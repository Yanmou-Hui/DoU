import torch
import os
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


# Load the CLIP model
def load_clip_to_cpu(cfg):
    backbone_name = cfg.model_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # Loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


# Text encoder that maps prompt tokens into feature vectors
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        :param prompts: Tensor representation of text prompts
        :param tokenized_prompts: Tokenized text prompts
        :return: Text features
        """
        # Add prompts and positional embeddings
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # Take features from the EOT embedding (EOT token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# Learn and generate context-aware prompts
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.img_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # Use CLIP tokenize to determine the actual number of tokens
            # instead of counting spaces manually
            ctx_init = ctx_init.replace("_", " ")
            with torch.no_grad():
                tok = clip.tokenize(ctx_init)  # [1, L]
                eot = tok.argmax(dim=-1).item()  # Index of the final EOT token
                n_ctx = eot - 1  # Remove [SOS] and keep only context tokens
                emb = clip_model.token_embedding(tok).to(torch.float32)  # Use FP32 for parameters, not fp16
            # Extract the token vectors corresponding to the context
            # remove [SOS] and keep n_ctx tokens
            ctx_vectors = emb[0, 1:1 + n_ctx, :].contiguous()  # [n_ctx, D]
            prompt_prefix = ctx_init
        else:
            # Random initialization: parameters must be FP32
            if cfg.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=torch.float32)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # To be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved in save_model(),
        # but they should be ignored in load_model() because we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        ctx = ctx.to(self.token_prefix.dtype)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


# Combine the image encoder and the text encoder
class DoUNet(nn.Module):
    def __init__(self, cfg, clip_model, proj_dim=1024):
        super().__init__()
        classnames = cfg.classnames
        self.clip = clip_model
        self.image_encoder = clip_model.visual
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Register hooks to capture intermediate layer outputs
        hook_candidates = [(name, module) for name, module in self.clip.visual.named_modules() if "ln_2" in name]
        hook_candidates.sort(key=lambda x: x[0])  # Ensure stable layer ordering
        self.hooks = [Hook(name, module) for name, module in hook_candidates]

        H = len(self.hooks)
        self.alpha = nn.Parameter(torch.randn(1, H, proj_dim))  # [1, H, D]

        self.layer_mu = nn.ModuleList([
            nn.Linear(proj_dim, proj_dim) for _ in range(H)
        ])
        self.layer_logvar = nn.ModuleList([
            nn.Linear(proj_dim, proj_dim) for _ in range(H)
        ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # logvar = log(σ^2)
        eps = torch.randn_like(std)  # Same shape/device/dtype as std
        return mu + std * eps

    def forward(self, image, depth=None, return_features=False):

        with torch.no_grad():
            _ = self.clip.encode_image(image)

        cls_list = [h.output[0, :, :].to(torch.float32) for h in self.hooks]
        g = torch.stack(cls_list, dim=1).float()

        mu_layers = []
        logvar_layers = []
        z_layers = []
        for idx, feat in enumerate(torch.unbind(g, dim=1)):
            mu_i = self.layer_mu[idx](feat)
            logvar_i = self.layer_logvar[idx](feat)
            z_i = self.reparameterize(mu_i, logvar_i)
            mu_layers.append(mu_i)
            logvar_layers.append(logvar_i)
            z_layers.append(z_i)

        mu = torch.stack(mu_layers, dim=1)
        logvar = torch.stack(logvar_layers, dim=1)
        z_stack = torch.stack(z_layers, dim=1)

        for h in self.hooks:
            h.output = None  # Optional: clear cache

        w = torch.softmax(self.alpha, dim=1).to(z_stack.dtype)
        if w.size(0) != z_stack.size(0):
            w = w.expand(z_stack.size(0), -1, -1)
        z = (w * z_stack).sum(dim=1)
        z = z.to(self.dtype)

        image_features = z @ self.clip.visual.proj

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        if return_features:
            return logits, (mu, logvar), image_features, text_features, z_stack
        else:
            return logits
