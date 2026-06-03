import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import random
import numpy as np
import os


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Enforce deterministic behavior for cuBLAS
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def cosine_warmup_scheduler(optimizer, base_lr, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def gather_trainable_params(model: nn.Module, cfg) -> list:
    """
    Collect trainable parameter groups for DoUNet.

    Returns:
        A list of param_groups that can be passed directly to torch.optim.AdamW.
    """
    params = []

    # 1) Prompt learner context
    if hasattr(model, "prompt_learner") and hasattr(model.prompt_learner, "ctx"):
        params.append({
            "params": [model.prompt_learner.ctx],
            "lr": getattr(cfg, "lr_text_ctx", cfg.lr),
            "weight_decay": cfg.weight_decay,
        })

    # 2) Alpha for layer aggregation
    if getattr(cfg, "train_alpha", True) and hasattr(model, "alpha"):
        print("Add alpha to training params.")
        params.append({
            "params": [model.alpha],
            "lr": getattr(cfg, "lr_alpha", cfg.lr),
            "weight_decay": cfg.weight_decay,
        })

    # 3) Projection layers: layer_mu and layer_logvar
    if getattr(cfg, "train_proj", True):
        proj_params = []

        if hasattr(model, "layer_mu"):
            for m in model.layer_mu:
                proj_params += list(m.parameters())

        if hasattr(model, "layer_logvar"):
            for m in model.layer_logvar:
                proj_params += list(m.parameters())

        if len(proj_params) > 0:
            print("Add layer_mu/layer_logvar to training params.")
            params.append({
                "params": proj_params,
                "lr": getattr(cfg, "lr_proj", cfg.lr),
                "weight_decay": cfg.weight_decay,
            })

    # 4) Optional logit scale
    if getattr(cfg, "learn_logit_scale", False) and hasattr(model, "logit_scale"):
        params.append({
            "params": [model.logit_scale],
            "lr": getattr(cfg, "lr_tau", cfg.lr),
            "weight_decay": 0.0,
        })

    return params


def freeze_model_except_prompts(model: nn.Module, cfg):
    """
    Freeze the entire model first, then unfreeze only the lightweight
    trainable modules used by DoUNet.

    Trainable modules:
      - prompt_learner.ctx
      - alpha           (optional, controlled by cfg.train_alpha)
      - layer_mu/logvar (optional, controlled by cfg.train_proj)
      - logit_scale     (optional, controlled by cfg.learn_logit_scale)
    """
    # 1) Freeze all parameters
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    # 2) Unfreeze prompt learner
    if hasattr(model, "prompt_learner") and hasattr(model.prompt_learner, "ctx"):
        model.prompt_learner.ctx.requires_grad_(True)

    # 3) Unfreeze alpha
    if getattr(cfg, "train_alpha", True) and hasattr(model, "alpha"):
        model.alpha.requires_grad_(True)

    # 4) Unfreeze variational projection layers
    if getattr(cfg, "train_proj", True):
        if hasattr(model, "layer_mu"):
            for m in model.layer_mu:
                for p in m.parameters():
                    p.requires_grad_(True)

        if hasattr(model, "layer_logvar"):
            for m in model.layer_logvar:
                for p in m.parameters():
                    p.requires_grad_(True)

    # 5) Optional logit scale
    if getattr(cfg, "learn_logit_scale", False) and hasattr(model, "logit_scale"):
        try:
            model.logit_scale.requires_grad_(True)
        except Exception:
            pass

    # 6) Report trainable parameters
    print(f"{'param name':<64} {'shape':<20} {'requires_grad'}")
    print("-" * 100)
    for n, p in model.named_parameters():
        print(f"{n:<64} {str(tuple(p.shape)):<20} {p.requires_grad}")
