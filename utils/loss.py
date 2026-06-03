from __future__ import annotations
import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F


def kl_normal(mu, logvar):
    if mu.dim() > 2:
        mu = mu.reshape(-1, mu.size(-1))
        logvar = logvar.reshape(-1, logvar.size(-1))
    q = Normal(mu, (0.5 * logvar).exp())
    p = Normal(mu.new_zeros(mu.size(-1)), mu.new_ones(mu.size(-1)))
    kld = kl_divergence(q, p).sum(dim=1).mean()
    return kld


import torch
import torch.nn.functional as F


def compute_cue_removal_consistency(
        z_stack,  # [B, H, D]   D=1024
        alpha,  # [1, H, D]
        visual_proj,  # [D, D_clip] usually [1024, 768]
        text_features,  # [C, D_clip]
        logit_scale,  # scalar
):
    """
    Cue-removal consistency loss.

    z_stack: per-layer latent features before final CLIP projection
    alpha: learnable aggregation weights
    visual_proj: CLIP visual projection matrix
    text_features: normalized text features in CLIP joint space
    logit_scale: exp(logit_scale)

    Returns:
        scalar loss
    """
    B, H, D = z_stack.shape

    # full aggregation
    w = torch.softmax(alpha, dim=1).to(z_stack.dtype)  # [1, H, D]
    if w.size(0) != B:
        w = w.expand(B, -1, -1)  # [B, H, D]

    z_full = (w * z_stack).sum(dim=1)  # [B, D]
    img_full = z_full @ visual_proj  # [B, D_clip]
    img_full = F.normalize(img_full, dim=-1)

    logits_full = logit_scale * (img_full @ text_features.t())  # [B, C]
    p_full = F.softmax(logits_full, dim=-1).detach()

    loss = 0.0

    for i in range(H):
        mask = torch.ones_like(w)
        mask[:, i, :] = 0.0

        w_minus = w * mask
        w_minus = w_minus / (w_minus.sum(dim=1, keepdim=True) + 1e-6)

        z_minus = (w_minus * z_stack).sum(dim=1)  # [B, D]
        img_minus = z_minus @ visual_proj  # [B, D_clip]
        img_minus = F.normalize(img_minus, dim=-1)

        logits_minus = logit_scale * (img_minus @ text_features.t())
        log_p_minus = F.log_softmax(logits_minus, dim=-1)

        loss += F.kl_div(log_p_minus, p_full, reduction="batchmean")

    return loss / H


def pairwise_linear_hsic(z_stack: torch.Tensor) -> torch.Tensor:
    """Compute average linear HSIC among all layer pairs in ``z_stack``.

    Args:
        z_stack: Tensor of shape [B, H, D] where B is the batch size, H the
            number of layers and D the feature dimension. Each slice
            ``z_stack[:, h, :]`` corresponds to the VIB output for layer ``h``.

    Returns:
        Scalar tensor with the mean HSIC value. Zero when there are fewer than
        two samples or layers.
    """
    if z_stack.dim() != 3:
        raise ValueError(f"Expected z_stack with 3 dims [B,H,D], got {z_stack.shape}")

    B, H, _ = z_stack.shape
    if B < 2 or H < 2:
        return z_stack.new_zeros(())

    z = z_stack.to(torch.float32)
    z = z - z.mean(dim=0, keepdim=True)

    hsic_total = z.new_zeros(())
    num_pairs = 0
    for i in range(H - 1):
        zi = z[:, i, :]
        for j in range(i + 1, H):
            zj = z[:, j, :]
            cov = zi.t().matmul(zj) / (B - 1)
            hsic_total = hsic_total + cov.pow(2).sum()
            num_pairs += 1

    if num_pairs == 0:
        return z.new_zeros(())

    return hsic_total / num_pairs
