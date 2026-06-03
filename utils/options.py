from dataclasses import dataclass
from typing import Tuple


# ============== Training Configuration ==============
@dataclass
class TrainCfg:
    method: str = "DouNet"

    model_name: str = "ViT-L/14"  # Only used if you are using the open_clip version
    pretrained: str = "openai"
    classnames: Tuple[str, str] = ("real", "generated")
    csc: bool = True  # Class-specific context
    n_ctx: int = 16
    ctx_init: str = None  # None = random initialization, or e.g. "a photo of a"
    class_token_position: str = "end"  # end|middle|front
    img_size: int = 224
    learn_logit_scale: bool = False

    # Optimization / scheduling
    lr: float = 1e-3  # Prompt context
    lr_alpha: float = 1e-3
    lr_tau: float = 1e-4  # Temperature
    weight_decay: float = 1e-2
    epochs: int = 30
    batch_size: int = 256
    workers: int = 0
    grad_accum_steps: int = 1
    amp: bool = True
    warmup_steps: int = 500

    # Early stopping / checkpoint saving
    early_stop_patience: int = 5
    outdir: str = "runs/dou"
    save_best_metric: str = "acc"  # auroc|acc

    # Data augmentation
    blur_prob: float = 0
    blur_sig = (0.0, 3.0)
    jpg_prob: float = 0
    jpg_method = ('cv2', 'pil')
    jpg_qual = (70, 100)

    # Miscellaneous
    seed: int = 42
    log_interval: int = 10
    alpha: float = 0.5  # Contrastive loss weight
    lambda_CIB: float = 1e-3  # Information bottleneck weight
    lambda_AFCL: float = 0.5  # HSIC loss weight
    lambda_reg: float = 0.5  # Alpha regularization weight
    train_alpha: bool = True
    train_proj: bool = True
