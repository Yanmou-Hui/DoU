import argparse
import os
import torch
from models.dou import DoUNet, load_clip_to_cpu
from utils.options import TrainCfg
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.utils.data import DataLoader
from dataset.dataset import ImageFolderWithPaths
import pandas as pd

DEFAULT_TEST_DATASETS = [
    "biggan", "cyclegan", "gaugan", "progan", "stargan", "stylegan", "dalle", "guided",
    "deepfake", "seeingdark", "san", "crn", "imle",
    "glide_50_27", "glide_100_10", "glide_100_27", "ldm_100", "ldm_200", "ldm_200_cfg",
    "biggan_genimage",
    "ADM", "Glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM",
    "wukong",
    "Infinity", "LlamaGen", "PixArt-XL", "SD35-L", "Show-o", "VAR",
    "FLUX", "Janus", "Janus-Pro-1B", "Janus-Pro-7B",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DoU on one or more test datasets")
    parser.add_argument("--method", default="DoUNet", help="Model method name stored in TrainCfg")
    parser.add_argument(
        "--data-root",
        help="Root directory containing test dataset folders",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_TEST_DATASETS,
        help="Dataset folder names under --data-root, separated by spaces",
    )
    parser.add_argument("--ckpt", default="runs/best.pth", help="Path to the saved model checkpoint")
    parser.add_argument("--output", default="result.xlsx", help="Path to save the Excel result file")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device, e.g. auto, cuda, cuda:0, or cpu",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--model-name", default="ViT-L/14", help="CLIP backbone name")
    parser.add_argument("--disable-amp", action="store_true", help="Disable AMP during evaluation")
    parser.add_argument("--skip-missing", action="store_true", help="Skip dataset folders that do not exist")
    return parser.parse_args()


def build_model(args, device):
    cfg = TrainCfg(
        method=args.method,
        model_name=args.model_name,
        classnames=("real", "generated"),
        learn_logit_scale=False,
        amp=not args.disable_amp,
    )

    clip_model = load_clip_to_cpu(cfg)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    model = DoUNet(cfg, clip_model).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, cfg


def evaluate_dataset(model, cfg, dataset_path, batch_size, num_workers, device):
    val_dataset = ImageFolderWithPaths(dataset_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    preds, truths, scores = [], [], []
    for imgs, y in val_dataloader:
        imgs = imgs.to(device)
        y = y.to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)

        preds.extend(pred.flatten().tolist())
        truths.extend(y.flatten().tolist())
        scores.extend(probs[:, 1].detach().cpu().numpy().tolist())

    y_true = np.array(truths)
    y_pred = np.array(preds)
    y_score = np.array(scores)

    acc = accuracy_score(y_true, y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
    f1 = f1_score(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_score)

    return {
        "ACC": acc * 100,
        "Real ACC": r_acc * 100,
        "Fake ACC": f_acc * 100,
        "F1": f1 * 100,
        "AP": ap_score * 100,
    }


def main():
    args = parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    model, cfg = build_model(args, device)

    max_len = max(len(dataset) for dataset in args.datasets) + 2
    results = []
    for dataset in args.datasets:
        dataset_path = os.path.join(args.data_root, dataset)
        if not os.path.isdir(dataset_path):
            message = f"Dataset folder not found: {dataset_path}"
            if args.skip_missing:
                print(f"Skip {dataset}: {message}")
                continue
            raise FileNotFoundError(message)

        metrics = evaluate_dataset(
            model=model,
            cfg=cfg,
            dataset_path=dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        print(
            f"{dataset:<{max_len}} | "
            f"{metrics['ACC']:6.2f}% | "
            f"{metrics['Real ACC']:7.2f}% | "
            f"{metrics['Fake ACC']:7.2f}% | "
            f"{metrics['F1']:6.2f}% | "
            f"{metrics['AP']:6.2f}%"
        )
        results.append({"Dataset": dataset, **metrics})

    if not results:
        raise RuntimeError("No datasets were evaluated")

    df = pd.DataFrame(results)
    df = df.set_index("Dataset").transpose()
    df.to_excel(args.output)

    acc_mean = sum(item["ACC"] for item in results) / len(results)
    print(f"Average ACC: {acc_mean:.2f}%")
    print(f"Done! Results saved to {args.output}")


if __name__ == "__main__":
    main()
