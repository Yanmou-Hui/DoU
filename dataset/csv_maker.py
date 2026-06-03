import os
import csv
import argparse
import random


def is_image_file(fname):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return os.path.splitext(fname.lower())[1] in IMG_EXTS


def make_csv(root_dir, out_csv, percent=1.0):
    real_items = []
    fake_items = []

    # ======== Traverse all images ========
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not is_image_file(fname):
                continue

            fpath = os.path.join(dirpath, fname)
            fpath = os.path.abspath(fpath)

            # Label assignment
            if "0_real" in fpath:
                real_items.append((fpath, 0))
            elif "1_fake" in fpath:
                fake_items.append((fpath, 1))

    print(f"Number of Real Images: {len(real_items)}, Number of Fake Images: {len(fake_items)}")

    if percent < 1.0:
        real_k = max(1, int(len(real_items) * percent))
        fake_k = max(1, int(len(fake_items) * percent))

        print(f"Sampling ratio: {percent * 100:.1f}%, keep {real_k} real images and {fake_k} fake images")

        real_items = random.sample(real_items, real_k) if real_k < len(real_items) else real_items
        fake_items = random.sample(fake_items, fake_k) if fake_k < len(fake_items) else fake_items

    # Merge
    items = real_items + fake_items
    random.shuffle(items)

    # ======== Write to CSV ========
    print(f"Write {len(items)} items -> {out_csv}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(items)


def make_train_val_csv(train_root, val_root, train_csv, val_csv, percent=1.0):
    print("Generate train CSV")
    make_csv(train_root, train_csv, percent)
    print("Generate val CSV")
    make_csv(val_root, val_csv, percent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Real/Fake image CSV files with optional sampling")
    parser.add_argument("--root", type=str, default="../dataset/genimage/stable_diffusion_v_1_4",
                        help="Image root for single CSV mode, or dataset root containing train/val folders")
    parser.add_argument("--output", type=str, default="train.csv", help="Output CSV path for single CSV mode")
    parser.add_argument("--train-root", type=str, default=None, help="Train image root")
    parser.add_argument("--val-root", type=str, default=None, help="Validation image root")
    parser.add_argument("--train-output", type=str, default="dataset/train.csv", help="Output train CSV path")
    parser.add_argument("--val-output", type=str, default="dataset/val.csv", help="Output val CSV path")
    parser.add_argument("--percent", type=float, default=1,
                        help="Sampling ratio, e.g. 0.1 = keep 10%% from each class")
    args = parser.parse_args()

    train_root = args.train_root
    val_root = args.val_root

    root_train = os.path.join(args.root, "train")
    root_val = os.path.join(args.root, "val")
    if train_root is None and val_root is None and os.path.isdir(root_train) and os.path.isdir(root_val):
        train_root = root_train
        val_root = root_val

    if train_root is not None or val_root is not None:
        if train_root is None or val_root is None:
            raise ValueError("--train-root and --val-root must be provided together")
        make_train_val_csv(train_root, val_root, args.train_output, args.val_output, args.percent)
    else:
        make_csv(args.root, args.output, args.percent)
