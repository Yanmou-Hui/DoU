# <p align="center">🚀 DoU</p>

## <p align="center">Diversity over Uniformity: Rethinking Representation in Generated Image Detection</p>

<p align="center">
Official PyTorch implementation for generated image detection.
</p>

---

## 🧠 Overview

DoU revisits representation learning for generated image detection. Instead of enforcing representation uniformity, DoU promotes **cue diversity** while preserving robustness and generalization ability.

---

## 🔑 Core Components

| Component | Full Name | Description |
| --- | --- | --- |
| 🧩 CIB | Cue Information Bottleneck | A VAE-style variational bottleneck applied to multi-stage visual cues. |
| 🔄 AFCL | Adaptive Feature Correlation Learning | An inter-cue decorrelation mechanism that encourages diverse visual cues. |
| 🖼️ Multi-Stage CLIP | Multi-Stage CLIP Aggregation | Hierarchical CLIP visual features for robust generated image detection. |

---

## 📢 Release Status

| Item | Status | Notes |
| --- | --- | --- |
| 🧪 Testing / evaluation code | ✅ Released | Use `test.py` for benchmark evaluation. |
| 🏆 Pretrained weights | ✅ Released | Use released checkpoints for direct testing. |
| 🏋️ Training code | 🚧 Coming next | The next update will provide the training pipeline. |
| ⚙️ Training configs | 🚧 Coming next | Full training configuration will be released with training code. |
| 📘 Reproducibility details | 🚧 Coming next | More instructions will be added in the next update. |

---

## 📁 Evaluation Data Format

For the current release, only evaluation is required. Each test dataset folder should follow the common real/fake structure:

```text
test_root/
  progan/
    0_real/
    1_fake/
  stylegan/
    0_real/
    1_fake/
```

Label meaning:

| Folder | Class |
| --- | --- |
| `0_real` | Real |
| `1_fake` | Generated / Fake |

---

## 🏆 Pretrained Weights

Download the released pretrained weights before running evaluation.

| Source | Link | Access Code / Notes |
| --- | --- |---------------------|
| 百度网盘 | [Download](https://pan.baidu.com/s/1tXBYHT2C2ixOoTBd0taPWA?pwd=8pkh) | 提取码: `8pkh`         |
| Google Drive | [Download](https://drive.google.com/file/d/1S8g99k-42NjulNGcGHxnm1xZuUWwvm9k/view?usp=drive_link) | \                   |

After downloading, place the checkpoint under `runs/DoU/`:

```text
runs/
  DoU/
    best.pth
```

Then use `--ckpt runs/DoU/best.pth` when running evaluation.

---

## ⚡ Usage

### ✅ Main Command

For the current release, run the evaluation command below with the released checkpoint:

```bash
python test.py \
  --data-root  xxx\
  --ckpt runs/DoU/best.pth \
  --output result.xlsx \
  --batch-size 256
```

---

### 🔧 Optional Evaluation Commands

The commands below are alternative or advanced evaluation usages. You do **not** need to run all of them.

#### Evaluate selected datasets only

```bash
python test.py \
  --data-root xxx \
  --datasets progan stylegan biggan \
  --ckpt runs/DoU/best.pth \
  --output result.xlsx
```

#### Run evaluation on CPU

```bash
python test.py \
  --data-root xxx \
  --ckpt runs/DoU/best.pth \
  --device cpu
```

#### Skip missing dataset folders

```bash
python test.py \
  --data-root xxx \
  --ckpt runs/DoU/best.pth \
  --skip-missing
```

#### Show all command-line options

```bash
python test.py --help
```

---

## 🏋️ Training

The training code is the next part to be released.

Planned next update:

| Item | Description |
| --- | --- |
| Training pipeline | End-to-end training entry point. |
| Dataset preparation | CSV generation for training and validation. |
| Configurations | Hyperparameters and training settings. |
| Reproducibility guide | Instructions for reproducing reported results. |

---

## 📌 Notes

- Released testing code and pretrained weights can already be used for evaluation.
- Training code and training data preparation instructions will be added in the next update.
- Please check the command-line options before running on a new environment.
