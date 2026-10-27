# <p align="center">🚀 DoU</p>

## <p align="center">Diversity over Uniformity: Rethinking Representation in Generated Image Detection</p>

<p align="center">
  Official PyTorch implementation for generated image detection.
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/He_Diversity_over_Uniformity_Rethinking_Representation_in_Generated_Image_Detection_CVPR_2026_paper.pdf">
    <img src="https://img.shields.io/badge/Paper-CVPR%202026-blue?logo=adobeacrobatreader&logoColor=white" alt="Paper">
  </a>
  <a href="https://huggingface.co/Yanmou/DoU">
    <img src="https://img.shields.io/badge/🤗-Hugging%20Face-yellow" alt="Hugging Face">
  </a>
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

---


## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{he2026diversity,
  title={Diversity over uniformity: Rethinking representation in generated image detection},
  author={He, Qinghui and Zhang, Haifeng and Qin, Qiao and Liu, Bo and Bi, Xiuli and Xiao, Bin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={40407--40417},
  year={2026}
}
```
