# MAFD-Net

**MAFD-Net: Efficient Intrusion Detection via Lightweight Convolutional Network with Multi-scale Attentive Feature Distillation**

> Zhenhua Cheng, Hui Hu*, Wengting Zhou, Shihong Deng, Longkun Cui  
> School of Information and Software Engineering, East China Jiaotong University  
> *Corresponding author: 2528@ecjtu.edu.cn

---

## Abstract

Network intrusion detection (NID) is a foundational technology for ensuring network security. However, deploying lightweight yet high-precision intrusion detection models (IDS) on resource-constrained devices remains a significant challenge. To address this, we propose **MAFD-Net**, a lightweight intrusion detection model that enhances detection accuracy through two key innovations:

1. **Multi-Head Attention (MHA)** mechanism integrated into EfficientNet-B0's MBConv modules, replacing the original Squeeze-and-Excitation (SE) attention to capture both local and global feature dependencies simultaneously.
2. **Multi-scale Attentive Feature Distillation (MAFD)**, which transfers intermediate layer features from the heavy teacher model (EfficientNet-B7) to the lightweight student model (MAFD-Net), bridging the performance gap while keeping model complexity low.

Experimental results show that MAFD-Net achieves **98.99%**, **78.32%**, and **99.66%** accuracy on NSL-KDD, UNSW-NB15, and QAX2024 datasets respectively, while reducing model parameters by **89%** compared to EfficientNet-B7.

**Keywords:** Intrusion Detection, Deep Learning, Multi-Head Attention, Lightweight Model, EfficientNet, Knowledge Distillation

---

## Architecture Overview

```
Network Traffic Data
        ↓
  Traffic-to-Image Conversion (16×16 PNG → resized to 64×64)
        ↓
┌─────────────────────────────────────────┐
│              MAFD-Net (Student)         │
│  EfficientNet-B0 + MHA-MBConv Modules   │
│  (Stages 3,4,5 - middle layer replaced) │
└─────────────────────────────────────────┘
        ↑  Knowledge Distillation (MAFD)
┌─────────────────────────────────────────┐
│        EfficientNet-B7 (Teacher)        │
│     Pre-trained on intrusion dataset    │
└─────────────────────────────────────────┘
        ↓
  Attack Classification Output
```

### MAFD Distillation Pipeline

The MAFD framework consists of three collaborative modules:

| Module | Description |
|--------|-------------|
| **Channel Adaptive Selection (CAS)** | Uses channel attention to filter noise and select informative feature channels from teacher features |
| **Dynamic Layer Matching (DLM)** | Dynamically matches each teacher layer to the top-K most relevant student layers via attention scores |
| **Multi-scale Feature Alignment & Distillation (MFAD)** | Projects aligned features into a unified embedding space and minimizes L2 distance for knowledge transfer |

**Total Loss:**
```
L_total = α · L_CE + γ · L_MAFD
```
where α = 0.8, γ = 0.2 (classification loss weighted higher than distillation loss).

---

## Datasets

This project is evaluated on three network intrusion detection datasets:

### NSL-KDD
A refined version of KDDCup99 with redundancies removed. **5-class classification** (Normal, DoS, Probe, R2L, U2R).

| Split | Total | Normal | DoS | Probe | R2L | U2R |
|-------|-------|--------|-----|-------|-----|-----|
| Train | 118,886 | 61,643 | 42,780 | 11,262 | 2,999 | 202 |
| Test  | 29,721  | 15,411 | 10,695 | 2,815 | 750  | 50  |

Download: http://nsl.cs.unb.ca/NSL-KDD

### UNSW-NB15
A comprehensive benchmark with real network traffic. **10-class classification** (Normal + 9 attack types).

| Split | Total | Normal | DoS | Backdoor | Exploits | Fuzzers | Generic | Recon | Shellcode | Worms |
|-------|-------|--------|-----|----------|----------|---------|---------|-------|-----------|-------|
| Train | 175,341 | 56,000 | 12,264 | 1,746 | 33,393 | 18,184 | 40,000 | 10,491 | 1,133 | 130 |
| Test  | 82,332  | 37,000 | 4,089 | 583 | 11,132 | 6,062 | 18,871 | 3,496 | 378 | 44 |

Download: https://www.unb.ca/cic/datasets/unsw-nb15.html

### QAX2024 (Private)
An enterprise-grade dataset with eight days of real network traffic, covering **81 attack categories** including CVEs for Apache Log4j2, Struts2, Shiro, etc. Train/Test split = 8:2. Due to confidentiality agreements, this dataset cannot be made publicly available.

**Data Preprocessing:** Raw traffic features are encoded into **16×16 PNG images**, then resized to 64×64 for model input.

Place datasets under `./data/` with the following structure:
```
data/
├── unsw/
│   ├── train_multi/{0.0, 1.0, ..., 9.0}/image_*.png
│   └── test_multi/ {0.0, 1.0, ..., 9.0}/image_*.png
├── nsl/
│   ├── train/
│   └── test/
└── QAX2024/
    ├── train/
    └── test/
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/shihenzhiren/MAFD-Net.git
cd MAFD-Net

# Install dependencies
pip install torch torchvision
pip install tensorboard_logger
```

**Computing Infrastructure:**
- Training (Teacher): NVIDIA A100-PCIE-40GB
- Testing / Student Training: NVIDIA GTX 1080 Ti

---

## Training

### Step 1: Train the Teacher Model (EfficientNet-B7)

```bash
# UNSW-NB15
python train_teacher.py \
  --batch_size 64 --epochs 240 \
  --dataset unsw \
  --model efficientnet_b7 \
  --learning_rate 0.05 \
  --lr_decay_epochs 150,180,210 \
  --weight_decay 5e-4 \
  --trial 0 --gpu_id 0

# NSL-KDD
python train_teacher.py \
  --batch_size 64 --epochs 240 \
  --dataset nsl \
  --model efficientnet_b7 \
  --learning_rate 0.05 \
  --lr_decay_epochs 150,180,210 \
  --weight_decay 5e-4 \
  --trial 0 --gpu_id 0
```

### Step 2: Train MAFD-Net (Student) with MAFD Distillation

```bash
# UNSW-NB15 - MAFD distillation
python train_student.py \
  --path_t ./save/teachers/models/efficientnet_b7_unsw/efficientnet_b7_best.pth \
  --distill afd \
  --dataset unsw \
  --model_s efficientnet_b0 \
  -c 1 -d 1 -b 1 \
  --trial 0 --gpu_id 0

# NSL-KDD - MAFD distillation
python train_student.py \
  --path_t ./save/teachers/models/efficientnet_b7_nsl/efficientnet_b7_best.pth \
  --distill afd \
  --dataset nsl \
  --model_s efficientnet_b0 \
  -c 1 -d 1 -b 1 \
  --trial 0 --gpu_id 0
```

More training scripts are available in `./scripts/`.

---

## Supported Distillation Methods

This codebase is built upon the [SimKD](https://github.com/DefangChen/SimKD) / [RepDistiller](https://github.com/HobbitLong/RepDistiller) toolbox and supports the following knowledge distillation methods for comparison:

| Method | Reference |
|--------|-----------|
| **MAFD (Ours)** | This paper |
| KD | Hinton et al., NeurIPS 2015 |
| FitNet | Romero et al., ICLR 2015 |
| AT | Zagoruyko & Komodakis, ICLR 2017 |
| SP | Tung & Mori, CVPR 2019 |
| VID | Ahn et al., CVPR 2019 |
| CRD | Tian et al., ICLR 2020 |
| SemCKD | Chen et al., AAAI 2021 |
| SimKD | Chen et al., CVPR 2022 |

---

## Experimental Results

### Ablation: MHA-MBConv Replacement Strategy (NSL-KDD / UNSW-NB15 / QAX2024)

| Exp. | Stages | Position | Params (M) | FLOPs (G) | NSL-KDD Acc (%) | UNSW-NB15 Acc (%) | QAX2024 Acc (%) |
|------|--------|----------|-----------|-----------|-----------------|-------------------|-----------------|
| A0 (Baseline) | None | — | 4.00 | 0.28 | 97.81 | 76.55 | 95.39 |
| A1 | [3] | Middle | 4.23 | 0.33 | 97.65 | 76.67 | 97.50 |
| A2 | [3,4] | Middle | 5.13 | 0.40 | 97.78 | 76.73 | 96.51 |
| **A3 (Ours)** | **[3,4,5]** | **Middle** | **6.90** | **0.51** | **97.94** | **77.02** | **96.86** |
| A4 | [3,4,5,6] | Middle | 12.10 | 0.84 | 97.75 | 76.91 | 97.65 |

> **A3 (stages 3,4,5 middle replacement) is adopted as the final MAFD-Net architecture.**

### Distillation Method Comparison

| Method | NSL-KDD Acc (%) | NSL-KDD F1 (%) | UNSW-NB15 Acc (%) | UNSW-NB15 F1 (%) | QAX2024 Acc (%) | QAX2024 F1 (%) |
|--------|----------------|----------------|-------------------|------------------|-----------------|----------------|
| Teacher (EfficientNet-B7) | 98.90 | 92.57 | 78.29 | 78.05 | 99.55 | 99.56 |
| Student (no distill) | 97.94 | 97.82 | 77.02 | 76.86 | 96.86 | 96.83 |
| KD | 98.58 | 91.55 | 77.23 | 77.06 | 98.63 | 98.64 |
| CRD | 98.90 | 92.81 | 77.73 | 77.53 | 99.31 | 99.32 |
| SimKD | 98.94 | 92.93 | 78.11 | 77.89 | 99.26 | 99.27 |
| **MAFD (Ours)** | **98.99** | **94.13** | **78.32** | **78.14** | **99.66** | **99.66** |
| MAFD w/o DLM | 98.73 | 92.37 | 78.03 | 78.02 | 99.05 | 99.06 |
| MAFD w/o CAS | 98.54 | 92.22 | 78.01 | 77.78 | 99.11 | 99.14 |

> MAFD achieves the best performance across all three datasets, with **89% parameter reduction** vs. the teacher model.

### Comparison with State-of-the-Art (NSL-KDD & UNSW-NB15)

| Method | NSL-KDD Acc (%) | NSL-KDD F1 (%) | UNSW-NB15 Acc (%) | UNSW-NB15 F1 (%) | Year | Preprocessing |
|--------|----------------|----------------|-------------------|------------------|------|---------------|
| RENOIR | 77.80 | 73.30 | 53.70 | 59.00 | 2021 | Raw traffic |
| ROULETTE | 81.50 | 79.00 | 76.30 | 76.70 | 2022 | Traffic-to-image |
| VINCENT | 82.90 | 81.80 | 78.40 | 78.10 | 2024 | Traffic-to-image |
| KD-TCNN | 98.20 | 86.63 | — | — | 2022 | Raw traffic |
| LNet-SKD | 98.66 | 89.03 | — | — | 2023 | Traffic-to-image |
| **MAFD-Net (Ours)** | **98.99** | **94.13** | **78.32** | **78.14** | **2025** | **Traffic-to-image** |

---

## Project Structure

```
MAFD-Net/
├── train_teacher.py          # Teacher model training script
├── train_student.py          # Student model training with KD
├── bohb_search.py            # Hyperparameter search (BOHB)
├── results_visualization.py  # Visualize training results
├── models/
│   ├── efficientnet.py       # EfficientNet student (MAFD-Net w/ MHA-MBConv)
│   ├── efficientnet_T.py     # EfficientNet teacher (B7)
│   ├── mobilenetv2.py
│   ├── resnet.py
│   ├── ShuffleNetv2.py
│   ├── vgg.py
│   └── util.py               # ConvReg, SRRL, SimKD, SelfA utilities
├── distiller_zoo/
│   ├── AFD.py                # Attentive Feature Distillation (MAFD core)
│   ├── AFD_improved.py       # Improved AFD variant
│   ├── KD.py                 # Vanilla Knowledge Distillation
│   ├── FitNet.py             # Hint-based distillation
│   ├── AT.py                 # Attention Transfer
│   ├── SP.py                 # Similarity Preserving
│   ├── VID.py                # Variational Information Distillation
│   └── SemCKD.py             # Semantic Calibration KD
├── dataset/
│   ├── unsw.py               # UNSW-NB15 dataloader (16×16 image format)
│   ├── QAX2024.py            # QAX2024 enterprise dataset loader
│   ├── cifar100.py           # CIFAR-100 (baseline reference)
│   └── imagenet.py           # ImageNet dataloader
├── helper/
│   ├── loops.py              # Training / validation loops
│   ├── util.py               # Learning rate scheduling, utilities
│   └── meters.py             # AverageMeter for logging
└── scripts/
    ├── run_vanilla.sh         # Train vanilla teacher/student models
    └── run_distill.sh         # Run various KD experiments
```

---

## Funding & Acknowledgements

This work was supported by the **National Natural Science Foundation of China (NSFC)** under Grant **61961020**.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{cheng2025mafdnet,
  title     = {MAFD-Net: Efficient Intrusion Detection via Lightweight Convolutional Network 
               with Multi-scale Attentive Feature Distillation},
  author    = {Cheng, Zhenhua and Hu, Hui and Zhou, Wengting and Deng, Shihong and Cui, Longkun},
  journal   = {--},
  year      = {2025},
  note      = {School of Information and Software Engineering, East China Jiaotong University}
}
```

This codebase is built upon the [SimKD](https://github.com/DefangChen/SimKD) toolbox and the [RepDistiller](https://github.com/HobbitLong/RepDistiller) benchmark. We sincerely thank the original authors for their open-source contributions.

---

## License

This project is for academic research purposes. The QAX2024 dataset is proprietary and cannot be redistributed. The NSL-KDD and UNSW-NB15 datasets are publicly available at their respective official links.
