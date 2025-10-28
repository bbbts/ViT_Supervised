# Segmenter: Transformer for Semantic Segmentation

![Figure 1 from paper](./overview.png)

[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)  
by Robin Strudel*, Ricardo Garcia*, Ivan Laptev and Cordelia Schmid, ICCV 2021.  
*Equal Contribution

🔥 **Segmenter is now available on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter).**

# 🌍 Vision Transformer (ViT) for Semantic Segmentation — Supervised Training on Flame & ADE20K

**Author:** Bijoya Bhattacharjee  
**Affiliation:** Ph.D. Student, Department of Electrical and Computer Engineering, University of Nevada, Las Vegas (UNLV)  
**Research Focus:** Computer Vision & Machine Learning — Semantic Segmentation for Wildfire Prediction  

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [Background & Related Works](#background--related-works)  
   - [Transformers in Vision](#transformers-in-vision)  
   - [Vision Transformer (ViT)](#vision-transformer-vit)  
   - [Segmenter: Transformer for Semantic Segmentation](#segmenter-transformer-for-semantic-segmentation)  
3. [Dataset Structure](#dataset-structure)  
   - [Flame Dataset](#flame-dataset)  
   - [ADE20K Dataset](#ade20k-dataset)  
4. [Installation](#installation)  
5. [Training Procedure](#training-procedure)  
6. [Evaluation & Results](#evaluation--results)  
7. [Inference](#inference)  
8. [Repository Structure](#repository-structure)  
9. [References](#references)  
10. [Author & Acknowledgments](#author--acknowledgments)  

---

## 1️⃣ Overview

This repository implements **semantic segmentation using Vision Transformers (ViT)** under a **fully supervised setup**, based on the **Segmenter** architecture ([Strudel et al., 2021](https://arxiv.org/abs/2105.05633v3)).

It extends the Segmenter framework to perform semantic segmentation on:
- **Flame Dataset (Fire Segmentation)** — from IEEE Dataport  
- **ADE20K Dataset (General Scene Understanding)** — a large-scale semantic segmentation benchmark  

The repository supports **end-to-end supervised training** using ViT backbones (e.g., `vit_tiny`, `vit_small`, `vit_base`) combined with the **Mask Transformer Decoder**.  
The experiments focus on **pixel-level classification**, comparing Transformer-based segmentation against traditional CNN-based methods.

---

## 2️⃣ Background & Related Works

### 🧠 Transformers in Vision

The Transformer architecture, introduced by Vaswani et al. (2017), revolutionized sequence modeling with the **“Attention is All You Need”** paradigm. Initially developed for NLP, it was later extended to vision tasks by partitioning images into patches, treating them as visual tokens.

**Paper:** [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)

---

### 🧩 Vision Transformer (ViT)

**ViT** (Dosovitskiy et al., 2020) demonstrated that pure Transformer models can perform competitively on image classification tasks when trained on large datasets.

- Images are split into fixed-size patches.  
- Each patch is linearly embedded and processed like tokens in NLP.  
- A classification token (CLS) aggregates global context.

**Paper:** [An Image is Worth 16x16 Words (ViT, 2020)](https://arxiv.org/abs/2010.11929)  
**Official Code:** [Google Research Vision Transformer](https://github.com/google-research/vision_transformer)

---

### 🎨 Segmenter: Transformer for Semantic Segmentation

The **Segmenter** model (Strudel et al., 2021) extended ViT to dense prediction tasks.  
Instead of a classification head, Segmenter uses a **mask transformer decoder** that learns contextual relationships between patches to predict segmentation masks.

**Key Highlights:**
- Employs ViT backbone for global feature extraction.  
- Uses a learnable set of mask tokens to decode semantic classes.  
- Achieves competitive performance on datasets like ADE20K.

**Paper:** [Segmenter: Transformer for Semantic Segmentation (2021)](https://arxiv.org/abs/2105.05633v3)  
**Official GitHub Repo:** [Segmenter Repository](https://github.com/rstrudel/segmenter)

---

## 3️⃣ Dataset Structure

### 🔥 Flame Dataset

The **Flame dataset** (IEEE Dataport) focuses on **fire segmentation** and contains annotated wildfire images with pixel-level masks.

```
Datasets/Flame/
    ├── images/
    │     ├── train/ (.jpg images)
    │     ├── test/ (.jpg images)
    │     └── validation/ (.jpg images)
    └── masks/
          ├── train/ (.png masks)
          ├── test/ (.png masks)
          └── validation/ (.png masks)
```

Each `.jpg` image has a corresponding `.png` mask representing different fire-related classes.  
You can download the Flame dataset from IEEE Dataport:  
🔗 [Flame Dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

---

### 🏙️ ADE20K Dataset

**ADE20K** is one of the most comprehensive datasets for semantic segmentation with **150 object categories**.

```
Datasets/ADE20K/ADEChallengeData2016/
    ├── images/
    │     ├── training/ (.jpg images)
    │     └── validation/ (.jpg images)
    └── annotations/
          ├── training/ (.png masks)
          └── validation/ (.png masks)
```

You can download the dataset from the official MIT Scene Parsing Benchmark:  
🔗 [ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

---

## 4️⃣ Installation

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/segmenter-ade20k-flame.git
cd segmenter-ade20k-flame
```

---

### Option 1: Conda Environment & Requirements
```bash
conda create -n segmenter_env python=3.8 -y
conda activate segmenter_env
pip install -r requirements.txt
```

### Option 2: PyTorch + pip install
1. Install [PyTorch 1.9](https://pytorch.org/) following your system configuration.
2. At the root of the repository, run:
```bash
pip install .
```

---

### Dataset Environment Variable
Define the dataset path:
```bash
export DATASET=/path/to/dataset/dir
```

### ADE20K Preparation
```bash
python -m segm.scripts.prepare_ade20k $DATASET
```

---

### Logs
You can visualize experiment logs using:
```bash
python -m segm.utils.logs logs.yml
```
`logs.yml` should specify your checkpoints, for example:
```yaml
root: /path/to/checkpoints/
logs:
  seg-t: seg_tiny_mask/log.txt
  seg-b: seg_base_mask/log.txt
```
This will generate plots of training/validation loss and accuracy for easy monitoring.


---

## 5️⃣ Training Procedure

### Flame
```bash
python3 train.py \
  --dataset flame \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --batch-size 8 \
  --epochs 50 \
  --learning-rate 0.0001 \
  --log-dir ./logs/Flame_ViT_Tiny/
```

### ADE20K
```bash
python3 train.py \
  --dataset ade20k \
  --backbone vit_small_patch16_384 \
  --decoder mask_transformer \
  --batch-size 8 \
  --epochs 50 \
  --learning-rate 0.0001 \
  --log-dir ./logs/ADE20K_ViT_Small/
```

Logs, validation accuracy, and loss curves are saved to the `--log-dir` path.

---

## 6️⃣ Evaluation & Results
```bash
python3 eval.py \
  --dataset flame \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --resume ./logs/Flame_ViT_Tiny/checkpoint.pth
```

- Saves **ground truth vs predicted masks** side-by-side.  
- Reports **mIoU, Pixel Accuracy, F1-score**.  
- Visual results are stored in the output directory.

---

## 7️⃣ Inference

```bash
python3 inference.py \
  --image /path/to/custom_image.jpg \
  --checkpoint ./logs/Flame_ViT_Tiny/checkpoint.pth \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --output ./inference_results/ \
  --overlay
```

- Generates segmentation mask for input image.  
- Optional `--overlay` shows predicted mask over original image.  

---

## 8️⃣ Repository Structure

```
segmenter-ade20k-flame/
│
├── segm/                    # Core Segmenter source code
├── train.py                 # Training script
├── eval.py                  # Evaluation script
├── inference.py             # Inference script
├── requirements.txt         # Dependencies
├── datasets/                # Dataset loaders
├── logs/                    # Saved training runs, checkpoints, and plots
├── README.md                # Project documentation
└── utils/                   # Helper functions
```

---

## 9️⃣ References

| Year | Paper | Authors | Link |
|------|--------|----------|------|
| 2017 | *Attention Is All You Need* | Vaswani et al. | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 2020 | *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* | Dosovitskiy et al. | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| 2020 | *Vision Transformer Code* | Google Research | [GitHub](https://github.com/google-research/vision_transformer) |
| 2021 | *Segmenter: Transformer for Semantic Segmentation* | Strudel et al. | [arXiv:2105.05633v3](https://arxiv.org/abs/2105.05633v3) |
| 2021 | *Segmenter Code Repository* | Strudel et al. | [GitHub](https://github.com/rstrudel/segmenter) |
| 2022 | *FLAME: A Large-Scale Dataset for Fire Segmentation* | IEEE Dataport | [Dataset Link](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) |
| 2017 | *ADE20K Scene Parsing Benchmark* | Zhou et al. | [Dataset Link](https://groups.csail.mit.edu/vision/datasets/ADE20K/) |

---

## 🔟 Author & Acknowledgments

**Author:**  
👩‍💻 **Bijoya Bhattacharjee**  
Ph.D. Student — Electrical & Computer Engineering  
University of Nevada, Las Vegas (UNLV)  

**Research Topics:**  
🔥 Wildfire Detection & Segmentation  
🛰️ Remote Sensing & Multimodal Data Fusion  
🧠 Vision Transformers & Self-Supervised Learning  

**Acknowledgments:**  
- Vision Transformer implementation uses [timm](https://github.com/rwightman/pytorch-image-models)  
- Semantic segmentation pipeline uses [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
- Builds upon **Segmenter (2021)** by Robin Strudel et al., extending supervised framework to Flame & ADE20K.

> *“Transformers reshaped NLP, ViT extended them to vision, and with Segmenter — we bring them to the realm of dense, pixel-level understanding.”*
