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

The repository explores **end-to-end supervised training** using ViT backbones (e.g., `vit_tiny`, `vit_small`, `vit_base`) combined with the **Mask Transformer Decoder**.  
The experiments focus on **pixel-level classification**, comparing Transformer-based segmentation against traditional CNN-based methods.

---

## 2️⃣ Background & Related Works

### 🧠 Transformers in Vision

The Transformer architecture, introduced by Vaswani et al. (2017), revolutionized sequence modeling with the **“Attention is All You Need”** paradigm.  
Initially developed for NLP, it was later extended to vision tasks by partitioning images into patches, treating them as visual tokens.

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
**Official GitHub Repo:** [https://github.com/rstrudel/segmenter](https://github.com/rstrudel/segmenter)

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

### Create & Activate a Conda Environment
```bash
conda create -n segmenter_env python=3.8 -y
conda activate segmenter_env
```

### Install Requirements
```bash
pip install -r requirements.txt
```

### Dataset Path
Update the dataset path in your training script:
```bash
DATASET=/path/to/Datasets/Flame
# or for ADE20K
DATASET=/path/to/Datasets/ADE20K/ADEChallengeData2016
```

---
## Installation

Define os environment variables pointing to your checkpoint and dataset directory, put in your `.bashrc`:
```sh
export DATASET=/path/to/dataset/dir
```

Install [PyTorch 1.9](https://pytorch.org/) then `pip install .` at the root of this repository.

To download ADE20K, use the following command:
```python
python -m segm.scripts.prepare_ade20k $DATASET
```


## Logs

To plot the logs of your experiments, you can use
```python
python -m segm.utils.logs logs.yml
```

with `logs.yml` located in `utils/` with the path to your experiments logs:
```yaml
root: /path/to/checkpoints/
logs:
  seg-t: seg_tiny_mask/log.txt
  seg-b: seg_base_mask/log.txt
```



## 5️⃣ Training Procedure

Example training command for **Flame**:
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

Example training command for **ADE20K**:
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

Training logs, validation accuracy, and loss curves will be automatically saved to the `--log-dir` path.

---

## 6️⃣ Evaluation & Results

To evaluate a trained model:
```bash
python3 eval.py \
  --dataset flame \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --resume ./logs/Flame_ViT_Tiny/checkpoint.pth
```

During evaluation:
- Ground truth masks and predicted segmentation maps are saved **side-by-side**.  
- Metrics like **mIoU**, **Pixel Accuracy**, and **F1-score** are reported.  
- Visual results (GT vs Prediction) are stored in the output directory.  

---

## 7️⃣ Inference

After training, you can run inference on any custom image using the trained model.

Example:
```bash
python3 inference.py \
  --image /path/to/custom_image.jpg \
  --checkpoint ./logs/Flame_ViT_Tiny/checkpoint.pth \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --output ./inference_results/
```

**Output:**  
- The model will generate a segmentation mask for the input image.  
- The script saves a visualization showing **input image**, **predicted mask**, and optionally **overlayed output**.  

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
| 2022 | *FLAME: A Large-Scale Dataset for Fire Segmentation* | IEEE Dataport | [Dataset Link]([https://ieee-dataport.org/open-access/flame-dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)) |
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
This work builds on the foundation of the **Segmenter (2021)** implementation by **Robin Strudel et al.**,  
and extends its supervised framework to domain-specific fire segmentation using the **Flame dataset**, alongside general-purpose **ADE20K** evaluation.

---

> *“Transformers reshaped NLP, ViT extended them to vision, and with Segmenter — we bring them to the realm of dense, pixel-level understanding.”*


## Acknowledgements

The Vision Transformer code is based on [timm](https://github.com/rwightman/pytorch-image-models) library and the semantic segmentation training and evaluation pipeline 
is using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
