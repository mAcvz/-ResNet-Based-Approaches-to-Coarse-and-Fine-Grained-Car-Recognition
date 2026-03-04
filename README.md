# ResNet-Based Approaches to Coarse and Fine-Grained Car Recognition

**Course:** Neural Networks and Deep Learning — University of Padua  
**Author:** Marco Cavazza  
**Dataset:** [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/)

---

## Overview

This project applies Convolutional Neural Networks (CNNs) to vehicle image classification using a subset of the CompCars dataset. Two tasks are addressed:

1. **Car Make Recognition** — coarse-grained classification across multiple manufacturers
2. **BMW Model Recognition** — fine-grained classification across 50 visually similar BMW models

---

## Repository Structure
```
├── Car_Make_recognition_Cavazza_Marco.ipynb   # Coarse-grained make classification
├── Models_recognition_Cavazza_Marco.ipynb     # Fine-grained BMW model classification
├── report/
│   └── cavazza_marco_REPORT.pdf               # Full project report
└── README.md
```

---

## Tasks

### Task 1 — Car Make Recognition
Classification across multiple manufacturers on a balanced CompCars subset. Three architectures of increasing complexity are compared, trained both from scratch and with pretrained ImageNet weights.

### Task 2 — BMW Model Recognition
Fine-grained classification of 50 BMW models (~5000 images). The class distribution is highly imbalanced. Focal Loss with class weighting is applied to address the imbalance.

---

## Models

| Model | Description |
|---|---|
| **ResNet_B** (Baseline) | Compact custom ResNet with 4 BasicBlocks, 13 trainable layers |
| **ResNet18** | Standard ResNet18 + custom two-layer head (BatchNorm + ReLU + Dropout) |
| **ResNet50** | Deep ResNet50 with IdentityBlocks and ConvolutionalBlocks across 4 stages |

All models trained from scratch; best architecture (ResNet18) also tested with pretrained ImageNet weights.

---

## Pipeline
```
Dataset (CompCars subset)
        ↓
Preprocessing + Data Augmentation
(resize 256×256 → crop 224×224, flips, rotations, color jitter, Gaussian blur)
        ↓
Model Training
(SGD + L2 decay, Cosine Annealing LR, 60 epochs, batch size 64)
        ↓
Evaluation
(Accuracy, Precision, Recall, macro-F1, ROC-AUC, Confusion Matrix)
```

---

## Training Details

- **Optimizer:** SGD with L2 weight decay
- **LR Scheduler:** CosineAnnealingLR (`T_max=60`, `η_min=1e-3`)
- **Batch size:** 64 — **Epochs:** 60
- **Loss functions:**
  - `CrossEntropyLoss` — standard task
  - `FocalLoss` — BMW imbalanced task: $\mathcal{L}_{focal}(p_t) = -\alpha(1-p_t)^\gamma \log(p_t)$
- **Class weighting (BMW task):** $w_i = \frac{1}{\sqrt{n_i}} \cdot \frac{1}{\frac{1}{C}\sum_j \frac{1}{\sqrt{n_j}}}$

---

## Results

### Task 1 — Car Make Classification

| Model | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| ResNet_B (scratch) | 0.8943 | 0.8950 | 0.9945 |
| ResNet18 (scratch) | 0.9338 | 0.9357 | 0.9979 |
| ResNet50 (scratch) | 0.5918 | 0.5918 | 0.5918 |
| **ResNet18 (pretrained)** | **0.9785** | **0.9782** | **0.9998** |

### Task 2 — BMW Model Classification (50 classes)

| Model | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| ResNet_B (scratch) | 0.5590 | 0.5497 | 0.9560 |
| ResNet18 (scratch) | 0.7266 | 0.7338 | 0.9811 |
| ResNet50 (scratch) | 0.3297 | 0.3151 | 0.8936 |
| **ResNet18 (pretrained)** | **0.8644** | **0.8598** | **0.9971** |

---

## Key Findings

- **ResNet18** is the best accuracy/complexity trade-off for limited data
- **Transfer learning** consistently outperforms training from scratch
- **ResNet50** overfits when data is scarce — more capacity is not always better
- **Focal Loss** alone is insufficient when classes are extremely rare; dataset size remains the main bottleneck
- Fine-grained single-brand recognition likely requires metric or contrastive learning for further improvement

---

## Requirements
```bash
pip install torch torchvision numpy matplotlib scikit-learn jupyter
```

---

## Notes

This project was developed as the final assignment for the Neural Networks and Deep Learning course at the University of Padua. It was the author's first hands-on experience with deep learning and object-oriented Python, coming from a theoretical physics background.
