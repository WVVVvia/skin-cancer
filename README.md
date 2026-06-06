**ECE 284: Digital Health Technologies — UCSD, Spring 2026**  
Yixian Wang | PID: A69044452

## Overview
This project replicates and extends the CNN-based skin lesion 
classification pipeline of Esteva et al. (Nature 2017) on the 
HAM10000 dataset. We compare three architectures — InceptionV3, 
ResNet50, and EfficientNet-B0 — using transfer learning, 
class-weighted loss, and data augmentation, and analyze model 
interpretability via Grad-CAM and age-stratified health equity.

**Best result:** InceptionV3 achieves 87.0% validation accuracy 
and macro-AUC 0.977 across 7 lesion classes.

## Dataset
[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
— 10,015 dermoscopy images, 7 lesion types, available on Kaggle.
No IRB required.

## Repository Structure
ece284-skin-cancer/
├── eda.ipynb                  # Exploratory data analysis & class distribution
├── inceptionv3.ipynb          # InceptionV3 training, AUC, Grad-CAM, health equity
├── resnet50.ipynb             # ResNet50 training, AUC, Grad-CAM
├── efficientnet.ipynb         # EfficientNet-B0 training, AUC, Grad-CAM
└── README.md

## Methods
- **Transfer learning** from ImageNet pretrained weights
- **Class-weighted cross-entropy loss** to handle severe class imbalance (nv = 67%)
- **Data augmentation**: random flip, rotation ±30°, color jitter
- **Cosine annealing** learning rate schedule (lr=1e-4, 15 epochs)
- **Grad-CAM** visualization on all three architectures
- **Age-stratified health equity analysis** on test set

## Results

| Model | Val Acc | Macro AUC | Val Loss |
|-------|---------|-----------|----------|
| InceptionV3 | **87.0%** | **0.977** | 0.487 |
| ResNet50 | 85.4% | 0.956 | 0.562 |
| EfficientNet-B0 | 82.7% | **0.975** | **0.495** |

**Key findings:**
- Melanoma (mel) is the hardest class across all models (AUC 0.869–0.931)
- EfficientNet-B0 achieves comparable AUC with only 4.0M parameters
- Model performance decreases with patient age (AUC gap: 0.989 vs 0.960 for <40 vs >60)

## Requirements
torch >= 2.0
torchvision
scikit-learn
matplotlib
pandas
numpy
Pillow
opencv-python

## How to Run
1. Download HAM10000 from Kaggle and add to your Kaggle input
2. Run notebooks in order:
   - `eda.ipynb` → generates class distribution figures
   - `inceptionv3.ipynb` → trains model, saves `InceptionV3_best.pth`
   - `resnet50.ipynb` → trains model, saves `ResNet50_best.pth`
   - `efficientnet.ipynb` → trains model, saves `EfficientNet-B0_best.pth`

## References
- Esteva et al. "Dermatologist-level classification of skin cancer 
  with deep neural networks." Nature 542 (2017)
- Tschandl et al. "The HAM10000 dataset." Scientific Data 5 (2018)
- Selvaraju et al. "Grad-CAM." ICCV (2017)

## AI Tool Disclosure
Claude (Anthropic) was used for report formatting and literature 
organization. All code was written and debugged independently.
