# pasta-shape-classifier
A model built using ResNet18 to classify different pasta shapes

## Overview
This is a basic image classifier project for learning purpose. It identifies pasta shapes (penne, fusilli, farfalle, etc.) from photos, built by fine-tuning a pretrained ResNet18 on the Pasta Shape Dataset.

## What's inside
- Data loading — Downloads the dataset via kagglehub, explores the (nested) folder structure, and previews a sample image from each pasta class.
- A "safe" dataset loader — Some images in the raw dataset are corrupted; SafeImageFolder verifies every file up front and skips the bad ones instead of crashing mid-training.  
- Three training runs, each building on the last:
  - Run 1 — Baseline run: full fine-tuning of ResNet18, no augmentation.
  - Run 2 — Adds data augmentation (flips, rotation, color jitter, affine shifts) and an LR scheduler.
  - Run 3 — Freezes all layers except layer4 and the final classifier head, refines augmentation, and trains for more epochs.
- Evaluation — Test-set accuracy after each run, plus a confusion matrix over all classes for the final model.
- Inference — Loads a single image and gets the model's top-3 predicted pasta shapes with confidence scores.

 <img width="759" height="607" alt="image" src="https://github.com/user-attachments/assets/fa3422d7-3c6a-4063-ab2c-56c5bcfdb1f8" />        
 
Sample inference on a farfalle pasta

 
- Explainability — Grad-CAM heatmaps showing which regions of an image drove the model's prediction.
  
<img width="696" height="697" alt="image" src="https://github.com/user-attachments/assets/76538c57-df6d-4969-8df9-63ce21156ea7" /> 

Grad-CAM: model attention on the pasta, not the background


## Requirements
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm kagglehub grad-cam opencv-python pillow
```
## Usage
1. Open pasta-vision.ipynb in Jupyter or Colab.
2. Run the cells top to bottom — the dataset downloads automatically via kagglehub on first run.
3. Each "RUN" section trains and saves its own checkpoint (best_model_run1.pth, run2, run3), so you can skip straight to Run 3 if you just want the best model.
4. To classify your own photo, drop an image into the working directory, update img_path in the inference cell, and re-run it.

## Results
| Run | Approach | Notes | Test Accuracy |
|---|---|---|---|
| 1 | Full fine-tune, no augmentation| Baseline | 70.5% |
| 2 | Full fine-tune + augmentation + LR scheduler | Reduces overfitting | 72.4% |
| 3 | Partial freeze (layer4 + fc only) + refined augmentation, 20 epochs | Best-performing model | 78.0% |

## Confusion Matrix Findings

**Model:** best_model_run3.pth  
**Overall test accuracy:** 78.0%%

### Most confused pairs

| True shape | Confused with | Roughly how often | Likely reason |
|---|---|---|---|
| rotini | fusili | 14 (Highest) | oth are twisted pasta sahpes with ridges |
| fettuccine | linguine | 10  | both long straight strand pastas |
| tagliatelle | fettuccine | 9  | both long straight strand pastas |
| fettuccine | tagliatelle | 8  | both long straight strand pastas |
| spaghetti | linguine | 8  | both long straight strand pastas |
| linguine | spaghetti | 8  | both long straight strand pastas |
| buccatini | linguine | 8  | both long straight strand pastas |
| fusili | rotini | 8 | both are twisted pasta sahpes with ridges |
| cavatappi | macaroni | 7 | cavatappi's tubed corkscrew shape at a small crop can resemble macaroni's tubed shape |

### Strongest classes
Shapes the model rarely misclassifies :
- tortellini (68 correct, minimal errors)
- orecchiette (59 correct, minimal errors)
- cannelloni (54 correct, minimal errors)

### Weakest classes
Shapes with the most spread-out row (getting predicted as several different things):
- fettuccine (10 lost to a linguine, 8 lost to tagliatelle)
- rotini (14 lost to a single other class alone)
- tagliatelle (9 to one class)

### Patterns noticed
-  Confusion is mostly between visually similar shapes as expected.  

### Takeaways
- The three cleanest classes (tortellini, orecchiette, cannelloni) share a trait: they're all fairly distinctive 3D/folded shapes rather than long strands — long-strand pastas (spaghetti, linguine, fettuccine, bucatini, rotini) are your harder cluster overall

## Acknowledgments
- Dataset: Pasta Shape Dataset by Zibran Zarif on Kaggle.
- Model: ResNet18 pretrained on ImageNet, via torchvision.






