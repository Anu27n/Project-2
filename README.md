# Deep Learning-Based Medical Image Analysis for Early Detection of Diabetic Retinopathy

**EfficientNet-B4 + CBAM Attention** model for automated DR severity grading from retinal fundus images.

## Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | APTOS 2019 Blindness Detection (3,662 images, 5 classes) |
| **Model** | EfficientNet-B4 backbone + CBAM attention + dual pooling |
| **Training** | 3-phase progressive fine-tuning with Focal Loss + Label Smoothing CE |
| **Imbalance Strategy** | Weighted sampler + dynamic class weights + optional conditional GAN augmentation |
| **Key Metrics** | QWK 0.878, AUC-ROC 0.964, Accuracy 84.2%, Sensitivity 85.3% |

### DR Severity Classes

| Grade | Class | Description |
|-------|-------|-------------|
| 0 | No DR | Healthy retina |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | More than just microaneurysms |
| 3 | Severe NPDR | Extensive haemorrhages, venous beading |
| 4 | Proliferative DR | Neovascularisation, vitreous haemorrhage |

## Preserving Notebook Outputs & Backups

**Training takes a long time** — protect your work:

1. **Backup notebooks with outputs** (run after training):
   ```bash
   python backup_notebooks.py
   ```
   Creates `backups/notebooks_backup_YYYYMMDD_HHMMSS.zip` with notebooks + results.

2. **Keep outputs in git**: Ensure notebooks are committed *with* outputs. Avoid "Clear All Outputs" before committing.

3. **Download the zip** from `backups/` to save locally if needed.

## Project Structure

```
Project-2/
├── app/
│   ├── api.py                 # FastAPI inference endpoint
│   └── streamlit_app.py       # Streamlit web interface
├── data/
│   └── aptos2019/             # Downloaded dataset (not in git)
│       ├── train.csv
│       ├── test.csv
│       ├── train_images/
│       └── test_images/
├── docs/
│   ├── project_proposal.md
│   ├── literature_review.md
│   └── methodology.md
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Results_Analysis.ipynb
├── results/
│   ├── models/                # Trained checkpoints (not in git)
│   │   ├── model_best_qwk.pth
│   │   ├── model_best_acc.pth
│   │   ├── model_last.pth
│   │   └── training_history.json
│   ├── metrics/
│   │   └── final_evaluation_results.json
│   └── figures/               # Generated plots
├── scripts/
│   └── download_dataset.py    # Kaggle dataset downloader
├── src/
│   ├── config.py              # All project configuration
│   ├── dataset.py             # PyTorch Dataset + DataLoaders
│   ├── augmentation/
│   │   └── cgan.py            # Conditional GAN training + minority sample generation
│   ├── preprocessing.py       # 6-step image preprocessing pipeline
│   ├── models/
│   │   └── efficientnet_model.py  # EfficientNet-B4 + CBAM + Ensemble
│   ├── training/
│   │   ├── losses.py          # Focal, Weighted Focal, Label Smoothing, Ordinal
│   │   └── trainer.py         # 3-phase progressive fine-tuning trainer
│   ├── evaluation/
│   │   └── metrics.py         # Comprehensive metrics evaluator (12+ metrics)
│   └── visualization/
│       └── gradcam.py         # Grad-CAM, Grad-CAM++, EigenCAM
├── run_train.py               # Training entry point
├── run_cgan_augment.py        # cGAN-based minority image synthesis
├── run_evaluate.py            # Evaluation entry point
├── run_inference.py           # Single image / batch inference
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download APTOS 2019 Dataset

Set up your Kaggle API credentials first:
- Go to https://www.kaggle.com/settings and create an API token
- Save `kaggle.json` to `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows)
- Accept competition rules at https://www.kaggle.com/c/aptos2019-blindness-detection/rules

```bash
python scripts/download_dataset.py
```

### 3. Train the Model

Optional (recommended for class imbalance): generate minority-class synthetic images with cGAN before training.

```bash
python run_cgan_augment.py
```

This creates:
- `data/aptos2019/train_augmented.csv`
- `data/aptos2019/cgan_augmentation_summary.json`
- synthetic images in `data/aptos2019/train_images/` with `synthetic_*` names

Then run training normally. `run_train.py` automatically uses `train_augmented.csv` when available.

```bash
python run_train.py
```

This runs the full 3-phase progressive fine-tuning pipeline:
- **Phase 1** (10 epochs): Feature extraction with frozen backbone, lr=1e-3
- **Phase 2** (15 epochs): Partial fine-tuning (top 50% unfrozen), lr=1e-4
- **Phase 3** (5 epochs): Full fine-tuning (all layers), lr=1e-5

### 4. Evaluate

```bash
python run_evaluate.py
python run_evaluate.py --checkpoint results/models/model_best_qwk.pth --save-figures
```

`run_evaluate.py` also reports compute-aware novelty metrics in `results/metrics/final_evaluation_results.json`:
- parameter count and model size
- MACs / FLOPs (approximate)
- local latency and throughput
- accuracy-per-GFLOP, QWK-per-GFLOP, and computation efficiency score

### 5. Run Inference

```bash
# Single image
python run_inference.py --image path/to/retinal_image.png

# Batch inference
python run_inference.py --image-dir path/to/images/ --output predictions.csv

# With Grad-CAM visualization
python run_inference.py --image path/to/image.png --gradcam
```

### 6. Web Interface

```bash
# Streamlit app
streamlit run app/streamlit_app.py

# FastAPI endpoint
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## Portable Run + ZIP Packaging

### Run Locally (CPU or GPU)

```bash
# Auto-detect device (GPU if available, else CPU)
python run_train.py --device auto

# Optional: resume from last checkpoint automatically
python run_train.py --resume auto

# Optional: quick debug run
python run_train.py --debug --epochs 6
```

### Run on Kaggle/Colab GPU

```bash
# Inside Kaggle/Colab after uploading/extracting the project
pip install -r requirements.txt

# Auto-select CUDA when available
python run_train.py --device auto
python run_evaluate.py
```

### Create Upload-Ready ZIP (Automatic)

```bash
python create_zip.py
```

This generates:
- `Project-2.zip`

The archive is created from the project root and excludes unnecessary files/folders such as:
- `.git/`, `.venv/`, `__pycache__/`, `data/`, `backups/`, temp/log files

It keeps core assets for training/evaluation portability:
- `src/`, `notebooks/`, `run_train.py`, `run_evaluate.py`, `run_cgan_augment.py`, `requirements.txt`, `README.md`
- `results/models/` checkpoint files (`.pth`, `.pt`, `.ckpt`, `.safetensors`)

## Architecture

### Model: EfficientNet-B4 + CBAM

```
Input Image (3, 512, 512)
    │
    ▼
EfficientNet-B4 Backbone (ImageNet pretrained)
    │
    ▼
CBAM Attention (Channel + Spatial)
    │
    ├──► Global Average Pooling ──┐
    │                              ├──► Concatenate (3584-d)
    └──► Global Max Pooling ──────┘
                │
                ▼
    Classification Head (MLP + BN + Dropout)
                │
                ▼
        5-class Softmax Output
```

### Preprocessing Pipeline

1. **Black border cropping** - Remove camera artifacts
2. **Resize** to 512x512
3. **Ben Graham enhancement** - Local contrast normalization
4. **Circular mask** - Focus on retinal region
5. **CLAHE** - Adaptive histogram equalization
6. **ImageNet normalization**

### Loss Function

Combined loss: **70% Weighted Focal Loss + 30% Label Smoothing CE**
- Focal Loss handles class imbalance (gamma=2.0, per-class weights)
- Label Smoothing (epsilon=0.1) prevents overconfident predictions

## Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| QWK | >= 0.85 | 0.878 | Achieved |
| AUC-ROC | >= 0.95 | 0.964 | Achieved |
| Sensitivity | >= 85% | 85.3% | Achieved |
| Specificity | >= 90% | 90.1% | Achieved |
| Accuracy | >= 90% | 84.2% | In Progress |

### Per-Class Performance

| Class | Sensitivity | Specificity | F1-Score | AUC-ROC |
|-------|-------------|-------------|----------|---------|
| No DR | 87.3% | 91.3% | 0.879 | 0.971 |
| Mild | 70.3% | 96.3% | 0.716 | 0.922 |
| Moderate | 82.0% | 89.3% | 0.816 | 0.949 |
| Severe | 68.4% | 96.6% | 0.684 | 0.931 |
| Proliferative | 90.0% | 98.5% | 0.871 | 0.987 |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA_Preprocessing.ipynb` | Dataset exploration, class distribution, preprocessing pipeline demo |
| `02_Model_Training.ipynb` | Full training pipeline with augmentation, model architecture, 3-phase training |
| `03_Results_Analysis.ipynb` | Training curves, confusion matrices, ROC curves, Grad-CAM, ablation study |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | Upload image for DR prediction |

## Technologies

- **PyTorch 2.0+** with mixed precision (AMP)
- **timm** for pretrained EfficientNet models
- **Albumentations** for data augmentation
- **scikit-learn** for metrics
- **FastAPI** for REST API
- **Streamlit** for web interface
- **OpenCV** for image processing
