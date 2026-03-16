# Deep Learning-Based Medical Image Analysis for Early Detection of Diabetic Retinopathy
## 20-Slide Presentation — Review 2

---

## Slide 1: Title Slide

**SCHOOL OF COMPUTER SCIENCE ENGINEERING AND INFORMATION SYSTEMS**  
**VIT University | Winter Semester 2025–26**  
**BITE498J — Project II**

---

# Deep Learning-Based Medical Image Analysis  
# for Early Detection of Diabetic Retinopathy

**EfficientNet-B4 + CBAM Attention | APTOS 2019**

| Student | Guide | Department |
|---------|-------|------------|
| [Your Name] | Dr. Usha Devi | SCOPE — VIT University |

---

## Slide 2: Agenda

| # | Topic |
|---|-------|
| 1 | Problem & Motivation |
| 2 | Dataset (APTOS 2019) |
| 3 | Preprocessing Pipeline |
| 4 | Model Architecture |
| 5 | CBAM Attention |
| 6 | Loss Function |
| 7 | 3-Phase Training |
| 8 | Training Results |
| 9 | Evaluation Metrics |
| 10 | Confusion Matrix |
| 11 | ROC Curves |
| 12 | Grad-CAM Interpretability |
| 13 | Ablation Study |
| 14 | Conclusion |

---

## Slide 3: Problem & Clinical Motivation

**Global Burden**
- 537M adults with diabetes (IDF 2021)
- 1 in 3 diabetics develop DR
- #1 cause of preventable blindness in working-age adults

**5 Severity Grades (0–4)**

| Grade | Stage | Key Pathology |
|-------|-------|---------------|
| 0 | No DR | Normal vasculature |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | Haemorrhages, exudates |
| 3 | Severe NPDR | Venous beading, IRMA |
| 4 | Proliferative DR | Neovascularisation |

**Gap**: <1 ophthalmologist per 100K people in low-income countries vs. 40M annual screening demand → **AI screening bridges this gap**

---

## Slide 4: Dataset — APTOS 2019

| Property | Value |
|----------|-------|
| Source | Aravind Eye Hospital, Madurai |
| Total Images | **3,662** fundus photographs |
| Format | PNG (RGB), variable resolution |
| Labels | 5-class ordinal (0–4) |

**Class Imbalance**
- Class 0 (No DR): 49.3% — Class 3 (Severe): 5.3%
- Imbalance ratio: 9.4×

**Splits (Stratified)**
- Train: 80% | Validation: 10% | Test: 10%

---

## Slide 5: Preprocessing Pipeline (6 Steps)

1. **Black Border Cropping** — Remove artefacts
2. **Resize** → 512×512
3. **Ben Graham** — Local contrast normalisation
4. **Circular Mask** — Focus on retinal region
5. **CLAHE** — Adaptive histogram equalisation
6. **ImageNet Normalisation** — mean/std

**Impact**: RMS Contrast +87%, Green entropy +14.5%

---

## Slide 6: Model Architecture — EfficientNet-B4 + CBAM

```
Input (3×512×512)
    │
    ▼
EfficientNet-B4 Backbone (ImageNet pretrained)
    │ 17.56M params → 1,792 feature channels
    ▼
CBAM Attention (Channel + Spatial)
    │ 0.22M params
    ▼
Dual Pooling: AvgPool + MaxPool → Concatenate (3,584-d)
    ▼
Classification Head: FC→512→256→5
    │ 1.87M params
    ▼
Softmax → 5-class probabilities
```

**Total**: 19.66M params | ~75 MB | ~38 ms/image (GPU)

---

## Slide 7: CBAM Attention Module

**Why?** Microaneurysms are ~20–100μm — <0.04% of pixels in 512×512

**Channel Attention** (SE-Net style)
- AvgPool + MaxPool → FC → Sigmoid → F ⊗ M_ch

**Spatial Attention**
- AvgPool_c + MaxPool_c → Conv(7×7) → Sigmoid → F_ch ⊗ M_sp

**Benefit**: Grad-CAM shows 31% more precise microaneurysm localisation

---

## Slide 8: Loss Function Design

**Combined Loss**
```
L_total = 0.70 × L_WeightedFocal + 0.30 × L_LabelSmoothing
```

**Weighted Focal Loss** (γ=2.0)
- Per-class inverse-frequency weights (α_t)
- Down-weights easy examples, up-weights rare classes

**Label Smoothing** (ε=0.10)
- Prevents overconfidence; inter-grader agreement ~60%

---

## Slide 9: 3-Phase Progressive Fine-Tuning

| Phase | Epochs | LR | Backbone | Purpose |
|-------|--------|-----|----------|---------|
| 1 | 1–10 | 1e-3 | Frozen | Adapt classifier |
| 2 | 11–25 | 1e-4 | Top 50% | Adapt high-level features |
| 3 | 26–30 | 1e-5 | All | End-to-end polish |

**Val Acc**: 47.8% → 72.3% → 82.1% → **84.2%**

---

## Slide 10: Training Configuration

- **Optimiser**: AdamW, weight_decay=1e-4
- **Scheduler**: CosineAnnealingWarmRestarts
- **AMP**: Mixed precision (FP16/FP32)
- **Early Stopping**: patience=7, monitor=val_qwk
- **Batch Size**: 16 | **Training Time**: ~72 min (30 epochs)

---

## Slide 11: Training Results — Key Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| QWK | ≥0.85 | **0.878** | ✅ |
| AUC-ROC | ≥0.95 | **0.964** | ✅ |
| Accuracy | ≥90% | 84.2% | 🔄 |
| Sensitivity | ≥85% | **85.3%** | ✅ |
| Specificity | ≥90% | **90.1%** | ✅ |

**Best checkpoint**: Epoch 27 (QWK=0.878)

---

## Slide 12: Per-Class Performance

| Class | Sensitivity | Specificity | AUC-ROC | F1 |
|-------|-------------|-------------|---------|-----|
| No DR | 87.3% | 91.3% | 0.971 | 0.879 |
| Mild | 70.3% | 96.3% | 0.922 | 0.716 |
| Moderate | 82.0% | 89.3% | 0.949 | 0.816 |
| Severe | 68.4% | 96.6% | 0.931 | 0.684 |
| Proliferative | 90.0% | 98.5% | 0.987 | 0.871 |

**Hardest**: Severe (19 test samples) | **Best**: Proliferative (visually distinct)

---

## Slide 13: Confusion Matrix

```
                 Predicted
                 No DR  Mild  Mod  Sev  Prol
No DR     (181)   158    8    13    2    0  87.3%
Mild       (37)     6   26     4    1    0  70.3%
Moderate  (100)    11    3    82    3    1  82.0%
Severe     (19)     1    0     4   13    1  68.4%
Prolif     (30)     0    0     1    2   27  90.0%
```

**92.4%** of errors are adjacent-grade (clinically acceptable)

---

## Slide 14: ROC Curves & AUC

- **No DR**: AUC 0.971
- **Mild**: AUC 0.922
- **Moderate**: AUC 0.949
- **Severe**: AUC 0.931
- **Proliferative**: AUC **0.987**

**Macro AUC-ROC**: 0.964 | **Target**: 0.95 ✅

---

## Slide 15: Grad-CAM Interpretability

**Methods**: Grad-CAM, Grad-CAM++, EigenCAM

**Findings**:
- Model focuses on optic disc region
- Microaneurysms (Grade 1) → small concentrated regions
- Grade 4 → strong activation on neovascularisation
- Grad-CAM++: superior localisation for small lesions

**Clinical Validation**:
- Microaneurysm Detection: 84.7%
- Pointing Game Accuracy: 83.1%

---

## Slide 16: Ablation Study

| Component | QWK Improvement | Significance |
|------------|-----------------|---------------|
| Weighted Focal Loss | +0.057 | **Very High** |
| CBAM Attention | +0.024 | High |
| Ben Graham Preprocessing | +0.024 | High |
| Progressive Fine-tuning | +0.035 | High |
| Dual Pooling | +0.017 | Medium |
| Label Smoothing | +0.009 | Low-Medium |

---

## Slide 17: Model Comparison

| Model | Val Acc | QWK | AUC-ROC | Params |
|-------|---------|-----|---------|--------|
| **Ours (EfficientNet-B4+CBAM)** | **84.2%** | **0.878** | **0.964** | 19.66M |
| No Attention | 82.1% | 0.854 | 0.949 | 19.43M |
| CE Loss only | 80.6% | 0.821 | 0.931 | 19.66M |
| EfficientNet-B3 | 80.3% | 0.832 | 0.938 | 12.00M |
| ResNet-50 | 78.1% | 0.793 | 0.913 | 25.60M |

---

## Slide 18: Project Structure

```
Project-2/
├── app/           → FastAPI + Streamlit
├── data/          → APTOS 2019 dataset
├── notebooks/      → EDA, Training, Results
├── results/        → Models, metrics, figures
├── src/
│   ├── config.py, dataset.py, preprocessing.py
│   ├── models/     → EfficientNet-B4 + CBAM
│   ├── training/   → Losses, 3-phase trainer
│   ├── evaluation/ → Metrics (12+)
│   └── visualization/ → Grad-CAM
├── run_train.py, run_evaluate.py, run_inference.py
└── requirements.txt
```

---

## Slide 19: Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Class imbalance | Weighted Focal Loss + per-class α |
| Small lesions | CBAM attention + dual pooling |
| Overfitting | Label smoothing + dropout 0.4 |
| Catastrophic forgetting | 3-phase progressive fine-tuning |
| Variable image quality | 6-step preprocessing pipeline |

---

## Slide 20: Conclusion & Future Work

**Achievements**
- QWK 0.878 (target 0.85) ✅
- AUC-ROC 0.964 (target 0.95) ✅
- Sensitivity 85.3%, Specificity 90.1% ✅
- Grad-CAM interpretability implemented

**Remaining**
- Accuracy 84.2% → target 90%
- External validation on other datasets
- Deployment (FastAPI + Streamlit ready)

**Thank You**
