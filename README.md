# Deep Learning-Based Medical Image Analysis for Early Detection of Diabetic Retinopathy

## 🎓 VIT University - B.Tech IT Capstone Project (BITE498J)
**Winter Semester 2025-26 | 5 Credits**

---

## 📋 Project Overview

Diabetic Retinopathy (DR) is a leading cause of preventable blindness affecting over 103 million people globally (IDF, 2024). This project develops an **automated deep learning system** using Convolutional Neural Networks (CNNs) to classify diabetic retinopathy severity levels from retinal fundus images, enabling early detection and timely intervention.

### 🎯 Objectives
- Develop an accurate multi-class classification system for DR severity (5 classes)
- Achieve classification accuracy > 90% and AUC-ROC > 0.95
- Implement interpretable AI using Grad-CAM visualizations
- Create a deployable screening tool for clinical settings

---

## 🔬 Technical Approach

### Architecture
- **Base Model**: EfficientNet-B4 with Transfer Learning
- **Framework**: TensorFlow/Keras or PyTorch
- **Dataset**: APTOS 2019 Blindness Detection Dataset (3,662 images)

### DR Severity Classes
| Grade | Description |
|-------|-------------|
| 0 | No DR |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | Proliferative DR |

### Preprocessing Pipeline
- Image resizing (512x512)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Ben Graham's preprocessing technique
- Data augmentation (rotation, flipping, brightness adjustment)

### Model Features
- Transfer learning from ImageNet weights
- Mixed precision training
- Grad-CAM for visual explanations
- Ensemble methods for improved accuracy

---

## 📂 Project Structure

```
Project-2/
├── README.md                    # Project overview
├── zeroth_review.md             # Zeroth review submission
├── docs/
│   ├── project_proposal.md      # Detailed proposal
│   ├── literature_review.md     # Latest literature survey
│   └── methodology.md           # Technical methodology
├── src/
│   ├── data/                    # Data processing scripts
│   ├── models/                  # Model architectures
│   ├── training/                # Training scripts
│   ├── evaluation/              # Evaluation metrics
│   └── visualization/           # Grad-CAM & plots
├── notebooks/
│   └── experiments.ipynb        # Jupyter experiments
├── results/
│   └── metrics/                 # Performance metrics
└── requirements.txt             # Dependencies
```

---

## 📊 Expected Outcomes

| Metric | Target |
|--------|--------|
| Accuracy | > 90% |
| AUC-ROC | > 0.95 |
| Sensitivity | > 85% |
| Specificity | > 90% |
| Quadratic Weighted Kappa | > 0.85 |

---

## 🛠️ Technology Stack

- **Language**: Python 3.10+
- **Deep Learning**: TensorFlow 2.15 / PyTorch 2.0
- **Image Processing**: OpenCV, Pillow, Albumentations
- **Visualization**: Matplotlib, Seaborn, Grad-CAM
- **Deployment**: Flask/FastAPI, Streamlit
- **Version Control**: Git, GitHub

---

## 📅 Project Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1 | Week 1-2 | Literature review, dataset preparation |
| Phase 2 | Week 3-4 | Baseline model implementation |
| Phase 3 | Week 5-6 | Model optimization & experiments |
| Phase 4 | Week 7-8 | Evaluation & interpretability |
| Phase 5 | Week 9-10 | Deployment & documentation |

---

## 📚 Key References

1. Gulshan, V., et al. (2016). Deep learning for diabetic retinopathy detection. *JAMA*, 316(22), 2402-2410.
2. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling. *ICML*.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations. *ICCV*.

---

## 👥 Project Team

- **Student**: [Your Name]
- **Guide**: Dr. Usha Devi
- **Department**: School of Computer Science Engineering and Information Systems

---

## 📄 License

This project is developed as part of academic requirements at VIT University.

---

*Last Updated: January 2026*