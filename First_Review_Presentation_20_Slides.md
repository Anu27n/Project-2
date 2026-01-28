# First Review - 20 Slides Presentation
# Deep Learning-Based Medical Image Analysis for Early Detection of Diabetic Retinopathy

---

## Slide 1: Title Slide

**SCHOOL OF COMPUTER SCIENCE ENGINEERING AND INFORMATION SYSTEMS**

**Winter Semester -- 2025-26**

**BITE498J -- Project II / Internship**

---

**Deep Learning-Based Medical Image Analysis for Early Detection of Diabetic Retinopathy**

---

**Student Name:** [Your Name]  
**Register Number:** [Your Register Number]  
**Guide:** Dr. Usha Devi

---

## Slide 2: Agenda

1. Introduction to Diabetic Retinopathy
2. Problem Statement
3. Motivation & Need
4. Literature Survey
5. Research Gap
6. Objectives
7. Scope of the Project
8. Dataset Description
9. Proposed Methodology
10. System Architecture
11. Preprocessing Pipeline
12. Deep Learning Model
13. Training Strategy
14. Grad-CAM Interpretability
15. Evaluation Metrics
16. Expected Outcomes
17. Implementation Timeline
18. Technology Stack
19. Risk Assessment
20. Conclusion & Future Work

---

## Slide 3: Introduction to Diabetic Retinopathy

### What is Diabetic Retinopathy?

- **Definition:** A diabetes complication affecting the blood vessels of the retina
- **Impact:** Leading cause of blindness in working-age adults (20-65 years)

### Global Statistics (2024)
| Metric | Value |
|--------|-------|
| People with Diabetes | 537 Million |
| Projected by 2045 | 783 Million |
| DR Prevalence | ~35% of diabetics |
| Vision Loss Preventable | Up to 95% with early detection |

### DR Severity Levels
- **0 - No DR:** Normal retina
- **1 - Mild:** Microaneurysms only
- **2 - Moderate:** More than mild but less than severe
- **3 - Severe:** Extensive abnormalities
- **4 - Proliferative:** Neovascularization, high risk of blindness

---

## Slide 4: Problem Statement

### Current Challenges in DR Screening

| Challenge | Description |
|-----------|-------------|
| **Time-Consuming** | Manual screening takes 5-10 minutes per patient |
| **Specialist Shortage** | Only 1 ophthalmologist per 100,000 people in many regions |
| **Human Error** | 15-30% inter-observer variability in grading |
| **Fatigue Effects** | Accuracy decreases after reviewing multiple images |
| **Scalability Issues** | Cannot meet growing diabetic population demands |
| **Healthcare Gaps** | Rural areas lack access to specialists |

### Key Problem
*"How can we develop an automated, accurate, and scalable screening system for diabetic retinopathy that can assist ophthalmologists and improve access to eye care in underserved regions?"*

---

## Slide 5: Motivation & Need

### Why Automated DR Detection?

**Clinical Need:**
- ✅ Early detection prevents 95% of vision loss cases
- ✅ Reduce screening bottlenecks
- ✅ Consistent, reproducible results
- ✅ 24/7 availability without fatigue

**Technological Enablers:**
- 🔬 Advances in Deep Learning (CNNs, Transformers)
- 📊 Availability of large annotated datasets
- 💻 GPU computing power accessibility
- 🌐 Cloud deployment capabilities

**Societal Impact:**
- 🌍 Improve healthcare accessibility
- 💰 Reduce screening costs
- 👥 Support specialist decision-making
- 🏥 Enable telemedicine solutions

---

## Slide 6: Literature Survey - Foundational Works

### Evolution of DR Detection Methods

**Traditional Methods (Pre-2012)**
- Handcrafted features (SIFT, SURF)
- Morphological operations
- Accuracy: 70-80%

**Deep Learning Era (2012-Present)**

| Author | Year | Method | Performance |
|--------|------|--------|-------------|
| Gulshan et al. (Google) | 2016 | Inception-v3 | 97.5% sensitivity |
| Gargeya & Leng | 2017 | Deep CNN | AUC 0.97 |
| He et al. | 2016 | ResNet | Foundation for transfer learning |
| Tan & Le | 2019 | EfficientNet | State-of-the-art efficiency |

---

## Slide 7: Literature Survey - Recent Advances (2023-2025)

### State-of-the-Art Methods

| Study | Year | Architecture | Accuracy | AUC |
|-------|------|--------------|----------|-----|
| Zhang et al. | 2024 | Vision Transformer | 94.2% | 0.972 |
| Chen et al. | 2024 | EfficientNetV2-L | 93.8% | 0.968 |
| Zhou et al. | 2024 | RETFound (Foundation Model) | 94.8% | 0.978 |
| Kumar et al. | 2024 | Multi-Task CNN | 93.1% | 0.962 |
| Patel et al. | 2024 | MobileViT-DR | 89.7% | 0.941 |

### Key Trends
- 🔄 Vision Transformers emerging
- 🎯 Multi-task learning for lesion detection
- 🔐 Federated learning for privacy
- 📱 Lightweight models for edge deployment
- 🔍 Explainable AI for clinical trust

---

## Slide 8: Research Gap

### Identified Gaps in Current Research

| Gap | Description | Our Approach |
|-----|-------------|--------------|
| **Class Imbalance** | Severe DR only 5.3% of data | Focal Loss + Weighted Sampling |
| **Generalization** | Models fail across different equipment | Robust preprocessing pipeline |
| **Real-time Processing** | Many models too slow for POC | Efficient architecture selection |
| **Clinical Integration** | Lack of interpretability | Grad-CAM visualization |
| **Deployment** | Research not production-ready | Web-based interface |

### Our Contribution
*Develop a robust, interpretable, and deployment-ready DR screening system addressing these gaps*

---

## Slide 9: Project Objectives

### Primary Objective
**Develop an automated deep learning system for 5-class classification of diabetic retinopathy severity from retinal fundus images**

### Secondary Objectives

1. **Architecture Implementation**
   - EfficientNet-B4 with transfer learning

2. **Preprocessing Excellence**
   - Ben Graham's method + CLAHE enhancement

3. **Class Imbalance Solution**
   - Focal Loss implementation

4. **Clinical Interpretability**
   - Grad-CAM visualization integration

5. **Deployment Readiness**
   - User-friendly web interface

6. **Performance Targets**
   - Accuracy > 90%, AUC > 0.95

---

## Slide 10: Scope of the Project

### In Scope ✅

| Component | Details |
|-----------|---------|
| Dataset | APTOS 2019 (3,662 images) |
| Classification | 5-class severity grading |
| Model | EfficientNet-B4 |
| Preprocessing | Ben Graham, CLAHE, Augmentation |
| Interpretability | Grad-CAM heatmaps |
| Deployment | Streamlit web application |

### Out of Scope ❌

- Real-time video processing
- Other retinal diseases (Glaucoma, AMD)
- Hospital EHR integration
- FDA regulatory approval
- Mobile application

---

## Slide 11: Dataset Description

### APTOS 2019 Blindness Detection Dataset

**Source:** Kaggle Competition (Asia Pacific Tele-Ophthalmology Society)

**Dataset Statistics:**
| Metric | Value |
|--------|-------|
| Training Images | 3,662 |
| Test Images | 1,928 |
| Image Format | High-resolution color fundus |
| Resolution | Variable (up to 3072×2048) |

**Class Distribution:**
| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative | 295 | 8.1% |

**Challenge:** Significant class imbalance (Severe DR only 5.3%)

---

## Slide 12: Proposed Methodology Overview

### End-to-End Pipeline

```
Input Image → Preprocessing → Augmentation → Model → Output + Visualization
```

### Key Stages:

**Stage 1: Data Preparation**
- Load APTOS dataset
- Quality filtering
- Train/validation split (stratified)

**Stage 2: Preprocessing**
- Black border cropping
- Ben Graham's enhancement
- CLAHE application
- Circle cropping

**Stage 3: Model Training**
- Transfer learning (ImageNet)
- Focal Loss optimization
- Multi-phase fine-tuning

**Stage 4: Evaluation & Deployment**
- Comprehensive metrics
- Grad-CAM integration
- Web interface

---

## Slide 13: System Architecture

### Complete System Design

```
┌────────────────────────────────────────────────┐
│              INPUT LAYER                        │
│    Retinal Fundus Image (RGB)                  │
└────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────┐
│           PREPROCESSING PIPELINE                │
│  • Crop → Resize → Ben Graham → CLAHE          │
└────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────┐
│            DATA AUGMENTATION                    │
│  • Rotation • Flip • Brightness • Noise        │
└────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────┐
│         EfficientNet-B4 BACKBONE                │
│  • MBConv Blocks with SE Attention             │
│  • Global Average Pooling                      │
└────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────┐
│        CLASSIFICATION HEAD                      │
│  • Dropout → Dense(512) → Dense(5)             │
└────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────┐
│              OUTPUT                             │
│  • DR Class (0-4) + Confidence + Grad-CAM      │
└────────────────────────────────────────────────┘
```

---

## Slide 14: Preprocessing Pipeline Details

### Step-by-Step Preprocessing

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Black Border Cropping | Remove irrelevant regions |
| 2 | Resize to 512×512 | Standardize input size |
| 3 | Ben Graham's Method | Enhance local contrast |
| 4 | Circle Cropping | Focus on retinal region |
| 5 | CLAHE Enhancement | Adaptive histogram equalization |
| 6 | Normalization | Scale to [0, 1] |

### Ben Graham's Preprocessing Formula
```
Enhanced = 4 × Original - 4 × GaussianBlur(σ=10) + 128
```

### CLAHE Parameters
- Clip Limit: 2.0
- Tile Grid: 8×8
- Applied to L channel (LAB color space)

---

## Slide 15: Deep Learning Model - EfficientNet-B4

### Why EfficientNet-B4?

| Factor | Advantage |
|--------|-----------|
| Compound Scaling | Optimal depth, width, resolution balance |
| Parameters | 19M (manageable for training) |
| SE Attention | Channel-wise feature recalibration |
| Transfer Learning | Strong ImageNet pretrained weights |
| Performance | Proven on medical imaging tasks |

### Architecture Details

| Component | Specification |
|-----------|---------------|
| Input Size | 512×512×3 |
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Feature Dimension | 1792 |
| Classification Head | Dropout(0.4) → Dense(512) → Dropout(0.3) → Dense(5) |
| Activation | Swish (backbone), ReLU (head), Softmax (output) |

---

## Slide 16: Training Strategy

### Three-Phase Training Approach

| Phase | Epochs | Layers | Learning Rate | Purpose |
|-------|--------|--------|---------------|---------|
| 1 | 10 | Head only | 1e-3 | Feature extraction |
| 2 | 20 | Top 50% | 1e-4 | Fine-tuning |
| 3 | 10 | All | 1e-5 | Full optimization |

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Batch Size | 16 |
| LR Scheduler | Cosine Annealing with Warm Restarts |
| Loss Function | Focal Loss (γ=2.0) |

### Focal Loss
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
- Addresses class imbalance
- Down-weights easy examples
- Focuses on hard cases (severe DR)

---

## Slide 17: Grad-CAM for Clinical Interpretability

### What is Grad-CAM?

**Gradient-weighted Class Activation Mapping**
- Generates visual explanations for CNN decisions
- Highlights regions influencing classification
- Essential for clinical trust and adoption

### How It Works

1. **Forward Pass:** Get feature maps from last conv layer
2. **Backward Pass:** Compute gradients w.r.t. target class
3. **Weight Computation:** Global average pooling of gradients
4. **CAM Generation:** Weighted combination of activations
5. **Visualization:** Overlay heatmap on original image

### Clinical Importance
- 🔍 Shows which retinal features influenced the decision
- ✅ Validates AI reasoning matches clinical knowledge
- 👨‍⚕️ Helps ophthalmologists trust and verify predictions
- 📊 Identifies microaneurysms, hemorrhages, exudates

---

## Slide 18: Evaluation Metrics

### Comprehensive Evaluation Strategy

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Overall correct predictions | > 90% |
| **QWK** | Quadratic Weighted Kappa (ordinal agreement) | > 0.85 |
| **AUC-ROC** | Area under ROC curve | > 0.95 |
| **Sensitivity** | True positive rate (detect disease) | > 85% |
| **Specificity** | True negative rate (avoid false alarms) | > 90% |
| **F1-Score** | Harmonic mean of precision & recall | > 0.88 |
| **Confusion Matrix** | Class-wise performance analysis | - |

### Why QWK?
- DR classes are ordinal (0 < 1 < 2 < 3 < 4)
- Penalizes disagreements by severity distance
- Standard metric in DR detection competitions

---

## Slide 19: Expected Outcomes & Deliverables

### Technical Outcomes

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Classification Accuracy | > 90% | > 93% |
| Quadratic Weighted Kappa | > 0.85 | > 0.90 |
| AUC-ROC | > 0.95 | > 0.97 |
| Sensitivity (Referable DR) | > 85% | > 90% |
| Inference Time | < 500ms | < 200ms |

### Project Deliverables

1. ✅ **Trained Model** - Optimized weights and architecture
2. ✅ **Web Application** - User-friendly Streamlit interface
3. ✅ **Technical Report** - Comprehensive documentation
4. ✅ **Research Paper** - Publication-ready manuscript
5. ✅ **Source Code** - Well-documented, modular codebase

### Societal Impact
- 🌍 Improved accessibility in underserved areas
- 👁️ Early detection preventing vision loss
- 💰 Reduced healthcare costs
- 👨‍⚕️ Augmented specialist capabilities

---

## Slide 20: Implementation Timeline & Conclusion

### Project Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Setup | Environment, dataset, baseline code |
| 3-4 | Development | Working model, initial results |
| 5-6 | Optimization | Tuned model, ablation studies |
| 7-8 | Evaluation | Complete metrics, Grad-CAM |
| 9-10 | Deployment | Web interface, documentation |

### Technology Stack
- **Language:** Python 3.10+
- **Framework:** PyTorch 2.0
- **Libraries:** timm, OpenCV, Albumentations
- **Deployment:** Streamlit/Gradio
- **Tracking:** Weights & Biases

### Conclusion

This project aims to develop a **robust, accurate, and interpretable** deep learning system for diabetic retinopathy detection. By leveraging:
- State-of-the-art EfficientNet-B4 architecture
- Advanced preprocessing techniques
- Explainable AI through Grad-CAM

We will create a **clinical-grade screening tool** that can improve early detection and reduce preventable blindness worldwide.

---

## Thank You!

**Questions?**

---

**Student:** [Your Name]  
**Guide:** Dr. Usha Devi  
**Course:** BITE498J - Project II  
**Semester:** Winter 2025-26

---
