![VIT Logo](media/image1.png)

**SCHOOL OF COMPUTER SCIENCE ENGINEERING AND INFORMATION SYSTEMS**

**Winter Semester -- 2025-26**

**BITE498J -- Project II / Internship**

**B.Tech (IT) First Review**

---

| **Field** | **Details** |
|-----------|-------------|
| **Register Number** | |
| **Student Name** | |
| **Project Code (Course Code)** | BITE498J -- Project II / Internship |
| **Project Domain** | Machine Learning - Deep Learning, Computer Vision, Medical Image Analysis |
| **Project Title** | Deep Learning-Based Medical Image Analysis for Early Detection of Diabetic Retinopathy |
| **Guide Name** | Dr. Usha Devi |

---

## 1. Title & Abstract

### 1.1 Project Title
**Deep Learning-Based Medical Image Analysis for Early Detection of Diabetic Retinopathy**

### 1.2 Abstract

Diabetic Retinopathy (DR) is a leading cause of blindness among working-age adults worldwide, affecting millions annually. According to the International Diabetes Federation (2024), approximately 537 million adults globally are living with diabetes, and this number is projected to reach 783 million by 2045. Early detection and timely treatment can prevent vision loss in up to 95% of cases. However, manual screening by ophthalmologists is time-consuming, expensive, and requires significant expertise, leading to delayed diagnosis especially in resource-constrained settings. The existing screening methods are subject to human error due to fatigue and workload, and the shortage of trained specialists, particularly in rural areas, creates a critical healthcare gap.

This project proposes to develop an automated deep learning system using Convolutional Neural Networks (CNNs) to classify diabetic retinopathy severity levels from retinal fundus images. The proposed system will utilize transfer learning with EfficientNet-B4 architecture, trained on the APTOS 2019 Blindness Detection Dataset containing over 3,662 images across 5 severity classes (No DR, Mild, Moderate, Severe, and Proliferative DR). Advanced preprocessing techniques including Ben Graham's method and CLAHE (Contrast Limited Adaptive Histogram Equalization) will enhance model robustness. The system will integrate Grad-CAM visualization for clinical interpretability, allowing ophthalmologists to understand the AI decision-making process. The proposed solution aims to achieve classification accuracy exceeding 90% with AUC-ROC greater than 0.95. This automated screening tool will significantly reduce diagnosis time, improve accessibility in underserved regions, and assist medical professionals in making faster, more accurate diagnostic decisions, ultimately contributing to early intervention and prevention of vision loss.

### 1.3 Keywords
Deep Learning, Diabetic Retinopathy, Medical Image Analysis, Convolutional Neural Networks, Transfer Learning, EfficientNet, Computer Vision, Automated Diagnosis, Grad-CAM, Retinal Image Classification

---

## 2. Problem Definition & Literature Survey

### 2.1 Problem Definition

#### 2.1.1 Background
Diabetic Retinopathy (DR) is a microvascular complication of diabetes mellitus that damages the blood vessels in the retina, potentially leading to vision impairment and blindness. The World Health Organization (WHO) reports that DR is the leading cause of preventable blindness among working-age adults globally.

#### 2.1.2 Problem Statement
The manual grading of retinal fundus images for diabetic retinopathy detection presents several critical challenges:

1. **Labor-intensive and time-consuming**: Manual screening requires specialized ophthalmologists to examine each retinal image, taking 5-10 minutes per patient
2. **Subject to inter-observer variability**: Studies show 15-30% disagreement between expert graders
3. **Prone to fatigue-related errors**: Accuracy decreases significantly after reviewing multiple images
4. **Inaccessible in resource-limited settings**: Only 1 ophthalmologist per 100,000 people in many developing countries
5. **Unable to scale with the growing diabetic population**: Current infrastructure cannot meet screening demands

#### 2.1.3 Need for Automated Solution
There is an urgent need for an automated, accurate, and scalable screening solution that can:
- Assist healthcare providers in early DR detection
- Reduce screening bottlenecks
- Provide consistent, reproducible results
- Enable screening in underserved areas
- Support ophthalmologists in clinical decision-making

### 2.2 Literature Survey

#### 2.2.1 Evolution of DR Detection Methods

**Traditional Methods (Pre-2012)**
Early automated DR detection relied on handcrafted features:
- Morphological operations for vessel segmentation
- SIFT and SURF descriptors for lesion detection
- Template matching for microaneurysm identification
- Achieved moderate accuracy (70-80%) but struggled with image quality variations

**Deep Learning Era (2012-Present)**
The introduction of deep learning revolutionized medical image analysis with significant improvements in accuracy and robustness.

#### 2.2.2 Foundational Works

| Author | Year | Key Contribution | Performance |
|--------|------|------------------|-------------|
| Gulshan et al. | 2016 | First large-scale deep learning validation for DR detection using Inception-v3 | 97.5% sensitivity, 93.4% specificity |
| Gargeya & Leng | 2017 | Data-driven approach without handcrafted features | AUC 0.97 |
| He et al. | 2016 | ResNet - skip connections enabling deeper networks | Foundation for transfer learning |
| Tan & Le | 2019 | EfficientNet - compound scaling for optimal accuracy-efficiency | State-of-the-art on ImageNet |

#### 2.2.3 Recent Advances (2023-2025)

| Study | Year | Architecture | Dataset | Accuracy | AUC |
|-------|------|--------------|---------|----------|-----|
| Zhang et al. | 2024 | Vision Transformer | APTOS + IDRiD | 94.2% | 0.972 |
| Chen et al. | 2024 | EfficientNetV2-L | Multi-source | 93.8% | 0.968 |
| Kumar et al. | 2024 | Multi-Task CNN | APTOS | 93.1% | 0.962 |
| Zhou et al. | 2024 | RETFound (Foundation Model) | 1.6M images | 94.8% | 0.978 |
| Wang et al. | 2024 | DenseNet-169 | Kaggle DR | 91.3% | 0.948 |
| Patel et al. | 2024 | MobileViT-DR | APTOS | 89.7% | 0.941 |

#### 2.2.4 Key Research Trends

1. **Vision Transformers**: Hybrid CNN-Transformer architectures showing promising results with multi-scale feature fusion
2. **Multi-Task Learning**: Simultaneous detection of multiple DR-related lesions (microaneurysms, hemorrhages, exudates)
3. **Self-Supervised Learning**: Contrastive pretraining achieving 91.2% accuracy with only 10% labeled data
4. **Federated Learning**: Privacy-preserving training across multiple institutions (Li et al., 2024)
5. **Explainable AI**: Grad-CAM and attention maps for clinical interpretability

#### 2.2.5 Research Gap Identified

While existing systems achieve high accuracy, several gaps remain:
1. Class imbalance handling in severe DR cases (only 5.3% of images)
2. Generalization across different imaging equipment and populations
3. Real-time processing for point-of-care deployment
4. Integration with clinical workflows
5. Comprehensive explainability for clinical trust

---

## 3. Objectives & Scope of the Project

### 3.1 Primary Objective
Develop an automated deep learning system for multi-class classification of diabetic retinopathy severity (5 classes) from retinal fundus images.

### 3.2 Secondary Objectives

1. **Implement State-of-the-Art Architecture**: Utilize EfficientNet-B4 with transfer learning from ImageNet for optimal accuracy-efficiency trade-off

2. **Advanced Preprocessing Pipeline**: Apply Ben Graham's preprocessing and CLAHE enhancement for robust feature extraction

3. **Address Class Imbalance**: Implement Focal Loss and weighted sampling strategies to improve detection of severe DR cases

4. **Integrate Explainable AI**: Develop Grad-CAM visualization for clinical interpretability and trust

5. **Create Deployment-Ready System**: Build a user-friendly web interface for clinical use

6. **Achieve Clinical-Grade Performance**: Target metrics suitable for screening assistance

### 3.3 Scope of the Project

#### 3.3.1 In Scope

| Component | Description |
|-----------|-------------|
| **Dataset** | APTOS 2019 Blindness Detection Dataset (3,662 training images) |
| **Classification** | 5-class severity grading (No DR, Mild, Moderate, Severe, Proliferative) |
| **Architecture** | EfficientNet-B4 with custom classification head |
| **Preprocessing** | Ben Graham's method, CLAHE, circle cropping, augmentation |
| **Interpretability** | Grad-CAM visualization for model decisions |
| **Evaluation** | Accuracy, QWK, AUC-ROC, Sensitivity, Specificity, F1-Score |
| **Deployment** | Web-based interface using Streamlit/Gradio |

#### 3.3.2 Out of Scope

- Real-time video processing of retinal images
- Detection of other retinal diseases (glaucoma, AMD)
- Integration with hospital EHR systems
- FDA/regulatory approval processes
- Mobile application development

### 3.4 Expected Outcomes

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Classification Accuracy | > 90% | > 93% |
| Quadratic Weighted Kappa | > 0.85 | > 0.90 |
| AUC-ROC (Weighted Average) | > 0.95 | > 0.97 |
| Sensitivity (Referable DR) | > 85% | > 90% |
| Specificity | > 90% | > 93% |
| Inference Time | < 500ms | < 200ms |

### 3.5 Deliverables

1. Trained Deep Learning Model with optimized weights
2. Web Application for clinical use
3. Comprehensive Technical Report
4. Research Paper (publication-ready)
5. Well-documented source code

---

## 4. Proposed Architecture

### 4.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
│  Retinal Fundus Image (RGB, Variable Resolution)                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                            │
│  1. Image Quality Assessment                                        │
│  2. Black Border Cropping                                           │
│  3. Resizing to 512×512                                             │
│  4. Ben Graham's Preprocessing (Local Contrast Enhancement)         │
│  5. Circle Cropping (Focus on Retinal Region)                       │
│  6. CLAHE Enhancement (Adaptive Histogram Equalization)             │
│  7. Normalization ([0, 1] range)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA AUGMENTATION (Training Only)                 │
│  • Random Rotation (±30°)      • Horizontal/Vertical Flip           │
│  • Brightness/Contrast Adjust  • Random Zoom (0.9-1.1)              │
│  • Gaussian Noise              • ShiftScaleRotate                   │
│  • CoarseDropout               • Mixup (optional)                   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DEEP LEARNING MODEL                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  EfficientNet-B4 Backbone (ImageNet Pretrained)               │  │
│  │  ├── Stem: Conv 3×3 (48 filters), BatchNorm, Swish            │  │
│  │  ├── MBConv Blocks (1-7) with Squeeze-and-Excitation          │  │
│  │  │   • Block 1: 24 channels, stride 1                         │  │
│  │  │   • Block 2: 32 channels, stride 2                         │  │
│  │  │   • Block 3: 56 channels, stride 2                         │  │
│  │  │   • Block 4: 112 channels, stride 2                        │  │
│  │  │   • Block 5: 160 channels, stride 1                        │  │
│  │  │   • Block 6: 272 channels, stride 2                        │  │
│  │  │   • Block 7: 448 channels, stride 1                        │  │
│  │  ├── Conv Head: 1792 channels                                 │  │
│  │  └── Global Average Pooling                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Custom Classification Head                                   │  │
│  │  ├── Dropout (0.4)                                            │  │
│  │  ├── Dense Layer (1792 → 512, ReLU)                           │  │
│  │  ├── BatchNorm1d (512)                                        │  │
│  │  ├── Dropout (0.3)                                            │  │
│  │  └── Dense Layer (512 → 5, Softmax)                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                      │
│  ├── DR Severity Classification (0: No DR, 1: Mild, 2: Moderate,   │
│  │                               3: Severe, 4: Proliferative)       │
│  ├── Confidence Score (Softmax Probability)                        │
│  └── Grad-CAM Heatmap Visualization                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Preprocessing Pipeline Details

#### 4.2.1 Ben Graham's Preprocessing
```
Enhanced_Image = 4 × Original - 4 × GaussianBlur(Original, σ=10) + 128
```
- Subtracts local average color to normalize lighting variations
- Enhances visibility of blood vessels and lesions
- Critical for handling variable illumination conditions

#### 4.2.2 CLAHE Enhancement
- Applied to L channel of LAB color space
- Clip Limit: 2.0
- Tile Grid Size: 8×8
- Improves local contrast without amplifying noise

### 4.3 Model Training Strategy

#### Phase 1: Feature Extraction (10 epochs)
- Freeze EfficientNet-B4 backbone
- Train only classification head
- Learning Rate: 1e-3

#### Phase 2: Fine-tuning (20 epochs)
- Unfreeze top 50% of backbone layers
- Learning Rate: 1e-4

#### Phase 3: Full Fine-tuning (10 epochs)
- Unfreeze all layers
- Learning Rate: 1e-5 (with cosine annealing)

### 4.4 Loss Function: Focal Loss

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

Where:
- α_t: Class weights (inverse frequency)
- γ = 2.0: Focusing parameter
- p_t: Predicted probability for correct class

Focal Loss addresses class imbalance by down-weighting easy examples and focusing on hard, misclassified cases.

### 4.5 Grad-CAM for Interpretability

Grad-CAM generates visual explanations by:
1. Computing gradients of target class w.r.t. final convolutional layer
2. Global average pooling of gradients to get weights
3. Weighted combination of activation maps
4. ReLU to keep only positive contributions

This provides clinically interpretable heatmaps showing regions influencing the model's decision.

### 4.6 Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.10+ |
| Deep Learning Framework | PyTorch 2.0 / TensorFlow 2.15 |
| Model Library | timm (PyTorch Image Models) |
| Image Processing | OpenCV, Albumentations |
| Visualization | Matplotlib, Seaborn |
| Experiment Tracking | Weights & Biases |
| Web Interface | Streamlit / Gradio |
| Version Control | Git, GitHub |

---

## 5. Project Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1-2 | Setup | Literature review, environment setup, dataset preparation | Dataset ready, baseline code |
| 3-4 | Development | Baseline model implementation, preprocessing pipeline | Working model with initial results |
| 5-6 | Optimization | Hyperparameter tuning, ablation studies | Optimized model |
| 7-8 | Evaluation | Comprehensive evaluation, Grad-CAM integration | Complete metrics, visualizations |
| 9-10 | Deployment | Web interface, documentation, final testing | Deployment-ready system |

---

## 6. References (APA Format)

1. Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., ... & Webster, D. R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*, 316(22), 2402-2410.

2. Gargeya, R., & Leng, T. (2017). Automated identification of diabetic retinopathy using deep learning. *Ophthalmology*, 124(7), 962-969.

3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning*, 6105-6114.

4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *IEEE International Conference on Computer Vision*, 618-626.

5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

6. Zhang, Y., Li, H., Wang, Q., & Chen, X. (2024). Vision Transformer with Multi-Scale Feature Fusion for Diabetic Retinopathy Grading. *IEEE Transactions on Medical Imaging*, 43(3), 892-905.

7. Chen, L., Wang, Y., & Zhang, X. (2024). Interpretable Deep Learning for Diabetic Retinopathy: A Grad-CAM++ Approach with Clinical Validation. *Artificial Intelligence in Medicine*, 148, 102756.

8. Zhou, H., Wang, L., & Zhang, P. (2024). RETFound: A Foundation Model for Retinal Image Analysis. *Nature*, 622, 156-163.

9. Kumar, S., Sharma, A., & Patel, R. (2024). Multi-Task Deep Learning for Simultaneous DR Grading and Lesion Segmentation. *Computers in Biology and Medicine*, 168, 107723.

10. Guo, X., Li, Y., & Huang, T. (2024). Focal Loss with Adaptive Class Weights for Imbalanced Diabetic Retinopathy Classification. *Neural Networks*, 172, 106089.

11. Wang, R., Zhou, L., & Ma, J. (2024). Advanced Preprocessing Pipeline for Robust DR Classification Across Diverse Imaging Conditions. *IEEE Access*, 12, 45623-45638.

12. Alyoubi, W. L., Shalash, W. M., & Abulkhair, M. F. (2020). Diabetic retinopathy detection through deep learning techniques: A review. *Informatics in Medicine Unlocked*, 20, 100377.

13. Pratt, H., Coenen, F., Broadbent, D. M., Harding, S. P., & Zheng, Y. (2016). Convolutional neural networks for diabetic retinopathy. *Procedia Computer Science*, 90, 200-205.

14. Quellec, G., Charrière, K., Boudi, Y., Cochener, B., & Lamard, M. (2017). Deep image mining for diabetic retinopathy screening. *Medical Image Analysis*, 39, 178-193.

15. Li, Q., Chen, Y., & Xu, Z. (2024). Federated Deep Learning for Multi-Institutional Diabetic Retinopathy Detection. *Nature Communications*, 15, 2341.

---

**Prepared by:** [Student Name]  
**Registration Number:** [Register Number]  
**Guide:** Dr. Usha Devi  
**Date:** January 2026

---
