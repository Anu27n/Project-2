# Technical Methodology: Diabetic Retinopathy Detection System

## Detailed Implementation Guide

---

## 1. Data Acquisition and Preparation

### 1.1 Dataset Download

The APTOS 2019 Blindness Detection dataset is available on Kaggle:

```bash
# Using Kaggle API
kaggle competitions download -c aptos2019-blindness-detection

# Extract the dataset
unzip aptos2019-blindness-detection.zip -d data/
```

### 1.2 Dataset Structure

```
data/
├── train_images/           # 3,662 training images
│   ├── 0005cfc8afb6.png
│   ├── 001639a390f0.png
│   └── ...
├── test_images/            # 1,928 test images
├── train.csv               # id_code, diagnosis (0-4)
├── test.csv                # id_code
└── sample_submission.csv
```

### 1.3 Class Distribution Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training labels
train_df = pd.read_csv('data/train.csv')

# Class distribution
class_counts = train_df['diagnosis'].value_counts().sort_index()
print("Class Distribution:")
print(class_counts)

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'], 
        class_counts.values, color=['green', 'yellow', 'orange', 'red', 'darkred'])
plt.xlabel('DR Severity')
plt.ylabel('Number of Images')
plt.title('APTOS 2019 Class Distribution')
plt.savefig('results/class_distribution.png')
```

---

## 2. Image Preprocessing Pipeline

### 2.1 Complete Preprocessing Code

```python
import cv2
import numpy as np
from PIL import Image

class DRPreprocessor:
    def __init__(self, img_size=512):
        self.img_size = img_size
    
    def crop_image_from_gray(self, img, tol=7):
        """
        Remove black borders around the retinal image
        """
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            
            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:
                return img
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img
    
    def circle_crop(self, img):
        """
        Apply circular mask to focus on retinal region
        """
        height, width = img.shape[:2]
        
        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))
        
        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        
        img = cv2.bitwise_and(img, img, mask=circle_img)
        return img
    
    def ben_graham_preprocessing(self, img, sigmaX=10):
        """
        Ben Graham's preprocessing for enhanced contrast
        - Subtracts local average color to normalize lighting
        - Enhances blood vessels and lesions visibility
        """
        img = cv2.addWeighted(
            img, 4,
            cv2.GaussianBlur(img, (0, 0), sigmaX),
            -4, 128
        )
        return img
    
    def apply_clahe(self, img, clip_limit=2.0, tile_size=8):
        """
        Contrast Limited Adaptive Histogram Equalization
        Applied to each channel separately
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return img
    
    def preprocess(self, image_path):
        """
        Complete preprocessing pipeline
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step 1: Crop black borders
        img = self.crop_image_from_gray(img)
        
        # Step 2: Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Step 3: Ben Graham preprocessing
        img = self.ben_graham_preprocessing(img)
        
        # Step 4: Circle crop
        img = self.circle_crop(img)
        
        # Step 5: CLAHE
        img = self.apply_clahe(img)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
```

### 2.2 Preprocessing Comparison Visualization

```python
def visualize_preprocessing_steps(image_path):
    """
    Visualize each preprocessing step
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    preprocessor = DRPreprocessor()
    
    # Original
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    
    # After cropping
    img_cropped = preprocessor.crop_image_from_gray(img)
    img_cropped = cv2.resize(img_cropped, (512, 512))
    axes[0, 1].imshow(img_cropped)
    axes[0, 1].set_title('After Cropping')
    
    # After Ben Graham
    img_ben = preprocessor.ben_graham_preprocessing(img_cropped)
    axes[0, 2].imshow(img_ben)
    axes[0, 2].set_title('Ben Graham Preprocessing')
    
    # After Circle Crop
    img_circle = preprocessor.circle_crop(img_ben)
    axes[1, 0].imshow(img_circle)
    axes[1, 0].set_title('Circle Cropped')
    
    # After CLAHE
    img_clahe = preprocessor.apply_clahe(img_circle)
    axes[1, 1].imshow(img_clahe)
    axes[1, 1].set_title('After CLAHE')
    
    # Final normalized
    img_final = img_clahe.astype(np.float32) / 255.0
    axes[1, 2].imshow(img_final)
    axes[1, 2].set_title('Final Preprocessed')
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_steps.png', dpi=150)
    plt.show()
```

---

## 3. Data Augmentation

### 3.1 Augmentation Pipeline with Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=512):
    """
    Training augmentation pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=5, p=1),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1
            ),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 16,
            max_width=img_size // 16,
            min_holes=4,
            fill_value=0,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_valid_transforms(img_size=512):
    """
    Validation/test transforms (no augmentation)
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
```

---

## 4. Model Architecture

### 4.1 EfficientNet-B4 with Custom Head

```python
import torch
import torch.nn as nn
import timm

class DRClassifier(nn.Module):
    """
    EfficientNet-B4 based classifier for Diabetic Retinopathy
    """
    def __init__(self, num_classes=5, pretrained=True, dropout=0.4):
        super(DRClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # 1792 for B4
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.75),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out
    
    def extract_features(self, x):
        """Extract features for Grad-CAM"""
        return self.backbone(x)
```

### 4.2 Alternative: Vision Transformer

```python
class DRViTClassifier(nn.Module):
    """
    Vision Transformer based classifier
    """
    def __init__(self, num_classes=5, pretrained=True, dropout=0.3):
        super(DRViTClassifier, self).__init__()
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            'vit_base_patch16_384',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.feature_dim = self.backbone.num_features  # 768
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out
```

---

## 5. Loss Functions

### 5.1 Focal Loss for Class Imbalance

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Calculate class weights based on distribution
def calculate_class_weights(train_df):
    """
    Calculate inverse frequency weights for classes
    """
    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    total_samples = len(train_df)
    
    weights = total_samples / (len(class_counts) * class_counts)
    weights = torch.FloatTensor(weights)
    
    return weights
```

### 5.2 Quadratic Weighted Kappa Loss

```python
class QWKLoss(nn.Module):
    """
    Differentiable approximation of Quadratic Weighted Kappa
    """
    def __init__(self, num_classes=5):
        super(QWKLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, preds, targets):
        # Soft predictions
        preds = torch.softmax(preds, dim=1)
        
        # Weight matrix
        weights = torch.zeros((self.num_classes, self.num_classes), 
                             device=preds.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                weights[i, j] = ((i - j) ** 2) / ((self.num_classes - 1) ** 2)
        
        # Observed and expected matrices
        O = torch.matmul(
            torch.nn.functional.one_hot(targets, self.num_classes).float().T,
            preds
        )
        
        # Expected under random agreement
        hist_true = torch.sum(
            torch.nn.functional.one_hot(targets, self.num_classes).float(), 
            dim=0
        )
        hist_pred = torch.sum(preds, dim=0)
        E = torch.outer(hist_true, hist_pred) / len(targets)
        
        # QWK calculation
        num = torch.sum(weights * O)
        den = torch.sum(weights * E)
        
        qwk = 1 - (num / (den + 1e-8))
        
        return 1 - qwk  # Loss = 1 - QWK (to minimize)
```

---

## 6. Training Pipeline

### 6.1 Complete Training Script

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
import wandb

class DRTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Calculate class weights
        self.class_weights = calculate_class_weights(config['train_df'])
        self.class_weights = self.class_weights.to(self.device)
        
        # Loss function
        self.criterion = FocalLoss(alpha=self.class_weights, gamma=2.0)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['T_0'],
            T_mult=config['T_mult'],
            eta_min=config['min_lr']
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            self.scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        
        # Calculate QWK
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        return avg_loss, accuracy, qwk, all_preds, all_labels
    
    def fit(self, train_loader, val_loader, epochs):
        best_qwk = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_qwk, _, _ = self.validate(val_loader)
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, QWK: {val_qwk:.4f}')
            
            # Save best model
            if val_qwk > best_qwk:
                best_qwk = val_qwk
                torch.save(self.model.state_dict(), 'models/best_model.pth')
                print(f'Best model saved with QWK: {val_qwk:.4f}')
            
            # Log to wandb
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_qwk': val_qwk,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return best_qwk
```

---

## 7. Grad-CAM Visualization

### 7.1 Grad-CAM Implementation

```python
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), target_class

def visualize_gradcam(model, image_path, preprocessor, target_layer):
    """
    Visualize Grad-CAM for a retinal image
    """
    # Preprocess image
    img = preprocessor.preprocess(image_path)
    input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).cuda()
    
    # Generate CAM
    grad_cam = GradCAM(model, target_layer)
    cam, pred_class = grad_cam.generate_cam(input_tensor)
    
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    result = 0.6 * img + 0.4 * heatmap
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[2].imshow(result)
    axes[2].set_title(f'Overlay (Predicted: Class {pred_class})')
    
    plt.tight_layout()
    plt.savefig('results/gradcam_visualization.png', dpi=150)
    plt.show()
```

---

## 8. Evaluation Metrics

### 8.1 Comprehensive Evaluation

```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score
)

def comprehensive_evaluation(y_true, y_pred, y_prob):
    """
    Calculate all evaluation metrics
    """
    results = {}
    
    # Accuracy
    results['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1
    
    # Quadratic Weighted Kappa
    results['qwk'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # AUC-ROC (one-vs-rest)
    results['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    # Confusion Matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    results['classification_report'] = classification_report(
        y_true, y_pred,
        target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    )
    
    return results

def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.show()
```

---

## 9. Deployment

### 9.1 FastAPI Web Service

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import io
from PIL import Image

app = FastAPI(title="DR Detection API")

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict DR severity from retinal image
    """
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess
    preprocessor = DRPreprocessor()
    img = preprocessor.preprocess_pil(image)
    
    # Predict
    model.eval()
    with torch.no_grad():
        input_tensor = transform(img).unsqueeze(0).to(device)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    return PredictionResponse(
        class_id=pred_class,
        class_name=class_names[pred_class],
        confidence=probs[pred_class].item(),
        probabilities={name: prob.item() for name, prob in zip(class_names, probs)}
    )
```

### 9.2 Streamlit Interface

```python
import streamlit as st

st.title("🔬 Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image for automated DR severity classification")

uploaded_file = st.file_uploader("Choose a retinal image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
    
    if st.button("Analyze"):
        with st.spinner("Processing..."):
            # Preprocessing and prediction
            result = predict_image(image)
            
            st.success(f"**Predicted Class:** {result['class_name']}")
            st.info(f"**Confidence:** {result['confidence']:.2%}")
            
            # Show Grad-CAM
            st.subheader("Explainability - Grad-CAM")
            gradcam_img = generate_gradcam(image, model)
            st.image(gradcam_img, caption="Regions of Interest", use_column_width=True)
```

---

## 10. Project Directory Structure

```
Project-2/
├── README.md
├── requirements.txt
├── docs/
│   ├── project_proposal.md
│   ├── literature_review.md
│   └── methodology.md
├── data/
│   ├── raw/                    # Original images
│   ├── processed/              # Preprocessed images
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters
│   ├── dataset.py             # PyTorch Dataset
│   ├── preprocessing.py       # Image preprocessing
│   ├── augmentation.py        # Data augmentation
│   ├── model.py               # Model architectures
│   ├── losses.py              # Loss functions
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation metrics
│   └── gradcam.py             # Grad-CAM visualization
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory analysis
│   ├── 02_Preprocessing.ipynb # Preprocessing experiments
│   ├── 03_Training.ipynb      # Model training
│   └── 04_Evaluation.ipynb    # Results analysis
├── models/
│   └── best_model.pth         # Saved weights
├── results/
│   ├── figures/               # Plots and visualizations
│   └── metrics/               # Evaluation results
└── app/
    ├── api.py                 # FastAPI service
    └── streamlit_app.py       # Streamlit interface
```

---

*Document Version: 1.0 | Last Updated: January 2026*
