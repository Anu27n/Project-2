# Diabetic Retinopathy Detection - Project Configuration

"""
Configuration settings for the DR Detection project
"""

import os
from pathlib import Path

# ============================================================
# PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET CONFIGURATION
# ============================================================
DATASET_CONFIG = {
    "name": "APTOS 2019 Blindness Detection",
    "num_classes": 5,
    "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
    "class_weights": [1.0, 4.88, 1.83, 9.50, 6.22],  # Inverse frequency weights
}

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
PREPROCESSING_CONFIG = {
    "image_size": 512,
    "clahe_clip_limit": 2.0,
    "clahe_tile_size": 8,
    "ben_graham_sigma": 10,
    "normalization_mean": [0.485, 0.456, 0.406],  # ImageNet stats
    "normalization_std": [0.229, 0.224, 0.225],
}

# ============================================================
# DATA AUGMENTATION
# ============================================================
AUGMENTATION_CONFIG = {
    "rotation_limit": 30,
    "shift_limit": 0.1,
    "scale_limit": 0.15,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "gaussian_noise_var": (10.0, 50.0),
    "coarse_dropout_prob": 0.3,
}

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_CONFIG = {
    "architecture": "efficientnet_b4",
    "pretrained": True,
    "dropout": 0.4,
    "num_classes": 5,
    "feature_dim": 1792,  # EfficientNet-B4 feature dimension
}

# Alternative architectures to experiment with
ALTERNATIVE_MODELS = [
    "efficientnet_b3",
    "efficientnet_b5",
    "efficientnetv2_m",
    "resnet50",
    "densenet121",
    "vit_base_patch16_384",
    "swin_base_patch4_window12_384",
]

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
TRAINING_CONFIG = {
    # General
    "seed": 42,
    "num_epochs": 30,
    "batch_size": 16,
    "num_workers": 4,
    "pin_memory": True,
    
    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "min_lr": 1e-6,
    
    # Scheduler
    "scheduler": "CosineAnnealingWarmRestarts",
    "T_0": 10,
    "T_mult": 2,
    
    # Loss
    "loss_function": "FocalLoss",
    "focal_gamma": 2.0,
    
    # Mixed Precision
    "use_amp": True,
    
    # Early Stopping
    "patience": 5,
    "min_delta": 0.001,
    
    # Cross Validation
    "n_folds": 5,
    "use_stratified": True,
}

# ============================================================
# TRAINING PHASES (Progressive Fine-tuning)
# ============================================================
TRAINING_PHASES = {
    "phase1": {
        "name": "Feature Extraction",
        "epochs": 10,
        "freeze_backbone": True,
        "lr": 1e-3,
    },
    "phase2": {
        "name": "Partial Fine-tuning",
        "epochs": 15,
        "freeze_backbone": False,
        "unfreeze_layers": 0.5,  # Top 50% of layers
        "lr": 1e-4,
    },
    "phase3": {
        "name": "Full Fine-tuning",
        "epochs": 5,
        "freeze_backbone": False,
        "unfreeze_layers": 1.0,  # All layers
        "lr": 1e-5,
    },
}

# ============================================================
# EVALUATION CONFIGURATION
# ============================================================
EVALUATION_CONFIG = {
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_roc",
        "quadratic_weighted_kappa",
        "sensitivity",
        "specificity",
    ],
    "target_metrics": {
        "accuracy": 0.90,
        "auc_roc": 0.95,
        "qwk": 0.85,
        "sensitivity": 0.85,
        "specificity": 0.90,
    },
}

# ============================================================
# GRAD-CAM CONFIGURATION
# ============================================================
GRADCAM_CONFIG = {
    "target_layer": "features.7",  # Last conv block for EfficientNet-B4
    "colormap": "jet",
    "alpha": 0.4,  # Overlay transparency
}

# ============================================================
# DEPLOYMENT CONFIGURATION
# ============================================================
DEPLOYMENT_CONFIG = {
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "max_file_size_mb": 10,
    "allowed_extensions": [".png", ".jpg", ".jpeg"],
    "inference_device": "cuda",  # or "cpu"
    "batch_inference": False,
}

# ============================================================
# EXPERIMENT TRACKING
# ============================================================
WANDB_CONFIG = {
    "project": "diabetic-retinopathy-detection",
    "entity": None,  # Set your wandb username/team
    "tags": ["efficientnet", "transfer-learning", "medical-imaging"],
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_device():
    """Get the best available device"""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("DIABETIC RETINOPATHY DETECTION - CONFIGURATION")
    print("=" * 60)
    print(f"\nModel: {MODEL_CONFIG['architecture']}")
    print(f"Image Size: {PREPROCESSING_CONFIG['image_size']}x{PREPROCESSING_CONFIG['image_size']}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"Device: {get_device()}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
