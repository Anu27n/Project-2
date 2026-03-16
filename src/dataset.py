"""
Diabetic Retinopathy Detection - PyTorch Dataset Module

Custom Dataset class for loading and preprocessing retinal fundus images.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

from preprocessing import DRPreprocessor


class DRDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy classification.
    
    Attributes:
        df: DataFrame with 'id_code' and 'diagnosis' columns
        image_dir: Directory containing the images
        transform: Albumentations transform pipeline
        preprocessor: DRPreprocessor instance for image preprocessing
    """
    
    SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg']

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        preprocessor: Optional[DRPreprocessor] = None,
        img_size: int = 512,
        file_extension: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.img_size = img_size
        self.file_extension = file_extension or self._detect_extension()

        if preprocessor is None:
            self.preprocessor = DRPreprocessor(img_size=img_size)
        else:
            self.preprocessor = preprocessor

        self.num_classes = 5
        self.class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    def _detect_extension(self) -> str:
        """Auto-detect image file extension from the image directory."""
        for ext in self.SUPPORTED_EXTENSIONS:
            sample = self.df.iloc[0]['id_code']
            if (self.image_dir / f"{sample}{ext}").exists():
                return ext
        return '.png'

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image tensor, label)
        """
        row = self.df.iloc[idx]
        
        # Get image path
        image_name = str(row['id_code']) + self.file_extension
        image_path = self.image_dir / image_name
        
        # Load and preprocess image
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default: convert to tensor
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            image = image / 255.0
        
        # Get label
        label = int(row['diagnosis'])
        
        return image, label
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image."""
        if not image_path.exists():
            for ext in self.SUPPORTED_EXTENSIONS:
                alt = image_path.with_suffix(ext)
                if alt.exists():
                    image_path = alt
                    break
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        image = self.preprocessor.preprocess(image, normalize=False)
        
        return image.astype(np.uint8)
    
    def get_labels(self) -> np.ndarray:
        """Get all labels for computing class weights."""
        return self.df['diagnosis'].values
    
    def get_class_counts(self) -> dict:
        """Get count of samples per class."""
        counts = self.df['diagnosis'].value_counts().sort_index()
        return {self.class_names[i]: counts.get(i, 0) for i in range(self.num_classes)}


def get_train_transforms(img_size: int = 512) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations Compose pipeline
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
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=1
            ),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 20,
            max_width=img_size // 20,
            min_holes=2,
            fill_value=0,
            p=0.2
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_valid_transforms(img_size: int = 512) -> A.Compose:
    """
    Get validation/test transform pipeline (no augmentation).
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_data_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    train_dir: str,
    valid_dir: str,
    batch_size: int = 16,
    img_size: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_df: Training DataFrame
        valid_df: Validation DataFrame
        train_dir: Training images directory
        valid_dir: Validation images directory
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, valid_loader)
    """
    # Create datasets
    train_dataset = DRDataset(
        df=train_df,
        image_dir=train_dir,
        transform=get_train_transforms(img_size),
        img_size=img_size
    )
    
    valid_dataset = DRDataset(
        df=valid_df,
        image_dir=valid_dir,
        transform=get_valid_transforms(img_size),
        img_size=img_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader


def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    
    # Inverse frequency weighting
    weights = total / (num_classes * class_counts + 1e-6)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("DR Dataset Module - Example Usage")
    print("=" * 50)
    
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'id_code': ['0_no_dr_sample1', '1_mild_sample1', '2_moderate_sample1'],
        'diagnosis': [0, 1, 2]
    })
    
    print("\nSample DataFrame:")
    print(sample_df)
    
    # Create dataset
    dataset = DRDataset(
        df=sample_df,
        image_dir='../data/aptos2019/train_images',
        transform=None,
        img_size=512
    )
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names: {dataset.class_names}")
    print(f"Class counts: {dataset.get_class_counts()}")
    
    # Compute class weights
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    weights = compute_class_weights(labels)
    print(f"\nClass weights: {weights}")
