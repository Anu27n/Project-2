"""
Diabetic Retinopathy Detection - Image Preprocessing Module

This module contains all preprocessing functions for retinal fundus images.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional


class DRPreprocessor:
    """
    Comprehensive image preprocessing pipeline for Diabetic Retinopathy detection.
    
    Implements:
    - Black border cropping
    - Circle cropping
    - Ben Graham's preprocessing
    - CLAHE enhancement
    - Normalization
    """
    
    def __init__(
        self,
        img_size: int = 512,
        ben_graham_sigma: int = 10,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8
    ):
        """
        Initialize the preprocessor.
        
        Args:
            img_size: Target image size (square)
            ben_graham_sigma: Sigma for Gaussian blur in Ben Graham's method
            clahe_clip_limit: Clip limit for CLAHE
            clahe_tile_size: Tile grid size for CLAHE
        """
        self.img_size = img_size
        self.ben_graham_sigma = ben_graham_sigma
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            RGB image as numpy array
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def crop_black_borders(self, img: np.ndarray, tol: int = 7) -> np.ndarray:
        """
        Remove black borders around the retinal image.
        
        Args:
            img: Input image (RGB)
            tol: Tolerance threshold for black pixels
            
        Returns:
            Cropped image
        """
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            
            # Check if cropping is valid
            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:
                return img
            
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        
        return img
    
    def circle_crop(self, img: np.ndarray) -> np.ndarray:
        """
        Apply circular mask to focus on the retinal region.
        
        Args:
            img: Input image (RGB)
            
        Returns:
            Circle-cropped image
        """
        height, width = img.shape[:2]
        
        x = int(width / 2)
        y = int(height / 2)
        r = min(x, y)
        
        # Create circular mask
        circle_mask = np.zeros((height, width), np.uint8)
        cv2.circle(circle_mask, (x, y), int(r), 1, thickness=-1)
        
        # Apply mask
        img = cv2.bitwise_and(img, img, mask=circle_mask)
        return img
    
    def ben_graham_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """
        Ben Graham's preprocessing technique.
        
        Enhances local contrast by subtracting the Gaussian-blurred version
        of the image. This removes lighting variations and enhances features
        like blood vessels and lesions.
        
        Args:
            img: Input image (RGB)
            
        Returns:
            Enhanced image
        """
        img = cv2.addWeighted(
            img, 4,
            cv2.GaussianBlur(img, (0, 0), self.ben_graham_sigma),
            -4, 128
        )
        return img
    
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Enhances contrast locally while limiting amplification to reduce noise.
        Applied to the L channel in LAB color space.
        
        Args:
            img: Input image (RGB)
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return img
    
    def resize(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            img: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(img, (self.img_size, self.img_size))
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            img: Input image (uint8)
            
        Returns:
            Normalized image (float32)
        """
        return img.astype(np.float32) / 255.0
    
    def preprocess(
        self,
        image: Union[str, Path, np.ndarray],
        apply_ben_graham: bool = True,
        apply_circle_crop: bool = True,
        apply_clahe: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            image: Image path or numpy array
            apply_ben_graham: Whether to apply Ben Graham's preprocessing
            apply_circle_crop: Whether to apply circle cropping
            apply_clahe: Whether to apply CLAHE
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = self.load_image(image)
        else:
            img = image.copy()
        
        # Step 1: Crop black borders
        img = self.crop_black_borders(img)
        
        # Step 2: Resize
        img = self.resize(img)
        
        # Step 3: Ben Graham preprocessing
        if apply_ben_graham:
            img = self.ben_graham_preprocessing(img)
        
        # Step 4: Circle crop
        if apply_circle_crop:
            img = self.circle_crop(img)
        
        # Step 5: CLAHE
        if apply_clahe:
            img = self.apply_clahe(img)
        
        # Step 6: Normalize
        if normalize:
            img = self.normalize(img)
        
        return img
    
    def visualize_steps(
        self,
        image_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Visualize each preprocessing step.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        img = self.load_image(image_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('1. Original')
        axes[0, 0].axis('off')
        
        # After cropping
        img_cropped = self.crop_black_borders(img)
        img_cropped = self.resize(img_cropped)
        axes[0, 1].imshow(img_cropped)
        axes[0, 1].set_title('2. After Cropping & Resize')
        axes[0, 1].axis('off')
        
        # After Ben Graham
        img_ben = self.ben_graham_preprocessing(img_cropped)
        axes[0, 2].imshow(img_ben)
        axes[0, 2].set_title('3. Ben Graham Preprocessing')
        axes[0, 2].axis('off')
        
        # After Circle Crop
        img_circle = self.circle_crop(img_ben)
        axes[1, 0].imshow(img_circle)
        axes[1, 0].set_title('4. Circle Cropped')
        axes[1, 0].axis('off')
        
        # After CLAHE
        img_clahe = self.apply_clahe(img_circle)
        axes[1, 1].imshow(img_clahe)
        axes[1, 1].set_title('5. After CLAHE')
        axes[1, 1].axis('off')
        
        # Final normalized
        img_final = self.normalize(img_clahe)
        axes[1, 2].imshow(img_final)
        axes[1, 2].set_title('6. Final Normalized')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def batch_preprocess(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    img_size: int = 512,
    num_workers: int = 4
) -> None:
    """
    Preprocess all images in a directory.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save preprocessed images
        img_size: Target image size
        num_workers: Number of parallel workers
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = DRPreprocessor(img_size=img_size)
    
    # Get all image files
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    def process_single(image_path):
        try:
            img = preprocessor.preprocess(image_path, normalize=False)
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single, image_files),
            total=len(image_files),
            desc="Preprocessing images"
        ))
    
    success = sum(results)
    print(f"Successfully preprocessed {success}/{len(image_files)} images")


if __name__ == "__main__":
    # Example usage
    preprocessor = DRPreprocessor(img_size=512)
    
    print("DRPreprocessor initialized with:")
    print(f"  - Image size: {preprocessor.img_size}x{preprocessor.img_size}")
    print(f"  - Ben Graham sigma: {preprocessor.ben_graham_sigma}")
    print(f"  - CLAHE clip limit: {preprocessor.clahe_clip_limit}")
    print(f"  - CLAHE tile size: {preprocessor.clahe_tile_size}")
