#!/usr/bin/env python
"""
Script to download the APTOS 2019 Blindness Detection Dataset from Kaggle.

Prerequisites:
1. Install Kaggle CLI: pip install kaggle
2. Get your Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New Token" under API section
   - Save the kaggle.json file to ~/.kaggle/kaggle.json
   - Run: chmod 600 ~/.kaggle/kaggle.json

Usage:
    python download_dataset.py
"""

import os
import subprocess
import sys
from pathlib import Path
import zipfile


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle credentials not found!")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' under API section")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def accept_competition_rules():
    """Remind user to accept competition rules."""
    print("\n⚠️  IMPORTANT: You must accept the competition rules first!")
    print("1. Go to: https://www.kaggle.com/c/aptos2019-blindness-detection/rules")
    print("2. Click 'I Understand and Accept' at the bottom")
    print("3. Then run this script again\n")


def download_aptos_dataset(output_dir: Path):
    """Download APTOS 2019 dataset from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading APTOS 2019 Blindness Detection Dataset...")
    print(f"   Output directory: {output_dir}")
    
    try:
        result = subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", "aptos2019-blindness-detection",
                "-p", str(output_dir)
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            if "403" in result.stderr or "Forbidden" in result.stderr:
                accept_competition_rules()
                return False
            elif "401" in result.stderr or "Unauthorized" in result.stderr:
                print("❌ Kaggle API authentication failed!")
                check_kaggle_credentials()
                return False
            else:
                print(f"❌ Download failed: {result.stderr}")
                return False
        
        print("✅ Download complete!")
        return True
        
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        return False


def extract_dataset(output_dir: Path):
    """Extract the downloaded zip file."""
    zip_path = output_dir / "aptos2019-blindness-detection.zip"
    
    if not zip_path.exists():
        print(f"❌ Zip file not found: {zip_path}")
        return False
    
    print("📦 Extracting dataset...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print("✅ Extraction complete!")
    
    # Remove zip file to save space
    zip_path.unlink()
    print("🗑️  Removed zip file to save space")
    
    return True


def verify_dataset(output_dir: Path):
    """Verify the dataset structure."""
    expected_files = [
        "train.csv",
        "test.csv",
        "train_images",
        "test_images"
    ]
    
    print("\n📋 Verifying dataset structure...")
    
    all_present = True
    for item in expected_files:
        path = output_dir / item
        if path.exists():
            if path.is_dir():
                count = len(list(path.glob("*.png")))
                print(f"   ✅ {item}/ ({count} images)")
            else:
                print(f"   ✅ {item}")
        else:
            print(f"   ❌ {item} NOT FOUND")
            all_present = False
    
    return all_present


def print_dataset_info(output_dir: Path):
    """Print dataset statistics."""
    import pandas as pd
    
    train_csv = output_dir / "train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        
        print("\n📊 Dataset Statistics:")
        print(f"   Total training images: {len(df)}")
        print("\n   Class distribution:")
        
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        for i, name in enumerate(class_names):
            count = len(df[df['diagnosis'] == i])
            pct = 100 * count / len(df)
            print(f"   {i} ({name}): {count} images ({pct:.1f}%)")


def main():
    """Main function to download and setup the dataset."""
    print("=" * 60)
    print("APTOS 2019 Blindness Detection Dataset Downloader")
    print("=" * 60)
    
    # Check credentials
    if not check_kaggle_credentials():
        return 1
    
    # Set output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "aptos2019"
    
    # Download
    if not download_aptos_dataset(output_dir):
        return 1
    
    # Extract
    if not extract_dataset(output_dir):
        return 1
    
    # Verify
    if not verify_dataset(output_dir):
        print("\n⚠️  Some files are missing!")
        return 1
    
    # Print info
    try:
        print_dataset_info(output_dir)
    except ImportError:
        pass
    
    print("\n" + "=" * 60)
    print("✅ Dataset ready for training!")
    print(f"   Location: {output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
