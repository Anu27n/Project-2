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
    """Check if Kaggle credentials are configured via env var or kaggle.json."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("[OK] Using KAGGLE_API_TOKEN environment variable")
        return True

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print("[OK] Using kaggle.json credentials")
        return True

    print("[X] Kaggle credentials not found!")
    print("\nTo set up Kaggle API (choose one method):")
    print("  Method 1 - API Token env var:")
    print("    export KAGGLE_API_TOKEN=<your_token>")
    print("  Method 2 - kaggle.json file:")
    print("    1. Go to https://www.kaggle.com/settings")
    print("    2. Click 'Create New Token' under API section")
    print("    3. Save kaggle.json to ~/.kaggle/kaggle.json")
    return False


def accept_competition_rules():
    """Remind user to accept competition rules."""
    print("\n[!] IMPORTANT: You must accept the competition rules first!")
    print("1. Go to: https://www.kaggle.com/c/aptos2019-blindness-detection/rules")
    print("2. Click 'I Understand and Accept' at the bottom")
    print("3. Then run this script again\n")


def download_aptos_dataset(output_dir: Path):
    """Download APTOS 2019 dataset from Kaggle (pre-resized 224x224 version)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading APTOS 2019 dataset (224x224 pre-resized)...")
    print(f"   Output directory: {output_dir}")
    
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "sovitrath/diabetic-retinopathy-224x224-2019-data",
                "-p", str(output_dir)
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            if "401" in result.stderr or "Unauthorized" in result.stderr:
                print("[X] Kaggle API authentication failed!")
                check_kaggle_credentials()
                return False
            else:
                print(f"[X] Download failed: {result.stderr}")
                return False
        
        print("[OK] Download complete!")
        return True
        
    except FileNotFoundError:
        print("[X] Kaggle CLI not found. Install with: pip install kaggle")
        return False


def extract_dataset(output_dir: Path):
    """Extract and reorganize the downloaded zip into train_images/ + train.csv."""
    import shutil
    import csv

    zip_path = output_dir / "diabetic-retinopathy-224x224-2019-data.zip"
    
    if not zip_path.exists():
        print(f"[X] Zip file not found: {zip_path}")
        return False
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("[OK] Extraction complete!")

    src_root = output_dir / "colored_images"
    if src_root.exists():
        print("Reorganizing into train_images/ + train.csv ...")
        class_map = {
            "No_DR": 0, "Mild": 1, "Moderate": 2,
            "Severe": 3, "Proliferate_DR": 4,
        }
        train_dir = output_dir / "train_images"
        train_dir.mkdir(exist_ok=True)

        rows = []
        for class_name, label in class_map.items():
            class_dir = src_root / class_name
            if not class_dir.exists():
                continue
            for img in class_dir.glob("*.png"):
                shutil.copy2(str(img), str(train_dir / img.name))
                rows.append({"id_code": img.stem, "diagnosis": label})

        csv_path = output_dir / "train.csv"
        with open(str(csv_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id_code", "diagnosis"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["id_code"]))

        shutil.rmtree(str(src_root))
        print(f"[OK] Created train.csv ({len(rows)} images)")

    zip_path.unlink()
    print("Removed zip file to save space")
    return True


def verify_dataset(output_dir: Path):
    """Verify the dataset structure."""
    expected_files = [
        "train.csv",
        "test.csv",
        "train_images",
        "test_images"
    ]
    
    print("\nVerifying dataset structure...")
    
    all_present = True
    for item in expected_files:
        path = output_dir / item
        if path.exists():
            if path.is_dir():
                count = len(list(path.glob("*.png")))
                print(f"   [OK] {item}/ ({count} images)")
            else:
                print(f"   [OK] {item}")
        else:
            print(f"   [X] {item} NOT FOUND")
            all_present = False
    
    return all_present


def print_dataset_info(output_dir: Path):
    """Print dataset statistics."""
    import pandas as pd
    
    train_csv = output_dir / "train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        
        print("\nDataset Statistics:")
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
        print("\n[!] Some files are missing!")
        return 1
    
    # Print info
    try:
        print_dataset_info(output_dir)
    except ImportError:
        pass
    
    print("\n" + "=" * 60)
    print("[OK] Dataset ready for training!")
    print(f"   Location: {output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
