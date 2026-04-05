#!/usr/bin/env python
"""
Run DR detection model training from project root.

Usage:
    python run_train.py

Expects data in:
  data/aptos2019/train.csv + data/aptos2019/train_images/

If missing, run: python scripts/download_dataset.py
"""

import sys
from pathlib import Path

# Add src to path so imports work when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split


def find_data():
    """Locate train CSV and image directory. Returns (train_csv_path, image_dir) or (None, None)."""
    aptos = PROJECT_ROOT / "data" / "aptos2019"
    if (aptos / "train.csv").exists() and (aptos / "train_images").exists():
        return aptos / "train.csv", aptos / "train_images"
    return None, None


def print_split_distribution(df: pd.DataFrame, split_name: str):
    counts = df["diagnosis"].value_counts().sort_index()
    total = max(1, len(df))
    print(f"\n  {split_name} class distribution:")
    for cls in range(5):
        c = int(counts.get(cls, 0))
        pct = (100.0 * c) / total
        print(f"    Class {cls}: {c:4d} ({pct:5.1f}%)")


def main():
    csv_path, image_dir = find_data()
    if csv_path is None or image_dir is None:
        print("=" * 60)
        print("  No dataset found.")
        print("  Download APTOS 2019:  python scripts/download_dataset.py")
        print("  Place train.csv and train_images/ in data/aptos2019/")
        print("=" * 60)
        return 1

    df = pd.read_csv(csv_path)

    # Prefer cGAN-augmented labels if available and enabled in config.
    from config import TRAINING_CONFIG
    augmented_csv = csv_path.parent / "train_augmented.csv"
    if TRAINING_CONFIG.get("use_augmented_csv_if_available", True) and augmented_csv.exists():
        csv_path = augmented_csv
        df = pd.read_csv(csv_path)
        print(f"  Using augmented CSV: {csv_path}")

    # APTOS uses 'id_code'; some CSVs use 'id_code' or similar
    if "id_code" not in df.columns and "image" in df.columns:
        df = df.rename(columns={"image": "id_code"})
    if "diagnosis" not in df.columns:
        print("  CSV must have 'diagnosis' column.")
        return 1

    # Stratified split 80/20; for tiny datasets ensure at least 5 valid (1 per class)
    n = len(df)
    test_ratio = 0.2 if n >= 25 else max(0.1, 5 / n)
    train_df, valid_df = train_test_split(
        df, test_size=test_ratio, stratify=df["diagnosis"], random_state=42
    )

    img_dir_str = str(image_dir)
    print(f"  Train samples: {len(train_df)}  |  Valid samples: {len(valid_df)}")
    print(f"  Image directory: {img_dir_str}")

    # Import after path is set
    from dataset import create_data_loaders, compute_class_weights
    from models.efficientnet_model import EfficientNetDR
    from training.trainer import DRTrainer
    from config import (
        PREPROCESSING_CONFIG,
        TRAINING_PHASES,
        get_device,
        set_seed,
    )

    set_seed(TRAINING_CONFIG["seed"])
    device = get_device()

    print_split_distribution(train_df, "Train")
    print_split_distribution(valid_df, "Validation")

    class_weights = compute_class_weights(
        train_df["diagnosis"].to_numpy(),
        num_classes=5,
    ).tolist()
    weights_str = ", ".join(f"{w:.3f}" for w in class_weights)
    print(f"\n  Dynamic class weights (train split): [{weights_str}]")

    img_size = PREPROCESSING_CONFIG["image_size"]
    batch_size = min(TRAINING_CONFIG["batch_size"], len(train_df))
    if batch_size < 1:
        batch_size = 1
    num_workers = 0  # avoid multiprocessing issues with small data

    train_loader, valid_loader = create_data_loaders(
        train_df,
        valid_df,
        train_dir=img_dir_str,
        valid_dir=img_dir_str,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_weighted_sampler=TRAINING_CONFIG.get("use_weighted_sampler", True),
        enable_minority_synthesis=TRAINING_CONFIG.get("enable_minority_synthesis", True),
        minority_ratio_threshold=TRAINING_CONFIG.get("minority_ratio_threshold", 0.65),
        synthesis_probability=TRAINING_CONFIG.get("synthesis_probability", 0.35),
        pin_memory=TRAINING_CONFIG.get("pin_memory", True),
    )

    model = EfficientNetDR(
        num_classes=5,
        pretrained=True,
        dropout=0.4,
        use_attention=True,
        freeze_backbone=True,
    )

    config = {
        "num_classes": 5,
        "num_epochs": (
            TRAINING_PHASES["phase1"]["epochs"]
            + TRAINING_PHASES["phase2"]["epochs"]
            + TRAINING_PHASES["phase3"]["epochs"]
        ),
        "batch_size": batch_size,
        "use_amp": TRAINING_CONFIG.get("use_amp", True),
        "gradient_clip": 1.0,
        "weight_decay": 1e-4,
        "focal_gamma": TRAINING_CONFIG.get("focal_gamma", 2.0),
        "class_weights": class_weights,
        "early_stopping_patience": TRAINING_CONFIG.get("patience", 7),
        "early_stopping_min_delta": TRAINING_CONFIG.get("min_delta", 1e-4),
        "use_weighted_sampler": TRAINING_CONFIG.get("use_weighted_sampler", True),
        "enable_minority_synthesis": TRAINING_CONFIG.get("enable_minority_synthesis", True),
        "minority_ratio_threshold": TRAINING_CONFIG.get("minority_ratio_threshold", 0.65),
        "synthesis_probability": TRAINING_CONFIG.get("synthesis_probability", 0.35),
        "phases": {
            "phase1": {
                "name": TRAINING_PHASES["phase1"]["name"],
                "epochs": TRAINING_PHASES["phase1"]["epochs"],
                "lr": TRAINING_PHASES["phase1"]["lr"],
                "freeze_backbone": True,
                "unfreeze_fraction": 0.0,
            },
            "phase2": {
                "name": TRAINING_PHASES["phase2"]["name"],
                "epochs": TRAINING_PHASES["phase2"]["epochs"],
                "lr": TRAINING_PHASES["phase2"]["lr"],
                "freeze_backbone": False,
                "unfreeze_fraction": 0.5,
            },
            "phase3": {
                "name": TRAINING_PHASES["phase3"]["name"],
                "epochs": TRAINING_PHASES["phase3"]["epochs"],
                "lr": TRAINING_PHASES["phase3"]["lr"],
                "freeze_backbone": False,
                "unfreeze_fraction": 1.0,
            },
        },
    }

    output_dir = PROJECT_ROOT / "results" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = DRTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=str(output_dir),
    )

    print("\n  Starting 3-phase training...\n")
    history = trainer.train(train_loader, valid_loader)
    print("\n  Training finished. Checkpoints in:", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
