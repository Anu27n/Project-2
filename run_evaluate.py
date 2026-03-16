#!/usr/bin/env python
"""
Evaluate a trained DR detection model on the validation set.

Usage:
    python run_evaluate.py
    python run_evaluate.py --checkpoint results/models/model_best_qwk.pth
    python run_evaluate.py --checkpoint results/models/model_best_qwk.pth --save-figures
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def find_data():
    aptos = PROJECT_ROOT / "data" / "aptos2019"
    if (aptos / "train.csv").exists() and (aptos / "train_images").exists():
        return aptos / "train.csv", aptos / "train_images"
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate DR detection model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "results" / "models" / "model_best_qwk.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save evaluation figures to results/figures/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    csv_path, image_dir = find_data()
    if csv_path is None:
        print("No dataset found. Run: python scripts/download_dataset.py")
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run training first: python run_train.py")
        return 1

    from config import PREPROCESSING_CONFIG, get_device, set_seed
    from dataset import DRDataset, get_valid_transforms
    from evaluation.metrics import DRMetricsEvaluator
    from models.efficientnet_model import EfficientNetDR

    set_seed(42)
    device = get_device()

    df = pd.read_csv(csv_path)
    if "id_code" not in df.columns and "image" in df.columns:
        df = df.rename(columns={"image": "id_code"})

    _, valid_df = train_test_split(
        df, test_size=0.2, stratify=df["diagnosis"], random_state=42
    )
    print(f"  Evaluation samples: {len(valid_df)}")

    img_size = PREPROCESSING_CONFIG["image_size"]

    valid_dataset = DRDataset(
        df=valid_df,
        image_dir=str(image_dir),
        transform=get_valid_transforms(img_size),
        img_size=img_size,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = EfficientNetDR(num_classes=5, pretrained=False, dropout=0.4, use_attention=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"  Device: {device}")

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    evaluator = DRMetricsEvaluator(num_classes=5)
    results = evaluator.evaluate(y_true, y_pred, y_proba)
    evaluator.print_report(results)

    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_serializable = {
        k: v for k, v in results.items()
        if k not in ("roc_curves", "pr_curves", "report")
    }
    results_serializable["report"] = results.get("report", "")

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2, default=str)
    print(f"\n  Results saved to {output_dir / 'evaluation_results.json'}")

    if args.save_figures:
        figures_dir = PROJECT_ROOT / "results" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            evaluator.plot_all(results, save_dir=str(figures_dir))
            print(f"  Figures saved to {figures_dir}")
        except AttributeError:
            print("  (plot_all not available; skipping figure generation)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
