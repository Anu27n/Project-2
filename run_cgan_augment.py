#!/usr/bin/env python
"""
Train a conditional GAN and generate minority-class synthetic images for APTOS 2019.

Usage examples:
    python run_cgan_augment.py
    python run_cgan_augment.py --epochs 60 --batch-size 64 --image-size 64
    python run_cgan_augment.py --ratio-threshold 0.7 --max-generate-per-class 1200
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def find_data_paths() -> tuple[Path | None, Path | None]:
    aptos = PROJECT_ROOT / "data" / "aptos2019"
    csv_path = aptos / "train.csv"
    image_dir = aptos / "train_images"

    if csv_path.exists() and image_dir.exists():
        return csv_path, image_dir
    return None, None


def main() -> int:
    from config import CGAN_CONFIG

    parser = argparse.ArgumentParser(
        description="Conditional GAN minority data augmentation for DR training"
    )
    parser.add_argument("--epochs", type=int, default=CGAN_CONFIG.get("epochs", 40), help="GAN training epochs")
    parser.add_argument("--batch-size", type=int, default=CGAN_CONFIG.get("batch_size", 64), help="GAN batch size")
    parser.add_argument("--image-size", type=int, default=CGAN_CONFIG.get("image_size", 64), help="GAN training image size")
    parser.add_argument("--latent-dim", type=int, default=CGAN_CONFIG.get("latent_dim", 128), help="Latent vector dimension")
    parser.add_argument("--embedding-dim", type=int, default=CGAN_CONFIG.get("embedding_dim", 64), help="Label embedding dimension")
    parser.add_argument("--lr", type=float, default=CGAN_CONFIG.get("learning_rate", 2e-4), help="Learning rate")
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=CGAN_CONFIG.get("ratio_threshold", 0.65),
        help="Classes below this fraction of majority count are treated as minority",
    )
    parser.add_argument(
        "--max-generate-per-class",
        type=int,
        default=CGAN_CONFIG.get("max_generate_per_class", None),
        help="Optional cap on generated images per minority class",
    )
    parser.add_argument(
        "--save-size",
        type=int,
        default=CGAN_CONFIG.get("save_size", 224),
        help="Saved synthetic image size for downstream classifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "aptos2019"),
        help="Directory for train_augmented.csv and cGAN summary",
    )
    args = parser.parse_args()

    csv_path, image_dir = find_data_paths()
    if csv_path is None or image_dir is None:
        print("Dataset not found. Expected data/aptos2019/train.csv and train_images/")
        return 1

    from augmentation.cgan import CGANConfig, run_cgan_augmentation

    gan_image_size = int(args.image_size)
    if gan_image_size != 64:
        print(
            f"Requested --image-size={gan_image_size} is not supported by the current cGAN architecture. "
            "Falling back to 64."
        )
        gan_image_size = 64

    config = CGANConfig(
        image_size=gan_image_size,
        latent_dim=max(16, int(args.latent_dim)),
        embedding_dim=max(8, int(args.embedding_dim)),
        batch_size=max(4, int(args.batch_size)),
        epochs=max(1, int(args.epochs)),
        learning_rate=float(args.lr),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Conditional GAN minority augmentation")
    print("=" * 70)
    print(f"CSV path     : {csv_path}")
    print(f"Image dir    : {image_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Epochs       : {config.epochs}")
    print(f"Batch size   : {config.batch_size}")
    print(f"Image size   : {config.image_size}")
    print(f"Latent dim   : {config.latent_dim}")
    print(f"Threshold    : {args.ratio_threshold}")

    summary = run_cgan_augmentation(
        csv_path=csv_path,
        image_dir=image_dir,
        output_dir=output_dir,
        config=config,
        ratio_threshold=float(args.ratio_threshold),
        max_generate_per_class=args.max_generate_per_class,
        save_size=max(64, int(args.save_size)),
    )

    print("\nResult summary")
    print("-" * 70)
    for k, v in summary.items():
        print(f"{k}: {v}")

    if summary.get("status") == "ok":
        print("\nCreated:")
        print(f"- {output_dir / 'train_augmented.csv'}")
        print(f"- {output_dir / 'cgan_augmentation_summary.json'}")
        print("Synthetic images are saved into train_images/ with synthetic_* names.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
