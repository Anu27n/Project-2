#!/usr/bin/env python
"""
Portable DR detection training entry point.

This refactor keeps existing 3-phase training logic intact while making the script:
  - OS portable (Windows/Linux/Kaggle/Colab)
  - GPU-ready with safe CPU fallback
  - Configurable via environment variables and CLI
  - Resume-capable from saved checkpoints
"""

import argparse
import copy
import os
import sys
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# KEY CHANGE: portable base path (no hardcoded Windows paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# KEY CHANGE: clean runtime config with env overrides
CONFIG: Dict[str, object] = {
    "data_dir": os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "aptos2019")),
    "save_dir": os.getenv("SAVE_DIR", os.path.join(BASE_DIR, "results", "models")),
    "batch_size": _env_int("BATCH_SIZE", 6),
    "epochs": _env_int("EPOCHS", 20),
    "device": os.getenv("DEVICE", "auto"),
    "num_workers": _env_int("NUM_WORKERS", 0),
    "cpu_threads": _env_int("CPU_THREADS", 4),
    "debug": _env_bool("DEBUG", False),
    "debug_fraction": _env_float("DEBUG_FRACTION", 0.20),
    "resume_checkpoint": os.getenv("RESUME_CHECKPOINT", ""),
    "auto_resume_last": _env_bool("AUTO_RESUME_LAST", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable DR training launcher")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--save-dir", type=str, default=None, help="Override checkpoint save directory")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--device", type=str, default=None, help="auto|cpu|cuda")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path or 'auto'")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (smaller dataset/fewer epochs)")
    parser.add_argument("--debug-fraction", type=float, default=None, help="Fraction of data to keep in debug mode")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> Dict[str, object]:
    runtime = dict(CONFIG)
    if args.data_dir:
        runtime["data_dir"] = args.data_dir
    if args.save_dir:
        runtime["save_dir"] = args.save_dir
    if args.batch_size is not None:
        runtime["batch_size"] = args.batch_size
    if args.epochs is not None:
        runtime["epochs"] = args.epochs
    if args.device:
        runtime["device"] = args.device
    if args.debug:
        runtime["debug"] = True
    if args.debug_fraction is not None:
        runtime["debug_fraction"] = args.debug_fraction
    if args.resume is not None:
        runtime["resume_checkpoint"] = args.resume
    return runtime


def find_data(data_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Locate train CSV and image directory."""
    csv_path = os.path.join(data_dir, "train.csv")
    image_dir = os.path.join(data_dir, "train_images")
    if os.path.exists(csv_path) and os.path.isdir(image_dir):
        return csv_path, image_dir
    return None, None


def resolve_device(device_pref: str) -> torch.device:
    """KEY CHANGE: robust device auto-detection for CPU/Kaggle/Colab."""
    pref = str(device_pref or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(pref)


def print_split_distribution(df: pd.DataFrame, split_name: str):
    counts = df["diagnosis"].value_counts().sort_index()
    total = max(1, len(df))
    print(f"\n  {split_name} class distribution:")
    for cls in range(5):
        c = int(counts.get(cls, 0))
        pct = (100.0 * c) / total
        print(f"    Class {cls}: {c:4d} ({pct:5.1f}%)")


def maybe_use_augmented_csv(csv_path: str, training_config: Dict[str, object]) -> str:
    augmented_csv = os.path.join(os.path.dirname(csv_path), "train_augmented.csv")
    if training_config.get("use_augmented_csv_if_available", True) and os.path.exists(augmented_csv):
        print(f"  Using augmented CSV: {augmented_csv}")
        return augmented_csv
    return csv_path


def apply_debug_subset(df: pd.DataFrame, enabled: bool, fraction: float) -> pd.DataFrame:
    """KEY CHANGE (BONUS): optional debug mode using a smaller stratified subset."""
    if not enabled:
        return df

    frac = min(max(float(fraction), 0.02), 1.0)
    grouped = []
    for _, grp in df.groupby("diagnosis"):
        n_keep = max(1, int(round(len(grp) * frac)))
        grouped.append(grp.sample(n=n_keep, random_state=42))

    debug_df = pd.concat(grouped, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"[DEBUG] Enabled -> using {len(debug_df)} / {len(df)} samples ({frac * 100:.1f}%)")
    return debug_df


def _scale_phase_epochs(phases: Dict[str, Dict[str, object]], total_epochs: int) -> Dict[str, Dict[str, object]]:
    """Scale phase epochs while preserving relative phase proportions."""
    phases_out = copy.deepcopy(phases)
    base_total = sum(int(phases_out[p]["epochs"]) for p in ["phase1", "phase2", "phase3"])
    target_total = max(3, int(total_epochs))
    if target_total == base_total:
        return phases_out

    ratio = target_total / max(base_total, 1)
    scaled = []
    for p in ["phase1", "phase2", "phase3"]:
        e = max(1, int(round(int(phases_out[p]["epochs"]) * ratio)))
        scaled.append(e)

    delta = target_total - sum(scaled)
    order = [1, 0, 2]  # favor phase2, then phase1, then phase3
    idx = 0
    while delta != 0:
        i = order[idx % len(order)]
        if delta > 0:
            scaled[i] += 1
            delta -= 1
        else:
            if scaled[i] > 1:
                scaled[i] -= 1
                delta += 1
        idx += 1

    phases_out["phase1"]["epochs"] = scaled[0]
    phases_out["phase2"]["epochs"] = scaled[1]
    phases_out["phase3"]["epochs"] = scaled[2]
    return phases_out


def resolve_resume_checkpoint(runtime: Dict[str, object], save_dir: str) -> Optional[str]:
    resume_arg = str(runtime.get("resume_checkpoint") or "").strip()

    if resume_arg.lower() == "auto":
        candidate = os.path.join(save_dir, "model_last.pth")
        return candidate if os.path.exists(candidate) else None

    if resume_arg:
        return os.path.abspath(resume_arg) if os.path.exists(resume_arg) else None

    if runtime.get("auto_resume_last", False):
        candidate = os.path.join(save_dir, "model_last.pth")
        return candidate if os.path.exists(candidate) else None

    return None


def main() -> int:
    args = parse_args()
    runtime = build_runtime_config(args)

    # KEY CHANGE: CPU optimization safe for local + Kaggle/Colab
    torch.set_num_threads(max(1, int(runtime["cpu_threads"])))

    csv_path, image_dir = find_data(str(runtime["data_dir"]))
    if csv_path is None or image_dir is None:
        print("=" * 70)
        print("  Dataset not found.")
        print(f"  Expected: {os.path.join(str(runtime['data_dir']), 'train.csv')}")
        print(f"            {os.path.join(str(runtime['data_dir']), 'train_images')}")
        print("  Download with: python scripts/download_dataset.py")
        print("=" * 70)
        return 1

    # Import after src path is configured
    from config import PREPROCESSING_CONFIG, TRAINING_CONFIG, TRAINING_PHASES, set_seed
    from dataset import create_data_loaders, compute_class_weights
    from models.efficientnet_model import EfficientNetDR
    from training.trainer import DRTrainer

    csv_path = maybe_use_augmented_csv(csv_path, TRAINING_CONFIG)
    df = pd.read_csv(csv_path)

    if "id_code" not in df.columns and "image" in df.columns:
        df = df.rename(columns={"image": "id_code"})
    if "diagnosis" not in df.columns:
        print("  CSV must have a diagnosis column.")
        return 1

    df = apply_debug_subset(df, bool(runtime["debug"]), float(runtime["debug_fraction"]))
    if len(df) < 10:
        print(f"  Not enough samples to train safely: {len(df)}")
        return 1

    # Keep existing split strategy
    n = len(df)
    test_ratio = 0.2 if n >= 25 else max(0.1, 5 / n)
    train_df, valid_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["diagnosis"],
        random_state=TRAINING_CONFIG.get("seed", 42),
    )

    set_seed(TRAINING_CONFIG.get("seed", 42))
    device = resolve_device(str(runtime["device"]))

    print("=" * 70)
    print("Portable training configuration")
    print("=" * 70)
    print(f"BASE_DIR          : {BASE_DIR}")
    print(f"CSV               : {csv_path}")
    print(f"Image directory   : {image_dir}")
    print(f"Save directory    : {runtime['save_dir']}")
    print(f"Device            : {device}")
    print(f"Batch size        : {runtime['batch_size']}")
    print(f"Total epochs cfg  : {runtime['epochs']}")
    print(f"Debug mode        : {runtime['debug']}")

    print_split_distribution(train_df, "Train")
    print_split_distribution(valid_df, "Validation")

    class_weights = compute_class_weights(train_df["diagnosis"].to_numpy(), num_classes=5).tolist()
    print(f"\n  Dynamic class weights: {[round(w, 3) for w in class_weights]}")

    img_size = int(PREPROCESSING_CONFIG["image_size"])
    batch_size = max(1, min(int(runtime["batch_size"]), len(train_df)))

    # KEY CHANGE: safe default workers across CPU/Kaggle/Colab
    num_workers = max(0, int(runtime["num_workers"]))
    pin_memory = bool(TRAINING_CONFIG.get("pin_memory", True)) and device.type == "cuda"

    train_loader, valid_loader = create_data_loaders(
        train_df,
        valid_df,
        train_dir=image_dir,
        valid_dir=image_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_weighted_sampler=TRAINING_CONFIG.get("use_weighted_sampler", True),
        enable_minority_synthesis=TRAINING_CONFIG.get("enable_minority_synthesis", True),
        minority_ratio_threshold=TRAINING_CONFIG.get("minority_ratio_threshold", 0.65),
        synthesis_probability=TRAINING_CONFIG.get("synthesis_probability", 0.35),
        pin_memory=pin_memory,
        max_samples_per_class=TRAINING_CONFIG.get("max_samples_per_class"),
        transform_use_clahe=TRAINING_CONFIG.get("transform_use_clahe", True),
        preprocess_apply_clahe=TRAINING_CONFIG.get("preprocess_apply_clahe", False),
    )

    model = EfficientNetDR(
        num_classes=5,
        pretrained=True,
        dropout=0.4,
        use_attention=True,
        freeze_backbone=True,
    )

    phases = _scale_phase_epochs(TRAINING_PHASES, int(runtime["epochs"]))
    trainer_config = {
        "num_classes": 5,
        "num_epochs": (
            int(phases["phase1"]["epochs"])
            + int(phases["phase2"]["epochs"])
            + int(phases["phase3"]["epochs"])
        ),
        "batch_size": batch_size,
        "use_amp": bool(TRAINING_CONFIG.get("use_amp", True)) and device.type == "cuda",
        "gradient_clip": 1.0,
        "weight_decay": float(TRAINING_CONFIG.get("weight_decay", 1e-4)),
        "focal_gamma": float(TRAINING_CONFIG.get("focal_gamma", 2.0)),
        "qwk_loss_weight": float(TRAINING_CONFIG.get("qwk_loss_weight", 0.30)),
        "mix_probability": float(TRAINING_CONFIG.get("mix_probability", 0.50)),
        "cutmix_probability": float(TRAINING_CONFIG.get("cutmix_probability", 0.50)),
        "mixup_alpha": float(TRAINING_CONFIG.get("mixup_alpha", 0.4)),
        "cutmix_alpha": float(TRAINING_CONFIG.get("cutmix_alpha", 1.0)),
        "min_lr": float(TRAINING_CONFIG.get("min_lr", 1e-5)),
        "class_weights": class_weights,
        "early_stopping_patience": int(TRAINING_CONFIG.get("patience", 7)),
        "early_stopping_min_delta": float(TRAINING_CONFIG.get("min_delta", 1e-4)),
        "phases": {
            "phase1": {
                "name": phases["phase1"]["name"],
                "epochs": int(phases["phase1"]["epochs"]),
                "lr": float(phases["phase1"]["lr"]),
                "img_size": int(phases["phase1"].get("img_size", 224)),
                "freeze_backbone": True,
                "unfreeze_fraction": 0.0,
                "scheduler_T0": int(phases["phase1"].get("scheduler_T0", 10)),
                "scheduler_Tmult": int(phases["phase1"].get("scheduler_Tmult", 1)),
            },
            "phase2": {
                "name": phases["phase2"]["name"],
                "epochs": int(phases["phase2"]["epochs"]),
                "lr": float(phases["phase2"]["lr"]),
                "img_size": int(phases["phase2"].get("img_size", 256)),
                "freeze_backbone": False,
                "unfreeze_fraction": 0.5,
                "scheduler_T0": int(phases["phase2"].get("scheduler_T0", 10)),
                "scheduler_Tmult": int(phases["phase2"].get("scheduler_Tmult", 2)),
            },
            "phase3": {
                "name": phases["phase3"]["name"],
                "epochs": int(phases["phase3"]["epochs"]),
                "lr": float(phases["phase3"]["lr"]),
                "img_size": int(phases["phase3"].get("img_size", 320)),
                "freeze_backbone": False,
                "unfreeze_fraction": 1.0,
                "scheduler_T0": int(phases["phase3"].get("scheduler_T0", 5)),
                "scheduler_Tmult": int(phases["phase3"].get("scheduler_Tmult", 1)),
            },
        },
    }

    output_dir = os.path.abspath(str(runtime["save_dir"]))
    os.makedirs(output_dir, exist_ok=True)

    trainer = DRTrainer(
        model=model,
        config=trainer_config,
        device=device,
        output_dir=output_dir,
    )

    resume_ckpt = resolve_resume_checkpoint(runtime, output_dir)
    if resume_ckpt is not None:
        # KEY CHANGE: explicit resume message requested
        print(f"\nResuming from checkpoint... {resume_ckpt}")

    print("\n  Starting 3-phase training...\n")
    trainer.train(train_loader, valid_loader, resume_from=resume_ckpt)
    print("\n  Training finished. Checkpoints in:", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
