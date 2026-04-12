"""
Diabetic Retinopathy Detection - Training Pipeline

Full training pipeline with:
    - 3-phase progressive fine-tuning (freeze → partial unfreeze → full unfreeze)
    - Mixed precision training (AMP)
    - Cosine Annealing with Warm Restarts LR scheduler
    - Early stopping with configurable patience
    - Model checkpointing (best QWK, best accuracy, last epoch)
    - Per-epoch metrics logging
    - Gradient clipping for stable training

Usage:
    trainer = DRTrainer(model, config)
    history = trainer.train(train_loader, valid_loader)
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import WeightedFocalLoss, DifferentiableQWKLoss

try:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        roc_auc_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not found. Some metrics will be unavailable.")


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Early stopping monitor.

    Stops training if the monitored metric does not improve
    by at least `min_delta` for `patience` consecutive epochs.

    Args:
        patience   : Number of epochs to wait before stopping
        min_delta  : Minimum improvement to reset the counter
        mode       : 'min' for loss, 'max' for accuracy / QWK
        verbose    : Print status messages
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Update the monitor.

        Args:
            score : Current epoch metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("  [EarlyStopping] Triggered - stopping training.")

        return self.early_stop

    def reset(self):
        """Reset the counter (called at the start of each phase)."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# ============================================================
# METRICS TRACKER
# ============================================================

class MetricsTracker:
    """
    Tracks and accumulates batch-level metrics across an epoch.

    Computes epoch-level summaries for:
        - Loss
        - Accuracy
        - Quadratic Weighted Kappa (QWK)
        - AUC-ROC (macro OvR)
        - Per-class accuracy
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.losses: List[float] = []
        self.all_preds: List[int] = []
        self.all_targets: List[int] = []
        self.all_probs: List[np.ndarray] = []

    def update(
        self,
        loss: float,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ):
        """
        Update with a batch of predictions.

        Args:
            loss    : Batch loss value
            preds   : (N,) predicted class indices
            targets : (N,) ground truth class indices
            probs   : (N, C) softmax probabilities (optional, for AUC)
        """
        self.losses.append(loss)
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_targets.extend(targets.cpu().numpy().tolist())

        if probs is not None:
            self.all_probs.extend(probs.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Compute epoch-level metrics.

        Returns:
            Dictionary of metric name → value
        """
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        metrics: Dict[str, float] = {}

        # ---- Loss ----
        metrics["loss"] = float(np.mean(self.losses))

        # ---- Accuracy ----
        if SKLEARN_AVAILABLE:
            metrics["accuracy"] = float(accuracy_score(targets, preds))

            # ---- Quadratic Weighted Kappa ----
            try:
                metrics["qwk"] = float(
                    cohen_kappa_score(targets, preds, weights="quadratic")
                )
            except Exception:
                metrics["qwk"] = 0.0

            # ---- AUC-ROC (macro, one-vs-rest) ----
            if self.all_probs:
                probs_arr = np.array(self.all_probs)
                try:
                    all_labels = list(range(self.num_classes))
                    metrics["auc_roc"] = float(
                        roc_auc_score(
                            targets,
                            probs_arr,
                            multi_class="ovr",
                            average="macro",
                            labels=all_labels,
                        )
                    )
                except Exception as e:
                    print(f"  [AUC] Could not compute: {e}")
                    metrics["auc_roc"] = 0.0

            # ---- Per-class accuracy ----
            cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))
            with np.errstate(divide="ignore", invalid="ignore"):
                per_class = np.where(
                    cm.sum(axis=1) > 0,
                    cm.diagonal() / cm.sum(axis=1),
                    0.0,
                )
            for i, acc in enumerate(per_class):
                metrics[f"class_{i}_acc"] = float(acc)
        else:
            # Fallback manual accuracy
            metrics["accuracy"] = float((preds == targets).mean())
            metrics["qwk"] = 0.0

        return metrics


# ============================================================
# MAIN TRAINER
# ============================================================

class DRTrainer:
    """
    End-to-end trainer for Diabetic Retinopathy classification.

    Training Phases (Progressive Fine-tuning):
    ┌──────────────────────────────────────────────────────────┐
    │ Phase 1 │ Feature Extraction   │ Backbone FROZEN          │
    │         │ 10 epochs, lr=1e-3   │ Train head only          │
    ├──────────────────────────────────────────────────────────┤
    │ Phase 2 │ Partial Fine-tuning  │ Top 50% unfrozen         │
    │         │ 15 epochs, lr=1e-4   │ Lower LR for backbone    │
    ├──────────────────────────────────────────────────────────┤
    │ Phase 3 │ Full Fine-tuning     │ All layers trainable      │
    │         │ 5 epochs,  lr=1e-5   │ Minimal LR, careful tuning│
    └──────────────────────────────────────────────────────────┘

    Args:
        model        : EfficientNetDR model instance
        config       : Training configuration dictionary
        device       : Torch device (auto-detected if None)
        output_dir   : Directory to save checkpoints and logs
    """

    CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "results/models",
    ):
        # ---- Device ----
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)

        # ---- Configuration ----
        self.config = self._default_config()
        if config is not None:
            self.config.update(config)

        # ---- Output directory ----
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Mixed precision scaler ----
        self.use_amp = self.config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # ---- Loss functions ----
        class_weights = self.config.get("class_weights")
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.criterion = WeightedFocalLoss(
            class_weights=class_weights,
            gamma=self.config.get("focal_gamma", 2.0),
            num_classes=self.config.get("num_classes", 5),
        )
        self.qwk_loss = DifferentiableQWKLoss(
            num_classes=self.config.get("num_classes", 5),
        )
        self.qwk_loss_weight = float(np.clip(self.config.get("qwk_loss_weight", 0.30), 0.0, 1.0))

        # ---- Batch mixing controls ----
        self.mix_probability = float(np.clip(self.config.get("mix_probability", 0.50), 0.0, 1.0))
        self.cutmix_probability = float(np.clip(self.config.get("cutmix_probability", 0.50), 0.0, 1.0))
        self.mixup_alpha = float(self.config.get("mixup_alpha", 0.4))
        self.cutmix_alpha = float(self.config.get("cutmix_alpha", 1.0))

        # ---- Training state ----
        self.current_epoch = 0
        self.current_phase = 1
        self.best_qwk = 0.0
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.history: Dict[str, List] = {
            "train_loss": [], "train_accuracy": [], "train_qwk": [],
            "val_loss":   [], "val_accuracy":   [], "val_qwk":   [],
            "val_auc_roc": [], "learning_rate": [],
            "epoch_time": [],
        }

        # ---- Optimizer / Scheduler (initialized per phase) ----
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self._pending_resume_state: Optional[Dict[str, Any]] = None

        print(f"\n{'='*60}")
        print(f"  DRTrainer initialized")
        print(f"  Device      : {self.device}")
        print(f"  AMP enabled : {self.use_amp}")
        print(f"  Mix prob    : {self.mix_probability:.2f}")
        print(f"  CutMix prob : {self.cutmix_probability:.2f}")
        print(f"  QWK weight  : {self.qwk_loss_weight:.2f}")
        print(f"  Output dir  : {self.output_dir}")
        print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # DEFAULT CONFIGURATION
    # ----------------------------------------------------------

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "num_classes": 5,
            "num_epochs": 30,
            "batch_size": 16,
            "use_amp": True,
            "gradient_clip": 1.0,
            "weight_decay": 1e-4,
            "focal_gamma": 2.0,
            "qwk_loss_weight": 0.30,
            "mix_probability": 0.50,
            "cutmix_probability": 0.50,
            "mixup_alpha": 0.4,
            "cutmix_alpha": 1.0,
            "min_lr": 1e-5,
            "early_stopping_patience": 7,
            "early_stopping_min_delta": 1e-4,
            "phases": {
                "phase1": {
                    "name": "Feature Extraction",
                    "epochs": 10,
                    "lr": 1e-3,
                    "img_size": 224,
                    "freeze_backbone": True,
                    "unfreeze_fraction": 0.0,
                    "scheduler_T0": 10,
                    "scheduler_Tmult": 1,
                },
                "phase2": {
                    "name": "Partial Fine-tuning",
                    "epochs": 15,
                    "lr": 1e-4,
                    "img_size": 256,
                    "freeze_backbone": False,
                    "unfreeze_fraction": 0.5,
                    "scheduler_T0": 10,
                    "scheduler_Tmult": 2,
                },
                "phase3": {
                    "name": "Full Fine-tuning",
                    "epochs": 5,
                    "lr": 1e-5,
                    "img_size": 320,
                    "freeze_backbone": False,
                    "unfreeze_fraction": 1.0,
                    "scheduler_T0": 5,
                    "scheduler_Tmult": 1,
                },
            },
        }

    # ----------------------------------------------------------
    # OPTIMIZER AND SCHEDULER SETUP
    # ----------------------------------------------------------

    def _setup_optimizer(self, lr: float) -> AdamW:
        """Create AdamW optimizer with layer-wise learning rates."""
        backbone_params = [
            p for name, p in self.model.named_parameters()
            if "backbone" in name and p.requires_grad
        ]
        head_params = [
            p for name, p in self.model.named_parameters()
            if "backbone" not in name and p.requires_grad
        ]

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": lr * 0.1,  # Lower LR for pretrained backbone
                "weight_decay": self.config["weight_decay"],
            })
        param_groups.append({
            "params": head_params,
            "lr": lr,
            "weight_decay": self.config["weight_decay"],
        })

        return AdamW(param_groups)

    def _setup_scheduler(
        self, optimizer: AdamW, T_0: int, T_mult: int
    ) -> CosineAnnealingWarmRestarts:
        """Create Cosine Annealing with Warm Restarts scheduler."""
        min_lr = float(self.config.get("min_lr", 1e-5))
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=max(min_lr, 1e-7),
        )

    def _configure_phase(self, phase_key: str):
        """
        Configure model, optimizer, and scheduler for a training phase.

        Args:
            phase_key : 'phase1' | 'phase2' | 'phase3'
        """
        phase_cfg = self.config["phases"][phase_key]
        phase_name = phase_cfg["name"]
        lr = phase_cfg["lr"]
        T0 = phase_cfg.get("scheduler_T0", 10)
        Tmult = phase_cfg.get("scheduler_Tmult", 1)

        print(f"\n{'='*60}")
        print(f"  PHASE {phase_key[-1]}: {phase_name}")
        print(f"  LR       : {lr}")
        print(f"  Epochs   : {phase_cfg['epochs']}")
        if "img_size" in phase_cfg:
            print(f"  Img Size : {phase_cfg['img_size']}")
        print(f"  Min LR   : {self.config.get('min_lr', 1e-5)}")
        print(f"{'='*60}")

        # Configure backbone freezing
        if phase_cfg["freeze_backbone"]:
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()
            else:
                for p in self.model.parameters():
                    p.requires_grad = False
                for p in self.model.classifier.parameters():
                    p.requires_grad = True
        else:
            frac = phase_cfg.get("unfreeze_fraction", 1.0)
            if hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone(frac)
            else:
                for p in self.model.parameters():
                    p.requires_grad = True

        # Build optimizer and scheduler
        self.optimizer = self._setup_optimizer(lr)
        self.scheduler = self._setup_scheduler(self.optimizer, T0, Tmult)

        # If checkpoint resume happened before optimizer/scheduler existed,
        # restore those states now on first configured phase.
        if self._pending_resume_state is not None:
            opt_state = self._pending_resume_state.get("optimizer_state_dict")
            sch_state = self._pending_resume_state.get("scheduler_state_dict")

            if opt_state:
                try:
                    self.optimizer.load_state_dict(opt_state)
                    print("  [Resume] Optimizer state restored")
                except Exception as exc:
                    print(f"  [Resume] Optimizer state restore skipped: {exc}")

            if sch_state:
                try:
                    self.scheduler.load_state_dict(sch_state)
                    print("  [Resume] Scheduler state restored")
                except Exception as exc:
                    print(f"  [Resume] Scheduler state restore skipped: {exc}")

            self._pending_resume_state = None

        total_p = sum(p.numel() for p in self.model.parameters())
        train_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable params : {train_p:,} / {total_p:,}")

    def _to_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels.long(), num_classes=self.config["num_classes"]).float()

    @staticmethod
    def _rand_bbox(size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
        _, _, h, w = size
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        return int(x1), int(y1), int(x2), int(y2)

    def _apply_mixup(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mixup_alpha <= 0.0:
            return images, self._to_one_hot(labels)

        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        index = torch.randperm(images.size(0), device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[index]
        targets_a = self._to_one_hot(labels)
        targets_b = self._to_one_hot(labels[index])
        mixed_targets = lam * targets_a + (1.0 - lam) * targets_b

        return mixed_images, mixed_targets

    def _apply_cutmix(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cutmix_alpha <= 0.0:
            return images, self._to_one_hot(labels)

        lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        index = torch.randperm(images.size(0), device=images.device)

        x1, y1, x2, y2 = self._rand_bbox(images.size(), lam)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        cut_area = (x2 - x1) * (y2 - y1)
        lam_adjusted = 1.0 - (cut_area / float(images.size(-1) * images.size(-2) + 1e-8))

        targets_a = self._to_one_hot(labels)
        targets_b = self._to_one_hot(labels[index])
        mixed_targets = lam_adjusted * targets_a + (1.0 - lam_adjusted) * targets_b

        return mixed_images, mixed_targets

    def _maybe_apply_batch_mixing(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.size(0) < 2 or self.mix_probability <= 0.0:
            return images, labels

        if np.random.rand() > self.mix_probability:
            return images, labels

        if np.random.rand() < self.cutmix_probability:
            return self._apply_cutmix(images, labels)
        return self._apply_mixup(images, labels)

    @staticmethod
    def _unwrap_dataset(dataset: Any) -> Any:
        current = dataset
        visited = set()
        while hasattr(current, "dataset") and id(current) not in visited:
            visited.add(id(current))
            current = current.dataset
        return current

    def _set_dataset_img_size(self, dataset: Any, img_size: int, is_train: bool) -> None:
        base_dataset = self._unwrap_dataset(dataset)

        if hasattr(base_dataset, "img_size"):
            base_dataset.img_size = int(img_size)

        if hasattr(base_dataset, "preprocessor") and hasattr(base_dataset.preprocessor, "img_size"):
            base_dataset.preprocessor.img_size = int(img_size)

        if not hasattr(base_dataset, "transform"):
            return

        try:
            from dataset import get_train_transforms, get_valid_transforms

            base_dataset.transform = (
                get_train_transforms(int(img_size), use_clahe=True)
                if is_train
                else get_valid_transforms(int(img_size), use_clahe=True)
            )
        except Exception as exc:
            print(f"  [Resize] Transform update skipped: {exc}")

    def _apply_progressive_resize(
        self,
        phase_key: str,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> None:
        phase_cfg = self.config["phases"].get(phase_key, {})
        phase_img_size = phase_cfg.get("img_size")

        if phase_img_size is None:
            return

        phase_img_size = int(phase_img_size)
        self._set_dataset_img_size(train_loader.dataset, phase_img_size, is_train=True)
        self._set_dataset_img_size(valid_loader.dataset, phase_img_size, is_train=False)
        print(f"  Progressive resize applied -> {phase_img_size}x{phase_img_size}")

    # ----------------------------------------------------------
    # TRAINING AND VALIDATION STEPS
    # ----------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """
        Run a single training epoch.

        Args:
            loader : Training DataLoader
            epoch  : Current epoch number (for display)

        Returns:
            Dictionary of training metrics for this epoch
        """
        self.model.train()
        tracker = MetricsTracker(num_classes=self.config["num_classes"])

        pbar = tqdm(loader, desc=f"  Train Ep {epoch:03d}", leave=False, ncols=100)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            metric_targets = labels
            images, train_targets = self._maybe_apply_batch_mixing(images, labels)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                focal_loss = self.criterion(logits, train_targets)
                qwk_loss = self.qwk_loss(logits, train_targets)
                loss = ((1.0 - self.qwk_loss_weight) * focal_loss) + (self.qwk_loss_weight * qwk_loss)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.get("gradient_clip"):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Gather predictions (cast to float32 so softmax sums to 1.0 under AMP)
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=1)
                preds = probs.argmax(dim=1)

            tracker.update(loss.item(), preds, metric_targets, probs)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{(preds == metric_targets).float().mean().item():.3f}",
            })

        # Step scheduler once per epoch
        if self.scheduler is not None:
            self.scheduler.step()

        return tracker.compute()

    @torch.no_grad()
    def _validate_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """
        Run a single validation epoch.

        Args:
            loader : Validation DataLoader
            epoch  : Current epoch number (for display)

        Returns:
            Dictionary of validation metrics for this epoch
        """
        self.model.eval()
        tracker = MetricsTracker(num_classes=self.config["num_classes"])

        pbar = tqdm(loader, desc=f"  Valid Ep {epoch:03d}", leave=False, ncols=100)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                focal_loss = self.criterion(logits, labels)
                qwk_loss = self.qwk_loss(logits, labels)
                loss = ((1.0 - self.qwk_loss_weight) * focal_loss) + (self.qwk_loss_weight * qwk_loss)

            probs = F.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)

            tracker.update(loss.item(), preds, labels, probs)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{(preds == labels).float().mean().item():.3f}",
            })

        return tracker.compute()

    # ----------------------------------------------------------
    # CHECKPOINTING
    # ----------------------------------------------------------

    def _save_checkpoint(
        self,
        metrics: Dict[str, float],
        epoch: int,
        tag: str = "best_qwk",
    ):
        """
        Save model checkpoint.

        Args:
            metrics : Current epoch metrics
            epoch   : Current epoch number
            tag     : Filename tag ('best_qwk' | 'best_acc' | 'last')
        """
        checkpoint = {
            "epoch": epoch,
            "model_name": getattr(self.model, "model_name", None),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict":    self.scaler.state_dict(),
            "best_qwk":     self.best_qwk,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "metrics":      metrics,
            "config":       self.config,
            "history":      self.history,
        }

        save_path = self.output_dir / f"model_{tag}.pth"
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved -> {save_path} (epoch {epoch})")

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """
        Load a checkpoint and restore training state.

        Args:
            path   : Path to checkpoint .pth file
            strict : Strict weight loading

        Returns:
            Checkpoint metadata dictionary
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        opt_state = checkpoint.get("optimizer_state_dict")
        sch_state = checkpoint.get("scheduler_state_dict")

        if self.optimizer and opt_state:
            self.optimizer.load_state_dict(opt_state)
        if self.scheduler and sch_state:
            self.scheduler.load_state_dict(sch_state)

        # Defer optimizer/scheduler restore if they are not initialized yet.
        if (self.optimizer is None and opt_state) or (self.scheduler is None and sch_state):
            self._pending_resume_state = {
                "optimizer_state_dict": opt_state,
                "scheduler_state_dict": sch_state,
            }

        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.best_qwk     = checkpoint.get("best_qwk", 0.0)
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_loss= checkpoint.get("best_val_loss", float("inf"))
        self.history      = checkpoint.get("history", self.history)

        print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

    # ----------------------------------------------------------
    # PER-PHASE TRAINING LOOP
    # ----------------------------------------------------------

    def _run_phase(
        self,
        phase_key: str,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> bool:
        """
        Run a complete training phase.

        Args:
            phase_key    : 'phase1' | 'phase2' | 'phase3'
            train_loader : Training DataLoader
            valid_loader : Validation DataLoader

        Returns:
            True if early stopping was triggered, False otherwise
        """
        phase_cfg = self.config["phases"][phase_key]
        num_epochs = phase_cfg["epochs"]

        # Configure model and optimizers for this phase
        self._configure_phase(phase_key)
        self._apply_progressive_resize(phase_key, train_loader, valid_loader)

        # Reset early stopping per phase
        early_stopper = EarlyStopping(
            patience=self.config["early_stopping_patience"],
            min_delta=self.config["early_stopping_min_delta"],
            mode="max",   # maximize QWK
            verbose=True,
        )

        for epoch_in_phase in range(1, num_epochs + 1):
            self.current_epoch += 1
            epoch_start = time.time()

            print(f"\n  -- Epoch {self.current_epoch:03d} "
                  f"[Phase {phase_key[-1]}: {epoch_in_phase}/{num_epochs}] --")

            # ---- Train ----
            train_metrics = self._train_epoch(train_loader, self.current_epoch)

            # ---- Validate ----
            val_metrics = self._validate_epoch(valid_loader, self.current_epoch)

            epoch_time = time.time() - epoch_start

            # ---- Current LR ----
            current_lr = self.optimizer.param_groups[-1]["lr"]

            # ---- Update history ----
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics.get("accuracy", 0.0))
            self.history["train_qwk"].append(train_metrics.get("qwk", 0.0))
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
            self.history["val_qwk"].append(val_metrics.get("qwk", 0.0))
            self.history["val_auc_roc"].append(val_metrics.get("auc_roc", 0.0))
            self.history["learning_rate"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # ---- Print epoch summary ----
            self._print_epoch_summary(
                epoch=self.current_epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                elapsed=epoch_time,
            )

            # ---- Checkpointing ----
            val_qwk = val_metrics.get("qwk", 0.0)
            val_acc = val_metrics.get("accuracy", 0.0)
            val_loss = val_metrics["loss"]

            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                self._save_checkpoint(val_metrics, self.current_epoch, tag="best_qwk")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(val_metrics, self.current_epoch, tag="best_acc")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Always save the last epoch checkpoint
            self._save_checkpoint(val_metrics, self.current_epoch, tag="last")

            # ---- Early Stopping ----
            if early_stopper(val_qwk):
                print(f"\n  [EarlyStop] Triggered at epoch {self.current_epoch}")
                return True  # signal that ES fired

        return False  # phase completed normally

    # ----------------------------------------------------------
    # MAIN TRAIN ENTRY POINT
    # ----------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict[str, List]:
        """
        Execute the full 3-phase progressive fine-tuning.

        Args:
            train_loader : Training DataLoader
            valid_loader : Validation DataLoader
            resume_from  : Optional path to a checkpoint to resume from

        Returns:
            Training history dictionary
        """
        total_start = time.time()

        print("\n" + "=" * 60)
        print("  STARTING FULL TRAINING")
        print(f"  Phases  : {list(self.config['phases'].keys())}")
        print(f"  Device  : {self.device}")
        print(f"  AMP     : {self.use_amp}")
        print("=" * 60)

        if resume_from:
            print(f"\n  Resuming from checkpoint: {resume_from}")
            self.load_checkpoint(resume_from)

        for phase_key in ["phase1", "phase2", "phase3"]:
            self.current_phase = int(phase_key[-1])
            triggered = self._run_phase(phase_key, train_loader, valid_loader)

            if triggered:
                # Restore best weights before next phase
                best_ckpt = self.output_dir / "model_best_qwk.pth"
                if best_ckpt.exists():
                    print(f"\n  Restoring best weights from {best_ckpt}")
                    self.model.load_state_dict(
                        torch.load(best_ckpt, map_location=self.device, weights_only=False)["model_state_dict"]
                    )

        total_time = time.time() - total_start

        # ---- Final summary ----
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print(f"  Total time       : {total_time / 60:.1f} min")
        print(f"  Total epochs     : {self.current_epoch}")
        print(f"  Best Val Accuracy: {self.best_val_acc:.4f} ({self.best_val_acc*100:.2f}%)")
        print(f"  Best Val QWK     : {self.best_qwk:.4f}")
        print(f"  Best Val Loss    : {self.best_val_loss:.6f}")
        print("=" * 60)

        # Save training history as JSON
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n  Training history saved -> {history_path}")

        return self.history

    # ----------------------------------------------------------
    # DISPLAY HELPERS
    # ----------------------------------------------------------

    @staticmethod
    def _print_epoch_summary(
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        elapsed: float,
    ):
        """Pretty-print a one-line epoch summary."""
        t_loss = train_metrics.get("loss", 0)
        t_acc  = train_metrics.get("accuracy", 0)
        t_qwk  = train_metrics.get("qwk", 0)

        v_loss = val_metrics.get("loss", 0)
        v_acc  = val_metrics.get("accuracy", 0)
        v_qwk  = val_metrics.get("qwk", 0)
        v_auc  = val_metrics.get("auc_roc", 0)

        print(
            f"\n  Epoch {epoch:03d} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )
        print(
            f"  Train | Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | QWK: {t_qwk:.4f}"
        )
        print(
            f"  Val   | Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | QWK: {v_qwk:.4f} | AUC: {v_auc:.4f}"
        )
