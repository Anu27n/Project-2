"""
Diabetic Retinopathy Detection - Loss Functions

This module implements custom loss functions optimized for the
highly imbalanced APTOS 2019 dataset:

    Class Distribution (approximate):
        Class 0 (No DR)        : ~49.5%  → 1,805 images
        Class 1 (Mild)         : ~10.2%  →   370 images
        Class 2 (Moderate)     : ~27.0%  →   999 images
        Class 3 (Severe)       :  ~5.2%  →   193 images
        Class 4 (Proliferative):  ~8.1%  →   295 images

Loss Functions Implemented:
    1. FocalLoss          - Down-weights easy examples, focuses on hard ones
    2. WeightedFocalLoss  - Focal Loss + class frequency weighting
    3. DifferentiableQWK  - Directly optimizes agreement-oriented behavior
    4. LabelSmoothingLoss - Prevents overconfident predictions
    5. OrdinalLoss        - Exploits ordinal nature of DR grading
    6. CombinedLoss       - Weighted combination of multiple losses
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. FOCAL LOSS
# ============================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Original paper: "Focal Loss for Dense Object Detection"
    (Lin et al., ICCV 2017)

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Key properties:
        - gamma = 0 → standard cross-entropy
        - Higher gamma → more focus on hard, misclassified examples
        - Downweights easy negatives (background / No DR cases)

    Args:
        alpha:  Weighting factor for rare classes. Can be:
                  - float  : scalar applied uniformly
                  - list   : per-class weights [alpha_0, ..., alpha_C]
                  - None   : no alpha weighting
        gamma:  Focusing parameter (default: 2.0)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int = 5,
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.register_buffer(
                "alpha", torch.tensor([alpha] * num_classes, dtype=torch.float32)
            )
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : Raw model outputs, shape (N, C)
            targets : Ground truth class indices, shape (N,)

        Returns:
            Scalar focal loss value
        """
        # Compute log softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)  # (N, C)
        probs = torch.exp(log_probs)  # (N, C)

        # Gather probabilities for the true class
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (N,)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (N,)

        # Focal modulating factor
        focal_weight = (1.0 - pt) ** self.gamma

        # Base focal loss
        loss = -focal_weight * log_pt  # (N,)

        # Apply per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


# ============================================================
# 2. WEIGHTED FOCAL LOSS  (Focal + class-frequency weights)
# ============================================================


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with inverse-frequency class weighting.

    Combines two complementary strategies:
      1. Focal mechanism  → hard example mining
      2. Class weighting  → explicit minority class upsampling

    Particularly effective for DR grading where:
      - Mild (class 1) and Severe (class 3) are highly under-represented
      - Misclassifying severe cases has higher clinical cost

    Args:
        class_weights : Tensor of per-class weights (inverse frequency).
                        If None, computed from dataset statistics.
        gamma         : Focal loss gamma parameter (default: 2.0)
        reduction     : 'mean' | 'sum' | 'none'
    """

    # Default APTOS 2019 inverse-frequency weights
    APTOS_WEIGHTS = [0.35, 1.70, 0.64, 3.32, 2.17]

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int = 5,
    ):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        if class_weights is None:
            weights = torch.tensor(self.APTOS_WEIGHTS, dtype=torch.float32)
        else:
            weights = class_weights.float()

        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Supports both hard labels (N,) and soft labels (N, C).
        Soft labels are required for MixUp/CutMix.
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        class_weights = self.class_weights.to(logits.device)

        if targets.ndim == 1:
            targets = targets.long()
            log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
            pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
            alpha_t = class_weights[targets]
        else:
            if targets.shape != logits.shape:
                raise ValueError(
                    "Soft targets must have shape (N, C) matching logits. "
                    f"Got {targets.shape} vs {logits.shape}."
                )
            soft_targets = targets.float()
            soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)

            log_pt = (soft_targets * log_probs).sum(dim=1)
            pt = (soft_targets * probs).sum(dim=1)
            alpha_t = (soft_targets * class_weights.unsqueeze(0)).sum(dim=1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -alpha_t * focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DifferentiableQWKLoss(nn.Module):
    """Differentiable surrogate of Quadratic Weighted Kappa loss."""

    def __init__(
        self,
        num_classes: int = 5,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super(DifferentiableQWKLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)

        if targets.ndim == 1:
            target_dist = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        else:
            if targets.shape[1] != self.num_classes:
                raise ValueError(
                    f"targets second dimension must be {self.num_classes}, got {targets.shape[1]}"
                )
            target_dist = targets.float()
            target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(self.eps)

        conf_mat = target_dist.t() @ probs
        conf_mat = conf_mat / conf_mat.sum().clamp_min(self.eps)

        hist_true = target_dist.sum(dim=0)
        hist_true = hist_true / hist_true.sum().clamp_min(self.eps)
        hist_pred = probs.sum(dim=0)
        hist_pred = hist_pred / hist_pred.sum().clamp_min(self.eps)

        expected = torch.outer(hist_true, hist_pred)
        expected = expected / expected.sum().clamp_min(self.eps)

        idx = torch.arange(self.num_classes, device=logits.device, dtype=torch.float32)
        denom_scale = float((self.num_classes - 1) ** 2) if self.num_classes > 1 else 1.0
        weight_matrix = (idx[:, None] - idx[None, :]) ** 2 / denom_scale

        numerator = (weight_matrix * conf_mat).sum()
        denominator = (weight_matrix * expected).sum().clamp_min(self.eps)

        loss = numerator / denominator

        if self.reduction == "sum":
            return loss
        if self.reduction == "none":
            return loss.unsqueeze(0)
        return loss


# ============================================================
# 3. LABEL SMOOTHING CROSS-ENTROPY LOSS
# ============================================================


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.

    Prevents the model from being overconfident by softening
    one-hot target distributions:
        y_smooth = (1 - eps) * y_onehot + eps / C

    Regularization benefit: reduces overfitting on small/noisy
    ophthalmology datasets where inter-grader agreement is ~0.6.

    Args:
        num_classes : Number of output classes
        smoothing   : Smoothing factor epsilon (default: 0.1)
        reduction   : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        num_classes: int = 5,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (N, C) raw model outputs
            targets : (N,)  integer class indices
        """
        log_probs = F.log_softmax(logits, dim=1)  # (N, C)

        # Smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.view(-1, 1), self.confidence)

        loss = -(smooth_targets * log_probs).sum(dim=1)  # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# 4. ORDINAL LOSS  (exploits DR grade ordering)
# ============================================================


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Cross-Entropy Loss for ordered DR severity grades.

    Motivation:
        DR severity is inherently ordinal:
            No DR < Mild < Moderate < Severe < Proliferative

        Standard CE treats all misclassifications equally, but
        confusing "Mild" with "Moderate" is less severe than
        confusing "No DR" with "Proliferative".

    Implementation (Frank & Hall, 2001):
        Decomposes K-class ordinal problem into K-1 binary
        classification tasks:
            P(Y > 0), P(Y > 1), P(Y > 2), P(Y > 3)

        Model needs 4 output nodes instead of 5.
        Each node predicts the probability of exceeding a threshold.

    Args:
        num_classes : Number of ordinal classes (default: 5)
        reduction   : 'mean' | 'sum' | 'none'
    """

    def __init__(self, num_classes: int = 5, reduction: str = "mean"):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1  # K-1 binary tasks
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (N, K-1) ordinal threshold logits
            targets : (N,)     integer class indices in [0, K-1]

        Returns:
            Ordinal regression loss
        """
        device = logits.device
        N = logits.size(0)

        # Build binary ordinal targets
        # For sample with class k: binary targets are [1,1,...,1,0,0,...,0]
        #                                              ^^^k ones^^^
        ordinal_targets = torch.zeros(N, self.num_tasks, device=device)
        for i in range(self.num_tasks):
            ordinal_targets[:, i] = (targets > i).float()

        # Binary cross-entropy on each threshold
        loss = F.binary_cross_entropy_with_logits(
            logits, ordinal_targets, reduction="none"
        )  # (N, K-1)

        loss = loss.sum(dim=1)  # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def decode_predictions(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert ordinal logits to class predictions.

        Args:
            logits : (N, K-1) ordinal threshold logits
        Returns:
            (N,) predicted class indices
        """
        probs = torch.sigmoid(logits)  # (N, K-1)
        # Class = number of thresholds exceeded
        preds = (probs > 0.5).sum(dim=1)
        return preds


# ============================================================
# 5. COMBINED LOSS
# ============================================================


class CombinedLoss(nn.Module):
    """
    Weighted combination of multiple loss functions.

    Recommended configuration for DR detection:
        - 60% Weighted Focal Loss (handles imbalance + hard examples)
        - 25% Label Smoothing CE  (regularization)
        - 15% Ordinal Loss        (exploits grade ordering)

    Args:
        focal_weight    : Weight for WeightedFocalLoss
        ce_weight       : Weight for LabelSmoothingCrossEntropyLoss
        ordinal_weight  : Weight for OrdinalRegressionLoss (0 to disable)
        class_weights   : Per-class frequency weights
        gamma           : Focal loss gamma
        smoothing       : Label smoothing epsilon
        num_classes     : Number of DR grades
    """

    def __init__(
        self,
        focal_weight: float = 0.60,
        ce_weight: float = 0.25,
        ordinal_weight: float = 0.15,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        num_classes: int = 5,
    ):
        super(CombinedLoss, self).__init__()

        assert abs(focal_weight + ce_weight + ordinal_weight - 1.0) < 1e-4, (
            f"Loss weights must sum to 1.0, got "
            f"{focal_weight + ce_weight + ordinal_weight:.4f}"
        )

        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.ordinal_weight = ordinal_weight
        self.use_ordinal = ordinal_weight > 0.0

        self.focal_loss = WeightedFocalLoss(
            class_weights=class_weights,
            gamma=gamma,
            num_classes=num_classes,
        )
        self.ce_loss = LabelSmoothingCrossEntropyLoss(
            num_classes=num_classes,
            smoothing=smoothing,
        )

        if self.use_ordinal:
            self.ordinal_loss = OrdinalRegressionLoss(num_classes=num_classes)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ordinal_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits         : (N, C) main classification logits
            targets        : (N,)   integer class labels
            ordinal_logits : (N, C-1) ordinal branch logits (optional)

        Returns:
            total_loss : Combined scalar loss
            loss_dict  : Dictionary of individual loss components
        """
        focal = self.focal_loss(logits, targets)
        ce = self.ce_loss(logits, targets)

        total = self.focal_weight * focal + self.ce_weight * ce

        loss_dict: Dict[str, float] = {
            "focal_loss": focal.item(),
            "ce_loss": ce.item(),
        }

        if self.use_ordinal and ordinal_logits is not None:
            ordinal = self.ordinal_loss(ordinal_logits, targets)
            total = total + self.ordinal_weight * ordinal
            loss_dict["ordinal_loss"] = ordinal.item()

        loss_dict["total_loss"] = total.item()

        return total, loss_dict


# ============================================================
# HELPER: Compute class weights from label array
# ============================================================


def compute_class_weights_from_labels(
    labels: np.ndarray,
    num_classes: int = 5,
    strategy: str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute per-class loss weights to handle imbalanced datasets.

    Args:
        labels      : 1-D array of integer class labels
        num_classes : Number of classes
        strategy    : Weight computation strategy:
                        'inverse_freq'     – w_c = N / (C * n_c)
                        'sqrt_inverse'     – w_c = sqrt(N / (C * n_c))
                        'effective_samples'– w_c = (1 - beta^n_c) / (1 - beta)
                                             (Cui et al., CVPR 2019)

    Returns:
        Normalized weight tensor of shape (num_classes,)
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = len(labels)

    if strategy == "inverse_freq":
        weights = total / (num_classes * (counts + 1e-6))

    elif strategy == "sqrt_inverse":
        weights = np.sqrt(total / (num_classes * (counts + 1e-6)))

    elif strategy == "effective_samples":
        beta = (total - 1.0) / total
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts + 1e-6))

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from: 'inverse_freq', 'sqrt_inverse', 'effective_samples'"
        )

    # Normalize so that mean weight = 1.0
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def get_loss_function(
    loss_name: str = "weighted_focal",
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    smoothing: float = 0.1,
    num_classes: int = 5,
    **kwargs,
) -> nn.Module:
    """
    Loss function factory.

    Args:
        loss_name     : One of 'focal', 'weighted_focal', 'ce_smooth',
                'qwk', 'ordinal', 'combined'
        class_weights : Per-class weights (used in focal losses)
        gamma         : Focal gamma parameter
        smoothing     : Label smoothing epsilon
        num_classes   : Number of classes
        **kwargs      : Additional arguments passed to the loss constructor

    Returns:
        Instantiated loss module
    """
    loss_name = loss_name.lower()

    registry = {
        "focal": lambda: FocalLoss(
            gamma=gamma,
            num_classes=num_classes,
        ),
        "weighted_focal": lambda: WeightedFocalLoss(
            class_weights=class_weights,
            gamma=gamma,
            num_classes=num_classes,
        ),
        "ce_smooth": lambda: LabelSmoothingCrossEntropyLoss(
            num_classes=num_classes,
            smoothing=smoothing,
        ),
        "qwk": lambda: DifferentiableQWKLoss(
            num_classes=num_classes,
        ),
        "ordinal": lambda: OrdinalRegressionLoss(
            num_classes=num_classes,
        ),
        "combined": lambda: CombinedLoss(
            class_weights=class_weights,
            gamma=gamma,
            smoothing=smoothing,
            num_classes=num_classes,
            **kwargs,
        ),
    }

    if loss_name not in registry:
        raise ValueError(
            f"Unknown loss function '{loss_name}'. Available: {list(registry.keys())}"
        )

    return registry[loss_name]()


# ============================================================
# MAIN — Smoke-test all loss functions
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DR Detection — Loss Functions Test")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cpu")

    N, C = 8, 5
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))

    print(f"\nBatch size : {N}")
    print(f"Classes    : {C}")
    print(f"Targets    : {targets.tolist()}\n")

    # ---- FocalLoss ----
    fl = FocalLoss(gamma=2.0, num_classes=C)
    fl_loss = fl(logits, targets)
    print(f"[1] FocalLoss                  : {fl_loss.item():.6f}")

    # ---- WeightedFocalLoss ----
    wfl = WeightedFocalLoss(gamma=2.0, num_classes=C)
    wfl_loss = wfl(logits, targets)
    print(f"[2] WeightedFocalLoss (APTOS)  : {wfl_loss.item():.6f}")

    # ---- Custom class weights ----
    labels_np = np.array([0] * 1805 + [1] * 370 + [2] * 999 + [3] * 193 + [4] * 295)
    cw = compute_class_weights_from_labels(labels_np, strategy="inverse_freq")
    print(f"\n[INFO] APTOS class weights (inverse_freq):")
    for i, w in enumerate(cw.tolist()):
        print(f"       Class {i}: {w:.4f}")

    wfl_custom = WeightedFocalLoss(class_weights=cw, gamma=2.0)
    wfl_custom_loss = wfl_custom(logits, targets)
    print(f"\n[3] WeightedFocalLoss (custom) : {wfl_custom_loss.item():.6f}")

    # ---- LabelSmoothingCE ----
    ls = LabelSmoothingCrossEntropyLoss(num_classes=C, smoothing=0.1)
    ls_loss = ls(logits, targets)
    print(f"[4] LabelSmoothingCE           : {ls_loss.item():.6f}")

    # ---- OrdinalLoss ----
    ordinal_logits = torch.randn(N, C - 1)
    ol = OrdinalRegressionLoss(num_classes=C)
    ol_loss = ol(ordinal_logits, targets)
    preds_ord = OrdinalRegressionLoss.decode_predictions(ordinal_logits)
    print(f"[5] OrdinalRegressionLoss      : {ol_loss.item():.6f}")
    print(f"    Ordinal predictions        : {preds_ord.tolist()}")

    # ---- CombinedLoss ----
    cl = CombinedLoss(
        focal_weight=0.60,
        ce_weight=0.25,
        ordinal_weight=0.15,
        class_weights=cw,
        gamma=2.0,
        smoothing=0.1,
    )
    total_loss, loss_dict = cl(logits, targets, ordinal_logits)
    print(f"\n[6] CombinedLoss breakdown:")
    for k, v in loss_dict.items():
        print(f"    {k:<18} : {v:.6f}")

    # ---- Factory ----
    print("\n[7] Loss factory test:")
    for name in ["focal", "weighted_focal", "ce_smooth", "combined"]:
        fn = get_loss_function(name, class_weights=cw, num_classes=C)
        print(f"    {name:<20} → {type(fn).__name__}")

    print("\n✅  All loss function tests passed!")
    print("=" * 60)
