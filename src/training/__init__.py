"""
DR Detection — Training Package
"""

from .losses import (
    CombinedLoss,
    FocalLoss,
    LabelSmoothingCrossEntropyLoss,
    OrdinalRegressionLoss,
    WeightedFocalLoss,
    compute_class_weights_from_labels,
    get_loss_function,
)
from .trainer import (
    DRTrainer,
    EarlyStopping,
    MetricsTracker,
)

__all__ = [
    # Loss functions
    "FocalLoss",
    "WeightedFocalLoss",
    "LabelSmoothingCrossEntropyLoss",
    "OrdinalRegressionLoss",
    "CombinedLoss",
    "compute_class_weights_from_labels",
    "get_loss_function",
    # Trainer
    "DRTrainer",
    "EarlyStopping",
    "MetricsTracker",
]
