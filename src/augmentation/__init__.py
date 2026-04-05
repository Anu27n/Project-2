"""Augmentation modules for DR training."""

from .cgan import (
    ConditionalGANTrainer,
    CGANConfig,
    compute_generation_plan,
    load_training_dataframe,
    run_cgan_augmentation,
    save_augmented_dataframe,
)

__all__ = [
    "ConditionalGANTrainer",
    "CGANConfig",
    "compute_generation_plan",
    "load_training_dataframe",
    "run_cgan_augmentation",
    "save_augmented_dataframe",
]
