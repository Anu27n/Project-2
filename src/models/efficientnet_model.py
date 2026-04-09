"""
Diabetic Retinopathy Detection - EfficientNet-B4 Model Architecture

This module implements the EfficientNet-B4 based deep learning model
for multi-class classification of Diabetic Retinopathy severity levels.

Classes:
    - 0: No DR
    - 1: Mild NPDR
    - 2: Moderate NPDR
    - 3: Severe NPDR
    - 4: Proliferative DR
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# ATTENTION MODULE
# ============================================================


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (SE-Net style).
    Recalibrates channel-wise feature responses adaptively.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        reduced = max(in_channels // reduction_ratio, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM style).
    Focuses on informative regions in the feature map.
    """

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines channel and spatial attention for enhanced feature selection.
    Particularly useful for detecting small lesions in retinal images.
    """

    def __init__(
        self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7
    ):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================
# CLASSIFICATION HEAD
# ============================================================


class DRClassificationHead(nn.Module):
    """
    Custom classification head for DR severity classification.

    Features:
    - Multi-layer perceptron with batch normalization
    - Dropout for regularization
    - Optional ordinal regression support (DR is ordinal)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ):
        super(DRClassificationHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ============================================================
# MAIN MODEL: EfficientNet-B4 for DR Detection
# ============================================================


class EfficientNetDR(nn.Module):
    """
    EfficientNet-B4 based model for Diabetic Retinopathy detection.

    Architecture:
        1. EfficientNet-B4 backbone (pretrained on ImageNet)
        2. CBAM attention module on final feature map
        3. Global Average Pooling + Global Max Pooling (concatenated)
        4. Custom classification head with BN + Dropout
        5. 5-class softmax output

    Args:
        num_classes: Number of output classes (default: 5)
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout: Dropout rate in classification head (default: 0.4)
        use_attention: Apply CBAM attention (default: True)
        freeze_backbone: Freeze backbone weights (for Phase 1 training)
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.4,
        use_attention: bool = True,
        freeze_backbone: bool = False,
        model_name: str = "efficientnet_b4",
    ):
        super(EfficientNetDR, self).__init__()

        self.num_classes = num_classes
        self.use_attention = use_attention
        self.model_name = model_name

        # ---- Backbone ----
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool="",  # Remove global pooling (we do custom pooling)
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

        # ---- Attention ----
        if use_attention:
            self.attention = CBAM(
                in_channels=self.feature_dim, reduction_ratio=16, kernel_size=7
            )

        # ---- Pooling ----
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # ---- Classification Head ----
        # We concatenate avg and max pool outputs → 2x feature_dim
        self.classifier = DRClassificationHead(
            in_features=self.feature_dim * 2,
            num_classes=num_classes,
            hidden_dim=512,
            dropout=dropout,
        )

        # ---- Freeze backbone if requested ----
        if freeze_backbone:
            self.freeze_backbone()

        # ---- Weight initialization for classifier ----
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1: feature extraction)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[OK] Backbone frozen - training classifier head only")

    def unfreeze_backbone(self, unfreeze_fraction: float = 1.0):
        """
        Unfreeze backbone parameters progressively.

        Args:
            unfreeze_fraction: Fraction of layers to unfreeze from the top.
                               0.5 = top 50%, 1.0 = all layers
        """
        all_params = list(self.backbone.parameters())
        num_to_unfreeze = int(len(all_params) * unfreeze_fraction)
        params_to_unfreeze = all_params[-num_to_unfreeze:]

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in params_to_unfreeze:
            param.requires_grad = True

        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.backbone.parameters())
        print(
            f"[OK] Unfrozen {unfreeze_fraction * 100:.0f}% of backbone | "
            f"Trainable backbone params: {trainable:,} / {total:,}"
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (before classification head).
        Used for feature visualization and embedding analysis.
        """
        features = self.backbone.forward_features(x)

        if self.use_attention:
            features = self.attention(features)

        avg_feat = self.avg_pool(features).flatten(1)
        max_feat = self.max_pool(features).flatten(1)

        return torch.cat([avg_feat, max_feat], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Extract backbone features → (B, feature_dim, H', W')
        features = self.backbone.forward_features(x)

        # Apply CBAM attention
        if self.use_attention:
            features = self.attention(features)

        # Dual global pooling → (B, feature_dim) each
        avg_feat = self.avg_pool(features).flatten(1)
        max_feat = self.max_pool(features).flatten(1)

        # Concatenate → (B, feature_dim * 2)
        pooled = torch.cat([avg_feat, max_feat], dim=1)

        # Classification head → (B, num_classes)
        logits = self.classifier(pooled)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        return self.predict_proba(x).argmax(dim=1)

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def print_summary(self):
        """Print model summary."""
        params = self.count_parameters()
        print("=" * 60)
        print(f"  MODEL: {self.model_name.upper()} for DR Detection")
        print("=" * 60)
        print(f"  Backbone        : {self.model_name}")
        print(f"  Feature Dim     : {self.feature_dim}")
        print(f"  Pooled Dim      : {self.feature_dim * 2} (avg + max concat)")
        print(f"  CBAM Attention  : {self.use_attention}")
        print(f"  Output Classes  : {self.num_classes}")
        print(f"  Total Params    : {params['total']:,}")
        print(f"  Trainable Params: {params['trainable']:,}")
        print(f"  Frozen Params   : {params['frozen']:,}")
        print("=" * 60)


# ============================================================
# ENSEMBLE MODEL
# ============================================================


class EnsembleDRModel(nn.Module):
    """
    Ensemble of multiple EfficientNet variants for improved DR detection.

    Combines predictions from:
    - EfficientNet-B3
    - EfficientNet-B4
    - EfficientNet-B5

    Ensemble strategy: Soft voting (average of softmax probabilities)
    """

    def __init__(
        self,
        num_classes: int = 5,
        model_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ):
        super(EnsembleDRModel, self).__init__()

        if model_names is None:
            model_names = ["efficientnet_b3", "efficientnet_b4", "efficientnet_b5"]

        self.model_names = model_names
        self.weights = weights or [1.0 / len(model_names)] * len(model_names)

        # Create individual models
        self.models = nn.ModuleList(
            [
                EfficientNetDR(
                    num_classes=num_classes, pretrained=True, model_name=name
                )
                for name in model_names
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average of softmax probabilities from all models."""
        probs = torch.stack(
            [
                F.softmax(model(x), dim=1) * w
                for model, w in zip(self.models, self.weights)
            ],
            dim=0,
        )

        ensemble_probs = probs.sum(dim=0)
        # Return log probabilities as logits equivalent
        return torch.log(ensemble_probs + 1e-8)

    def load_model_weights(self, model_paths: List[str], device: torch.device):
        """Load saved weights for each model in the ensemble."""
        for model, path in zip(self.models, model_paths):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[OK] Loaded weights from {path}")


# ============================================================
# MODEL FACTORY
# ============================================================


def infer_timm_efficientnet_name(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Guess timm efficientnet_* name from weights (conv stem + head + backbone size).

    B0 and B1 share the same stem/head tensor shapes in timm; we use backbone param count.
    """
    prefix = "backbone."
    stem = state_dict.get(prefix + "conv_stem.weight")
    head = state_dict.get(prefix + "conv_head.weight")
    if stem is None or head is None:
        return "efficientnet_b4"
    stem_c = int(stem.shape[0])
    out_c = int(head.shape[0])
    in_c = int(head.shape[1])
    backbone_n = sum(
        int(v.numel())
        for k, v in state_dict.items()
        if k.startswith(prefix) and isinstance(v, torch.Tensor)
    )
    if stem_c == 32 and out_c == 1280 and in_c == 320:
        return "efficientnet_b0" if backbone_n < 5_300_000 else "efficientnet_b1"
    if stem_c == 32 and out_c == 1408 and in_c == 352:
        return "efficientnet_b2"
    if stem_c == 40 and out_c == 1536 and in_c == 384:
        return "efficientnet_b3"
    if stem_c == 48 and out_c == 1792 and in_c == 448:
        return "efficientnet_b4"
    if stem_c == 48 and out_c == 2048 and in_c == 512:
        return "efficientnet_b5"
    return "efficientnet_b4"


def load_efficientnet_dr_from_checkpoint(
    checkpoint_path: Union[str, Path],
    *,
    map_location=None,
    num_classes: int = 5,
    dropout: float = 0.4,
    use_attention: bool = True,
    weights_only: bool = False,
) -> Tuple[EfficientNetDR, Dict]:
    """
    Build EfficientNetDR to match a saved checkpoint (architecture inferred if needed).
    """
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location=map_location, weights_only=weights_only)
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict):
        raise ValueError(f"No model_state_dict in checkpoint: {path}")
    model_name = ckpt.get("model_name") or infer_timm_efficientnet_name(sd)
    model = EfficientNetDR(
        num_classes=num_classes,
        pretrained=False,
        dropout=dropout,
        use_attention=use_attention,
        model_name=str(model_name),
    )
    if map_location is not None:
        model = model.to(map_location)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ckpt


def build_model(
    architecture: str = "efficientnet_b4",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.4,
    use_attention: bool = True,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None,
) -> EfficientNetDR:
    """
    Factory function to build the DR detection model.

    Args:
        architecture: Model architecture name (timm-compatible)
        num_classes: Number of DR severity classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
        use_attention: Apply CBAM attention module
        freeze_backbone: Freeze backbone for Phase 1 training
        device: Target device (auto-detect if None)

    Returns:
        Initialized EfficientNetDR model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetDR(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        use_attention=use_attention,
        freeze_backbone=freeze_backbone,
        model_name=architecture,
    )

    model = model.to(device)
    model.print_summary()

    return model


def load_checkpoint(
    model: EfficientNetDR,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> Tuple[EfficientNetDR, Dict]:
    """
    Load model from checkpoint.

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Target device
        strict: Strict weight loading

    Returns:
        Tuple of (model with loaded weights, checkpoint metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "best_val_acc": checkpoint.get("best_val_acc", 0.0),
        "best_qwk": checkpoint.get("best_qwk", 0.0),
    }

    print(f"[OK] Loaded checkpoint from epoch {metadata['epoch']}")
    print(f"   Best Val Accuracy : {metadata['best_val_acc']:.4f}")
    print(f"   Best QWK          : {metadata['best_qwk']:.4f}")

    return model, metadata


# ============================================================
# AVAILABLE ARCHITECTURES INFO
# ============================================================

SUPPORTED_ARCHITECTURES = {
    "efficientnet_b3": {
        "feature_dim": 1536,
        "params_M": 12.0,
        "image_size": 300,
        "description": "Lightweight - good for quick experiments",
    },
    "efficientnet_b4": {
        "feature_dim": 1792,
        "params_M": 19.3,
        "image_size": 380,
        "description": "Primary model - best accuracy/speed tradeoff",
    },
    "efficientnet_b5": {
        "feature_dim": 2048,
        "params_M": 30.4,
        "image_size": 456,
        "description": "Higher accuracy - requires more GPU memory",
    },
    "efficientnetv2_m": {
        "feature_dim": 1280,
        "params_M": 54.1,
        "image_size": 480,
        "description": "EfficientNetV2 - faster training convergence",
    },
    "resnet50": {
        "feature_dim": 2048,
        "params_M": 25.6,
        "image_size": 224,
        "description": "Baseline comparison model",
    },
    "densenet121": {
        "feature_dim": 1024,
        "params_M": 8.0,
        "image_size": 224,
        "description": "Dense connections - good feature reuse",
    },
}


def list_supported_architectures():
    """Print all supported model architectures."""
    print("\n" + "=" * 65)
    print(f"  {'Architecture':<22} {'Features':>10} {'Params (M)':>12} {'ImgSize':>9}")
    print("=" * 65)
    for name, info in SUPPORTED_ARCHITECTURES.items():
        print(
            f"  {name:<22} "
            f"{info['feature_dim']:>10,} "
            f"{info['params_M']:>12.1f} "
            f"{info['image_size']:>9}"
        )
        print(f"    → {info['description']}")
    print("=" * 65)


# ============================================================
# MAIN - Example Usage & Model Verification
# ============================================================

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 65)
    print("  DR Detection - EfficientNet Model Module")
    print("=" * 65)

    # List supported architectures
    list_supported_architectures()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Build primary model
    print("\n📦 Building EfficientNet-B4 model...")
    model = build_model(
        architecture="efficientnet_b4",
        num_classes=5,
        pretrained=False,  # False for testing (no download needed)
        dropout=0.4,
        use_attention=True,
        freeze_backbone=False,
        device=device,
    )

    # Test forward pass
    print("\n🔬 Testing forward pass...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)
        preds = model.predict(dummy_input)
        features = model.get_features(dummy_input)

    print(f"  Input shape    : {dummy_input.shape}")
    print(f"  Logits shape   : {logits.shape}")
    print(f"  Probs shape    : {probs.shape}")
    print(f"  Predictions    : {preds.cpu().numpy()}")
    print(f"  Features shape : {features.shape}")
    print(f"  Probs sum (≈1) : {probs.sum(dim=1).cpu().numpy()}")

    # Test freeze / unfreeze
    print("\n🔧 Testing freeze/unfreeze...")
    model.freeze_backbone()
    frozen_params = model.count_parameters()
    print(f"  After freeze   : trainable = {frozen_params['trainable']:,}")

    model.unfreeze_backbone(unfreeze_fraction=0.5)
    partial_params = model.count_parameters()
    print(f"  After 50% unfreeze : trainable = {partial_params['trainable']:,}")

    model.unfreeze_backbone(unfreeze_fraction=1.0)
    full_params = model.count_parameters()
    print(f"  After full unfreeze: trainable = {full_params['trainable']:,}")

    print("\n[OK] All model tests passed!")
    print("=" * 65)
