"""
Diabetic Retinopathy Detection - Grad-CAM Visualization Module

Implements gradient-based visual explanation methods for interpreting
EfficientNet-B4 predictions on retinal fundus images:

    1. Grad-CAM      : Gradient-weighted Class Activation Mapping
                       (Selvaraju et al., ICCV 2017)
    2. Grad-CAM++    : Improved Grad-CAM with better localization
                       (Chattopadhay et al., WACV 2018)
    3. EigenCAM      : PCA-based CAM, gradient-free alternative
    4. Overlay utils : Heatmap blending, multi-class grid, clinical report

Clinical motivation:
    Grad-CAM highlights the retinal regions that drove the model's
    DR severity prediction — microaneurysms, exudates, haemorrhages,
    neovascularisation — enabling clinician validation and trust.

Usage:
    cam = GradCAM(model, target_layer="backbone.blocks.6")
    heatmap, overlay = cam.generate(image_tensor, target_class=2)
    cam.visualize(original_image, overlay, predicted_class=2)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

CLASS_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
COLORMAPS    = ["jet", "hot", "plasma", "viridis", "RdYlGn_r"]

# Clinical lesion labels for annotation overlays
DR_LESION_LABELS = {
    0: [],
    1: ["Microaneurysms"],
    2: ["Microaneurysms", "Hard Exudates", "Haemorrhages"],
    3: ["Microaneurysms", "Haemorrhages", "Venous Beading", "IRMA"],
    4: ["Neovascularisation", "Vitreous Haemorrhage", "Pre-retinal Haemorrhage"],
}


# ─────────────────────────────────────────────────────────────
# HOOK MANAGER
# ─────────────────────────────────────────────────────────────

class HookManager:
    """
    Context-manager-safe forward/backward hook handler.

    Attaches hooks to a target layer and captures:
        - Forward activations  (feature maps)
        - Backward gradients   (gradient signals)

    Usage:
        with HookManager(target_layer) as hooks:
            output = model(x)
            output[0, class_idx].backward()
            activations = hooks.activations
            gradients   = hooks.gradients
    """

    def __init__(self, layer: nn.Module):
        self.layer       = layer
        self.activations : Optional[torch.Tensor] = None
        self.gradients   : Optional[torch.Tensor] = None
        self._fwd_hook   = None
        self._bwd_hook   = None

    def __enter__(self) -> "HookManager":
        self._fwd_hook = self.layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.layer.register_full_backward_hook(self._save_gradient)
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def _remove_hooks(self):
        if self._fwd_hook is not None:
            self._fwd_hook.remove()
            self._fwd_hook = None
        if self._bwd_hook is not None:
            self._bwd_hook.remove()
            self._bwd_hook = None

    def remove(self):
        """Manually remove hooks (use if not using context manager)."""
        self._remove_hooks()


# ─────────────────────────────────────────────────────────────
# BASE CAM CLASS
# ─────────────────────────────────────────────────────────────

class BaseCAM:
    """
    Abstract base class for all Class Activation Map methods.

    Subclasses must implement `_compute_weights()` which defines
    how gradient/activation information is aggregated into channel
    weights for the final saliency map.

    Args:
        model        : Trained EfficientNetDR (or any nn.Module)
        target_layer : Name of the convolutional layer to hook.
                       For EfficientNet-B4 (timm), use the string path
                       to the layer, e.g. "backbone.blocks.6.0.conv_pw"
        device       : Torch device (auto-detected if None)
    """

    def __init__(
        self,
        model       : nn.Module,
        target_layer: str,
        device      : Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or (
            next(model.parameters()).device
            if len(list(model.parameters())) > 0
            else torch.device("cpu")
        )
        self.model.eval()

        # Resolve target layer by dot-separated name
        self.target_layer = self._get_layer(target_layer)
        self.layer_name   = target_layer

    def _get_layer(self, layer_name: str) -> nn.Module:
        """Traverse the model's module tree by dot-separated name."""
        parts = layer_name.split(".")
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def _compute_weights(
        self,
        activations : torch.Tensor,
        gradients   : torch.Tensor,
    ) -> torch.Tensor:
        """
        Abstract: compute per-channel weights from activations and gradients.

        Args:
            activations : (1, C, H, W) forward feature maps
            gradients   : (1, C, H, W) backward gradient maps

        Returns:
            weights : (C,) channel importance weights
        """
        raise NotImplementedError

    def generate(
        self,
        input_tensor  : torch.Tensor,
        target_class  : Optional[int] = None,
        smooth        : bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate a Grad-CAM saliency map for a single image.

        Args:
            input_tensor  : (1, 3, H, W) preprocessed image tensor
            target_class  : Class index to explain. If None, uses the
                            predicted (argmax) class.
            smooth        : Apply Gaussian smoothing to the heatmap

        Returns:
            heatmap   : (H, W) float32 array in [0, 1] — raw saliency map
            confidence: Softmax probability of the explained class
        """
        input_tensor = input_tensor.to(self.device)

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with HookManager(self.target_layer) as hooks:
            # Forward pass
            self.model.zero_grad()
            logits = self.model(input_tensor)               # (1, num_classes)
            probs  = F.softmax(logits, dim=1)               # (1, num_classes)

            # Choose class to explain
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            confidence = float(probs[0, target_class].item())

            # Backward pass: gradient of class score w.r.t. activations
            score = logits[0, target_class]
            score.backward()

            activations = hooks.activations  # (1, C, H', W')
            gradients   = hooks.gradients    # (1, C, H', W')

        # Compute channel weights
        weights = self._compute_weights(activations, gradients)  # (C,)

        # Weighted combination of activation maps
        cam = (weights.view(-1, 1, 1) * activations.squeeze(0)).sum(dim=0)  # (H', W')

        # ReLU: only keep positive contributions
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        cam = self._normalize(cam)

        # Resize to input spatial dimensions
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam = self._normalize(cam)

        # Optional smoothing
        if smooth:
            cam = cv2.GaussianBlur(cam, (11, 11), sigmaX=4)
            cam = self._normalize(cam)

        return cam.astype(np.float32), confidence

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise array to [0, 1]."""
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)

    def generate_for_all_classes(
        self,
        input_tensor : torch.Tensor,
        smooth       : bool = True,
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Generate saliency maps for all DR severity classes.

        Args:
            input_tensor : (1, 3, H, W) preprocessed image tensor

        Returns:
            Dictionary mapping class name → (heatmap, confidence)
        """
        results: Dict[str, Tuple[np.ndarray, float]] = {}
        for c, name in enumerate(CLASS_NAMES):
            heatmap, conf = self.generate(input_tensor, target_class=c, smooth=smooth)
            results[name] = (heatmap, conf)
        return results


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM(BaseCAM):
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017.

    Weights are computed as the Global Average Pooling of gradients
    over the spatial dimensions:
        w_k = (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{ij}

    CAM = ReLU(Σ_k w_k * A^k)
    """

    def _compute_weights(
        self,
        activations : torch.Tensor,
        gradients   : torch.Tensor,
    ) -> torch.Tensor:
        # Global Average Pool over spatial dims (H', W')
        return gradients.squeeze(0).mean(dim=(-2, -1))   # (C,)


# ─────────────────────────────────────────────────────────────
# GRAD-CAM++
# ─────────────────────────────────────────────────────────────

class GradCAMPlusPlus(BaseCAM):
    """
    Grad-CAM++: Improved gradient-weighted Class Activation Mapping.

    Reference:
        Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based
        Visual Explanations for Deep Convolutional Networks", WACV 2018.

    Improvement over Grad-CAM:
        Uses second-order gradients to more accurately localize
        multiple instances of the same class in one image.

    Weight formula:
        alpha^{kc}_{ij} = (∂²y^c / ∂(A^k_{ij})²) /
                          (2 * ∂²y^c / ∂(A^k_{ij})² + Σ A^k_{ab} * ∂³y^c/∂(A^k_{ij})³)

        w_k = Σ_{i,j} alpha^{kc}_{ij} * ReLU(∂y^c / ∂A^k_{ij})
    """

    def _compute_weights(
        self,
        activations : torch.Tensor,
        gradients   : torch.Tensor,
    ) -> torch.Tensor:
        grads    = gradients.squeeze(0)      # (C, H', W')
        acts     = activations.squeeze(0)    # (C, H', W')

        grads_sq  = grads ** 2              # second-order approx
        grads_cu  = grads ** 3              # third-order approx

        # Sum of activations per channel
        acts_sum  = acts.sum(dim=(-2, -1), keepdim=True)  # (C, 1, 1)

        # Alpha weights (element-wise, spatial)
        denom  = 2.0 * grads_sq + grads_cu * acts_sum + 1e-7
        alpha  = grads_sq / denom                         # (C, H', W')

        # Final per-channel weights
        weights = (alpha * F.relu(grads)).sum(dim=(-2, -1))  # (C,)
        return weights


# ─────────────────────────────────────────────────────────────
# EIGEN-CAM  (gradient-free)
# ─────────────────────────────────────────────────────────────

class EigenCAM(BaseCAM):
    """
    EigenCAM: Principal Component Analysis-based Class Activation Map.

    Reference:
        Muhammad & Yeasin, "EigenCAM: Class Activation Map using
        Principal Components", IJCNN 2020.

    Key property:
        Gradient-free — uses PCA on the activation tensor.
        The first principal component captures the most salient
        spatial structure in the feature maps.

    Advantage:
        More stable than gradient-based methods for pre-trained
        models fine-tuned on small medical datasets.
    """

    def generate(
        self,
        input_tensor  : torch.Tensor,
        target_class  : Optional[int] = None,
        smooth        : bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Override generate: no backward pass needed."""
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        activations_captured: List[torch.Tensor] = []

        def fwd_hook(module, inp, out):
            activations_captured.append(out.detach())

        hook = self.target_layer.register_forward_hook(fwd_hook)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs  = F.softmax(logits, dim=1)

        hook.remove()

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        confidence = float(probs[0, target_class].item())
        acts = activations_captured[0].squeeze(0)   # (C, H', W')

        # PCA: reshape to (C, H'*W') and compute first PC
        C, H, W   = acts.shape
        acts_flat  = acts.view(C, -1).cpu().numpy()     # (C, H'*W')
        acts_flat -= acts_flat.mean(axis=0, keepdims=True)

        try:
            # Compact SVD for efficiency
            _, _, Vt = np.linalg.svd(acts_flat, full_matrices=False)
            # Project activations onto first principal component
            cam = np.abs(acts_flat.T @ Vt[0]).reshape(H, W)
        except np.linalg.LinAlgError:
            # Fallback: channel-mean
            cam = acts.mean(dim=0).cpu().numpy()

        cam = self._normalize(cam)

        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam  = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam  = self._normalize(cam)

        if smooth:
            cam = cv2.GaussianBlur(cam, (11, 11), sigmaX=4)
            cam = self._normalize(cam)

        return cam.astype(np.float32), confidence

    def _compute_weights(self, activations, gradients):
        # Not used — overridden by generate()
        return activations.squeeze(0).mean(dim=(-2, -1))


# ─────────────────────────────────────────────────────────────
# HEATMAP OVERLAY UTILITIES
# ─────────────────────────────────────────────────────────────

def heatmap_to_rgb(
    heatmap   : np.ndarray,
    colormap  : str = "jet",
) -> np.ndarray:
    """
    Convert a float [0, 1] heatmap to an RGB colour image.

    Args:
        heatmap  : (H, W) float array in [0, 1]
        colormap : Matplotlib colormap name

    Returns:
        (H, W, 3) uint8 RGB image
    """
    cmap  = plt.get_cmap(colormap)
    rgb   = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_heatmap(
    original_image : np.ndarray,
    heatmap        : np.ndarray,
    alpha          : float = 0.45,
    colormap       : str   = "jet",
    threshold      : Optional[float] = None,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original retinal image.

    Args:
        original_image : (H, W, 3) uint8 RGB retinal image
        heatmap        : (H, W) float array in [0, 1]
        alpha          : Heatmap opacity (0.0 = invisible, 1.0 = opaque)
        colormap       : Matplotlib colormap for the heatmap
        threshold      : If set, mask out regions below this saliency value

    Returns:
        (H, W, 3) uint8 blended image
    """
    # Ensure correct size
    h, w  = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # Optional thresholding (highlight only high-saliency regions)
    if threshold is not None:
        heatmap_resized = np.where(
            heatmap_resized >= threshold, heatmap_resized, 0.0
        ).astype(np.float32)

    heatmap_rgb = heatmap_to_rgb(heatmap_resized, colormap)  # (H, W, 3) uint8

    # Ensure original is uint8 RGB
    if original_image.dtype != np.uint8:
        img = (original_image * 255).clip(0, 255).astype(np.uint8)
    else:
        img = original_image.copy()

    # Alpha blend
    blended = cv2.addWeighted(img, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended


def apply_threshold_mask(
    heatmap   : np.ndarray,
    percentile: float = 70.0,
) -> np.ndarray:
    """
    Zero out heatmap values below the given percentile.

    Useful for isolating the most salient regions (lesion areas)
    while suppressing background noise.

    Args:
        heatmap    : (H, W) float heatmap in [0, 1]
        percentile : Threshold percentile (e.g., 70 → top 30% kept)

    Returns:
        Thresholded heatmap of same shape
    """
    thresh = np.percentile(heatmap, percentile)
    return np.where(heatmap >= thresh, heatmap, 0.0).astype(np.float32)


def draw_contour_on_image(
    image   : np.ndarray,
    heatmap : np.ndarray,
    level   : float = 0.5,
    color   : Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw contour lines from the heatmap onto the image.

    Highlights the boundary of high-saliency regions (e.g., lesion edges).

    Args:
        image    : (H, W, 3) uint8 RGB image
        heatmap  : (H, W) float [0, 1] saliency map
        level    : Contour level threshold (default: 0.5)
        color    : BGR/RGB contour colour
        thickness: Contour line thickness in pixels

    Returns:
        Image with contours drawn
    """
    result = image.copy()

    # Binarize heatmap at the given level
    binary = (heatmap >= level).astype(np.uint8) * 255

    # Find and draw contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


# ─────────────────────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def visualize_gradcam(
    original_image  : np.ndarray,
    heatmap         : np.ndarray,
    predicted_class : int,
    confidence      : float,
    true_class      : Optional[int] = None,
    method_name     : str  = "Grad-CAM",
    colormap        : str  = "jet",
    alpha           : float = 0.45,
    save_path       : Optional[Union[str, Path]] = None,
    show            : bool = True,
) -> np.ndarray:
    """
    Create a 3-panel Grad-CAM visualization figure:
        [Original] | [Heatmap] | [Overlay]

    Args:
        original_image  : (H, W, 3) uint8 RGB retinal image
        heatmap         : (H, W) float [0, 1] saliency map
        predicted_class : Predicted DR severity index
        confidence      : Softmax confidence for the predicted class
        true_class      : Ground truth class (if available)
        method_name     : CAM method label for title
        colormap        : Colormap for the heatmap panel
        alpha           : Overlay alpha
        save_path       : If given, saves the figure to this path
        show            : Whether to call plt.show()

    Returns:
        The blended overlay as a uint8 RGB array
    """
    pred_name  = CLASS_NAMES[predicted_class]
    pred_color = CLASS_COLORS[predicted_class]
    overlay    = overlay_heatmap(original_image, heatmap, alpha=alpha, colormap=colormap)
    heatmap_rgb = heatmap_to_rgb(heatmap, colormap)

    # Build title
    title_parts = [f"{method_name} — Predicted: {pred_name} ({confidence*100:.1f}%)"]
    if true_class is not None:
        true_name = CLASS_NAMES[true_class]
        status    = "✓ CORRECT" if true_class == predicted_class else "✗ WRONG"
        title_parts.append(f"True: {true_name}  |  {status}")
    title = "\n".join(title_parts)

    # Lesion hints
    lesions = DR_LESION_LABELS.get(predicted_class, [])
    lesion_str = ("Key features: " + ", ".join(lesions)) if lesions else "No lesions expected"

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    panels = [
        (original_image, "Original Retinal Image",    None),
        (heatmap_rgb,    f"{method_name} Heatmap",    colormap),
        (overlay,        "Grad-CAM Overlay",           None),
    ]

    for ax, (img, panel_title, cmap) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(panel_title, fontsize=11, color="white", fontweight="bold", pad=8)
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")

        # Colourbar on heatmap panel
        if cmap is not None:
            sm = plt.cm.ScalarMappable(
                cmap=plt.get_cmap(cmap),
                norm=plt.Normalize(vmin=0, vmax=1),
            )
            plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # Coloured border on the predicted-class label
    fig.suptitle(title, fontsize=12, color=pred_color, fontweight="bold", y=1.02)
    fig.text(
        0.5, -0.02, lesion_str,
        ha="center", fontsize=9, color="#aaaaaa", style="italic"
    )

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  💾 Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return overlay


def visualize_all_classes(
    original_image : np.ndarray,
    cam_results    : Dict[str, Tuple[np.ndarray, float]],
    predicted_class: int,
    method_name    : str = "Grad-CAM",
    colormap       : str = "jet",
    alpha          : float = 0.40,
    save_path      : Optional[Union[str, Path]] = None,
    show           : bool = True,
) -> plt.Figure:
    """
    Create a 2-row grid showing Grad-CAM overlays for all 5 DR classes.

    Layout:
        Row 1: Original | Class 0 | Class 1 | Class 2
        Row 2:           | Class 3 | Class 4 | Legend

    Args:
        original_image  : (H, W, 3) uint8 RGB retinal image
        cam_results     : Dict from generate_for_all_classes()
        predicted_class : Index of the model's predicted class
        method_name     : CAM method label
        colormap        : Heatmap colormap
        alpha           : Overlay alpha
        save_path       : Save path for the figure
        show            : Call plt.show()

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor("#0d0d1a")

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.12)

    # ── Row 0, Col 0: Original ────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_image)
    ax_orig.set_title("Original\nRetinal Image", color="white",
                      fontsize=10, fontweight="bold")
    ax_orig.axis("off")

    # Predicted-class text box on original
    pred_name  = CLASS_NAMES[predicted_class]
    pred_color = CLASS_COLORS[predicted_class]
    ax_orig.text(
        0.5, -0.08, f"Predicted: {pred_name}",
        transform=ax_orig.transAxes,
        ha="center", fontsize=9, color=pred_color, fontweight="bold"
    )

    # ── All-class overlays ────────────────────────────────────
    positions = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2),
    ]

    for idx, (cls_name, (heatmap, conf)) in enumerate(cam_results.items()):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])

        overlay = overlay_heatmap(original_image, heatmap, alpha=alpha, colormap=colormap)
        ax.imshow(overlay)

        cls_idx   = CLASS_NAMES.index(cls_name)
        cls_color = CLASS_COLORS[cls_idx]
        is_pred   = cls_idx == predicted_class
        border_w  = 5 if is_pred else 1

        # Highlight predicted class with coloured border
        for spine in ax.spines.values():
            spine.set_edgecolor(cls_color)
            spine.set_linewidth(border_w)

        title_suffix = " ★" if is_pred else ""
        ax.set_title(
            f"Class {cls_idx}: {cls_name}{title_suffix}\n"
            f"Conf: {conf*100:.1f}%",
            color=cls_color, fontsize=9, fontweight="bold",
        )
        ax.axis("off")

    # ── Legend panel (row 1, col 3) ───────────────────────────
    ax_leg = fig.add_subplot(gs[1, 3])
    ax_leg.set_facecolor("#0d0d1a")
    ax_leg.axis("off")

    patches = [
        mpatches.Patch(color=CLASS_
