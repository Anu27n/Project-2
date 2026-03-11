"""
DR Detection — Visualization Package
"""

from .gradcam import (
    BaseCAM,
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    HookManager,
    apply_threshold_mask,
    draw_contour_on_image,
    heatmap_to_rgb,
    overlay_heatmap,
    visualize_all_classes,
    visualize_gradcam,
)

__all__ = [
    # CAM classes
    "BaseCAM",
    "GradCAM",
    "GradCAMPlusPlus",
    "EigenCAM",
    "HookManager",
    # Overlay utilities
    "heatmap_to_rgb",
    "overlay_heatmap",
    "apply_threshold_mask",
    "draw_contour_on_image",
    # Visualization functions
    "visualize_gradcam",
    "visualize_all_classes",
]
