#!/usr/bin/env python
"""
Run inference on a single retinal image or a directory of images.

Usage:
    python run_inference.py --image path/to/image.png
    python run_inference.py --image-dir path/to/images/ --output results/predictions.csv
    python run_inference.py --image path/to/image.png --gradcam
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
import torch

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def load_model(checkpoint_path: str, device: torch.device):
    from models.efficientnet_model import EfficientNetDR

    model = EfficientNetDR(num_classes=5, pretrained=False, dropout=0.4, use_attention=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, img_size: int = 512):
    from preprocessing import DRPreprocessor

    preprocessor = DRPreprocessor(img_size=img_size)
    img = preprocessor.preprocess(image_path, normalize=False)
    img = img.astype(np.uint8)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=img)["image"].unsqueeze(0)
    return tensor, img


def predict_single(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    return pred, confidence, probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="DR Detection Inference")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, help="Directory of images to process")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "results" / "models" / "model_best_qwk.pth"),
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM overlay")
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image-dir")

    from config import get_device
    device = get_device()
    model = load_model(args.checkpoint, device)
    print(f"  Model loaded on {device}")

    results = []

    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
        )

    for img_path in image_paths:
        tensor, original = preprocess_image(str(img_path), args.img_size)
        pred, conf, probs = predict_single(model, tensor, device)

        results.append({
            "image": img_path.name,
            "prediction": pred,
            "class_name": CLASS_NAMES[pred],
            "confidence": round(conf, 4),
            **{f"prob_{CLASS_NAMES[i]}": round(float(probs[i]), 4) for i in range(5)},
        })

        severity = "LOW" if pred <= 1 else ("MEDIUM" if pred == 2 else "HIGH")
        print(f"  {img_path.name}: {CLASS_NAMES[pred]} (conf={conf:.1%}) [{severity} risk]")

        if args.gradcam:
            try:
                from visualization.gradcam import GradCAM, visualize_gradcam
                cam = GradCAM(model, target_layer="backbone.blocks.6")
                heatmap, cam_conf = cam.generate(tensor.to(device), target_class=pred)
                save_path = PROJECT_ROOT / "results" / "figures" / f"gradcam_{img_path.stem}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                visualize_gradcam(
                    original, heatmap, pred, cam_conf,
                    save_path=str(save_path), show=False,
                )
            except Exception as e:
                print(f"  Grad-CAM error: {e}")

    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n  Predictions saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
