"""
FastAPI endpoint for Diabetic Retinopathy detection inference.

Usage:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
    # Then POST an image to http://localhost:8000/predict
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import MODEL_CONFIG, get_device
from models.efficientnet_model import EfficientNetDR, load_efficientnet_dr_from_checkpoint
from preprocessing import DRPreprocessor

app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="EfficientNet-B4 + CBAM model for DR severity classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
RISK_LEVELS = ["None", "Low", "Medium", "High", "Critical"]
RECOMMENDATIONS = [
    "No action needed. Routine screening in 12 months.",
    "Monitor closely. Follow-up screening in 6-9 months.",
    "Refer to ophthalmologist within 3-6 months.",
    "Urgent referral to ophthalmologist within 1 month.",
    "Immediate referral to retinal specialist.",
]

_model = None
_device = None
_preprocessor = None


def get_model():
    global _model, _device, _preprocessor
    if _model is None:
        _device = get_device()
        checkpoint_path = PROJECT_ROOT / "results" / "models" / "model_best_qwk.pth"
        if checkpoint_path.exists():
            _model, _ = load_efficientnet_dr_from_checkpoint(
                checkpoint_path,
                map_location=_device,
                num_classes=5,
                dropout=float(MODEL_CONFIG.get("dropout", 0.4)),
                use_attention=True,
                weights_only=False,
            )
        else:
            _model = EfficientNetDR(
                num_classes=5,
                pretrained=False,
                dropout=float(MODEL_CONFIG.get("dropout", 0.4)),
                use_attention=True,
                model_name=str(MODEL_CONFIG.get("architecture", "efficientnet_b4")),
            ).to(_device)
            _model.eval()
        _preprocessor = DRPreprocessor(img_size=512)
    return _model, _device, _preprocessor


@app.get("/")
async def root():
    return {
        "service": "Diabetic Retinopathy Detection API",
        "model": "EfficientNet-B4 + CBAM",
        "classes": CLASS_NAMES,
        "endpoints": {"/predict": "POST image for DR prediction", "/health": "Health check"},
    }


@app.get("/health")
async def health():
    model, device, _ = get_model()
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (png/jpg)")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")

    try:
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    model, device, preprocessor = get_model()

    processed = preprocessor.preprocess(img, normalize=False).astype(np.uint8)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=processed)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    return JSONResponse({
        "prediction": {
            "class_index": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "confidence": round(confidence, 4),
            "risk_level": RISK_LEVELS[pred_class],
            "recommendation": RECOMMENDATIONS[pred_class],
        },
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(5)
        },
        "filename": file.filename,
    })
