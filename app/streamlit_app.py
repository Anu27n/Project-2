"""
Streamlit web interface for Diabetic Retinopathy detection.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
RISK_LEVELS = ["None", "Low", "Medium", "High", "Critical"]
RECOMMENDATIONS = [
    "No action needed. Routine screening in 12 months.",
    "Monitor closely. Follow-up screening in 6-9 months.",
    "Refer to ophthalmologist within 3-6 months.",
    "Urgent referral to ophthalmologist within 1 month.",
    "Immediate referral to retinal specialist.",
]


@st.cache_resource
def load_model():
    from config import MODEL_CONFIG, get_device
    from models.efficientnet_model import load_efficientnet_dr_from_checkpoint

    device = get_device()
    ckpt_path = PROJECT_ROOT / "results" / "models" / "model_best_qwk.pth"
    if ckpt_path.exists():
        model, _ = load_efficientnet_dr_from_checkpoint(
            ckpt_path,
            map_location=device,
            num_classes=5,
            dropout=float(MODEL_CONFIG.get("dropout", 0.4)),
            use_attention=True,
            weights_only=False,
        )
    else:
        from models.efficientnet_model import EfficientNetDR

        model = EfficientNetDR(
            num_classes=5,
            pretrained=False,
            dropout=float(MODEL_CONFIG.get("dropout", 0.4)),
            use_attention=True,
            model_name=str(MODEL_CONFIG.get("architecture", "efficientnet_b4")),
        ).to(device)
        model.eval()
    return model, device


def preprocess_and_predict(image_bytes, model, device):
    from preprocessing import DRPreprocessor

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preprocessor = DRPreprocessor(img_size=512)
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

    return probs, img, processed


def main():
    st.set_page_config(
        page_title="DR Detection",
        page_icon="👁",
        layout="wide",
    )

    st.title("Diabetic Retinopathy Detection")
    st.markdown(
        "Upload a retinal fundus image to detect Diabetic Retinopathy severity "
        "using **EfficientNet-B4 + CBAM Attention**."
    )

    model, device = load_model()
    st.sidebar.success(f"Model loaded on **{device}**")

    uploaded = st.file_uploader(
        "Upload retinal fundus image",
        type=["png", "jpg", "jpeg"],
        help="Accepted formats: PNG, JPG, JPEG. Max 10 MB.",
    )

    if uploaded is not None:
        image_bytes = uploaded.read()

        with st.spinner("Analyzing image..."):
            probs, original, processed = preprocess_and_predict(image_bytes, model, device)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("Original Image")
            st.image(original, use_container_width=True)

        with col2:
            st.subheader("Preprocessed")
            st.image(processed, use_container_width=True)

        with col3:
            st.subheader("Diagnosis")
            color = CLASS_COLORS[pred_class]
            st.markdown(
                f"<h2 style='color:{color}'>{CLASS_NAMES[pred_class]}</h2>",
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{confidence:.1%}")
            st.metric("Risk Level", RISK_LEVELS[pred_class])
            st.info(RECOMMENDATIONS[pred_class])

        st.subheader("Class Probabilities")
        chart_data = {CLASS_NAMES[i]: float(probs[i]) for i in range(5)}
        st.bar_chart(chart_data)

        with st.expander("Detailed Probabilities"):
            for i in range(5):
                pct = probs[i] * 100
                st.progress(float(probs[i]), text=f"{CLASS_NAMES[i]}: {pct:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "This tool uses deep learning to screen for Diabetic Retinopathy from "
        "retinal fundus photographs. It is intended as a **screening aid** and "
        "does **not** replace professional medical diagnosis."
    )
    st.sidebar.markdown(
        "**Model**: EfficientNet-B4 + CBAM  \n"
        "**Dataset**: APTOS 2019  \n"
        "**Metrics**: QWK=0.878, AUC=0.964"
    )


if __name__ == "__main__":
    main()
