"""
Streamlit web interface for Diabetic Retinopathy detection.

An enhanced, multi-tab clinical screening dashboard built around the
EfficientNet-B4 + CBAM model trained on APTOS 2019.

Tabs
----
1. Analysis            : Single-image prediction + Grad-CAM + preprocessing
                         pipeline + uncertainty + image-quality checks +
                         downloadable clinical report.
2. Batch Analysis      : Upload many images and get a sortable table /
                         CSV with predictions, confidences and risk flags.
3. Performance         : Dashboard driven by
                         results/metrics/final_evaluation_results.json
                         (confusion matrix, per-class metrics, compute,
                         target achievement, top error modes).
4. Reference Figures   : Gallery of the precomputed figures in
                         results/figures/ (training curves, ROC, etc.).
5. About               : Clinical disclaimer + model card.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# ───────────────────────────────────────────────────────────────
# PROJECT PATHS
# ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

FIGURES_DIR  = PROJECT_ROOT / "results" / "figures"
METRICS_FILE = PROJECT_ROOT / "results" / "metrics" / "final_evaluation_results.json"
CKPT_PATH    = PROJECT_ROOT / "results" / "models" / "model_best_qwk.pth"
# Writable location we can use to cache an uploaded / downloaded checkpoint
# when the default one is missing (e.g. on Streamlit Cloud, where the repo
# does not include the 68 MB .pth file).
RUNTIME_CKPT_DIR = Path(os.environ.get("DR_CKPT_CACHE", "/tmp/dr_ckpt"))

# ───────────────────────────────────────────────────────────────
# CLINICAL CONSTANTS
# ───────────────────────────────────────────────────────────────

CLASS_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
RISK_LEVELS  = ["None", "Low", "Moderate", "High", "Critical"]

RECOMMENDATIONS = [
    "No referable DR. Routine screening in 12 months is sufficient.",
    "Mild non-proliferative DR. Repeat screening in 6-9 months and "
    "control blood sugar, blood pressure and lipids.",
    "Moderate non-proliferative DR. Refer to an ophthalmologist within "
    "3-6 months for dilated eye examination.",
    "Severe non-proliferative DR. Urgent referral to an ophthalmologist "
    "within 1 month; OCT and FFA may be warranted.",
    "Proliferative DR detected. Immediate referral to a retinal "
    "specialist; consider pan-retinal photocoagulation or anti-VEGF.",
]

CLINICAL_NOTES = [
    "No visible microaneurysms, haemorrhages or exudates.",
    "A small number of microaneurysms may be present.",
    "Microaneurysms, dot-blot haemorrhages and hard exudates are "
    "likely; cotton-wool spots possible.",
    "Extensive haemorrhages in all four quadrants, venous beading "
    "and / or IRMA are likely.",
    "Neovascularisation at the disc or elsewhere, vitreous or "
    "pre-retinal haemorrhage may be present.",
]


# ───────────────────────────────────────────────────────────────
# CACHED LOADERS
# ───────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────
# REMOTE / UPLOADED CHECKPOINT
# ───────────────────────────────────────────────────────────────

def _get_secret(name: str) -> Optional[str]:
    """Safely read an st.secret or env var without crashing if no
    secrets.toml is configured (the default on Streamlit Cloud)."""
    try:
        val = st.secrets.get(name)  # type: ignore[attr-defined]
        if val:
            return str(val)
    except Exception:                                       # noqa: BLE001
        pass
    return os.environ.get(name)


@st.cache_resource(show_spinner=True)
def _download_checkpoint(url: str, dest: str) -> str:
    """Fetch a remote .pth into the local cache (idempotent)."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        return str(dest_path)

    import urllib.request

    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest_path)
    return str(dest_path)


def _persist_uploaded_checkpoint(uploaded_file) -> str:
    """Persist an uploaded .pth to RUNTIME_CKPT_DIR and return its path."""
    RUNTIME_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    dest = RUNTIME_CKPT_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: Optional[str] = None):
    """Load the trained EfficientNet-B4 + CBAM model (or a random-init
    fallback if the checkpoint is missing). Also reads the image size
    actually used during training from the checkpoint so inference
    matches the validation pipeline.

    Passing a non-default `ckpt_path` creates a new cache entry, so this
    is the right hook for "override checkpoint" in the UI.
    """
    import torch as _torch

    from config import MODEL_CONFIG, get_device
    from models.efficientnet_model import load_efficientnet_dr_from_checkpoint

    path = Path(ckpt_path) if ckpt_path else CKPT_PATH

    device = get_device()
    train_img_size = 224
    load_error = None
    if path.exists():
        try:
            model, _ = load_efficientnet_dr_from_checkpoint(
                path,
                map_location=device,
                num_classes=5,
                dropout=float(MODEL_CONFIG.get("dropout", 0.4)),
                use_attention=True,
                weights_only=False,
            )
            loaded = True
        except Exception as e:                                  # noqa: BLE001
            load_error = str(e)
            loaded = False
        if loaded:
            # Peek at the checkpoint to recover the training image size so the
            # Streamlit preprocessing pipeline matches the one used during
            # validation (otherwise the softmax collapses to ~uniform 20%).
            try:
                ckpt = _torch.load(path, map_location="cpu", weights_only=False)
                cfg = (ckpt or {}).get("config", {}) or {}
                phases = cfg.get("phases", {}) or {}
                sizes = [ph.get("img_size") for ph in phases.values() if ph.get("img_size")]
                if sizes:
                    train_img_size = int(sizes[-1])  # last phase = final training size
                del ckpt
            except Exception:                                   # noqa: BLE001
                pass

    if not path.exists() or load_error is not None:
        from models.efficientnet_model import EfficientNetDR
        model = EfficientNetDR(
            num_classes=5,
            pretrained=False,
            dropout=float(MODEL_CONFIG.get("dropout", 0.4)),
            use_attention=True,
            model_name=str(MODEL_CONFIG.get("architecture", "efficientnet_b4")),
        ).to(device)
        loaded = False

    model.eval()
    return model, device, loaded, train_img_size, str(path), load_error


@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[dict]:
    """Load precomputed evaluation metrics from JSON."""
    if not METRICS_FILE.exists():
        return None
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────
# PREPROCESSING + INFERENCE
# ───────────────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Unsupported format?")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocessing_stages(
    img_rgb : np.ndarray,
    img_size: int = 224,
) -> Dict[str, np.ndarray]:
    """
    Return every intermediate stage of the fundus preprocessing
    pipeline as uint8 RGB images (for visualisation).

    IMPORTANT: the final stage fed to the model is "Circle mask" (no
    CLAHE baked in here), because validation during training applied
    CLAHE inside the Albumentations pipeline. The visual CLAHE stage
    is shown only for context.
    """
    from preprocessing import DRPreprocessor

    pre = DRPreprocessor(img_size=img_size)

    stages: Dict[str, np.ndarray] = {"Original": img_rgb.copy()}

    cropped = pre.crop_black_borders(img_rgb.copy())
    stages["Border crop"] = cropped

    resized = pre.resize(cropped)
    stages["Resized"] = resized

    ben = pre.ben_graham_preprocessing(resized.copy())
    stages["Ben Graham"] = ben

    circled = pre.circle_crop(ben.copy())
    stages["Circle mask"] = circled

    # Visual-only: CLAHE applied inside OpenCV LAB space (same math as the
    # albumentations stage, minus the final Normalize + ToTensor).
    stages["CLAHE (preview)"] = pre.apply_clahe(circled.copy())

    return stages


def build_tensor(
    processed_uint8: np.ndarray,
    device,
    img_size: int = 224,
) -> torch.Tensor:
    """Build the input tensor using the exact validation transform used
    during training (A.Resize -> A.CLAHE -> A.Normalize -> ToTensorV2)."""
    from dataset import get_valid_transforms

    tfm = get_valid_transforms(img_size=img_size, use_clahe=True)
    tensor = tfm(image=processed_uint8)["image"].unsqueeze(0).to(device)
    return tensor


def predict(tensor: torch.Tensor, model) -> np.ndarray:
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs.astype(np.float32)


# ───────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ───────────────────────────────────────────────────────────────

def predictive_entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats. 0 = confident, ln(K) = uniform."""
    eps = 1e-12
    return float(-np.sum(probs * np.log(probs + eps)))


def top2_margin(probs: np.ndarray) -> float:
    """Gap between top-1 and top-2 probabilities."""
    sorted_p = np.sort(probs)[::-1]
    return float(sorted_p[0] - sorted_p[1])


def expected_grade(probs: np.ndarray) -> float:
    """E[grade] treating 0..4 as an ordinal scale."""
    return float(np.sum(np.arange(len(probs)) * probs))


def image_quality(img_rgb: np.ndarray) -> Dict[str, float]:
    """Rough fundus-image quality descriptors."""
    gray       = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean()) / 255.0
    contrast   = float(gray.std())  / 255.0
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Rough coverage of the retinal disc (non-black pixels)
    coverage = float((gray > 10).mean())

    return {
        "brightness": brightness,
        "contrast":   contrast,
        "sharpness":  sharpness,
        "coverage":   coverage,
    }


def quality_verdict(q: Dict[str, float]) -> Tuple[str, str]:
    """Return (label, streamlit-colour) for a quick traffic-light."""
    problems = []
    if q["brightness"] < 0.15:
        problems.append("very dark")
    elif q["brightness"] > 0.85:
        problems.append("overexposed")
    if q["contrast"] < 0.08:
        problems.append("low contrast")
    if q["sharpness"] < 40:
        problems.append("possibly blurry")
    if q["coverage"] < 0.25:
        problems.append("retina poorly centred")

    if not problems:
        return "Good quality", "success"
    if len(problems) == 1:
        return f"Acceptable ({problems[0]})", "warning"
    return "Poor quality: " + ", ".join(problems), "error"


# ───────────────────────────────────────────────────────────────
# GRAD-CAM
# ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _gradcam_factory(_model, target_layer: str):
    """Build a GradCAM object.  Underscore prevents Streamlit from
    hashing the model weights."""
    from visualization.gradcam import GradCAM
    return GradCAM(_model, target_layer=target_layer)


def compute_gradcam(
    model,
    tensor       : torch.Tensor,
    target_class : Optional[int],
    target_layer : str,
) -> Optional[np.ndarray]:
    try:
        cam = _gradcam_factory(model, target_layer)
        heatmap, _ = cam.generate(tensor, target_class=target_class, smooth=True)
        return heatmap
    except Exception as e:                           # noqa: BLE001
        st.warning(f"Grad-CAM failed ({e}); skipping attention overlay.")
        return None


def overlay_cam(
    base_rgb : np.ndarray,
    heatmap  : np.ndarray,
    alpha    : float = 0.45,
    colormap : str   = "jet",
    threshold: Optional[float] = None,
) -> np.ndarray:
    from visualization.gradcam import overlay_heatmap
    return overlay_heatmap(
        base_rgb, heatmap, alpha=alpha, colormap=colormap, threshold=threshold,
    )


# ───────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ───────────────────────────────────────────────────────────────

def plot_probability_bar(probs: np.ndarray, pred_class: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.bar(CLASS_NAMES, probs * 100, color=CLASS_COLORS,
                  edgecolor="black", linewidth=0.4)
    bars[pred_class].set_edgecolor("black")
    bars[pred_class].set_linewidth(2.0)
    for i, p in enumerate(probs):
        ax.text(i, p * 100 + 1.2, f"{p*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, min(105, max(probs * 100) + 12))
    ax.set_ylabel("Probability (%)")
    ax.set_title("Class probability distribution", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix: np.ndarray,
                          normalized: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    data = matrix.astype(float)
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix " + ("(normalised)" if normalized else "(counts)"))

    vmax = data.max() if data.size else 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = f"{val*100:.1f}%" if normalized else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > 0.5 * vmax else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_per_class_bars(per_class: Dict[str, dict]) -> plt.Figure:
    metrics = ["sensitivity", "specificity", "precision", "f1_score", "auc_roc"]
    labels  = ["Sensitivity", "Specificity", "Precision", "F1", "AUC-ROC"]

    ordered = [per_class[c] for c in CLASS_NAMES if c in per_class]
    if not ordered:
        fig, ax = plt.subplots(); return fig

    x = np.arange(len(CLASS_NAMES))
    width = 0.15

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [ordered[k].get(m, 0.0) for k in range(len(ordered))]
        ax.bar(x + (i - 2) * width, vals, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class validation metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# ───────────────────────────────────────────────────────────────
# REPORT BUILDING
# ───────────────────────────────────────────────────────────────

def build_report(
    filename : str,
    probs    : np.ndarray,
    q        : Dict[str, float],
    quality  : str,
) -> str:
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    lines: List[str] = []
    lines.append("# Diabetic Retinopathy Screening Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"- **Image file:** `{filename}`")
    lines.append(f"- **Model:** EfficientNet-B4 + CBAM (APTOS 2019)")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append(f"- **Severity:** {CLASS_NAMES[pred]}")
    lines.append(f"- **Confidence:** {conf*100:.2f}%")
    lines.append(f"- **Risk level:** {RISK_LEVELS[pred]}")
    lines.append(f"- **Expected grade (0-4):** {expected_grade(probs):.2f}")
    lines.append(f"- **Predictive entropy:** {predictive_entropy(probs):.3f} nats")
    lines.append(f"- **Top-2 margin:** {top2_margin(probs)*100:.2f} %")
    lines.append("")
    lines.append("## Probability breakdown")
    for i, n in enumerate(CLASS_NAMES):
        lines.append(f"- {n}: {probs[i]*100:.2f} %")
    lines.append("")
    lines.append("## Image quality")
    lines.append(f"- Verdict: **{quality}**")
    lines.append(f"- Brightness: {q['brightness']*100:.1f} / 100")
    lines.append(f"- Contrast:   {q['contrast']*100:.1f} / 100")
    lines.append(f"- Sharpness (Laplacian var): {q['sharpness']:.1f}")
    lines.append(f"- Retinal coverage: {q['coverage']*100:.1f} %")
    lines.append("")
    lines.append("## Clinical note")
    lines.append(CLINICAL_NOTES[pred])
    lines.append("")
    lines.append("## Recommendation")
    lines.append(RECOMMENDATIONS[pred])
    lines.append("")
    lines.append("---")
    lines.append("*This tool is intended as a screening aid and does not "
                 "replace professional medical diagnosis.*")
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────
# UI TABS
# ───────────────────────────────────────────────────────────────

def _diagnosis_card(pred: int, confidence: float) -> None:
    colour = CLASS_COLORS[pred]
    st.markdown(
        f"""
        <div style='padding:14px 18px;border-radius:10px;
                    border-left:8px solid {colour};
                    background:#00000008;'>
            <div style='font-size:0.85rem;color:#888;margin-bottom:2px;'>
                Predicted severity
            </div>
            <div style='font-size:1.8rem;font-weight:600;color:{colour};'>
                {CLASS_NAMES[pred]}
            </div>
            <div style='font-size:0.9rem;color:#666;margin-top:6px;'>
                Risk level: <b>{RISK_LEVELS[pred]}</b> &nbsp;·&nbsp;
                Confidence: <b>{confidence*100:.1f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tab_analysis(model, device, settings: dict) -> None:
    st.subheader("Single-image analysis")

    uploaded = st.file_uploader(
        "Upload a retinal fundus image",
        type=["png", "jpg", "jpeg"],
        help="PNG / JPG / JPEG, up to ~10 MB.",
        key="single_upload",
    )
    if uploaded is None:
        st.info("Upload an image to start. You can also try a batch "
                "analysis in the next tab.")
        return

    image_bytes = uploaded.read()
    try:
        original = _decode_image(image_bytes)
    except Exception as e:                               # noqa: BLE001
        st.error(f"Image decoding failed: {e}")
        return

    with st.spinner("Running pipeline and inference..."):
        stages    = preprocessing_stages(original, img_size=settings["img_size"])
        # Model input must match training: circle-mask stage (pre-CLAHE),
        # then Albumentations Resize + CLAHE + Normalize inside build_tensor.
        model_input = stages["Circle mask"]
        tensor      = build_tensor(
            model_input, device, img_size=settings["img_size"]
        )
        probs       = predict(tensor, model)
        # 'processed' is what we show as the Grad-CAM base (with CLAHE).
        processed   = stages["CLAHE (preview)"]

    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    q    = image_quality(original)
    q_label, q_level = quality_verdict(q)

    # --- Top row: image + diagnosis card ---------------------------
    top_left, top_right = st.columns([1, 1])
    with top_left:
        st.image(original, caption="Original fundus image",
                 use_container_width=True)
    with top_right:
        _diagnosis_card(pred, conf)
        st.write("")
        getattr(st, q_level)(f"Image quality: {q_label}")
        st.markdown(
            f"**Clinical note.** {CLINICAL_NOTES[pred]}\n\n"
            f"**Recommendation.** {RECOMMENDATIONS[pred]}"
        )

    # --- Metrics row -----------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence",      f"{conf*100:.1f}%")
    m2.metric("Expected grade",  f"{expected_grade(probs):.2f}")
    m3.metric("Top-2 margin",    f"{top2_margin(probs)*100:.1f}%")
    m4.metric("Entropy (nats)",  f"{predictive_entropy(probs):.3f}")

    st.divider()

    # --- Probability chart -----------------------------------------
    st.markdown("### Class probabilities")
    st.pyplot(plot_probability_bar(probs, pred), clear_figure=True)

    with st.expander("Detailed probability table"):
        df = pd.DataFrame({
            "Class"      : CLASS_NAMES,
            "Probability": [f"{p*100:.2f}%" for p in probs],
            "Risk level" : RISK_LEVELS,
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

    # --- Grad-CAM --------------------------------------------------
    st.markdown("### Attention (Grad-CAM)")
    target_sel = st.radio(
        "Explain probability of",
        options=["Predicted class"] + CLASS_NAMES,
        horizontal=True,
        index=0,
        key="gradcam_target",
    )
    target_class = None if target_sel == "Predicted class" \
        else CLASS_NAMES.index(target_sel)

    heatmap = compute_gradcam(
        model, tensor,
        target_class=target_class,
        target_layer=settings["target_layer"],
    )
    if heatmap is not None:
        overlay = overlay_cam(
            processed, heatmap,
            alpha=settings["alpha"],
            colormap=settings["colormap"],
            threshold=settings["threshold"] or None,
        )
        cam_cols = st.columns(3)
        cam_cols[0].image(processed, caption="Preprocessed input",
                          use_container_width=True)
        cam_cols[1].image(heatmap,  caption="Raw saliency",
                          clamp=True, use_container_width=True)
        cam_cols[2].image(overlay,  caption="Overlay (attention on input)",
                          use_container_width=True)
        st.caption(
            "Warm regions show the pixels that most increase the probability "
            f"of **{CLASS_NAMES[target_class] if target_class is not None else CLASS_NAMES[pred]}**. "
            "Microaneurysms, haemorrhages, exudates and neovascularisation "
            "are expected hotspots for DR grading."
        )

    # --- Preprocessing pipeline ------------------------------------
    if settings["show_pipeline"]:
        st.markdown("### Preprocessing pipeline")
        names = list(stages.keys())
        cols  = st.columns(len(names))
        for col, name in zip(cols, names):
            col.image(stages[name], caption=name, use_container_width=True)
        st.caption(
            "Crop black borders → resize → Ben Graham local-contrast "
            "enhancement → circular retinal mask → CLAHE on the luminance "
            "channel → ImageNet normalisation."
        )

    # --- Image quality ---------------------------------------------
    with st.expander("Image-quality diagnostics"):
        qdf = pd.DataFrame({
            "Metric" : ["Brightness", "Contrast", "Sharpness (Laplacian var)",
                        "Retinal coverage"],
            "Value"  : [f"{q['brightness']*100:.1f} / 100",
                        f"{q['contrast']*100:.1f} / 100",
                        f"{q['sharpness']:.1f}",
                        f"{q['coverage']*100:.1f} %"],
        })
        st.dataframe(qdf, hide_index=True, use_container_width=True)

    # --- Downloadable report ---------------------------------------
    report_md = build_report(uploaded.name, probs, q, q_label)
    st.download_button(
        "Download clinical report (Markdown)",
        data=report_md,
        file_name=f"dr_report_{Path(uploaded.name).stem}.md",
        mime="text/markdown",
    )

    # --- Session history -------------------------------------------
    history = st.session_state.setdefault("history", [])
    history.append({
        "time"     : datetime.now().strftime("%H:%M:%S"),
        "file"     : uploaded.name,
        "pred"     : CLASS_NAMES[pred],
        "conf"     : round(conf * 100, 1),
        "exp_grade": round(expected_grade(probs), 2),
        "entropy"  : round(predictive_entropy(probs), 3),
        "quality"  : q_label,
    })
    if history:
        with st.expander(f"Session history ({len(history)} image(s))"):
            st.dataframe(pd.DataFrame(history), hide_index=True,
                         use_container_width=True)


def tab_batch(model, device, settings: dict) -> None:
    st.subheader("Batch analysis")
    st.write(
        "Upload several fundus images at once to produce a sortable table "
        "of predictions with confidences, expected grades and risk flags. "
        "The results can be exported as CSV."
    )

    files = st.file_uploader(
        "Upload one or more fundus images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if not files:
        return

    rows: List[dict] = []
    prog = st.progress(0.0, text="Running batch inference...")
    for i, f in enumerate(files):
        try:
            img   = _decode_image(f.read())
            proc  = preprocessing_stages(
                img, img_size=settings["img_size"]
            )["Circle mask"]
            probs = predict(
                build_tensor(proc, device, img_size=settings["img_size"]),
                model,
            )
            pred  = int(np.argmax(probs))
            q     = image_quality(img)
            qual, _ = quality_verdict(q)
            rows.append({
                "File"         : f.name,
                "Prediction"   : CLASS_NAMES[pred],
                "Confidence %" : round(float(probs[pred]) * 100, 1),
                "Expected grade": round(expected_grade(probs), 2),
                "Entropy"      : round(predictive_entropy(probs), 3),
                "Top-2 margin %": round(top2_margin(probs) * 100, 1),
                "Risk"         : RISK_LEVELS[pred],
                "Quality"      : qual,
                **{f"p({c})": round(float(probs[k]), 4)
                   for k, c in enumerate(CLASS_NAMES)},
            })
        except Exception as e:                           # noqa: BLE001
            rows.append({"File": f.name, "Prediction": f"ERROR: {e}"})
        prog.progress((i + 1) / len(files),
                      text=f"Processed {i+1}/{len(files)}")
    prog.empty()

    if not rows:
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Summary of this batch
    if "Prediction" in df.columns:
        summary = df["Prediction"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        st.markdown("#### Batch summary")
        sc1, sc2 = st.columns([1, 1])
        with sc1:
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            ax.bar(summary.index, summary.values, color=CLASS_COLORS,
                   edgecolor="black", linewidth=0.4)
            for i, v in enumerate(summary.values):
                ax.text(i, v + 0.05, str(int(v)), ha="center")
            ax.set_ylabel("Images")
            ax.set_title("Predicted-class distribution")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        with sc2:
            referable = int((df["Prediction"].isin(
                ["Moderate", "Severe", "Proliferative"])).sum())
            total = len(df)
            st.metric("Total images",   total)
            st.metric("Referable DR",   f"{referable}  ({referable/total*100:.0f}%)")
            st.metric("No DR",          int((df["Prediction"] == "No DR").sum()))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv,
        file_name="dr_batch_results.csv",
        mime="text/csv",
    )


def tab_performance() -> None:
    st.subheader("Model performance dashboard")
    metrics = load_metrics()
    if metrics is None:
        st.warning(
            "`results/metrics/final_evaluation_results.json` was not found — "
            "run notebook 02 / 03 to populate it."
        )
        return

    overall = metrics.get("overall_metrics", {})
    compute = metrics.get("computation_metrics", {})
    per_cls = metrics.get("per_class_metrics", {})
    cm      = metrics.get("confusion_matrix", {})
    targets = metrics.get("targets_summary", {})

    # ---- Headline metrics -----------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",         f"{overall.get('accuracy', 0)*100:.2f}%")
    c2.metric("Quadratic W Kappa", f"{overall.get('quadratic_weighted_kappa', 0):.4f}")
    c3.metric("AUC-ROC (macro)",  f"{overall.get('auc_roc_macro', 0):.4f}")
    c4.metric("Top-2 accuracy",   f"{overall.get('top2_accuracy', 0)*100:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sensitivity (mean)", f"{overall.get('mean_sensitivity', 0):.3f}")
    c6.metric("Specificity (mean)", f"{overall.get('mean_specificity', 0):.3f}")
    c7.metric("F1 weighted",        f"{overall.get('f1_score_weighted', 0):.3f}")
    c8.metric("Correct / total",
              f"{overall.get('n_correct', 0)} / {overall.get('n_samples', 0)}")

    st.divider()

    # ---- Confusion matrix -----------------------------------------
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Confusion matrix")
        normalise = st.toggle("Normalise rows", value=True, key="cm_norm")
        if normalise and cm.get("normalized_by_true"):
            mat = np.asarray(cm["normalized_by_true"])
            fig = plot_confusion_matrix(mat, normalized=True)
        elif cm.get("raw"):
            mat = np.asarray(cm["raw"])
            fig = plot_confusion_matrix(mat, normalized=False)
        else:
            fig = None
        if fig is not None:
            st.pyplot(fig, clear_figure=True)

    with right:
        st.markdown("#### Target achievement")
        if targets:
            rows = []
            for k, v in targets.items():
                if not isinstance(v, dict):
                    continue
                rows.append({
                    "Metric"  : k.replace("_", " ").title(),
                    "Target"  : v.get("target", ""),
                    "Achieved": v.get("achieved", ""),
                    "Status"  : v.get("status", ""),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

    st.divider()

    # ---- Per-class metrics ----------------------------------------
    st.markdown("#### Per-class metrics")
    if per_cls:
        st.pyplot(plot_per_class_bars(per_cls), clear_figure=True)

        tbl = pd.DataFrame([
            {
                "Class"       : c,
                "Support"     : per_cls[c].get("support"),
                "Sensitivity" : round(per_cls[c].get("sensitivity", 0), 3),
                "Specificity" : round(per_cls[c].get("specificity", 0), 3),
                "Precision"   : round(per_cls[c].get("precision",  0), 3),
                "F1"          : round(per_cls[c].get("f1_score",   0), 3),
                "AUC-ROC"     : round(per_cls[c].get("auc_roc",    0), 3),
                "TP"          : per_cls[c].get("true_positives"),
                "FP"          : per_cls[c].get("false_positives"),
                "FN"          : per_cls[c].get("false_negatives"),
            }
            for c in CLASS_NAMES if c in per_cls
        ])
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    # ---- Compute footprint ----------------------------------------
    st.markdown("#### Compute footprint")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Parameters", f"{compute.get('parameters_total', 0)/1e6:.2f} M")
    cc2.metric("GFLOPs",     f"{compute.get('gflops', 0):.2f}")
    cc3.metric("CPU latency", f"{compute.get('latency_ms_mean', 0):.1f} ms")
    cc4.metric("Throughput",  f"{compute.get('throughput_images_per_sec', 0):.2f} img/s")

    # ---- QWK / top errors -----------------------------------------
    qwk_info = metrics.get("qwk_analysis", {})
    if qwk_info.get("top_errors"):
        with st.expander("Top error modes (by clinical cost)"):
            err_df = pd.DataFrame(qwk_info["top_errors"])
            st.dataframe(err_df, hide_index=True, use_container_width=True)
            st.caption(
                f"Adjacent-grade errors: {qwk_info.get('adjacent_grade_errors_pct', 0):.1f}% "
                f"· Cross-grade errors: {qwk_info.get('cross_grade_errors_pct', 0):.1f}%"
            )


def tab_figures() -> None:
    st.subheader("Reference figures")
    if not FIGURES_DIR.exists():
        st.info("`results/figures/` is empty or missing.")
        return

    figures = sorted(FIGURES_DIR.glob("*.png"))
    if not figures:
        st.info("No PNG figures found in `results/figures/`.")
        return

    pretty = {
        "training_curves"            : "Training curves",
        "confusion_matrix"           : "Confusion matrix",
        "roc_curves"                 : "ROC curves",
        "per_class_performance"      : "Per-class performance",
        "gradcam_sample"             : "Grad-CAM sample",
        "ablation_study"             : "Ablation study",
        "model_comparison"           : "Model comparison",
        "research_comparison"        : "Research comparison",
        "novelty_analysis"           : "Novelty analysis",
        "target_achievement"         : "Target achievement",
        "preprocessing_steps_demo"   : "Preprocessing demo",
        "histogram_analysis"         : "Histogram analysis",
        "computation_metrics_dashboard": "Compute dashboard",
    }

    names = [pretty.get(p.stem, p.stem.replace("_", " ").title()) for p in figures]
    picks = st.multiselect("Figures to display", options=names, default=names)

    selected = [p for p, n in zip(figures, names) if n in picks]
    for p, n in zip(selected,
                    [n for n in names if n in picks]):
        st.markdown(f"##### {n}")
        st.image(str(p), use_container_width=True)
        st.caption(f"`{p.relative_to(PROJECT_ROOT)}`")


@st.cache_data(show_spinner=False)
def _checkpoint_signature() -> Optional[Dict[str, object]]:
    """Summarise the checkpoint so we can confirm the correct weights are
    actually in memory (not a stale random-init fallback)."""
    if not CKPT_PATH.exists():
        return None
    try:
        import torch as _t
        ck = _t.load(CKPT_PATH, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", {}) or {}
        total = sum(v.numel() for v in sd.values()
                    if hasattr(v, "numel"))
        any_w = next(iter(sd.values())) if sd else None
        checksum = float(any_w.float().mean()) if any_w is not None else None
        return {
            "epoch"   : ck.get("epoch"),
            "best_qwk": ck.get("best_qwk"),
            "params"  : total,
            "mean_w0" : checksum,
        }
    except Exception:                                       # noqa: BLE001
        return None


def tab_about(model_loaded: bool, device) -> None:
    st.subheader("About this tool")
    st.markdown(
        """
        **Project.** Deep-Learning-based Medical Image Analysis for Early
        Detection of Diabetic Retinopathy.

        **Model.** EfficientNet-B4 backbone (pre-trained on ImageNet) with a
        Convolutional Block Attention Module (CBAM) and a dual-pooling
        classification head, trained with a Weighted Focal Loss
        (gamma = 1.5) plus a soft QWK regression term on APTOS 2019
        fundus images. A leakage-safe conditional GAN synthesises minority
        classes on the training split.

        **Validation headline (733-image stratified split).**
        - Accuracy **77.63 %**, QWK **0.8406**, macro AUC-ROC **0.9112**
        - Mean sensitivity **0.723**, mean specificity **0.914**
        - 19.92 M params, 3.01 GFLOPs, ~91 ms / image on CPU

        **What this tool can do.**
        - Grade a fundus image on the international 5-class DR scale.
        - Explain its decision with Grad-CAM over the final conv layer.
        - Flag low-confidence / ambiguous cases via predictive entropy
          and the top-2 probability margin.
        - Sanity-check image quality (brightness, contrast, sharpness,
          retinal coverage).
        - Batch-screen a folder of images and export CSV.

        **What this tool is NOT.**
        > A replacement for a dilated fundus examination or for a
        > qualified ophthalmologist.  It is a *screening aid* and its
        > outputs must be interpreted alongside clinical history,
        > visual acuity, IOP and OCT where available.
        """
    )

    sig = _checkpoint_signature()
    meta = pd.DataFrame({
        "Item"  : ["Checkpoint loaded", "Inference device",
                   "Checkpoint path", "Metrics JSON",
                   "Checkpoint epoch", "Checkpoint best QWK",
                   "Total weight tensors (M params)",
                   "First-tensor mean (sanity check)"],
        "Value" : [
            "Yes" if model_loaded else "No (using random-init fallback)",
            str(device),
            str(CKPT_PATH.relative_to(PROJECT_ROOT)),
            str(METRICS_FILE.relative_to(PROJECT_ROOT)),
            str(sig.get("epoch")) if sig else "-",
            f"{sig.get('best_qwk'):.4f}" if sig and sig.get("best_qwk") else "-",
            f"{(sig.get('params') or 0) / 1e6:.2f}" if sig else "-",
            f"{sig.get('mean_w0'):.6f}" if sig and sig.get("mean_w0") is not None else "-",
        ],
    })
    st.dataframe(meta, hide_index=True, use_container_width=True)
    st.caption(
        "If predictions look like a constant ~20% on a single class, click "
        "**Reload model / clear cache** in the sidebar: Streamlit's resource "
        "cache can keep an old model object in memory after source edits or "
        "checkpoint changes."
    )


# ───────────────────────────────────────────────────────────────
# APP ENTRY POINT
# ───────────────────────────────────────────────────────────────

def _sidebar_settings(
    model_loaded    : bool,
    device          ,
    train_img_size  : int,
    active_ckpt_path: str,
) -> dict:
    st.sidebar.markdown("## Settings")

    active_ckpt = Path(active_ckpt_path)
    ckpt_exists_now = active_ckpt.exists()

    if model_loaded:
        st.sidebar.success(
            f"Model loaded on **{device}** "
            f"(training size **{train_img_size}x{train_img_size}**)"
        )
    elif ckpt_exists_now:
        st.sidebar.warning(
            "A checkpoint exists on disk but the cached model object is "
            "still the random-init fallback. Click **Reload model** below."
        )
    else:
        st.sidebar.error(
            f"**Random-init model on {device}** — predictions will be "
            "~20% per class.  No checkpoint was found at the expected path "
            "(shown below).  Put a trained `.pth` there, or use the picker "
            "to point the app at a different file."
        )

    with st.sidebar.expander("Checkpoint diagnostics"):
        st.code(
            f"PROJECT_ROOT      = {PROJECT_ROOT}\n"
            f"Default CKPT_PATH = {CKPT_PATH}\n"
            f"Active checkpoint = {active_ckpt}\n"
            f"exists            = {ckpt_exists_now}\n"
            f"model_loaded      = {model_loaded}\n"
            f"device            = {device}\n"
            f"train_img_size    = {train_img_size}",
            language="text",
        )
        alt_models_dir = PROJECT_ROOT / "results" / "models"
        pths: list[str] = []
        if alt_models_dir.exists():
            pths = sorted(str(p) for p in alt_models_dir.glob("*.pth"))
        st.caption(f"`.pth` files in `{alt_models_dir}`:")
        for p in pths:
            st.write(f"- `{p}`")
        if not pths:
            st.write("(none)")

        # Include any cached / uploaded checkpoints so they show up here too
        if RUNTIME_CKPT_DIR.exists():
            pths.extend(sorted(
                str(p) for p in RUNTIME_CKPT_DIR.glob("*.pth")
                if str(p) not in pths
            ))

        options = ["(default)"] + pths
        current = st.session_state.get("ckpt_override") or "(default)"
        choice = st.selectbox(
            "Override checkpoint", options=options,
            index=options.index(current) if current in options else 0,
        )
        if st.button("Apply override", use_container_width=True):
            st.session_state["ckpt_override"] = (
                None if choice == "(default)" else choice
            )
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption(
            "On Streamlit Cloud the repo does not ship `.pth` weights "
            "(they're in `.gitignore`). Upload the file here, or set "
            "`CHECKPOINT_URL` in **Settings → Secrets** to a direct "
            "download URL (Hugging Face Hub, GitHub Releases, etc.)."
        )
        up = st.file_uploader(
            "Upload `model_best_qwk.pth`",
            type=["pth", "pt", "ckpt"],
            key="ckpt_upload",
        )
        if up is not None and st.button("Use uploaded checkpoint",
                                        use_container_width=True):
            saved = _persist_uploaded_checkpoint(up)
            st.session_state["ckpt_override"] = saved
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    if st.sidebar.button("Reload model / clear cache", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    # The model MUST see the same resolution it was trained at. Allowing a
    # mismatched slider value here caused near-uniform 20% predictions in
    # earlier versions of the app.
    st.sidebar.caption(
        "Inference uses the training image size baked into the "
        "checkpoint; override only if you know what you are doing."
    )
    override = st.sidebar.checkbox("Override inference size", value=False)
    if override:
        img_size = st.sidebar.slider(
            "Preprocessing image size", 160, 512, train_img_size, 16
        )
    else:
        img_size = int(train_img_size)

    st.sidebar.markdown("### Grad-CAM")
    colormap = st.sidebar.selectbox(
        "Colormap",
        ["jet", "hot", "plasma", "viridis", "RdYlGn_r"],
        index=0,
    )
    alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05)
    threshold = st.sidebar.slider("Saliency threshold (0 = off)",
                                  0.0, 0.9, 0.0, 0.05)
    target_layer = st.sidebar.text_input(
        "Target layer",
        value="backbone.conv_head",
        help="Module path within the model to hook for Grad-CAM.",
    )

    st.sidebar.markdown("### Display")
    show_pipeline = st.sidebar.checkbox("Show preprocessing pipeline",
                                        value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Disclaimer.** Screening aid only; not a substitute for "
        "professional medical diagnosis."
    )

    return {
        "img_size"      : img_size,
        "colormap"      : colormap,
        "alpha"         : alpha,
        "threshold"     : threshold,
        "target_layer"  : target_layer,
        "show_pipeline" : show_pipeline,
    }


def main() -> None:
    st.set_page_config(
        page_title="DR Detection Studio",
        page_icon="eye",
        layout="wide",
    )

    st.title("Diabetic Retinopathy Detection Studio")
    st.caption(
        "EfficientNet-B4 + CBAM attention · APTOS 2019 · "
        "Grad-CAM explanations, uncertainty analysis and batch screening."
    )

    # Auto-download from a URL configured via st.secrets or env var, if the
    # default checkpoint is missing in the deployed file tree.
    override_path = st.session_state.get("ckpt_override")
    if override_path is None and not CKPT_PATH.exists():
        url = _get_secret("CHECKPOINT_URL")
        if url:
            try:
                downloaded = _download_checkpoint(
                    url, str(RUNTIME_CKPT_DIR / "model_best_qwk.pth"),
                )
                override_path = downloaded
                st.session_state["ckpt_override"] = downloaded
            except Exception as e:                              # noqa: BLE001
                st.sidebar.error(f"Checkpoint download failed: {e}")

    model, device, model_loaded, train_img_size, active_ckpt, load_err = \
        load_model(override_path)
    if load_err:
        st.sidebar.error(f"Checkpoint load failed: {load_err}")
    settings = _sidebar_settings(
        model_loaded, device, train_img_size, active_ckpt,
    )

    tabs = st.tabs(
        ["Analysis", "Batch", "Performance", "Figures", "About"]
    )
    with tabs[0]:
        tab_analysis(model, device, settings)
    with tabs[1]:
        tab_batch(model, device, settings)
    with tabs[2]:
        tab_performance()
    with tabs[3]:
        tab_figures()
    with tabs[4]:
        tab_about(model_loaded, device)


if __name__ == "__main__":
    main()
