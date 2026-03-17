"""
Diabetic Retinopathy Detection - Comprehensive Evaluation Metrics Module

Implements all evaluation metrics required for DR severity classification:

    Primary Metric  : Quadratic Weighted Kappa (QWK)  — official APTOS metric
    Secondary       : Accuracy, AUC-ROC (macro OvR), F1-Score (weighted)
    Clinical        : Sensitivity, Specificity per class
    Diagnostic      : Confusion Matrix, Classification Report, ROC Curves

DR Grade Scale (Ordinal):
    0 → No DR          (healthy)
    1 → Mild NPDR
    2 → Moderate NPDR
    3 → Severe NPDR
    4 → Proliferative  (most severe)

Usage:
    evaluator = DRMetricsEvaluator(num_classes=5)
    results   = evaluator.evaluate(y_true, y_pred, y_proba)
    evaluator.print_report(results)
    evaluator.plot_all(results, save_dir="results/figures")
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

CLASS_NAMES     = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CLASS_COLORS    = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
NUM_CLASSES     = 5

# Severity penalty matrix — used in the QWK computation display
# Entry [i,j] = how bad it is to predict j when true label is i
SEVERITY_PENALTY = np.array([
    [0.00, 0.06, 0.25, 0.56, 1.00],
    [0.06, 0.00, 0.06, 0.25, 0.56],
    [0.25, 0.06, 0.00, 0.06, 0.25],
    [0.56, 0.25, 0.06, 0.00, 0.06],
    [1.00, 0.56, 0.25, 0.06, 0.00],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _ensure_numpy(*arrays) -> List[np.ndarray]:
    """Convert any array-like inputs to NumPy arrays."""
    result = []
    for arr in arrays:
        if arr is None:
            result.append(None)
        else:
            result.append(np.asarray(arr))
    return result


def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    num_classes: int,
):
    """Validate shapes and value ranges of prediction arrays."""
    assert y_true.ndim == 1, f"y_true must be 1-D, got shape {y_true.shape}"
    assert y_pred.ndim == 1, f"y_pred must be 1-D, got shape {y_pred.shape}"
    assert len(y_true) == len(y_pred), (
        f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
    )
    assert set(np.unique(y_true)).issubset(set(range(num_classes))), (
        f"y_true contains labels outside [0, {num_classes-1}]"
    )
    if y_proba is not None:
        assert y_proba.ndim == 2, f"y_proba must be 2-D, got shape {y_proba.shape}"
        assert y_proba.shape == (len(y_true), num_classes), (
            f"y_proba shape {y_proba.shape} != ({len(y_true)}, {num_classes})"
        )


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    """
    Quadratic Weighted Kappa (QWK) — primary metric for APTOS 2019 competition.

    QWK measures the agreement between two raters (ground truth vs. model)
    while penalizing disagreements proportionally to the squared difference
    in ordinal grades.

    Interpretation:
        < 0.00  → Less than chance agreement
        0.00–0.20 → Slight agreement
        0.21–0.40 → Fair agreement
        0.41–0.60 → Moderate agreement
        0.61–0.80 → Substantial agreement   ← clinical requirement
        0.81–1.00 → Almost perfect agreement ← target (> 0.85)
    """
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return 0.0


def compute_per_class_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-class Sensitivity (Recall) and Specificity using
    one-vs-rest (OvR) binarization.

    Sensitivity = TP / (TP + FN)   — ability to detect positive cases
    Specificity = TN / (TN + FP)   — ability to rule out negative cases

    Returns:
        sensitivity : array of shape (num_classes,)
        specificity : array of shape (num_classes,)
    """
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)

    for c in range(num_classes):
        # Binary labels for class c
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)

        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

        sensitivity[c] = tp / (tp + fn + 1e-9)
        specificity[c] = tn / (tn + fp + 1e-9)

    return sensitivity, specificity


def compute_auc_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute AUC-ROC scores.

    Returns a dictionary with:
        'macro'     : Macro-averaged AUC-ROC (one-vs-rest)
        'weighted'  : Sample-weighted AUC-ROC
        'class_{i}' : Per-class AUC-ROC
    """
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    auc_dict: Dict[str, float] = {}

    # Per-class AUC
    for c in range(num_classes):
        try:
            auc_dict[f"class_{c}"] = float(
                roc_auc_score(y_true_bin[:, c], y_proba[:, c])
            )
        except Exception:
            auc_dict[f"class_{c}"] = 0.0

    # Macro and weighted
    try:
        auc_dict["macro"] = float(
            roc_auc_score(
                y_true, y_proba,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        auc_dict["macro"] = float(np.mean([auc_dict[f"class_{c}"] for c in range(num_classes)]))

    try:
        auc_dict["weighted"] = float(
            roc_auc_score(
                y_true, y_proba,
                multi_class="ovr",
                average="weighted",
            )
        )
    except Exception:
        auc_dict["weighted"] = auc_dict["macro"]

    return auc_dict


def compute_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute F1 scores at multiple averaging levels.

    Returns:
        macro     : Unweighted mean of per-class F1
        weighted  : Class-frequency weighted mean F1
        class_{i} : Per-class F1 score
    """
    f1_dict: Dict[str, float] = {}

    f1_dict["macro"]    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_dict["weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_dict["micro"]    = float(f1_score(y_true, y_pred, average="micro",    zero_division=0))

    per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
    for c, score in enumerate(per_class):
        f1_dict[f"class_{c}"] = float(score)

    return f1_dict


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute precision and recall at macro, weighted, and per-class level.

    Returns:
        precision_dict, recall_dict
    """
    prec_dict: Dict[str, float] = {}
    rec_dict:  Dict[str, float] = {}

    for avg in ["macro", "weighted", "micro"]:
        prec_dict[avg] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        rec_dict[avg]  = float(recall_score(y_true, y_pred, average=avg, zero_division=0))

    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
    per_rec  = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))

    for c in range(num_classes):
        prec_dict[f"class_{c}"] = float(per_prec[c])
        rec_dict[f"class_{c}"]  = float(per_rec[c])

    return prec_dict, rec_dict


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        normalize : None | 'true' | 'pred' | 'all'
                    'true'  → normalize over actual (rows)
                    'pred'  → normalize over predicted (cols)
                    'all'   → normalize over all samples

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(num_classes)),
        normalize=normalize,
    )
    return cm


def compute_average_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute Average Precision (area under the Precision-Recall curve).

    Returns per-class and macro-averaged AP scores.
    """
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    ap_dict: Dict[str, float] = {}

    for c in range(num_classes):
        try:
            ap_dict[f"class_{c}"] = float(
                average_precision_score(y_true_bin[:, c], y_proba[:, c])
            )
        except Exception:
            ap_dict[f"class_{c}"] = 0.0

    ap_dict["macro"] = float(np.mean([ap_dict[f"class_{c}"] for c in range(num_classes)]))

    return ap_dict


def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 2,
    num_classes: int = NUM_CLASSES,
) -> float:
    """
    Compute Top-K accuracy.

    A prediction is correct if the true label appears in the top-K
    most probable predictions. Useful for borderline DR grades.

    Args:
        k : Number of top predictions to consider (default: 2)
    """
    if y_proba is None or k >= num_classes:
        return 1.0

    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = np.array([
        y_true[i] in top_k_preds[i] for i in range(len(y_true))
    ])
    return float(correct.mean())


# ─────────────────────────────────────────────────────────────
# COMPREHENSIVE EVALUATOR CLASS
# ─────────────────────────────────────────────────────────────

class DRMetricsEvaluator:
    """
    Comprehensive metrics evaluator for Diabetic Retinopathy classification.

    Computes and reports:
        ┌─────────────────────────────────────────────────────┐
        │ Overall  │ Accuracy, QWK, AUC-ROC, F1 (macro/wtd)  │
        │ Clinical │ Sensitivity, Specificity per class        │
        │ Detailed │ Precision, Recall, AP per class           │
        │ Visual   │ Confusion matrix, ROC curves, PR curves   │
        └─────────────────────────────────────────────────────┘

    Args:
        num_classes : Number of DR severity classes (default: 5)
        class_names : Names for each class
    """

    def __init__(
        self,
        num_classes:  int = NUM_CLASSES,
        class_names:  List[str] = CLASS_NAMES,
    ):
        self.num_classes  = num_classes
        self.class_names  = class_names

    # ──────────────────────────────────────────────────────────
    # CORE EVALUATION
    # ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        y_true:  Union[np.ndarray, List[int]],
        y_pred:  Union[np.ndarray, List[int]],
        y_proba: Optional[Union[np.ndarray, List[List[float]]]] = None,
    ) -> Dict:
        """
        Run full evaluation and return all metrics in a nested dictionary.

        Args:
            y_true  : Ground truth labels, shape (N,)
            y_pred  : Predicted class indices, shape (N,)
            y_proba : Softmax probabilities, shape (N, C) — enables AUC metrics

        Returns:
            Dictionary with keys:
                'overall'      : scalar metrics (accuracy, qwk, …)
                'per_class'    : per-class sensitivity, specificity, F1, …
                'confusion'    : raw and normalized confusion matrices
                'auc_roc'      : per-class and averaged AUC-ROC
                'average_prec' : per-class and averaged AP
                'roc_curves'   : (fpr, tpr, thresholds) per class (if y_proba given)
                'report'       : sklearn classification report string
        """
        # ── Input validation ──────────────────────────────────
        y_true, y_pred, y_proba = _ensure_numpy(y_true, y_pred, y_proba)
        _validate_inputs(y_true, y_pred, y_proba, self.num_classes)

        results: Dict = {}

        # ── 1. Overall metrics ────────────────────────────────
        accuracy = compute_accuracy(y_true, y_pred)
        qwk      = compute_quadratic_weighted_kappa(y_true, y_pred, self.num_classes)
        f1       = compute_f1_scores(y_true, y_pred, self.num_classes)
        prec, rec = compute_precision_recall(y_true, y_pred, self.num_classes)
        sensitivity, specificity = compute_per_class_sensitivity_specificity(
            y_true, y_pred, self.num_classes
        )

        results["overall"] = {
            "accuracy":           accuracy,
            "qwk":                qwk,
            "f1_macro":           f1["macro"],
            "f1_weighted":        f1["weighted"],
            "f1_micro":           f1["micro"],
            "precision_macro":    prec["macro"],
            "precision_weighted": prec["weighted"],
            "recall_macro":       rec["macro"],
            "recall_weighted":    rec["weighted"],
            "mean_sensitivity":   float(sensitivity.mean()),
            "mean_specificity":   float(specificity.mean()),
            "n_samples":          int(len(y_true)),
        }

        # ── 2. Per-class metrics ──────────────────────────────
        per_class: Dict[str, Dict] = {}
        for c in range(self.num_classes):
            per_class[self.class_names[c]] = {
                "class_index":  c,
                "sensitivity":  float(sensitivity[c]),
                "specificity":  float(specificity[c]),
                "f1_score":     f1.get(f"class_{c}", 0.0),
                "precision":    prec.get(f"class_{c}", 0.0),
                "recall":       rec.get(f"class_{c}", 0.0),
                "support":      int(np.sum(y_true == c)),
            }

        results["per_class"] = per_class

        # ── 3. Confusion matrices ─────────────────────────────
        cm_raw  = compute_confusion_matrix(y_true, y_pred, self.num_classes, normalize=None)
        cm_norm = compute_confusion_matrix(y_true, y_pred, self.num_classes, normalize="true")

        results["confusion"] = {
            "raw":        cm_raw.tolist(),
            "normalized": cm_norm.tolist(),
        }

        # ── 4. AUC-ROC, Average Precision, ROC curves ─────────
        if y_proba is not None:
            auc_roc = compute_auc_roc(y_true, y_proba, self.num_classes)
            avg_prec = compute_average_precision(y_true, y_proba, self.num_classes)
            top2_acc = compute_top_k_accuracy(y_true, y_proba, k=2, num_classes=self.num_classes)

            # Add per-class AUC to per_class dict
            for c in range(self.num_classes):
                name = self.class_names[c]
                results["per_class"][name]["auc_roc"]      = auc_roc.get(f"class_{c}", 0.0)
                results["per_class"][name]["average_prec"] = avg_prec.get(f"class_{c}", 0.0)

            results["overall"]["auc_roc_macro"]    = auc_roc["macro"]
            results["overall"]["auc_roc_weighted"]  = auc_roc["weighted"]
            results["overall"]["avg_precision_macro"] = avg_prec["macro"]
            results["overall"]["top2_accuracy"]     = top2_acc

            results["auc_roc"]      = auc_roc
            results["average_prec"] = avg_prec

            # Compute ROC curve data per class
            y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
            roc_curves: Dict[str, Dict] = {}
            for c in range(self.num_classes):
                try:
                    fpr, tpr, thresholds = roc_curve(y_true_bin[:, c], y_proba[:, c])
                    roc_curves[self.class_names[c]] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": float(auc(fpr, tpr)),
                    }
                except Exception:
                    roc_curves[self.class_names[c]] = {"fpr": [], "tpr": [], "auc": 0.0}
            results["roc_curves"] = roc_curves

            # Precision-Recall curve data per class
            pr_curves: Dict[str, Dict] = {}
            for c in range(self.num_classes):
                try:
                    prec_c, rec_c, _ = precision_recall_curve(y_true_bin[:, c], y_proba[:, c])
                    pr_curves[self.class_names[c]] = {
                        "precision": prec_c.tolist(),
                        "recall":    rec_c.tolist(),
                        "ap":        avg_prec.get(f"class_{c}", 0.0),
                    }
                except Exception:
                    pr_curves[self.class_names[c]] = {"precision": [], "recall": [], "ap": 0.0}
            results["pr_curves"] = pr_curves

        # ── 5. Classification report ──────────────────────────
        results["report"] = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )

        # ── 6. QWK penalty analysis ───────────────────────────
        results["qwk_analysis"] = self._compute_qwk_analysis(y_true, y_pred)

        return results

    def _compute_qwk_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Detailed analysis of QWK score components.

        Breaks down which class pairs contribute most to kappa loss
        so that training can be targeted at the most costly errors.
        """
        cm = compute_confusion_matrix(y_true, y_pred, self.num_classes, normalize=None)
        n = cm.sum()

        if n == 0:
            return {}

        # Expected matrix (product of marginals)
        row_sums = cm.sum(axis=1, keepdims=True)
        col_sums = cm.sum(axis=0, keepdims=True)
        expected = (row_sums @ col_sums).astype(np.float64) / n

        # Weighted observed and expected
        w_obs = (SEVERITY_PENALTY * cm).sum() / n
        w_exp = (SEVERITY_PENALTY * expected).sum() / n
        qwk_manual = 1.0 - (w_obs / (w_exp + 1e-9))

        # Find costliest error pairs
        cost_matrix = SEVERITY_PENALTY * cm
        flat_idx = np.argsort(cost_matrix.ravel())[::-1]

        top_errors = []
        for idx in flat_idx[:10]:
            r, c = np.unravel_index(idx, (self.num_classes, self.num_classes))
            if r != c and cm[r, c] > 0:
                top_errors.append({
                    "true_class":  self.class_names[r],
                    "pred_class":  self.class_names[c],
                    "count":       int(cm[r, c]),
                    "penalty":     float(SEVERITY_PENALTY[r, c]),
                    "total_cost":  float(cost_matrix[r, c]),
                })

        return {
            "qwk_manual":   float(qwk_manual),
            "w_observed":   float(w_obs),
            "w_expected":   float(w_exp),
            "top_errors":   top_errors,
        }

    # ──────────────────────────────────────────────────────────
    # REPORTING
    # ──────────────────────────────────────────────────────────

    def print_report(self, results: Dict, show_per_class: bool = True):
        """
        Pretty-print a comprehensive evaluation report to stdout.

        Args:
            results       : Output from evaluate()
            show_per_class: Whether to include per-class breakdown table
        """
        ov = results["overall"]
        sep = "=" * 70

        print(f"\n{sep}")
        print("  DIABETIC RETINOPATHY DETECTION — EVALUATION REPORT")
        print(sep)

        # ── Overall ──────────────────────────────────────────
        print(f"\n  {'OVERALL METRICS':^66}")
        print(f"  {'-'*66}")
        print(f"  {'Metric':<38} {'Value':>12}   {'Target':>10}")
        print(f"  {'-'*66}")

        targets = {
            "Accuracy":                 "≥ 90.0%",
            "Quadratic Weighted Kappa": "≥ 0.850",
            "AUC-ROC (macro)":          "≥ 0.950",
            "F1-Score (weighted)":      "≥ 0.880",
            "Mean Sensitivity":         "≥ 85.0%",
            "Mean Specificity":         "≥ 90.0%",
        }

        metrics_display = [
            ("Accuracy",                    f"{ov['accuracy']*100:.2f}%"),
            ("Quadratic Weighted Kappa",     f"{ov['qwk']:.4f}"),
            ("AUC-ROC (macro)",              f"{ov.get('auc_roc_macro', 0):.4f}"),
            ("AUC-ROC (weighted)",           f"{ov.get('auc_roc_weighted', 0):.4f}"),
            ("F1-Score (macro)",             f"{ov['f1_macro']:.4f}"),
            ("F1-Score (weighted)",          f"{ov['f1_weighted']:.4f}"),
            ("Precision (macro)",            f"{ov['precision_macro']:.4f}"),
            ("Recall (macro)",               f"{ov['recall_macro']:.4f}"),
            ("Mean Sensitivity",             f"{ov['mean_sensitivity']*100:.2f}%"),
            ("Mean Specificity",             f"{ov['mean_specificity']*100:.2f}%"),
            ("Top-2 Accuracy",               f"{ov.get('top2_accuracy', 0)*100:.2f}%"),
            ("Avg Precision (macro, PR-AUC)",f"{ov.get('avg_precision_macro', 0):.4f}"),
            ("Total Samples",                str(ov["n_samples"])),
        ]

        for name, val in metrics_display:
            tgt = targets.get(name, "—")
            print(f"  {name:<38} {val:>12}   {tgt:>10}")

        # ── Per-class ─────────────────────────────────────────
        if show_per_class and "per_class" in results:
            print(f"\n  {'PER-CLASS BREAKDOWN':^66}")
            print(f"  {'-'*66}")
            header = f"  {'Class':<18} {'Sens':>7} {'Spec':>7} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Supp':>6}"
            print(header)
            print(f"  {'-'*66}")

            for cls_name, m in results["per_class"].items():
                auc_val = m.get("auc_roc", 0.0)
                print(
                    f"  {cls_name:<18} "
                    f"{m['sensitivity']*100:>6.1f}% "
                    f"{m['specificity']*100:>6.1f}% "
                    f"{m['f1_score']:>7.4f} "
                    f"{auc_val:>7.4f} "
                    f"{m['precision']:>7.4f} "
                    f"{m['support']:>6}"
                )

        # ── QWK analysis ──────────────────────────────────────
        if "qwk_analysis" in results and results["qwk_analysis"].get("top_errors"):
            print(f"\n  {'TOP MISCLASSIFICATION ERRORS (by QWK cost)':^66}")
            print(f"  {'-'*66}")
            print(f"  {'True → Predicted':<35} {'Count':>7} {'Penalty':>8} {'Cost':>8}")
            print(f"  {'-'*66}")
            for err in results["qwk_analysis"]["top_errors"][:5]:
                true_c = err.get("true_class", "?")
                pred_c = err.get("pred_class", "?")
                cnt = err.get("count", 0)
                pen = err.get("penalty", 0)
                cost = err.get("total_cost", 0)
                print(f"  {true_c} → {pred_c:<25} {cnt:>7} {pen:>8.4f} {cost:>8.2f}")
