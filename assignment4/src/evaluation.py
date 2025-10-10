"""
Metric computation and evaluation helpers.
"""
from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute a comprehensive metric set for binary classification."""
    # Guard: if classifier didn't produce proba for class 1, fallback
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        p1 = y_proba[:, 1]
    else:
        p1 = y_proba.ravel()
    metrics = {
        "roc_auc": roc_auc_score(y_true, p1),
        "pr_auc": average_precision_score(y_true, p1),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, p1),
    }
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return metrics

def plot_roc(ax, y_true, y_proba, label: str):
    RocCurveDisplay.from_predictions(y_true, y_proba, name=label, ax=ax)

def plot_pr(ax, y_true, y_proba, label: str):
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=label, ax=ax)

def plot_calibration(ax, y_true, y_proba, n_bins: int = 10, label: str = "model"):
    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=n_bins, name=label, ax=ax)

def aggregate_confusion(cm_list: List[np.ndarray]) -> np.ndarray:
    """Sum confusion matrices over folds."""
    agg = np.zeros((2,2), dtype=int)
    for cm in cm_list:
        agg += cm
    return agg
