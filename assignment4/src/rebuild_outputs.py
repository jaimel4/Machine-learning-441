# src/rebuild_outputs.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix
)
from sklearn.calibration import CalibrationDisplay

# Project imports
from .dataset import load_dataset, map_target, TARGET_COL
from .plots import savefig, bar_with_error

# Models expected (only those with saved files will be included)
MODEL_NAMES = [
    "baseline_lr",
    "baseline_dt",
    "homog_rf",
    "hetero_vote_soft",
    "hetero_vote_hard",
]

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _get_proba(estimator, X):
    """Return probability for class 1 from any sklearn estimator/pipeline."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    elif hasattr(estimator, "decision_function"):
        from scipy.special import expit
        return expit(estimator.decision_function(X))
    else:
        return estimator.predict(X).astype(float)

def main():
    ap = argparse.ArgumentParser(
        description="Rebuild metrics/figures from saved per-fold models (no retraining)."
    )
    ap.add_argument("--data", type=Path, required=True, help="Path to breastCancer.csv")
    ap.add_argument("--models_dir", type=Path, default=Path("results/models"), help="Folder with saved fold models")
    ap.add_argument("--out", type=Path, default=Path("results"), help="Output directory (CSV + figures)")
    ap.add_argument("--seed", type=int, default=42, help="Seed used in the original run")
    ap.add_argument("--outer", type=int, default=5, help="Number of outer folds used originally")
    args = ap.parse_args()

    outdir = args.out
    figs_dir = outdir / "figures"
    _ensure_dir(figs_dir)

    # Load data and rebuild the SAME outer splits
    df = map_target(load_dataset(args.data))
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    skf = StratifiedKFold(n_splits=args.outer, shuffle=True, random_state=args.seed)
    fold_indices = list(skf.split(X, y))  # list of (train_idx, test_idx)

    # Collectors
    per_model_fold_metrics = {m: [] for m in MODEL_NAMES}
    roc_storage = {m: [] for m in MODEL_NAMES}
    pr_storage  = {m: [] for m in MODEL_NAMES}
    cm_storage  = {m: [] for m in MODEL_NAMES}
    rows = []  # per-fold metrics
    per_fold_pred_rows = []
    oof_store = {m: {"y_true": [], "y_proba": [], "y_pred": []} for m in MODEL_NAMES}

    # Iterate folds and models; load and predict
    for fold_idx, (_, test_idx) in enumerate(fold_indices, start=1):
        X_te, y_te = X.iloc[test_idx], y[test_idx]

        for model_name in MODEL_NAMES:
            model_path = args.models_dir / model_name / f"fold{fold_idx}_best.joblib"
            if not model_path.exists():
                continue  # model for this fold not present

            est = joblib.load(model_path)

            # Predict
            y_proba = _get_proba(est, X_te)
            y_pred  = (y_proba >= 0.5).astype(int)

            # Metrics
            m = {
                "model": model_name,
                "fold": fold_idx,
                "roc_auc": roc_auc_score(y_te, y_proba),
                "pr_auc": average_precision_score(y_te, y_proba),
                "accuracy": accuracy_score(y_te, y_pred),
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall": recall_score(y_te, y_pred, zero_division=0),
                "f1": f1_score(y_te, y_pred, zero_division=0),
                "brier": brier_score_loss(y_te, y_proba),
            }
            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
            m.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
            rows.append(m)
            per_model_fold_metrics[model_name].append(m["roc_auc"])

            # Curves
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            prec, rec, _ = precision_recall_curve(y_te, y_proba)
            roc_storage[model_name].append((fpr, tpr))
            pr_storage[model_name].append((rec, prec))
            cm_storage[model_name].append(np.array([[tn, fp], [fn, tp]]))

            # OOF buffers
            oof_store[model_name]["y_true"].extend(y_te.tolist())
            oof_store[model_name]["y_proba"].extend(y_proba.tolist())
            oof_store[model_name]["y_pred"].extend(y_pred.tolist())

            # Row-wise predictions
            for idx, yt, yp, yhat in zip(X_te.index.tolist(), y_te.tolist(), y_proba.tolist(), y_pred.tolist()):
                per_fold_pred_rows.append({
                    "fold": fold_idx,
                    "model": model_name,
                    "index": int(idx),
                    "y_true": int(yt),
                    "y_proba": float(yp),
                    "y_pred": int(yhat),
                })

    # Save rebuilt metrics and predictions
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "metrics_outer_cv_rebuilt.csv", index=False)
    if per_fold_pred_rows:
        pd.DataFrame(per_fold_pred_rows).to_csv(outdir / "per_fold_predictions_rebuilt.csv", index=False)

    # Plots (mean ± std bands)
    for store, name, xlabel, ylabel in [
        (roc_storage, "roc", "False Positive Rate", "True Positive Rate"),
        (pr_storage,  "pr",  "Recall",               "Precision"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        for m in MODEL_NAMES:
            curves = store[m]
            if not curves:
                continue
            grid = np.linspace(0, 1, 200)
            if name == "roc":
                interped = [np.interp(grid, fpr, tpr) for (fpr, tpr) in curves]
            else:
                # PR stored as (rec, prec); reverse recall for monotonic interp
                interped = [np.interp(grid, rec[::-1], prec[::-1]) for (rec, prec) in curves]
            mean_curve = np.mean(interped, axis=0)
            std_curve  = np.std(interped,  axis=0)
            ax.plot(grid, mean_curve, label=m)
            ax.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{name.upper()} (mean ± std across folds")
        ax.legend(); ax.grid(alpha=0.3)
        savefig(fig, figs_dir / f"{name}.png")

    # Confusion matrices (aggregated)
    try:
        import seaborn as sns
        for m in MODEL_NAMES:
            cms = cm_storage[m]
            if not cms:
                continue
            cm = np.sum(cms, axis=0)
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix (agg) — {m}")
            savefig(fig, figs_dir / f"confusion_{m}.png")
    except Exception:
        # Fallback if seaborn not available
        for m in MODEL_NAMES:
            cms = cm_storage[m]
            if not cms:
                continue
            cm = np.sum(cms, axis=0)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, int(val), ha="center", va="center")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix (agg) — {m}")
            savefig(fig, figs_dir / f"confusion_{m}.png")

    # Summary bar chart (ROC AUC)
    labels, means, stds = [], [], []
    for m in MODEL_NAMES:
        vals = per_model_fold_metrics[m]
        if not vals:
            continue
        labels.append(m)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    if labels:
        fig, ax = bar_with_error(labels, means, stds, ylabel="ROC AUC", title="Outer-CV ROC AUC")
        savefig(fig, figs_dir / "summary_bar_auc.png")

    # Calibration (OOF) for RF vs Hetero Soft (if present)
    fig, ax = plt.subplots(figsize=(6, 5))
    any_curve = False
    for m in ["homog_rf", "hetero_vote_soft"]:
        if oof_store[m]["y_true"]:
            y_true_all  = np.array(oof_store[m]["y_true"])
            y_proba_all = np.array(oof_store[m]["y_proba"])
            CalibrationDisplay.from_predictions(y_true_all, y_proba_all, n_bins=10, name=m, ax=ax)
            any_curve = True
    if any_curve:
        ax.set_title("Calibration (Reliability) Curves")
        ax.grid(alpha=0.3)
        savefig(fig, figs_dir / "calibration.png")
    else:
        plt.close(fig)

    # Error profile CSV
    err_rows = []
    for m in MODEL_NAMES:
        if not oof_store[m]["y_true"]:
            continue
        yt = np.array(oof_store[m]["y_true"])
        yp = np.array(oof_store[m]["y_pred"])
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        total = tn + fp + fn + tp
        err_rows.append({
            "model": m,
            "fn": int(fn), "fp": int(fp), "tn": int(tn), "tp": int(tp),
            "fn_rate": float(fn / (fn + tp) if (fn + tp) > 0 else 0.0),
            "fp_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0.0),
            "error_rate": float((fn + fp) / total if total > 0 else 0.0),
        })
    if err_rows:
        pd.DataFrame(err_rows).to_csv(outdir / "error_profile_rebuilt.csv", index=False)

    print("Rebuild complete. Outputs written under:", outdir)

if __name__ == "__main__":
    main()
