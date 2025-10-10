"""
Experiment runner: nested cross-validation, plotting, and outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

from .dataset import load_dataset, map_target, infer_feature_types, basic_data_checks, save_json, TARGET_COL
from .preprocessing import build_preprocessor
from .models import build_baseline_logreg, build_baseline_tree, build_homogeneous_rf, build_heterogeneous_voting, build_heterogeneous_voting_hard
from .evaluation import compute_metrics, plot_roc, plot_pr, plot_calibration, aggregate_confusion
from .plots import savefig, bar_with_error
from .stats import wilcoxon_signed_rank

@dataclass
class ModelSpec:
    name: str
    builder: callable
    calibrate: bool = False  # optionally wrap best estimator with calibration inside inner-CV

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _get_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:,1]
        return proba.ravel()
    elif hasattr(estimator, "decision_function"):
        # Map decision scores to [0,1] via logistic for comparability
        from scipy.special import expit
        return expit(estimator.decision_function(X))
    else:
        # fallback: binary predictions as pseudo-proba
        return estimator.predict(X).astype(float)

def run_nested_cv(
    df: pd.DataFrame,
    outdir: Path,
    seed: int = 42,
    n_outer: int = 5,
    n_inner: int = 3,
    do_calibration: bool = False,
) -> None:
    _ensure_dir(outdir)
    figs_dir = outdir / "figures"
    _ensure_dir(figs_dir)

    # Dataset checks
    save_json(basic_data_checks(df), outdir / "dataset_summary.json")

    # Feature types & preprocess
    num_cols, cat_cols = infer_feature_types(df)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Define model specs
    specs: List[ModelSpec] = [
        ModelSpec("baseline_lr", build_baseline_logreg, calibrate=False),
        ModelSpec("baseline_dt", build_baseline_tree, calibrate=False),
        ModelSpec("homog_rf", build_homogeneous_rf, calibrate=False),  # RF probs are ok
        ModelSpec("hetero_vote_soft", build_heterogeneous_voting, calibrate=False),
        ModelSpec("hetero_vote_hard", build_heterogeneous_voting_hard, calibrate=False),
    ]

    # Prepare collectors
    all_results = []
    # For calibration & error analysis
    oof_store = {s.name: {"y_true": [], "y_proba": [], "y_pred": []} for s in specs}
    per_fold_pred_rows = []

    per_model_fold_metrics: Dict[str, List[float]] = {s.name: [] for s in specs}  # for primary metric
    roc_storage: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {s.name: [] for s in specs}
    pr_storage: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {s.name: [] for s in specs}
    cm_storage: Dict[str, List[np.ndarray]] = {s.name: [] for s in specs}

    # Outer CV
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        for spec in specs:
            pipe, grid = spec.builder(preprocessor)

            inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 100)
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                cv=inner,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            gs.fit(X_tr, y_tr)

            best_est = gs.best_estimator_

            # Optional calibration (kept off by default for speed)
            if spec.calibrate and do_calibration:
                best_est = CalibratedClassifierCV(best_est, method="isotonic", cv=3)
                best_est.fit(X_tr, y_tr)

            # Evaluate on outer test
            y_proba = _get_proba(best_est, X_te)
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = compute_metrics(y_te, y_proba, y_pred)

            # Store out-of-fold preds for calibration & error analysis
            oof_store[spec.name]["y_true"].extend(y_te.tolist())
            oof_store[spec.name]["y_proba"].extend(y_proba.tolist())
            oof_store[spec.name]["y_pred"].extend(y_pred.tolist())

            # Row-wise persistence
            for idx, yt, yp, yhat in zip(X_te.index.tolist(), y_te.tolist(), y_proba.tolist(), y_pred.tolist()):
                per_fold_pred_rows.append({
                    "fold": fold_idx,
                    "model": spec.name,
                    "index": int(idx),
                    "y_true": int(yt),
                    "y_proba": float(yp),
                    "y_pred": int(yhat),
                })

            metrics.update({
                "model": spec.name,
                "fold": fold_idx,
                "best_params": gs.best_params_
            })
            all_results.append(metrics)
            per_model_fold_metrics[spec.name].append(metrics["roc_auc"])

            # Store curves and confusion
            from sklearn.metrics import roc_curve, precision_recall_curve
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            prec, rec, _ = precision_recall_curve(y_te, y_proba)
            roc_storage[spec.name].append((fpr, tpr))
            pr_storage[spec.name].append((rec, prec))
            cm = np.array([[metrics["tn"], metrics["fp"]],[metrics["fn"], metrics["tp"]]])
            cm_storage[spec.name].append(cm)

            # Persist per-fold model (optional for appendix)
            model_dir = outdir / "models" / spec.name
            _ensure_dir(model_dir)
            joblib.dump(best_est, model_dir / f"fold{fold_idx}_best.joblib")

    # Save metrics CSV
    df_metrics = pd.DataFrame(all_results)
    df_metrics.to_csv(outdir / "metrics_outer_cv.csv", index=False)
    # Save per-fold predictions
    pd.DataFrame(per_fold_pred_rows).to_csv(outdir / "per_fold_predictions.csv", index=False)

    # Calibration plot using out-of-fold predictions for key contenders
    key_models = ["homog_rf", "hetero_vote_soft"]
    fig, ax = plt.subplots(figsize=(6,5))
    from sklearn.calibration import CalibrationDisplay
    for m in key_models:
        y_true_all = np.array(oof_store[m]["y_true"])
        y_proba_all = np.array(oof_store[m]["y_proba"])
        CalibrationDisplay.from_predictions(y_true_all, y_proba_all, n_bins=10, name=m, ax=ax)
    ax.set_title("Calibration (Reliability) Curves — OOF Predictions")
    ax.grid(alpha=0.3)
    savefig(fig, figs_dir / "calibration.png")

    # Error profile CSV (aggregate FN/FP rates)
    err_rows = []
    for m in oof_store:
        yt = np.array(oof_store[m]["y_true"])
        yp = np.array(oof_store[m]["y_pred"])
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        total = tn+fp+fn+tp
        err_rows.append({
            "model": m,
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "tp": int(tp),
            "fn_rate": float(fn / (fn + tp) if (fn+tp)>0 else 0.0),
            "fp_rate": float(fp / (fp + tn) if (fp+tn)>0 else 0.0),
            "error_rate": float((fn+fp)/total if total>0 else 0.0),
        })
    pd.DataFrame(err_rows).to_csv(outdir / "error_profile.csv", index=False)

    # Permutation importance for the best headline model (by mean AUC)
    # Refit a best model on the full dataset via inner CV (cv=3) to choose params; then compute permutation importance.
    headline_means = {k: v["mean"] for k, v in headline["models"].items()}
    best_model_name = max(headline_means, key=headline_means.get)

    # Rebuild pipeline & grid
    builder_map = {
        "baseline_lr": build_baseline_logreg,
        "baseline_dt": build_baseline_tree,
        "homog_rf": build_homogeneous_rf,
        "hetero_vote_soft": build_heterogeneous_voting,
        "hetero_vote_hard": build_heterogeneous_voting_hard,
    }
    pipe_best, grid_best = builder_map[best_model_name](preprocessor)

    gs_full = GridSearchCV(estimator=pipe_best, param_grid=grid_best, cv=3, scoring="roc_auc", n_jobs=-1)
    gs_full.fit(X, y)
    best_full = gs_full.best_estimator_

    # Permutation importance on full dataset
    from sklearn.inspection import permutation_importance
    r = permutation_importance(best_full, X, y, n_repeats=10, random_state=seed, scoring="roc_auc")
    feat_names = X.columns.tolist()
    import pandas as pd
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp_df.to_csv(outdir / "permutation_importance.csv", index=False)

    # Plot top-15
    top = imp_df.head(15)[::-1]  # reverse for horizontal bar
    fig, ax = plt.subplots(figsize=(7,6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Permutation importance (mean decrease in ROC AUC)")
    ax.set_title(f"Top-15 Features — {best_model_name}")
    ax.grid(axis="x", alpha=0.3)
    savefig(fig, figs_dir / "perm_importance_top15.png")


    # Aggregate & plot: ROC
    import matplotlib.pyplot as plt
    for store, name, xlabel, ylabel in [
        (roc_storage, "roc", "False Positive Rate", "True Positive Rate"),
        (pr_storage, "pr", "Recall", "Precision"),
    ]:
        fig, ax = plt.subplots(figsize=(6,5))
        for spec in specs:
            # Average by simple interpolation on common grid
            grid = np.linspace(0, 1, 200)
            curves = store[spec.name]
            if name == "roc":
                interped = [np.interp(grid, fpr, tpr) for (fpr, tpr) in curves]
            else:
                # PR stored as (rec, prec) with rec descending from 1->0 typically
                interped = [np.interp(grid, rec[::-1], prec[::-1]) for (rec, prec) in curves]
            mean_curve = np.mean(interped, axis=0)
            std_curve = np.std(interped, axis=0)
            ax.plot(grid, mean_curve, label=f"{spec.name}")
            ax.fill_between(grid, mean_curve-std_curve, mean_curve+std_curve, alpha=0.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name.upper()} (mean ± std across {len(curves)} folds)")
        ax.legend()
        ax.grid(alpha=0.3)
        savefig(fig, figs_dir / f"{name}.png")

    # Confusion matrices aggregated
    import seaborn as sns  # Only for heatmap of confusion; if not desired, can use matplotlib.imshow
    for spec in specs:
        cm_agg = cm_storage[spec.name]
        cm = np.sum(cm_agg, axis=0)
        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (agg) — {spec.name}")
        savefig(fig, figs_dir / f"confusion_{spec.name}.png")

    # Bar chart of primary metric mean±std
    labels = []
    means = []
    stds = []
    for spec in specs:
        vals = per_model_fold_metrics[spec.name]
        labels.append(spec.name)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    fig, ax = bar_with_error(labels, means, stds, ylabel="ROC AUC", title="Outer-CV ROC AUC (mean ± std)")
    savefig(fig, figs_dir / "summary_bar_auc.png")

    # Statistical test RF vs Hetero
    rf_vals = per_model_fold_metrics["homog_rf"]
    het_vals = per_model_fold_metrics["hetero_vote_soft"]
    w = wilcoxon_signed_rank(rf_vals, het_vals)
    save_json({"rf_vs_hetero_wilcoxon": w, "rf": rf_vals, "hetero": het_vals}, outdir / "wilcoxon_rf_vs_hetero.json")

    # Small report JSON for convenience
    headline = {
        "primary_metric": "roc_auc",
        "models": {labels[i]: {"mean": means[i], "std": stds[i]} for i in range(len(labels))},
    }
    save_json(headline, outdir / "headline.json")

