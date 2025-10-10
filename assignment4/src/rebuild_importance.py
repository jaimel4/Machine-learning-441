from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# Project imports
from .dataset import load_dataset, map_target, TARGET_COL

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _get_proba(estimator, X):
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
        description="Rebuild permutation importance from saved per-fold models (no retraining)."
    )
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models_dir", type=Path, default=Path("results/models"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--model", type=str, default=None,
                    help="Model name to analyse (e.g., 'hetero_vote_soft', 'homog_rf'). "
                         "If omitted, selects the best by mean ROC AUC from metrics_outer_cv_rebuilt.csv.")
    ap.add_argument("--repeats", type=int, default=10, help="Permutation repeats per fold")
    args = ap.parse_args()

    outdir = args.out
    figs_dir = outdir / "figures"
    _ensure_dir(figs_dir)

    # Load data
    df = map_target(load_dataset(args.data))
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    feature_names = X.columns.tolist()

    # Choose model to analyse
    model_name = args.model
    metrics_path = outdir / "metrics_outer_cv_rebuilt.csv"
    if model_name is None:
        if metrics_path.exists():
            met = pd.read_csv(metrics_path)
            best = (met.groupby("model")["roc_auc"]
                        .mean()
                        .sort_values(ascending=False))
            if not best.empty:
                model_name = best.index[0]
            else:
                # fallback preference
                model_name = "hetero_vote_soft"
        else:
            model_name = "hetero_vote_soft"

    # Reconstruct outer splits
    skf = StratifiedKFold(n_splits=args.outer, shuffle=True, random_state=args.seed)
    fold_indices = list(skf.split(X, y))

    # Collect per-fold importances
    imp_means = []
    imp_stds  = []
    used_folds = 0

    for fold_idx, (_, test_idx) in enumerate(fold_indices, start=1):
        model_path = args.models_dir / model_name / f"fold{fold_idx}_best.joblib"
        if not model_path.exists():
            continue

        est = joblib.load(model_path)
        X_te, y_te = X.iloc[test_idx], y[test_idx]

        # quick sanity check to avoid degenerate cases
        try:
            _ = roc_auc_score(y_te, _get_proba(est, X_te))
        except Exception:
            # if the loaded estimator cannot score, skip
            continue

        r = permutation_importance(
            est, X_te, y_te,
            n_repeats=args.repeats,
            random_state=args.seed,
            scoring="roc_auc"
        )
        # r.importances_mean is aligned with columns of X_te (original feature names)
        imp_means.append(r.importances_mean)
        imp_stds.append(r.importances_std)
        used_folds += 1

    if used_folds == 0:
        raise SystemExit(f"No saved models found for '{model_name}'. "
                         f"Checked: {args.models_dir}/{model_name}/fold*_best.joblib")

    # Aggregate across folds
    imp_means = np.vstack(imp_means)
    imp_stds  = np.vstack(imp_stds)
    agg_mean  = imp_means.mean(axis=0)
    agg_std   = imp_means.std(axis=0)

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": agg_mean,
        "importance_std": agg_std,
        "folds_used": used_folds,
        "model": model_name,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = outdir / "permutation_importance_rebuilt.csv"
    imp_df.to_csv(csv_path, index=False)

    # Plot top-15
    top = imp_df.head(15).iloc[::-1]  # reverse for horizontal bar
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Permutation importance (Δ ROC AUC)")
    ax.set_title(f"Top-15 Features — {model_name} (rebuilt, n_folds={used_folds})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs_dir / "perm_importance_top15.png", dpi=150)
    plt.close(fig)

    print("Permutation importance saved:",
          csv_path, "and", figs_dir / "perm_importance_top15.png")

if __name__ == "__main__":
    main()
