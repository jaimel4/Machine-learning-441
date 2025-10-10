"""
Permutation importance aggregation across outer folds.
"""
from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def compute_permutation_importance(estimator, X, y, n_repeats: int = 10, random_state: int = 0) -> pd.DataFrame:
    """Compute permutation importance for a fitted pipeline.
    Works with Pipeline(preprocessor, model). Column names lost after preprocessing;
    will return feature indices of the transformed space. For report, consider using RF feature_importances_ too.
    """
    r = permutation_importance(estimator, X, y, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc")
    df = pd.DataFrame({
        "feature_index": np.arange(len(r.importances_mean)),
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df
