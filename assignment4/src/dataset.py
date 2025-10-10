"""
Dataset utilities for the breast cancer classification task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import json

TARGET_COL = "diagnosis"
DROP_COLS = ["id"]
LIKELY_CATEGORICAL = {"gender"}

def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV/TSV robustly and normalise headers + values."""
    # Auto-detect delimiter; handle quotes
    df = pd.read_csv(path, sep=None, engine="python")

    # Normalise column names
    df.columns = [c.strip().strip('"').strip("'").replace(" ", "_").lower() for c in df.columns]

    # Strip quotes/whitespace from string cells
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.strip('"').str.strip("'")

    # Ensure target present
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}"

    # Coerce all non-target, non-categorical columns to numeric
    for c in df.columns:
        if c in (TARGET_COL, "gender"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map diagnosis B/M -> 0/1 and drop unused columns."""
    out = df.copy()
    # Drop id if present
    for c in DROP_COLS:
        if c in out.columns:
            out = out.drop(columns=[c])
    # Map target
    out[TARGET_COL] = out[TARGET_COL].map({"B": 0, "M": 1}).astype(int)
    return out

def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical features. Force 'gender' to categorical if present.
    Excludes target from both lists.
    """
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c == TARGET_COL:
            continue
        if c in LIKELY_CATEGORICAL:
            categorical_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols

def basic_data_checks(df: pd.DataFrame) -> Dict:
    """Return simple data quality stats for the report and save JSON alongside results."""
    n_rows, n_cols = df.shape
    class_counts = df[TARGET_COL].value_counts(dropna=False).to_dict()
    missing_per_col = df.isna().sum().to_dict()
    duplicated_rows = int(df.duplicated().sum())
    zero_var_cols = [c for c in df.columns if c != TARGET_COL and df[c].nunique(dropna=False) <= 1]

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "class_counts": class_counts,
        "missing_per_col": missing_per_col,
        "duplicated_rows": duplicated_rows,
        "zero_variance_cols": zero_var_cols,
    }

def save_json(d: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
