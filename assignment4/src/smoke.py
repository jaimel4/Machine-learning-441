# src/smoke.py
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, brier_score_loss

from .dataset import load_dataset, map_target, infer_feature_types
from .preprocessing import build_preprocessor

def main():
    data_path = Path("data/breastCancer.csv")
    df = map_target(load_dataset(data_path))
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].values

    num_cols, cat_cols = infer_feature_types(df)
    pre = build_preprocessor(num_cols, cat_cols)

    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))])

    # tiny grid = fast
    grid = {"clf__C": [1.0], "clf__penalty": ["l2"], "clf__class_weight": [None]}

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (tr_idx, te_idx), = sss.split(X, y)
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    gs = GridSearchCV(pipe, grid, cv=2, scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)

    proba = gs.predict_proba(X_te)[:,1]
    pred = (proba >= 0.5).astype(int)

    print("=== SMOKE TEST (LogReg, 80/20 split) ===")
    print("Best params:", gs.best_params_)
    print("ROC AUC:", round(roc_auc_score(y_te, proba), 4))
    print("PR AUC:", round(average_precision_score(y_te, proba), 4))
    print("Accuracy:", round(accuracy_score(y_te, pred), 4))
    print("Precision:", round(precision_score(y_te, pred), 4))
    print("Recall:", round(recall_score(y_te, pred), 4))
    print("F1:", round(f1_score(y_te, pred), 4))
    print("Brier:", round(brier_score_loss(y_te, proba), 4))

if __name__ == "__main__":
    main()
