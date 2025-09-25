from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes, fetch_california_housing

def load_dataset(name: str, data_dir: str = "data/raw"):
    name = name.lower()
    if name in {"breast_cancer", "cancer"}:
        ds = load_breast_cancer()
        X, y = ds.data.astype(np.float32), ds.target.astype(int)
        task = "classification"
    elif name in {"digits"}:
        ds = load_digits()
        X, y = ds.data.astype(np.float32), ds.target.astype(int)
        task = "classification"
    elif name in {"clf_synth"}:
        X, y = make_classification(
            n_samples=4000, n_features=20, n_informative=10, n_redundant=5,
            n_classes=3, weights=[0.2, 0.3, 0.5], random_state=0
        )
        X, y = X.astype(np.float32), y.astype(int)
        task = "classification"
    elif name in {"energy"}:
        # UCI Energy Efficiency: if CSV exists
        path = os.path.join(data_dir, "energy_efficiency.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected {path}. Place CSV into data/raw/")
        df = pd.read_csv(path)
        # Assume columns: features + target named 'y'; adapt as needed
        y = df["y"].to_numpy(np.float32)
        X = df.drop(columns=["y"]).to_numpy(np.float32)
        task = "regression"
    elif name in {"airfoil"}:
        path = os.path.join(data_dir, "airfoil.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected {path}. Place CSV into data/raw/")
        df = pd.read_csv(path)
        y = df["y"].to_numpy(np.float32)
        X = df.drop(columns=["y"]).to_numpy(np.float32)
        task = "regression"
    elif name in {"sine"}:
        rng = np.random.RandomState(0)
        X = rng.uniform(-6, 6, size=(3000, 1)).astype(np.float32)
        y = (np.sin(X) + 0.3*np.sin(3*X) + 0.1*rng.randn(*X.shape)).astype(np.float32).ravel()
        task = "regression"
    elif name in {"diabetes"}:
        ds = load_diabetes()
        X, y = ds.data.astype(np.float32), ds.target.astype(np.float32)
        task = "regression"
    elif name in {"california"}:
        ds = fetch_california_housing()
        X, y = ds.data.astype(np.float32), ds.target.astype(np.float32)
        task = "regression"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return X, y, task

def make_splits(X, y, task: str, test_size=0.2, val_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if task=="classification" else None)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train if task=="classification" else None)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return (X_tr, y_tr), (X_val, y_val), (X_test, y_test), scaler
