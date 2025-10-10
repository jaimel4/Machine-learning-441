"""
Model builders and parameter grids.
"""
from __future__ import annotations

from typing import Dict, Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def build_baseline_logreg(preprocessor) -> Tuple[Pipeline, Dict]:
    model = LogisticRegression(max_iter=2000, solver="liblinear")
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"],
        "clf__class_weight": [None, "balanced"],
    }
    return pipe, grid

def build_baseline_tree(preprocessor) -> Tuple[Pipeline, Dict]:
    model = DecisionTreeClassifier(random_state=0)
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    grid = {
        "clf__max_depth": [3, 5, 8, None],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__criterion": ["gini", "entropy"],
        "clf__class_weight": [None, "balanced"],
    }
    return pipe, grid

def build_homogeneous_rf(preprocessor) -> Tuple[Pipeline, Dict]:
    model = RandomForestClassifier(random_state=0, n_jobs=-1, bootstrap=True)
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2", 0.5],
        "clf__class_weight": [None, "balanced"],
    }
    return pipe, grid

def build_heterogeneous_voting(preprocessor) -> Tuple[Pipeline, Dict]:
    lr = LogisticRegression(max_iter=2000, solver="liblinear")
    svc = SVC(probability=True, kernel="rbf", random_state=0)
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=0)

    vclf = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("svc", svc),
            ("knn", knn),
            ("rf", rf),
            ("gb", gb),
        ],
        voting="soft",
        weights=None,  # tuned via grid
        n_jobs=None,   # VotingClassifier ignores n_jobs for members; set on members if needed
        flatten_transform=True,
    )
    pipe = Pipeline([("pre", preprocessor), ("vclf", vclf)])

    # Keep grids intentionally small to maintain run-time sanity
    grid = {
        "vclf__weights": [
            None,
            [1,1,1,1,1],
            [2,1,1,2,1],
            [2,2,1,2,1],
            [3,1,1,2,1],
        ],
        "vclf__lr__C": [0.5, 1.0, 2.0],
        "vclf__lr__class_weight": [None, "balanced"],
        "vclf__svc__C": [0.5, 1.0, 2.0],
        "vclf__svc__gamma": ["scale", 0.1],
        "vclf__knn__n_neighbors": [5, 9, 15],
        "vclf__knn__weights": ["uniform", "distance"],
        "vclf__rf__n_estimators": [150, 300],
        "vclf__rf__max_depth": [None, 10],
        "vclf__gb__n_estimators": [100, 200],
        "vclf__gb__learning_rate": [0.05, 0.1],
        "vclf__gb__max_depth": [2, 3],
    }
    return pipe, grid


def build_heterogeneous_voting_hard(preprocessor) -> Tuple[Pipeline, Dict]:
    """Hard-voting heterogeneous ensemble for ablation."""
    lr = LogisticRegression(max_iter=2000, solver="liblinear")
    svc = SVC(probability=True, kernel="rbf", random_state=0)
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=0)

    vclf = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("svc", svc),
            ("knn", knn),
            ("rf", rf),
            ("gb", gb),
        ],
        voting="hard",
        weights=None,
        flatten_transform=True,
    )
    pipe = Pipeline([("pre", preprocessor), ("vclf", vclf)])
    grid = {
        "vclf__lr__C": [0.5, 1.0, 2.0],
        "vclf__lr__class_weight": [None, "balanced"],
        "vclf__svc__C": [0.5, 1.0, 2.0],
        "vclf__svc__gamma": ["scale", 0.1],
        "vclf__knn__n_neighbors": [5, 9, 15],
        "vclf__knn__weights": ["uniform", "distance"],
        "vclf__rf__n_estimators": [150, 300],
        "vclf__rf__max_depth": [None, 10],
        "vclf__gb__n_estimators": [100, 200],
        "vclf__gb__learning_rate": [0.05, 0.1],
        "vclf__gb__max_depth": [2, 3],
    }
    return pipe, grid
