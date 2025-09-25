from __future__ import annotations
import numpy as np
from typing import Dict, List, Set
from .passive import query_passive
from .uncertainty import query_uncertainty
from .sensitivity import query_sensitivity
from ..training.sgd import train_sgd, evaluate
from ..models.mlp import MLP1H

def run_active_learning(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    task: str,
    hidden_dim: int,
    activation: str,
    weight_decay: float,
    lr: float,
    batch_size: int,
    patience: int,
    init_frac: float,
    query_size: int,
    budget: int,
    strategy: str = "passive",
    mc_dropout_passes: int = 0,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    n = len(X_tr)
    idxs = np.arange(n)
    rng.shuffle(idxs)

    k0 = max(1, int(init_frac * n))
    labeled: Set[int] = set(idxs[:k0].tolist())
    unlabeled: Set[int] = set(idxs[k0:].tolist())

    # Build model
    out_dim = int(y_tr.max()+1) if task == "classification" else 1
    model = MLP1H(X_tr.shape[1], hidden_dim, out_dim, activation=activation)

    logs = []
    steps = 0
    while len(labeled) < k0 + budget and len(unlabeled) > 0:
        # Train on current labeled set
        L = sorted(list(labeled))
        model, _ = train_sgd(
            model,
            X_tr[L], y_tr[L],
            X_val, y_val,
            task,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size, epochs=200, patience=patience
        )

        # Evaluate on test
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        te_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=1024)
        te_metrics = evaluate(model, te_loader, task)
        logs.append({"labeled": len(labeled), **te_metrics})

        # Select new queries
        k = min(query_size, len(unlabeled))
        if k == 0:
            break

        if strategy == "passive":
            new_idxs = query_passive(unlabeled, k)
        elif strategy == "uncertainty":
            new_idxs = query_uncertainty(model, X_tr, unlabeled, k, task, mc_passes=mc_dropout_passes)
        elif strategy == "sensitivity":
            new_idxs = query_sensitivity(model, X_tr, unlabeled, k, task)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for j in new_idxs:
            labeled.add(j)
            unlabeled.remove(j)
        steps += 1

    return logs
