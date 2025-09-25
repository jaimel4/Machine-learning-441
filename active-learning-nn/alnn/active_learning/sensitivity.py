from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Set

def query_sensitivity(model, X_pool: np.ndarray, unlabeled: Set[int], k: int, task: str):
    """
    Sensitivity via gradient of a surrogate loss wrt inputs:
      - Classification: CE(logits, argmax(logits)) on pseudo-labels
      - Regression: 0.5 * y_pred^2
    """
    model.eval()
    idxs = np.fromiter(unlabeled, dtype=int)
    X = torch.from_numpy(X_pool[idxs]).float()
    X.requires_grad_(True)

    scores = []
    for i in range(len(X)):
        xi = X[i:i+1]
        yi = model(xi)
        if task == "classification":
            yhat = yi.argmax(dim=1)      # pseudo-label
            loss = F.cross_entropy(yi, yhat)
        else:
            y_pred = yi.view(-1)
            loss = 0.5 * (y_pred ** 2).mean()

        if X.grad is not None:
            X.grad.zero_()
        loss.backward(retain_graph=True)
        g = X.grad[i].detach()
        scores.append(g.norm().item())

    order = np.argsort(np.array(scores))[::-1]
    chosen = idxs[order[:k]]
    return chosen.tolist()
