from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Set

def _entropy(p):
    eps = 1e-12
    p = p.clamp(eps, 1.0)
    return -(p * p.log()).sum(dim=1)

def query_uncertainty(model, X_pool: np.ndarray, unlabeled: Set[int], k: int, task: str, mc_passes: int = 0):
    model.eval()
    idxs = np.fromiter(unlabeled, dtype=int)
    X = torch.from_numpy(X_pool[idxs]).float()

    if task == "classification":
        if mc_passes and mc_passes > 0:
            model.train()  # enable dropout
            ps = []
            for _ in range(mc_passes):
                logits = model(X)
                ps.append(F.softmax(logits, dim=1).detach())
            P = torch.stack(ps).mean(dim=0)
        else:
            with torch.no_grad():
                logits = model(X)
                P = F.softmax(logits, dim=1)
        H = _entropy(P)  # higher = more uncertain
        order = torch.argsort(H, descending=True).cpu().numpy()
    else:
        if mc_passes and mc_passes > 0:
            model.train()  # enable dropout
            preds = []
            for _ in range(mc_passes):
                with torch.no_grad():
                    preds.append(model(X).squeeze(1).detach())
            P = torch.stack(preds, dim=0)  # [T, N]
            var = P.var(dim=0)             # predictive variance
            order = torch.argsort(var, descending=True).cpu().numpy()
        else:
            # fallback: magnitude heuristic
            with torch.no_grad():
                preds = model(X).squeeze(1)
            order = torch.argsort(preds.abs(), descending=True).cpu().numpy()

    chosen = idxs[order[:k]]
    return chosen.tolist()
