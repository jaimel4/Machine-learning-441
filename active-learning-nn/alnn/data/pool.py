import numpy as np

def initial_label_pool(y, init_frac: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(y)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    k = max(1, int(init_frac * n))
    labeled = set(idxs[:k].tolist())
    unlabeled = set(idxs[k:].tolist())
    return labeled, unlabeled
