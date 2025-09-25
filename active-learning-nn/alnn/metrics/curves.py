import numpy as np

def area_under_learning_curve(xs, ys):
    """Trapezoidal area assuming xs increasing (labeled counts) and ys are scores (acc or -loss)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return np.trapz(ys, xs)
