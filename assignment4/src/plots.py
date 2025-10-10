"""
Plotting utilities.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import itertools

def savefig(fig, path: str, tight: bool = True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def bar_with_error(xlabels: List[str], means: List[float], stds: List[float], ylabel: str, title: str):
    import matplotlib.pyplot as plt
    x = np.arange(len(xlabels))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x, xlabels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return fig, ax
