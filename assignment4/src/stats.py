"""
Statistical tests for paired model comparisons.
"""
from __future__ import annotations

from typing import List, Dict
import numpy as np
from scipy.stats import wilcoxon

def wilcoxon_signed_rank(a: List[float], b: List[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test.
    Returns test statistic and p-value.
    """
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", correction=False)
    return {"statistic": float(stat), "p_value": float(p)}
