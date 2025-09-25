import numpy as np
from scipy.stats import ttest_rel, wilcoxon

def paired_tests(a, b):
    a, b = np.asarray(a), np.asarray(b)
    res = {}
    try:
        res['ttest_rel_p'] = float(ttest_rel(a, b, nan_policy='omit').pvalue)
    except Exception:
        res['ttest_rel_p'] = float('nan')
    try:
        res['wilcoxon_p'] = float(wilcoxon(a, b).pvalue)
    except Exception:
        res['wilcoxon_p'] = float('nan')
    return res
