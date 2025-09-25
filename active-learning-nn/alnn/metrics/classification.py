import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def final_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def maybe_auc(y_true, proba):
    try:
        return roc_auc_score(y_true, proba, multi_class='ovr')
    except Exception:
        return np.nan
