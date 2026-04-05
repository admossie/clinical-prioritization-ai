
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix


def total_cost(y_true, y_prob, threshold, fp_cost=120.0, fn_cost=8000.0, tp_benefit=2500.0):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": float(fp * fp_cost + fn * fn_cost - tp * tp_benefit),
    }


def optimize_threshold(y_true, y_prob, fp_cost=120.0, fn_cost=8000.0, tp_benefit=2500.0, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    rows = [total_cost(y_true, y_prob, t, fp_cost, fn_cost, tp_benefit) for t in thresholds]
    table = pd.DataFrame(rows).sort_values(["total_cost", "threshold"]).reset_index(drop=True)
    return table.iloc[0].to_dict(), table
