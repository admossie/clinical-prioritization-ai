from __future__ import annotations
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return pd.DataFrame(
        {"predicted_probability": prob_pred, "observed_rate": prob_true}
    )


def calibration_metrics(y_true, y_prob) -> dict:
    return {"brier_score": float(brier_score_loss(y_true, y_prob))}
