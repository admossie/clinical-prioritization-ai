from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from .preprocess import load_and_prepare_data, transform_with_feature_names
from .temporal_features import add_temporal_features
from .train import split_data
from .schemas import TARGET_COLUMN
from .calibration import calibration_table, calibration_metrics
from .fairness import run_fairness_suite


def top_n_capture(y_true, probs, frac=0.2):
    order = probs.argsort()[::-1]
    cutoff = max(1, int(len(probs) * frac))
    idx = order[:cutoff]
    return float(y_true.iloc[idx].sum() / max(y_true.sum(), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    args = p.parse_args()
    df = add_temporal_features(load_and_prepare_data(args.data_path))
    _, test_df = split_data(df)
    model = joblib.load("models/best_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    X_test, y_test = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]
    Xt_test = transform_with_feature_names(X_test, preprocessor)
    probs = model.predict_proba(Xt_test)[:, 1]
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [
            {
                "roc_auc": float(roc_auc_score(y_test, probs)),
                "pr_auc": float(average_precision_score(y_test, probs)),
                "top_10_capture": top_n_capture(y_test, probs, 0.10),
                "top_20_capture": top_n_capture(y_test, probs, 0.20),
                "top_30_capture": top_n_capture(y_test, probs, 0.30),
                **calibration_metrics(y_test, probs),
            }
        ]
    )
    metrics.to_csv("outputs/tables/evaluation_metrics.csv", index=False)
    scored = test_df.copy()
    scored["risk_score"] = probs
    scored.to_csv("outputs/tables/test_scored.csv", index=False)
    for group_col, table in run_fairness_suite(
        scored, prob_col="risk_score", target_col=TARGET_COLUMN
    ).items():
        table.to_csv(f"outputs/tables/fairness_{group_col}.csv", index=False)
    calib = calibration_table(y_test, probs)
    calib.to_csv("outputs/tables/calibration_table.csv", index=False)
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("outputs/figures/roc_curve.png", bbox_inches="tight")
    plt.close()
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("outputs/figures/pr_curve.png", bbox_inches="tight")
    plt.close()
    plt.figure()
    plt.plot(calib["predicted_probability"], calib["observed_rate"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Rate")
    plt.title("Calibration Curve")
    plt.savefig("outputs/figures/calibration_curve.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
