from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from .preprocess import align_to_preprocessor_input, transform_with_feature_names


def map_external_dataset(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame()
    for target_col, source_col in mapping.items():
        out[target_col] = df[source_col]
    return out


def evaluate_external(model, preprocessor, df, target_col="target"):
    X = align_to_preprocessor_input(df.drop(columns=[target_col]), preprocessor)
    y = df[target_col]
    Xt = transform_with_feature_names(X, preprocessor)
    probs = model.predict_proba(Xt)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
    }, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-path", default="models/best_model.joblib")
    p.add_argument("--preprocessor-path", default="models/preprocessor.joblib")
    p.add_argument("--mapping-json", default=None)
    args = p.parse_args()
    raw = pd.read_csv(args.data_path)
    if args.mapping_json:
        mapping = json.load(open(args.mapping_json, "r", encoding="utf-8"))[
            "column_mapping"
        ]
        df = map_external_dataset(raw, mapping)
    else:
        df = raw.copy()
    model = joblib.load(args.model_path)
    preprocessor = joblib.load(args.preprocessor_path)
    metrics, probs = evaluate_external(model, preprocessor, df)
    pd.DataFrame([metrics]).to_csv(
        "outputs/tables/external_validation_metrics.csv", index=False
    )
    scored = df.copy()
    scored["risk_score"] = probs
    scored.to_csv("outputs/tables/external_scored.csv", index=False)


if __name__ == "__main__":
    main()
