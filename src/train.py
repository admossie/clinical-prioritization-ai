from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier

from .cost_sensitive import optimize_threshold
from .preprocess import build_preprocessor, load_and_prepare_data
from .schemas import TARGET_COLUMN
from .temporal_features import add_temporal_features

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


DEFAULT_RANDOM_STATE = 42


def split_data(df: pd.DataFrame, target_col: str = TARGET_COLUMN):
    if "patient_nbr" in df.columns:
        x = df.drop(columns=[target_col])
        y = df[target_col]
        groups = df["patient_nbr"]
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=DEFAULT_RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(x, y, groups))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        stratify=df[target_col],
        random_state=DEFAULT_RANDOM_STATE,
    )
    return train_df.copy(), test_df.copy()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder for future feature additions while keeping the pipeline stable.
    return df.copy()


def build_candidate_models():
    models = {
        "logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        ),
        "xgb": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=DEFAULT_RANDOM_STATE,
        ),
    }

    if LGBMClassifier is not None:
        models["lgbm"] = LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=DEFAULT_RANDOM_STATE,
            verbose=-1,
        )

    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
            random_state=DEFAULT_RANDOM_STATE,
        )

    return models


def evaluate_predictions(y_true, probs):
    roc_auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    fpr, tpr, _ = roc_curve(y_true, probs)
    recall_candidates = [rec for fp, rec in zip(fpr, tpr) if fp <= 0.10]
    recall_at_10fpr = float(max(recall_candidates)) if recall_candidates else 0.0
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall@10%FPR": recall_at_10fpr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()

    df = add_temporal_features(load_and_prepare_data(args.data_path))
    df = feature_engineering(df)

    train_df, test_df = split_data(df)
    preprocessor = build_preprocessor(train_df)

    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    xt_train = preprocessor.fit_transform(x_train)
    xt_test = preprocessor.transform(x_test)

    models = build_candidate_models()
    fitted = {}
    rows = []

    for name, model in models.items():
        model.fit(xt_train, y_train)
        probs = model.predict_proba(xt_test)[:, 1]
        metrics = evaluate_predictions(y_test, probs)
        rows.append({"model": name, **metrics})
        fitted[name] = (model, probs)

    if len(models) >= 2:
        ensemble = VotingClassifier(
            estimators=[
                (name, model) for name, model in build_candidate_models().items()
            ],
            voting="soft",
        )
        ensemble.fit(xt_train, y_train)
        ensemble_probs = ensemble.predict_proba(xt_test)[:, 1]
        rows.append(
            {"model": "ensemble", **evaluate_predictions(y_test, ensemble_probs)}
        )
        fitted["ensemble"] = (ensemble, ensemble_probs)

    results_df = (
        pd.DataFrame(rows)
        .sort_values(
            ["roc_auc", "recall@10%FPR", "pr_auc"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_name = str(results_df.iloc[0]["model"])
    best_model, best_probs = fitted[best_name]
    best_threshold, threshold_table = optimize_threshold(y_test, best_probs)

    Path("models").mkdir(exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    results_df.to_csv("outputs/tables/model_comparison.csv", index=False)
    pd.DataFrame([best_threshold]).to_csv(
        "outputs/tables/best_threshold.csv", index=False
    )
    threshold_table.to_csv("outputs/tables/threshold_sweep.csv", index=False)

    scored = test_df.copy()
    scored["risk_score"] = best_probs
    scored.to_csv("outputs/tables/test_scored.csv", index=False)

    print("\n--- Quick Validation ---")
    print("Best model:", best_name)
    print("Best threshold (cost-sensitive):", best_threshold)
    print("Score percentiles:")
    print(np.percentile(best_probs, [10, 25, 50, 75, 90, 95]))
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
