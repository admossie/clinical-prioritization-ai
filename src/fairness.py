from __future__ import annotations
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def subgroup_metrics(
    df: pd.DataFrame, prob_col: str, group_col: str, target_col: str = "target"
) -> pd.DataFrame:
    rows = []
    for group_value, sub in df.groupby(group_col):
        if sub[target_col].nunique() < 2:
            rows.append(
                {
                    group_col: group_value,
                    "count": len(sub),
                    "roc_auc": None,
                    "pr_auc": None,
                    "positive_rate": float(sub[target_col].mean()),
                }
            )
        else:
            rows.append(
                {
                    group_col: group_value,
                    "count": len(sub),
                    "roc_auc": float(roc_auc_score(sub[target_col], sub[prob_col])),
                    "pr_auc": float(
                        average_precision_score(sub[target_col], sub[prob_col])
                    ),
                    "positive_rate": float(sub[target_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def run_fairness_suite(
    df: pd.DataFrame, prob_col: str = "risk_score", target_col: str = "target"
) -> dict:
    return {
        col: subgroup_metrics(df, prob_col, col, target_col)
        for col in ["gender", "race", "age"]
        if col in df.columns
    }
