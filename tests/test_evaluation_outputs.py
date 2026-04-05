from pathlib import Path

import pandas as pd


def test_evaluation_tables_exist_with_expected_columns():
    metrics_path = Path("outputs/tables/evaluation_metrics.csv")
    comparison_path = Path("outputs/tables/model_comparison.csv")

    assert metrics_path.exists()
    assert comparison_path.exists()

    metrics = pd.read_csv(metrics_path)
    comparison = pd.read_csv(comparison_path)

    assert {
        "roc_auc",
        "pr_auc",
        "top_10_capture",
        "top_20_capture",
        "top_30_capture",
        "brier_score",
    }.issubset(metrics.columns)
    assert {"model", "roc_auc", "pr_auc", "recall@10%FPR"}.issubset(comparison.columns)
    assert not comparison.empty
