import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.preprocess import load_and_prepare_data, transform_with_feature_names
from src.schemas import TARGET_COLUMN
from src.temporal_features import add_temporal_features

MODEL_PATH = ROOT / "models" / "best_model.joblib"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.joblib"
SCORED_PATH = ROOT / "outputs" / "tables" / "test_scored.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "diabetic_data.csv"
DOCS_OUTPUT_PATH = ROOT / "docs" / "03_explainability.png"


def format_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__", "num_missing__", "cat_missing__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ")


def load_high_risk_example(scored: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    key_cols = [column for column in ("encounter_id", "patient_nbr") if column in scored.columns and column in raw.columns]
    if not key_cols:
        return raw.head(1).copy()

    row_key = scored.sort_values("risk_score", ascending=False).iloc[0][key_cols]
    mask = pd.Series(True, index=raw.index)
    for column in key_cols:
        mask &= raw[column] == row_key[column]
    return raw.loc[mask].head(1).copy()


def save_bar_chart(contribution_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor="white")
    colors = ["#1565C0" if value < 0 else "#FF0051" for value in contribution_df["impact"]]
    ax.bar(range(len(contribution_df)), contribution_df["impact"], color=colors, width=0.8)
    ax.set_title("Top feature contributions", loc="left", fontsize=14, fontweight="bold")
    ax.axhline(0, color="#CCCCCC", linewidth=1)
    ax.grid(axis="y", color="#EAEAEA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(contribution_df)))
    ax.set_xticklabels(contribution_df["feature"], rotation=90, fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_waterfall_plot(shap_values, out_path: Path) -> None:
    plt.close("all")
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.gcf().set_size_inches(8.6, 5.9)
    plt.gcf().tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close("all")


def combine_panels(bar_path: Path, waterfall_path: Path, risk: float, out_path: Path) -> None:
    bar_img = plt.imread(bar_path)
    waterfall_img = plt.imread(waterfall_path)

    fig = plt.figure(figsize=(8.6, 11.2), facecolor="white")
    grid = fig.add_gridspec(14, 1, top=0.975, bottom=0.035, hspace=0.32)

    header = fig.add_subplot(grid[0:2, 0])
    header.axis("off")
    header.text(0.0, 0.80, "Prediction Explainability", fontsize=30, fontweight="bold", color="#1F2A44")
    header.text(
        0.0,
        0.36,
        f"Example high-risk patient from test cohort | predicted risk = {risk:.3f}",
        fontsize=15,
        color="#666666",
    )

    top_panel = fig.add_subplot(grid[2:6, 0])
    top_panel.imshow(bar_img)
    top_panel.axis("off")

    bottom_panel = fig.add_subplot(grid[6:14, 0])
    bottom_panel.imshow(waterfall_img)
    bottom_panel.axis("off")

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scored = pd.read_csv(SCORED_PATH, low_memory=False)
    raw = add_temporal_features(load_and_prepare_data(str(RAW_DATA_PATH)))

    example_row = load_high_risk_example(scored, raw)
    x = example_row.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in example_row.columns else example_row.copy()
    xt = transform_with_feature_names(x, preprocessor)
    risk = float(model.predict_proba(xt)[:, 1][0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(xt)
    shap_values.feature_names = [format_feature_name(column) for column in xt.columns]

    raw_values = np.asarray(shap_values.values)
    if raw_values.ndim == 3:
        raw_values = raw_values[:, :, -1]
    row_values = raw_values[0] if raw_values.ndim > 1 else raw_values
    feature_names = shap_values.feature_names[: len(row_values)]

    contribution_df = (
        pd.DataFrame({"feature": feature_names, "impact": row_values})
        .assign(abs_impact=lambda frame: frame["impact"].abs())
        .sort_values("abs_impact", ascending=False)
        .head(10)
    )

    tmp_bar = ROOT / "docs" / ".tmp_explainability_bar.png"
    tmp_waterfall = ROOT / "docs" / ".tmp_explainability_waterfall.png"

    save_bar_chart(contribution_df, tmp_bar)
    save_waterfall_plot(shap_values, tmp_waterfall)
    combine_panels(tmp_bar, tmp_waterfall, risk, DOCS_OUTPUT_PATH)

    tmp_bar.unlink(missing_ok=True)
    tmp_waterfall.unlink(missing_ok=True)
    print(f"Saved {DOCS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()