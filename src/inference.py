from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .preprocess import (
    build_preprocessor,
    load_and_prepare_data,
    transform_with_feature_names,
)
from .schemas import TARGET_COLUMN
from .temporal_features import add_temporal_features

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
PREPROC_PATH = ROOT / "models" / "preprocessor.joblib"
MODEL_METADATA_PATH = ROOT / "models" / "model_metadata.json"
REFERENCE_SCORES_PATH = ROOT / "outputs" / "tables" / "test_scored.csv"
FALLBACK_DATA_PATHS = [
    ROOT / "data" / "raw" / "diabetic_data.csv",
    ROOT / "data" / "raw" / "sample_diabetic_data.csv",
]
FALLBACK_TRAIN_ROWS = 5000

DEFAULT_DEMO_INPUTS: dict[str, Any] = {
    "age": "[50-60)",
    "gender": "Female",
    "race": "Caucasian",
    "time_in_hospital": 4,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 12,
    "number_diagnoses": 8,
    "number_outpatient": 1,
    "number_emergency": 0,
    "number_inpatient": 1,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "prior_inpatient": 0,
    "prior_outpatient": 0,
    "prior_emergency": 0,
    "prior_positive_count": 0,
    "diag_delta": 0,
    "med_delta": 0,
}

REQUIRED_INPUT_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "encounter_number",
    "prior_encounters",
    "prior_number_inpatient_sum",
    "prior_number_inpatient_mean",
    "prior_number_outpatient_sum",
    "prior_number_outpatient_mean",
    "prior_number_emergency_sum",
    "prior_number_emergency_mean",
    "prior_total_visits",
    "diag_delta",
    "med_delta",
    "prior_positive_count",
    "ever_prior_positive",
]

CATEGORICAL_NONE_DEFAULTS = {
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
}


def fit_fallback_pipeline():
    data_path = next((path for path in FALLBACK_DATA_PATHS if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "No dataset is available to build the fallback demo model."
        )

    df = add_temporal_features(load_and_prepare_data(str(data_path)))
    df = df.head(FALLBACK_TRAIN_ROWS).copy()

    preprocessor = build_preprocessor(df)
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    xt = preprocessor.fit_transform(x)

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="liblinear",
    )
    model.fit(xt, y)
    return model, preprocessor


@lru_cache(maxsize=1)
def load_pipeline():
    if MODEL_PATH.exists() and PREPROC_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROC_PATH)
            return model, preprocessor, "saved-artifacts"
        except Exception:
            LOGGER.debug(
                "Failed to load saved model artifacts; using fallback pipeline.",
                exc_info=True,
            )

    model, preprocessor = fit_fallback_pipeline()
    return model, preprocessor, "fallback-demo"


@lru_cache(maxsize=4)
def load_model_metadata(
    path: str | None = None, use_saved_artifacts: bool | None = None
) -> dict[str, str]:
    metadata = {
        "model_name": "clinical-readmission-prioritizer",
        "version": "v1.0.9",
        "artifact_source": "fallback-demo",
    }

    metadata_path = Path(path) if path else MODEL_METADATA_PATH
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({key: str(value) for key, value in loaded.items()})
        except Exception:
            LOGGER.debug(
                "Model metadata parsing failed; using defaults.", exc_info=True
            )

    if use_saved_artifacts is not None:
        metadata["artifact_source"] = (
            "saved-artifacts" if use_saved_artifacts else "fallback-demo"
        )

    return metadata


@lru_cache(maxsize=2)
def load_reference_cohort(path: str | None = None) -> pd.DataFrame:
    reference_path = Path(path) if path else REFERENCE_SCORES_PATH
    if not reference_path.exists():
        return pd.DataFrame(columns=["risk_score", "target"])

    candidate_cols = {"encounter_id", "patient_nbr", "risk_score", "target"}
    cohort = pd.read_csv(
        reference_path,
        usecols=lambda column_name: column_name in candidate_cols,
        low_memory=False,
    )
    cohort["risk_score"] = pd.to_numeric(cohort.get("risk_score"), errors="coerce")
    if "target" in cohort.columns:
        target_series = pd.Series(
            pd.to_numeric(cohort["target"], errors="coerce"), index=cohort.index
        )
        cohort["target"] = target_series.fillna(0)
    else:
        cohort["target"] = 0.0
    return cohort.dropna(subset=["risk_score"]).reset_index(drop=True)


def get_reference_scores() -> np.ndarray:
    reference_cohort = load_reference_cohort(str(REFERENCE_SCORES_PATH))
    if reference_cohort.empty:
        return np.array([], dtype=float)
    return reference_cohort["risk_score"].to_numpy(dtype=float)


def get_percentile_thresholds(
    reference_scores: np.ndarray, medium_pct: float = 70.0, high_pct: float = 90.0
) -> tuple[float, float]:
    if reference_scores.size == 0:
        return 0.12, 0.20
    medium_cut = float(np.percentile(reference_scores, medium_pct))
    high_cut = float(np.percentile(reference_scores, high_pct))
    return medium_cut, high_cut


def assign_percentile_tier(risk: float, medium_cut: float, high_cut: float) -> str:
    if risk >= high_cut:
        return "High"
    if risk >= medium_cut:
        return "Medium"
    return "Low"


def get_risk_percentile(risk: float, reference_scores: np.ndarray) -> float:
    if reference_scores.size == 0:
        return 100.0
    return float((reference_scores < risk).mean() * 100.0)


def apply_missing_input_defaults(row: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_INPUT_COLUMNS:
        if col in row.columns:
            continue
        if col == "age":
            row[col] = DEFAULT_DEMO_INPUTS["age"]
        elif col == "gender":
            row[col] = DEFAULT_DEMO_INPUTS["gender"]
        elif col == "race":
            row[col] = DEFAULT_DEMO_INPUTS["race"]
        elif col == "weight":
            row[col] = "Unknown"
        elif col in CATEGORICAL_NONE_DEFAULTS:
            row[col] = "None"
        else:
            row[col] = 0
    return row


def build_prediction_row(
    payload: Mapping[str, Any], encounter_id: int = 999999, patient_nbr: int = 9999
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged_payload = {**DEFAULT_DEMO_INPUTS, **dict(payload)}

    prior_inpatient = int(merged_payload.get("prior_inpatient", 0) or 0)
    prior_outpatient = int(merged_payload.get("prior_outpatient", 0) or 0)
    prior_emergency = int(merged_payload.get("prior_emergency", 0) or 0)
    prior_positive_count = int(merged_payload.get("prior_positive_count", 0) or 0)

    prior_encounters = prior_inpatient + prior_outpatient + prior_emergency
    encounter_number = prior_encounters + 1
    ever_prior_positive = 1 if prior_positive_count > 0 else 0

    row_dict: dict[str, Any] = {col: 0 for col in REQUIRED_INPUT_COLUMNS}
    row_dict.update(
        {
            "encounter_id": int(merged_payload.get("encounter_id", encounter_id)),
            "patient_nbr": int(merged_payload.get("patient_nbr", patient_nbr)),
            "race": str(merged_payload.get("race", DEFAULT_DEMO_INPUTS["race"])),
            "gender": str(merged_payload.get("gender", DEFAULT_DEMO_INPUTS["gender"])),
            "age": str(merged_payload.get("age", DEFAULT_DEMO_INPUTS["age"])),
            "admission_type_id": int(
                merged_payload.get(
                    "admission_type_id", DEFAULT_DEMO_INPUTS["admission_type_id"]
                )
            ),
            "discharge_disposition_id": int(
                merged_payload.get(
                    "discharge_disposition_id",
                    DEFAULT_DEMO_INPUTS["discharge_disposition_id"],
                )
            ),
            "admission_source_id": int(
                merged_payload.get(
                    "admission_source_id",
                    DEFAULT_DEMO_INPUTS["admission_source_id"],
                )
            ),
            "time_in_hospital": int(
                merged_payload.get(
                    "time_in_hospital", DEFAULT_DEMO_INPUTS["time_in_hospital"]
                )
            ),
            "num_lab_procedures": int(
                merged_payload.get(
                    "num_lab_procedures",
                    DEFAULT_DEMO_INPUTS["num_lab_procedures"],
                )
            ),
            "num_procedures": int(
                merged_payload.get(
                    "num_procedures", DEFAULT_DEMO_INPUTS["num_procedures"]
                )
            ),
            "num_medications": int(
                merged_payload.get(
                    "num_medications", DEFAULT_DEMO_INPUTS["num_medications"]
                )
            ),
            "number_outpatient": int(
                merged_payload.get(
                    "number_outpatient", DEFAULT_DEMO_INPUTS["number_outpatient"]
                )
            ),
            "number_emergency": int(
                merged_payload.get(
                    "number_emergency", DEFAULT_DEMO_INPUTS["number_emergency"]
                )
            ),
            "number_inpatient": int(
                merged_payload.get(
                    "number_inpatient", DEFAULT_DEMO_INPUTS["number_inpatient"]
                )
            ),
            "number_diagnoses": int(
                merged_payload.get(
                    "number_diagnoses", DEFAULT_DEMO_INPUTS["number_diagnoses"]
                )
            ),
            "encounter_number": encounter_number,
            "prior_encounters": prior_encounters,
            "prior_number_inpatient_sum": prior_inpatient,
            "prior_number_inpatient_mean": float(prior_inpatient),
            "prior_number_outpatient_sum": prior_outpatient,
            "prior_number_outpatient_mean": float(prior_outpatient),
            "prior_number_emergency_sum": prior_emergency,
            "prior_number_emergency_mean": float(prior_emergency),
            "prior_total_visits": prior_encounters,
            "diag_delta": int(merged_payload.get("diag_delta", 0) or 0),
            "med_delta": int(merged_payload.get("med_delta", 0) or 0),
            "prior_positive_count": prior_positive_count,
            "ever_prior_positive": ever_prior_positive,
        }
    )

    row = pd.DataFrame([row_dict])
    row = apply_missing_input_defaults(row)
    normalized_row_dict = {
        str(key): value.item() if isinstance(value, np.generic) else value
        for key, value in row.iloc[0].to_dict().items()
    }
    return row, normalized_row_dict


def score_patient_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    model, preprocessor, pipeline_source = load_pipeline()
    row, row_dict = build_prediction_row(payload)

    t0 = time.time()
    transformed = transform_with_feature_names(row, preprocessor)
    t1 = time.time()
    risk = float(model.predict_proba(transformed)[:, 1][0])
    t2 = time.time()

    reference_scores = get_reference_scores()
    medium_cut, high_cut = get_percentile_thresholds(reference_scores)
    tier = assign_percentile_tier(risk, medium_cut, high_cut)
    risk_percentile = get_risk_percentile(risk, reference_scores)

    model_metadata = load_model_metadata(
        str(MODEL_METADATA_PATH), pipeline_source == "saved-artifacts"
    )

    return {
        "risk": risk,
        "tier": tier,
        "risk_percentile": risk_percentile,
        "medium_cut": medium_cut,
        "high_cut": high_cut,
        "row_dict": row_dict,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_version": model_metadata.get("version", "unversioned"),
        "artifact_source": model_metadata.get("artifact_source", pipeline_source),
        "timing": {
            "preprocessor": t1 - t0,
            "model": t2 - t1,
            "total": t2 - t0,
        },
    }


def score_batch_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model, preprocessor, pipeline_source = load_pipeline()
    reference_scores = get_reference_scores()
    medium_cut, high_cut = get_percentile_thresholds(reference_scores)
    model_metadata = load_model_metadata(
        str(MODEL_METADATA_PATH), pipeline_source == "saved-artifacts"
    )

    rows: list[pd.DataFrame] = []
    row_dicts: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads, start=1):
        row, row_dict = build_prediction_row(
            payload,
            encounter_id=999999 + index,
            patient_nbr=9999 + index,
        )
        rows.append(row)
        row_dicts.append(row_dict)

    if not rows:
        return {
            "count": 0,
            "predictions": [],
            "model_version": model_metadata.get("version", "unversioned"),
            "artifact_source": model_metadata.get("artifact_source", pipeline_source),
            "tier_counts": {"High": 0, "Medium": 0, "Low": 0},
        }

    batch_frame = pd.concat(rows, ignore_index=True)
    t0 = time.time()
    transformed = transform_with_feature_names(batch_frame, preprocessor)
    t1 = time.time()
    risks = model.predict_proba(transformed)[:, 1]
    t2 = time.time()

    predictions: list[dict[str, Any]] = []
    for row_dict, risk_value in zip(row_dicts, risks):
        risk = float(risk_value)
        tier = assign_percentile_tier(risk, medium_cut, high_cut)
        risk_percentile = get_risk_percentile(risk, reference_scores)
        predictions.append(
            {
                "risk": risk,
                "tier": tier,
                "risk_percentile": risk_percentile,
                "medium_cut": medium_cut,
                "high_cut": high_cut,
                "row_dict": row_dict,
                "generated_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                "model_version": model_metadata.get("version", "unversioned"),
                "artifact_source": model_metadata.get(
                    "artifact_source", pipeline_source
                ),
            }
        )

    tier_counts = {
        tier_name: sum(1 for item in predictions if item["tier"] == tier_name)
        for tier_name in ("High", "Medium", "Low")
    }

    return {
        "count": len(predictions),
        "predictions": predictions,
        "model_version": model_metadata.get("version", "unversioned"),
        "artifact_source": model_metadata.get("artifact_source", pipeline_source),
        "tier_counts": tier_counts,
        "timing": {
            "preprocessor": t1 - t0,
            "model": t2 - t1,
            "total": t2 - t0,
        },
    }
