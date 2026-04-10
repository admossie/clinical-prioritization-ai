import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="AI Care Prioritization Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
PREPROC_PATH = ROOT / "models" / "preprocessor.joblib"
MODEL_METADATA_PATH = ROOT / "models" / "model_metadata.json"
REFERENCE_SCORES_PATH = ROOT / "outputs" / "tables" / "test_scored.csv"
FALLBACK_DATA_PATHS = [
    ROOT / "data" / "raw" / "diabetic_data.csv",
    ROOT / "data" / "raw" / "sample_diabetic_data.csv",
]
FALLBACK_TRAIN_ROWS = 250
sys.path.append(str(ROOT))
from src.preprocess import (  # noqa: E402
    build_preprocessor,
    load_and_prepare_data,
    transform_with_feature_names,
)
from src.schemas import TARGET_COLUMN  # noqa: E402
from src.temporal_features import add_temporal_features  # noqa: E402
from src.workflow_simulation import hospital_roi  # noqa: E402


def get_screenshot_mode() -> str:
    value = st.query_params.get("screenshot", "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip().lower()


def format_explainability_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__", "num_missing__", "cat_missing__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ")


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
            .hero-card {
                padding: 1.35rem 1.25rem;
                border-radius: 18px;
                background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
                color: white;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
            }
            .hero-card h1 {
                margin: 0 0 0.35rem 0;
                font-size: 2rem;
            }
            .hero-card p {
                margin: 0.2rem 0;
                opacity: 0.95;
            }
            .pill-row {
                margin-top: 0.75rem;
            }
            .pill {
                display: inline-block;
                padding: 0.25rem 0.6rem;
                margin-right: 0.4rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.16);
                font-size: 0.82rem;
            }
            .result-banner {
                padding: 1rem 1.1rem;
                border-radius: 14px;
                margin: 0.5rem 0 1rem 0;
                border-left: 6px solid transparent;
            }
            .result-banner.high {
                background: #fef2f2;
                border-left-color: #dc2626;
            }
            .result-banner.medium {
                background: #fff7ed;
                border-left-color: #ea580c;
            }
            .result-banner.low {
                background: #ecfdf5;
                border-left-color: #16a34a;
            }
            .result-banner h3 {
                margin: 0 0 0.3rem 0;
            }
            .info-card {
                padding: 1rem;
                border-radius: 14px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                margin-bottom: 0.75rem;
            }
            .info-card h4 {
                margin: 0 0 0.3rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_sidebar(model_metadata: dict[str, str]) -> None:
    with st.sidebar:
        st.markdown("## Demo overview")
        st.caption(
            "A startup-style clinical operations demo for readmission prioritization."
        )
        st.markdown(
            "- Predict readmission risk\n"
            "- Prioritize limited care-team capacity\n"
            "- Estimate operational ROI"
        )
        st.markdown("### Best for")
        st.write("Care coordination leaders, hospital ops teams, and pilot partners.")
        st.markdown("### Model status")
        st.caption(f"Version: {model_metadata.get('version', 'unversioned')}")
        artifact_source = model_metadata.get("artifact_source", "unknown")
        st.caption(f"Source: {artifact_source.replace('-', ' ').title()}")
        st.info(
            "This tool supports prioritization decisions; "
            "it does not replace clinical judgment."
        )


def render_hero(reference_cohort: pd.DataFrame, reference_scores: np.ndarray) -> None:
    total_patients = int(len(reference_cohort)) if not reference_cohort.empty else 0
    observed_positives = (
        int(reference_cohort["target"].sum())
        if not reference_cohort.empty and "target" in reference_cohort.columns
        else 0
    )
    mode_label = (
        "Live reference mode" if reference_scores.size else "Fallback demo mode"
    )

    st.markdown(
        f"""
        <div class="hero-card">
            <h1>AI Care Prioritization Engine</h1>
            <p>
                Capacity-aware readmission prioritization for care teams,
                pilots, and operational decision support.
            </p>
            <div class="pill-row">
                <span class="pill">{mode_label}</span>
                <span class="pill">{total_patients:,} reference patients</span>
                <span class="pill">{observed_positives:,} observed positives</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Reference cohort", f"{total_patients:,}")
    k2.metric("Observed positives", f"{observed_positives:,}")
    k3.metric("Operating mode", "Live" if reference_scores.size else "Fallback")


def render_prediction_banner(tier: str, risk: float, percentile: float) -> None:
    tone = tier.lower()
    st.markdown(
        f"""
        <div class="result-banner {tone}">
            <h3>{tier} operational priority</h3>
            <p>
                Estimated readmission risk: <strong>{format_percent(risk)}</strong>
                · Higher than <strong>{percentile:.1f}%</strong> of the reference cohort
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_startup_highlights() -> None:
    c1, c2, c3 = st.columns(3)
    c1.markdown("### Faster triage")
    c1.caption("Turn model output into a daily intervention queue for care managers.")
    c2.markdown("### ROI visibility")
    c2.caption("Show expected savings, program cost, and operational tradeoffs.")
    c3.markdown("### Pilot-ready")
    c3.caption(
        "Built for demos with hospital operations, clinical leadership, and partners."
    )


def render_landing_section() -> None:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("## Why hospitals would use this")
        st.markdown(
            "- identify high-risk discharges earlier\n"
            "- focus outreach resources where they matter most\n"
            "- connect model outputs to operational ROI and queue planning"
        )
    with right:
        st.markdown(
            """
            <div class="info-card">
                <h4>Problem</h4>
                <p>
                    Readmission models often stop at scoring risk and do not help
                    teams act under real staffing limits.
                </p>
            </div>
            <div class="info-card">
                <h4>Solution</h4>
                <p>
                    This app turns risk into an intervention queue, estimated
                    savings, and workflow-ready prioritization.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_demo_kpis(reference_cohort: pd.DataFrame) -> None:
    scored_count = int(len(reference_cohort)) if not reference_cohort.empty else 0
    positives = (
        int(reference_cohort["target"].sum())
        if not reference_cohort.empty and "target" in reference_cohort.columns
        else 0
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scored patients", f"{scored_count:,}")
    k2.metric("Observed readmissions", f"{positives:,}")
    k3.metric("Demo mode", "Pilot-ready")
    k4.metric("Workflow focus", "Capacity-aware")


def safe_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
THRESHOLD_CACHE_PATH = "outputs/derived_thresholds.pkl"
DEFAULT_CAPACITY = 200
USE_CAPACITY_OVERRIDE = True

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


@st.cache_resource
def load_pipeline():
    if MODEL_PATH.exists() and PREPROC_PATH.exists():
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROC_PATH)
        return model, preprocessor

    return fit_fallback_pipeline()


@st.cache_data
def load_model_metadata(path: str, use_saved_artifacts: bool) -> dict[str, str]:
    metadata = {
        "model_name": "clinical-readmission-prioritizer",
        "version": "v1.0.5",
        "artifact_source": (
            "saved-artifacts" if use_saved_artifacts else "fallback-demo"
        ),
    }

    metadata_path = Path(path)
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({key: str(value) for key, value in loaded.items()})
        except Exception:
            pass

    return metadata


@st.cache_data
def load_reference_cohort(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["risk_score", "target"])

    candidate_cols = {"encounter_id", "patient_nbr", "risk_score", "target"}
    cohort = pd.read_csv(path, usecols=lambda c: c in candidate_cols, low_memory=False)
    cohort["risk_score"] = pd.to_numeric(cohort.get("risk_score"), errors="coerce")
    if "target" in cohort.columns:
        target_series = pd.Series(
            pd.to_numeric(cohort["target"], errors="coerce"), index=cohort.index
        )
        cohort["target"] = target_series.fillna(0)
    else:
        cohort["target"] = 0.0
    return cohort.dropna(subset=["risk_score"]).reset_index(drop=True)


def get_percentile_thresholds(
    reference_scores: np.ndarray, medium_pct: float = 70.0, high_pct: float = 90.0
):
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


def assign_tiers_to_cohort(
    cohort: pd.DataFrame, medium_cut: float, high_cut: float
) -> pd.DataFrame:
    tiered = cohort.copy()
    tiered["tier"] = np.where(
        tiered["risk_score"] >= high_cut,
        "High",
        np.where(tiered["risk_score"] >= medium_cut, "Medium", "Low"),
    )
    return tiered


def apply_missing_input_defaults(row: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_INPUT_COLUMNS:
        if col in row.columns:
            continue
        if col == "age":
            row[col] = "[70-80]"
        elif col == "gender":
            row[col] = "Unknown"
        elif col == "race":
            row[col] = "Unknown"
        elif col == "weight":
            row[col] = "Unknown"
        elif col in CATEGORICAL_NONE_DEFAULTS:
            row[col] = "None"
        else:
            row[col] = 0
    return row


using_saved_artifacts = MODEL_PATH.exists() and PREPROC_PATH.exists()
model_metadata = load_model_metadata(str(MODEL_METADATA_PATH), using_saved_artifacts)

reference_cohort = load_reference_cohort(REFERENCE_SCORES_PATH)
reference_scores = (
    reference_cohort["risk_score"].to_numpy(dtype=float)
    if not reference_cohort.empty
    else np.array([], dtype=float)
)
inject_custom_styles()
render_sidebar(model_metadata)
render_hero(reference_cohort, reference_scores)
st.success("Quick demo: choose a preset and click `Predict risk` below.")
with st.expander("About this solution", expanded=False):
    render_startup_highlights()
    render_landing_section()
    render_demo_kpis(reference_cohort)
if reference_scores.size == 0:
    st.warning(
        "Reference risk cohort not found. Falling back to default percentile cutoffs."
    )

if not using_saved_artifacts:
    st.warning(
        "Saved model artifacts were not found, so the app is using a lightweight "
        "fallback demo model built from the included dataset."
    )

if "model" not in st.session_state or "preprocessor" not in st.session_state:
    with st.spinner("Loading model artifacts..."):
        cached_model, cached_preprocessor = load_pipeline()
        st.session_state["model"] = cached_model
        st.session_state["preprocessor"] = cached_preprocessor


model = st.session_state["model"]
preprocessor = st.session_state["preprocessor"]
if not st.session_state.get("_predictor_warmed", False):
    try:
        warm_row = pd.DataFrame([{col: 0 for col in REQUIRED_INPUT_COLUMNS}])
        warm_row = apply_missing_input_defaults(warm_row)
        warm_xt = transform_with_feature_names(warm_row, preprocessor)
        _ = model.predict_proba(warm_xt)
    except Exception:
        pass
    st.session_state["_predictor_warmed"] = True

screenshot_mode = get_screenshot_mode()
docs_explainability_only = screenshot_mode == "explainability"


# SHAP explainer cache
@st.cache_resource
def load_explainer(_model):
    model_name = getattr(getattr(_model, "__class__", None), "__name__", "")
    try:
        if model_name in {"XGBClassifier", "LGBMClassifier", "CatBoostClassifier"}:
            return shap.TreeExplainer(_model)
        return shap.Explainer(_model)
    except Exception:
        return None


explainer = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

default_inputs = {
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

patient_presets = {
    "Balanced demo": {
        "description": "Typical medical patient with moderate complexity.",
        "values": default_inputs,
    },
    "High-risk frequent utilizer": {
        "description": "Multiple prior visits and higher expected readmission risk.",
        "values": {
            **default_inputs,
            "age": "[70-80)",
            "time_in_hospital": 9,
            "num_medications": 22,
            "number_diagnoses": 12,
            "number_outpatient": 3,
            "number_emergency": 2,
            "number_inpatient": 3,
            "prior_inpatient": 4,
            "prior_outpatient": 3,
            "prior_emergency": 2,
            "prior_positive_count": 2,
            "diag_delta": 3,
            "med_delta": 6,
        },
    },
    "Lower-risk stable discharge": {
        "description": (
            "Simpler case with limited prior utilization " "and fewer risk signals."
        ),
        "values": {
            **default_inputs,
            "age": "[40-50)",
            "time_in_hospital": 2,
            "num_medications": 6,
            "number_diagnoses": 4,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "prior_inpatient": 0,
            "prior_outpatient": 0,
            "prior_emergency": 0,
            "prior_positive_count": 0,
            "diag_delta": -1,
            "med_delta": -2,
        },
    },
    "Chronic care follow-up": {
        "description": (
            "Moderate-to-high complexity patient needing "
            "proactive follow-up planning."
        ),
        "values": {
            **default_inputs,
            "age": "[60-70)",
            "time_in_hospital": 6,
            "num_lab_procedures": 55,
            "num_medications": 18,
            "number_diagnoses": 10,
            "number_outpatient": 2,
            "number_emergency": 1,
            "number_inpatient": 2,
            "prior_inpatient": 2,
            "prior_outpatient": 3,
            "prior_emergency": 1,
            "prior_positive_count": 1,
            "diag_delta": 2,
            "med_delta": 4,
        },
    },
}

if docs_explainability_only:
    st.title("Prediction Explainability")
    age = default_inputs["age"]
    gender = default_inputs["gender"]
    race = default_inputs["race"]
    time_in_hospital = default_inputs["time_in_hospital"]
    num_lab_procedures = default_inputs["num_lab_procedures"]
    num_procedures = default_inputs["num_procedures"]
    num_medications = default_inputs["num_medications"]
    number_diagnoses = default_inputs["number_diagnoses"]
    number_outpatient = default_inputs["number_outpatient"]
    number_emergency = default_inputs["number_emergency"]
    number_inpatient = default_inputs["number_inpatient"]
    admission_type_id = default_inputs["admission_type_id"]
    discharge_disposition_id = default_inputs["discharge_disposition_id"]
    admission_source_id = default_inputs["admission_source_id"]
    prior_inpatient = default_inputs["prior_inpatient"]
    prior_outpatient = default_inputs["prior_outpatient"]
    prior_emergency = default_inputs["prior_emergency"]
    prior_positive_count = default_inputs["prior_positive_count"]
    diag_delta = default_inputs["diag_delta"]
    med_delta = default_inputs["med_delta"]
    show_explainability = True
    submitted = True
else:
    st.markdown("## Patient Risk Simulator")
    st.caption(
        "Choose a demo preset, optionally adjust the values, then click `Predict risk`."
    )
    with st.container(border=True):
        preset_options = list(patient_presets.keys())
        preset_name = st.selectbox(
            "Demo patient preset",
            preset_options,
            help="Use a preset to make demos faster and more realistic.",
        )
        preset_values = patient_presets[preset_name]["values"]
        st.caption(patient_presets[preset_name]["description"])

        s1, s2, s3 = st.columns(3)
        s1.metric("Hospital days", int(preset_values["time_in_hospital"]))
        s2.metric("Medications", int(preset_values["num_medications"]))
        s3.metric("Prior readmissions", int(preset_values["prior_positive_count"]))

        age_options = [
            "[30-40)",
            "[40-50)",
            "[50-60)",
            "[60-70)",
            "[70-80)",
            "[80-90)",
        ]
        gender_options = ["Female", "Male"]
        race_options = ["Caucasian", "Hispanic", "Other"]

        with st.form("prediction_form"):
            st.markdown("### Patient profile")
            p1, p2, p3 = st.columns(3)
            with p1:
                age = st.selectbox(
                    "Age band",
                    age_options,
                    index=safe_index(age_options, str(preset_values["age"])),
                )
            with p2:
                gender = st.selectbox(
                    "Gender",
                    gender_options,
                    index=safe_index(gender_options, str(preset_values["gender"])),
                )
            with p3:
                race = st.selectbox(
                    "Race",
                    race_options,
                    index=safe_index(race_options, str(preset_values["race"])),
                )

            st.markdown("### Encounter and utilization")
            c1, c2 = st.columns(2)
            with c1:
                time_in_hospital = st.slider(
                    "Time in hospital", 1, 14, int(preset_values["time_in_hospital"])
                )
                num_lab_procedures = st.slider(
                    "Lab procedures", 1, 100, int(preset_values["num_lab_procedures"])
                )
                num_procedures = st.slider(
                    "Procedures", 0, 6, int(preset_values["num_procedures"])
                )
                num_medications = st.slider(
                    "Medications", 1, 40, int(preset_values["num_medications"])
                )
                number_diagnoses = st.slider(
                    "Diagnoses", 1, 16, int(preset_values["number_diagnoses"])
                )

            with c2:
                number_outpatient = st.slider(
                    "Outpatient visits", 0, 20, int(preset_values["number_outpatient"])
                )
                number_emergency = st.slider(
                    "Emergency visits", 0, 10, int(preset_values["number_emergency"])
                )
                number_inpatient = st.slider(
                    "Inpatient visits", 0, 10, int(preset_values["number_inpatient"])
                )
                admission_type_id = st.slider(
                    "Admission type ID", 1, 8, int(preset_values["admission_type_id"])
                )
                discharge_disposition_id = st.slider(
                    "Discharge disposition ID",
                    1,
                    30,
                    int(preset_values["discharge_disposition_id"]),
                )
                admission_source_id = st.slider(
                    "Admission source ID",
                    1,
                    25,
                    int(preset_values["admission_source_id"]),
                )

            st.markdown("### Prior history")
            h1, h2, h3 = st.columns(3)
            with h1:
                prior_inpatient = st.slider(
                    "Prior inpatient total",
                    0,
                    20,
                    int(preset_values["prior_inpatient"]),
                )
                prior_outpatient = st.slider(
                    "Prior outpatient total",
                    0,
                    20,
                    int(preset_values["prior_outpatient"]),
                )
            with h2:
                prior_emergency = st.slider(
                    "Prior emergency total",
                    0,
                    20,
                    int(preset_values["prior_emergency"]),
                )
                prior_positive_count = st.slider(
                    "Prior readmissions",
                    0,
                    10,
                    int(preset_values["prior_positive_count"]),
                )
            with h3:
                diag_delta = st.slider(
                    "Diagnosis change", -10, 10, int(preset_values["diag_delta"])
                )
                med_delta = st.slider(
                    "Medication change", -20, 20, int(preset_values["med_delta"])
                )

            show_explainability = st.checkbox(
                "Include explainability details (slower)",
                value=False,
                help="Enable SHAP feature contribution charts for deeper review.",
            )
            submitted = st.form_submit_button(
                "Predict risk", type="primary", use_container_width=True
            )

if submitted:
    prior_encounters = prior_inpatient + prior_outpatient + prior_emergency
    encounter_number = prior_encounters + 1
    ever_prior_positive = 1 if prior_positive_count > 0 else 0

    with st.spinner("Scoring patient and preparing results..."):
        row_dict: dict[str, object] = {col: 0 for col in REQUIRED_INPUT_COLUMNS}
        row_dict.update(
            {
                "encounter_id": 999999,
                "patient_nbr": 9999,
                "race": race,
                "gender": gender,
                "age": age,
                "admission_type_id": admission_type_id,
                "discharge_disposition_id": discharge_disposition_id,
                "admission_source_id": admission_source_id,
                "time_in_hospital": time_in_hospital,
                "num_lab_procedures": num_lab_procedures,
                "num_procedures": num_procedures,
                "num_medications": num_medications,
                "number_outpatient": number_outpatient,
                "number_emergency": number_emergency,
                "number_inpatient": number_inpatient,
                "number_diagnoses": number_diagnoses,
                "encounter_number": encounter_number,
                "prior_encounters": prior_encounters,
                "prior_number_inpatient_sum": prior_inpatient,
                "prior_number_inpatient_mean": float(prior_inpatient),
                "prior_number_outpatient_sum": prior_outpatient,
                "prior_number_outpatient_mean": float(prior_outpatient),
                "prior_number_emergency_sum": prior_emergency,
                "prior_number_emergency_mean": float(prior_emergency),
                "prior_total_visits": prior_encounters,
                "diag_delta": diag_delta,
                "med_delta": med_delta,
                "prior_positive_count": prior_positive_count,
                "ever_prior_positive": ever_prior_positive,
            }
        )
        row = pd.DataFrame([row_dict])
        row = apply_missing_input_defaults(row)

        t0 = time.time()
        Xt = transform_with_feature_names(row, preprocessor)
        t1 = time.time()
        risk = float(model.predict_proba(Xt)[:, 1][0])
        t2 = time.time()

        medium_cut, high_cut = get_percentile_thresholds(reference_scores)
        tier = assign_percentile_tier(risk, medium_cut, high_cut)
        risk_percentile = get_risk_percentile(risk, reference_scores)

    st.session_state["last_prediction"] = {
        "risk": risk,
        "tier": tier,
        "risk_percentile": risk_percentile,
        "medium_cut": medium_cut,
        "high_cut": high_cut,
        "show_explainability": show_explainability,
        "row_dict": row_dict,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_version": model_metadata.get("version", "unversioned"),
        "artifact_source": model_metadata.get("artifact_source", "unknown"),
        "timing": {
            "preprocessor": t1 - t0,
            "model": t2 - t1,
            "total": t2 - t0,
        },
    }

prediction_result = st.session_state.get("last_prediction")

if prediction_result is None and not docs_explainability_only:
    st.info("Prediction results will appear here after you click `Predict risk`.")

if prediction_result is not None:
    risk = float(prediction_result["risk"])
    tier = str(prediction_result["tier"])
    risk_percentile = float(prediction_result["risk_percentile"])
    medium_cut = float(prediction_result["medium_cut"])
    high_cut = float(prediction_result["high_cut"])
    show_explainability = bool(prediction_result["show_explainability"])
    row = pd.DataFrame([prediction_result["row_dict"]])
    Xt = transform_with_feature_names(row, preprocessor)
    timing = prediction_result["timing"]

    st.caption(
        "Preprocessor: "
        f"{timing['preprocessor']:.3f}s, "
        f"Model: {timing['model']:.3f}s, "
        f"Total: {timing['total']:.3f}s"
    )

    render_prediction_banner(tier, risk, risk_percentile)

    st.subheader("Prediction Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated risk", format_percent(risk))
    m2.metric("Priority tier", tier)
    m3.metric("Relative percentile", f"{risk_percentile:.1f}%")
    m4.metric("High-priority cutoff", format_percent(high_cut))

    interpretation_md = (
        "**How to read this result**\n"
        f"- **{format_percent(risk)} estimated risk** means the model predicts "
        "roughly this chance of readmission.\n"
        f"- **{risk_percentile:.1f}th percentile** means this patient scores "
        "higher than most patients in the reference cohort.\n"
        f"- **{tier} priority** is an operational queue label based on cohort "
        f"thresholds (`Medium ≥ {format_percent(medium_cut)}`, "
        f"`High ≥ {format_percent(high_cut)}`)."
    )
    st.markdown(interpretation_md)

    action_map = {
        "High": "Recommended action: same-day outreach and care coordination review.",
        "Medium": "Recommended action: include in the next outreach planning cycle.",
        "Low": "Recommended action: routine follow-up is likely sufficient.",
    }
    st.info(action_map[tier])
    with st.expander("Prediction metadata", expanded=False):
        st.json(
            {
                "generated_at": prediction_result.get("generated_at", "unknown"),
                "model_version": prediction_result.get("model_version", "unversioned"),
                "artifact_source": prediction_result.get("artifact_source", "unknown"),
                "reference_mode": (
                    "saved trained artifacts"
                    if using_saved_artifacts
                    else "fallback demo model"
                ),
            }
        )
    st.caption(
        "This is a prioritization aid for demos and workflow planning, not a diagnosis."
    )

    # Explainability can be slower, so keep it optional for faster demos.
    if show_explainability:
        explainer = load_explainer(model)
        if not docs_explainability_only:
            st.subheader("Prediction Explainability")
        if explainer is None:
            st.info(
                "SHAP explainer is not available for the current model in this session."
            )
        else:
            try:
                t_shap0 = time.time()
                if hasattr(Xt, "sparse"):
                    Xt_dense = Xt.sparse.to_dense()
                else:
                    to_array = getattr(Xt, "toarray", None)
                    Xt_dense = to_array() if callable(to_array) else Xt
                dense_frame = (
                    Xt_dense
                    if isinstance(Xt_dense, pd.DataFrame)
                    else pd.DataFrame(Xt_dense)
                )
                shap_values = explainer(dense_frame)
                t_shap1 = time.time()

                feature_names = [
                    format_explainability_feature_name(str(col))
                    for col in dense_frame.columns
                ]
                if hasattr(shap_values, "feature_names"):
                    shap_values.feature_names = feature_names
                raw_values = np.asarray(getattr(shap_values, "values", shap_values))
                if raw_values.ndim == 3:
                    raw_values = raw_values[:, :, -1]
                row_values = raw_values[0] if raw_values.ndim > 1 else raw_values

                contribution_df = (
                    pd.DataFrame(
                        {
                            "feature": feature_names[: len(row_values)],
                            "impact": row_values,
                        }
                    )
                    .assign(abs_impact=lambda d: d["impact"].abs())
                    .sort_values("abs_impact", ascending=False)
                    .head(10)
                    .set_index("feature")
                )

                st.write("**Top feature contributions**")
                st.bar_chart(contribution_df["impact"])

                plt.close("all")
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
                if not docs_explainability_only:
                    st.caption(f"SHAP explanation time: {t_shap1 - t_shap0:.3f}s")
            except Exception as exc:
                st.info(f"Explainability preview is temporarily unavailable: {exc}")
    elif not docs_explainability_only:
        st.caption(
            "Tip: enable `Include explainability details` "
            "if you want SHAP-based feature insights."
        )

    if docs_explainability_only:
        st.stop()

    # Hospital-grade triage planning powered by the scored cohort.
    st.subheader("Capacity-Aware Triage Dashboard")
    st.caption(
        "Plan queue size, expected capture, and ROI using the scored "
        "reference population."
    )
    with st.expander("Queue and ROI Parameters", expanded=True):
        max_capacity = int(max(len(reference_cohort), 1))
        default_capacity = int(min(DEFAULT_CAPACITY, max_capacity))
        capacity = st.number_input(
            "Capacity (patients)",
            min_value=1,
            max_value=max_capacity,
            value=default_capacity,
        )
        intervention_cost = st.number_input(
            "Intervention cost per patient ($)",
            min_value=0,
            value=150,
        )
        readmission_cost = st.number_input(
            "Readmission cost per event ($)", min_value=0, value=12000
        )
        intervention_effectiveness = st.slider(
            "Intervention effectiveness (fraction)",
            min_value=0.0,
            max_value=1.0,
            value=0.18,
            step=0.01,
        )

    if not reference_cohort.empty:
        tiered_cohort = assign_tiers_to_cohort(reference_cohort, medium_cut, high_cut)
        tier_counts = (
            tiered_cohort["tier"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
        )
        total_patients = int(len(tiered_cohort))

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "High Tier",
            f"{tier_counts['High']:,}",
            f"{(tier_counts['High']/total_patients)*100:.1f}%",
        )
        c2.metric(
            "Medium Tier",
            f"{tier_counts['Medium']:,}",
            f"{(tier_counts['Medium']/total_patients)*100:.1f}%",
        )
        c3.metric(
            "Low Tier",
            f"{tier_counts['Low']:,}",
            f"{(tier_counts['Low']/total_patients)*100:.1f}%",
        )

        ranked = tiered_cohort.sort_values("risk_score", ascending=False).reset_index(
            drop=True
        )
        queue = ranked.head(int(capacity))
        total_readmissions = float(ranked["target"].sum())
        queue_readmissions = float(queue["target"].sum())
        capture_rate = (
            0.0
            if total_readmissions == 0
            else 100.0 * queue_readmissions / total_readmissions
        )

        rank_desc = int(np.sum(reference_scores > risk)) + 1
        in_queue = rank_desc <= int(capacity)

        q1, q2, q3 = st.columns(3)
        q1.metric("Queue Capture", f"{capture_rate:.1f}%")
        q2.metric("Patient Rank", f"{rank_desc:,} / {total_patients:,}")
        q3.metric("In Current Queue", "Yes" if in_queue else "No")

        roi = hospital_roi(
            ranked[["risk_score", "target"]],
            capacity=int(capacity),
            intervention_cost=intervention_cost,
            readmission_cost=readmission_cost,
            intervention_effectiveness=intervention_effectiveness,
        )
        st.write("**ROI Results**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Patients targeted", f"{roi['patients_targeted']:,}")
        r2.metric("Prevented readmissions", f"{roi['readmissions_prevented']:.1f}")
        r3.metric("Program cost", format_currency(roi["cost"]))
        r4.metric("Net ROI", format_currency(roi["net_roi"]))

        with st.expander("Detailed economics"):
            st.json(roi)

        display_cols = [
            c
            for c in ["encounter_id", "patient_nbr", "risk_score", "tier", "target"]
            if c in queue.columns
        ]
        st.write("**Top Queue Preview**")
        queue_export = queue[display_cols].copy()
        st.dataframe(queue_export.head(20), width="stretch")

        csv_bytes = queue_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download current queue (CSV)",
            data=csv_bytes,
            file_name=f"triage_queue_top_{int(capacity)}.csv",
            mime="text/csv",
        )
