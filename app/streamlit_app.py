import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

st.set_page_config(
    page_title="AI Care Prioritization Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.inference import (  # noqa: E402
    DEFAULT_DEMO_INPUTS,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PREPROC_PATH,
    REFERENCE_SCORES_PATH,
    REQUIRED_INPUT_COLUMNS,
    apply_missing_input_defaults,
    load_model_metadata,
    load_pipeline,
    load_reference_cohort,
    score_batch_payloads,
    score_patient_payload,
)
from src.preprocess import transform_with_feature_names  # noqa: E402
from src.workflow_simulation import hospital_roi  # noqa: E402

LOGGER = logging.getLogger(__name__)


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


def render_sidebar(
    model_metadata: dict[str, str], api_health: Optional[dict[str, str]]
) -> None:
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
        st.markdown("### Inference layer")
        if api_health:
            st.caption("Mode: FastAPI service")
            st.caption(f"API status: {api_health.get('status', 'ok')}")
        else:
            st.caption("Mode: Local in-app scoring")
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
DEFAULT_CAPACITY = 200
API_BASE_URL = (
    os.environ.get("CARE_API_URL", "http://127.0.0.1:8000").strip().rstrip("/")
)


@st.cache_data(ttl=30, show_spinner=False)
def get_api_health(base_url: str) -> Optional[dict[str, str]]:
    if not base_url:
        return None

    try:
        response = requests.get(f"{base_url}/health", timeout=0.4)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    if payload.get("status") != "ok":
        return None

    return {key: str(value) for key, value in payload.items()}


def score_patient_with_optional_api(payload: dict[str, Any]) -> dict[str, Any]:
    api_health = get_api_health(API_BASE_URL)
    if api_health:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=1.5,
            )
            response.raise_for_status()
            result = response.json()
            result["inference_mode"] = "api"
            result["api_base_url"] = API_BASE_URL
            return result
        except (requests.RequestException, ValueError):
            LOGGER.debug(
                "Falling back to local single-patient inference.", exc_info=True
            )

    result = score_patient_payload(payload)
    result["inference_mode"] = "local"
    result["api_base_url"] = None
    return result


def score_batch_with_optional_api(
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    api_health = get_api_health(API_BASE_URL)
    if api_health:
        try:
            response = requests.post(
                f"{API_BASE_URL}/batch_predict",
                json={"patients": payloads},
                timeout=3.0,
            )
            response.raise_for_status()
            result = response.json()
            result["inference_mode"] = "api"
            result["api_base_url"] = API_BASE_URL
            return result
        except (requests.RequestException, ValueError):
            LOGGER.debug("Falling back to local batch inference.", exc_info=True)

    result = score_batch_payloads(payloads)
    result["inference_mode"] = "local"
    result["api_base_url"] = None
    return result


def build_batch_template_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": "[70-80)",
                "gender": "Female",
                "race": "Caucasian",
                "time_in_hospital": 8,
                "num_lab_procedures": 60,
                "num_medications": 20,
                "number_diagnoses": 11,
                "number_outpatient": 2,
                "number_emergency": 1,
                "number_inpatient": 2,
                "prior_inpatient": 3,
                "prior_outpatient": 2,
                "prior_emergency": 1,
                "prior_positive_count": 1,
                "diag_delta": 2,
                "med_delta": 4,
            },
            {
                "age": "[40-50)",
                "gender": "Male",
                "race": "Other",
                "time_in_hospital": 2,
                "num_lab_procedures": 20,
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
        ]
    )


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


artifacts_present = MODEL_PATH.exists() and PREPROC_PATH.exists()

if (
    "model" not in st.session_state
    or "preprocessor" not in st.session_state
    or "pipeline_source" not in st.session_state
):
    with st.spinner("Loading model artifacts..."):
        cached_model, cached_preprocessor, pipeline_source = load_pipeline()
        st.session_state["model"] = cached_model
        st.session_state["preprocessor"] = cached_preprocessor
        st.session_state["pipeline_source"] = pipeline_source

using_saved_artifacts = st.session_state.get("pipeline_source") == "saved-artifacts"
model_metadata = load_model_metadata(str(MODEL_METADATA_PATH), using_saved_artifacts)
api_health = get_api_health(API_BASE_URL)

reference_cohort = load_reference_cohort(str(REFERENCE_SCORES_PATH))
reference_scores = (
    reference_cohort["risk_score"].to_numpy(dtype=float)
    if not reference_cohort.empty
    else np.array([], dtype=float)
)
inject_custom_styles()
render_sidebar(model_metadata, api_health)
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

if not artifacts_present:
    st.warning(
        "Saved model artifacts were not found, so the app is using a lightweight "
        "fallback demo model built from the included dataset."
    )
elif not using_saved_artifacts:
    st.warning(
        "Saved artifacts were present but could not be loaded in this environment, "
        "so the app switched to the fallback demo model."
    )

model = st.session_state["model"]
preprocessor = st.session_state["preprocessor"]
if not st.session_state.get("_predictor_warmed", False):
    try:
        warm_row = pd.DataFrame([{col: 0 for col in REQUIRED_INPUT_COLUMNS}])
        warm_row = apply_missing_input_defaults(warm_row)
        warm_xt = transform_with_feature_names(warm_row, preprocessor)
        _ = model.predict_proba(warm_xt)
    except Exception:
        LOGGER.debug(
            "Predictor warm-up failed; continuing without warm cache.", exc_info=True
        )
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
        LOGGER.debug(
            "Could not initialize SHAP explainer for current model.", exc_info=True
        )
        return None


explainer = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_batch_result" not in st.session_state:
    st.session_state["last_batch_result"] = None

default_inputs = DEFAULT_DEMO_INPUTS.copy()

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
                    key=f"age_{preset_name}",
                )
            with p2:
                gender = st.selectbox(
                    "Gender",
                    gender_options,
                    index=safe_index(gender_options, str(preset_values["gender"])),
                    key=f"gender_{preset_name}",
                )
            with p3:
                race = st.selectbox(
                    "Race",
                    race_options,
                    index=safe_index(race_options, str(preset_values["race"])),
                    key=f"race_{preset_name}",
                )

            st.markdown("### Encounter and utilization")
            c1, c2 = st.columns(2)
            with c1:
                time_in_hospital = st.slider(
                    "Time in hospital",
                    1,
                    14,
                    int(preset_values["time_in_hospital"]),
                    key=f"time_in_hospital_{preset_name}",
                )
                num_lab_procedures = st.slider(
                    "Lab procedures",
                    1,
                    100,
                    int(preset_values["num_lab_procedures"]),
                    key=f"num_lab_procedures_{preset_name}",
                )
                num_procedures = st.slider(
                    "Procedures",
                    0,
                    6,
                    int(preset_values["num_procedures"]),
                    key=f"num_procedures_{preset_name}",
                )
                num_medications = st.slider(
                    "Medications",
                    1,
                    40,
                    int(preset_values["num_medications"]),
                    key=f"num_medications_{preset_name}",
                )
                number_diagnoses = st.slider(
                    "Diagnoses",
                    1,
                    16,
                    int(preset_values["number_diagnoses"]),
                    key=f"number_diagnoses_{preset_name}",
                )

            with c2:
                number_outpatient = st.slider(
                    "Outpatient visits",
                    0,
                    20,
                    int(preset_values["number_outpatient"]),
                    key=f"number_outpatient_{preset_name}",
                )
                number_emergency = st.slider(
                    "Emergency visits",
                    0,
                    10,
                    int(preset_values["number_emergency"]),
                    key=f"number_emergency_{preset_name}",
                )
                number_inpatient = st.slider(
                    "Inpatient visits",
                    0,
                    10,
                    int(preset_values["number_inpatient"]),
                    key=f"number_inpatient_{preset_name}",
                )
                admission_type_id = st.slider(
                    "Admission type ID",
                    1,
                    8,
                    int(preset_values["admission_type_id"]),
                    key=f"admission_type_id_{preset_name}",
                )
                discharge_disposition_id = st.slider(
                    "Discharge disposition ID",
                    1,
                    30,
                    int(preset_values["discharge_disposition_id"]),
                    key=f"discharge_disposition_id_{preset_name}",
                )
                admission_source_id = st.slider(
                    "Admission source ID",
                    1,
                    25,
                    int(preset_values["admission_source_id"]),
                    key=f"admission_source_id_{preset_name}",
                )

            st.markdown("### Prior history")
            h1, h2, h3 = st.columns(3)
            with h1:
                prior_inpatient = st.slider(
                    "Prior inpatient total",
                    0,
                    20,
                    int(preset_values["prior_inpatient"]),
                    key=f"prior_inpatient_{preset_name}",
                )
                prior_outpatient = st.slider(
                    "Prior outpatient total",
                    0,
                    20,
                    int(preset_values["prior_outpatient"]),
                    key=f"prior_outpatient_{preset_name}",
                )
            with h2:
                prior_emergency = st.slider(
                    "Prior emergency total",
                    0,
                    20,
                    int(preset_values["prior_emergency"]),
                    key=f"prior_emergency_{preset_name}",
                )
                prior_positive_count = st.slider(
                    "Prior readmissions",
                    0,
                    10,
                    int(preset_values["prior_positive_count"]),
                    key=f"prior_positive_count_{preset_name}",
                )
            with h3:
                diag_delta = st.slider(
                    "Diagnosis change",
                    -10,
                    10,
                    int(preset_values["diag_delta"]),
                    key=f"diag_delta_{preset_name}",
                )
                med_delta = st.slider(
                    "Medication change",
                    -20,
                    20,
                    int(preset_values["med_delta"]),
                    key=f"med_delta_{preset_name}",
                )

            show_explainability = st.checkbox(
                "Include explainability details (slower)",
                value=False,
                help="Enable SHAP feature contribution charts for deeper review.",
                key=f"show_explainability_{preset_name}",
            )
            submitted = st.form_submit_button(
                "Predict risk", type="primary", use_container_width=True
            )

if submitted:
    prediction_payload: dict[str, Any] = {
        "age": age,
        "gender": gender,
        "race": race,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "prior_inpatient": prior_inpatient,
        "prior_outpatient": prior_outpatient,
        "prior_emergency": prior_emergency,
        "prior_positive_count": prior_positive_count,
        "diag_delta": diag_delta,
        "med_delta": med_delta,
    }

    try:
        with st.spinner("Scoring patient and preparing results..."):
            result = score_patient_with_optional_api(prediction_payload)

        st.session_state["last_prediction"] = {
            "risk": float(result["risk"]),
            "tier": str(result["tier"]),
            "risk_percentile": float(result["risk_percentile"]),
            "medium_cut": float(result["medium_cut"]),
            "high_cut": float(result["high_cut"]),
            "show_explainability": show_explainability,
            "row_dict": dict(result["row_dict"]),
            "generated_at": str(result.get("generated_at", "unknown")),
            "model_version": str(result.get("model_version", "unversioned")),
            "artifact_source": str(result.get("artifact_source", "unknown")),
            "inference_mode": str(result.get("inference_mode", "local")),
            "api_base_url": result.get("api_base_url"),
            "timing": dict(result.get("timing", {})),
        }
    except Exception as exc:
        st.error(
            f"Failed to score patient: {str(exc)}. "
            "Please check your inputs and try again."
        )
        LOGGER.error(f"Prediction failed with error: {exc}", exc_info=True)
        st.session_state["last_prediction"] = None

prediction_result = st.session_state.get("last_prediction")

if not docs_explainability_only:
    st.markdown("---")
    st.markdown("## Batch Queue Scoring")
    st.caption(
        "Upload a CSV of patients to score an outreach queue, "
        "review tier mix, and export the prioritized results."
    )

    batch_template = build_batch_template_dataframe()
    st.download_button(
        label="Download batch template (CSV)",
        data=batch_template.to_csv(index=False).encode("utf-8"),
        file_name="sample_batch_patients.csv",
        mime="text/csv",
    )
    st.caption(
        "You can upload a small subset of columns like age, gender, race, "
        "hospital days, meds, diagnoses, and prior utilization. Missing values "
        "fall back to the demo defaults."
    )

    uploaded_batch_file = st.file_uploader(
        "Upload patient CSV for batch scoring",
        type=["csv"],
        key="batch_queue_upload",
    )
    if uploaded_batch_file is not None:
        try:
            uploaded_batch_df = pd.read_csv(uploaded_batch_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
        else:
            if uploaded_batch_df.empty:
                st.warning(
                    "The uploaded CSV is empty. Please provide at least one row."
                )
            else:
                st.write("**Uploaded batch preview**")
                st.dataframe(uploaded_batch_df.head(10), width="stretch")
                if st.button(
                    "Score uploaded queue",
                    type="primary",
                    use_container_width=True,
                ):
                    payloads = uploaded_batch_df.where(
                        pd.notnull(uploaded_batch_df), None
                    ).to_dict(orient="records")
                    try:
                        with st.spinner("Scoring uploaded patient batch..."):
                            st.session_state["last_batch_result"] = (
                                score_batch_with_optional_api(payloads)
                            )
                    except Exception as exc:
                        st.error(
                            f"Failed to score batch: {str(exc)}. "
                            "Please check your CSV and try again."
                        )
                        LOGGER.error(
                            f"Batch prediction failed with error: {exc}", exc_info=True
                        )
                        st.session_state["last_batch_result"] = None

    batch_result = st.session_state.get("last_batch_result")
    if batch_result:
        batch_predictions = batch_result.get("predictions", [])
        if batch_predictions:
            scored_batch_df = pd.DataFrame(
                [
                    {
                        "encounter_id": item["row_dict"].get("encounter_id"),
                        "patient_nbr": item["row_dict"].get("patient_nbr"),
                        "age": item["row_dict"].get("age"),
                        "gender": item["row_dict"].get("gender"),
                        "race": item["row_dict"].get("race"),
                        "time_in_hospital": item["row_dict"].get("time_in_hospital"),
                        "num_medications": item["row_dict"].get("num_medications"),
                        "number_diagnoses": item["row_dict"].get("number_diagnoses"),
                        "risk_score": float(item["risk"]),
                        "risk_percentile": float(item["risk_percentile"]),
                        "priority_tier": item["tier"],
                    }
                    for item in batch_predictions
                ]
            ).sort_values("risk_score", ascending=False, ignore_index=True)

            tier_counts = batch_result.get("tier_counts", {})
            batch_timing = batch_result.get("timing", {})
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Batch patients", int(batch_result.get("count", 0)))
            b2.metric("High priority", int(tier_counts.get("High", 0)))
            b3.metric("Medium priority", int(tier_counts.get("Medium", 0)))
            b4.metric(
                "Average risk",
                format_percent(float(scored_batch_df["risk_score"].mean())),
            )

            if batch_timing:
                st.caption(
                    "Batch preprocessor: "
                    f"{float(batch_timing.get('preprocessor', 0.0)):.3f}s, "
                    f"Model: {float(batch_timing.get('model', 0.0)):.3f}s, "
                    f"Total: {float(batch_timing.get('total', 0.0)):.3f}s"
                )

            st.write("**Prioritized batch preview**")
            st.dataframe(scored_batch_df.head(25), width="stretch")

            st.download_button(
                label="Download scored batch (CSV)",
                data=scored_batch_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_batch_queue.csv",
                mime="text/csv",
            )

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
        metadata_payload = {
            "generated_at": prediction_result.get("generated_at", "unknown"),
            "model_version": prediction_result.get("model_version", "unversioned"),
            "artifact_source": prediction_result.get("artifact_source", "unknown"),
            "inference_mode": prediction_result.get("inference_mode", "local"),
            "reference_mode": (
                "saved trained artifacts"
                if using_saved_artifacts
                else "fallback demo model"
            ),
        }
        if prediction_result.get("api_base_url"):
            metadata_payload["api_base_url"] = prediction_result["api_base_url"]
        st.json(metadata_payload)
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
