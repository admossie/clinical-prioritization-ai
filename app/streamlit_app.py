
import sys
import time
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
PREPROC_PATH = ROOT / "models" / "preprocessor.joblib"
REFERENCE_SCORES_PATH = ROOT / "outputs" / "tables" / "test_scored.csv"
sys.path.append(str(ROOT))
from src.preprocess import transform_with_feature_names
from src.workflow_simulation import hospital_roi


def get_screenshot_mode() -> str:
    value = st.query_params.get("screenshot", "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip().lower()


def format_explainability_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__", "num_missing__", "cat_missing__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("_", " ")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
THRESHOLD_CACHE_PATH = "outputs/derived_thresholds.pkl"
DEFAULT_CAPACITY = 200
USE_CAPACITY_OVERRIDE = True

REQUIRED_INPUT_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight", "admission_type_id",
    "discharge_disposition_id", "admission_source_id", "time_in_hospital", "payer_code",
    "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
    "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone", "change",
    "diabetesMed", "encounter_number", "prior_encounters", "prior_number_inpatient_sum",
    "prior_number_inpatient_mean", "prior_number_outpatient_sum", "prior_number_outpatient_mean",
    "prior_number_emergency_sum", "prior_number_emergency_mean", "prior_total_visits",
    "diag_delta", "med_delta", "prior_positive_count", "ever_prior_positive",
]

CATEGORICAL_NONE_DEFAULTS = {
    "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed",
}

@st.cache_resource
def load_pipeline():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)
    return model, preprocessor


@st.cache_data
def load_reference_cohort(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["risk_score", "target"])

    candidate_cols = {"encounter_id", "patient_nbr", "risk_score", "target"}
    cohort = pd.read_csv(path, usecols=lambda c: c in candidate_cols, low_memory=False)
    cohort["risk_score"] = pd.to_numeric(cohort.get("risk_score"), errors="coerce")
    if "target" in cohort.columns:
        cohort["target"] = pd.to_numeric(cohort["target"], errors="coerce").fillna(0)
    else:
        cohort["target"] = 0.0
    return cohort.dropna(subset=["risk_score"]).reset_index(drop=True)


def get_percentile_thresholds(reference_scores: np.ndarray, medium_pct: float = 70.0, high_pct: float = 90.0):
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


def assign_tiers_to_cohort(cohort: pd.DataFrame, medium_cut: float, high_cut: float) -> pd.DataFrame:
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

if not MODEL_PATH.exists() or not PREPROC_PATH.exists():
    st.warning("Train the model first.")
    st.stop()

reference_cohort = load_reference_cohort(REFERENCE_SCORES_PATH)
reference_scores = reference_cohort["risk_score"].to_numpy(dtype=float) if not reference_cohort.empty else np.array([], dtype=float)
st.title("AI Care Prioritization Engine")
st.caption("Capacity-aware readmission prioritization demo")
if reference_scores.size == 0:
    st.warning("Reference risk cohort not found. Falling back to default percentile cutoffs.")

if "model" not in st.session_state or "preprocessor" not in st.session_state:
    with st.spinner("Loading model artifacts..."):
        cached_model, cached_preprocessor = load_pipeline()
        st.session_state["model"] = cached_model
        st.session_state["preprocessor"] = cached_preprocessor


model = st.session_state["model"]
preprocessor = st.session_state["preprocessor"]
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

explainer = load_explainer(model)

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
    submitted = True
else:
    with st.form("prediction_form"):
        # Only allow age bands, genders, and races present in the model's features
        age = st.selectbox(
            "Age band",
            ["[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"],
            index=2
        )
        gender = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox("Race", ["Caucasian", "Hispanic", "Other"])

        c1, c2 = st.columns(2)
        with c1:
            time_in_hospital = st.slider("Time in hospital", 1, 14, 4)
            num_lab_procedures = st.slider("Lab procedures", 1, 100, 40)
            num_procedures = st.slider("Procedures", 0, 6, 1)
            num_medications = st.slider("Medications", 1, 40, 12)
            number_diagnoses = st.slider("Diagnoses", 1, 16, 8)

        with c2:
            number_outpatient = st.slider("Outpatient visits", 0, 20, 1)
            number_emergency = st.slider("Emergency visits", 0, 10, 0)
            number_inpatient = st.slider("Inpatient visits", 0, 10, 1)
            admission_type_id = st.slider("Admission type ID", 1, 8, 1)
            discharge_disposition_id = st.slider("Discharge disposition ID", 1, 30, 1)
            admission_source_id = st.slider("Admission source ID", 1, 25, 7)

        prior_inpatient = st.slider("Prior inpatient total", 0, 20, 0)
        prior_outpatient = st.slider("Prior outpatient total", 0, 20, 0)
        prior_emergency = st.slider("Prior emergency total", 0, 20, 0)
        prior_positive_count = st.slider("Prior readmissions", 0, 10, 0)
        diag_delta = st.slider("Diagnosis change", -10, 10, 0)
        med_delta = st.slider("Medication change", -20, 20, 0)

        submitted = st.form_submit_button("Predict risk")

if submitted:
    prior_encounters = prior_inpatient + prior_outpatient + prior_emergency
    encounter_number = prior_encounters + 1
    ever_prior_positive = 1 if prior_positive_count > 0 else 0


    # Build the row with user inputs and sensible defaults
    row_dict = {col: 0 for col in REQUIRED_INPUT_COLUMNS}
    # Fill in user-provided values
    row_dict.update({
        'encounter_id': 999999,
        'patient_nbr': 9999,
        'race': race,
        'gender': gender,
        'age': age,
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'encounter_number': encounter_number,
        'prior_encounters': prior_encounters,
        'prior_number_inpatient_sum': prior_inpatient,
        'prior_number_inpatient_mean': float(prior_inpatient),
        'prior_number_outpatient_sum': prior_outpatient,
        'prior_number_outpatient_mean': float(prior_outpatient),
        'prior_number_emergency_sum': prior_emergency,
        'prior_number_emergency_mean': float(prior_emergency),
        'prior_total_visits': prior_encounters,
        'diag_delta': diag_delta,
        'med_delta': med_delta,
        'prior_positive_count': prior_positive_count,
        'ever_prior_positive': ever_prior_positive,
    })
    row = pd.DataFrame([row_dict])
    row = apply_missing_input_defaults(row)

    t0 = time.time()
    Xt = transform_with_feature_names(row, preprocessor)
    t1 = time.time()
    risk = float(model.predict_proba(Xt)[:, 1][0])
    t2 = time.time()

    st.caption(f"Preprocessor: {t1-t0:.3f}s, Model: {t2-t1:.3f}s, Total: {t2-t0:.3f}s")

    medium_cut, high_cut = get_percentile_thresholds(reference_scores)
    tier = assign_percentile_tier(risk, medium_cut, high_cut)
    risk_percentile = get_risk_percentile(risk, reference_scores)

    st.subheader("Prediction Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", f"{risk:.3f}")
    m2.metric("Risk tier", tier)
    m3.metric("Risk Percentile", f"{risk_percentile:.1f}%")
    st.write("Raw risk score:", risk)
    st.caption("Risk tiers are based on population percentiles, not fixed thresholds.")
    st.caption(f"Current cutoffs: Medium >= {medium_cut:.3f}, High >= {high_cut:.3f}")

    if tier == "High":
        st.error("High-priority patient for follow-up.")
    elif tier == "Medium":
        st.warning("Moderate-priority patient for review.")
    else:
        st.success("Low-priority patient based on current model.")

    st.caption("Demo prediction only. Final trustworthiness depends on training with the full public dataset.")

    # Model explanation shown automatically for screenshots and demos.
    if not docs_explainability_only:
        st.subheader("Prediction Explainability")
    if explainer is None:
        st.info("SHAP explainer is not available for the current model in this session.")
    else:
        try:
            t_shap0 = time.time()
            Xt_dense = Xt.sparse.to_dense() if hasattr(Xt, "sparse") else Xt.toarray() if hasattr(Xt, "toarray") else Xt
            shap_values = explainer(Xt_dense)
            t_shap1 = time.time()

            feature_names = (
                [format_explainability_feature_name(col) for col in Xt_dense.columns]
                if hasattr(Xt_dense, "columns")
                else [f"feature_{i}" for i in range(Xt_dense.shape[1])]
            )
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

    if docs_explainability_only:
        st.stop()

    # Hospital-grade triage planning powered by the scored cohort.
    st.subheader("Capacity-Aware Triage Dashboard")
    st.caption("Plan queue size, expected capture, and ROI using the scored reference population.")
    with st.expander("Queue and ROI Parameters", expanded=True):
        max_capacity = int(max(len(reference_cohort), 1))
        default_capacity = int(min(DEFAULT_CAPACITY, max_capacity))
        capacity = st.number_input("Capacity (patients)", min_value=1, max_value=max_capacity, value=default_capacity)
        intervention_cost = st.number_input("Intervention cost per patient ($)", min_value=0, value=150)
        readmission_cost = st.number_input("Readmission cost per event ($)", min_value=0, value=12000)
        intervention_effectiveness = st.slider("Intervention effectiveness (fraction)", min_value=0.0, max_value=1.0, value=0.18, step=0.01)

    if not reference_cohort.empty:
        tiered_cohort = assign_tiers_to_cohort(reference_cohort, medium_cut, high_cut)
        tier_counts = tiered_cohort["tier"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        total_patients = int(len(tiered_cohort))

        c1, c2, c3 = st.columns(3)
        c1.metric("High Tier", f"{tier_counts['High']:,}", f"{(tier_counts['High']/total_patients)*100:.1f}%")
        c2.metric("Medium Tier", f"{tier_counts['Medium']:,}", f"{(tier_counts['Medium']/total_patients)*100:.1f}%")
        c3.metric("Low Tier", f"{tier_counts['Low']:,}", f"{(tier_counts['Low']/total_patients)*100:.1f}%")

        ranked = tiered_cohort.sort_values("risk_score", ascending=False).reset_index(drop=True)
        queue = ranked.head(int(capacity))
        total_readmissions = float(ranked["target"].sum())
        queue_readmissions = float(queue["target"].sum())
        capture_rate = 0.0 if total_readmissions == 0 else 100.0 * queue_readmissions / total_readmissions

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
        st.json(roi)

        display_cols = [c for c in ["encounter_id", "patient_nbr", "risk_score", "tier", "target"] if c in queue.columns]
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