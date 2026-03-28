from pathlib import Path
import time
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
PREPROC_PATH = ROOT / "models" / "preprocessor.joblib"


def align_to_preprocessor_input(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    # Get all expected feature names from the preprocessor
    try:
        expected_cols = list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn
        expected_cols = []

    # One-hot encode categorical variables to match preprocessor expectations
    aligned = df.copy()
    # Age
    for age_band in ["[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]:
        aligned[f"cat__age_{age_band}"] = 1 if df.loc[0, "cat__age"] == age_band else 0
    # Gender
    for gender_val in ["Female", "Male"]:
        aligned[f"cat__gender_{gender_val}"] = 1 if df.loc[0, "cat__gender"] == gender_val else 0
    # Race
    for race_val in ["Caucasian", "Hispanic", "Other"]:
        aligned[f"cat__race_{race_val}"] = 1 if df.loc[0, "cat__race"] == race_val else 0

    # Remove original categorical columns
    aligned = aligned.drop(["cat__age", "cat__gender", "cat__race"], axis=1)

    # Add missing columns as zeros
    for col in expected_cols:
        if col not in aligned.columns:
            aligned[col] = 0

    # Ensure column order matches expected_cols
    return aligned[expected_cols] if expected_cols else aligned


st.set_page_config(page_title="AI Care Prioritization Engine", layout="wide")
st.title("AI Care Prioritization Engine")
st.caption("Capacity-aware readmission prioritization demo")

if not MODEL_PATH.exists() or not PREPROC_PATH.exists():
    st.warning("Train the model first.")
    st.stop()

if "model" not in st.session_state or "preprocessor" not in st.session_state:
    with st.spinner("Loading model artifacts..."):
        st.session_state["model"] = joblib.load(MODEL_PATH)
        st.session_state["preprocessor"] = joblib.load(PREPROC_PATH)

model = st.session_state["model"]
preprocessor = st.session_state["preprocessor"]

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

    row = pd.DataFrame([{ 
        "num__encounter_id": 999999,
        "num__patient_nbr": 9999,
        "cat__race": race,
        "cat__gender": gender,
        "cat__age": age,
        "num__admission_type_id": admission_type_id,
        "num__discharge_disposition_id": discharge_disposition_id,
        "num__admission_source_id": admission_source_id,
        "num__time_in_hospital": time_in_hospital,
        "num__num_lab_procedures": num_lab_procedures,
        "num__num_procedures": num_procedures,
        "num__num_medications": num_medications,
        "num__number_outpatient": number_outpatient,
        "num__number_emergency": number_emergency,
        "num__number_inpatient": number_inpatient,
        "num__number_diagnoses": number_diagnoses,
        "num__encounter_number": encounter_number,
        "num__prior_encounters": prior_encounters,
        "num__prior_number_inpatient_sum": prior_inpatient,
        "num__prior_number_inpatient_mean": float(prior_inpatient),
        "num__prior_number_outpatient_sum": prior_outpatient,
        "num__prior_number_outpatient_mean": float(prior_outpatient),
        "num__prior_number_emergency_sum": prior_emergency,
        "num__prior_number_emergency_mean": float(prior_emergency),
        "num__prior_total_visits": prior_encounters,
        "num__diag_delta": diag_delta,
        "num__med_delta": med_delta,
        "num__prior_positive_count": prior_positive_count,
        "num__ever_prior_positive": ever_prior_positive,
    }])

    row = align_to_preprocessor_input(row, preprocessor)


    import time
    t0 = time.time()
    Xt = preprocessor.transform(row)
    t1 = time.time()
    risk = float(model.predict_proba(Xt)[:, 1][0])
    t2 = time.time()

    st.caption(f"Preprocessor: {t1-t0:.3f}s, Model: {t2-t1:.3f}s, Total: {t2-t0:.3f}s")

    tier = "Low" if risk < 0.30 else "Medium" if risk < 0.60 else "High"

    st.subheader("Prediction Result")
    m1, m2 = st.columns(2)
    m1.metric("Risk score", f"{risk:.3f}")
    m2.metric("Risk tier", tier)

    if tier == "High":
        st.error("High-priority patient for follow-up.")
    elif tier == "Medium":
        st.warning("Moderate-priority patient for review.")
    else:
        st.success("Low-priority patient based on current model.")

    st.caption("Demo prediction only. Final trustworthiness depends on training with the full public dataset.")