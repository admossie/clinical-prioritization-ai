## Model Performance

### ROC Curve
![ROC](outputs/figures/roc_curve.png)

### Precision-Recall Curve
![PR](outputs/figures/pr_curve.png)

### Calibration Curve
![Calibration](outputs/figures/calibration_curve.png)

# Clinical Prioritization AI

## Overview

The AI Care Prioritization Engine is a capacity-aware, machine learning-based system for predicting hospital readmission risk and supporting clinical prioritization. Designed for both scientific research and real-world deployment, it features:

- **Temporal feature engineering**
- **Model calibration and fairness analysis**
- **Cost-sensitive thresholding**
- **Workflow simulation**
- **External validation**
- **Interactive Streamlit app**

This project is suitable for scientific publication, reproducible research, and as a foundation for healthcare AI startups.

## Features

- **Temporal Modeling:** Extracts and uses patient history for improved prediction.
- **Calibration:** Evaluates and visualizes model probability calibration.
- **Fairness:** Subgroup analysis for bias and equity.
- **Cost-sensitive Thresholding:** Optimizes decision thresholds for operational cost.
- **Workflow Simulation:** Simulates real-world triage and prioritization.
- **External Validation:** Validates model on external datasets.
- **Streamlit App:** User-friendly interface for demo and clinical exploration.

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```


## Results

- ROC-AUC: 0.78
- PR-AUC: 0.41
- Top 20% capture: 65%
- Brier Score: 0.18

The model demonstrates strong ranking performance and reliable probability calibration for clinical prioritization.

## Usage

### 1. Train the Model
```bash
python -m src.train --data-path data/raw/diabetic_data.csv
```

### 2. Evaluate Model Performance
```bash
python -m src.evaluate --data-path data/raw/diabetic_data.csv
```

### 3. External Validation
```bash
python -m src.external_validate --data-path data/raw/diabetic_data.csv
```

### 4. Workflow Simulation
```bash
python -m src.workflow_simulation --scored-data outputs/tables/test_scored.csv
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
or
```bash
streamlit run app/streamlit_app.py
```

## Project Structure

- `src/` — Core modules: training, evaluation, calibration, fairness, simulation
- `app/` — Streamlit app code
- `models/` — Saved model and preprocessor artifacts
- `data/` — Example raw and external datasets
- `outputs/` — Generated results, figures, and tables
- `notebooks/` — Jupyter notebooks for reproducibility and exploration
- `paper/` — Manuscript draft for publication

## Reproducibility

Jupyter notebooks in `notebooks/` demonstrate temporal modeling, calibration, fairness, and workflow analysis. See `paper/manuscript_draft.md` for a publication-ready manuscript template.

## Screenshots

Add screenshots of the Streamlit app to `docs/README_screenshots.md` after running the app.

## Citation

If you use this project in your research, please cite:

> Clinical Prioritization AI (2026). https://github.com/admossie/clinical-prioritization-ai

## Contact

For questions or collaboration, contact: Abebaw Mossie <abebawdebas7@gmail.com>
