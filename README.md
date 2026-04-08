# Operationalizing Machine Learning for Clinical Decision Support: A Capacity-Aware Framework for Hospital Readmission Prioritization

[![Release](https://img.shields.io/github/v/release/admossie/clinical-prioritization-ai)](https://github.com/admossie/clinical-prioritization-ai/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Capacity-aware hospital readmission prioritization with calibrated machine learning, workflow simulation, fairness analysis, and an interactive Streamlit demo.

## Why this project exists

Most readmission models stop at ranking risk. Operational teams need one step more: which patients should be prioritized under real capacity constraints, what return can be expected, and how stable the decision logic remains when the queue size or intervention assumptions change.

This repository packages that full workflow into a reproducible research project and demo application.

## What it includes

- Temporal feature engineering from patient history
- Model training, calibration, and subgroup fairness analysis
- Cost-sensitive thresholding and queue-based prioritization
- External validation pipeline
- Workflow simulation for operational planning
- Streamlit app for live prediction, explainability, and queue export

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train --data-path data/raw/diabetic_data.csv
python -m src.evaluate --data-path data/raw/diabetic_data.csv
streamlit run app/streamlit_app.py
```

`app.py` is a thin wrapper. `app/streamlit_app.py` is the canonical app entrypoint.

## Startup resources

- `STARTUP_GUIDE.md` - product positioning and pilot-readiness checklist
- `README_STARTUP.md` - investor/demo-friendly project overview
- `CONTRIBUTING.md` - contributor and team workflow guidance
- `Dockerfile` - simple containerized app deployment

## Demo workflow

1. Train the model on the public diabetic readmission dataset.
2. Evaluate discrimination and calibration.
3. Run external validation.
4. Simulate a top-N queue using operational assumptions.
5. Launch the Streamlit demo.
6. Review the queue, explainability output, and download the current queue as CSV.

## Key outputs

- ROC-AUC: 0.69
- PR-AUC: 0.24
- Top 20% capture: 41%
- Brier score: 0.09

These results indicate moderate ranking performance with usable probability calibration for prioritization-oriented workflows.

## Streamlit screenshots

See [docs/README_screenshots.md](docs/README_screenshots.md) for the current screenshot notes.

- Risk prediction: [docs/01_risk_prediction.png](docs/01_risk_prediction.png)
- Triage dashboard: [docs/02_triage_dashboard.png](docs/02_triage_dashboard.png)
- Explainability: [docs/03_explainability.png](docs/03_explainability.png)
- Queue export: [docs/04_queue_export.png](docs/04_queue_export.png)
- Social preview asset: [docs/social_preview.png](docs/social_preview.png)

## Usage

### Train the model

```bash
python -m src.train --data-path data/raw/diabetic_data.csv
```

### Evaluate model performance

```bash
python -m src.evaluate --data-path data/raw/diabetic_data.csv
```

### Run external validation

```bash
python -m src.external_validate --data-path data/raw/diabetic_data.csv
```

### Simulate operational workflow

```bash
python -m src.workflow_simulation --scored-data outputs/tables/test_scored.csv
```

### Launch the app

```bash
streamlit run app/streamlit_app.py
```

### Use the triage dashboard

- Generate an individual patient risk prediction
- Review percentile-based risk tier assignment
- Adjust queue capacity and intervention assumptions
- Inspect ROI estimates and queue capture
- Export the current active queue with the in-app CSV button

## Decision-theoretic formulation

Define:

- Risk score:
  $$p_i = P(y_i = 1 \mid x_i)$$
  where $p_i$ is the predicted probability of readmission for patient $i$ given features $x_i$.
- Capacity constraint:
  $$K$$
  where $K$ is the number of patients that can be prioritized or intervened upon.
- Objective:
  $$
  \max_{i \in \text{Top-}K} \sum p_i \cdot \text{benefit} - \text{cost}
  $$

This framing makes the prioritization step explicit rather than treating prediction as the end product.

## Model performance figures

### ROC curve

![ROC](outputs/figures/roc_curve.png)

### Precision-recall curve

![PR](outputs/figures/pr_curve.png)

### Calibration curve

![Calibration](outputs/figures/calibration_curve.png)

## Project structure

- `src/` - Core modules for training, evaluation, calibration, fairness, and simulation
- `app/` - Streamlit application code
- `models/` - Saved model and preprocessor artifacts
- `data/` - Raw and example external datasets
- `outputs/` - Generated figures, tables, and scored outputs
- `notebooks/` - Reproducibility and exploratory notebooks
- `paper/` - Manuscript draft and publication material

### Structure convention

- Keep runtime artifacts in root-level `models/` and `outputs/` only
- Avoid duplicate artifact copies under `src/`

## Reproducibility

- Notebooks under `notebooks/` cover temporal modeling, calibration, fairness, and workflow analysis
- The manuscript template lives in `paper/manuscript_draft.md`
- The explainability figure can be regenerated deterministically from `outputs/generate_explainability_figure.py`
- The GitHub social preview asset can be regenerated from `outputs/generate_social_preview.py`

## Citation

If you use this project in research, please cite:

> Clinical Prioritization AI (2026). https://github.com/admossie/clinical-prioritization-ai

## Contact

Abebaw Mossie  
abebawdebas7@gmail.com

