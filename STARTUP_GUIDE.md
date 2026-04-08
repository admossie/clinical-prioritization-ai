# Startup Guide

## Product positioning

AI Care Prioritization Engine is designed for early-stage healthcare analytics pilots. It combines readmission-risk prediction with operational triage, enabling health systems to prioritize patients who need follow-up care, transitional support, and clinical coordination.

## Startup-ready highlights

- Deployable Streamlit app for rapid stakeholder demos.
- Reproducible Python packaging via `pyproject.toml`.
- Developer tooling and test support with `requirements-dev.txt`.
- Docker container support for cloud and on-premises pilots.

## Recommended next steps for a startup

1. Validate the model against a real pilot dataset.
2. Track key adoption metrics: intervention lift, workload reduction, and readmission avoidance.
3. Build a simple clinician dashboard for daily patient prioritization.
4. Prepare integration points for EHR and care management systems.

## How to use

- Run the app locally:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

- Run tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

- Build the Docker container:

```bash
docker build -t ai-care-prioritization-engine .
```
