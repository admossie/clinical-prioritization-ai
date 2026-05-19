# Deployment Guide

## Recommended public demo path: Streamlit Community Cloud

This repository is ready to deploy using `app.py` as the main entrypoint.

## Deployment checklist

Use this as the minimum release-ready checklist before sharing the app publicly:

1. Confirm the latest tag is present on GitHub.
2. Verify `app.py` launches locally with `streamlit run app.py`.
3. Confirm the model artifacts load and the fallback path still works.
4. Publish the Streamlit app with `app.py` as the main file.
5. Test a single-patient prediction and a CSV batch upload in the hosted app.
6. Capture a screenshot or short demo note for the README.

### Steps
1. Push the repository to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select the repository: `admossie/clinical-prioritization-ai`.
4. Set the main file path to `app.py`.
5. Deploy.

### Included deployment helpers
- `.streamlit/config.toml` for theme and hosted app defaults
- `packages.txt` for system packages often needed by `lightgbm` / `xgboost`
- `runtime.txt` to keep the hosted Python version aligned with CI and Docker
- a fallback demo model path so the app can still render even if saved artifacts are missing
- a built-in Batch Queue Scoring section for CSV upload and prioritized export

## Docker deployment

### Streamlit only

```bash
docker build -t ai-care-prioritization-engine .
docker run -p 8501:8501 ai-care-prioritization-engine
```

Then open `http://localhost:8501`.

### Streamlit + FastAPI together

```bash
docker compose up --build
```

This launches:
- `http://localhost:8501` for the Streamlit app
- `http://localhost:8000/health` for API health
- `http://localhost:8000/docs` for interactive API docs

## Troubleshooting

- If saved model artifacts are missing, the app now falls back to a lightweight demo model automatically.
- To regenerate the full trained artifacts anyway, run:
  ```bash
  python -m src.train --data-path data/raw/diabetic_data.csv
  ```
- If hosted deployment fails on system libraries, confirm `packages.txt` is being used.
- Use `streamlit run app.py` locally before deploying to verify the app still launches.

## Pilot readiness notes

If this deployment is being used for a pilot, collect these checks before widening access:

- Prediction load time in the hosted environment
- Whether the risk tier is understandable to the intended users
- Whether queue export works for the pilot cohort size
- Whether the current fallback behavior is acceptable for demo use
- Whether the pilot metrics are being stored somewhere reproducible
