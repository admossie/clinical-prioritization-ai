# Deployment Guide

## Recommended public demo path: Streamlit Community Cloud

This repository is ready to deploy using `app.py` as the main entrypoint.

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

## Docker deployment

```bash
docker build -t ai-care-prioritization-engine .
docker run -p 8501:8501 ai-care-prioritization-engine
```

Then open `http://localhost:8501`.

## Troubleshooting

- If saved model artifacts are missing, the app now falls back to a lightweight demo model automatically.
- To regenerate the full trained artifacts anyway, run:
  ```bash
  python -m src.train --data-path data/raw/diabetic_data.csv
  ```
- If hosted deployment fails on system libraries, confirm `packages.txt` is being used.
- Use `streamlit run app.py` locally before deploying to verify the app still launches.
