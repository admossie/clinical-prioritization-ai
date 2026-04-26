# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- `DEPLOYMENT.md`, `.streamlit/config.toml`, and `packages.txt` for a smoother public demo deployment path.
- A Streamlit smoke test to catch app-launch and prediction-flow regressions earlier.
- A lightweight fallback demo-model path so the app still renders on clean checkouts without saved artifacts.
- Model metadata display in the UI and tracked trained artifacts for more reliable hosted demos.
- `src/inference.py` shared scoring module used by both the Streamlit app and the FastAPI service.
- `api/main.py` production-style FastAPI inference service with `/health`, `/predict`, and `/batch_predict` endpoints.
- `tests/test_api_smoke.py` to verify API health, single predict, and batch predict endpoints.
- `docker-compose.yml` for running Streamlit and FastAPI together as a local product-style stack.
- `POST /batch_predict` plus a Batch Queue Scoring upload workflow in the Streamlit app.
- `data/external/sample_batch_patients.csv` for quick demo-ready batch scoring.
- `SECURITY.md` and `SUPPORT.md` for a more professional GitHub repository structure.
- `conftest.py` at the project root to ensure `src` and `api` packages are importable in all pytest invocations.

### Changed
- Improved Streamlit prediction wording to show **estimated risk as a percentage**, percentile-based priority context, and clearer operational guidance.
- Updated GitHub-facing documentation to explain how to interpret `Estimated risk`, `Relative percentile`, and `Priority tier`.
- Standardized demo launch instructions around `streamlit run app.py`.
- Expanded the app from single-patient prediction to both individual and uploaded CSV queue scoring.
- Restructured `README.md` and `CONTRIBUTING.md` into a cleaner, more professional GitHub presentation format.

## [1.0.2] - 2026-04-07
### Added
- Initial startup-ready packaging and documentation improvements.
- `pyproject.toml` for Python packaging and dependency metadata.
- `requirements-dev.txt` for developer tooling.
- `CONTRIBUTING.md` and `CHANGELOG.md` for project collaboration.
