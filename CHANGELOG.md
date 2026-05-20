# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Added clearer public-story documentation in `README.md` and a deployment checklist in `DEPLOYMENT.md`.
- Added a concise pilot validation plan so the next phase can focus on workflow proof instead of more model changes.
- Added a formal reproducibility appendix section to `paper/manuscript_draft.md` for journal submission readiness.
- Added `paper/model_card.md` with intended use, performance, fairness outputs, limitations, and governance notes.
- Strengthened top-level scientific framing in `README.md` with Research Overview, Clinical Relevance, Evaluation Summary, Calibration/Fairness, and Limitations sections.
- Added citation metadata via `CITATION.cff` and a repository-level `model_card.md` entrypoint.

## [1.0.9] - 2026-05-04
### Fixed
- Retrained `models/best_model.joblib` and `models/preprocessor.joblib` with scikit-learn 1.8.0. The preprocessor saved under 1.6.1 referenced an internal class (`_RemainderColsList`) removed in 1.8.0, causing a silent load failure and every patient returning the same 52.9% risk score regardless of inputs.
- Added try-except error handling to single-patient and batch scoring calls in the Streamlit app. Users now see a clear error message instead of a silent failure when prediction fails.
- Fixed demo preset switching: form widget values now reset when the user changes the demo preset, and stale prediction results are cleared automatically.
- Increased `FALLBACK_TRAIN_ROWS` from 250 to 5000 so the fallback demo model is more representative when saved artifacts cannot be loaded.

## [1.0.8] - 2026-05-04
### Fixed
- Relaxed `scikit-learn==1.6.1` pin to `>=1.6,<2.0` to support Python 3.14 (resolves to 1.8.0 which ships a pre-built wheel for CPython 3.14).
- Rebuilt virtual environment on Python 3.14.4 — all runtime and dev dependencies install cleanly with no source compilation required.
- `pip-audit` now reports **0 known vulnerabilities** across all runtime dependencies (previously 9 CVEs were blocked by the Python 3.9 interpreter; all patched versions are now installed).

## [1.0.7] - 2026-05-02
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
- `.github/workflows/security.yml` to run `pip-audit` and `bandit` on pushes, pull requests, and a weekly schedule.
- `[project.optional-dependencies].dev` in `pyproject.toml` to formalize developer and security tooling.
- `setup_windows_env.bat` to bootstrap a Python `3.10+` virtual environment, install dependencies, and run quality/security checks in one command.

### Changed
- Improved Streamlit prediction wording to show **estimated risk as a percentage**, percentile-based priority context, and clearer operational guidance.
- Updated GitHub-facing documentation to explain how to interpret `Estimated risk`, `Relative percentile`, and `Priority tier`.
- Standardized demo launch instructions around `streamlit run app.py`.
- Expanded the app from single-patient prediction to both individual and uploaded CSV queue scoring.
- Restructured `README.md` and `CONTRIBUTING.md` into a cleaner, more professional GitHub presentation format.
- Replaced `urllib` API calls in the Streamlit app with `requests` calls that enforce status checks and timeouts.
- Replaced silent `except ...: pass` fallback paths with debug logging in `app/streamlit_app.py` and `src/inference.py`.
- Tightened dependency version constraints in `requirements.txt` and aligned package metadata in `pyproject.toml` for better reproducibility.
- Raised the declared minimum Python version to `3.10+` to align with secured dependency support and deployment/runtime configuration.
- Updated `run_windows.bat` to fail fast on unsupported Python versions with a clear upgrade path.

## [1.0.2] - 2026-04-07
### Added
- Initial startup-ready packaging and documentation improvements.
- `pyproject.toml` for Python packaging and dependency metadata.
- `requirements-dev.txt` for developer tooling.
- `CONTRIBUTING.md` and `CHANGELOG.md` for project collaboration.
