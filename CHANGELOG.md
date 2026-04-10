# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- `DEPLOYMENT.md`, `.streamlit/config.toml`, and `packages.txt` for a smoother public demo deployment path.
- A Streamlit smoke test to catch app-launch and prediction-flow regressions earlier.
- A lightweight fallback demo-model path so the app still renders on clean checkouts without saved artifacts.
- Model metadata display in the UI and tracked trained artifacts for more reliable hosted demos.

### Changed
- Improved Streamlit prediction wording to show **estimated risk as a percentage**, percentile-based priority context, and clearer operational guidance.
- Updated GitHub-facing documentation to explain how to interpret `Estimated risk`, `Relative percentile`, and `Priority tier`.
- Standardized demo launch instructions around `streamlit run app.py`.

## [1.0.2] - 2026-04-07
### Added
- Initial startup-ready packaging and documentation improvements.
- `pyproject.toml` for Python packaging and dependency metadata.
- `requirements-dev.txt` for developer tooling.
- `CONTRIBUTING.md` and `CHANGELOG.md` for project collaboration.
