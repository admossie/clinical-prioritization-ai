# Contributing

Thanks for your interest in improving AI Care Prioritization Engine.
This project is designed to be startup-ready, with a focus on rapid iteration, reproducible results, and easy handoff.

## How to contribute

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make changes with clear, small commits.
4. Open a pull request with a concise summary and any validation steps.

## Local setup

- Install the main dependencies with `pip install -r requirements.txt`.
- Install development tools with `pip install -r requirements-dev.txt`.
- Run the main application with `streamlit run app/streamlit_app.py`.

## Testing

- Run unit tests with `pytest -q`.
- Add tests for new features and bug fixes.

## Startup focus

- Keep the product story clear: risk prediction, care prioritization, and operational triage.
- Favor reproducibility, simple deployment, and integration with existing hospital workflows.
- Use lightweight interfaces and metrics that stakeholders can explain to clinicians.
