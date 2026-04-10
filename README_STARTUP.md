# AI Care Prioritization Engine (Startup Edition)

AI Care Prioritization Engine is a clinical workflow platform prototype built to make patient prioritization actionable for hospitals and care teams.

## Value proposition

- Predict which patients are at highest risk of hospital readmission.
- Help care coordinators allocate follow-up resources to the right patients.
- Show results as an easy-to-understand **risk percentage, cohort percentile, and priority tier**.
- Provide a demo-ready interface using Streamlit for rapid stakeholder review.

## Startup-ready capabilities

- Faster pilot deployment with Docker support.
- Clear developer onboarding via `CONTRIBUTING.md`.
- Reproducible packaging with `pyproject.toml`.
- Issue and PR templates for scalable collaboration.

## Portfolio positioning

Use this project summary for resumes, interviews, and portfolio descriptions:

> Built and deployed a production-grade ML decision support system for clinical readmission prioritization using Streamlit, integrating risk prediction, capacity-aware triage, and ROI simulation.

- Designed an interactive patient simulation interface with real-time inference
- Implemented percentile-based prioritization and operational thresholds
- Modeled business impact with workflow and ROI simulation outputs
- Deployed publicly with CI validation and GitHub-based release workflows

## Get started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the demo app:
   ```bash
   streamlit run app.py
   ```
3. Run validation tests:
   ```bash
   pytest -q
   ```
4. Build the container:
   ```bash
   docker build -t ai-care-prioritization-engine .
   ```

## Next-step product work

- Add a clinician-facing dashboard for daily worklists.
- Build performance monitoring for intervention impact.
- Connect outputs to care management and referral systems.
