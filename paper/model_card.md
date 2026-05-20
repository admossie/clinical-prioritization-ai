# Model Card: Clinical Prioritization AI (v1.0.9)

## Model Details
- Model name: clinical-readmission-prioritizer
- Version: v1.0.9
- Primary algorithm: LightGBM (selected by benchmark comparison)
- Supporting preprocessor: scikit-learn pipeline stored in `models/preprocessor.joblib`
- Training target: binary readmission outcome (30-day style proxy from source dataset conventions)

## Intended Use
This model is intended for operational triage support in care coordination workflows where teams must prioritize follow-up under limited capacity.

Appropriate use cases:
- Queue ranking for outreach prioritization
- Scenario analysis for threshold and capacity planning
- Demonstration and research workflows in clinical informatics

Out-of-scope use cases:
- Diagnostic decision making
- Fully automated treatment decisions
- Use without local validation and governance review

## Data
- Primary development data: `data/raw/diabetic_data.csv`
- External validation inputs: files in `data/external/`
- Feature engineering includes temporal encounter history features and utilization summaries.

## Performance Summary
Internal held-out test results (v1.0.9 release artifacts):
- ROC-AUC: 0.689
- PR-AUC: 0.244
- Brier score: 0.091
- Top-10% capture: 26.1%
- Top-20% capture: 41.2%

Artifacts:
- Metrics table: `outputs/tables/evaluation_metrics.csv`
- Curves: `outputs/figures/roc_curve.png`, `outputs/figures/pr_curve.png`, `outputs/figures/calibration_curve.png`

## Explainability
- Global explainability artifact: `outputs/figures/shap_summary.png`
- Dashboard includes prediction context and operational priority tiers.

## Fairness and Subgroup Reporting
Subgroup analyses are exported for audit and review:
- `outputs/tables/fairness_age.csv`
- `outputs/tables/fairness_gender.csv`
- `outputs/tables/fairness_race.csv`

These outputs should be interpreted with local population context and sample-size awareness.

## External Validation
Preliminary external validation artifacts are provided:
- `outputs/tables/external_validation_metrics.csv`
- `outputs/tables/external_scored.csv`

External performance should be treated as preliminary unless confirmed in the target deployment environment.

## Risks and Limitations
- Model trained on structured historical data; distribution shift can degrade reliability.
- Performance and fairness may vary across institutions and cohorts.
- Missingness patterns and coding practices can materially affect predictions.
- Intended as decision support, not a substitute for clinical judgment.

## Ethical and Governance Notes
Before real-world adoption:
1. Conduct local validation and calibration checks.
2. Review subgroup metrics in target population.
3. Define human oversight and escalation policies.
4. Document intended intervention pathways and accountability.

## Monitoring Recommendations
- Track drift in feature distributions and score distributions.
- Monitor calibration drift and update thresholds periodically.
- Recompute subgroup fairness metrics on a regular cadence.
- Version all model and preprocessor artifacts with release tags.

## Reproducibility and Deployment
- Source code: https://github.com/admossie/clinical-prioritization-ai
- Release: https://github.com/admossie/clinical-prioritization-ai/releases/tag/v1.0.9
- Live demo: https://admossie-clinical-prioritization-ai-app-hzx0qb.streamlit.app/
