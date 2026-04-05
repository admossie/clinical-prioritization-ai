


# Clinical Prioritization AI: Capacity-Aware Open-Source Engine

## Authors

[Your Name], [Collaborators], et al.

## Abstract

We present Clinical Prioritization AI, an open-source, capacity-aware engine for hospital readmission risk prediction and operational triage. The pipeline combines temporal feature engineering, calibrated machine learning, subgroup fairness analysis, cost-sensitive threshold selection, workflow simulation, and external validation. On the held-out benchmark cohort, the best model (LightGBM) achieved a ROC-AUC of 0.689, a PR-AUC of 0.244, and captured 41.2% of true readmissions within the top 20% highest-risk patients. A companion Streamlit app exposes percentile-based risk tiers, queue ranking, explainability, and ROI-aware workflow simulation for practical demonstration. All code, sample data, and reproducibility artifacts are openly available at https://github.com/admossie/clinical-prioritization-ai.


## Introduction

Hospital readmissions are a persistent challenge for healthcare systems, impacting patient outcomes and operational costs. Accurate, fair, and operationally-aware risk prediction is essential for effective care management and resource allocation. We introduce a reproducible pipeline and interactive app for developing, validating, and deploying readmission risk models, with a focus on transparency, fairness, and real-world applicability.


## Methods

### Data and Preprocessing
We use publicly available datasets, including MIMIC-like samples, and provide preprocessing scripts for data cleaning and feature engineering. Temporal features are constructed from patient encounter histories to enable dynamic risk estimation.

### Model Development
We benchmark logistic regression, XGBoost, LightGBM, CatBoost, and a soft-voting ensemble using group-aware train/test splitting at the patient level. Models are compared with ROC-AUC, PR-AUC, and recall at low false-positive rates, after which the best-performing pipeline is serialized for evaluation and app deployment. Probability calibration and reliability plots are generated to support clinically interpretable risk estimates.

### Fairness and Thresholding
Subgroup metrics are computed to assess and mitigate bias across demographics. Cost-sensitive thresholding is applied to optimize for operational cost and workflow constraints.

### Workflow Simulation
Workflow simulation modules quantify the impact of triage thresholds and care-team capacity on patient prioritization and outcomes.

### External Validation
The engine supports validation on external datasets to assess transportability and generalizability.


## Results

### Figure 1. Model Calibration Curve
![Calibration Curve](../outputs/figures/calibration_curve.png)
*Caption: Calibration curve showing predicted vs. observed risk probabilities.*

### Figure 2. Precision-Recall Curve
![Precision-Recall Curve](../outputs/figures/pr_curve.png)
*Caption: Precision-recall curve for model discrimination.*

### Figure 3. Workflow Simulation Output
*See outputs/tables/workflow_scenarios.csv for simulated prioritization results.*

On the internal held-out test set, the best LightGBM model achieved a ROC-AUC of 0.689, a PR-AUC of 0.244, and a Brier score of 0.091. The ranked-risk workflow captured 26.1% of readmissions in the top 10% of patients and 41.2% in the top 20%, supporting targeted care-management outreach under limited capacity. Fairness, calibration, and queue-simulation outputs are exported as reproducible tables and figures for auditability.


## Reproducibility

All code, data, and Jupyter notebooks are provided for full reproducibility. The Streamlit app enables interactive exploration and demonstration. Detailed instructions are available in the README.


## Discussion

This engine provides a robust, extensible foundation for research and deployment of clinical risk models. Its modular design supports rapid experimentation and adaptation to new datasets or clinical settings. Limitations include reliance on structured EHR data and the need for local validation before deployment.


## Conclusion

The AI Care Prioritization Engine advances the state of the art in clinical risk prediction by integrating fairness, calibration, and operational awareness. It is suitable for both scientific publication and startup use.


## Code and Data Availability


All code and sample data are available at: https://github.com/admossie/clinical-prioritization-ai

## References

1. Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Sci Data. 2016;3:160035.
2. Lundberg SM, Lee S-I. A Unified Approach to Interpreting Model Predictions. Adv Neural Inf Process Syst. 2017;30.
3. Rajkomar A, Dean J, Kohane I. Machine Learning in Medicine. N Engl J Med. 2019;380(14):1347-1358.
4. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. J Mach Learn Res. 2011;12:2825-2830.
5. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. Proc 22nd ACM SIGKDD Int Conf Knowl Discov Data Min. 2016:785-794.

## Acknowledgments

We thank the open-source and clinical informatics communities for their contributions and feedback.
