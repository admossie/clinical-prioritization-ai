import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss
import shap
import joblib

# Load test scored data
scored = pd.read_csv('outputs/tables/test_scored.csv')

# ROC and PR curves
fpr, tpr, _ = roc_curve(scored['target'], scored['risk_score'])
precision, recall, _ = precision_recall_curve(scored['target'], scored['risk_score'])
roc_auc = roc_auc_score(scored['target'], scored['risk_score'])
avg_prec = average_precision_score(scored['target'], scored['risk_score'])

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/roc_curve.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(recall, precision, label=f'PR AUC={avg_prec:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/pr_curve.png', dpi=300)

# Cost vs threshold
costs = pd.read_csv('outputs/tables/threshold_sweep.csv')
plt.figure(figsize=(6,4))
plt.plot(costs['threshold'], costs['total_cost'], marker='o')
plt.xlabel('Threshold')
plt.ylabel('Total Cost')
plt.title('Cost vs. Threshold')
plt.tight_layout()
plt.savefig('outputs/figures/cost_vs_threshold.png', dpi=300)

# Risk score distribution
plt.figure(figsize=(6,4))
sns.histplot(scored['risk_score'], bins=30, kde=True)
plt.xlabel('Risk Score')
plt.title('Risk Score Distribution')
plt.tight_layout()
plt.savefig('outputs/figures/risk_score_distribution.png', dpi=300)

# Calibration curve
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(scored['target'], scored['risk_score'], n_bins=10)
plt.figure(figsize=(6,4))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0,1],[0,1],'--',color='gray',label='Perfectly calibrated')
plt.xlabel('Mean Predicted Value')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/calibration_curve.png', dpi=300)

# Brier score
brier = brier_score_loss(scored['target'], scored['risk_score'])
with open('outputs/tables/brier_score.txt','w') as f:
    f.write(f'Brier score: {brier:.4f}\n')

# SHAP feature importance (XGBoost)
model = joblib.load('models/best_model.joblib')
preprocessor = joblib.load('models/preprocessor.joblib')

# Use a sample for SHAP to save time
X = scored.drop(columns=['target','risk_score'])
X_proc = preprocessor.transform(X)
explainer = shap.Explainer(model)
shap_values = explainer(X_proc[:200])
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_proc[:200], show=False)
plt.tight_layout()
plt.savefig('outputs/figures/shap_summary.png', dpi=300)

print('All figures and tables generated in outputs/figures and outputs/tables.')
