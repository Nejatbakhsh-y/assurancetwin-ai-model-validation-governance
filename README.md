# AssuranceTwin AI

## 1. Project Overview

**AssuranceTwin AI** is an end-to-end model validation and AI governance project for financial-services machine learning. The project demonstrates how a high-stakes predictive model should be evaluated beyond standard performance metrics.

The central argument of this repository is:

> The best predictive model is not automatically the best governed model.

This project simulates the work of an independent model validation, model risk management, and responsible AI governance team. It includes model inventory, champion/challenger modeling, independent validation metrics, fairness testing, calibration review, explainability analysis, stress testing, drift monitoring, governance documentation, and a Streamlit dashboard.

The project is designed to demonstrate practical experience in:

- Model Validation
- AI Governance
- Model Risk Management
- Responsible AI
- Financial Services AI
- Risk Analytics
- GenAI-supported governance documentation

---

## 2. Why This Project Matters

Financial-services AI systems are often used in high-impact decision environments. In these settings, model governance requires more than predictive accuracy.

A responsible validation process should examine:

- Model purpose and intended use
- Data quality and limitations
- Champion and challenger performance
- Calibration quality
- Fairness and bias risk
- Explainability and feature stability
- Stress-test behavior
- Drift and monitoring readiness
- Human oversight
- Documentation completeness
- Model risk committee readiness

This project demonstrates how those controls can be organized into a reproducible GitHub portfolio project.

---

## 3. Dataset

This project uses a public mortgage lending dataset structure inspired by HMDA-style Loan/Application Records.

The modeling task is framed as a binary classification problem:

- `approved = 1`: application is approval or origination related
- `approved = 0`: application is not approval or origination related

The project uses the data for educational, research, and portfolio demonstration purposes only.

Raw data files are not committed to the repository. Users should place raw data locally under:

```text
data/raw/
```

Processed modeling outputs are saved under:

```text
data/processed/
reports/tables/
reports/figures/
reports/validation/
```

---

## 4. Model Validation and AI Governance Framework

The repository follows a structured model validation and AI governance workflow.

| Component | Purpose |
|---|---|
| Model Inventory | Documents ownership, business use, risk tier, monitoring frequency, and approval status |
| Data Inspection | Reviews schema, missingness, target distribution, and data usability |
| Champion/Challenger Modeling | Compares multiple model candidates under both performance and governance criteria |
| Independent Validation Metrics | Evaluates predictive quality, confusion matrix behavior, approval rate, and error rates |
| Fairness and Bias Testing | Reviews group-level outcome differences and potential disparate impact |
| Calibration Analysis | Tests whether predicted probabilities are reliable |
| Explainability Review | Evaluates feature importance and explanation stability |
| Stress Testing | Simulates adverse scenarios and distribution shifts |
| Drift Monitoring | Evaluates data drift, prediction drift, performance drift, fairness drift, and calibration drift |
| AssuranceTwin Score | Combines validation and governance dimensions into a structured scorecard |
| Governance Documentation | Produces model card, AI governance card, validation checklist, and committee summary |

---

## 5. Champion and Challenger Model Design

The project trains and compares several model candidates:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Calibrated Logistic Regression
- Optional XGBoost or LightGBM, depending on local package availability

The champion model is not selected only by the strongest predictive metric. The project also considers:

- Calibration quality
- Fairness behavior
- Robustness under stress
- Drift stability
- Explainability stability
- Monitoring readiness
- Documentation completeness

This structure reflects a model risk management principle: strong AUC alone is not sufficient for approval in a high-stakes financial-services setting.

---

## 6. Independent Validation Metrics

The independent validation module evaluates:

- AUC
- Accuracy
- Precision
- Recall
- F1 Score
- Balanced Accuracy
- Brier Score
- Calibration Error
- Confusion Matrix
- Approval Rate
- False Positive Rate
- False Negative Rate

Output files:

- [reports/validation/model_validation_report.md](reports/validation/model_validation_report.md)
- [reports/tables/independent_validation_metrics.csv](reports/tables/independent_validation_metrics.csv)

---

## 7. Fairness and Bias Review

The fairness review tests group-level differences across available borrower and geography-related attributes.

Metrics include:

- Approval-rate difference
- Disparate impact ratio
- False-negative-rate gap
- False-positive-rate gap
- Equal opportunity difference
- Calibration by group

Output files:

- [reports/validation/fairness_validation_report.md](reports/validation/fairness_validation_report.md)
- [reports/tables/fairness_metrics.csv](reports/tables/fairness_metrics.csv)
- [reports/figures/fairness_group_comparison.png](reports/figures/fairness_group_comparison.png)

---

## 8. Calibration Review

Calibration matters because financial decision models often rely on probability estimates, not only classification labels.

A model with strong AUC can still produce unreliable probabilities. Poor calibration can distort:

- Risk ranking
- Threshold selection
- Approval logic
- Portfolio monitoring
- Model risk reporting

Output files:

- [reports/tables/calibration_summary.csv](reports/tables/calibration_summary.csv)
- [reports/figures/calibration_curve.png](reports/figures/calibration_curve.png)

---

## 9. Explainability and Stability Review

The explainability module evaluates whether model explanations are stable across models, time periods, and group-level segments.

Methods include:

- Feature importance
- Permutation importance
- SHAP-style explanation analysis, when available
- Group-level explanation comparison
- Explanation stability across time splits

Output files:

- [reports/figures/feature_importance.png](reports/figures/feature_importance.png)
- [reports/figures/shap_summary.png](reports/figures/shap_summary.png)
- [reports/tables/explanation_stability.csv](reports/tables/explanation_stability.csv)

---

## 10. Stress Testing

The stress-testing module evaluates model behavior under adverse or shifted conditions.

Stress scenarios include:

- Income shock
- Loan amount increase
- LTV increase
- Missing-data shock
- Minority-tract distribution shift
- Out-of-time validation
- Recession-like synthetic scenario

Output files:

- [reports/tables/stress_test_results.csv](reports/tables/stress_test_results.csv)
- [reports/figures/stress_test_model_sensitivity.png](reports/figures/stress_test_model_sensitivity.png)
- [reports/validation/stress_testing_report.md](reports/validation/stress_testing_report.md)

---

## 11. Drift Monitoring

The monitoring module simulates lifecycle governance after model development.

It evaluates:

- Population Stability Index
- Characteristic Stability Index
- Data drift
- Prediction drift
- Performance drift
- Fairness drift
- Calibration drift

Output files:

- [reports/tables/drift_monitoring_summary.csv](reports/tables/drift_monitoring_summary.csv)
- [reports/figures/drift_dashboard_plot.png](reports/figures/drift_dashboard_plot.png)
- [reports/validation/monitoring_plan.md](reports/validation/monitoring_plan.md)

---

## 12. AssuranceTwin Score

The AssuranceTwin Score combines model performance and AI governance dimensions into a single scorecard.

| Dimension | Weight |
|---|---:|
| Predictive Performance | 20% |
| Calibration | 15% |
| Fairness | 20% |
| Robustness / Stress Testing | 15% |
| Drift Stability | 10% |
| Explainability Stability | 10% |
| Documentation Completeness | 5% |
| Monitoring Readiness | 5% |

Output files:

- [reports/tables/assurancetwin_scorecard.csv](reports/tables/assurancetwin_scorecard.csv)
- [reports/figures/assurancetwin_score_radar.png](reports/figures/assurancetwin_score_radar.png)

---

## 13. Final Governance and Validation Deliverables

The repository includes the following final model validation and AI governance artifacts.

| Deliverable | File |
|---|---|
| Final Project Report | [reports/final/final_project_report.md](reports/final/final_project_report.md) |
| Independent Model Validation Report | [reports/validation/model_validation_report.md](reports/validation/model_validation_report.md) |
| Fairness Validation Report | [reports/validation/fairness_validation_report.md](reports/validation/fairness_validation_report.md) |
| Monitoring Plan | [reports/validation/monitoring_plan.md](reports/validation/monitoring_plan.md) |
| Model Risk Committee Summary | [reports/validation/model_risk_committee_summary.md](reports/validation/model_risk_committee_summary.md) |
| Model Card | [docs/model_card.md](docs/model_card.md) |
| AI Governance Card | [docs/ai_governance_card.md](docs/ai_governance_card.md) |
| Validation Checklist | [docs/validation_checklist.md](docs/validation_checklist.md) |
| Streamlit Governance Dashboard | [dashboard/streamlit_app.py](dashboard/streamlit_app.py) |

---

## 14. Streamlit Governance Dashboard

The repository includes a Streamlit dashboard for reviewing model validation and AI governance results.

Dashboard file:

- [dashboard/streamlit_app.py](dashboard/streamlit_app.py)

Dashboard tabs include:

- Model Inventory
- Champion vs Challenger
- Validation Metrics
- Fairness Review
- Calibration
- Stress Testing
- Drift Monitoring
- AssuranceTwin Score
- Model Risk Committee Summary

Run locally with:

```powershell
streamlit run dashboard/streamlit_app.py
```

The dashboard runs locally on the user's machine. The local Streamlit URL should not be treated as a public GitHub deployment link.

---

## 15. Repository Structure

```text
assurancetwin-ai-model-validation-governance/
│
├── README.md
│
├── dashboard/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── model_card.md
│   ├── ai_governance_card.md
│   └── validation_checklist.md
│
├── models/
│   ├── champion_model.pkl
│   └── challenger_model.pkl
│
├── reports/
│   ├── final/
│   │   └── final_project_report.md
│   │
│   ├── figures/
│   │   ├── roc_curves.png
│   │   ├── precision_recall_curves.png
│   │   ├── calibration_curve.png
│   │   ├── fairness_group_comparison.png
│   │   ├── feature_importance.png
│   │   ├── shap_summary.png
│   │   ├── stress_test_model_sensitivity.png
│   │   ├── drift_dashboard_plot.png
│   │   └── assurancetwin_score_radar.png
│   │
│   ├── tables/
│   │   ├── model_inventory.csv
│   │   ├── modeling_dataset_summary.csv
│   │   ├── model_performance_summary.csv
│   │   ├── independent_validation_metrics.csv
│   │   ├── fairness_metrics.csv
│   │   ├── calibration_summary.csv
│   │   ├── explanation_stability.csv
│   │   ├── stress_test_results.csv
│   │   ├── drift_monitoring_summary.csv
│   │   └── assurancetwin_scorecard.csv
│   │
│   └── validation/
│       ├── model_validation_report.md
│       ├── fairness_validation_report.md
│       ├── stress_testing_report.md
│       ├── monitoring_plan.md
│       └── model_risk_committee_summary.md
│
└── scripts/
    ├── 00_environment_check.py
    ├── 02_inspect_hmda_schema.py
    ├── 03_create_clean_hmda_dataset.py
    ├── 04_create_model_inventory.py
    ├── 05_train_champion_challenger_models.py
    ├── 06_independent_validation_metrics.py
    ├── 07_fairness_bias_testing.py
    ├── 08_calibration_analysis.py
    ├── 09_explainability_stability.py
    ├── 10_stress_testing.py
    ├── 11_monitoring_drift_simulation.py
    ├── 12_assurancetwin_score.py
    ├── 13_generate_model_validation_report.py
    └── 14_generate_model_card.py
```

---

## 16. How to Run the Project

Run the scripts from the repository root.

### Step 1: Environment Check

```powershell
python scripts/00_environment_check.py
```

### Step 2: Inspect Raw Dataset

```powershell
python scripts/02_inspect_hmda_schema.py
```

### Step 3: Create Clean Modeling Dataset

```powershell
python scripts/03_create_clean_hmda_dataset.py
```

### Step 4: Create Model Inventory

```powershell
python scripts/04_create_model_inventory.py
```

### Step 5: Train Champion and Challenger Models

```powershell
python scripts/05_train_champion_challenger_models.py
```

### Step 6: Independent Validation Metrics

```powershell
python scripts/06_independent_validation_metrics.py
```

### Step 7: Fairness and Bias Testing

```powershell
python scripts/07_fairness_bias_testing.py
```

### Step 8: Calibration Analysis

```powershell
python scripts/08_calibration_analysis.py
```

### Step 9: Explainability and Stability

```powershell
python scripts/09_explainability_stability.py
```

### Step 10: Stress Testing

```powershell
python scripts/10_stress_testing.py
```

### Step 11: Monitoring and Drift Simulation

```powershell
python scripts/11_monitoring_drift_simulation.py
```

### Step 12: AssuranceTwin Score

```powershell
python scripts/12_assurancetwin_score.py
```

### Step 13: Generate Model Validation Report

```powershell
python scripts/13_generate_model_validation_report.py
```

### Step 14: Generate Model Card and AI Governance Card

```powershell
python scripts/14_generate_model_card.py
```

### Step 15: Run the Streamlit Dashboard

```powershell
streamlit run dashboard/streamlit_app.py
```

---

## 17. Model Risk Committee Summary

The project includes a committee-style summary that consolidates the main governance findings.

File:

- [reports/validation/model_risk_committee_summary.md](reports/validation/model_risk_committee_summary.md)

The recommendation is conditional approval for demonstration and research purposes only.

Production use would require:

- Business-owner approval
- Legal and compliance review
- Fair lending review
- Data lineage review
- Threshold governance
- Human oversight procedures
- Ongoing monitoring controls
- Periodic revalidation
- Formal model risk committee approval

---

## 18. Project Limitations

This repository is a portfolio and research demonstration. It is not a production lending system.

Important limitations:

- The project is based on public mortgage data structure and simulated governance workflows.
- The outputs are for educational and professional demonstration purposes.
- Any production use would require institutional review, legal review, compliance review, fair lending review, data governance review, and formal model risk approval.
- Group-level fairness findings require careful interpretation and domain review.
- Model monitoring thresholds should be customized before production deployment.
- The AssuranceTwin Score is a governance aid, not a substitute for expert validation judgment.

---

## 19. Professional Relevance

This project is directly relevant to roles involving:

- Model Validation
- AI Governance
- Model Risk Management
- Responsible AI
- Credit Risk Analytics
- Financial Services AI
- Risk Analytics
- Fairness and Bias Testing
- Model Monitoring
- GenAI-supported Documentation
- Regulatory and Committee-Level Model Review

---

## 20. Final Positioning

AssuranceTwin AI demonstrates a practical, reproducible, and governance-centered approach to machine learning validation in financial services.

The project moves beyond standard machine-learning evaluation by combining:

- Predictive performance
- Calibration
- Fairness
- Explainability
- Stress testing
- Drift monitoring
- Documentation
- Committee-level governance review

This makes the repository suitable for demonstrating applied capability in model validation, responsible AI, and financial-services AI governance.