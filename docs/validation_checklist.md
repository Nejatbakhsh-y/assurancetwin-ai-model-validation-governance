# Validation Checklist

## Document Purpose

This checklist summarizes the evidence required for the AssuranceTwin AI model validation and AI governance package. It is intended for independent validation review, model risk management review, audit readiness, and model risk committee submission.

| Field | Description |
|---|---|
| Project | AssuranceTwin AI - Model Validation Governance |
| Generated Date | June 26, 2026 |
| AssuranceTwin Score | 73.37/100 |
| Approval Recommendation | Conditional Approval |

## Validation Checklist

| Validation Control | Status | Review Notes |
| --- | --- | --- |
| Model inventory completed | Available | Confirm model owner, validator, model type, business use, risk tier, target variable, approval status, and monitoring frequency. |
| Training and validation data documented | Available | Confirm dataset source, target construction, exclusions, missingness, and known limitations. |
| Target distribution reviewed | Available | Confirm class balance and approval outcome definition. |
| Champion-challenger models trained | Available | Confirm that model selection considered governance quality, not only predictive performance. |
| Independent validation metrics generated | Available | Review AUC, accuracy, precision, recall, F1, balanced accuracy, Brier score, approval rate, FPR, and FNR. |
| Calibration reviewed | Available | Review calibration curve, Brier score, expected calibration error, and group-level calibration. |
| Fairness and bias testing completed | Available | Review approval-rate differences, disparate impact, FPR gaps, FNR gaps, equal opportunity, and group calibration. |
| Explainability reviewed | Available | Review feature importance, permutation importance, SHAP-style explanations, and stability across groups, time, and models. |
| Stress testing completed | Available | Review income shock, loan amount shock, LTV shock, missing-data shock, minority-tract shift, out-of-time validation, and recession-like scenario. |
| Drift monitoring simulated | Available | Review PSI, CSI, data drift, prediction drift, performance drift, fairness drift, and calibration drift. |
| AssuranceTwin score generated | 73.37/100 | Confirm final score and component scores across performance, calibration, fairness, robustness, drift, explainability, documentation, and monitoring. |
| Model card generated | Generated | Confirm intended use, out-of-scope use, data summary, validation evidence, limitations, monitoring, and approval recommendation. |
| AI governance card generated | Generated | Confirm governance use, protected attribute handling, model risks, monitoring requirements, human oversight, and documentation status. |
| Human oversight defined | Required | Confirm accountable reviewers, override procedures, escalation path, and committee approval process. |
| Production use restriction documented | Required | Confirm that the model is not used for actual underwriting, pricing, adverse action, or consumer-impacting automated decisions. |
| Final model risk committee recommendation documented | Conditional Approval | The AssuranceTwin score is 73.37/100. The model may be considered for limited or controlled use only after remediation items are documented, owners are assigned, monitoring controls are approved, and residual risks are accepted by the appropriate governance authority. |

## Evidence Inventory

| Evidence Item | Expected File | Status |
| --- | --- | --- |
| Model Inventory | reports/tables/model_inventory.csv | Available |
| Model Performance Summary | reports/tables/model_performance_summary.csv | Available |
| Independent Validation Metrics | reports/tables/independent_validation_metrics.csv | Available |
| Fairness Metrics | reports/tables/fairness_metrics.csv | Available |
| Calibration Summary | reports/tables/calibration_summary.csv | Available |
| Stress Test Results | reports/tables/stress_test_results.csv | Available |
| Drift Monitoring Summary | reports/tables/drift_monitoring_summary.csv | Available |
| Explanation Stability | reports/tables/explanation_stability.csv | Available |
| AssuranceTwin Scorecard | reports/tables/assurancetwin_scorecard.csv | Available |
| Modeling Dataset Summary | reports/tables/modeling_dataset_summary.csv | Available |
| Target Distribution | reports/tables/modeling_target_distribution.csv | Available |

## Final Review Notes

The model should not be approved solely on predictive performance. Approval should depend on the full governance profile, including calibration quality, fairness behavior, stress-test robustness, drift stability, explanation stability, documentation completeness, and monitoring readiness.

Final approval should be issued only after all material findings are closed or formally accepted as residual risk by the appropriate governance authority.
