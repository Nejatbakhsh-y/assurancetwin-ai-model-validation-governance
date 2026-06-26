# Model Card

## Model Identification

| Field | Description |
|---|---|
| Model Name | HMDA Mortgage Application Approval Classifier |
| Model Type | Supervised binary classification |
| Project | AssuranceTwin AI - Model Validation Governance |
| Target Variable | approved |
| Business Use | Research and model-validation demonstration using public HMDA loan/application records to estimate mortgage application approval status. |
| Risk Tier | High |
| Model Owner | Yousef Nejatbakhsh |
| Independent Validator | Independent model validation reviewer / project validator |
| Generated Date | June 26, 2026 |
| Documentation Completeness | 11 of 11 expected evidence files available (100.00%) |
| AssuranceTwin Score | 73.37/100 |
| Approval Recommendation | Conditional Approval |

## Intended Use

This model is intended for a controlled model validation and AI governance project. It estimates a binary mortgage application approval outcome using a structured HMDA-based modeling dataset. The primary purpose is to demonstrate a complete model risk management workflow, including model inventory, champion-challenger comparison, independent validation, calibration review, fairness testing, explainability review, stress testing, monitoring simulation, and governance documentation.

The model card is designed for use by model developers, independent validators, model risk managers, AI governance reviewers, audit stakeholders, and a model risk committee.

## Out-of-Scope Use

This model should not be used for actual mortgage underwriting, consumer credit decisions, pricing, adverse action notices, regulatory reporting, or automated decision-making affecting real applicants. The data and validation framework support research, demonstration, and governance prototyping only.

The model should not be used outside the documented population, geography, time period, product scope, or data-generating process without additional independent validation.

## Dataset Summary

The model uses a cleaned HMDA-based modeling dataset created in earlier project steps. The target variable is constructed from mortgage application action outcomes. Because HMDA is observational and does not contain all underwriting variables, the model should be interpreted as a governance and validation artifact rather than a production underwriting system.

### Modeling Dataset Summary

| created_at | script | raw_file | raw_file_size_mb | raw_rows | raw_columns | modeling_rows | modeling_columns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-26 09:21:01 | scripts/03_create_clean_hmda_dataset.py | data\raw\hmda_lar_nj_2024.csv | 117.19 | 323940 | 99 | 290004 | 36 |

### Target Distribution

| approved | target_label | row_count | percent |
| --- | --- | --- | --- |
| 0 | not_approved_or_not_completed | 123427 | 42.5604 |
| 1 | approved_or_origination_related | 166577 | 57.4396 |

## Model Development Summary

The project compares champion and challenger model candidates. The best predictive model is not automatically treated as the best governed model. Selection should consider predictive performance, calibration, fairness, robustness, drift stability, explainability stability, documentation quality, and monitoring readiness.

### Champion-Challenger Performance Evidence

| model_name | roc_auc | average_precision | accuracy | balanced_accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.887724 | 0.882492 | 0.832373 | 0.825169 | 0.840795 | 0.873584 | 0.856876 |
| Calibrated Logistic Regression | 0.888446 | 0.883337 | 0.834347 | 0.825402 | 0.835846 | 0.885515 | 0.859964 |
| Gradient Boosting | 0.957832 | 0.960454 | 0.908987 | 0.897946 | 0.881573 | 0.972145 | 0.924646 |
| Random Forest | 0.944959 | 0.946295 | 0.89408 | 0.882998 | 0.870946 | 0.957474 | 0.912163 |
| XGBoost | 0.95125 | 0.95491 | 0.896853 | 0.887839 | 0.881092 | 0.948422 | 0.913518 |

## Independent Validation Results

Independent validation reviews model performance using metrics such as AUC, accuracy, precision, recall, F1, balanced accuracy, Brier score, calibration error, confusion matrix outputs, approval rate, false positive rate, and false negative rate.

### Independent Validation Metrics

| model_name | model_file | saved_object_type | saved_object_keys | classification_threshold | validation_n | actual_approval_rate | predicted_approval_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Champion Model | models\champion_model.pkl | dict | model; model_name; selection_role; selection_logic; target_column; feature_columns; numeric_features; categorical_features; removed_columns; performance_summary | 0.5 | 58001 | 0.5744038895881105 | 0.5984551990482923 |
| Challenger Model | models\challenger_model.pkl | dict | model; model_name; selection_role; selection_logic; target_column; feature_columns; numeric_features; categorical_features; removed_columns; performance_summary | 0.5 | 58001 | 0.5744038895881105 | 0.6326097825899554 |

## Calibration Review

Calibration is reviewed because probability quality is central in financial risk settings. A model with strong discrimination can still produce poorly calibrated probabilities. The validation team should review calibration curves, Brier score, expected calibration error, and group-level calibration.

### Calibration Evidence

| model_name | n_observations | observed_approval_rate | mean_predicted_probability | probability_min | probability_25pct | probability_median | probability_75pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Champion Model | 58001 | 0.5744038895881105 | 0.5377931951262532 | 0.0036413273733564 | 0.1002716847043391 | 0.6834716454196658 | 0.8538006787357949 |
| Challenger Model | 58001 | 0.5744038895881105 | 0.5755439647190581 | 0.0004923056130782 | 0.0238324291927907 | 0.7947809056660741 | 0.9335443637194346 |

## Fairness and Bias Review

Fairness testing evaluates whether model outcomes and errors differ materially across protected or governance-relevant groups. The review should include approval-rate differences, disparate impact ratios, false-negative-rate gaps, false-positive-rate gaps, equal opportunity differences, and group calibration.

Protected attributes should be used for validation, bias testing, monitoring, and governance review. They should not be used for direct production decisioning unless explicitly approved by legal, compliance, and governance stakeholders.

### Fairness Evidence

| attribute | group | reference_group | n | small_group_flag | actual_positives | actual_negatives | actual_approval_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| race | Free Form Text Only | Joint | 155 | False | 41 | 114 | 0.2645161290322581 |
| race | Race Not Available | Joint | 62332 | False | 32187 | 30145 | 0.516380029519348 |
| race | Native Hawaiian or Other Pacific Islander | Joint | 624 | False | 235 | 389 | 0.3766025641025641 |
| race | American Indian or Alaska Native | Joint | 1253 | False | 505 | 748 | 0.4030327214684757 |
| race | 2 or more minority races | Joint | 631 | False | 281 | 350 | 0.4453248811410459 |
| race | Black or African American | Joint | 25436 | False | 12310 | 13126 | 0.4839597420978141 |
| race | Asian | Joint | 31821 | False | 18142 | 13679 | 0.5701266459256465 |
| race | White | Joint | 162731 | False | 99592 | 63139 | 0.6120038591294836 |
| race | Joint | Joint | 5021 | False | 3284 | 1737 | 0.654052977494523 |
| ethnicity | Ethnicity Not Available | Joint | 54597 | False | 28680 | 25917 | 0.5253035881092367 |
| ethnicity | Free Form Text Only | Joint | 224 | False | 86 | 138 | 0.3839285714285714 |
| ethnicity | Hispanic or Latino | Joint | 36919 | False | 18785 | 18134 | 0.5088165984994176 |

## Explainability Review

Explainability review evaluates feature importance, permutation importance, SHAP-based explanations where available, group-level explanation differences, and explanation stability across time splits or model variants.

The governance concern is not only whether the model is explainable, but whether explanations remain stable across demographic groups, time periods, and challenger models.

### Explanation Stability Evidence

| comparison_type | dimension | segment | model | n_rows | n_positive | positive_rate | reference_model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference | overall | overall | champion | 290004 | 166577 | 0.5743955255789576 | champion |
| group_vs_overall | race | White | champion | 162731 | 99592 | 0.6120038591294836 | champion |
| group_vs_overall | race | Race Not Available | champion | 62332 | 32187 | 0.516380029519348 | champion |
| group_vs_overall | race | Asian | champion | 31821 | 18142 | 0.5701266459256465 | champion |
| group_vs_overall | race | Black or African American | champion | 25436 | 12310 | 0.4839597420978141 | champion |
| group_vs_overall | race | Joint | champion | 5021 | 3284 | 0.654052977494523 | champion |
| group_vs_overall | race | American Indian or Alaska Native | champion | 1253 | 505 | 0.4030327214684757 | champion |
| group_vs_overall | race | 2 or more minority races | champion | 631 | 281 | 0.4453248811410459 | champion |
| group_vs_overall | race | Native Hawaiian or Other Pacific Islander | champion | 624 | 235 | 0.3766025641025641 | champion |
| group_vs_overall | race | Free Form Text Only | champion | 155 | 41 | 0.2645161290322581 | champion |
| group_vs_overall | ethnicity | Not Hispanic or Latino | champion | 190910 | 114482 | 0.5996647635011262 | champion |
| group_vs_overall | ethnicity | Ethnicity Not Available | champion | 54597 | 28680 | 0.5253035881092367 | champion |

## Stress Testing and Robustness Review

Stress testing evaluates sensitivity under adverse or shifted scenarios such as income shocks, loan amount increases, LTV increases, missing-data shocks, minority-tract distribution shifts, out-of-time validation, and recession-like synthetic scenarios.

### Stress Testing Evidence

| scenario | model_name | model_source | n_records | actual_approval_rate | predicted_approval_rate | predicted_denial_rate | average_predicted_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | logistic_regression | models\champion_model.pkl | 87002 | 0.5744005884922185 | 0.3634858968759339 | 0.6365141031240661 | 0.3338187191376405 |
| baseline | gradient_boosting | models\challenger_model.pkl | 87002 | 0.5744005884922185 | 0.5284821038596814 | 0.4715178961403186 | 0.4859776158973241 |
| income_shock | logistic_regression | models\champion_model.pkl | 87002 | 0.5744005884922185 | 0.3635088848532217 | 0.6364911151467783 | 0.3338754498009811 |
| income_shock | gradient_boosting | models\challenger_model.pkl | 87002 | 0.5744005884922185 | 0.5228270614468633 | 0.4771729385531367 | 0.4810368127952423 |
| loan_amount_increase | logistic_regression | models\champion_model.pkl | 87002 | 0.5744005884922185 | 0.3636583067055929 | 0.636341693294407 | 0.3340918723763509 |
| loan_amount_increase | gradient_boosting | models\challenger_model.pkl | 87002 | 0.5744005884922185 | 0.5299533344061056 | 0.4700466655938944 | 0.4876737624074747 |
| ltv_increase | logistic_regression | models\champion_model.pkl | 87002 | 0.5744005884922185 | 0.3634858968759339 | 0.6365141031240661 | 0.3338170308233301 |
| ltv_increase | gradient_boosting | models\challenger_model.pkl | 87002 | 0.5744005884922185 | 0.5020574239672652 | 0.4979425760327348 | 0.4552022154837379 |
| missing_data_shock | logistic_regression | models\champion_model.pkl | 87002 | 0.5744005884922185 | 0.3136249741385255 | 0.6863750258614745 | 0.3012704159237526 |
| missing_data_shock | gradient_boosting | models\challenger_model.pkl | 87002 | 0.5744005884922185 | 0.4921840877221213 | 0.5078159122778787 | 0.4566392259251768 |
| minority_tract_distribution_shift | logistic_regression | models\champion_model.pkl | 87002 | 0.5744005884922185 | 0.3634858968759339 | 0.6365141031240661 | 0.3338187191376405 |
| minority_tract_distribution_shift | gradient_boosting | models\challenger_model.pkl | 87002 | 0.5744005884922185 | 0.5284821038596814 | 0.4715178961403186 | 0.4859776158973241 |

## Drift and Monitoring Review

Ongoing monitoring should track population stability, characteristic stability, data drift, prediction drift, performance drift, fairness drift, and calibration drift. Material drift should trigger escalation, root-cause analysis, remediation, and possible model recalibration or redevelopment.

### Drift Monitoring Evidence

| monitoring_period | period_role | n_records | model_source | used_existing_model_artifact | event_rate | score_mean | approval_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024Q1_sim_1 | Baseline train period | 36251 | models\champion_model.pkl | True | 0.4366224380017103 | 0.4455309861480557 | 0.5259441118865686 |
| 2024Q2_sim_2 | Baseline train period | 36250 | models\champion_model.pkl | True | 0.4908137931034482 | 0.4715989404719782 | 0.5625103448275862 |
| 2024Q3_sim_3 | Baseline train period | 36251 | models\champion_model.pkl | True | 0.5523709690767151 | 0.5117074541536569 | 0.5762047943505006 |
| 2024Q4_sim_4 | Baseline train period | 36250 | models\champion_model.pkl | True | 0.5846620689655172 | 0.5383029437390441 | 0.5949793103448275 |
| 2024Q1_sim_5 | Monitoring period | 36250 | models\champion_model.pkl | True | 0.6212965517241379 | 0.5699842948233377 | 0.6220137931034483 |
| 2024Q2_sim_6 | Monitoring period | 36251 | models\champion_model.pkl | True | 0.6335273509696284 | 0.5756083377741847 | 0.6299136575542744 |
| 2024Q3_sim_7 | Monitoring period | 36250 | models\champion_model.pkl | True | 0.6329103448275862 | 0.5676798598695628 | 0.6174068965517241 |
| 2024Q4_sim_8 | Monitoring period | 36251 | models\champion_model.pkl | True | 0.6429615734738352 | 0.5894061255834626 | 0.6303274392430553 |

## AssuranceTwin Score

The AssuranceTwin score combines predictive performance, calibration, fairness, robustness, drift stability, explainability stability, documentation completeness, and monitoring readiness into a single governance-oriented score.

### Scorecard Evidence

| component | weight | score_0_to_100 | weighted_points | status | evidence_file |
| --- | --- | --- | --- | --- | --- |
| Predictive Performance Score | 0.2 | 94.4 | 18.88 | Strong | reports\tables\independent_validation_metrics.csv; reports\tables\model_performance_summary.csv |
| Calibration Score | 0.15 | 74.5 | 11.17 | Acceptable | reports\tables\calibration_summary.csv; reports\tables\independent_validation_metrics.csv |
| Fairness Score | 0.2 | 65.84 | 13.17 | Needs remediation | reports\tables\fairness_metrics.csv |
| Robustness / Stress-Test Score | 0.15 | 43.82 | 6.57 | Insufficient evidence or high risk | reports\tables\stress_test_results.csv |
| Drift Stability Score | 0.1 | 68.2 | 6.82 | Needs remediation | reports\tables\drift_monitoring_summary.csv |
| Explainability Stability Score | 0.1 | 67.63 | 6.76 | Needs remediation | reports\tables\explanation_stability.csv |
| Documentation Completeness Score | 0.05 | 100.0 | 5.0 | Strong | README.md; docs\model_inventory_template.md; reports\tables\model_inventory.csv; reports\validation\model_validation_report.md; reports\validation\fairness_validation_report.md; reports\validation\stress_testing_report.md; reports\validation\monitoring_plan.md |
| Monitoring Readiness Score | 0.05 | 100.0 | 5.0 | Strong | reports\validation\monitoring_plan.md; reports\tables\drift_monitoring_summary.csv; reports\figures\drift_dashboard_plot.png |
| AssuranceTwin Score | 1.0 | 73.37 | 73.37 | Acceptable | Aggregate score from all model-governance components |

## Approval Recommendation

**Recommendation:** Conditional Approval

**Rationale:** The AssuranceTwin score is 73.37/100. The model may be considered for limited or controlled use only after remediation items are documented, owners are assigned, monitoring controls are approved, and residual risks are accepted by the appropriate governance authority.

## Key Limitations

1. The project uses HMDA-based observational data and may not include all variables used in actual underwriting.
2. The target variable is derived from historical action outcomes and may reflect historical policy, operational, market, or institutional patterns.
3. Fairness metrics identify group-level differences but do not by themselves establish legal compliance or causality.
4. The model should not be treated as production-ready without legal, compliance, privacy, security, and business-owner review.
5. Performance and fairness may degrade under population shift, macroeconomic stress, geographic expansion, or policy changes.
6. Explainability outputs may be unstable across models, time windows, or subpopulations.
7. Monitoring thresholds require business and risk appetite approval before production use.

## Required Monitoring

The following controls are required before any controlled deployment:

- Monthly or quarterly performance monitoring.
- Population Stability Index and Characteristic Stability Index monitoring.
- Prediction distribution monitoring.
- Fairness metric monitoring across protected and governance-relevant groups.
- Calibration monitoring by time period and group.
- Data quality checks for missingness, outliers, invalid categories, and schema changes.
- Human review for adverse model behavior, exceptions, and overrides.
- Periodic independent validation refresh.
- Documented escalation thresholds and remediation owners.

## Human Oversight

Human oversight is required for interpretation, approval, override review, exception management, and periodic governance review. The model should support decision analysis and validation research; it should not independently make or execute consumer-impacting decisions.

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

## Documentation Status

This model card was automatically generated from available project evidence. Missing evidence should be completed before the model is submitted for final approval.
