# Fairness and Bias Validation Report

## 1. Validation Purpose

This report evaluates whether the HMDA approval model creates material group-level disparities even when overall model performance appears acceptable. This is a central AI-governance concern for high-stakes credit and lending models.

## 2. Data and Model Scope

- Modeling dataset: `C:/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/AssuranceTwin AI - Model Validation Governance/assurancetwin-ai-model-validation-governance/data/processed/hmda_modeling_dataset.csv`
- Number of evaluated records: `290,004`
- Target column: `approved`
- Prediction source: `C:\Users\nejat\OneDrive\Desktop\UN\Skills\GitHub 2026\AssuranceTwin AI - Model Validation Governance\assurancetwin-ai-model-validation-governance\models\champion_model.pkl using saved feature columns`
- Classification threshold: `0.50`

## 3. Protected-Group Construction

- race derived from `derived_race`.
- ethnicity derived from `derived_ethnicity`.
- sex derived from `derived_sex`.
- income_band derived from `income`.
- minority_tract_band could not be derived because no minority-tract-like column was found.

## 4. Overall Model Context

| Metric | Value |
|---|---|
| AUC | 0.8872 |
| Accuracy | 0.8327 |
| Actual approval rate | 57.44% |
| Predicted approval rate | 59.75% |
| False positive rate | 22.37% |
| False negative rate | 12.55% |
| True positive rate | 87.45% |
| Brier score | 0.1240 |

## 5. Attribute-Level Fairness Summary

| attribute | groups_tested | reference_group | lowest_dir_group | lowest_disparate_impact_ratio | largest_fnr_gap_group | largest_fnr_gap | largest_fpr_gap_group | largest_fpr_gap | largest_equal_opportunity_gap_group | largest_equal_opportunity_difference | largest_calibration_error_group | largest_calibration_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ethnicity | 5 | Joint | Ethnicity Not Available | 0.6887 | Ethnicity Not Available | 0.2288 | Ethnicity Not Available | -0.0540 | Ethnicity Not Available | -0.2288 | Ethnicity Not Available | 0.0498 |
| income_band | 5 | Income > $150K | Missing/Unknown | 0.1432 | Missing/Unknown | 0.8112 | Missing/Unknown | -0.2236 | Missing/Unknown | -0.8112 | Missing/Unknown | 0.2806 |
| race | 9 | Joint | Free Form Text Only | 0.6093 | Race Not Available | 0.2102 | Black or African American | 0.0452 | Race Not Available | -0.2102 | Free Form Text Only | 0.0963 |
| sex | 4 | Joint | Sex Not Available | 0.5505 | Sex Not Available | 0.3597 | Sex Not Available | -0.0682 | Sex Not Available | -0.3597 | Sex Not Available | 0.0753 |

## 6. Governance Attention Items

The following groups triggered at least one governance attention flag. Flags are based on disparate impact ratio below 0.80, absolute error-rate gaps of at least 0.10, equal-opportunity gaps of at least 0.10, or group calibration error of at least 0.10.

| attribute | group | reference_group | n | predicted_approval_rate | approval_rate_difference | disparate_impact_ratio | false_negative_rate_gap | false_positive_rate_gap | equal_opportunity_difference | calibration_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| race | Free Form Text Only | Joint | 155 | 0.4194 | -0.2690 | 0.6093 | 0.0361 | 0.0303 | -0.0361 | 0.0963 |
| race | Race Not Available | Joint | 62332 | 0.4717 | -0.2166 | 0.6853 | 0.2102 | -0.0176 | -0.2102 | 0.0412 |
| race | Native Hawaiian or Other Pacific Islander | Joint | 624 | 0.4776 | -0.2107 | 0.6938 | 0.0832 | 0.0340 | -0.0832 | 0.0553 |
| race | American Indian or Alaska Native | Joint | 1253 | 0.5084 | -0.1799 | 0.7386 | 0.0454 | 0.0333 | -0.0454 | 0.0518 |
| race | 2 or more minority races | Joint | 631 | 0.5277 | -0.1606 | 0.7667 | 0.0381 | 0.0133 | -0.0381 | 0.0302 |
| ethnicity | Ethnicity Not Available | Joint | 54597 | 0.4633 | -0.2094 | 0.6887 | 0.2288 | -0.0540 | -0.2288 | 0.0498 |
| ethnicity | Free Form Text Only | Joint | 224 | 0.4911 | -0.1816 | 0.7300 | 0.0746 | 0.0125 | -0.0746 | 0.0363 |
| sex | Sex Not Available | Joint | 31220 | 0.3772 | -0.3079 | 0.5505 | 0.3597 | -0.0682 | -0.3597 | 0.0753 |
| income_band | Missing/Unknown | Income > $150K | 21099 | 0.1004 | -0.6007 | 0.1432 | 0.8112 | -0.2236 | -0.8112 | 0.2806 |
| income_band | Income <= $50K | Income > $150K | 19785 | 0.3382 | -0.3629 | 0.4823 | 0.2760 | -0.0821 | -0.2760 | 0.0238 |

## 7. Interpretation of Metrics

- **Approval-rate difference** compares each group approval rate with the reference group approval rate.
- **Disparate impact ratio** is the group approval rate divided by the reference group approval rate. Values below 0.80 are commonly used as a screening signal for adverse impact.
- **False-negative-rate gap** identifies groups more likely to be incorrectly classified as not approved when the true target is approved.
- **False-positive-rate gap** identifies groups more likely to be incorrectly classified as approved when the true target is not approved.
- **Equal opportunity difference** compares true-positive rates by group.
- **Calibration error** compares mean predicted probability with observed approval rate within each group.

## 8. Validation Conclusion

The fairness screen identified one or more group-level disparities that require governance review before the model can be considered suitable for high-stakes deployment. Recommended actions include feature review, threshold sensitivity analysis, reject-inference assessment, subgroup stability testing, and challenger-model comparison.

## 9. Limitations

- This script performs statistical fairness screening; it does not establish legal compliance.
- Protected-group variables may include missing, unavailable, or self-reported categories that require careful interpretation.
- Small groups may produce unstable error-rate estimates.
- Fairness results are sensitive to target construction, classification threshold, sample design, and model-feature choices.

## 10. Output Files

- Fairness metrics table: `C:/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/AssuranceTwin AI - Model Validation Governance/assurancetwin-ai-model-validation-governance/reports/tables/fairness_metrics.csv`
- Fairness comparison figure: `C:/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/AssuranceTwin AI - Model Validation Governance/assurancetwin-ai-model-validation-governance/reports/figures/fairness_group_comparison.png`
- Fairness validation report: `C:/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/AssuranceTwin AI - Model Validation Governance/assurancetwin-ai-model-validation-governance/reports/validation/fairness_validation_report.md`
