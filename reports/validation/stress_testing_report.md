# Stress Testing Report

## Purpose

This report evaluates whether model outputs remain stable under adverse financial-risk, data-quality, portfolio-mix, and time-shift conditions. The analysis is designed as an independent model-validation stress test, not merely as a predictive-performance exercise.

## Inputs

- Modeling dataset: `data\processed\hmda_modeling_dataset.csv`
- Target variable: `approved`
- Validation sample size: `87,002`
- Probability threshold: `0.50`
- Results table: `reports\tables\stress_test_results.csv`
- Sensitivity figure: `reports\figures\stress_test_model_sensitivity.png`

## Models evaluated

| Model | Source |
|---|---|
| gradient_boosting | `models\challenger_model.pkl` |
| logistic_regression | `models\champion_model.pkl` |

## Stress scenarios

| Scenario | Implementation note |
|---|---|
| baseline | Original validation sample. |
| income_shock | Reduced 'income' by 20 percent. |
| loan_amount_increase | Increased 'loan_amount' by 15 percent. |
| ltv_increase | Increased 'loan_to_value_ratio' by 10 percentage points. |
| minority_tract_distribution_shift | Minority-tract column not found; scenario returned unchanged data. |
| missing_data_shock | Set approximately 15 percent of values to missing for 8 high-impact columns. |
| out_of_time_validation | No usable multi-period time column found; used final 20 percent of ordered records as a proxy out-of-time window. |
| recession_like_synthetic | income -15 percent using 'income'; loan amount +10 percent using 'loan_amount'; LTV +15 percentage points using 'loan_to_value_ratio'; property value -10 percent using 'property_value'; DTI +10 percent using 'debt_to_income_ratio'; 10 percent missingness introduced into stressed financial fields. |

## Baseline model behavior

| Model | AUC | Brier score | Approval rate | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| gradient_boosting | 0.9227 | 0.1270 | 52.85% | 14.62% | 18.83% |
| logistic_regression | 0.7574 | 0.2614 | 36.35% | 12.24% | 45.79% |

## Most sensitive stress results

| Model | Scenario | Approval-rate delta | Probability delta | AUC delta | Brier delta |
|---|---|---:|---:|---:|---:|
| gradient_boosting | recession_like_synthetic | -7.06 pp | -6.41 pp | -0.0158 | 0.0338 |
| gradient_boosting | missing_data_shock | -3.63 pp | -2.93 pp | -0.0127 | 0.0200 |
| gradient_boosting | ltv_increase | -2.64 pp | -3.08 pp | -0.0052 | 0.0122 |
| logistic_regression | missing_data_shock | -4.99 pp | -3.25 pp | -0.0285 | 0.0294 |
| logistic_regression | recession_like_synthetic | -3.35 pp | -2.19 pp | -0.0187 | 0.0196 |
| logistic_regression | loan_amount_increase | 0.02 pp | 0.03 pp | 0.0000 | -0.0001 |

## Governance interpretation

A governed model should not be selected only because it has the highest baseline AUC. A model that shows large adverse changes in approval rate, calibration, false-negative rate, or false-positive rate under stress may create operational, consumer-compliance, or reputational risk. The appropriate governance question is whether model behavior remains explainable, directionally plausible, and stable under realistic adverse conditions.

Key validation considerations:

- **Income shock:** evaluates whether the model becomes excessively restrictive when borrower income deteriorates.
- **Loan amount and LTV shocks:** evaluate sensitivity to higher credit exposure and weaker collateral coverage.
- **Missing-data shock:** evaluates operational resilience when upstream data quality deteriorates.
- **Minority-tract distribution shift:** evaluates whether portfolio composition changes materially alter predicted approvals.
- **Out-of-time validation:** evaluates temporal robustness. If no true multi-period field exists, the script uses a deterministic final-window proxy and explicitly flags this limitation.
- **Recession-like synthetic scenario:** evaluates combined macroeconomic stress, including lower income, weaker collateral, higher loan burden, and missingness.

## Primary sensitivity finding

The largest approval-rate movement was observed for model `gradient_boosting` under scenario `recession_like_synthetic`, with a change of -7.06 pp relative to baseline.

## Model-risk conclusion

The stress-test results should be reviewed jointly with independent validation metrics, fairness testing, calibration analysis, and explanation stability. A model with strong predictive performance but weak stress robustness should not automatically be treated as the preferred governed model.

## Files produced

- `reports\tables\stress_test_results.csv`
- `reports\figures\stress_test_model_sensitivity.png`
- `reports\validation\stress_testing_report.md`
