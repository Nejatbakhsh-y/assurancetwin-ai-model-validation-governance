# Independent Model Validation Report

## Purpose

This report simulates the work of an independent model-validation team. The validation process loads the saved champion and challenger models, evaluates them on a held-out validation sample, and compares predictive, classification, calibration, and decision-risk metrics.

The validation step does not retrain the models and does not tune hyperparameters. It is intended to represent independent post-development review rather than model-development activity.

## Validation Design

- Source dataset: `data\processed\hmda_modeling_dataset.csv`
- Target variable: `approved`
- Validation split: `20%`
- Random state: `42`
- Calibration bins: `10`

## Model Loading Review

| Model | Model File | Saved Object Type | Saved Object Keys | Threshold |
|---|---|---|---|---:|
| Champion Model | `models\champion_model.pkl` | dict | model; model_name; selection_role; selection_logic; target_column; feature_columns; numeric_features; categorical_features; removed_columns; performance_summary | 0.5000 |
| Challenger Model | `models\challenger_model.pkl` | dict | model; model_name; selection_role; selection_logic; target_column; feature_columns; numeric_features; categorical_features; removed_columns; performance_summary | 0.5000 |

## Validation Metrics Summary

| Model | AUC | Accuracy | Precision | Recall | F1 | Balanced Accuracy | Brier Score | Calibration Error | Approval Rate | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Champion Model | 0.8854 | 0.8313 | 0.8390 | 0.8741 | 0.8562 | 0.8238 | 0.1250 | 0.0523 | 0.5985 | 0.2265 | 0.1259 |
| Challenger Model | 0.9573 | 0.9086 | 0.8817 | 0.9711 | 0.9242 | 0.8976 | 0.0699 | 0.0150 | 0.6326 | 0.1758 | 0.0289 |

## Confusion Matrices

### Champion Model

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 19,095 | 5,590 |
| Actual 1 | 4,195 | 29,121 |

### Challenger Model

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 20,345 | 4,340 |
| Actual 1 | 964 | 32,352 |

## Independent Validation Findings

- Best discriminatory performance by AUC: **Challenger Model**.
- Best class-balance performance by balanced accuracy: **Challenger Model**.
- Best probability accuracy by Brier Score: **Challenger Model**.
- Best calibration by Expected Calibration Error: **Challenger Model**.
- Lowest false-positive rate: **Challenger Model**.
- Lowest false-negative rate: **Challenger Model**.

## Governance Interpretation

A model with the highest predictive performance is not automatically the best governed model. Independent validation must also consider calibration quality, approval-rate behavior, false-positive risk, false-negative risk, operational consequences, and documentation quality.

For model-risk governance, the preferred model should be selected using a documented trade-off among predictive strength, calibration, stability, interpretability, and business-use risk. This is especially important for credit, insurance, lending, and other regulated decision environments.

## Validation Conclusion

This independent validation step provides evidence that model selection should not rely only on AUC or accuracy. The challenger model may be preferable from a governance perspective if it has stronger calibration, lower error asymmetry, lower approval-rate distortion, or better documented risk controls. Conversely, the champion model remains supportable only if its predictive gains justify its validation, monitoring, and governance risks.

## Output Files

- Metrics table: `reports\tables\independent_validation_metrics.csv`
- Validation report: `reports\validation\model_validation_report.md`
