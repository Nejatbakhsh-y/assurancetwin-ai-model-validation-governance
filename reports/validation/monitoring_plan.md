# Monitoring and Drift Simulation Plan

## 1. Purpose
This document defines an ongoing monitoring framework for the HMDA approval model used in this repository. The objective is to detect material deterioration after model development, including data drift, prediction drift, performance drift, fairness drift, and calibration drift.

## 2. Monitoring Design
- **Target variable:** `approved`
- **Model used for monitoring:** `models\champion_model.pkl`
- **Existing model artifact used:** `True`
- **Period source:** Deterministic pseudo-quarter simulation because no usable timestamp was found
- **Baseline train periods:** 2024Q1_sim_1, 2024Q2_sim_2, 2024Q3_sim_3, 2024Q4_sim_4
- **Monitoring periods:** 2024Q1_sim_5, 2024Q2_sim_6, 2024Q3_sim_7, 2024Q4_sim_8
- **Synthetic periods used:** `True`
- **Limitation:** The processed public HMDA file did not provide a usable transaction-level timestamp. The script therefore created deterministic pseudo-periods. These periods support governance simulation, but they should be replaced with true monthly or quarterly production cohorts in a deployed system.

## 3. Metrics
The monitoring process computes the following controls:
- **Population Stability Index:** drift in the model score distribution versus the baseline period.
- **Characteristic Stability Index:** drift in selected input-feature distributions versus the baseline period.
- **Data drift:** maximum and average CSI across monitored characteristics.
- **Prediction drift:** PSI of predicted probabilities and movement in the model approval rate.
- **Performance drift:** AUC, accuracy, balanced accuracy, precision, recall, F1, Brier score, false-positive rate, and false-negative rate.
- **Fairness drift:** group-level approval-rate gap, false-negative-rate gap, and false-positive-rate gap.
- **Calibration drift:** expected calibration error and Brier score movement versus baseline.

## 4. Thresholds and Escalation
| Metric family | Green | Amber | Red |
|---|---:|---:|---:|
| PSI or CSI | < 0.10 | 0.10 to < 0.25 | >= 0.25 |
| AUC change vs. baseline | > -0.03 | -0.05 to -0.03 | <= -0.05 |
| Brier score increase | < 0.015 | 0.015 to < 0.030 | >= 0.030 |
| ECE increase | < 0.020 | 0.020 to < 0.040 | >= 0.040 |
| Group approval-rate gap | < 0.10 | 0.10 to < 0.15 | >= 0.15 |

Recommended escalation:
- **Green:** Continue scheduled monitoring.
- **Amber:** Perform analyst review, feature-level diagnosis, and business-context assessment.
- **Red:** Open a model-risk issue, notify the model owner and validator, assess customer impact, and consider recalibration, challenger review, policy override, or model retirement.

## 5. Monitoring Results Summary
- Green monitoring periods: **0**
- Amber monitoring periods: **0**
- Red monitoring periods: **4**
- Largest characteristic drift: **loan_amount** with CSI **22.6398** during period **2024Q4_sim_8**.

## 6. Fairness Monitoring Scope
The following group variables were tested for fairness drift:
- `derived_race`
- `derived_ethnicity`
- `derived_sex`
- `income_band`

## 7. Governance Alignment
This monitoring plan supports model lifecycle governance by defining measurable post-development controls, thresholds, escalation criteria, documentation artifacts, and review evidence. For high-risk AI use cases, this type of monitoring also supports post-market or post-deployment control expectations by collecting and analyzing performance and compliance evidence over time.

## 8. Generated Evidence
- `reports\tables\drift_monitoring_summary.csv`
- `reports\figures\drift_dashboard_plot.png`
- `reports\validation\monitoring_plan.md`

## 9. Implementation Notes
- Number of model features considered: **30**
- The baseline period is used as the reference distribution for PSI and CSI.
- The script is intentionally conservative: when a champion artifact is incompatible, it trains a fallback monitoring model and records that fact.
- Production implementation should replace pseudo-periods, if used, with actual monthly or quarterly production cohorts.
