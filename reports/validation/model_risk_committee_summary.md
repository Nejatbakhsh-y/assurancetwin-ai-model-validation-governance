# Model Risk Committee Summary

Generated: 2026-06-26 12:15:37

## Executive Decision View

**AssuranceTwin Score:** 73.37 / 100

**Risk Band:** Moderate model-risk concern

**Recommended Committee Action:** Conditionally approve, subject to remediation tracking and enhanced monitoring.

The AssuranceTwin Score is a composite model-governance score that combines predictive performance, calibration, fairness, stress robustness, drift stability, explainability stability, documentation completeness, and monitoring readiness.

## Scorecard

| Component | Weight | Score | Weighted Points | Status |
|---|---:|---:|---:|---|
| Predictive Performance Score | 0.20 | 94.40 | 18.88 | Strong |
| Calibration Score | 0.15 | 74.50 | 11.17 | Acceptable |
| Fairness Score | 0.20 | 65.84 | 13.17 | Needs remediation |
| Robustness / Stress-Test Score | 0.15 | 43.82 | 6.57 | Insufficient evidence or high risk |
| Drift Stability Score | 0.10 | 68.20 | 6.82 | Needs remediation |
| Explainability Stability Score | 0.10 | 67.63 | 6.76 | Needs remediation |
| Documentation Completeness Score | 0.05 | 100.00 | 5.00 | Strong |
| Monitoring Readiness Score | 0.05 | 100.00 | 5.00 | Strong |
| AssuranceTwin Score | 1.00 | 73.37 | 73.37 | Acceptable |

## Main Strengths

- Documentation Completeness Score: 100.00 (Strong)
- Monitoring Readiness Score: 100.00 (Strong)
- Predictive Performance Score: 94.40 (Strong)

## Main Weaknesses / Remediation Priorities

- Robustness / Stress-Test Score: 43.82 (Insufficient evidence or high risk)
- Fairness Score: 65.84 (Needs remediation)
- Explainability Stability Score: 67.63 (Needs remediation)

## Required Committee Action Items

- Remediate **Fairness Score**. Current score: 65.84.
- Remediate **Robustness / Stress-Test Score**. Current score: 43.82.
- Remediate **Drift Stability Score**. Current score: 68.20.
- Remediate **Explainability Stability Score**. Current score: 67.63.

## Evidence Used

- **Predictive Performance Score**: reports\tables\independent_validation_metrics.csv; reports\tables\model_performance_summary.csv
- **Calibration Score**: reports\tables\calibration_summary.csv; reports\tables\independent_validation_metrics.csv
- **Fairness Score**: reports\tables\fairness_metrics.csv
- **Robustness / Stress-Test Score**: reports\tables\stress_test_results.csv
- **Drift Stability Score**: reports\tables\drift_monitoring_summary.csv
- **Explainability Stability Score**: reports\tables\explanation_stability.csv
- **Documentation Completeness Score**: README.md; docs\model_inventory_template.md; reports\tables\model_inventory.csv; reports\validation\model_validation_report.md; reports\validation\fairness_validation_report.md; reports\validation\stress_testing_report.md; reports\validation\monitoring_plan.md
- **Monitoring Readiness Score**: reports\validation\monitoring_plan.md; reports\tables\drift_monitoring_summary.csv; reports\figures\drift_dashboard_plot.png

## Scoring Formula

```text
AssuranceTwin Score =
0.20 * Predictive Performance
+ 0.15 * Calibration
+ 0.20 * Fairness
+ 0.15 * Robustness / Stress-Test
+ 0.10 * Drift Stability
+ 0.10 * Explainability Stability
+ 0.05 * Documentation Completeness
+ 0.05 * Monitoring Readiness
```

## Interpretation Guide

- **85 to 100:** Strong model-assurance posture.
- **70 to 84.99:** Acceptable, with active monitoring and documented limitations.
- **55 to 69.99:** Elevated model-risk concern; remediation should precede broad deployment.
- **Below 55:** High model-risk concern or insufficient validation evidence.

## Governance Limitation

This score is a structured decision-support artifact, not an automatic model approval mechanism. Final approval should also consider model materiality, regulatory exposure, business use, implementation controls, change management, and independent validation sign-off.
