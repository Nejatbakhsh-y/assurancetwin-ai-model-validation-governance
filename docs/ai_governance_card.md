# AI Governance Card

## Governance Summary

| Field | Description |
|---|---|
| Project | AssuranceTwin AI - Model Validation Governance |
| System Type | AI/ML model validation and governance framework |
| Model Risk Tier | High / Material model risk for demonstration purposes |
| Generated Date | June 26, 2026 |
| Evidence Completeness | 11 of 11 expected evidence files available (100.00%) |
| AssuranceTwin Score | 73.37/100 |
| Approval Recommendation | Conditional Approval |

## Intended Use

The system is intended to support model validation and AI governance analysis for a supervised mortgage approval classification model. It provides structured documentation, validation evidence, risk assessment, and approval support for a model risk committee or AI governance review process.

Permitted use cases include:

- Model validation demonstration.
- AI governance documentation.
- Champion-challenger model review.
- Fairness, calibration, explainability, stress, and drift analysis.
- Portfolio artifact for model risk management and AI governance work.
- Research and educational use.

## Out-of-Scope Use

The system is not approved for:

- Actual mortgage underwriting.
- Automated approval or denial of real credit applications.
- Consumer-facing decisioning.
- Pricing or adverse action notice generation.
- Legal or regulatory compliance certification.
- Production deployment without additional enterprise review.
- Use in geographies, products, populations, or time periods not independently validated.

## Data Limitations

The project uses a cleaned HMDA-based modeling dataset. HMDA is valuable for public mortgage market analysis, but it does not contain the full set of variables required for actual underwriting. Historical action outcomes may reflect lender policies, macroeconomic conditions, applicant behavior, market structure, institutional practices, and historical disparities.

Key data limitations include:

- Observational data rather than randomized experimental data.
- Potential historical bias in the target variable.
- Limited underwriting feature coverage.
- Possible missingness, coding differences, or reporting inconsistencies.
- Possible time-period and geography-specific patterns.
- Limited ability to infer causal relationships.
- Need for ongoing schema and data-quality checks.

## Protected Attribute Handling

Protected and governance-relevant attributes may be used for validation, fairness testing, bias analysis, monitoring, and governance reporting. Their use should be controlled, documented, and reviewed by appropriate legal, compliance, privacy, and model risk stakeholders.

Protected attributes should not be used for direct production decisioning unless there is explicit approval and a documented legal and compliance basis. Any use of protected attributes should be subject to access controls, audit logging, minimization principles, and clear documentation.

## Known Model Risks

| Risk Area | Governance Concern | Required Control |
|---|---|---|
| Performance Risk | The model may perform well overall but poorly in specific subgroups or time periods. | Independent validation, subgroup testing, and monitoring. |
| Calibration Risk | Predicted probabilities may not represent reliable empirical likelihoods. | Calibration curve review, Brier score, calibration error, and recalibration triggers. |
| Fairness Risk | Approval rates or error rates may differ materially across protected or governance-relevant groups. | Fairness metrics, group-level calibration, documented thresholds, and escalation. |
| Drift Risk | Data, predictions, performance, calibration, or fairness may change over time. | PSI, CSI, performance drift, prediction drift, fairness drift, and calibration drift monitoring. |
| Explainability Risk | Feature importance and SHAP-style explanations may be unstable across groups, time, or models. | Explanation stability testing and documented interpretation controls. |
| Robustness Risk | The model may be sensitive to macroeconomic or borrower-profile stress scenarios. | Stress testing and scenario analysis. |
| Documentation Risk | Missing documentation can prevent effective review, audit, and approval. | Required governance artifacts and evidence inventory. |
| Misuse Risk | The model may be misinterpreted as production-ready or used outside scope. | Explicit use restrictions and human oversight requirements. |

## Monitoring Requirements

The following monitoring controls are required before any controlled deployment:

1. Population Stability Index monitoring.
2. Characteristic Stability Index monitoring.
3. Input missingness and schema monitoring.
4. Prediction distribution monitoring.
5. AUC, accuracy, recall, precision, F1, and balanced accuracy monitoring.
6. Brier score and calibration error monitoring.
7. Group-level fairness monitoring.
8. Approval-rate and error-rate disparity monitoring.
9. Explanation stability monitoring.
10. Stress scenario refresh.
11. Documented thresholds for green, amber, and red status.
12. Assigned owners for monitoring review and remediation.
13. Escalation path to model risk management and governance committee.
14. Periodic independent validation refresh.

## Human Oversight Requirements

Human oversight is required at the following points:

- Model approval and conditional approval decisions.
- Review of validation findings.
- Review of fairness and bias results.
- Review of stress testing and drift findings.
- Approval of monitoring thresholds.
- Override and exception review.
- Remediation prioritization.
- Retirement, redevelopment, or recalibration decisions.

The system should support expert review rather than replace accountable human judgment.

## Documentation Status

| Documentation Item | Status |
|---|---|
| Model Inventory | Available |
| Champion-Challenger Performance Evidence | Available |
| Independent Validation Metrics | Available |
| Calibration Evidence | Available |
| Fairness Evidence | Available |
| Explainability Evidence | Available |
| Stress Testing Evidence | Available |
| Drift Monitoring Evidence | Available |
| AssuranceTwin Scorecard | Available |
| Model Card | Generated by this script |
| AI Governance Card | Generated by this script |
| Validation Checklist | Generated by this script |

## Approval Recommendation

**Recommendation:** Conditional Approval

**Rationale:** The AssuranceTwin score is 73.37/100. The model may be considered for limited or controlled use only after remediation items are documented, owners are assigned, monitoring controls are approved, and residual risks are accepted by the appropriate governance authority.

## Missing Evidence

No expected evidence files are missing.

## Governance Conclusion

The model should be evaluated as a high-governance-use AI/ML artifact. The project demonstrates a complete model risk management lifecycle, but any real-world use would require additional enterprise controls, legal review, compliance review, privacy review, security review, business-owner sign-off, and independent model validation approval.
