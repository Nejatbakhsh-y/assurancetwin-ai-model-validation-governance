# Model Inventory Template

Generated on: 2026-06-26

## Purpose

This document provides a governance-ready model inventory template for the HMDA model validation and AI governance project.

A model inventory is used to document model ownership, business use, risk tiering, validation status, monitoring expectations, approval status, and known limitations across the model lifecycle.

---

## Inventory Fields

| Field | Description |
|---|---|
| model_id | Unique model identifier used for tracking and governance. |
| model_name | Descriptive model name. |
| model_type | Modeling approach, such as binary classification, regression, scorecard, or AI/ML model. |
| business_use | Intended use of the model and decision context. |
| risk_tier | Risk rating such as Low, Moderate, or High. |
| training_data | Source data used to develop the model. |
| target_variable | Outcome variable predicted by the model. |
| owner | Business or technical owner responsible for the model. |
| validator | Independent reviewer or validation function. |
| materiality | Business, regulatory, fairness, financial, or operational importance. |
| validation_status | Current validation state, such as Not started, In progress, Completed, or Requires remediation. |
| monitoring_frequency | Expected frequency of model performance, drift, and fairness monitoring. |
| approval_status | Governance approval state, such as Not approved, Conditionally approved, or Approved. |
| known_limitations | Documented limitations, exclusions, assumptions, and use restrictions. |

---

## Initial Inventory Entry

| Field | Value |
|---|---|
| model_id | HMDA-APPROVAL-001 |
| model_name | HMDA Mortgage Application Approval Classifier |
| model_type | Supervised binary classification |
| business_use | Research and model-validation demonstration using public HMDA loan/application records to estimate mortgage application approval status. |
| risk_tier | High |
| training_data | data/processed/hmda_modeling_dataset.csv; derived from data/raw/hmda_lar_nj_2024.csv |
| target_variable | approved |
| owner | Yousef Nejatbakhsh |
| validator | Independent model validation reviewer / project validator |
| materiality | High; the project concerns lending outcomes, approval classification, fairness review, and governance documentation. |
| validation_status | Not started |
| monitoring_frequency | Quarterly after model deployment; ad hoc after material data or methodology changes |
| approval_status | Not approved; development-stage inventory entry |
| known_limitations | Uses public HMDA data; does not include all underwriting variables, credit scores, debt-to-income details, internal lender policies, full applicant context, or post-application operational information. Results are for research and governance demonstration only, not production credit decisioning. |

---

## Governance Use

This inventory should be updated whenever:

1. A new model candidate is created.
2. A model is trained, validated, approved, rejected, or retired.
3. Material model changes occur.
4. Training data, target definition, feature set, or methodology changes.
5. Monitoring identifies drift, instability, fairness concerns, or performance degradation.

---

## Lifecycle Status Definitions

| Status | Meaning |
|---|---|
| Not started | Model validation has not begun. |
| In progress | Validation review is underway. |
| Completed | Validation has been completed with documented findings. |
| Requires remediation | Issues were identified and must be corrected before approval. |
| Approved | Model has passed governance review for its intended use. |
| Retired | Model is no longer active or recommended for use. |

---

## Notes

This inventory is part of the broader model-risk management documentation package for the project. It should be used together with:

- dataset documentation
- data-quality checks
- model development documentation
- validation report
- fairness and bias assessment
- monitoring plan
- model card
