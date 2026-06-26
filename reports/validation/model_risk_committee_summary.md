# Model Risk Committee Summary

## Project Name

AssuranceTwin AI — Model Validation and AI Governance Framework

## Committee Review Purpose

This summary is intended for model risk committee review. It consolidates the key findings from model development, independent validation, fairness testing, calibration review, stress testing, explainability review, drift monitoring, and AI governance documentation.

## Model Use Case

The project evaluates a mortgage lending decision model using public HMDA-style data. The model is designed as a controlled demonstration of financial-services AI governance, not as a production lending system.

## Business Context

The model supports binary classification of mortgage application outcomes. Because the use case is high-stakes and financial in nature, the model requires strong validation, fairness review, monitoring controls, and documentation before any production consideration.

## Validation Scope

The validation review covers:

- Model inventory completeness
- Champion and challenger comparison
- Predictive performance
- Calibration quality
- Fairness and bias metrics
- Explainability and feature-importance stability
- Stress testing
- Drift monitoring
- Documentation completeness
- Monitoring readiness

## Key Governance Principle

The best predictive model is not automatically the best governed model. Committee review should consider performance together with fairness, calibration, transparency, robustness, monitoring design, and documented limitations.

## Main Validation Findings

The project includes a structured validation workflow with performance metrics, confusion-matrix analysis, approval-rate analysis, calibration testing, group-level fairness analysis, stress testing, and monitoring simulation.

The validation package is designed to show how an independent validation team would assess a model before recommending approval, conditional approval, or rejection.

## Fairness and Responsible AI Review

The fairness review evaluates whether model outcomes differ meaningfully across borrower and geography-related groups. The review includes approval-rate differences, disparate impact ratios, false-positive-rate gaps, false-negative-rate gaps, equal opportunity differences, and group-level calibration.

Any material group-level difference should be treated as a governance issue requiring additional review before production use.

## Calibration Review

Calibration is reviewed because financial-services models often rely on probability outputs, not only classification labels. Poor calibration can create incorrect risk ranking, distorted threshold decisions, and unreliable monitoring triggers.

## Stress Testing Review

Stress testing evaluates whether the model remains stable under adverse or shifted scenarios. The tested scenarios include income shock, loan amount increase, LTV increase, missing-data shock, minority-tract distribution shift, out-of-time validation, and recession-like synthetic stress.

## Monitoring and Drift Review

The monitoring framework includes data drift, prediction drift, performance drift, fairness drift, and calibration drift. This supports lifecycle governance and ongoing oversight after deployment.

## Documentation Review

The repository includes:

- Model card
- AI governance card
- Validation checklist
- Independent validation report
- Fairness validation report
- Monitoring plan
- Final project report
- Streamlit governance dashboard

These artifacts provide committee-level traceability and support reproducible model governance.

## Recommendation

The model should be treated as conditionally approvable for demonstration and research purposes only.

Production approval would require:

- Business-owner signoff
- Legal and compliance review
- Fair lending review
- Data lineage review
- Threshold governance
- Human oversight procedures
- Ongoing monitoring controls
- Periodic revalidation
- Formal model-risk committee approval

## Final Committee Position

Conditional approval for portfolio demonstration and GitHub presentation.

Not approved for production lending use without additional institutional controls, legal review, compliance review, and formal model-risk governance approval.
