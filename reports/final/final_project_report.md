# AssuranceTwin AI — Final Project Report

## Executive Summary

AssuranceTwin AI is a model validation and AI governance project designed for high-stakes financial-services machine learning. The project demonstrates how a model can be evaluated not only for predictive performance, but also for calibration, fairness, robustness, drift stability, explainability, documentation quality, and monitoring readiness.

The central governance argument is that the best predictive model is not automatically the best governed model. A strong model-risk-management process must consider performance, transparency, fairness, lifecycle controls, business use, model limitations, and committee-level approval evidence.

## Project Purpose

This repository simulates the workflow of an independent model validation and AI governance team. It uses a mortgage lending dataset to build, compare, validate, and document machine-learning models under a financial risk management framework.

The project supports the following professional areas:

- Model Validation
- AI Governance
- Model Risk Management
- Responsible AI
- Financial Services AI
- Risk Analytics
- GenAI-supported documentation

## Main Deliverables

The final repository includes the following governance and validation artifacts:

- README.md
- docs/model_card.md
- docs/ai_governance_card.md
- docs/validation_checklist.md
- reports/validation/model_validation_report.md
- reports/validation/fairness_validation_report.md
- reports/validation/monitoring_plan.md
- reports/validation/model_risk_committee_summary.md
- dashboard/streamlit_app.py

## Model Development Summary

The project trains and compares multiple model types, including logistic regression, random forest, gradient boosting, and calibrated models. The champion/challenger design demonstrates that model selection should not rely only on AUC or accuracy. Governance quality, calibration, fairness, robustness, and monitoring readiness are also part of the final assessment.

## Independent Validation Summary

The independent validation layer evaluates:

- AUC
- Accuracy
- Precision
- Recall
- F1 score
- Balanced accuracy
- Brier score
- Calibration error
- Approval rate
- False positive rate
- False negative rate
- Confusion matrix

These metrics provide a structured view of both predictive performance and decision-risk behavior.

## Fairness and Bias Review

The fairness review evaluates group-level model behavior across relevant borrower and geography-related attributes. The analysis includes approval-rate difference, disparate impact ratio, false-positive-rate gaps, false-negative-rate gaps, equal opportunity differences, and calibration by group.

This section is central to the project because aggregate model performance can hide group-level harm.

## Calibration Review

The calibration analysis evaluates whether predicted probabilities are reliable. In financial decisioning, probability quality matters because poorly calibrated models can distort risk estimation, approval thresholds, pricing logic, and monitoring triggers.

## Stress Testing Summary

The stress-testing module evaluates model sensitivity under adverse or shifted conditions, including income shock, loan amount increase, LTV increase, missing-data shock, minority-tract distribution shift, out-of-time validation, and recession-like synthetic scenarios.

This gives the project a financial-risk-management structure rather than a generic machine-learning structure.

## Drift Monitoring Summary

The monitoring and drift simulation framework evaluates data drift, prediction drift, performance drift, fairness drift, and calibration drift. This supports model lifecycle governance and ongoing post-deployment oversight.

## AssuranceTwin Score

The AssuranceTwin Score combines multiple governance dimensions into one scorecard:

- Predictive performance
- Calibration
- Fairness
- Robustness
- Drift stability
- Explainability stability
- Documentation completeness
- Monitoring readiness

The score is intended to support committee-level review, not to replace expert judgment.

## Governance Conclusion

The project demonstrates an end-to-end AI governance workflow for financial-services machine learning. It includes model inventory, model card documentation, independent validation, fairness review, calibration review, stress testing, drift monitoring, and model-risk committee documentation.

The repository is suitable for demonstrating practical experience in model validation, AI governance, model risk management, responsible AI, risk analytics, and financial-services AI controls.
