"""
Step P — Generate the AI Governance Card and Model Card

Creates:
    docs/model_card.md
    docs/ai_governance_card.md
    docs/validation_checklist.md

This script is designed for the AssuranceTwin AI - Model Validation Governance project.

It reads available validation evidence from prior project steps and generates
professional model governance documentation suitable for a model risk committee package.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCS_DIR = REPO_ROOT / "docs"
TABLES_DIR = REPO_ROOT / "reports" / "tables"
VALIDATION_DIR = REPO_ROOT / "reports" / "validation"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL_CARD = DOCS_DIR / "model_card.md"
OUTPUT_GOVERNANCE_CARD = DOCS_DIR / "ai_governance_card.md"
OUTPUT_CHECKLIST = DOCS_DIR / "validation_checklist.md"


# ---------------------------------------------------------------------
# Expected evidence files
# ---------------------------------------------------------------------

EXPECTED_INPUTS = {
    "Model Inventory": TABLES_DIR / "model_inventory.csv",
    "Model Performance Summary": TABLES_DIR / "model_performance_summary.csv",
    "Independent Validation Metrics": TABLES_DIR / "independent_validation_metrics.csv",
    "Fairness Metrics": TABLES_DIR / "fairness_metrics.csv",
    "Calibration Summary": TABLES_DIR / "calibration_summary.csv",
    "Stress Test Results": TABLES_DIR / "stress_test_results.csv",
    "Drift Monitoring Summary": TABLES_DIR / "drift_monitoring_summary.csv",
    "Explanation Stability": TABLES_DIR / "explanation_stability.csv",
    "AssuranceTwin Scorecard": TABLES_DIR / "assurancetwin_scorecard.csv",
    "Modeling Dataset Summary": TABLES_DIR / "modeling_dataset_summary.csv",
    "Target Distribution": TABLES_DIR / "modeling_target_distribution.csv",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV safely. Return an empty DataFrame if missing or unreadable."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return pd.DataFrame()


def clean_text(value) -> str:
    """Convert values to clean markdown-safe text."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    text = text.replace("\n", "<br>")
    text = text.replace("|", "\\|")
    return text


def markdown_table_from_rows(headers: List[str], rows: List[List[object]]) -> str:
    """Create a markdown table from headers and rows."""
    if not rows:
        return "_No data available._"

    header_line = "| " + " | ".join(clean_text(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"

    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(clean_text(x) for x in row) + " |")

    return "\n".join([header_line, sep_line] + body_lines)


def dataframe_to_markdown(
    df: pd.DataFrame,
    max_rows: int = 12,
    max_cols: int = 8,
) -> str:
    """Convert a DataFrame to a compact markdown table."""
    if df is None or df.empty:
        return "_No data available._"

    compact = df.copy()

    if compact.shape[1] > max_cols:
        compact = compact.iloc[:, :max_cols]

    if compact.shape[0] > max_rows:
        compact = compact.head(max_rows)

    compact = compact.fillna("")
    headers = list(compact.columns)
    rows = compact.values.tolist()

    return markdown_table_from_rows(headers, rows)


def evidence_status(path: Path) -> str:
    """Return evidence availability status."""
    return "Available" if path.exists() else "Missing"


def load_all_inputs() -> Dict[str, pd.DataFrame]:
    """Load all expected evidence files."""
    return {name: read_csv_safe(path) for name, path in EXPECTED_INPUTS.items()}


def normalize_column_name(column_name: str) -> str:
    """Normalize a column name for flexible matching."""
    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find a column using flexible normalized matching."""
    if df.empty:
        return None

    normalized_map = {normalize_column_name(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized_map:
            return normalized_map[key]

    return None


def get_inventory_value(
    inventory_df: pd.DataFrame,
    column_candidates: List[str],
    default: str,
) -> str:
    """Extract first-row value from model inventory if available."""
    if inventory_df.empty:
        return default

    col = find_first_existing_column(inventory_df, column_candidates)
    if col is None:
        return default

    value = inventory_df.iloc[0].get(col, default)

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    value = str(value).strip()
    return value if value else default


def normalize_score(score: float) -> float:
    """
    Normalize a score to a 0-100 scale.

    If the value is between 0 and 1, treat it as a proportion.
    Otherwise, treat it as already being on a 0-100 scale.
    """
    if score <= 1.0:
        score = score * 100.0

    return round(max(0.0, min(100.0, score)), 2)


def extract_assurancetwin_score(scorecard_df: pd.DataFrame) -> Optional[float]:
    """
    Extract the final AssuranceTwin score from reports/tables/assurancetwin_scorecard.csv.

    Supported scorecard structures include:

    1. A final row labeled:
       - AssuranceTwin Score
       - Assurance Twin Score
       - Overall Score
       - Final Score
       - Total Score

    2. Columns such as:
       - governance_component
       - component
       - metric
       - score
       - component_score
       - weighted_score

    3. A weighted component table where final score must be computed as:
       sum(weight * component_score)

    This corrected function handles the current project scorecard where the final row is:
       AssuranceTwin Score | 1.00 | 73.37 | 73.37 | Acceptable
    """
    if scorecard_df is None or scorecard_df.empty:
        return None

    df = scorecard_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    normalized_columns = {normalize_column_name(c): c for c in df.columns}

    # Direct final-score columns.
    direct_score_columns = [
        "assurancetwin_score",
        "assurance_twin_score",
        "overall_score",
        "final_score",
        "total_score",
        "score_total",
    ]

    for candidate in direct_score_columns:
        if candidate in normalized_columns:
            col = normalized_columns[candidate]
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                return normalize_score(float(values.iloc[0]))

    # Label/component column candidates.
    label_col = None
    label_candidates = [
        "governance_component",
        "component",
        "metric",
        "score_component",
        "section",
        "category",
        "score_name",
        "name",
    ]

    for candidate in label_candidates:
        if candidate in normalized_columns:
            label_col = normalized_columns[candidate]
            break

    # Score-like columns.
    score_col = None
    score_candidates = [
        "component_score",
        "score",
        "value",
        "raw_score",
        "final_score",
        "overall_score",
    ]

    for candidate in score_candidates:
        if candidate in normalized_columns:
            score_col = normalized_columns[candidate]
            break

    weighted_col = None
    weighted_candidates = [
        "weighted_score",
        "weighted_component_score",
        "weighted_value",
    ]

    for candidate in weighted_candidates:
        if candidate in normalized_columns:
            weighted_col = normalized_columns[candidate]
            break

    # Best case: find final row and extract component_score/score first.
    if label_col is not None:
        labels = df[label_col].astype(str).str.lower().str.strip()

        final_score_mask = labels.str.contains(
            "assurancetwin score|assurance twin score|overall score|final score|total score",
            regex=True,
            na=False,
        )

        if final_score_mask.any():
            if score_col is not None:
                values = pd.to_numeric(df.loc[final_score_mask, score_col], errors="coerce").dropna()
                if not values.empty:
                    return normalize_score(float(values.iloc[0]))

            if weighted_col is not None:
                values = pd.to_numeric(df.loc[final_score_mask, weighted_col], errors="coerce").dropna()
                if not values.empty:
                    return normalize_score(float(values.iloc[0]))

    # Fallback: compute weighted score from component rows, excluding final total row.
    weight_col = None
    weight_candidates = [
        "weight",
        "component_weight",
    ]

    for candidate in weight_candidates:
        if candidate in normalized_columns:
            weight_col = normalized_columns[candidate]
            break

    if score_col is not None and weight_col is not None:
        work = df.copy()

        if label_col is not None:
            labels = work[label_col].astype(str).str.lower().str.strip()
            total_mask = labels.str.contains(
                "assurancetwin score|assurance twin score|overall score|final score|total score",
                regex=True,
                na=False,
            )
            work = work.loc[~total_mask].copy()

        scores = pd.to_numeric(work[score_col], errors="coerce")
        weights = pd.to_numeric(work[weight_col], errors="coerce")

        valid = scores.notna() & weights.notna()

        if valid.any():
            raw_score = float((scores[valid] * weights[valid]).sum())

            # If weights are percentages rather than proportions, convert.
            if float(weights[valid].sum()) > 1.5:
                raw_score = raw_score / 100.0

            return normalize_score(raw_score)

    return None


def approval_recommendation(score: Optional[float]) -> Tuple[str, str]:
    """Return approval recommendation and rationale."""
    if score is None:
        return (
            "Conditional Approval Pending Score Extraction",
            (
                "The final approval recommendation remains conditional because the "
                "AssuranceTwin score could not be extracted from the scorecard. "
                "The evidence files may be complete, but the scorecard structure should "
                "be reviewed before final committee approval."
            ),
        )

    if score >= 80:
        return (
            "Approval with Ongoing Monitoring",
            (
                f"The AssuranceTwin score is {score:.2f}/100, which supports approval "
                "subject to documented monitoring, periodic validation refresh, and "
                "continued performance, calibration, fairness, drift, robustness, and "
                "explainability surveillance."
            ),
        )

    if score >= 65:
        return (
            "Conditional Approval",
            (
                f"The AssuranceTwin score is {score:.2f}/100. The model may be considered "
                "for limited or controlled use only after remediation items are documented, "
                "owners are assigned, monitoring controls are approved, and residual risks "
                "are accepted by the appropriate governance authority."
            ),
        )

    return (
        "Rejection / Not Approved for Production Use",
        (
            f"The AssuranceTwin score is {score:.2f}/100. The model does not meet the "
            "minimum governance threshold for production use. Remediation, redevelopment, "
            "or additional validation is required before approval."
        ),
    )


def completeness_summary() -> Tuple[int, int, float]:
    """Calculate evidence completeness."""
    total = len(EXPECTED_INPUTS)
    available = sum(1 for path in EXPECTED_INPUTS.values() if path.exists())
    pct = round(100.0 * available / total, 2) if total else 0.0
    return available, total, pct


def source_files_missing() -> List[str]:
    """Return names of missing expected evidence files."""
    return [name for name, path in EXPECTED_INPUTS.items() if not path.exists()]


def evidence_inventory_markdown() -> str:
    """Create markdown evidence inventory table."""
    rows = []

    for name, path in EXPECTED_INPUTS.items():
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        rows.append([name, relative_path, evidence_status(path)])

    return markdown_table_from_rows(
        ["Evidence Item", "Expected File", "Status"],
        rows,
    )


# ---------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------

def generate_model_card(data: Dict[str, pd.DataFrame]) -> str:
    inventory = data["Model Inventory"]
    performance = data["Model Performance Summary"]
    independent = data["Independent Validation Metrics"]
    fairness = data["Fairness Metrics"]
    calibration = data["Calibration Summary"]
    stress = data["Stress Test Results"]
    drift = data["Drift Monitoring Summary"]
    explanation = data["Explanation Stability"]
    scorecard = data["AssuranceTwin Scorecard"]
    dataset_summary = data["Modeling Dataset Summary"]
    target_distribution = data["Target Distribution"]

    score = extract_assurancetwin_score(scorecard)
    recommendation, recommendation_rationale = approval_recommendation(score)

    model_name = get_inventory_value(
        inventory,
        ["model_name", "name"],
        "AssuranceTwin AI Champion Model",
    )

    model_type = get_inventory_value(
        inventory,
        ["model_type", "type", "algorithm"],
        "Supervised binary classification model",
    )

    business_use = get_inventory_value(
        inventory,
        ["business_use", "use_case", "purpose"],
        "Governed prediction of mortgage application approval outcome for model validation demonstration.",
    )

    risk_tier = get_inventory_value(
        inventory,
        ["risk_tier", "risk_level", "materiality"],
        "High / Material model risk due to potential credit-decision relevance.",
    )

    target_variable = get_inventory_value(
        inventory,
        ["target_variable", "target", "outcome"],
        "approved",
    )

    owner = get_inventory_value(
        inventory,
        ["owner", "model_owner"],
        "Model Development Owner",
    )

    validator = get_inventory_value(
        inventory,
        ["validator", "validation_owner"],
        "Independent Model Validation Function",
    )

    generated_date = datetime.now().strftime("%B %d, %Y")
    available, total, completeness = completeness_summary()

    score_display = "Not extracted" if score is None else f"{score:.2f}/100"

    markdown = f"""# Model Card

## Model Identification

| Field | Description |
|---|---|
| Model Name | {clean_text(model_name)} |
| Model Type | {clean_text(model_type)} |
| Project | AssuranceTwin AI - Model Validation Governance |
| Target Variable | {clean_text(target_variable)} |
| Business Use | {clean_text(business_use)} |
| Risk Tier | {clean_text(risk_tier)} |
| Model Owner | {clean_text(owner)} |
| Independent Validator | {clean_text(validator)} |
| Generated Date | {generated_date} |
| Documentation Completeness | {available} of {total} expected evidence files available ({completeness:.2f}%) |
| AssuranceTwin Score | {score_display} |
| Approval Recommendation | {clean_text(recommendation)} |

## Intended Use

This model is intended for a controlled model validation and AI governance project. It estimates a binary mortgage application approval outcome using a structured HMDA-based modeling dataset. The primary purpose is to demonstrate a complete model risk management workflow, including model inventory, champion-challenger comparison, independent validation, calibration review, fairness testing, explainability review, stress testing, monitoring simulation, and governance documentation.

The model card is designed for use by model developers, independent validators, model risk managers, AI governance reviewers, audit stakeholders, and a model risk committee.

## Out-of-Scope Use

This model should not be used for actual mortgage underwriting, consumer credit decisions, pricing, adverse action notices, regulatory reporting, or automated decision-making affecting real applicants. The data and validation framework support research, demonstration, and governance prototyping only.

The model should not be used outside the documented population, geography, time period, product scope, or data-generating process without additional independent validation.

## Dataset Summary

The model uses a cleaned HMDA-based modeling dataset created in earlier project steps. The target variable is constructed from mortgage application action outcomes. Because HMDA is observational and does not contain all underwriting variables, the model should be interpreted as a governance and validation artifact rather than a production underwriting system.

### Modeling Dataset Summary

{dataframe_to_markdown(dataset_summary)}

### Target Distribution

{dataframe_to_markdown(target_distribution)}

## Model Development Summary

The project compares champion and challenger model candidates. The best predictive model is not automatically treated as the best governed model. Selection should consider predictive performance, calibration, fairness, robustness, drift stability, explainability stability, documentation quality, and monitoring readiness.

### Champion-Challenger Performance Evidence

{dataframe_to_markdown(performance)}

## Independent Validation Results

Independent validation reviews model performance using metrics such as AUC, accuracy, precision, recall, F1, balanced accuracy, Brier score, calibration error, confusion matrix outputs, approval rate, false positive rate, and false negative rate.

### Independent Validation Metrics

{dataframe_to_markdown(independent)}

## Calibration Review

Calibration is reviewed because probability quality is central in financial risk settings. A model with strong discrimination can still produce poorly calibrated probabilities. The validation team should review calibration curves, Brier score, expected calibration error, and group-level calibration.

### Calibration Evidence

{dataframe_to_markdown(calibration)}

## Fairness and Bias Review

Fairness testing evaluates whether model outcomes and errors differ materially across protected or governance-relevant groups. The review should include approval-rate differences, disparate impact ratios, false-negative-rate gaps, false-positive-rate gaps, equal opportunity differences, and group calibration.

Protected attributes should be used for validation, bias testing, monitoring, and governance review. They should not be used for direct production decisioning unless explicitly approved by legal, compliance, and governance stakeholders.

### Fairness Evidence

{dataframe_to_markdown(fairness)}

## Explainability Review

Explainability review evaluates feature importance, permutation importance, SHAP-based explanations where available, group-level explanation differences, and explanation stability across time splits or model variants.

The governance concern is not only whether the model is explainable, but whether explanations remain stable across demographic groups, time periods, and challenger models.

### Explanation Stability Evidence

{dataframe_to_markdown(explanation)}

## Stress Testing and Robustness Review

Stress testing evaluates sensitivity under adverse or shifted scenarios such as income shocks, loan amount increases, LTV increases, missing-data shocks, minority-tract distribution shifts, out-of-time validation, and recession-like synthetic scenarios.

### Stress Testing Evidence

{dataframe_to_markdown(stress)}

## Drift and Monitoring Review

Ongoing monitoring should track population stability, characteristic stability, data drift, prediction drift, performance drift, fairness drift, and calibration drift. Material drift should trigger escalation, root-cause analysis, remediation, and possible model recalibration or redevelopment.

### Drift Monitoring Evidence

{dataframe_to_markdown(drift)}

## AssuranceTwin Score

The AssuranceTwin score combines predictive performance, calibration, fairness, robustness, drift stability, explainability stability, documentation completeness, and monitoring readiness into a single governance-oriented score.

### Scorecard Evidence

{dataframe_to_markdown(scorecard)}

## Approval Recommendation

**Recommendation:** {clean_text(recommendation)}

**Rationale:** {clean_text(recommendation_rationale)}

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

{evidence_inventory_markdown()}

## Documentation Status

This model card was automatically generated from available project evidence. Missing evidence should be completed before the model is submitted for final approval.
"""

    return markdown


def generate_governance_card(data: Dict[str, pd.DataFrame]) -> str:
    scorecard = data["AssuranceTwin Scorecard"]
    score = extract_assurancetwin_score(scorecard)
    recommendation, recommendation_rationale = approval_recommendation(score)

    generated_date = datetime.now().strftime("%B %d, %Y")
    available, total, completeness = completeness_summary()
    missing = source_files_missing()

    missing_text = (
        "No expected evidence files are missing."
        if not missing
        else "; ".join(missing)
    )

    score_display = "Not extracted" if score is None else f"{score:.2f}/100"

    markdown = f"""# AI Governance Card

## Governance Summary

| Field | Description |
|---|---|
| Project | AssuranceTwin AI - Model Validation Governance |
| System Type | AI/ML model validation and governance framework |
| Model Risk Tier | High / Material model risk for demonstration purposes |
| Generated Date | {generated_date} |
| Evidence Completeness | {available} of {total} expected evidence files available ({completeness:.2f}%) |
| AssuranceTwin Score | {score_display} |
| Approval Recommendation | {clean_text(recommendation)} |

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
| Model Inventory | {evidence_status(EXPECTED_INPUTS["Model Inventory"])} |
| Champion-Challenger Performance Evidence | {evidence_status(EXPECTED_INPUTS["Model Performance Summary"])} |
| Independent Validation Metrics | {evidence_status(EXPECTED_INPUTS["Independent Validation Metrics"])} |
| Calibration Evidence | {evidence_status(EXPECTED_INPUTS["Calibration Summary"])} |
| Fairness Evidence | {evidence_status(EXPECTED_INPUTS["Fairness Metrics"])} |
| Explainability Evidence | {evidence_status(EXPECTED_INPUTS["Explanation Stability"])} |
| Stress Testing Evidence | {evidence_status(EXPECTED_INPUTS["Stress Test Results"])} |
| Drift Monitoring Evidence | {evidence_status(EXPECTED_INPUTS["Drift Monitoring Summary"])} |
| AssuranceTwin Scorecard | {evidence_status(EXPECTED_INPUTS["AssuranceTwin Scorecard"])} |
| Model Card | Generated by this script |
| AI Governance Card | Generated by this script |
| Validation Checklist | Generated by this script |

## Approval Recommendation

**Recommendation:** {clean_text(recommendation)}

**Rationale:** {clean_text(recommendation_rationale)}

## Missing Evidence

{clean_text(missing_text)}

## Governance Conclusion

The model should be evaluated as a high-governance-use AI/ML artifact. The project demonstrates a complete model risk management lifecycle, but any real-world use would require additional enterprise controls, legal review, compliance review, privacy review, security review, business-owner sign-off, and independent model validation approval.
"""

    return markdown


def generate_validation_checklist(data: Dict[str, pd.DataFrame]) -> str:
    scorecard = data["AssuranceTwin Scorecard"]
    score = extract_assurancetwin_score(scorecard)
    recommendation, recommendation_rationale = approval_recommendation(score)

    generated_date = datetime.now().strftime("%B %d, %Y")
    score_display = "Not extracted" if score is None else f"{score:.2f}/100"

    rows = [
        [
            "Model inventory completed",
            evidence_status(EXPECTED_INPUTS["Model Inventory"]),
            "Confirm model owner, validator, model type, business use, risk tier, target variable, approval status, and monitoring frequency.",
        ],
        [
            "Training and validation data documented",
            evidence_status(EXPECTED_INPUTS["Modeling Dataset Summary"]),
            "Confirm dataset source, target construction, exclusions, missingness, and known limitations.",
        ],
        [
            "Target distribution reviewed",
            evidence_status(EXPECTED_INPUTS["Target Distribution"]),
            "Confirm class balance and approval outcome definition.",
        ],
        [
            "Champion-challenger models trained",
            evidence_status(EXPECTED_INPUTS["Model Performance Summary"]),
            "Confirm that model selection considered governance quality, not only predictive performance.",
        ],
        [
            "Independent validation metrics generated",
            evidence_status(EXPECTED_INPUTS["Independent Validation Metrics"]),
            "Review AUC, accuracy, precision, recall, F1, balanced accuracy, Brier score, approval rate, FPR, and FNR.",
        ],
        [
            "Calibration reviewed",
            evidence_status(EXPECTED_INPUTS["Calibration Summary"]),
            "Review calibration curve, Brier score, expected calibration error, and group-level calibration.",
        ],
        [
            "Fairness and bias testing completed",
            evidence_status(EXPECTED_INPUTS["Fairness Metrics"]),
            "Review approval-rate differences, disparate impact, FPR gaps, FNR gaps, equal opportunity, and group calibration.",
        ],
        [
            "Explainability reviewed",
            evidence_status(EXPECTED_INPUTS["Explanation Stability"]),
            "Review feature importance, permutation importance, SHAP-style explanations, and stability across groups, time, and models.",
        ],
        [
            "Stress testing completed",
            evidence_status(EXPECTED_INPUTS["Stress Test Results"]),
            "Review income shock, loan amount shock, LTV shock, missing-data shock, minority-tract shift, out-of-time validation, and recession-like scenario.",
        ],
        [
            "Drift monitoring simulated",
            evidence_status(EXPECTED_INPUTS["Drift Monitoring Summary"]),
            "Review PSI, CSI, data drift, prediction drift, performance drift, fairness drift, and calibration drift.",
        ],
        [
            "AssuranceTwin score generated",
            score_display,
            "Confirm final score and component scores across performance, calibration, fairness, robustness, drift, explainability, documentation, and monitoring.",
        ],
        [
            "Model card generated",
            "Generated",
            "Confirm intended use, out-of-scope use, data summary, validation evidence, limitations, monitoring, and approval recommendation.",
        ],
        [
            "AI governance card generated",
            "Generated",
            "Confirm governance use, protected attribute handling, model risks, monitoring requirements, human oversight, and documentation status.",
        ],
        [
            "Human oversight defined",
            "Required",
            "Confirm accountable reviewers, override procedures, escalation path, and committee approval process.",
        ],
        [
            "Production use restriction documented",
            "Required",
            "Confirm that the model is not used for actual underwriting, pricing, adverse action, or consumer-impacting automated decisions.",
        ],
        [
            "Final model risk committee recommendation documented",
            recommendation,
            recommendation_rationale,
        ],
    ]

    checklist_table = markdown_table_from_rows(
        ["Validation Control", "Status", "Review Notes"],
        rows,
    )

    evidence_table = evidence_inventory_markdown()

    markdown = f"""# Validation Checklist

## Document Purpose

This checklist summarizes the evidence required for the AssuranceTwin AI model validation and AI governance package. It is intended for independent validation review, model risk management review, audit readiness, and model risk committee submission.

| Field | Description |
|---|---|
| Project | AssuranceTwin AI - Model Validation Governance |
| Generated Date | {generated_date} |
| AssuranceTwin Score | {score_display} |
| Approval Recommendation | {clean_text(recommendation)} |

## Validation Checklist

{checklist_table}

## Evidence Inventory

{evidence_table}

## Final Review Notes

The model should not be approved solely on predictive performance. Approval should depend on the full governance profile, including calibration quality, fairness behavior, stress-test robustness, drift stability, explanation stability, documentation completeness, and monitoring readiness.

Final approval should be issued only after all material findings are closed or formally accepted as residual risk by the appropriate governance authority.
"""

    return markdown


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    print("Step P — Generating AI Governance Card and Model Card")
    print(f"Repository root: {REPO_ROOT}")

    data = load_all_inputs()

    model_card = generate_model_card(data)
    governance_card = generate_governance_card(data)
    checklist = generate_validation_checklist(data)

    OUTPUT_MODEL_CARD.write_text(model_card, encoding="utf-8")
    OUTPUT_GOVERNANCE_CARD.write_text(governance_card, encoding="utf-8")
    OUTPUT_CHECKLIST.write_text(checklist, encoding="utf-8")

    score = extract_assurancetwin_score(data["AssuranceTwin Scorecard"])
    recommendation, _ = approval_recommendation(score)

    available, total, completeness = completeness_summary()
    missing = source_files_missing()

    print("\nGenerated documentation:")
    print(f"  - {OUTPUT_MODEL_CARD.relative_to(REPO_ROOT)}")
    print(f"  - {OUTPUT_GOVERNANCE_CARD.relative_to(REPO_ROOT)}")
    print(f"  - {OUTPUT_CHECKLIST.relative_to(REPO_ROOT)}")

    print("\nEvidence completeness:")
    print(f"  - {available} of {total} expected evidence files available ({completeness:.2f}%)")

    print("\nAssuranceTwin score:")
    if score is None:
        print("  - Not extracted")
    else:
        print(f"  - {score:.2f}/100")

    print("\nApproval recommendation:")
    print(f"  - {recommendation}")

    if missing:
        print("\nMissing expected evidence files:")
        for item in missing:
            print(f"  - {item}: {EXPECTED_INPUTS[item].relative_to(REPO_ROOT)}")
        print("\nThe markdown files were still generated. Missing evidence is marked in the documents.")
    else:
        print("\nAll expected evidence files are available.")

    print("\nStep P complete.")


if __name__ == "__main__":
    main()