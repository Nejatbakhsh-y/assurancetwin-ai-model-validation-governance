"""
Step O — Generate the Model Validation Report

This script creates a final Model Validation Committee-style report for the
AssuranceTwin AI — Model Validation and Governance project.

Primary output:
    reports/validation/model_validation_committee_report.md

Secondary output:
    reports/tables/model_validation_report_inputs.csv

The report consolidates outputs from prior steps:
    - model inventory
    - model performance summary
    - independent validation metrics
    - calibration analysis
    - fairness and bias testing
    - explainability stability
    - stress testing
    - monitoring and drift simulation
    - AssuranceTwin scorecard
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
VALIDATION_DIR = REPORTS_DIR / "validation"
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT = VALIDATION_DIR / "model_validation_committee_report.md"
OUTPUT_INPUT_INVENTORY = TABLES_DIR / "model_validation_report_inputs.csv"


# ---------------------------------------------------------------------
# Expected project artifacts
# ---------------------------------------------------------------------

EXPECTED_INPUTS = {
    "Model inventory": TABLES_DIR / "model_inventory.csv",
    "Model performance summary": TABLES_DIR / "model_performance_summary.csv",
    "Independent validation metrics": TABLES_DIR / "independent_validation_metrics.csv",
    "Calibration summary": TABLES_DIR / "calibration_summary.csv",
    "Fairness metrics": TABLES_DIR / "fairness_metrics.csv",
    "Explanation stability": TABLES_DIR / "explanation_stability.csv",
    "Stress test results": TABLES_DIR / "stress_test_results.csv",
    "Drift monitoring summary": TABLES_DIR / "drift_monitoring_summary.csv",
    "AssuranceTwin scorecard": TABLES_DIR / "assurancetwin_scorecard.csv",
    "Clean HMDA modeling dataset": PROCESSED_DATA_DIR / "hmda_modeling_dataset.csv",
    "Fairness validation report": VALIDATION_DIR / "fairness_validation_report.md",
    "Stress testing report": VALIDATION_DIR / "stress_testing_report.md",
    "Monitoring plan": VALIDATION_DIR / "monitoring_plan.md",
    "Champion model": MODELS_DIR / "champion_model.pkl",
    "Challenger model": MODELS_DIR / "challenger_model.pkl",
}

EXPECTED_FIGURES = {
    "ROC curves": FIGURES_DIR / "roc_curves.png",
    "Precision-recall curves": FIGURES_DIR / "precision_recall_curves.png",
    "Calibration curve": FIGURES_DIR / "calibration_curve.png",
    "Fairness group comparison": FIGURES_DIR / "fairness_group_comparison.png",
    "Feature importance": FIGURES_DIR / "feature_importance.png",
    "SHAP summary": FIGURES_DIR / "shap_summary.png",
    "Stress test sensitivity": FIGURES_DIR / "stress_test_model_sensitivity.png",
    "Drift dashboard": FIGURES_DIR / "drift_dashboard_plot.png",
    "AssuranceTwin radar": FIGURES_DIR / "assurancetwin_score_radar.png",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def relative(path: Path) -> str:
    """Return path relative to project root when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def markdown_link(path: Path, label: Optional[str] = None) -> str:
    """Return a Markdown link relative to the report location."""
    label = label or path.name
    try:
        rel = path.relative_to(VALIDATION_DIR).as_posix()
    except ValueError:
        rel = Path("..") / path.relative_to(REPORTS_DIR)
        rel = rel.as_posix()
    return f"[{label}]({rel})"


def figure_link(path: Path) -> str:
    """Return a Markdown image link relative to reports/validation."""
    rel = Path("../figures") / path.name
    return f"![{path.stem}]({rel.as_posix()})"


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Read CSV safely."""
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read {relative(path)} because: {exc}")
        return None


def read_text_if_exists(path: Path, max_chars: int = 4000) -> Optional[str]:
    """Read Markdown/text safely."""
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "\n\n[Excerpt truncated.]"
        return text
    except Exception as exc:
        print(f"Warning: could not read {relative(path)} because: {exc}")
        return None


def format_value(value) -> str:
    """Format values for Markdown tables."""
    if pd.isna(value):
        return ""

    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return ""

    text = str(value)
    text = text.replace("\n", " ").replace("|", "/").strip()

    if len(text) > 120:
        text = text[:117] + "..."

    return text


def dataframe_to_markdown(df: Optional[pd.DataFrame], max_rows: int = 12, max_cols: int = 8) -> str:
    """Convert a DataFrame to a compact Markdown table without requiring tabulate."""
    if df is None:
        return "_Input file not found or could not be read._"

    if df.empty:
        return "_Input file exists but contains no rows._"

    display_df = df.copy()

    if display_df.shape[1] > max_cols:
        display_df = display_df.iloc[:, :max_cols]

    if display_df.shape[0] > max_rows:
        display_df = display_df.head(max_rows)

    columns = [str(c) for c in display_df.columns]

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in display_df.iterrows():
        row_values = [format_value(row[col]) for col in display_df.columns]
        rows.append("| " + " | ".join(row_values) + " |")

    note = ""
    if df.shape[0] > max_rows or df.shape[1] > max_cols:
        note = f"\n\n_Table preview limited to {min(max_rows, df.shape[0])} rows and {min(max_cols, df.shape[1])} columns._"

    return "\n".join([header, separator] + rows) + note


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find a column using case-insensitive exact or partial matching."""
    if df is None or df.empty:
        return None

    normalized = {str(c).lower().strip(): c for c in df.columns}

    for cand in candidates:
        key = cand.lower().strip()
        if key in normalized:
            return normalized[key]

    for cand in candidates:
        key = cand.lower().strip()
        for col in df.columns:
            if key in str(col).lower():
                return col

    return None


def numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to numeric safely."""
    return pd.to_numeric(series, errors="coerce")


def count_csv_rows_and_columns(path: Path) -> Tuple[Optional[int], Optional[int], List[str]]:
    """Count rows and columns without loading the full file into memory."""
    if not path.exists():
        return None, None, []

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            row_count = sum(1 for _ in reader)
            return row_count, len(header), header
    except Exception:
        return None, None, []


def target_distribution(path: Path, target_col: str = "approved") -> Optional[pd.DataFrame]:
    """Compute target distribution for approved if the column exists."""
    if not path.exists():
        return None

    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if target_col not in header:
            return None

        s = pd.read_csv(path, usecols=[target_col])[target_col]
        out = (
            s.value_counts(dropna=False)
            .rename_axis(target_col)
            .reset_index(name="count")
        )
        out["share"] = out["count"] / out["count"].sum()
        return out
    except Exception:
        return None


def extract_overall_score(scorecard: Optional[pd.DataFrame]) -> Optional[float]:
    """Extract AssuranceTwin overall score from a flexible scorecard format."""
    if scorecard is None or scorecard.empty:
        return None

    df = scorecard.copy()
    lower_cols = [str(c).lower() for c in df.columns]

    # Case 1: explicit overall / total row
    first_col = df.columns[0]
    for idx, val in df[first_col].astype(str).str.lower().items():
        if any(key in val for key in ["overall", "total", "assurancetwin"]):
            numeric_values = pd.to_numeric(df.loc[idx], errors="coerce")
            numeric_values = numeric_values.dropna()
            if not numeric_values.empty:
                return float(numeric_values.iloc[-1])

    # Case 2: column containing final score
    for col in df.columns:
        col_lower = str(col).lower()
        if "overall" in col_lower or "assurancetwin" in col_lower or "final" in col_lower:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.iloc[0])

    # Case 3: weighted score calculation
    score_col = find_column(df, ["score", "component_score", "normalized_score"])
    weight_col = find_column(df, ["weight", "component_weight"])

    if score_col and weight_col:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        weights = pd.to_numeric(df[weight_col], errors="coerce")
        valid = scores.notna() & weights.notna()

        if valid.any():
            weighted_sum = float((scores[valid] * weights[valid]).sum())

            # If weights look like percentages, divide by 100.
            if weights[valid].sum() > 1.5:
                weighted_sum = weighted_sum / 100.0

            return weighted_sum

    return None


def summarize_model_performance(performance_df: Optional[pd.DataFrame]) -> Dict[str, Optional[str]]:
    """Identify likely champion and challenger models from model performance table."""
    result = {
        "champion_model": None,
        "challenger_model": None,
        "selection_metric": None,
        "champion_value": None,
        "challenger_value": None,
    }

    if performance_df is None or performance_df.empty:
        return result

    df = performance_df.copy()
    model_col = find_column(df, ["model", "model_name", "model_type", "algorithm"])
    if model_col is None:
        return result

    metric_priority = [
        ("AUC", ["auc", "roc_auc", "roc auc", "validation_auc"], False),
        ("Balanced Accuracy", ["balanced_accuracy", "balanced accuracy"], False),
        ("F1", ["f1", "f1_score", "f1 score"], False),
        ("Accuracy", ["accuracy"], False),
        ("Brier Score", ["brier", "brier_score"], True),
    ]

    metric_col = None
    metric_label = None
    ascending = False

    for label, candidates, lower_is_better in metric_priority:
        col = find_column(df, candidates)
        if col is not None:
            metric_col = col
            metric_label = label
            ascending = lower_is_better
            break

    if metric_col is None:
        return result

    df["_metric_value"] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["_metric_value"])

    if df.empty:
        return result

    df = df.sort_values("_metric_value", ascending=ascending)

    champion = df.iloc[0]
    result["champion_model"] = str(champion[model_col])
    result["selection_metric"] = metric_label
    result["champion_value"] = f"{champion['_metric_value']:.4f}"

    if len(df) > 1:
        challenger = df.iloc[1]
        result["challenger_model"] = str(challenger[model_col])
        result["challenger_value"] = f"{challenger['_metric_value']:.4f}"

    return result


def evaluate_fairness_flags(fairness_df: Optional[pd.DataFrame]) -> List[str]:
    """Generate fairness risk flags from flexible fairness metrics table."""
    flags = []

    if fairness_df is None or fairness_df.empty:
        return ["Fairness metrics were not available for final report consolidation."]

    df = fairness_df.copy()

    metric_col = find_column(df, ["metric", "fairness_metric", "measure"])
    value_col = find_column(df, ["value", "metric_value", "score"])
    group_col = find_column(df, ["group", "group_value", "category"])

    if metric_col and value_col:
        values = pd.to_numeric(df[value_col], errors="coerce")

        for _, row in df.iterrows():
            metric = str(row[metric_col]).lower()
            value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]

            if pd.isna(value):
                continue

            group_suffix = ""
            if group_col and pd.notna(row[group_col]):
                group_suffix = f" for group `{row[group_col]}`"

            if "disparate impact" in metric or "dir" in metric:
                if value < 0.80:
                    flags.append(
                        f"Potential adverse impact: disparate impact ratio below 0.80{group_suffix}."
                    )

            if any(key in metric for key in ["gap", "difference", "equal opportunity", "fnr", "fpr"]):
                if abs(value) > 0.10:
                    flags.append(
                        f"Material group-level difference: `{row[metric_col]}` exceeds 0.10{group_suffix}."
                    )

        return sorted(set(flags))

    # Fallback: inspect numeric columns for extreme values.
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        for col in numeric_df.columns:
            col_lower = str(col).lower()
            min_val = numeric_df[col].min()
            max_val = numeric_df[col].max()

            if "disparate" in col_lower and min_val < 0.80:
                flags.append(f"Potential adverse impact: minimum `{col}` is below 0.80.")

            if any(key in col_lower for key in ["gap", "difference", "fnr", "fpr"]):
                if max(abs(min_val), abs(max_val)) > 0.10:
                    flags.append(f"Material group-level difference detected in `{col}`.")

    if not flags:
        flags.append("No automated high-severity fairness flag was detected from the available fairness table.")

    return sorted(set(flags))


def evaluate_calibration_flags(calibration_df: Optional[pd.DataFrame]) -> List[str]:
    """Generate calibration risk flags."""
    flags = []

    if calibration_df is None or calibration_df.empty:
        return ["Calibration summary was not available for final report consolidation."]

    df = calibration_df.copy()

    metric_col = find_column(df, ["metric", "measure"])
    value_col = find_column(df, ["value", "metric_value", "score"])

    if metric_col and value_col:
        for _, row in df.iterrows():
            metric = str(row[metric_col]).lower()
            value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]

            if pd.isna(value):
                continue

            if any(key in metric for key in ["ece", "calibration error", "expected calibration"]):
                if value > 0.08:
                    flags.append("Expected calibration error exceeds 0.08 and should be remediated or monitored.")

            if "brier" in metric and value > 0.20:
                flags.append("Brier score is above 0.20, indicating possible probability-quality weakness.")

    else:
        numeric_df = df.select_dtypes(include="number")
        for col in numeric_df.columns:
            col_lower = str(col).lower()
            max_val = numeric_df[col].max()

            if any(key in col_lower for key in ["ece", "calibration_error"]) and max_val > 0.08:
                flags.append(f"Calibration error column `{col}` exceeds 0.08.")

    if not flags:
        flags.append("No automated high-severity calibration flag was detected from the available calibration table.")

    return sorted(set(flags))


def evaluate_drift_flags(drift_df: Optional[pd.DataFrame]) -> List[str]:
    """Generate drift and monitoring risk flags."""
    flags = []

    if drift_df is None or drift_df.empty:
        return ["Drift monitoring summary was not available for final report consolidation."]

    df = drift_df.copy()
    metric_col = find_column(df, ["metric", "drift_metric", "measure"])
    value_col = find_column(df, ["value", "metric_value", "score", "psi", "csi"])

    if metric_col and value_col:
        for _, row in df.iterrows():
            metric = str(row[metric_col]).lower()
            value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]

            if pd.isna(value):
                continue

            if "psi" in metric or "population stability" in metric:
                if value >= 0.25:
                    flags.append("Population Stability Index is at or above 0.25, indicating material drift.")
                elif value >= 0.10:
                    flags.append("Population Stability Index is at or above 0.10, indicating moderate drift.")

            if "csi" in metric or "characteristic stability" in metric:
                if value >= 0.25:
                    flags.append("Characteristic Stability Index is at or above 0.25 for at least one feature.")
    else:
        numeric_df = df.select_dtypes(include="number")
        for col in numeric_df.columns:
            col_lower = str(col).lower()
            max_val = numeric_df[col].max()

            if any(key in col_lower for key in ["psi", "csi", "stability"]):
                if max_val >= 0.25:
                    flags.append(f"Material drift detected in `{col}`.")
                elif max_val >= 0.10:
                    flags.append(f"Moderate drift detected in `{col}`.")

    if not flags:
        flags.append("No automated high-severity drift flag was detected from the available monitoring table.")

    return sorted(set(flags))


def determine_committee_recommendation(
    score: Optional[float],
    fairness_flags: List[str],
    calibration_flags: List[str],
    drift_flags: List[str],
    missing_critical_inputs: List[str],
) -> Tuple[str, List[str]]:
    """Determine final model validation recommendation."""
    reasons = []

    serious_missing = [
        item for item in missing_critical_inputs
        if item in {
            "Model performance summary",
            "Independent validation metrics",
            "Calibration summary",
            "Fairness metrics",
            "Stress test results",
            "Drift monitoring summary",
            "AssuranceTwin scorecard",
        }
    ]

    severe_fairness = [
        flag for flag in fairness_flags
        if "below 0.80" in flag or "exceeds 0.10" in flag
    ]

    severe_calibration = [
        flag for flag in calibration_flags
        if "exceeds 0.08" in flag or "above 0.20" in flag
    ]

    severe_drift = [
        flag for flag in drift_flags
        if "material drift" in flag.lower() or "at or above 0.25" in flag
    ]

    if serious_missing:
        reasons.append(
            "One or more critical validation artifacts are missing; approval should not be unconditional."
        )

    if severe_fairness:
        reasons.append(
            "Fairness review identified potential adverse impact or material group-level performance differences."
        )

    if severe_calibration:
        reasons.append(
            "Calibration review identified probability-quality concerns requiring remediation or monitoring."
        )

    if severe_drift:
        reasons.append(
            "Monitoring review identified material drift risk requiring lifecycle controls."
        )

    if score is not None:
        if score >= 85:
            score_recommendation = "Approval"
            reasons.append(f"AssuranceTwin Score is {score:.2f}, which is in the approval range.")
        elif score >= 70:
            score_recommendation = "Conditional Approval"
            reasons.append(f"AssuranceTwin Score is {score:.2f}, which supports conditional approval.")
        else:
            score_recommendation = "Rejection"
            reasons.append(f"AssuranceTwin Score is {score:.2f}, which is below the acceptable threshold.")
    else:
        score_recommendation = "Conditional Approval"
        reasons.append("Final AssuranceTwin Score was unavailable; recommendation is conservative.")

    if serious_missing:
        final = "Conditional Approval"
    elif severe_fairness or severe_calibration or severe_drift:
        if score is not None and score < 70:
            final = "Rejection"
        else:
            final = "Conditional Approval"
    else:
        final = score_recommendation

    if final == "Approval":
        reasons.append("No automated high-severity fairness, calibration, or drift flags were detected.")
    elif final == "Conditional Approval":
        reasons.append("Use should be permitted only with documented remediation, monitoring, and governance controls.")
    else:
        reasons.append("The model should not be approved until material validation weaknesses are resolved.")

    return final, sorted(set(reasons))


def create_input_inventory() -> Tuple[pd.DataFrame, List[str]]:
    """Create inventory of all report inputs."""
    rows = []
    missing = []

    for label, path in EXPECTED_INPUTS.items():
        exists = path.exists()
        rows.append(
            {
                "artifact": label,
                "path": relative(path),
                "exists": exists,
                "artifact_type": path.suffix.lower().replace(".", "") or "file",
            }
        )
        if not exists:
            missing.append(label)

    for label, path in EXPECTED_FIGURES.items():
        exists = path.exists()
        rows.append(
            {
                "artifact": label,
                "path": relative(path),
                "exists": exists,
                "artifact_type": "figure",
            }
        )

    inventory = pd.DataFrame(rows)
    inventory.to_csv(OUTPUT_INPUT_INVENTORY, index=False)

    return inventory, missing


def section(title: str) -> str:
    return f"\n## {title}\n"


def subsection(title: str) -> str:
    return f"\n### {title}\n"


# ---------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------

def main() -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    input_inventory, missing_inputs = create_input_inventory()

    model_inventory = read_csv_if_exists(TABLES_DIR / "model_inventory.csv")
    performance = read_csv_if_exists(TABLES_DIR / "model_performance_summary.csv")
    independent = read_csv_if_exists(TABLES_DIR / "independent_validation_metrics.csv")
    calibration = read_csv_if_exists(TABLES_DIR / "calibration_summary.csv")
    fairness = read_csv_if_exists(TABLES_DIR / "fairness_metrics.csv")
    explanation = read_csv_if_exists(TABLES_DIR / "explanation_stability.csv")
    stress = read_csv_if_exists(TABLES_DIR / "stress_test_results.csv")
    drift = read_csv_if_exists(TABLES_DIR / "drift_monitoring_summary.csv")
    scorecard = read_csv_if_exists(TABLES_DIR / "assurancetwin_scorecard.csv")

    fairness_report_excerpt = read_text_if_exists(VALIDATION_DIR / "fairness_validation_report.md", max_chars=2500)
    stress_report_excerpt = read_text_if_exists(VALIDATION_DIR / "stress_testing_report.md", max_chars=2500)
    monitoring_plan_excerpt = read_text_if_exists(VALIDATION_DIR / "monitoring_plan.md", max_chars=2500)

    dataset_path = PROCESSED_DATA_DIR / "hmda_modeling_dataset.csv"
    dataset_rows, dataset_cols, dataset_columns = count_csv_rows_and_columns(dataset_path)
    target_dist = target_distribution(dataset_path, target_col="approved")

    overall_score = extract_overall_score(scorecard)
    model_summary = summarize_model_performance(performance)

    fairness_flags = evaluate_fairness_flags(fairness)
    calibration_flags = evaluate_calibration_flags(calibration)
    drift_flags = evaluate_drift_flags(drift)

    recommendation, recommendation_reasons = determine_committee_recommendation(
        score=overall_score,
        fairness_flags=fairness_flags,
        calibration_flags=calibration_flags,
        drift_flags=drift_flags,
        missing_critical_inputs=missing_inputs,
    )

    lines: List[str] = []

    lines.append("# Model Validation Committee Report")
    lines.append("")
    lines.append("**Project:** AssuranceTwin AI — Model Validation and Governance")
    lines.append("")
    lines.append(f"**Generated:** {generated_at}")
    lines.append("")
    lines.append(f"**Final Validation Recommendation:** **{recommendation}**")
    lines.append("")

    if overall_score is not None:
        lines.append(f"**AssuranceTwin Score:** **{overall_score:.2f}**")
    else:
        lines.append("**AssuranceTwin Score:** Not available")

    lines.append("")
    lines.append("---")

    # 1. Executive Summary
    lines.append(section("1. Executive Summary"))
    lines.append(
        "This report consolidates the independent validation evidence for the candidate credit decision model. "
        "The review evaluates predictive performance, calibration, fairness, challenger model behavior, explainability, "
        "stress testing, drift stability, documentation completeness, and monitoring readiness."
    )
    lines.append("")
    lines.append(f"The final model validation recommendation is **{recommendation}**.")
    lines.append("")
    lines.append("Primary basis for recommendation:")
    lines.append("")
    for reason in recommendation_reasons:
        lines.append(f"- {reason}")

    lines.append("")
    lines.append("Key model summary:")
    lines.append("")
    lines.append(f"- Champion model identified from performance table: **{model_summary['champion_model'] or 'Not available'}**")
    lines.append(f"- Challenger model identified from performance table: **{model_summary['challenger_model'] or 'Not available'}**")
    lines.append(f"- Selection metric: **{model_summary['selection_metric'] or 'Not available'}**")
    lines.append(f"- Champion metric value: **{model_summary['champion_value'] or 'Not available'}**")
    lines.append(f"- Challenger metric value: **{model_summary['challenger_value'] or 'Not available'}**")

    # 2. Model Purpose and Use
    lines.append(section("2. Model Purpose and Use"))
    lines.append(
        "The model is designed to support credit decision analysis using HMDA-style mortgage application data. "
        "The target variable is a binary approval outcome, where approved applications are coded as 1 and non-approved "
        "applications are coded as 0. The intended use is analytical model validation, governance demonstration, "
        "and model-risk committee review rather than live automated underwriting."
    )
    lines.append("")
    lines.append("The model should be treated as a high-impact decision-support model because the output relates to credit access. "
                 "Accordingly, the validation standard emphasizes transparency, fairness, lifecycle monitoring, "
                 "and documented approval controls.")

    lines.append(subsection("Model Inventory"))
    lines.append(dataframe_to_markdown(model_inventory, max_rows=12, max_cols=10))

    # 3. Data Description
    lines.append(section("3. Data Description"))

    if dataset_path.exists():
        lines.append(f"The clean modeling dataset was found at `{relative(dataset_path)}`.")
        lines.append("")
        lines.append(f"- Number of records: **{dataset_rows if dataset_rows is not None else 'Not available'}**")
        lines.append(f"- Number of columns: **{dataset_cols if dataset_cols is not None else 'Not available'}**")
        lines.append("")
        if dataset_columns:
            preview_cols = ", ".join(dataset_columns[:25])
            if len(dataset_columns) > 25:
                preview_cols += ", ..."
            lines.append(f"Available column preview: `{preview_cols}`")
    else:
        lines.append("The clean modeling dataset was not found. Data description is therefore based only on available summary artifacts.")

    lines.append(subsection("Target Distribution"))
    lines.append(dataframe_to_markdown(target_dist, max_rows=10, max_cols=5))

    modeling_summary = read_csv_if_exists(TABLES_DIR / "modeling_dataset_summary.csv")
    modeling_target_summary = read_csv_if_exists(TABLES_DIR / "modeling_target_distribution.csv")

    lines.append(subsection("Modeling Dataset Summary"))
    lines.append(dataframe_to_markdown(modeling_summary, max_rows=12, max_cols=8))

    lines.append(subsection("Existing Target Distribution Table"))
    lines.append(dataframe_to_markdown(modeling_target_summary, max_rows=12, max_cols=8))

    # 4. Conceptual Soundness Review
    lines.append(section("4. Conceptual Soundness Review"))
    lines.append(
        "Conceptual soundness was reviewed by assessing whether the model objective, target construction, "
        "candidate predictors, validation framework, and governance controls are aligned with the stated business use. "
        "The model development design separates predictive performance from governance acceptability. "
        "A model with stronger AUC is not automatically considered acceptable unless calibration, fairness, stress behavior, "
        "monitoring readiness, and documentation quality are also satisfactory."
    )
    lines.append("")
    lines.append("Conceptual review observations:")
    lines.append("")
    lines.append("- The binary approval target is appropriate for a credit decision modeling demonstration.")
    lines.append("- The validation workflow includes independent testing rather than relying only on development metrics.")
    lines.append("- Champion/challenger comparison supports model-risk governance by testing whether alternatives produce more stable or more governable behavior.")
    lines.append("- Fairness and monitoring components are necessary because aggregate accuracy can mask group-level harm and post-deployment degradation.")

    # 5. Development Methodology
    lines.append(section("5. Development Methodology"))
    lines.append(
        "The development methodology uses a reproducible Python workflow with separate scripts for data inspection, "
        "dataset cleaning, model inventory, champion/challenger model training, independent validation, calibration analysis, "
        "fairness testing, explainability, stress testing, monitoring simulation, and final scoring."
    )
    lines.append("")
    lines.append("Development artifacts reviewed:")
    lines.append("")
    reviewed_artifacts = [
        "scripts/03_create_clean_hmda_dataset.py",
        "scripts/04_create_model_inventory.py",
        "scripts/05_train_champion_challenger_models.py",
        "scripts/06_independent_validation_metrics.py",
        "scripts/07_fairness_bias_testing.py",
        "scripts/08_calibration_analysis.py",
        "scripts/09_explainability_stability.py",
        "scripts/10_stress_testing.py",
        "scripts/11_monitoring_drift_simulation.py",
        "scripts/12_assurancetwin_score.py",
    ]
    for artifact in reviewed_artifacts:
        path = ROOT / artifact
        status = "Found" if path.exists() else "Not found"
        lines.append(f"- `{artifact}` — {status}")

    # 6. Independent Validation Results
    lines.append(section("6. Independent Validation Results"))
    lines.append(
        "Independent validation results assess discrimination, classification quality, probability accuracy, "
        "confusion-matrix behavior, approval rates, and error rates. These metrics are used to evaluate whether "
        "the model is sufficiently reliable for the stated validation use case."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(independent, max_rows=20, max_cols=8))

    if (FIGURES_DIR / "roc_curves.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "roc_curves.png"))

    if (FIGURES_DIR / "precision_recall_curves.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "precision_recall_curves.png"))

    # 7. Challenger Model Review
    lines.append(section("7. Challenger Model Review"))
    lines.append(
        "The challenger model review compares the selected champion model against alternative algorithms. "
        "The purpose is not only to identify the most predictive model, but also to assess whether the selected model "
        "is defensible from a validation, stability, transparency, and governance perspective."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(performance, max_rows=20, max_cols=10))

    lines.append("")
    lines.append("Challenger review conclusion:")
    lines.append("")
    if model_summary["champion_model"]:
        lines.append(
            f"The apparent champion model is **{model_summary['champion_model']}**, based on "
            f"**{model_summary['selection_metric']}** with value **{model_summary['champion_value']}**."
        )
    else:
        lines.append("A champion model could not be identified automatically because the performance table was unavailable or incomplete.")

    if model_summary["challenger_model"]:
        lines.append(
            f"The apparent challenger model is **{model_summary['challenger_model']}**, with value "
            f"**{model_summary['challenger_value']}** on the same selection metric."
        )

    # 8. Calibration Review
    lines.append(section("8. Calibration Review"))
    lines.append(
        "Calibration review evaluates whether predicted probabilities are reliable. In credit decision settings, "
        "probability quality matters because poorly calibrated scores can lead to incorrect approval thresholds, "
        "misleading risk segmentation, and weak monitoring triggers."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(calibration, max_rows=20, max_cols=8))

    lines.append("")
    lines.append("Calibration risk flags:")
    lines.append("")
    for flag in calibration_flags:
        lines.append(f"- {flag}")

    if (FIGURES_DIR / "calibration_curve.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "calibration_curve.png"))

    # 9. Fairness and Bias Review
    lines.append(section("9. Fairness and Bias Review"))
    lines.append(
        "Fairness review evaluates group-level performance and approval behavior across protected or policy-relevant segments. "
        "The review considers approval-rate differences, disparate impact ratio, false-negative-rate gaps, "
        "false-positive-rate gaps, equal opportunity differences, and calibration by group when available."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(fairness, max_rows=30, max_cols=10))

    lines.append("")
    lines.append("Fairness risk flags:")
    lines.append("")
    for flag in fairness_flags:
        lines.append(f"- {flag}")

    if fairness_report_excerpt:
        lines.append(subsection("Fairness Validation Report Excerpt"))
        lines.append(fairness_report_excerpt)

    if (FIGURES_DIR / "fairness_group_comparison.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "fairness_group_comparison.png"))

    # 10. Explainability Review
    lines.append(section("10. Explainability Review"))
    lines.append(
        "Explainability review assesses whether the model's drivers are understandable and whether explanations remain stable "
        "across time splits, demographic groups, and model alternatives. Explanation instability is a governance concern because "
        "it can indicate that the model is relying on unstable proxy relationships."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(explanation, max_rows=20, max_cols=8))

    if (FIGURES_DIR / "feature_importance.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "feature_importance.png"))

    if (FIGURES_DIR / "shap_summary.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "shap_summary.png"))

    # 11. Stress Testing
    lines.append(section("11. Stress Testing"))
    lines.append(
        "Stress testing evaluates model sensitivity under adverse or unusual scenarios, including income shocks, "
        "loan amount increases, LTV increases, missing-data shocks, minority-tract distribution shifts, out-of-time validation, "
        "and recession-like synthetic conditions. The objective is to determine whether model behavior remains plausible "
        "under conditions that differ from ordinary validation data."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(stress, max_rows=30, max_cols=10))

    if stress_report_excerpt:
        lines.append(subsection("Stress Testing Report Excerpt"))
        lines.append(stress_report_excerpt)

    if (FIGURES_DIR / "stress_test_model_sensitivity.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "stress_test_model_sensitivity.png"))

    # 12. Drift and Monitoring Plan
    lines.append(section("12. Drift and Monitoring Plan"))
    lines.append(
        "The monitoring plan evaluates population stability, characteristic stability, data drift, prediction drift, "
        "performance drift, fairness drift, and calibration drift. These controls are required because model performance "
        "and fairness characteristics can degrade after deployment or when applicant populations shift."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(drift, max_rows=30, max_cols=10))

    lines.append("")
    lines.append("Drift and monitoring risk flags:")
    lines.append("")
    for flag in drift_flags:
        lines.append(f"- {flag}")

    if monitoring_plan_excerpt:
        lines.append(subsection("Monitoring Plan Excerpt"))
        lines.append(monitoring_plan_excerpt)

    if (FIGURES_DIR / "drift_dashboard_plot.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "drift_dashboard_plot.png"))

    # 13. Model Limitations
    lines.append(section("13. Model Limitations"))
    lines.append(
        "The following limitations should be considered before any production interpretation:"
    )
    lines.append("")
    lines.append("- The project is a validation and governance demonstration, not a production underwriting system.")
    lines.append("- HMDA-style public data may not contain all variables used in an actual credit underwriting process.")
    lines.append("- Approval outcome labels may reflect historical decision patterns and may therefore embed historical policy, market, or institutional bias.")
    lines.append("- Fairness metrics are sensitive to group definitions, sample size, threshold choice, and target construction.")
    lines.append("- Calibration and drift results should be re-evaluated periodically using fresh out-of-time data.")
    lines.append("- Explainability outputs should be interpreted as model-behavior diagnostics, not causal proof.")
    lines.append("- Stress scenarios are synthetic and should be supplemented with institution-specific macroeconomic and portfolio-risk assumptions.")

    # 14. Governance Recommendations
    lines.append(section("14. Governance Recommendations"))
    lines.append("Recommended governance actions:")
    lines.append("")
    lines.append("- Maintain this model in a formal model inventory with owner, validator, use case, materiality, and approval status.")
    lines.append("- Require independent validation refresh before any change in target definition, feature set, population, or decision threshold.")
    lines.append("- Track monthly or quarterly drift metrics, including PSI, CSI, prediction drift, calibration drift, and fairness drift.")
    lines.append("- Establish escalation thresholds for material drift, adverse impact, or degraded calibration.")
    lines.append("- Review approval-rate differences and error-rate differences by protected or policy-relevant groups.")
    lines.append("- Document challenger model results and explain why the champion model is acceptable from both performance and governance perspectives.")
    lines.append("- Store all validation artifacts, figures, scorecards, and committee recommendations in the repository.")
    lines.append("- Require model-risk committee approval before any production use or external decision-support use.")

    lines.append(subsection("AssuranceTwin Scorecard"))
    lines.append(dataframe_to_markdown(scorecard, max_rows=20, max_cols=10))

    if (FIGURES_DIR / "assurancetwin_score_radar.png").exists():
        lines.append("")
        lines.append(figure_link(FIGURES_DIR / "assurancetwin_score_radar.png"))

    # 15. Approval / Conditional Approval / Rejection Recommendation
    lines.append(section("15. Approval / Conditional Approval / Rejection Recommendation"))
    lines.append(f"**Final recommendation:** **{recommendation}**")
    lines.append("")
    lines.append("Recommendation rationale:")
    lines.append("")
    for reason in recommendation_reasons:
        lines.append(f"- {reason}")

    if recommendation == "Approval":
        lines.append("")
        lines.append(
            "The model may be approved for the stated validation use case, subject to normal monitoring, "
            "documentation retention, and periodic validation refresh."
        )
    elif recommendation == "Conditional Approval":
        lines.append("")
        lines.append(
            "The model may be conditionally approved for the stated validation use case only if the documented "
            "conditions are addressed. Required conditions include continued monitoring, review of flagged fairness, "
            "calibration, or drift issues, and maintenance of complete validation documentation."
        )
    else:
        lines.append("")
        lines.append(
            "The model should not be approved until material validation deficiencies are resolved and the model is "
            "resubmitted for independent validation review."
        )

    # Appendix
    lines.append(section("Appendix A. Report Input Inventory"))
    lines.append(
        f"The full report input inventory was saved to `{relative(OUTPUT_INPUT_INVENTORY)}`."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(input_inventory, max_rows=60, max_cols=5))

    lines.append(section("Appendix B. Missing Inputs"))
    if missing_inputs:
        lines.append("The following expected inputs were not found:")
        lines.append("")
        for item in missing_inputs:
            lines.append(f"- {item}")
    else:
        lines.append("No expected core inputs were missing.")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("Step O completed successfully.")
    print(f"Created report: {relative(OUTPUT_REPORT)}")
    print(f"Created input inventory: {relative(OUTPUT_INPUT_INVENTORY)}")
    print(f"Final recommendation: {recommendation}")

    if overall_score is not None:
        print(f"AssuranceTwin Score: {overall_score:.2f}")
    else:
        print("AssuranceTwin Score: not available")


if __name__ == "__main__":
    main()