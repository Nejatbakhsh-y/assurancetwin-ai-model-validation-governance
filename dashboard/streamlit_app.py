"""
AssuranceTwin AI - Streamlit Governance Dashboard

Step Q:
Builds a visual AI governance and model validation dashboard for:

1. Model Inventory
2. Champion vs Challenger
3. Validation Metrics
4. Fairness Review
5. Calibration
6. Stress Testing
7. Drift Monitoring
8. AssuranceTwin Score
9. Model Risk Committee Summary

Run from the repository root:

    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="AssuranceTwin AI Governance Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Path utilities
# =============================================================================

def find_project_root() -> Path:
    """
    Finds the repository root whether Streamlit is run from the root folder
    or from inside the dashboard folder.
    """
    current = Path(__file__).resolve().parent

    for candidate in [current, *current.parents]:
        has_reports = (candidate / "reports").exists()
        has_docs = (candidate / "docs").exists()
        has_dashboard = (candidate / "dashboard").exists()

        if has_dashboard and (has_reports or has_docs):
            return candidate

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()


def project_path(relative_path: str | Path) -> Path:
    return PROJECT_ROOT / Path(relative_path)


def first_existing_path(paths: Iterable[str | Path]) -> Optional[Path]:
    for path in paths:
        full_path = project_path(path)
        if full_path.exists() and full_path.is_file():
            return full_path
    return None


def file_exists(relative_path: str | Path) -> bool:
    return project_path(relative_path).exists()


# =============================================================================
# Loading utilities
# =============================================================================

@st.cache_data(show_spinner=False)
def read_csv_cached(path_string: str) -> pd.DataFrame:
    path = Path(path_string)

    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


@st.cache_data(show_spinner=False)
def read_text_cached(path_string: str) -> str:
    path = Path(path_string)

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin1")


def load_csv(candidates: Iterable[str | Path]) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    path = first_existing_path(candidates)

    if path is None:
        return None, None

    try:
        df = read_csv_cached(str(path))
        return df, path
    except Exception as exc:
        st.error(f"Could not load CSV file: {path}")
        st.exception(exc)
        return None, path


def load_markdown(candidates: Iterable[str | Path]) -> tuple[Optional[str], Optional[Path]]:
    path = first_existing_path(candidates)

    if path is None:
        return None, None

    try:
        text = read_text_cached(str(path))
        return text, path
    except Exception as exc:
        st.error(f"Could not load Markdown file: {path}")
        st.exception(exc)
        return None, path


def relative_display_path(path: Optional[Path]) -> str:
    if path is None:
        return "Not found"

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# =============================================================================
# Data utilities
# =============================================================================

def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized_to_original = {
        normalize_column_name(column): column for column in df.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)

        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]

    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_number(value) -> str:
    if pd.isna(value):
        return "N/A"

    try:
        numeric_value = float(value)
    except Exception:
        return str(value)

    if abs(numeric_value) >= 100:
        return f"{numeric_value:,.1f}"

    if abs(numeric_value) >= 10:
        return f"{numeric_value:,.2f}"

    return f"{numeric_value:,.3f}"


def dataframe_height(df: pd.DataFrame) -> int:
    return min(650, max(250, 38 * (len(df) + 1)))


def show_missing_box(title: str, expected_files: Iterable[str]) -> None:
    st.info(
        f"{title} is not available yet. Run the related project script first, "
        "then refresh this dashboard."
    )

    with st.expander("Expected file locations"):
        for file in expected_files:
            st.code(file, language="text")


def show_table(
    title: str,
    candidates: Iterable[str | Path],
    caption: Optional[str] = None,
) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    df, path = load_csv(candidates)

    st.subheader(title)

    if df is None:
        show_missing_box(title, [str(item) for item in candidates])
        return None, path

    st.caption(caption or f"Loaded from: {relative_display_path(path)}")
    st.dataframe(df, width="stretch", height=dataframe_height(df))

    return df, path


def show_markdown(
    title: str,
    candidates: Iterable[str | Path],
    fallback_message: Optional[str] = None,
) -> tuple[Optional[str], Optional[Path]]:
    text, path = load_markdown(candidates)

    st.subheader(title)

    if text is None:
        st.info(
            fallback_message
            or f"{title} is not available yet. Run the related script first."
        )

        with st.expander("Expected file locations"):
            for file in candidates:
                st.code(str(file), language="text")

        return None, path

    st.caption(f"Loaded from: {relative_display_path(path)}")
    st.markdown(text)

    return text, path


def show_image(
    title: str,
    candidates: Iterable[str | Path],
    fallback_message: Optional[str] = None,
) -> Optional[Path]:
    path = first_existing_path(candidates)

    st.subheader(title)

    if path is None:
        st.info(
            fallback_message
            or f"{title} is not available yet. Run the related script first."
        )

        with st.expander("Expected file locations"):
            for file in candidates:
                st.code(str(file), language="text")

        return None

    st.caption(f"Loaded from: {relative_display_path(path)}")
    st.image(str(path), width="stretch")

    return path


def show_download_button(path: Path, label: str) -> None:
    try:
        data = path.read_bytes()
    except Exception:
        return

    st.download_button(
        label=label,
        data=data,
        file_name=path.name,
        mime="application/octet-stream",
    )


def metric_cards(metrics: dict[str, object]) -> None:
    if not metrics:
        return

    cols = st.columns(min(4, len(metrics)))

    for index, (label, value) in enumerate(metrics.items()):
        with cols[index % len(cols)]:
            st.metric(label=label, value=str(value))


def chart_by_category(
    df: pd.DataFrame,
    category_candidates: Iterable[str],
    metric_candidates: Iterable[str],
    title: str,
) -> None:
    category_col = find_column(df, category_candidates)

    if category_col is None:
        return

    metric_cols: list[str] = []

    for candidate in metric_candidates:
        col = find_column(df, [candidate])
        if col is not None and col not in metric_cols:
            metric_cols.append(col)

    if not metric_cols:
        return

    chart_df = df[[category_col, *metric_cols]].copy()

    for col in metric_cols:
        chart_df[col] = coerce_numeric(chart_df[col])

    chart_df = chart_df.dropna(how="all", subset=metric_cols)

    if chart_df.empty:
        return

    st.subheader(title)
    st.bar_chart(chart_df.set_index(category_col)[metric_cols])


def extract_metric_from_name_value_table(
    df: pd.DataFrame,
    name_candidates: Iterable[str],
    value_candidates: Iterable[str],
    target_names: Iterable[str],
) -> Optional[object]:
    name_col = find_column(df, name_candidates)
    value_col = find_column(df, value_candidates)

    if name_col is None or value_col is None:
        return None

    normalized_targets = {
        normalize_column_name(target) for target in target_names
    }

    for _, row in df.iterrows():
        metric_name = normalize_column_name(row.get(name_col, ""))

        if metric_name in normalized_targets:
            return row.get(value_col)

    return None


def find_best_model(df: pd.DataFrame) -> Optional[dict[str, object]]:
    model_col = find_column(df, ["model", "model_name", "algorithm"])
    score_col = find_column(
        df,
        [
            "auc",
            "roc_auc",
            "validation_auc",
            "test_auc",
            "f1",
            "f1_score",
            "balanced_accuracy",
            "accuracy",
        ],
    )

    if model_col is None or score_col is None:
        return None

    temp = df[[model_col, score_col]].copy()
    temp[score_col] = coerce_numeric(temp[score_col])
    temp = temp.dropna(subset=[score_col])

    if temp.empty:
        return None

    best_row = temp.sort_values(score_col, ascending=False).iloc[0]

    return {
        "model": best_row[model_col],
        "metric": score_col,
        "score": best_row[score_col],
    }


def get_overall_assurancetwin_score(df: pd.DataFrame) -> Optional[float]:
    component_col = find_column(df, ["component", "score_component", "metric"])
    score_col = find_column(df, ["score", "component_score"])
    weighted_col = find_column(df, ["weighted_score", "weighted_component_score"])

    if component_col is not None:
        normalized_components = df[component_col].astype(str).map(normalize_column_name)

        for target in [
            "assurancetwin_score",
            "overall_score",
            "overall_assurancetwin_score",
        ]:
            match = df[normalized_components == target]

            if not match.empty:
                for candidate_col in [score_col, weighted_col]:
                    if candidate_col is not None:
                        value = coerce_numeric(match[candidate_col]).dropna()
                        if not value.empty:
                            return float(value.iloc[0])

    if weighted_col is not None:
        values = coerce_numeric(df[weighted_col]).dropna()
        if not values.empty:
            return float(values.sum())

    if score_col is not None:
        values = coerce_numeric(df[score_col]).dropna()
        if not values.empty:
            return float(values.mean())

    return None


def status_label_from_score(score: Optional[float]) -> str:
    if score is None:
        return "Not available"

    if score >= 85:
        return "Strong"

    if score >= 70:
        return "Acceptable with monitoring"

    if score >= 55:
        return "Conditional approval likely"

    return "Remediation required"


def find_min_disparate_impact(df: pd.DataFrame) -> Optional[float]:
    col = find_column(
        df,
        [
            "disparate_impact_ratio",
            "disparate_impact",
            "di_ratio",
            "selection_rate_ratio",
        ],
    )

    if col is None:
        return None

    values = coerce_numeric(df[col]).dropna()

    if values.empty:
        return None

    return float(values.min())


def find_max_fairness_gap(df: pd.DataFrame) -> Optional[float]:
    possible_cols = [
        "approval_rate_difference",
        "approval_rate_gap",
        "false_negative_rate_gap",
        "false_positive_rate_gap",
        "equal_opportunity_difference",
        "calibration_gap",
    ]

    values: list[float] = []

    for candidate in possible_cols:
        col = find_column(df, [candidate])

        if col is not None:
            numeric_values = coerce_numeric(df[col]).abs().dropna()
            values.extend(numeric_values.tolist())

    if not values:
        return None

    return float(max(values))


def committee_file_status() -> pd.DataFrame:
    expected_files = [
        ("Model Inventory", "reports/tables/model_inventory.csv"),
        ("Performance Summary", "reports/tables/model_performance_summary.csv"),
        ("Independent Validation Metrics", "reports/tables/independent_validation_metrics.csv"),
        ("Fairness Metrics", "reports/tables/fairness_metrics.csv"),
        ("Calibration Summary", "reports/tables/calibration_summary.csv"),
        ("Stress Testing Results", "reports/tables/stress_test_results.csv"),
        ("Drift Monitoring Summary", "reports/tables/drift_monitoring_summary.csv"),
        ("AssuranceTwin Scorecard", "reports/tables/assurancetwin_scorecard.csv"),
        ("Model Validation Report", "reports/validation/model_validation_report.md"),
        ("Fairness Validation Report", "reports/validation/fairness_validation_report.md"),
        ("Stress Testing Report", "reports/validation/stress_testing_report.md"),
        ("Monitoring Plan", "reports/validation/monitoring_plan.md"),
        ("Model Card", "docs/model_card.md"),
        ("AI Governance Card", "docs/ai_governance_card.md"),
        ("Validation Checklist", "docs/validation_checklist.md"),
    ]

    rows = []

    for item_name, relative_path in expected_files:
        path = project_path(relative_path)
        rows.append(
            {
                "artifact": item_name,
                "path": relative_path,
                "status": "Available" if path.exists() else "Missing",
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("AssuranceTwin AI")
st.sidebar.markdown("Model Validation and AI Governance Dashboard")

st.sidebar.divider()

st.sidebar.write("Repository root")
st.sidebar.code(str(PROJECT_ROOT), language="text")

st.sidebar.write("Dashboard refreshed")
st.sidebar.code(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), language="text")

st.sidebar.divider()

status_df = committee_file_status()
available_count = int((status_df["status"] == "Available").sum())
total_count = int(len(status_df))

st.sidebar.metric("Governance artifacts available", f"{available_count}/{total_count}")

with st.sidebar.expander("Artifact status"):
    st.dataframe(status_df, width="stretch", height=420)


# =============================================================================
# Main title
# =============================================================================

st.title("AssuranceTwin AI Governance Dashboard")

st.markdown(
    """
This dashboard consolidates the project outputs into a model-risk and AI-governance
review interface. It is designed to show model inventory, validation evidence,
fairness review, calibration quality, stress testing, drift monitoring, and the
final AssuranceTwin governance score.
"""
)


# =============================================================================
# Top-level readiness indicators
# =============================================================================

score_df, score_path = load_csv(["reports/tables/assurancetwin_scorecard.csv"])
validation_df, validation_path = load_csv(["reports/tables/independent_validation_metrics.csv"])
fairness_df, fairness_path = load_csv(["reports/tables/fairness_metrics.csv"])
model_perf_df, model_perf_path = load_csv(["reports/tables/model_performance_summary.csv"])

overall_score = get_overall_assurancetwin_score(score_df) if score_df is not None else None
min_di = find_min_disparate_impact(fairness_df) if fairness_df is not None else None
max_gap = find_max_fairness_gap(fairness_df) if fairness_df is not None else None
best_model = find_best_model(model_perf_df) if model_perf_df is not None else None

top_metrics = {
    "AssuranceTwin Status": status_label_from_score(overall_score),
    "AssuranceTwin Score": "N/A" if overall_score is None else format_number(overall_score),
    "Artifacts Available": f"{available_count}/{total_count}",
}

if best_model is not None:
    top_metrics["Best Predictive Model"] = str(best_model["model"])

if min_di is not None:
    top_metrics["Minimum DI Ratio"] = format_number(min_di)

if max_gap is not None:
    top_metrics["Maximum Fairness Gap"] = format_number(max_gap)

metric_cards(top_metrics)

st.divider()


# =============================================================================
# Dashboard tabs
# =============================================================================

tabs = st.tabs(
    [
        "Model Inventory",
        "Champion vs Challenger",
        "Validation Metrics",
        "Fairness Review",
        "Calibration",
        "Stress Testing",
        "Drift Monitoring",
        "AssuranceTwin Score",
        "Model Risk Committee Summary",
    ]
)


# =============================================================================
# Tab 1: Model Inventory
# =============================================================================

with tabs[0]:
    st.header("Model Inventory")

    inventory_df, inventory_path = show_table(
        "Inventory Table",
        ["reports/tables/model_inventory.csv"],
        caption="Central model inventory for lifecycle governance and model-risk tracking.",
    )

    if inventory_df is not None:
        model_id_col = find_column(inventory_df, ["model_id", "id"])
        risk_col = find_column(inventory_df, ["risk_tier", "risk_rating", "tier"])
        approval_col = find_column(
            inventory_df,
            ["approval_status", "validation_status", "status"],
        )

        metrics = {}

        if model_id_col is not None:
            metrics["Models in Inventory"] = inventory_df[model_id_col].nunique()
        else:
            metrics["Models in Inventory"] = len(inventory_df)

        if risk_col is not None:
            high_risk_count = inventory_df[risk_col].astype(str).str.lower().str.contains(
                "high", na=False
            ).sum()
            metrics["High-Risk Models"] = int(high_risk_count)

        if approval_col is not None:
            approved_count = inventory_df[approval_col].astype(str).str.lower().str.contains(
                "approved", na=False
            ).sum()
            metrics["Approved or Conditionally Approved"] = int(approved_count)

        metric_cards(metrics)

    show_markdown(
        "Inventory Documentation",
        ["docs/model_inventory_template.md"],
        fallback_message="The inventory template has not been generated yet.",
    )


# =============================================================================
# Tab 2: Champion vs Challenger
# =============================================================================

with tabs[1]:
    st.header("Champion vs Challenger")

    performance_df, _ = show_table(
        "Model Performance Summary",
        ["reports/tables/model_performance_summary.csv"],
        caption=(
            "Compares champion and challenger models. The best predictive model is "
            "not automatically the best governed model."
        ),
    )

    if performance_df is not None:
        best = find_best_model(performance_df)

        if best is not None:
            metric_cards(
                {
                    "Best Model by Available Score": best["model"],
                    "Selection Metric": best["metric"],
                    "Score": format_number(best["score"]),
                }
            )

        chart_by_category(
            performance_df,
            category_candidates=["model", "model_name", "algorithm"],
            metric_candidates=[
                "auc",
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "f1_score",
                "balanced_accuracy",
                "brier_score",
            ],
            title="Model Comparison Chart",
        )

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            "ROC Curves",
            ["reports/figures/roc_curves.png"],
            fallback_message="ROC curve figure has not been generated yet.",
        )

    with col2:
        show_image(
            "Precision-Recall Curves",
            ["reports/figures/precision_recall_curves.png"],
            fallback_message="Precision-recall figure has not been generated yet.",
        )


# =============================================================================
# Tab 3: Validation Metrics
# =============================================================================

with tabs[2]:
    st.header("Validation Metrics")

    validation_metrics_df, _ = show_table(
        "Independent Validation Metrics",
        ["reports/tables/independent_validation_metrics.csv"],
        caption="Independent validation metrics used for model-risk review.",
    )

    if validation_metrics_df is not None:
        auc_value = extract_metric_from_name_value_table(
            validation_metrics_df,
            name_candidates=["metric", "name"],
            value_candidates=["value", "score", "metric_value"],
            target_names=["auc", "roc_auc"],
        )
        accuracy_value = extract_metric_from_name_value_table(
            validation_metrics_df,
            name_candidates=["metric", "name"],
            value_candidates=["value", "score", "metric_value"],
            target_names=["accuracy"],
        )
        brier_value = extract_metric_from_name_value_table(
            validation_metrics_df,
            name_candidates=["metric", "name"],
            value_candidates=["value", "score", "metric_value"],
            target_names=["brier_score"],
        )
        calibration_error_value = extract_metric_from_name_value_table(
            validation_metrics_df,
            name_candidates=["metric", "name"],
            value_candidates=["value", "score", "metric_value"],
            target_names=["calibration_error", "expected_calibration_error"],
        )

        metrics = {}

        if auc_value is not None:
            metrics["AUC"] = format_number(auc_value)

        if accuracy_value is not None:
            metrics["Accuracy"] = format_number(accuracy_value)

        if brier_value is not None:
            metrics["Brier Score"] = format_number(brier_value)

        if calibration_error_value is not None:
            metrics["Calibration Error"] = format_number(calibration_error_value)

        metric_cards(metrics)

    show_markdown(
        "Independent Validation Report",
        [
            "reports/validation/model_validation_report.md",
            "reports/validation/independent_validation_report.md",
        ],
        fallback_message="The independent validation report has not been generated yet.",
    )


# =============================================================================
# Tab 4: Fairness Review
# =============================================================================

with tabs[3]:
    st.header("Fairness Review")

    fairness_metrics_df, _ = show_table(
        "Fairness Metrics",
        ["reports/tables/fairness_metrics.csv"],
        caption=(
            "Fairness metrics by race, ethnicity, sex, income band, and "
            "minority-tract band where available."
        ),
    )

    if fairness_metrics_df is not None:
        min_disparate_impact = find_min_disparate_impact(fairness_metrics_df)
        max_fairness_gap = find_max_fairness_gap(fairness_metrics_df)

        fairness_cards = {}

        if min_disparate_impact is not None:
            fairness_cards["Minimum Disparate Impact Ratio"] = format_number(
                min_disparate_impact
            )

        if max_fairness_gap is not None:
            fairness_cards["Maximum Absolute Fairness Gap"] = format_number(
                max_fairness_gap
            )

        metric_cards(fairness_cards)

        if min_disparate_impact is not None and min_disparate_impact < 0.80:
            st.warning(
                "The minimum disparate impact ratio is below 0.80. "
                "This should be reviewed before approval."
            )

        chart_by_category(
            fairness_metrics_df,
            category_candidates=[
                "group",
                "group_name",
                "protected_group",
                "segment",
                "attribute",
            ],
            metric_candidates=[
                "approval_rate",
                "approval_rate_difference",
                "disparate_impact_ratio",
                "false_negative_rate_gap",
                "false_positive_rate_gap",
                "equal_opportunity_difference",
                "calibration_by_group",
                "calibration_gap",
            ],
            title="Fairness Metric Comparison",
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        show_image(
            "Fairness Group Comparison",
            ["reports/figures/fairness_group_comparison.png"],
            fallback_message="Fairness comparison figure has not been generated yet.",
        )

    with col2:
        show_markdown(
            "Fairness Validation Report",
            ["reports/validation/fairness_validation_report.md"],
            fallback_message="Fairness validation report has not been generated yet.",
        )


# =============================================================================
# Tab 5: Calibration
# =============================================================================

with tabs[4]:
    st.header("Calibration")

    calibration_df, _ = show_table(
        "Calibration Summary",
        ["reports/tables/calibration_summary.csv"],
        caption="Calibration quality is critical when model outputs are interpreted as probabilities.",
    )

    if calibration_df is not None:
        brier_value = extract_metric_from_name_value_table(
            calibration_df,
            name_candidates=["metric", "name"],
            value_candidates=["value", "score", "metric_value"],
            target_names=["brier_score"],
        )
        ece_value = extract_metric_from_name_value_table(
            calibration_df,
            name_candidates=["metric", "name"],
            value_candidates=["value", "score", "metric_value"],
            target_names=[
                "expected_calibration_error",
                "ece",
                "calibration_error",
            ],
        )

        calibration_cards = {}

        if brier_value is not None:
            calibration_cards["Brier Score"] = format_number(brier_value)

        if ece_value is not None:
            calibration_cards["Expected Calibration Error"] = format_number(ece_value)

        metric_cards(calibration_cards)

    show_image(
        "Calibration Curve",
        ["reports/figures/calibration_curve.png"],
        fallback_message="Calibration curve has not been generated yet.",
    )


# =============================================================================
# Tab 6: Stress Testing
# =============================================================================

with tabs[5]:
    st.header("Stress Testing")

    stress_df, _ = show_table(
        "Stress Test Results",
        ["reports/tables/stress_test_results.csv"],
        caption=(
            "Stress scenarios include income shock, loan amount increase, LTV increase, "
            "missing-data shock, minority-tract shift, out-of-time validation, and "
            "recession-like synthetic scenarios where available."
        ),
    )

    if stress_df is not None:
        scenario_col = find_column(stress_df, ["scenario", "stress_scenario"])
        metric_candidates = [
            "auc",
            "accuracy",
            "approval_rate",
            "performance_change",
            "auc_change",
            "approval_rate_change",
            "fairness_change",
            "calibration_change",
        ]

        chart_by_category(
            stress_df,
            category_candidates=["scenario", "stress_scenario"],
            metric_candidates=metric_candidates,
            title="Stress Scenario Sensitivity",
        )

        if scenario_col is not None:
            metric_cards({"Stress Scenarios Tested": stress_df[scenario_col].nunique()})

    col1, col2 = st.columns([1, 1])

    with col1:
        show_image(
            "Stress Test Sensitivity Plot",
            ["reports/figures/stress_test_model_sensitivity.png"],
            fallback_message="Stress test sensitivity figure has not been generated yet.",
        )

    with col2:
        show_markdown(
            "Stress Testing Report",
            ["reports/validation/stress_testing_report.md"],
            fallback_message="Stress testing report has not been generated yet.",
        )


# =============================================================================
# Tab 7: Drift Monitoring
# =============================================================================

with tabs[6]:
    st.header("Drift Monitoring")

    drift_df, _ = show_table(
        "Drift Monitoring Summary",
        ["reports/tables/drift_monitoring_summary.csv"],
        caption=(
            "Monitoring includes data drift, prediction drift, performance drift, "
            "fairness drift, calibration drift, PSI, and CSI where available."
        ),
    )

    if drift_df is not None:
        period_col = find_column(
            drift_df,
            ["period", "monitoring_period", "month", "quarter", "time_period"],
        )

        if period_col is not None:
            metric_cards({"Monitoring Periods": drift_df[period_col].nunique()})

        chart_by_category(
            drift_df,
            category_candidates=[
                "period",
                "monitoring_period",
                "month",
                "quarter",
                "time_period",
            ],
            metric_candidates=[
                "psi",
                "population_stability_index",
                "csi",
                "characteristic_stability_index",
                "data_drift",
                "prediction_drift",
                "performance_drift",
                "fairness_drift",
                "calibration_drift",
            ],
            title="Drift Monitoring Chart",
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        show_image(
            "Drift Dashboard Plot",
            ["reports/figures/drift_dashboard_plot.png"],
            fallback_message="Drift dashboard figure has not been generated yet.",
        )

    with col2:
        show_markdown(
            "Monitoring Plan",
            ["reports/validation/monitoring_plan.md"],
            fallback_message="Monitoring plan has not been generated yet.",
        )


# =============================================================================
# Tab 8: AssuranceTwin Score
# =============================================================================

with tabs[7]:
    st.header("AssuranceTwin Score")

    assurance_df, assurance_path = show_table(
        "AssuranceTwin Scorecard",
        ["reports/tables/assurancetwin_scorecard.csv"],
        caption=(
            "Composite score combining performance, calibration, fairness, robustness, "
            "drift stability, explainability stability, documentation completeness, "
            "and monitoring readiness."
        ),
    )

    if assurance_df is not None:
        score = get_overall_assurancetwin_score(assurance_df)

        metric_cards(
            {
                "Overall Score": "N/A" if score is None else format_number(score),
                "Governance Status": status_label_from_score(score),
            }
        )

        chart_by_category(
            assurance_df,
            category_candidates=[
                "component",
                "score_component",
                "metric",
                "dimension",
            ],
            metric_candidates=[
                "score",
                "component_score",
                "weighted_score",
                "weighted_component_score",
            ],
            title="AssuranceTwin Component Scores",
        )

    show_image(
        "AssuranceTwin Radar Plot",
        ["reports/figures/assurancetwin_score_radar.png"],
        fallback_message="AssuranceTwin radar figure has not been generated yet.",
    )

    show_markdown(
        "AssuranceTwin Score Report",
        [
            "reports/validation/assurancetwin_score_report.md",
            "reports/validation/assurancetwin_summary.md",
        ],
        fallback_message=(
            "No AssuranceTwin Markdown report was found. The scorecard and radar plot "
            "are sufficient for this dashboard tab if those files exist."
        ),
    )


# =============================================================================
# Tab 9: Model Risk Committee Summary
# =============================================================================

with tabs[8]:
    st.header("Model Risk Committee Summary")

    st.markdown(
        """
This tab organizes the final documentation package for review by a model-risk
committee, AI governance committee, or senior validation stakeholder.
"""
    )

    committee_status = committee_file_status()

    available = int((committee_status["status"] == "Available").sum())
    missing = int((committee_status["status"] == "Missing").sum())

    metric_cards(
        {
            "Available Artifacts": available,
            "Missing Artifacts": missing,
            "Documentation Readiness": f"{available}/{len(committee_status)}",
        }
    )

    st.subheader("Governance Package Status")
    st.dataframe(
        committee_status,
        width="stretch",
        height=dataframe_height(committee_status),
    )

    st.subheader("Core Committee Documents")

    doc_candidates = [
        ("AI Governance Card", ["docs/ai_governance_card.md"]),
        ("Model Card", ["docs/model_card.md"]),
        ("Validation Checklist", ["docs/validation_checklist.md"]),
        (
            "Final Model Validation Report",
            [
                "reports/validation/final_model_validation_report.md",
                "reports/validation/model_validation_report.md",
                "reports/validation/model_risk_committee_report.md",
            ],
        ),
    ]

    selected_doc_name = st.selectbox(
        "Select document to review",
        [item[0] for item in doc_candidates],
    )

    selected_candidates = next(
        paths for name, paths in doc_candidates if name == selected_doc_name
    )

    selected_text, selected_path = load_markdown(selected_candidates)

    if selected_text is None:
        show_missing_box(selected_doc_name, selected_candidates)
    else:
        st.caption(f"Loaded from: {relative_display_path(selected_path)}")
        st.markdown(selected_text)

        if selected_path is not None:
            show_download_button(
                selected_path,
                f"Download {selected_doc_name}",
            )

    st.subheader("Suggested Committee Review Questions")

    st.markdown(
        """
1. Is the model purpose clearly defined and consistent with the documented intended use?
2. Are the training data, target definition, and known limitations adequately documented?
3. Does the champion model remain acceptable after challenger comparison, calibration review, and stress testing?
4. Are fairness gaps, disparate impact ratios, and group-level calibration differences acceptable?
5. Are drift thresholds, monitoring frequency, escalation rules, and human oversight requirements defined?
6. Is approval, conditional approval, or rejection supported by the validation evidence?
"""
    )


# =============================================================================
# Footer
# =============================================================================

st.divider()

st.caption(
    "AssuranceTwin AI - Model Validation and AI Governance Dashboard. "
    "Generated from local project artifacts."
)
