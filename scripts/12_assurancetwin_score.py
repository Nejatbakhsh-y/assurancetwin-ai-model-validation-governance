"""
AssuranceTwin AI - Model Validation Governance
Step N: AssuranceTwin Score

Creates:
    reports/tables/assurancetwin_scorecard.csv
    reports/figures/assurancetwin_score_radar.png
    reports/validation/model_risk_committee_summary.md
"""

from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
VALIDATION = REPORTS / "validation"
DOCS = ROOT / "docs"

for folder in [TABLES, FIGURES, VALIDATION, DOCS]:
    folder.mkdir(parents=True, exist_ok=True)

SCORECARD_PATH = TABLES / "assurancetwin_scorecard.csv"
RADAR_PATH = FIGURES / "assurancetwin_score_radar.png"
SUMMARY_PATH = VALIDATION / "model_risk_committee_summary.md"


# ---------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------

WEIGHTS = {
    "Predictive Performance Score": 0.20,
    "Calibration Score": 0.15,
    "Fairness Score": 0.20,
    "Robustness / Stress-Test Score": 0.15,
    "Drift Stability Score": 0.10,
    "Explainability Stability Score": 0.10,
    "Documentation Completeness Score": 0.05,
    "Monitoring Readiness Score": 0.05,
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def clean_name(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def exists_nonempty(path):
    return path.exists() and path.is_file() and path.stat().st_size > 0


def read_csv(path):
    if not exists_nonempty(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def to_float(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    text = str(x).strip().replace(",", "")
    if text.lower() in ["", "nan", "none", "null", "na", "n/a"]:
        return None

    percent = text.endswith("%")
    text = text.replace("%", "")

    try:
        value = float(text)
        if percent:
            value = value / 100.0
        return value
    except Exception:
        return None


def as_prop(x):
    x = float(x)
    if abs(x) > 1 and abs(x) <= 100:
        return x / 100.0
    return x


def clip100(x):
    return float(np.clip(x, 0, 100))


def higher_better_score(x):
    x = as_prop(x)
    return clip100(100 * x)


def lower_better_score(x, bad_level=0.25):
    x = abs(as_prop(x))
    return clip100(100 * (1 - x / bad_level))


def ratio_near_one_score(x):
    x = as_prop(x)
    return clip100(100 * (1 - min(abs(1 - x), 1)))


def psi_score(x):
    x = abs(as_prop(x))
    if x <= 0.10:
        return 100.0
    if x <= 0.25:
        return clip100(100 - ((x - 0.10) / 0.15) * 50)
    return clip100(50 - ((x - 0.25) / 0.25) * 50)


def mean_score(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return 0.0
    return clip100(float(np.mean(values)))


def status(score):
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Acceptable"
    if score >= 55:
        return "Needs remediation"
    return "Insufficient evidence or high risk"


def risk_band(score):
    if score >= 85:
        return "Low model-risk concern"
    if score >= 70:
        return "Moderate model-risk concern"
    if score >= 55:
        return "Elevated model-risk concern"
    return "High model-risk concern"


def source_text(paths):
    found = [str(p.relative_to(ROOT)) for p in paths if exists_nonempty(p)]
    if found:
        return "; ".join(found)
    return "No source file found"


def extract_values(df, keywords):
    if df is None or df.empty:
        return []

    keywords = [clean_name(k) for k in keywords]
    values = []

    columns = list(df.columns)
    clean_columns = {col: clean_name(col) for col in columns}

    # Case 1: metric/value style table
    metric_cols = [
        col for col in columns
        if clean_columns[col] in [
            "metric", "metrics", "measure", "statistic",
            "variable", "name", "component", "test", "indicator"
        ]
    ]

    value_cols = [
        col for col in columns
        if clean_columns[col] in [
            "value", "score", "metric_value", "result",
            "estimate", "mean", "average", "avg"
        ]
    ]

    numeric_cols = [
        col for col in columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    candidate_value_cols = value_cols if value_cols else numeric_cols

    if metric_cols and candidate_value_cols:
        for _, row in df.iterrows():
            metric_text = " ".join(str(row.get(col, "")) for col in metric_cols)
            metric_text = clean_name(metric_text)

            if any(k in metric_text for k in keywords):
                for col in candidate_value_cols:
                    value = to_float(row.get(col))
                    if value is not None:
                        values.append(value)

    # Case 2: wide table where metric names are column names
    for col in columns:
        col_clean = clean_columns[col]
        if any(k in col_clean for k in keywords):
            numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            values.extend(numeric.astype(float).tolist())

    return values


# ---------------------------------------------------------------------
# Component scoring
# ---------------------------------------------------------------------

def predictive_performance_score():
    paths = [
        TABLES / "independent_validation_metrics.csv",
        TABLES / "model_performance_summary.csv",
    ]
    dfs = [read_csv(p) for p in paths]

    metric_groups = [
        ["auc", "roc_auc"],
        ["accuracy"],
        ["precision"],
        ["recall"],
        ["f1", "f1_score"],
        ["balanced_accuracy"],
    ]

    scores = []
    for keywords in metric_groups:
        values = []
        for df in dfs:
            values.extend(extract_values(df, keywords))
        if values:
            scores.append(max(higher_better_score(v) for v in values))

    return mean_score(scores), source_text(paths)


def calibration_score():
    paths = [
        TABLES / "calibration_summary.csv",
        TABLES / "independent_validation_metrics.csv",
    ]
    dfs = [read_csv(p) for p in paths]

    scores = []

    brier = []
    ece = []
    calibration_error = []

    for df in dfs:
        brier.extend(extract_values(df, ["brier", "brier_score"]))
        ece.extend(extract_values(df, ["ece", "expected_calibration_error"]))
        calibration_error.extend(
            extract_values(df, ["calibration_error", "calibration_gap"])
        )

    if brier:
        scores.append(mean_score([lower_better_score(v, 0.35) for v in brier]))

    if ece:
        scores.append(mean_score([lower_better_score(v, 0.20) for v in ece]))

    if calibration_error:
        scores.append(mean_score([lower_better_score(v, 0.20) for v in calibration_error]))

    return mean_score(scores), source_text(paths)


def fairness_score():
    path = TABLES / "fairness_metrics.csv"
    df = read_csv(path)

    scores = []

    gap_metrics = [
        ["approval_rate_difference", "approval_rate_gap", "selection_rate_difference"],
        ["false_positive_rate_gap", "fpr_gap"],
        ["false_negative_rate_gap", "fnr_gap"],
        ["equal_opportunity_difference", "tpr_gap", "recall_gap"],
        ["calibration_gap", "group_calibration_error"],
    ]

    for keywords in gap_metrics:
        values = extract_values(df, keywords)
        if values:
            scores.append(mean_score([lower_better_score(v, 0.25) for v in values]))

    ratio_values = extract_values(
        df,
        ["disparate_impact_ratio", "disparate_impact", "impact_ratio", "di_ratio"]
    )
    if ratio_values:
        scores.append(mean_score([ratio_near_one_score(v) for v in ratio_values]))

    if not scores:
        final = 0.0
    else:
        final = clip100(0.70 * np.mean(scores) + 0.30 * np.min(scores))

    return final, source_text([path])


def robustness_score():
    path = TABLES / "stress_test_results.csv"
    df = read_csv(path)

    scores = []

    drop_values = []
    drop_values.extend(extract_values(df, ["drop", "degradation", "sensitivity"]))
    drop_values.extend(extract_values(df, ["approval_rate_change", "prediction_shift"]))

    if drop_values:
        scores.append(mean_score([lower_better_score(v, 0.30) for v in drop_values]))

    for keywords in [["auc"], ["f1"], ["balanced_accuracy"], ["recall"]]:
        values = extract_values(df, keywords)
        if values:
            scores.append(min(higher_better_score(v) for v in values))

    return mean_score(scores), source_text([path])


def drift_stability_score():
    path = TABLES / "drift_monitoring_summary.csv"
    df = read_csv(path)

    scores = []

    psi_values = extract_values(df, ["psi", "population_stability_index"])
    csi_values = extract_values(df, ["csi", "characteristic_stability_index"])

    if psi_values:
        scores.append(mean_score([psi_score(v) for v in psi_values]))

    if csi_values:
        scores.append(mean_score([psi_score(v) for v in csi_values]))

    drift_metrics = [
        ["data_drift", "feature_drift"],
        ["prediction_drift", "score_drift"],
        ["performance_drift"],
        ["fairness_drift"],
        ["calibration_drift"],
    ]

    for keywords in drift_metrics:
        values = extract_values(df, keywords)
        if values:
            scores.append(mean_score([lower_better_score(v, 0.25) for v in values]))

    return mean_score(scores), source_text([path])


def explainability_stability_score():
    path = TABLES / "explanation_stability.csv"
    df = read_csv(path)

    scores = []

    higher_metrics = [
        ["rank_correlation", "spearman", "kendall", "correlation"],
        ["top_feature_overlap", "feature_overlap", "jaccard"],
        ["stability_score", "explanation_stability"],
    ]

    lower_metrics = [
        ["importance_shift", "feature_importance_shift", "rank_shift"],
        ["shap_shift", "shap_drift", "shap_difference"],
        ["group_explanation_gap", "explanation_gap"],
    ]

    for keywords in higher_metrics:
        values = extract_values(df, keywords)
        if values:
            scores.append(mean_score([higher_better_score(v) for v in values]))

    for keywords in lower_metrics:
        values = extract_values(df, keywords)
        if values:
            scores.append(mean_score([lower_better_score(v, 0.50) for v in values]))

    return mean_score(scores), source_text([path])


def documentation_score():
    required = [
        ROOT / "README.md",
        DOCS / "model_inventory_template.md",
        TABLES / "model_inventory.csv",
        VALIDATION / "model_validation_report.md",
        VALIDATION / "fairness_validation_report.md",
        VALIDATION / "stress_testing_report.md",
        VALIDATION / "monitoring_plan.md",
    ]

    found = [p for p in required if exists_nonempty(p)]
    score = clip100(100 * len(found) / len(required))

    return score, source_text(required)


def monitoring_readiness_score():
    required = [
        VALIDATION / "monitoring_plan.md",
        TABLES / "drift_monitoring_summary.csv",
        FIGURES / "drift_dashboard_plot.png",
    ]

    score = 0

    if exists_nonempty(required[0]):
        score += 45

        text = required[0].read_text(encoding="utf-8", errors="ignore").lower()
        terms = [
            "psi", "csi", "population stability", "characteristic stability",
            "fairness", "calibration", "threshold", "monthly", "quarterly",
            "owner", "escalation", "remediation"
        ]
        hits = sum(1 for t in terms if t in text)
        score += min(10, hits)

    if exists_nonempty(required[1]):
        score += 30

    if exists_nonempty(required[2]):
        score += 15

    return clip100(score), source_text(required)


# ---------------------------------------------------------------------
# Scorecard, radar plot, and committee summary
# ---------------------------------------------------------------------

def build_scorecard():
    component_functions = {
        "Predictive Performance Score": predictive_performance_score,
        "Calibration Score": calibration_score,
        "Fairness Score": fairness_score,
        "Robustness / Stress-Test Score": robustness_score,
        "Drift Stability Score": drift_stability_score,
        "Explainability Stability Score": explainability_stability_score,
        "Documentation Completeness Score": documentation_score,
        "Monitoring Readiness Score": monitoring_readiness_score,
    }

    rows = []

    for component, func in component_functions.items():
        score, evidence = func()
        weight = WEIGHTS[component]
        rows.append(
            {
                "component": component,
                "weight": weight,
                "score_0_to_100": round(score, 2),
                "weighted_points": round(score * weight, 2),
                "status": status(score),
                "evidence_file": evidence,
            }
        )

    df = pd.DataFrame(rows)

    total = float(df["weighted_points"].sum())

    total_row = {
        "component": "AssuranceTwin Score",
        "weight": 1.00,
        "score_0_to_100": round(total, 2),
        "weighted_points": round(total, 2),
        "status": status(total),
        "evidence_file": "Aggregate score from all model-governance components",
    }

    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df


def create_radar_plot(scorecard):
    df = scorecard[scorecard["component"] != "AssuranceTwin Score"].copy()

    labels = df["component"].tolist()
    values = df["score_0_to_100"].astype(float).tolist()

    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    ax.set_title("AssuranceTwin Score Radar", fontsize=14, pad=24)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(RADAR_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def markdown_scorecard(scorecard):
    lines = []
    lines.append("| Component | Weight | Score | Weighted Points | Status |")
    lines.append("|---|---:|---:|---:|---|")

    for _, row in scorecard.iterrows():
        lines.append(
            "| {} | {:.2f} | {:.2f} | {:.2f} | {} |".format(
                row["component"],
                float(row["weight"]),
                float(row["score_0_to_100"]),
                float(row["weighted_points"]),
                row["status"],
            )
        )

    return "\n".join(lines)


def create_committee_summary(scorecard):
    total = float(
        scorecard.loc[
            scorecard["component"] == "AssuranceTwin Score",
            "score_0_to_100"
        ].iloc[0]
    )

    if total >= 85:
        recommendation = "Approve for controlled production use with standard monitoring."
    elif total >= 70:
        recommendation = "Conditionally approve, subject to remediation tracking and enhanced monitoring."
    elif total >= 55:
        recommendation = "Do not approve for broad production use until remediation is completed."
    else:
        recommendation = "Do not approve. Validation evidence is insufficient or the model-risk profile is too high."

    component_df = scorecard[scorecard["component"] != "AssuranceTwin Score"].copy()

    weakest = component_df.sort_values("score_0_to_100").head(3)
    strongest = component_df.sort_values("score_0_to_100", ascending=False).head(3)

    weak_lines = []
    for _, row in weakest.iterrows():
        weak_lines.append(
            "- {}: {:.2f} ({})".format(
                row["component"],
                float(row["score_0_to_100"]),
                row["status"],
            )
        )

    strong_lines = []
    for _, row in strongest.iterrows():
        strong_lines.append(
            "- {}: {:.2f} ({})".format(
                row["component"],
                float(row["score_0_to_100"]),
                row["status"],
            )
        )

    evidence_lines = []
    for _, row in component_df.iterrows():
        evidence_lines.append(
            "- **{}**: {}".format(row["component"], row["evidence_file"])
        )

    action_items = []
    weak_components = component_df[component_df["score_0_to_100"] < 70]

    if weak_components.empty:
        action_items.append(
            "- Maintain periodic validation refresh, monitoring evidence, and change-management controls."
        )
    else:
        for _, row in weak_components.iterrows():
            action_items.append(
                "- Remediate **{}**. Current score: {:.2f}.".format(
                    row["component"],
                    float(row["score_0_to_100"]),
                )
            )

    formula_lines = [
        "AssuranceTwin Score =",
        "0.20 * Predictive Performance",
        "+ 0.15 * Calibration",
        "+ 0.20 * Fairness",
        "+ 0.15 * Robustness / Stress-Test",
        "+ 0.10 * Drift Stability",
        "+ 0.10 * Explainability Stability",
        "+ 0.05 * Documentation Completeness",
        "+ 0.05 * Monitoring Readiness",
    ]

    lines = []
    lines.append("# Model Risk Committee Summary")
    lines.append("")
    lines.append("Generated: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("")
    lines.append("## Executive Decision View")
    lines.append("")
    lines.append("**AssuranceTwin Score:** {:.2f} / 100".format(total))
    lines.append("")
    lines.append("**Risk Band:** {}".format(risk_band(total)))
    lines.append("")
    lines.append("**Recommended Committee Action:** {}".format(recommendation))
    lines.append("")
    lines.append(
        "The AssuranceTwin Score is a composite model-governance score that combines predictive performance, calibration, fairness, stress robustness, drift stability, explainability stability, documentation completeness, and monitoring readiness."
    )
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append(markdown_scorecard(scorecard))
    lines.append("")
    lines.append("## Main Strengths")
    lines.append("")
    lines.extend(strong_lines)
    lines.append("")
    lines.append("## Main Weaknesses / Remediation Priorities")
    lines.append("")
    lines.extend(weak_lines)
    lines.append("")
    lines.append("## Required Committee Action Items")
    lines.append("")
    lines.extend(action_items)
    lines.append("")
    lines.append("## Evidence Used")
    lines.append("")
    lines.extend(evidence_lines)
    lines.append("")
    lines.append("## Scoring Formula")
    lines.append("")
    lines.append("```text")
    lines.extend(formula_lines)
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- **85 to 100:** Strong model-assurance posture.")
    lines.append("- **70 to 84.99:** Acceptable, with active monitoring and documented limitations.")
    lines.append("- **55 to 69.99:** Elevated model-risk concern; remediation should precede broad deployment.")
    lines.append("- **Below 55:** High model-risk concern or insufficient validation evidence.")
    lines.append("")
    lines.append("## Governance Limitation")
    lines.append("")
    lines.append(
        "This score is a structured decision-support artifact, not an automatic model approval mechanism. Final approval should also consider model materiality, regulatory exposure, business use, implementation controls, change management, and independent validation sign-off."
    )
    lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    scorecard = build_scorecard()

    scorecard.to_csv(SCORECARD_PATH, index=False)
    create_radar_plot(scorecard)
    create_committee_summary(scorecard)

    total = float(
        scorecard.loc[
            scorecard["component"] == "AssuranceTwin Score",
            "score_0_to_100"
        ].iloc[0]
    )

    print("")
    print("Step N complete: AssuranceTwin Score created.")
    print("AssuranceTwin Score: {:.2f} / 100".format(total))
    print("Risk band: {}".format(risk_band(total)))
    print("")
    print("Outputs created:")
    print("- {}".format(SCORECARD_PATH.relative_to(ROOT)))
    print("- {}".format(RADAR_PATH.relative_to(ROOT)))
    print("- {}".format(SUMMARY_PATH.relative_to(ROOT)))
    print("")
    print(scorecard[["component", "weight", "score_0_to_100", "weighted_points", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()