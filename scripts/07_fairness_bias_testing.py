"""
Step I — Fairness and Bias Testing

Purpose
-------
This script simulates an independent AI/model-governance fairness review for a
high-stakes HMDA approval model.

It tests group-level performance by:
    - race
    - ethnicity
    - sex
    - income band
    - minority-tract band

Metrics produced:
    - approval-rate difference
    - disparate impact ratio
    - false-negative-rate gap
    - false-positive-rate gap
    - equal opportunity difference
    - calibration by group

Outputs
-------
    reports/validation/fairness_validation_report.md
    reports/tables/fairness_metrics.csv
    reports/figures/fairness_group_comparison.png
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hmda_modeling_dataset.csv"

MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "champion_model.pkl",
    PROJECT_ROOT / "models" / "champion_model.joblib",
    PROJECT_ROOT / "models" / "best_model.pkl",
    PROJECT_ROOT / "models" / "best_model.joblib",
    PROJECT_ROOT / "models" / "challenger_model.pkl",
]

REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "fairness_validation_report.md"
TABLE_PATH = PROJECT_ROOT / "reports" / "tables" / "fairness_metrics.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "fairness_group_comparison.png"

TARGET_CANDIDATES = [
    "approved",
    "approval",
    "loan_approved",
    "target",
    "y",
    "action_taken",
]

PROTECTED_ATTRIBUTES = [
    "race",
    "ethnicity",
    "sex",
    "income_band",
    "minority_tract_band",
]

PREDICTION_PROBABILITY_CANDIDATES = [
    "approval_probability",
    "predicted_probability",
    "predicted_approval_probability",
    "probability_approved",
    "champion_probability",
    "model_probability",
    "y_probability",
    "y_score",
    "score",
]

PREDICTION_LABEL_CANDIDATES = [
    "predicted_approved",
    "approval_prediction",
    "predicted_label",
    "prediction",
    "champion_prediction",
    "model_prediction",
    "y_pred",
]

MIN_GROUP_SIZE = 30
DEFAULT_THRESHOLD = 0.50


# =============================================================================
# Utility functions
# =============================================================================

def ensure_output_directories() -> None:
    """Create output directories if they do not already exist."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    """Normalize a column name for flexible matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column using case-insensitive and separator-insensitive matching."""
    normalized_lookup = {normalize_name(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized_lookup:
            return normalized_lookup[key]

    return None


def clean_group_labels(series: pd.Series) -> pd.Series:
    """Clean group labels and replace missing values with a governance-friendly label."""
    out = series.astype("object").copy()
    out = out.where(pd.notna(out), "Missing/Unknown")
    out = out.astype(str).str.strip()

    missing_like = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "missing",
        "not available",
        "not_available",
    }

    out = out.apply(
        lambda x: "Missing/Unknown"
        if normalize_name(x) in missing_like
        else x
    )

    return out


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers and return NaN if the denominator is zero."""
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return float("nan")
    return float(numerator) / float(denominator)


def safe_percentage(value: float) -> str:
    """Format a decimal as a percentage string."""
    if pd.isna(value):
        return "NA"
    return f"{100 * value:.2f}%"


def safe_number(value: float, digits: int = 4) -> str:
    """Format a number safely for markdown reports."""
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """
    Convert a DataFrame to a simple markdown table without requiring tabulate.
    """
    if df.empty:
        return "_No rows available._"

    table_df = df.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)

    table_df = table_df.fillna("NA").astype(str)

    headers = list(table_df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    body_rows = []
    for _, row in table_df.iterrows():
        body_rows.append("| " + " | ".join(row.values.tolist()) + " |")

    return "\n".join([header_row, separator_row] + body_rows)


# =============================================================================
# Protected-group derivation
# =============================================================================

def map_race(series: pd.Series) -> pd.Series:
    """Map HMDA applicant race codes to broad group labels where possible."""
    race_map = {
        1: "American Indian or Alaska Native",
        2: "Asian",
        21: "Asian",
        22: "Asian",
        23: "Asian",
        24: "Asian",
        25: "Asian",
        26: "Asian",
        27: "Asian",
        3: "Black or African American",
        4: "Native Hawaiian or Other Pacific Islander",
        41: "Native Hawaiian or Other Pacific Islander",
        42: "Native Hawaiian or Other Pacific Islander",
        43: "Native Hawaiian or Other Pacific Islander",
        44: "Native Hawaiian or Other Pacific Islander",
        5: "White",
        6: "Information not provided",
        7: "Not applicable",
    }

    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.round().astype("Int64").map(race_map)

    out = series.astype("object").copy()
    out = out.where(mapped.isna(), mapped)
    return clean_group_labels(out)


def map_ethnicity(series: pd.Series) -> pd.Series:
    """Map HMDA applicant ethnicity codes to broad group labels where possible."""
    ethnicity_map = {
        1: "Hispanic or Latino",
        11: "Hispanic or Latino",
        12: "Hispanic or Latino",
        13: "Hispanic or Latino",
        14: "Hispanic or Latino",
        2: "Not Hispanic or Latino",
        3: "Information not provided",
        4: "Not applicable",
    }

    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.round().astype("Int64").map(ethnicity_map)

    out = series.astype("object").copy()
    out = out.where(mapped.isna(), mapped)
    return clean_group_labels(out)


def map_sex(series: pd.Series) -> pd.Series:
    """Map HMDA applicant sex codes to group labels where possible."""
    sex_map = {
        1: "Male",
        2: "Female",
        3: "Information not provided",
        4: "Not applicable",
        6: "Applicant selected both male and female",
    }

    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.round().astype("Int64").map(sex_map)

    out = series.astype("object").copy()
    out = out.where(mapped.isna(), mapped)
    return clean_group_labels(out)


def derive_income_band(series: pd.Series) -> pd.Series:
    """
    Derive applicant income bands.

    HMDA income is commonly reported in thousands of dollars. If the values look
    like raw dollar amounts, this function converts them to thousands first.
    """
    income = pd.to_numeric(series, errors="coerce")

    if income.dropna().empty:
        return pd.Series(["Missing/Unknown"] * len(series), index=series.index)

    median_income = income.dropna().median()

    # If values appear to be raw dollars rather than thousands, convert to 000s.
    if median_income > 1000:
        income = income / 1000.0

    bins = [-np.inf, 50, 100, 150, np.inf]
    labels = [
        "Income <= $50K",
        "$50K < Income <= $100K",
        "$100K < Income <= $150K",
        "Income > $150K",
    ]

    band = pd.cut(income, bins=bins, labels=labels)
    band = band.astype("object").where(pd.notna(band), "Missing/Unknown")

    return clean_group_labels(band)


def derive_minority_tract_band(series: pd.Series) -> pd.Series:
    """
    Derive minority-tract bands from tract minority population percentage.

    The function accepts either 0-1 proportions or 0-100 percentages.
    """
    minority_pct = pd.to_numeric(series, errors="coerce")

    if minority_pct.dropna().empty:
        return pd.Series(["Missing/Unknown"] * len(series), index=series.index)

    if minority_pct.dropna().max() <= 1:
        minority_pct = minority_pct * 100.0

    bins = [-np.inf, 20, 50, 80, np.inf]
    labels = [
        "Minority tract <= 20%",
        "20% < Minority tract <= 50%",
        "50% < Minority tract <= 80%",
        "Minority tract > 80%",
    ]

    band = pd.cut(minority_pct, bins=bins, labels=labels)
    band = band.astype("object").where(pd.notna(band), "Missing/Unknown")

    return clean_group_labels(band)


def add_protected_group_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add canonical protected-group columns:
        race
        ethnicity
        sex
        income_band
        minority_tract_band
    """
    df = df.copy()
    notes: list[str] = []

    race_source = find_column(
        df,
        [
            "race",
            "race_group",
            "applicant_race",
            "applicant_race_1",
            "applicant_race-1",
            "derived_race",
            "derived_race_ethnicity",
        ],
    )
    if race_source:
        df["race"] = map_race(df[race_source])
        notes.append(f"race derived from `{race_source}`.")
    else:
        notes.append("race could not be derived because no race-like column was found.")

    ethnicity_source = find_column(
        df,
        [
            "ethnicity",
            "ethnicity_group",
            "applicant_ethnicity",
            "applicant_ethnicity_1",
            "applicant_ethnicity-1",
            "derived_ethnicity",
        ],
    )
    if ethnicity_source:
        df["ethnicity"] = map_ethnicity(df[ethnicity_source])
        notes.append(f"ethnicity derived from `{ethnicity_source}`.")
    else:
        notes.append("ethnicity could not be derived because no ethnicity-like column was found.")

    sex_source = find_column(
        df,
        [
            "sex",
            "sex_group",
            "applicant_sex",
            "applicant_sex_1",
            "applicant_sex-1",
            "derived_sex",
        ],
    )
    if sex_source:
        df["sex"] = map_sex(df[sex_source])
        notes.append(f"sex derived from `{sex_source}`.")
    else:
        notes.append("sex could not be derived because no sex-like column was found.")

    income_band_source = find_column(
        df,
        [
            "income_band",
            "applicant_income_band",
            "income_group",
            "borrower_income_band",
        ],
    )
    income_source = find_column(
        df,
        [
            "income",
            "applicant_income",
            "borrower_income",
            "applicant_income_000s",
        ],
    )

    if income_band_source:
        df["income_band"] = clean_group_labels(df[income_band_source])
        notes.append(f"income_band taken from `{income_band_source}`.")
    elif income_source:
        df["income_band"] = derive_income_band(df[income_source])
        notes.append(f"income_band derived from `{income_source}`.")
    else:
        notes.append("income_band could not be derived because no income-like column was found.")

    minority_band_source = find_column(
        df,
        [
            "minority_tract_band",
            "tract_minority_band",
            "minority_population_band",
            "tract_minority_population_band",
        ],
    )
    minority_source = find_column(
        df,
        [
            "tract_minority_population_percent",
            "tract_minority_population_pct",
            "minority_population_percent",
            "minority_population_pct",
            "tract_minority_percent",
            "tract_minority_pct",
        ],
    )

    if minority_band_source:
        df["minority_tract_band"] = clean_group_labels(df[minority_band_source])
        notes.append(f"minority_tract_band taken from `{minority_band_source}`.")
    elif minority_source:
        df["minority_tract_band"] = derive_minority_tract_band(df[minority_source])
        notes.append(f"minority_tract_band derived from `{minority_source}`.")
    else:
        notes.append(
            "minority_tract_band could not be derived because no minority-tract-like column was found."
        )

    return df, notes


# =============================================================================
# Target and prediction handling
# =============================================================================

def convert_action_taken_to_target(series: pd.Series) -> pd.Series:
    """
    Convert HMDA action_taken to an approval target.

    Approval-related HMDA action_taken codes:
        1 = Loan originated
        2 = Application approved but not accepted
        8 = Preapproval request approved but not accepted
    """
    numeric = pd.to_numeric(series, errors="coerce")

    approved_codes = {1, 2, 8}

    if numeric.notna().any():
        y = numeric.round().astype("Int64").apply(
            lambda x: 1 if pd.notna(x) and int(x) in approved_codes else 0
        )
        return y.astype(int)

    text = series.astype(str).str.lower().str.strip()

    approved_text_patterns = [
        "loan originated",
        "approved but not accepted",
        "preapproval request approved",
        "approved",
        "originated",
    ]

    denied_text_patterns = [
        "denied",
        "withdrawn",
        "incomplete",
        "purchased loan",
        "preapproval request denied",
        "not applicable",
    ]

    y = pd.Series(np.nan, index=series.index)

    for pattern in approved_text_patterns:
        y = y.where(~text.str.contains(pattern, na=False), 1)

    for pattern in denied_text_patterns:
        y = y.where(~text.str.contains(pattern, na=False), 0)

    return y


def convert_binary_like_to_target(series: pd.Series) -> pd.Series:
    """Convert common binary-like values to 0/1."""
    numeric = pd.to_numeric(series, errors="coerce")

    unique_numeric = set(numeric.dropna().unique().tolist())
    if unique_numeric and unique_numeric.issubset({0, 1, 0.0, 1.0}):
        return numeric.astype("Int64")

    text = series.astype(str).str.lower().str.strip()

    positive_values = {
        "1",
        "true",
        "yes",
        "y",
        "approved",
        "approve",
        "loan approved",
        "originated",
        "loan originated",
    }

    negative_values = {
        "0",
        "false",
        "no",
        "n",
        "denied",
        "deny",
        "not approved",
        "rejected",
    }

    y = pd.Series(np.nan, index=series.index)

    y = y.where(~text.isin(positive_values), 1)
    y = y.where(~text.isin(negative_values), 0)

    return y


def get_target(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Find and convert the target column."""
    target_col = find_column(df, TARGET_CANDIDATES)

    if not target_col:
        raise ValueError(
            "No target column found. Expected one of: "
            + ", ".join(TARGET_CANDIDATES)
        )

    if normalize_name(target_col) == "action_taken":
        y = convert_action_taken_to_target(df[target_col])
    else:
        y = convert_binary_like_to_target(df[target_col])

    valid = y.notna()
    if valid.sum() == 0:
        raise ValueError(
            f"Target column `{target_col}` was found, but it could not be converted to binary 0/1."
        )

    return target_col, y


def load_model_payload() -> tuple[Any, list[str] | None, float, str]:
    """
    Load the champion model or an available fallback model.

    The script supports either:
        - a plain sklearn estimator/pipeline
        - a dictionary payload containing model and feature metadata
    """
    model_path = None
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(
            "No model file found. Expected one of:\n"
            + "\n".join(str(path) for path in MODEL_CANDIDATES)
        )

    payload = joblib.load(model_path)

    feature_columns = None
    threshold = DEFAULT_THRESHOLD

    if isinstance(payload, dict):
        model = None

        for key in [
            "model",
            "champion_model",
            "best_model",
            "estimator",
            "pipeline",
            "trained_model",
        ]:
            if key in payload:
                model = payload[key]
                break

        if model is None:
            raise ValueError(
                f"Model payload at {model_path} is a dictionary, but no model object was found."
            )

        for key in [
            "feature_columns",
            "features",
            "model_features",
            "training_features",
            "selected_features",
        ]:
            if key in payload:
                feature_columns = list(payload[key])
                break

        for key in ["threshold", "classification_threshold", "decision_threshold"]:
            if key in payload:
                try:
                    threshold = float(payload[key])
                except Exception:
                    threshold = DEFAULT_THRESHOLD
                break
    else:
        model = payload

    if feature_columns is None and hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)

    return model, feature_columns, threshold, str(model_path)


def get_positive_class_index(model: Any) -> int:
    """Find the probability column corresponding to the positive class."""
    classes = getattr(model, "classes_", None)

    if classes is None:
        return 1

    class_values = [str(value).lower() for value in list(classes)]

    for positive_label in ["1", "true", "approved", "yes"]:
        if positive_label in class_values:
            return class_values.index(positive_label)

    if len(class_values) > 1:
        return 1

    return 0


def predict_from_model(
    model: Any,
    x: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate probability and class predictions from a fitted model."""
    if hasattr(model, "predict_proba"):
        proba_raw = model.predict_proba(x)

        if len(proba_raw.shape) == 1:
            probability = np.asarray(proba_raw, dtype=float)
        else:
            positive_index = get_positive_class_index(model)
            probability = np.asarray(proba_raw[:, positive_index], dtype=float)

        prediction = (probability >= threshold).astype(int)
        return probability, prediction

    if hasattr(model, "decision_function"):
        decision_score = np.asarray(model.decision_function(x), dtype=float)
        probability = 1.0 / (1.0 + np.exp(-decision_score))
        prediction = (probability >= threshold).astype(int)
        return probability, prediction

    raw_prediction = model.predict(x)
    prediction_series = convert_binary_like_to_target(pd.Series(raw_prediction))
    prediction = prediction_series.fillna(0).astype(int).to_numpy()
    probability = prediction.astype(float)

    return probability, prediction


def get_predictions(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
    Obtain model predictions.

    Priority:
        1. Use existing prediction probability column if present.
        2. Use existing prediction label column if present.
        3. Load the saved champion model and generate predictions.
    """
    probability_col = find_column(df, PREDICTION_PROBABILITY_CANDIDATES)
    prediction_col = find_column(df, PREDICTION_LABEL_CANDIDATES)

    if probability_col:
        probability = pd.to_numeric(df[probability_col], errors="coerce").fillna(0.0).to_numpy()
        prediction = (probability >= DEFAULT_THRESHOLD).astype(int)
        return (
            probability,
            prediction,
            f"Existing probability column `{probability_col}`",
            DEFAULT_THRESHOLD,
        )

    if prediction_col:
        prediction_series = convert_binary_like_to_target(df[prediction_col])
        prediction = prediction_series.fillna(0).astype(int).to_numpy()
        probability = prediction.astype(float)
        return (
            probability,
            prediction,
            f"Existing prediction column `{prediction_col}`",
            DEFAULT_THRESHOLD,
        )

    model, feature_columns, threshold, model_source = load_model_payload()

    candidate_feature_sets: list[tuple[str, pd.DataFrame]] = []

    if feature_columns:
        x_saved = df.reindex(columns=feature_columns)
        candidate_feature_sets.append(("saved feature columns", x_saved))

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        x_model_features = df.reindex(columns=model_features)
        candidate_feature_sets.append(("model feature_names_in_", x_model_features))

    non_feature_columns = set(
        [target_col]
        + PROTECTED_ATTRIBUTES
        + PREDICTION_PROBABILITY_CANDIDATES
        + PREDICTION_LABEL_CANDIDATES
    )

    general_feature_columns = [
        col for col in df.columns if col not in non_feature_columns
    ]

    if general_feature_columns:
        candidate_feature_sets.append(
            ("all non-target non-group columns", df[general_feature_columns])
        )

    numeric_feature_columns = [
        col
        for col in general_feature_columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if numeric_feature_columns:
        candidate_feature_sets.append(
            ("numeric non-target non-group columns", df[numeric_feature_columns])
        )

    errors = []

    for feature_set_name, x_candidate in candidate_feature_sets:
        try:
            probability, prediction = predict_from_model(
                model=model,
                x=x_candidate,
                threshold=threshold,
            )
            return (
                probability,
                prediction,
                f"{model_source} using {feature_set_name}",
                threshold,
            )
        except Exception as exc:
            errors.append(f"{feature_set_name}: {exc}")

    raise RuntimeError(
        "The saved model was found, but predictions could not be generated.\n\n"
        "Attempted feature sets:\n"
        + "\n".join(errors)
    )


# =============================================================================
# Fairness metrics
# =============================================================================

def compute_group_base_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
    group: pd.Series,
) -> pd.DataFrame:
    """Compute raw performance metrics for each group."""
    rows = []

    working = pd.DataFrame(
        {
            "y_true": y_true.astype(int).to_numpy(),
            "y_pred": np.asarray(y_pred, dtype=int),
            "y_probability": np.asarray(y_probability, dtype=float),
            "group": clean_group_labels(group),
        }
    )

    for group_name, group_df in working.groupby("group", dropna=False):
        yt = group_df["y_true"].astype(int)
        yp = group_df["y_pred"].astype(int)
        prob = group_df["y_probability"].astype(float)

        n = int(len(group_df))
        positives = int((yt == 1).sum())
        negatives = int((yt == 0).sum())

        if n == 0:
            continue

        tn, fp, fn, tp = confusion_matrix(
            yt,
            yp,
            labels=[0, 1],
        ).ravel()

        approval_rate = float(yp.mean())
        actual_approval_rate = float(yt.mean())

        false_positive_rate = safe_divide(fp, fp + tn)
        false_negative_rate = safe_divide(fn, fn + tp)
        true_positive_rate = safe_divide(tp, tp + fn)
        true_negative_rate = safe_divide(tn, tn + fp)

        mean_predicted_probability = float(prob.mean())
        calibration_error = abs(mean_predicted_probability - actual_approval_rate)

        try:
            group_brier_score = brier_score_loss(yt, prob)
        except Exception:
            group_brier_score = float("nan")

        rows.append(
            {
                "group": str(group_name),
                "n": n,
                "actual_positives": positives,
                "actual_negatives": negatives,
                "actual_approval_rate": actual_approval_rate,
                "predicted_approval_rate": approval_rate,
                "true_positive_rate": true_positive_rate,
                "true_negative_rate": true_negative_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "mean_predicted_probability": mean_predicted_probability,
                "calibration_error": calibration_error,
                "brier_score": group_brier_score,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "small_group_flag": n < MIN_GROUP_SIZE,
            }
        )

    return pd.DataFrame(rows)


def compute_fairness_metrics_for_attribute(
    attribute: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
    group: pd.Series,
) -> pd.DataFrame:
    """
    Compute fairness metrics for one protected attribute.

    Reference group:
        The group with the highest predicted approval rate among groups with
        at least MIN_GROUP_SIZE observations. If no group satisfies the size
        threshold, the largest group is used.
    """
    base = compute_group_base_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
        group=group,
    )

    if base.empty:
        return base

    eligible = base[base["n"] >= MIN_GROUP_SIZE].copy()

    if eligible.empty:
        reference_row = base.sort_values("n", ascending=False).iloc[0]
    else:
        reference_row = eligible.sort_values(
            ["predicted_approval_rate", "n"],
            ascending=[False, False],
        ).iloc[0]

    reference_group = reference_row["group"]
    reference_approval_rate = reference_row["predicted_approval_rate"]
    reference_fnr = reference_row["false_negative_rate"]
    reference_fpr = reference_row["false_positive_rate"]
    reference_tpr = reference_row["true_positive_rate"]

    base["attribute"] = attribute
    base["reference_group"] = reference_group

    base["approval_rate_difference"] = (
        base["predicted_approval_rate"] - reference_approval_rate
    )

    base["disparate_impact_ratio"] = base["predicted_approval_rate"].apply(
        lambda value: safe_divide(value, reference_approval_rate)
    )

    base["false_negative_rate_gap"] = base["false_negative_rate"] - reference_fnr
    base["false_positive_rate_gap"] = base["false_positive_rate"] - reference_fpr

    base["equal_opportunity_difference"] = (
        base["true_positive_rate"] - reference_tpr
    )

    base["adverse_impact_flag"] = base["disparate_impact_ratio"].apply(
        lambda value: bool(pd.notna(value) and value < 0.80)
    )

    base["large_fnr_gap_flag"] = base["false_negative_rate_gap"].apply(
        lambda value: bool(pd.notna(value) and abs(value) >= 0.10)
    )

    base["large_fpr_gap_flag"] = base["false_positive_rate_gap"].apply(
        lambda value: bool(pd.notna(value) and abs(value) >= 0.10)
    )

    base["large_equal_opportunity_gap_flag"] = base[
        "equal_opportunity_difference"
    ].apply(
        lambda value: bool(pd.notna(value) and abs(value) >= 0.10)
    )

    base["large_calibration_error_flag"] = base["calibration_error"].apply(
        lambda value: bool(pd.notna(value) and value >= 0.10)
    )

    base["governance_attention_flag"] = (
        base["adverse_impact_flag"]
        | base["large_fnr_gap_flag"]
        | base["large_fpr_gap_flag"]
        | base["large_equal_opportunity_gap_flag"]
        | base["large_calibration_error_flag"]
    )

    ordered_columns = [
        "attribute",
        "group",
        "reference_group",
        "n",
        "small_group_flag",
        "actual_positives",
        "actual_negatives",
        "actual_approval_rate",
        "predicted_approval_rate",
        "approval_rate_difference",
        "disparate_impact_ratio",
        "true_positive_rate",
        "equal_opportunity_difference",
        "false_negative_rate",
        "false_negative_rate_gap",
        "false_positive_rate",
        "false_positive_rate_gap",
        "mean_predicted_probability",
        "calibration_error",
        "brier_score",
        "tp",
        "fp",
        "tn",
        "fn",
        "adverse_impact_flag",
        "large_fnr_gap_flag",
        "large_fpr_gap_flag",
        "large_equal_opportunity_gap_flag",
        "large_calibration_error_flag",
        "governance_attention_flag",
    ]

    return base[ordered_columns].sort_values(
        ["attribute", "predicted_approval_rate"],
        ascending=[True, True],
    )


def compute_all_fairness_metrics(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> pd.DataFrame:
    """Compute fairness metrics for all available protected attributes."""
    all_results = []

    available_attributes = [
        attribute for attribute in PROTECTED_ATTRIBUTES if attribute in df.columns
    ]

    if not available_attributes:
        raise ValueError(
            "No protected-group columns were available for fairness testing."
        )

    for attribute in available_attributes:
        result = compute_fairness_metrics_for_attribute(
            attribute=attribute,
            y_true=y_true,
            y_pred=y_pred,
            y_probability=y_probability,
            group=df[attribute],
        )

        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise ValueError("Fairness metrics could not be computed for any attribute.")

    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# Figure and report
# =============================================================================

def create_fairness_figure(fairness_df: pd.DataFrame) -> None:
    """Create a horizontal bar chart comparing predicted approval rates by group."""
    plot_df = fairness_df.copy()

    if plot_df.empty:
        plt.figure(figsize=(10, 4))
        plt.text(
            0.5,
            0.5,
            "No fairness metrics available.",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
        plt.close()
        return

    plot_df = plot_df[plot_df["n"] >= MIN_GROUP_SIZE].copy()

    if plot_df.empty:
        plot_df = fairness_df.copy()

    plot_df["label"] = plot_df["attribute"] + ": " + plot_df["group"].astype(str)

    plot_df = plot_df.sort_values(
        ["attribute", "predicted_approval_rate"],
        ascending=[True, True],
    )

    # Keep figure readable if many groups exist.
    if len(plot_df) > 35:
        plot_df = plot_df.sort_values(
            "predicted_approval_rate",
            ascending=True,
        ).head(35)

    height = max(6, 0.35 * len(plot_df))

    plt.figure(figsize=(12, height))
    plt.barh(plot_df["label"], plot_df["predicted_approval_rate"])
    plt.xlabel("Predicted Approval Rate")
    plt.ylabel("Protected Attribute and Group")
    plt.title("Fairness Group Comparison: Predicted Approval Rates")
    plt.xlim(0, 1)

    overall_approval_rate = fairness_df["predicted_approval_rate"].mean()
    if not pd.isna(overall_approval_rate):
        plt.axvline(
            overall_approval_rate,
            linestyle="--",
            linewidth=1,
            label="Mean group approval rate",
        )
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def compute_overall_model_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Compute high-level overall model metrics for report context."""
    y_true_array = y_true.astype(int).to_numpy()
    y_pred_array = np.asarray(y_pred, dtype=int)
    y_probability_array = np.asarray(y_probability, dtype=float)

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true_array, y_pred_array)
    metrics["overall_approval_rate"] = float(y_pred_array.mean())
    metrics["actual_approval_rate"] = float(y_true_array.mean())

    try:
        metrics["auc"] = roc_auc_score(y_true_array, y_probability_array)
    except Exception:
        metrics["auc"] = float("nan")

    try:
        metrics["brier_score"] = brier_score_loss(
            y_true_array,
            y_probability_array,
        )
    except Exception:
        metrics["brier_score"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
    ).ravel()

    metrics["false_positive_rate"] = safe_divide(fp, fp + tn)
    metrics["false_negative_rate"] = safe_divide(fn, fn + tp)
    metrics["true_positive_rate"] = safe_divide(tp, tp + fn)

    return metrics


def summarize_fairness_by_attribute(fairness_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact attribute-level summary for the markdown report."""
    rows = []

    for attribute, sub_df in fairness_df.groupby("attribute"):
        eligible = sub_df[sub_df["n"] >= MIN_GROUP_SIZE].copy()
        if eligible.empty:
            eligible = sub_df.copy()

        min_dir_row = eligible.sort_values(
            "disparate_impact_ratio",
            ascending=True,
        ).iloc[0]

        max_calibration_row = eligible.sort_values(
            "calibration_error",
            ascending=False,
        ).iloc[0]

        max_fnr_gap_row = eligible.assign(
            abs_fnr_gap=eligible["false_negative_rate_gap"].abs()
        ).sort_values("abs_fnr_gap", ascending=False).iloc[0]

        max_fpr_gap_row = eligible.assign(
            abs_fpr_gap=eligible["false_positive_rate_gap"].abs()
        ).sort_values("abs_fpr_gap", ascending=False).iloc[0]

        max_eod_row = eligible.assign(
            abs_eod=eligible["equal_opportunity_difference"].abs()
        ).sort_values("abs_eod", ascending=False).iloc[0]

        rows.append(
            {
                "attribute": attribute,
                "groups_tested": int(sub_df["group"].nunique()),
                "reference_group": str(sub_df["reference_group"].iloc[0]),
                "lowest_dir_group": str(min_dir_row["group"]),
                "lowest_disparate_impact_ratio": safe_number(
                    min_dir_row["disparate_impact_ratio"]
                ),
                "largest_fnr_gap_group": str(max_fnr_gap_row["group"]),
                "largest_fnr_gap": safe_number(
                    max_fnr_gap_row["false_negative_rate_gap"]
                ),
                "largest_fpr_gap_group": str(max_fpr_gap_row["group"]),
                "largest_fpr_gap": safe_number(
                    max_fpr_gap_row["false_positive_rate_gap"]
                ),
                "largest_equal_opportunity_gap_group": str(max_eod_row["group"]),
                "largest_equal_opportunity_difference": safe_number(
                    max_eod_row["equal_opportunity_difference"]
                ),
                "largest_calibration_error_group": str(max_calibration_row["group"]),
                "largest_calibration_error": safe_number(
                    max_calibration_row["calibration_error"]
                ),
            }
        )

    return pd.DataFrame(rows)


def write_report(
    fairness_df: pd.DataFrame,
    overall_metrics: dict[str, float],
    protected_notes: list[str],
    target_col: str,
    prediction_source: str,
    threshold: float,
    n_records: int,
) -> None:
    """Write the fairness validation markdown report."""
    summary_df = summarize_fairness_by_attribute(fairness_df)

    flagged_df = fairness_df[
        fairness_df["governance_attention_flag"] == True
    ].copy()

    flagged_display = flagged_df[
        [
            "attribute",
            "group",
            "reference_group",
            "n",
            "predicted_approval_rate",
            "approval_rate_difference",
            "disparate_impact_ratio",
            "false_negative_rate_gap",
            "false_positive_rate_gap",
            "equal_opportunity_difference",
            "calibration_error",
        ]
    ].copy()

    for col in [
        "predicted_approval_rate",
        "approval_rate_difference",
        "false_negative_rate_gap",
        "false_positive_rate_gap",
        "equal_opportunity_difference",
        "calibration_error",
    ]:
        flagged_display[col] = flagged_display[col].apply(safe_number)

    flagged_display["disparate_impact_ratio"] = flagged_display[
        "disparate_impact_ratio"
    ].apply(safe_number)

    summary_display = summary_df.copy()

    report_lines = []

    report_lines.append("# Fairness and Bias Validation Report")
    report_lines.append("")
    report_lines.append("## 1. Validation Purpose")
    report_lines.append("")
    report_lines.append(
        "This report evaluates whether the HMDA approval model creates "
        "material group-level disparities even when overall model performance "
        "appears acceptable. This is a central AI-governance concern for "
        "high-stakes credit and lending models."
    )
    report_lines.append("")

    report_lines.append("## 2. Data and Model Scope")
    report_lines.append("")
    report_lines.append(f"- Modeling dataset: `{DATA_PATH.as_posix()}`")
    report_lines.append(f"- Number of evaluated records: `{n_records:,}`")
    report_lines.append(f"- Target column: `{target_col}`")
    report_lines.append(f"- Prediction source: `{prediction_source}`")
    report_lines.append(f"- Classification threshold: `{threshold:.2f}`")
    report_lines.append("")

    report_lines.append("## 3. Protected-Group Construction")
    report_lines.append("")
    for note in protected_notes:
        report_lines.append(f"- {note}")
    report_lines.append("")

    report_lines.append("## 4. Overall Model Context")
    report_lines.append("")
    report_lines.append(
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| AUC | {safe_number(overall_metrics.get('auc'))} |\n"
        f"| Accuracy | {safe_number(overall_metrics.get('accuracy'))} |\n"
        f"| Actual approval rate | {safe_percentage(overall_metrics.get('actual_approval_rate'))} |\n"
        f"| Predicted approval rate | {safe_percentage(overall_metrics.get('overall_approval_rate'))} |\n"
        f"| False positive rate | {safe_percentage(overall_metrics.get('false_positive_rate'))} |\n"
        f"| False negative rate | {safe_percentage(overall_metrics.get('false_negative_rate'))} |\n"
        f"| True positive rate | {safe_percentage(overall_metrics.get('true_positive_rate'))} |\n"
        f"| Brier score | {safe_number(overall_metrics.get('brier_score'))} |"
    )
    report_lines.append("")

    report_lines.append("## 5. Attribute-Level Fairness Summary")
    report_lines.append("")
    report_lines.append(markdown_table(summary_display))
    report_lines.append("")

    report_lines.append("## 6. Governance Attention Items")
    report_lines.append("")
    if flagged_display.empty:
        report_lines.append(
            "No groups triggered the configured governance attention flags. "
            "This does not prove the model is fair; it only means that no tested "
            "group exceeded the screening thresholds used in this validation script."
        )
    else:
        report_lines.append(
            "The following groups triggered at least one governance attention flag. "
            "Flags are based on disparate impact ratio below 0.80, absolute error-rate "
            "gaps of at least 0.10, equal-opportunity gaps of at least 0.10, or "
            "group calibration error of at least 0.10."
        )
        report_lines.append("")
        report_lines.append(markdown_table(flagged_display, max_rows=50))
    report_lines.append("")

    report_lines.append("## 7. Interpretation of Metrics")
    report_lines.append("")
    report_lines.append(
        "- **Approval-rate difference** compares each group approval rate with the "
        "reference group approval rate."
    )
    report_lines.append(
        "- **Disparate impact ratio** is the group approval rate divided by the "
        "reference group approval rate. Values below 0.80 are commonly used as "
        "a screening signal for adverse impact."
    )
    report_lines.append(
        "- **False-negative-rate gap** identifies groups more likely to be incorrectly "
        "classified as not approved when the true target is approved."
    )
    report_lines.append(
        "- **False-positive-rate gap** identifies groups more likely to be incorrectly "
        "classified as approved when the true target is not approved."
    )
    report_lines.append(
        "- **Equal opportunity difference** compares true-positive rates by group."
    )
    report_lines.append(
        "- **Calibration error** compares mean predicted probability with observed "
        "approval rate within each group."
    )
    report_lines.append("")

    report_lines.append("## 8. Validation Conclusion")
    report_lines.append("")
    if flagged_display.empty:
        report_lines.append(
            "The fairness screen did not identify a material disparity under the "
            "configured thresholds. The model should still be subject to periodic "
            "monitoring, documentation review, and challenger-model comparison."
        )
    else:
        report_lines.append(
            "The fairness screen identified one or more group-level disparities that "
            "require governance review before the model can be considered suitable "
            "for high-stakes deployment. Recommended actions include feature review, "
            "threshold sensitivity analysis, reject-inference assessment, subgroup "
            "stability testing, and challenger-model comparison."
        )
    report_lines.append("")

    report_lines.append("## 9. Limitations")
    report_lines.append("")
    report_lines.append(
        "- This script performs statistical fairness screening; it does not establish "
        "legal compliance."
    )
    report_lines.append(
        "- Protected-group variables may include missing, unavailable, or self-reported "
        "categories that require careful interpretation."
    )
    report_lines.append(
        "- Small groups may produce unstable error-rate estimates."
    )
    report_lines.append(
        "- Fairness results are sensitive to target construction, classification "
        "threshold, sample design, and model-feature choices."
    )
    report_lines.append("")

    report_lines.append("## 10. Output Files")
    report_lines.append("")
    report_lines.append(f"- Fairness metrics table: `{TABLE_PATH.as_posix()}`")
    report_lines.append(f"- Fairness comparison figure: `{FIGURE_PATH.as_posix()}`")
    report_lines.append(f"- Fairness validation report: `{REPORT_PATH.as_posix()}`")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_output_directories()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean modeling dataset not found:\n{DATA_PATH}\n\n"
            "Run the clean HMDA dataset step before running this script."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    if df.empty:
        raise ValueError(f"The dataset is empty: {DATA_PATH}")

    target_col, y_raw = get_target(df)

    valid_target_mask = y_raw.notna()
    df = df.loc[valid_target_mask].reset_index(drop=True)
    y_true = y_raw.loc[valid_target_mask].astype(int).reset_index(drop=True)

    df, protected_notes = add_protected_group_columns(df)

    missing_attributes = [
        attribute for attribute in PROTECTED_ATTRIBUTES if attribute not in df.columns
    ]

    if len(missing_attributes) == len(PROTECTED_ATTRIBUTES):
        raise ValueError(
            "None of the requested protected attributes could be created. "
            "Check whether the clean HMDA dataset retains race, ethnicity, sex, "
            "income, and tract-minority fields."
        )

    y_probability, y_pred, prediction_source, threshold = get_predictions(
        df=df,
        target_col=target_col,
    )

    if len(y_pred) != len(df):
        raise ValueError(
            "Prediction length does not match the number of evaluated records."
        )

    fairness_df = compute_all_fairness_metrics(
        df=df,
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    fairness_df.to_csv(TABLE_PATH, index=False)

    create_fairness_figure(fairness_df)

    overall_metrics = compute_overall_model_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    write_report(
        fairness_df=fairness_df,
        overall_metrics=overall_metrics,
        protected_notes=protected_notes,
        target_col=target_col,
        prediction_source=prediction_source,
        threshold=threshold,
        n_records=len(df),
    )

    print("\nStep I — Fairness and Bias Testing completed successfully.")
    print(f"Fairness metrics table: {TABLE_PATH}")
    print(f"Fairness validation report: {REPORT_PATH}")
    print(f"Fairness comparison figure: {FIGURE_PATH}")

    available_attributes = [
        attribute for attribute in PROTECTED_ATTRIBUTES if attribute in df.columns
    ]

    print("\nProtected attributes tested:")
    for attribute in available_attributes:
        print(f"  - {attribute}")

    attention_count = int(fairness_df["governance_attention_flag"].sum())
    print(f"\nGovernance attention flags: {attention_count}")


if __name__ == "__main__":
    main()