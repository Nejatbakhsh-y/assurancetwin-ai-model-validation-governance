"""
Step L — Stress Testing
Project: AssuranceTwin AI - Model Validation Governance

Creates:
    reports/tables/stress_test_results.csv
    reports/figures/stress_test_model_sensitivity.png
    reports/validation/stress_testing_report.md

Stress scenarios:
    1. Income shock
    2. Loan amount increase
    3. LTV increase
    4. Missing-data shock
    5. Minority-tract distribution shift
    6. Out-of-time validation
    7. Recession-like synthetic scenario

Design principle:
    Stress testing evaluates whether model behavior remains acceptable under plausible
    adverse portfolio and data-quality conditions. This is not ordinary model scoring.
    It is a governance-oriented robustness check.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "hmda_modeling_dataset.csv"

MODELS_DIR = ROOT / "models"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.pkl"
CHALLENGER_MODEL_PATH = MODELS_DIR / "challenger_model.pkl"

TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"
VALIDATION_DIR = ROOT / "reports" / "validation"

RESULTS_PATH = TABLES_DIR / "stress_test_results.csv"
FIGURE_PATH = FIGURES_DIR / "stress_test_model_sensitivity.png"
REPORT_PATH = VALIDATION_DIR / "stress_testing_report.md"


RANDOM_STATE = 42
TEST_SIZE = 0.30
PROBABILITY_THRESHOLD = 0.50


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_CANDIDATES = [
    "approved",
    "target",
    "y",
    "loan_approved",
    "application_approved",
]

TIME_CANDIDATES = [
    "activity_year",
    "year",
    "application_year",
    "loan_year",
    "origination_year",
    "submission_year",
]

INCOME_CANDIDATES = [
    "income",
    "applicant_income",
    "annual_income",
    "income_000s",
]

LOAN_AMOUNT_CANDIDATES = [
    "loan_amount",
    "loan_amount_000s",
    "loan_amt",
    "amount",
]

LTV_CANDIDATES = [
    "combined_loan_to_value_ratio",
    "loan_to_value_ratio",
    "ltv",
    "cltv",
]

PROPERTY_VALUE_CANDIDATES = [
    "property_value",
    "property_value_000s",
    "collateral_value",
]

DTI_CANDIDATES = [
    "debt_to_income_ratio",
    "dti",
    "debt_income_ratio",
]

MINORITY_TRACT_CANDIDATES = [
    "tract_minority_population_percent",
    "minority_population_percent",
    "tract_minority_pct",
    "minority_tract_share",
    "minority_tract_percent",
    "tract_minority_population",
]

RACE_CANDIDATES = [
    "race",
    "derived_race",
    "applicant_race",
    "applicant_race_1",
]

ETHNICITY_CANDIDATES = [
    "ethnicity",
    "derived_ethnicity",
    "applicant_ethnicity",
    "applicant_ethnicity_1",
]

SEX_CANDIDATES = [
    "sex",
    "derived_sex",
    "applicant_sex",
    "applicant_sex_1",
]

POST_OUTCOME_OR_LEAKAGE_PATTERNS = [
    "action_taken",
    "denial_reason",
    "purchaser_type",
    "rate_spread",
    "hoepa_status",
    "lien_status_post",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

@dataclass
class ModelBundle:
    name: str
    estimator: Any
    feature_columns: Optional[List[str]]
    source: str


def ensure_directories() -> None:
    for directory in [TABLES_DIR, FIGURES_DIR, VALIDATION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_text(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_text(c) for c in df.columns]
    return df


def find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    candidate_set = {normalize_text(c) for c in candidates}
    for col in df.columns:
        if normalize_text(col) in candidate_set:
            return col

    # Flexible fallback for common naming variants.
    for col in df.columns:
        col_norm = normalize_text(col)
        for candidate in candidate_set:
            if candidate in col_norm:
                return col

    return None


def find_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    found = []
    candidate_set = {normalize_text(c) for c in candidates}

    for col in df.columns:
        col_norm = normalize_text(col)
        if col_norm in candidate_set:
            found.append(col)

    for col in df.columns:
        col_norm = normalize_text(col)
        if col not in found and any(candidate in col_norm for candidate in candidate_set):
            found.append(col)

    return found


def coerce_binary_target(series: pd.Series) -> pd.Series:
    y = series.copy()

    if y.dtype == "bool":
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        unique_values = sorted(pd.Series(y.dropna().unique()).tolist())
        if set(unique_values).issubset({0, 1}):
            return y.astype(int)

        # Conservative fallback: positive values become 1.
        return (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)

    y_str = y.astype(str).str.strip().str.lower()

    positive_values = {
        "1",
        "yes",
        "y",
        "true",
        "approved",
        "approve",
        "originated",
        "loan originated",
        "application approved",
    }

    negative_values = {
        "0",
        "no",
        "n",
        "false",
        "denied",
        "deny",
        "not approved",
        "application denied",
    }

    mapped = y_str.map(
        lambda v: 1 if v in positive_values else 0 if v in negative_values else np.nan
    )

    if mapped.isna().mean() < 0.20:
        return mapped.fillna(0).astype(int)

    # Last-resort deterministic encoding.
    codes, uniques = pd.factorize(y_str)
    return pd.Series(codes, index=series.index).astype(int).clip(lower=0, upper=1)


def safe_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("Exempt", "", regex=False)
        .str.replace("exempt", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def convert_obvious_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    likely_numeric_tokens = [
        "income",
        "amount",
        "value",
        "ratio",
        "ltv",
        "dti",
        "tract",
        "population",
        "percent",
        "rate",
        "age",
        "term",
    ]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_norm = normalize_text(col)
        if any(token in col_norm for token in likely_numeric_tokens):
            converted = safe_numeric(df[col])
            if converted.notna().sum() >= max(10, int(0.20 * len(df))):
                df[col] = converted

    return df


def drop_obvious_leakage_columns(X: pd.DataFrame, target_col: str) -> pd.DataFrame:
    X = X.copy()
    drop_cols = []

    for col in X.columns:
        col_norm = normalize_text(col)
        if col_norm == normalize_text(target_col):
            drop_cols.append(col)
        elif any(pattern in col_norm for pattern in POST_OUTCOME_OR_LEAKAGE_PATTERNS):
            drop_cols.append(col)

    if drop_cols:
        X = X.drop(columns=sorted(set(drop_cols)), errors="ignore")

    return X


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find modeling dataset at: {DATA_PATH}\n"
            "Run the clean dataset step first: scripts/03_create_clean_hmda_dataset.py"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = normalize_columns(df)
    df = convert_obvious_numeric_columns(df)

    target_col = find_first_column(df, TARGET_CANDIDATES)

    if target_col is None:
        raise ValueError(
            "Could not identify the target column. Expected one of: "
            + ", ".join(TARGET_CANDIDATES)
        )

    y = coerce_binary_target(df[target_col])
    X = df.drop(columns=[target_col], errors="ignore")
    X = drop_obvious_leakage_columns(X, target_col=target_col)

    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if y.nunique() < 2:
        raise ValueError(
            f"Target column '{target_col}' has fewer than two classes after cleaning."
        )

    return X, y, target_col


def extract_estimator_and_features(raw_artifact: Any) -> Tuple[Any, Optional[List[str]], Optional[str]]:
    """
    Supports several common artifact formats:
        - estimator / pipeline directly
        - dict with keys such as model, estimator, pipeline
        - dict with feature list
    """
    if isinstance(raw_artifact, dict):
        estimator = None
        for key in ["model", "estimator", "pipeline", "best_model", "fitted_model"]:
            if key in raw_artifact:
                estimator = raw_artifact[key]
                break

        if estimator is None:
            estimator = raw_artifact

        feature_columns = None
        for key in [
            "feature_columns",
            "features",
            "feature_names",
            "input_features",
            "selected_features",
        ]:
            if key in raw_artifact and raw_artifact[key] is not None:
                feature_columns = list(raw_artifact[key])
                feature_columns = [normalize_text(c) for c in feature_columns]
                break

        model_name = raw_artifact.get("model_name") or raw_artifact.get("name")
        return estimator, feature_columns, model_name

    estimator = raw_artifact
    feature_columns = None

    if hasattr(estimator, "feature_names_in_"):
        feature_columns = list(getattr(estimator, "feature_names_in_"))
        feature_columns = [normalize_text(c) for c in feature_columns]

    return estimator, feature_columns, None


def load_model_bundle(path: Path, fallback_name: str) -> Optional[ModelBundle]:
    if not path.exists():
        return None

    raw_artifact = joblib.load(path)
    estimator, feature_columns, artifact_name = extract_estimator_and_features(raw_artifact)

    if feature_columns is None and hasattr(estimator, "feature_names_in_"):
        feature_columns = list(getattr(estimator, "feature_names_in_"))
        feature_columns = [normalize_text(c) for c in feature_columns]

    name = normalize_text(artifact_name or fallback_name)

    return ModelBundle(
        name=name,
        estimator=estimator,
        feature_columns=feature_columns,
        source=str(path.relative_to(ROOT)),
    )


def fit_fallback_models(X_train: pd.DataFrame, y_train: pd.Series) -> List[ModelBundle]:
    """
    If saved models are not found, fit two simple validation-only fallback models.
    This keeps the stress-testing script executable, but the report will state that
    fallback models were used.
    """
    categorical_cols = [
        c for c in X_train.columns
        if X_train[c].dtype == "object" or str(X_train[c].dtype) == "category"
    ]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(random_state=RANDOM_STATE),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    gradient_boosting.fit(X_train, y_train)

    features = list(X_train.columns)

    return [
        ModelBundle(
            name="fallback_logistic_regression",
            estimator=logistic,
            feature_columns=features,
            source="trained inside scripts/10_stress_testing.py because saved model was unavailable",
        ),
        ModelBundle(
            name="fallback_gradient_boosting",
            estimator=gradient_boosting,
            feature_columns=features,
            source="trained inside scripts/10_stress_testing.py because saved model was unavailable",
        ),
    ]


def prepare_model_input(X: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    X_model = X.copy()
    X_model.columns = [normalize_text(c) for c in X_model.columns]

    if bundle.feature_columns is None:
        return X_model

    expected = [normalize_text(c) for c in bundle.feature_columns]

    for col in expected:
        if col not in X_model.columns:
            X_model[col] = np.nan

    return X_model[expected]


def get_positive_class_index(estimator: Any) -> int:
    classes = getattr(estimator, "classes_", None)

    if classes is not None:
        classes_list = list(classes)
        if 1 in classes_list:
            return classes_list.index(1)
        if "1" in classes_list:
            return classes_list.index("1")
        if True in classes_list:
            return classes_list.index(True)
        return len(classes_list) - 1

    return 1


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def predict_probability(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    X_model = prepare_model_input(X, bundle)

    estimator = bundle.estimator

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_model)
        if proba.ndim == 1:
            return np.asarray(proba).astype(float)

        positive_idx = get_positive_class_index(estimator)
        positive_idx = min(positive_idx, proba.shape[1] - 1)
        return np.asarray(proba[:, positive_idx]).astype(float)

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_model)
        return sigmoid(np.asarray(scores).astype(float))

    if hasattr(estimator, "predict"):
        predictions = estimator.predict(X_model)
        return np.asarray(predictions).astype(float)

    raise TypeError(f"Model '{bundle.name}' does not support prediction.")


def safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def expected_calibration_error(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true_array = np.asarray(y_true).astype(float)
    y_prob_array = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_array, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true_array)

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue

        bin_weight = np.mean(mask)
        observed_rate = np.mean(y_true_array[mask])
        predicted_rate = np.mean(y_prob_array[mask])
        ece += bin_weight * abs(observed_rate - predicted_rate)

    return float(ece)


def compute_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = PROBABILITY_THRESHOLD,
) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    approval_rate = float(np.mean(y_pred))
    actual_approval_rate = float(np.mean(y_true))
    avg_probability = float(np.mean(y_prob))

    metrics = {
        "n_records": int(len(y_true)),
        "actual_approval_rate": actual_approval_rate,
        "predicted_approval_rate": approval_rate,
        "predicted_denial_rate": float(1.0 - approval_rate),
        "average_predicted_probability": avg_probability,
        "auc": safe_auc(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": expected_calibration_error(y_true, y_prob),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else np.nan,
    }

    return metrics


# ---------------------------------------------------------------------
# Stress scenario transformations
# ---------------------------------------------------------------------

def scenario_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    return X.copy(), y.copy(), "Original validation sample."


def scenario_income_shock(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    Xs = X.copy()
    income_col = find_first_column(Xs, INCOME_CANDIDATES)

    if income_col is None:
        return Xs, y.copy(), "Income column not found; scenario returned unchanged data."

    Xs[income_col] = safe_numeric(Xs[income_col]) * 0.80
    Xs[income_col] = Xs[income_col].clip(lower=0)

    return Xs, y.copy(), f"Reduced '{income_col}' by 20 percent."


def scenario_loan_amount_increase(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    Xs = X.copy()
    loan_col = find_first_column(Xs, LOAN_AMOUNT_CANDIDATES)

    if loan_col is None:
        return Xs, y.copy(), "Loan amount column not found; scenario returned unchanged data."

    Xs[loan_col] = safe_numeric(Xs[loan_col]) * 1.15
    Xs[loan_col] = Xs[loan_col].clip(lower=0)

    return Xs, y.copy(), f"Increased '{loan_col}' by 15 percent."


def scenario_ltv_increase(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    Xs = X.copy()
    ltv_col = find_first_column(Xs, LTV_CANDIDATES)

    if ltv_col is None:
        return Xs, y.copy(), "LTV column not found; scenario returned unchanged data."

    ltv = safe_numeric(Xs[ltv_col])
    Xs[ltv_col] = (ltv + 10.0).clip(lower=0, upper=250)

    return Xs, y.copy(), f"Increased '{ltv_col}' by 10 percentage points."


def scenario_missing_data_shock(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    Xs = X.copy()

    candidate_cols = []
    for candidate_group in [
        INCOME_CANDIDATES,
        LOAN_AMOUNT_CANDIDATES,
        LTV_CANDIDATES,
        PROPERTY_VALUE_CANDIDATES,
        DTI_CANDIDATES,
        MINORITY_TRACT_CANDIDATES,
        RACE_CANDIDATES,
        ETHNICITY_CANDIDATES,
        SEX_CANDIDATES,
    ]:
        candidate_cols.extend(find_columns(Xs, candidate_group))

    candidate_cols = sorted(set(candidate_cols))

    if not candidate_cols:
        # Fallback: shock up to 10 non-target features.
        candidate_cols = list(Xs.columns[: min(10, Xs.shape[1])])

    missing_fraction = 0.15
    row_count = len(Xs)

    for col in candidate_cols:
        mask = rng.random(row_count) < missing_fraction
        Xs.loc[mask, col] = np.nan

    return (
        Xs,
        y.copy(),
        f"Set approximately {int(missing_fraction * 100)} percent of values to missing "
        f"for {len(candidate_cols)} high-impact columns.",
    )


def scenario_minority_tract_distribution_shift(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    Xs = X.copy()
    ys = y.copy()

    minority_col = find_first_column(Xs, MINORITY_TRACT_CANDIDATES)

    if minority_col is None:
        return Xs, ys, "Minority-tract column not found; scenario returned unchanged data."

    minority_values = safe_numeric(Xs[minority_col])

    if minority_values.notna().sum() < 20:
        return Xs, ys, f"Column '{minority_col}' had insufficient numeric data; scenario returned unchanged data."

    threshold = minority_values.quantile(0.75)
    high_minority_idx = Xs.index[minority_values >= threshold].to_numpy()
    other_idx = Xs.index[minority_values < threshold].to_numpy()

    if len(high_minority_idx) < 10 or len(other_idx) < 10:
        return Xs, ys, f"Column '{minority_col}' did not support stable resampling; scenario returned unchanged data."

    n = len(Xs)
    n_high = int(0.60 * n)
    n_other = n - n_high

    sampled_high = rng.choice(high_minority_idx, size=n_high, replace=True)
    sampled_other = rng.choice(other_idx, size=n_other, replace=True)

    sampled_idx = np.concatenate([sampled_high, sampled_other])
    rng.shuffle(sampled_idx)

    X_shifted = Xs.loc[sampled_idx].reset_index(drop=True)
    y_shifted = ys.loc[sampled_idx].reset_index(drop=True)

    return (
        X_shifted,
        y_shifted,
        f"Resampled validation portfolio so approximately 60 percent of records came from "
        f"the highest quartile of '{minority_col}'.",
    )


def scenario_out_of_time_validation(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    Xs = X.copy()
    ys = y.copy()

    time_col = find_first_column(Xs, TIME_CANDIDATES)

    if time_col is not None:
        time_values = safe_numeric(Xs[time_col])

        if time_values.notna().sum() >= 20 and time_values.nunique(dropna=True) > 1:
            latest_time = time_values.max()
            mask = time_values == latest_time

            if mask.sum() >= 20 and mask.sum() < len(Xs):
                return (
                    Xs.loc[mask].reset_index(drop=True),
                    ys.loc[mask].reset_index(drop=True),
                    f"Used records from latest available period in '{time_col}': {latest_time}.",
                )

    # HMDA state-year extracts often contain only one activity year.
    # In that case, use a deterministic last-window proxy.
    n = len(Xs)
    start = int(0.80 * n)

    return (
        Xs.iloc[start:].reset_index(drop=True),
        ys.iloc[start:].reset_index(drop=True),
        "No usable multi-period time column found; used final 20 percent of ordered records as a proxy out-of-time window.",
    )


def scenario_recession_like_synthetic(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    Xs = X.copy()

    income_col = find_first_column(Xs, INCOME_CANDIDATES)
    loan_col = find_first_column(Xs, LOAN_AMOUNT_CANDIDATES)
    ltv_col = find_first_column(Xs, LTV_CANDIDATES)
    property_value_col = find_first_column(Xs, PROPERTY_VALUE_CANDIDATES)
    dti_col = find_first_column(Xs, DTI_CANDIDATES)

    applied = []

    if income_col is not None:
        Xs[income_col] = (safe_numeric(Xs[income_col]) * 0.85).clip(lower=0)
        applied.append(f"income -15 percent using '{income_col}'")

    if loan_col is not None:
        Xs[loan_col] = (safe_numeric(Xs[loan_col]) * 1.10).clip(lower=0)
        applied.append(f"loan amount +10 percent using '{loan_col}'")

    if ltv_col is not None:
        Xs[ltv_col] = (safe_numeric(Xs[ltv_col]) + 15.0).clip(lower=0, upper=250)
        applied.append(f"LTV +15 percentage points using '{ltv_col}'")

    if property_value_col is not None:
        Xs[property_value_col] = (safe_numeric(Xs[property_value_col]) * 0.90).clip(lower=0)
        applied.append(f"property value -10 percent using '{property_value_col}'")

    if dti_col is not None:
        Xs[dti_col] = (safe_numeric(Xs[dti_col]) * 1.10).clip(lower=0)
        applied.append(f"DTI +10 percent using '{dti_col}'")

    # Add moderate data-quality deterioration.
    critical_cols = []
    for col in [income_col, loan_col, ltv_col, property_value_col, dti_col]:
        if col is not None:
            critical_cols.append(col)

    if critical_cols:
        for col in critical_cols:
            mask = rng.random(len(Xs)) < 0.10
            Xs.loc[mask, col] = np.nan
        applied.append("10 percent missingness introduced into stressed financial fields")

    if not applied:
        return Xs, y.copy(), "No recession-relevant columns found; scenario returned unchanged data."

    return Xs, y.copy(), "; ".join(applied) + "."


SCENARIOS = [
    ("baseline", scenario_baseline),
    ("income_shock", scenario_income_shock),
    ("loan_amount_increase", scenario_loan_amount_increase),
    ("ltv_increase", scenario_ltv_increase),
    ("missing_data_shock", scenario_missing_data_shock),
    ("minority_tract_distribution_shift", scenario_minority_tract_distribution_shift),
    ("out_of_time_validation", scenario_out_of_time_validation),
    ("recession_like_synthetic", scenario_recession_like_synthetic),
]


# ---------------------------------------------------------------------
# Results and reporting
# ---------------------------------------------------------------------

def evaluate_models_under_scenarios(
    bundles: List[ModelBundle],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    baseline_by_model: Dict[str, Dict[str, float]] = {}

    for scenario_name, scenario_function in SCENARIOS:
        X_scenario, y_scenario, scenario_note = scenario_function(X_valid, y_valid, rng)

        for bundle in bundles:
            try:
                y_prob = predict_probability(bundle, X_scenario)
                metrics = compute_metrics(y_scenario, y_prob)
                error_message = ""
            except Exception as exc:
                metrics = {
                    "n_records": len(y_scenario),
                    "actual_approval_rate": np.nan,
                    "predicted_approval_rate": np.nan,
                    "predicted_denial_rate": np.nan,
                    "average_predicted_probability": np.nan,
                    "auc": np.nan,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "balanced_accuracy": np.nan,
                    "brier_score": np.nan,
                    "expected_calibration_error": np.nan,
                    "true_negative": np.nan,
                    "false_positive": np.nan,
                    "false_negative": np.nan,
                    "true_positive": np.nan,
                    "false_positive_rate": np.nan,
                    "false_negative_rate": np.nan,
                }
                error_message = str(exc)

            row = {
                "scenario": scenario_name,
                "model_name": bundle.name,
                "model_source": bundle.source,
                "scenario_note": scenario_note,
                "error_message": error_message,
            }
            row.update(metrics)

            if scenario_name == "baseline":
                baseline_by_model[bundle.name] = metrics.copy()
                row["approval_rate_delta_pp"] = 0.0
                row["avg_probability_delta_pp"] = 0.0
                row["auc_delta"] = 0.0
                row["brier_score_delta"] = 0.0
                row["fpr_delta_pp"] = 0.0
                row["fnr_delta_pp"] = 0.0
            else:
                baseline = baseline_by_model.get(bundle.name, {})
                row["approval_rate_delta_pp"] = percentage_point_delta(
                    metrics.get("predicted_approval_rate"),
                    baseline.get("predicted_approval_rate"),
                )
                row["avg_probability_delta_pp"] = percentage_point_delta(
                    metrics.get("average_predicted_probability"),
                    baseline.get("average_predicted_probability"),
                )
                row["auc_delta"] = safe_delta(
                    metrics.get("auc"),
                    baseline.get("auc"),
                )
                row["brier_score_delta"] = safe_delta(
                    metrics.get("brier_score"),
                    baseline.get("brier_score"),
                )
                row["fpr_delta_pp"] = percentage_point_delta(
                    metrics.get("false_positive_rate"),
                    baseline.get("false_positive_rate"),
                )
                row["fnr_delta_pp"] = percentage_point_delta(
                    metrics.get("false_negative_rate"),
                    baseline.get("false_negative_rate"),
                )

            rows.append(row)

    results = pd.DataFrame(rows)
    return results


def safe_delta(value: Any, baseline: Any) -> float:
    try:
        if value is None or baseline is None:
            return np.nan
        if pd.isna(value) or pd.isna(baseline):
            return np.nan
        return float(value) - float(baseline)
    except Exception:
        return np.nan


def percentage_point_delta(value: Any, baseline: Any) -> float:
    delta = safe_delta(value, baseline)
    if pd.isna(delta):
        return np.nan
    return float(delta * 100.0)


def save_results_table(results: pd.DataFrame) -> None:
    preferred_order = [
        "scenario",
        "model_name",
        "model_source",
        "n_records",
        "actual_approval_rate",
        "predicted_approval_rate",
        "predicted_denial_rate",
        "average_predicted_probability",
        "approval_rate_delta_pp",
        "avg_probability_delta_pp",
        "auc",
        "auc_delta",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "brier_score",
        "brier_score_delta",
        "expected_calibration_error",
        "false_positive_rate",
        "fpr_delta_pp",
        "false_negative_rate",
        "fnr_delta_pp",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "scenario_note",
        "error_message",
    ]

    ordered_cols = [c for c in preferred_order if c in results.columns]
    remaining_cols = [c for c in results.columns if c not in ordered_cols]

    results = results[ordered_cols + remaining_cols]
    results.to_csv(RESULTS_PATH, index=False)


def create_sensitivity_plot(results: pd.DataFrame) -> None:
    plot_df = results.copy()
    plot_df = plot_df[plot_df["scenario"] != "baseline"]

    if plot_df.empty:
        return

    pivot = plot_df.pivot_table(
        index="scenario",
        columns="model_name",
        values="approval_rate_delta_pp",
        aggfunc="mean",
    )

    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(12, 7))
    ax.axhline(0, linewidth=1)
    ax.set_title("Stress Test Model Sensitivity")
    ax.set_xlabel("Stress Scenario")
    ax.set_ylabel("Predicted Approval Rate Change vs Baseline, percentage points")
    ax.legend(title="Model", loc="best")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()


def format_pct(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "NA"


def format_pp(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f} pp"
    except Exception:
        return "NA"


def format_float(value: Any, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def identify_most_sensitive_scenarios(results: pd.DataFrame) -> pd.DataFrame:
    stressed = results[results["scenario"] != "baseline"].copy()

    if stressed.empty:
        return stressed

    stressed["abs_approval_rate_delta_pp"] = stressed["approval_rate_delta_pp"].abs()
    stressed = stressed.sort_values(
        ["model_name", "abs_approval_rate_delta_pp"],
        ascending=[True, False],
    )

    return stressed.groupby("model_name").head(3)


def create_markdown_report(
    results: pd.DataFrame,
    target_col: str,
    used_fallback_models: bool,
    X_valid: pd.DataFrame,
) -> None:
    baseline = results[results["scenario"] == "baseline"].copy()
    stressed = results[results["scenario"] != "baseline"].copy()
    sensitive = identify_most_sensitive_scenarios(results)

    model_sources = (
        results[["model_name", "model_source"]]
        .drop_duplicates()
        .sort_values("model_name")
    )

    scenario_notes = (
        results[["scenario", "scenario_note"]]
        .drop_duplicates()
        .sort_values("scenario")
    )

    lines = []

    lines.append("# Stress Testing Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report evaluates whether model outputs remain stable under adverse "
        "financial-risk, data-quality, portfolio-mix, and time-shift conditions. "
        "The analysis is designed as an independent model-validation stress test, "
        "not merely as a predictive-performance exercise."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Modeling dataset: `{DATA_PATH.relative_to(ROOT)}`")
    lines.append(f"- Target variable: `{target_col}`")
    lines.append(f"- Validation sample size: `{len(X_valid):,}`")
    lines.append(f"- Probability threshold: `{PROBABILITY_THRESHOLD:.2f}`")
    lines.append(f"- Results table: `{RESULTS_PATH.relative_to(ROOT)}`")
    lines.append(f"- Sensitivity figure: `{FIGURE_PATH.relative_to(ROOT)}`")
    lines.append("")

    lines.append("## Models evaluated")
    lines.append("")
    lines.append("| Model | Source |")
    lines.append("|---|---|")
    for _, row in model_sources.iterrows():
        lines.append(f"| {row['model_name']} | `{row['model_source']}` |")
    lines.append("")

    if used_fallback_models:
        lines.append("> Note: Saved champion/challenger model files were not found. ")
        lines.append(
            "> This script trained fallback validation-only models so that the stress-testing "
            "workflow could run. For final governance use, rerun Step G and confirm that "
            "`models/champion_model.pkl` and `models/challenger_model.pkl` exist."
        )
        lines.append("")

    lines.append("## Stress scenarios")
    lines.append("")
    lines.append("| Scenario | Implementation note |")
    lines.append("|---|---|")
    for _, row in scenario_notes.iterrows():
        lines.append(f"| {row['scenario']} | {row['scenario_note']} |")
    lines.append("")

    lines.append("## Baseline model behavior")
    lines.append("")
    lines.append("| Model | AUC | Brier score | Approval rate | FPR | FNR |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in baseline.sort_values("model_name").iterrows():
        lines.append(
            "| "
            f"{row['model_name']} | "
            f"{format_float(row['auc'])} | "
            f"{format_float(row['brier_score'])} | "
            f"{format_pct(row['predicted_approval_rate'])} | "
            f"{format_pct(row['false_positive_rate'])} | "
            f"{format_pct(row['false_negative_rate'])} |"
        )
    lines.append("")

    lines.append("## Most sensitive stress results")
    lines.append("")
    lines.append("| Model | Scenario | Approval-rate delta | Probability delta | AUC delta | Brier delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in sensitive.iterrows():
        lines.append(
            "| "
            f"{row['model_name']} | "
            f"{row['scenario']} | "
            f"{format_pp(row['approval_rate_delta_pp'])} | "
            f"{format_pp(row['avg_probability_delta_pp'])} | "
            f"{format_float(row['auc_delta'])} | "
            f"{format_float(row['brier_score_delta'])} |"
        )
    lines.append("")

    lines.append("## Governance interpretation")
    lines.append("")
    lines.append(
        "A governed model should not be selected only because it has the highest baseline "
        "AUC. A model that shows large adverse changes in approval rate, calibration, "
        "false-negative rate, or false-positive rate under stress may create operational, "
        "consumer-compliance, or reputational risk. The appropriate governance question is "
        "whether model behavior remains explainable, directionally plausible, and stable "
        "under realistic adverse conditions."
    )
    lines.append("")
    lines.append("Key validation considerations:")
    lines.append("")
    lines.append(
        "- **Income shock:** evaluates whether the model becomes excessively restrictive "
        "when borrower income deteriorates."
    )
    lines.append(
        "- **Loan amount and LTV shocks:** evaluate sensitivity to higher credit exposure "
        "and weaker collateral coverage."
    )
    lines.append(
        "- **Missing-data shock:** evaluates operational resilience when upstream data "
        "quality deteriorates."
    )
    lines.append(
        "- **Minority-tract distribution shift:** evaluates whether portfolio composition "
        "changes materially alter predicted approvals."
    )
    lines.append(
        "- **Out-of-time validation:** evaluates temporal robustness. If no true multi-period "
        "field exists, the script uses a deterministic final-window proxy and explicitly "
        "flags this limitation."
    )
    lines.append(
        "- **Recession-like synthetic scenario:** evaluates combined macroeconomic stress, "
        "including lower income, weaker collateral, higher loan burden, and missingness."
    )
    lines.append("")

    if not stressed.empty:
        worst = stressed.copy()
        worst["abs_approval_rate_delta_pp"] = worst["approval_rate_delta_pp"].abs()
        worst = worst.sort_values("abs_approval_rate_delta_pp", ascending=False).head(1)

        if not worst.empty:
            row = worst.iloc[0]
            lines.append("## Primary sensitivity finding")
            lines.append("")
            lines.append(
                f"The largest approval-rate movement was observed for model "
                f"`{row['model_name']}` under scenario `{row['scenario']}`, with a change "
                f"of {format_pp(row['approval_rate_delta_pp'])} relative to baseline."
            )
            lines.append("")

    lines.append("## Model-risk conclusion")
    lines.append("")
    lines.append(
        "The stress-test results should be reviewed jointly with independent validation "
        "metrics, fairness testing, calibration analysis, and explanation stability. "
        "A model with strong predictive performance but weak stress robustness should not "
        "automatically be treated as the preferred governed model."
    )
    lines.append("")

    lines.append("## Files produced")
    lines.append("")
    lines.append(f"- `{RESULTS_PATH.relative_to(ROOT)}`")
    lines.append(f"- `{FIGURE_PATH.relative_to(ROOT)}`")
    lines.append(f"- `{REPORT_PATH.relative_to(ROOT)}`")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_run_metadata(
    target_col: str,
    bundles: List[ModelBundle],
    used_fallback_models: bool,
) -> None:
    metadata_path = VALIDATION_DIR / "stress_testing_metadata.json"

    metadata = {
        "script": "scripts/10_stress_testing.py",
        "target_column": target_col,
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "results_path": str(RESULTS_PATH.relative_to(ROOT)),
        "figure_path": str(FIGURE_PATH.relative_to(ROOT)),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "probability_threshold": PROBABILITY_THRESHOLD,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "used_fallback_models": used_fallback_models,
        "models": [
            {
                "model_name": bundle.name,
                "model_source": bundle.source,
                "feature_column_count": (
                    len(bundle.feature_columns)
                    if bundle.feature_columns is not None
                    else None
                ),
            }
            for bundle in bundles
        ],
        "stress_scenarios": [name for name, _ in SCENARIOS],
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    ensure_directories()

    print("=" * 80)
    print("Step L — Stress Testing")
    print("=" * 80)

    print(f"Loading dataset: {DATA_PATH}")
    X, y, target_col = load_dataset()

    print(f"Dataset loaded: {len(X):,} rows, {X.shape[1]:,} candidate features")
    print(f"Target column: {target_col}")
    print(f"Target approval rate: {y.mean():.4f}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Validation sample: {len(X_valid):,} rows")

    bundles: List[ModelBundle] = []

    champion_bundle = load_model_bundle(CHAMPION_MODEL_PATH, fallback_name="champion_model")
    challenger_bundle = load_model_bundle(CHALLENGER_MODEL_PATH, fallback_name="challenger_model")

    if champion_bundle is not None:
        bundles.append(champion_bundle)
        print(f"Loaded champion model: {CHAMPION_MODEL_PATH}")

    if challenger_bundle is not None:
        bundles.append(challenger_bundle)
        print(f"Loaded challenger model: {CHALLENGER_MODEL_PATH}")

    used_fallback_models = False

    if not bundles:
        print("Saved champion/challenger models were not found.")
        print("Training fallback validation-only models for stress testing.")
        bundles = fit_fallback_models(X_train, y_train)
        used_fallback_models = True

    print("Evaluating stress scenarios...")
    results = evaluate_models_under_scenarios(bundles, X_valid, y_valid)

    print(f"Saving results table: {RESULTS_PATH}")
    save_results_table(results)

    print(f"Saving sensitivity figure: {FIGURE_PATH}")
    create_sensitivity_plot(results)

    print(f"Saving validation report: {REPORT_PATH}")
    create_markdown_report(
        results=results,
        target_col=target_col,
        used_fallback_models=used_fallback_models,
        X_valid=X_valid,
    )

    save_run_metadata(
        target_col=target_col,
        bundles=bundles,
        used_fallback_models=used_fallback_models,
    )

    print("")
    print("Stress testing complete.")
    print("")
    print("Created:")
    print(f"  - {RESULTS_PATH.relative_to(ROOT)}")
    print(f"  - {FIGURE_PATH.relative_to(ROOT)}")
    print(f"  - {REPORT_PATH.relative_to(ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()