"""
Step M — Monitoring and Drift Simulation

Creates lifecycle monitoring artifacts for the AssuranceTwin AI Model Validation
and Governance project.

Outputs:
    reports/tables/drift_monitoring_summary.csv
    reports/figures/drift_dashboard_plot.png
    reports/validation/monitoring_plan.md

Design goals:
    - Time-based train/monitoring split when a usable date/year/period column exists.
    - Deterministic pseudo-period monitoring simulation when the public file lacks
      a true timestamp.
    - Population Stability Index, Characteristic Stability Index, data drift,
      prediction drift, performance drift, fairness drift, and calibration drift.
    - Robust fallbacks if the previously saved champion model cannot be loaded.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "hmda_modeling_dataset.csv"
TABLE_DIR = ROOT_DIR / "reports" / "tables"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"
VALIDATION_DIR = ROOT_DIR / "reports" / "validation"
MODEL_DIR = ROOT_DIR / "models"

SUMMARY_PATH = TABLE_DIR / "drift_monitoring_summary.csv"
FIGURE_PATH = FIGURE_DIR / "drift_dashboard_plot.png"
PLAN_PATH = VALIDATION_DIR / "monitoring_plan.md"

RANDOM_STATE = 42
MAX_TRAIN_ROWS = 120_000
MAX_FEATURES_FOR_CSI = 25
PSI_BINS = 10


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class PeriodSplitInfo:
    period_column: str
    period_source: str
    train_periods: List[str]
    monitor_periods: List[str]
    synthetic_periods_used: bool


@dataclass
class ModelInfo:
    model: object
    feature_columns: List[str]
    source: str
    used_existing_artifact: bool


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def ensure_directories() -> None:
    for folder in [TABLE_DIR, FIGURE_DIR, VALIDATION_DIR, MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_column_name(c) for c in out.columns]
    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def classify_stability_index(value: float) -> str:
    if pd.isna(value):
        return "Not available"
    if value < 0.10:
        return "Low"
    if value < 0.25:
        return "Moderate"
    return "High"


def status_from_breaches(breaches: Sequence[str]) -> str:
    if any("High" in b or "Red" in b for b in breaches):
        return "Red"
    if len(breaches) > 0:
        return "Amber"
    return "Green"


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run the clean HMDA dataset step first."
        )
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if df.empty:
        raise ValueError(f"{DATA_PATH} is empty.")
    return normalize_columns(df)


def identify_or_create_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Find an existing binary approval target or derive one from action_taken."""
    candidate_targets = [
        "approved",
        "approval",
        "loan_approved",
        "target",
        "target_approved",
        "y",
    ]

    for col in candidate_targets:
        if col in df.columns:
            target = safe_numeric(df[col])
            unique_values = sorted(target.dropna().unique().tolist())
            if set(unique_values).issubset({0, 1}) and len(unique_values) >= 1:
                out = df.copy()
                out[col] = target.astype("Int64")
                out = out[out[col].notna()].copy()
                out[col] = out[col].astype(int)
                return out, col

    if "action_taken" in df.columns:
        out = df.copy()
        action = out["action_taken"].astype(str).str.extract(r"(\d+)", expand=False)
        # HMDA-style approval/origination-related outcomes.
        approved_codes = {"1", "2", "6", "8"}
        out["approved"] = action.isin(approved_codes).astype(int)
        out = out[action.notna()].copy()
        return out, "approved"

    raise ValueError(
        "No binary target column was found and action_taken is unavailable. "
        "Expected a target such as 'approved' or an HMDA 'action_taken' column."
    )


def add_governance_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Create monitoring-friendly bands when the raw numeric columns exist."""
    out = df.copy()

    income_candidates = ["income", "applicant_income", "ffiec_msa_md_median_family_income"]
    income_col = next((c for c in income_candidates if c in out.columns), None)
    if income_col and "income_band" not in out.columns:
        income = safe_numeric(out[income_col])
        try:
            out["income_band"] = pd.qcut(
                income.rank(method="first"),
                q=4,
                labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                duplicates="drop",
            ).astype(str)
            out.loc[income.isna(), "income_band"] = "Missing"
        except Exception:
            out["income_band"] = np.where(income.isna(), "Missing", "Available")

    minority_candidates = [
        "tract_minority_population_percent",
        "minority_tract_percent",
        "minority_population_percent",
    ]
    minority_col = next((c for c in minority_candidates if c in out.columns), None)
    if minority_col and "minority_tract_band" not in out.columns:
        minority = safe_numeric(out[minority_col])
        out["minority_tract_band"] = pd.cut(
            minority,
            bins=[-np.inf, 20, 50, 80, np.inf],
            labels=["0_20", "20_50", "50_80", "80_plus"],
        ).astype(str)
        out.loc[minority.isna(), "minority_tract_band"] = "Missing"

    ltv_candidates = ["loan_to_value_ratio", "ltv", "combined_loan_to_value_ratio"]
    ltv_col = next((c for c in ltv_candidates if c in out.columns), None)
    if ltv_col and "ltv_band" not in out.columns:
        ltv = safe_numeric(out[ltv_col])
        out["ltv_band"] = pd.cut(
            ltv,
            bins=[-np.inf, 60, 80, 90, 100, np.inf],
            labels=["ltv_le_60", "ltv_60_80", "ltv_80_90", "ltv_90_100", "ltv_gt_100"],
        ).astype(str)
        out.loc[ltv.isna(), "ltv_band"] = "Missing"

    return out


def parse_period_from_column(series: pd.Series, column_name: str) -> Optional[pd.Series]:
    """Return a string period series if the column can form useful monitoring periods."""
    name = column_name.lower()
    non_missing = series.dropna()
    if non_missing.empty:
        return None

    if "quarter" in name or name.endswith("_qtr"):
        return series.astype(str).replace({"nan": np.nan, "None": np.nan})

    if "month" in name and not pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().sum() >= max(20, int(0.05 * len(series))):
            return parsed.dt.to_period("M").astype(str)
        return series.astype(str).replace({"nan": np.nan, "None": np.nan})

    if "year" in name:
        years = safe_numeric(series)
        if years.notna().sum() > 0:
            return years.round().astype("Int64").astype(str).replace({"<NA>": np.nan})

    if "date" in name or "time" in name or "period" in name:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().sum() >= max(20, int(0.05 * len(series))):
            unique_months = parsed.dt.to_period("M").nunique(dropna=True)
            if unique_months >= 6:
                return parsed.dt.to_period("M").astype(str)
            return parsed.dt.to_period("Q").astype(str)
        return series.astype(str).replace({"nan": np.nan, "None": np.nan})

    return None


def assign_monitoring_periods(df: pd.DataFrame) -> Tuple[pd.DataFrame, PeriodSplitInfo]:
    """Use real periods if possible; otherwise create deterministic pseudo-quarters."""
    out = df.copy()

    preferred = [
        "application_date",
        "origination_date",
        "action_taken_date",
        "submission_date",
        "year_month",
        "reporting_period",
        "activity_year",
    ]
    date_like = [
        c
        for c in out.columns
        if any(token in c for token in ["date", "month", "quarter", "period", "year", "time"])
    ]
    candidates = list(dict.fromkeys([c for c in preferred if c in out.columns] + date_like))

    for col in candidates:
        period = parse_period_from_column(out[col], col)
        if period is None:
            continue
        period = period.astype("object")
        unique_periods = sorted(pd.Series(period).dropna().unique().tolist())
        if len(unique_periods) >= 3:
            out["monitoring_period"] = period
            out = out[out["monitoring_period"].notna()].copy()
            periods = sorted(out["monitoring_period"].astype(str).unique().tolist())
            train_count = max(1, int(math.ceil(len(periods) * 0.50)))
            train_periods = periods[:train_count]
            monitor_periods = periods[train_count:]
            if len(monitor_periods) == 0:
                monitor_periods = periods[-1:]
                train_periods = periods[:-1]
            return out, PeriodSplitInfo(
                period_column="monitoring_period",
                period_source=f"Derived from {col}",
                train_periods=train_periods,
                monitor_periods=monitor_periods,
                synthetic_periods_used=False,
            )

    # Fallback: deterministic pseudo-periods. This is common for public HMDA LAR
    # samples that contain activity_year but no transaction month/date.
    base_year = "Synthetic"
    if "activity_year" in out.columns:
        years = safe_numeric(out["activity_year"]).dropna()
        if not years.empty:
            base_year = str(int(years.mode().iloc[0]))

    sort_cols = [c for c in ["activity_year", "loan_amount", "income", "debt_to_income_ratio"] if c in out.columns]
    temp = out.copy()
    temp["_original_order"] = np.arange(len(temp))
    if sort_cols:
        for c in sort_cols:
            if c in temp.columns:
                numeric = safe_numeric(temp[c])
                if numeric.notna().sum() > 0:
                    temp[f"_sort_{c}"] = numeric
                else:
                    temp[f"_sort_{c}"] = temp[c].astype(str)
        sort_by = [f"_sort_{c}" for c in sort_cols] + ["_original_order"]
        temp = temp.sort_values(sort_by, kind="mergesort")
    else:
        temp = temp.sort_values("_original_order", kind="mergesort")

    n_periods = 8 if len(temp) >= 800 else 4
    labels = [f"{base_year}Q{(i % 4) + 1}_sim_{i + 1}" for i in range(n_periods)]
    period_numbers = pd.qcut(
        np.arange(len(temp)),
        q=n_periods,
        labels=False,
        duplicates="drop",
    )
    temp["monitoring_period"] = [labels[int(i)] for i in period_numbers]
    temp = temp.sort_values("_original_order", kind="mergesort")
    out["monitoring_period"] = temp["monitoring_period"].values

    periods = labels[: out["monitoring_period"].nunique()]
    train_count = max(1, len(periods) // 2)
    return out, PeriodSplitInfo(
        period_column="monitoring_period",
        period_source="Deterministic pseudo-quarter simulation because no usable timestamp was found",
        train_periods=periods[:train_count],
        monitor_periods=periods[train_count:],
        synthetic_periods_used=True,
    )


# -----------------------------------------------------------------------------
# Feature preparation and model handling
# -----------------------------------------------------------------------------
def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    leakage_or_admin_patterns = [
        "approved",
        "approval",
        "action_taken",
        "denial_reason",
        "purchaser_type",
        "monitoring_period",
        "row_id",
        "record_id",
        "respondent_id",
    ]

    feature_cols: List[str] = []
    n = len(df)
    for col in df.columns:
        if col == target_col:
            continue
        if any(pattern in col for pattern in leakage_or_admin_patterns):
            continue
        if df[col].isna().mean() > 0.98:
            continue

        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            continue

        # Avoid unstable high-cardinality identifiers in fallback model training.
        if not pd.api.types.is_numeric_dtype(df[col]) and nunique > min(100, max(25, int(0.05 * n))):
            governance_keep = any(
                key in col
                for key in ["race", "ethnicity", "sex", "income_band", "minority_tract", "ltv_band"]
            )
            if not governance_keep:
                continue

        feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No usable feature columns were found for drift simulation.")
    return feature_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore")

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def extract_model_from_artifact(artifact: object) -> Tuple[object, Optional[List[str]]]:
    """Support plain model objects and common dictionary-style artifacts."""
    if isinstance(artifact, dict):
        model_keys = ["model", "pipeline", "champion_model", "estimator", "best_model"]
        feature_keys = ["feature_columns", "features", "model_features", "X_columns", "input_features"]
        model = None
        features = None
        for key in model_keys:
            if key in artifact:
                model = artifact[key]
                break
        for key in feature_keys:
            if key in artifact and artifact[key] is not None:
                features = list(artifact[key])
                features = [clean_column_name(f) for f in features]
                break
        if model is not None:
            return model, features
    return artifact, None


def align_features(X: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    aligned = X.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[list(feature_columns)]


def predict_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):
            proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).ravel()

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-decision))

    pred = np.asarray(model.predict(X), dtype=float).ravel()
    return np.clip(pred, 0.0, 1.0)


def try_load_existing_model(X_reference: pd.DataFrame) -> Optional[ModelInfo]:
    candidates = [
        MODEL_DIR / "champion_model.pkl",
        MODEL_DIR / "champion_model.joblib",
        MODEL_DIR / "calibrated_logistic_regression.pkl",
        MODEL_DIR / "best_model.pkl",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            artifact = joblib.load(path)
            model, features = extract_model_from_artifact(artifact)
            if features is None:
                features = list(X_reference.columns)
            features = [c for c in features if c in X_reference.columns] + [
                c for c in features if c not in X_reference.columns
            ]
            X_aligned = align_features(X_reference, features)
            _ = predict_scores(model, X_aligned.head(min(25, len(X_aligned))))
            return ModelInfo(
                model=model,
                feature_columns=list(features),
                source=str(path.relative_to(ROOT_DIR)),
                used_existing_artifact=True,
            )
        except Exception as exc:
            print(f"Warning: could not use model artifact {path.name}: {exc}")

    return None


def train_fallback_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_periods: Sequence[str],
) -> ModelInfo:
    train_df = df[df["monitoring_period"].astype(str).isin(train_periods)].copy()
    train_df = train_df[train_df[target_col].notna()].copy()
    if train_df[target_col].nunique() < 2:
        raise ValueError("Training split has fewer than two target classes; cannot train fallback model.")

    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE, stratify=train_df[target_col])

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(int)

    preprocessor = make_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    fallback_path = MODEL_DIR / "monitoring_fallback_model.pkl"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "purpose": "Fallback monitoring model trained by scripts/11_monitoring_drift_simulation.py",
        },
        fallback_path,
    )

    return ModelInfo(
        model=model,
        feature_columns=feature_cols,
        source=str(fallback_path.relative_to(ROOT_DIR)),
        used_existing_artifact=False,
    )


# -----------------------------------------------------------------------------
# Drift metrics
# -----------------------------------------------------------------------------
def distribution_index(expected_props: np.ndarray, actual_props: np.ndarray) -> float:
    eps = 1e-6
    expected = np.clip(np.asarray(expected_props, dtype=float), eps, None)
    actual = np.clip(np.asarray(actual_props, dtype=float), eps, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = PSI_BINS) -> float:
    exp = safe_numeric(expected).dropna()
    act = safe_numeric(actual).dropna()
    if len(exp) == 0 or len(act) == 0:
        return np.nan

    if exp.nunique() <= 1:
        return 0.0 if act.nunique() <= 1 else np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(exp, quantiles))
    if len(edges) < 3:
        edges = np.linspace(exp.min(), exp.max(), min(bins, exp.nunique()) + 1)
    if len(edges) < 3:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts = pd.cut(exp, bins=edges, include_lowest=True).value_counts(sort=False)
    act_counts = pd.cut(act, bins=edges, include_lowest=True).value_counts(sort=False)
    return distribution_index(exp_counts.values, act_counts.values)


def psi_categorical(expected: pd.Series, actual: pd.Series, max_categories: int = 30) -> float:
    exp = expected.astype("object").where(expected.notna(), "Missing").astype(str)
    act = actual.astype("object").where(actual.notna(), "Missing").astype(str)
    if exp.empty or act.empty:
        return np.nan

    top_categories = exp.value_counts().head(max_categories).index.tolist()
    exp_grouped = exp.where(exp.isin(top_categories), "Other")
    act_grouped = act.where(act.isin(top_categories), "Other")
    categories = sorted(set(exp_grouped.unique()).union(set(act_grouped.unique())))
    exp_counts = exp_grouped.value_counts().reindex(categories, fill_value=0)
    act_counts = act_grouped.value_counts().reindex(categories, fill_value=0)
    return distribution_index(exp_counts.values, act_counts.values)


def characteristic_stability(expected_df: pd.DataFrame, actual_df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in expected_df.columns or col not in actual_df.columns:
            continue
        exp = expected_df[col]
        act = actual_df[col]
        numeric_ratio = safe_numeric(exp).notna().mean()
        if pd.api.types.is_numeric_dtype(exp) or numeric_ratio > 0.80:
            value = psi_numeric(exp, act)
            metric_type = "numeric_csi"
        else:
            value = psi_categorical(exp, act)
            metric_type = "categorical_csi"
        rows.append(
            {
                "feature": col,
                "metric_type": metric_type,
                "csi": value,
                "drift_level": classify_stability_index(value),
            }
        )
    return pd.DataFrame(rows).sort_values("csi", ascending=False, na_position="last")


def expected_calibration_error(y_true: pd.Series, y_score: pd.Series, bins: int = 10) -> float:
    y = pd.Series(y_true).dropna().astype(int)
    s = pd.Series(y_score).loc[y.index].astype(float)
    if y.empty or s.empty:
        return np.nan

    edges = np.linspace(0, 1, bins + 1)
    bucket = pd.cut(s, bins=edges, include_lowest=True, labels=False)
    ece = 0.0
    total = len(y)
    for b in range(bins):
        idx = bucket == b
        if idx.sum() == 0:
            continue
        observed = y[idx].mean()
        predicted = s[idx].mean()
        ece += (idx.sum() / total) * abs(observed - predicted)
    return float(ece)


def binary_classification_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.50) -> Dict[str, float]:
    y = pd.Series(y_true).astype(int)
    score = pd.Series(y_score).astype(float)
    pred = (score >= threshold).astype(int)

    metrics: Dict[str, float] = {}
    metrics["event_rate"] = float(y.mean()) if len(y) else np.nan
    metrics["score_mean"] = float(score.mean()) if len(score) else np.nan
    metrics["approval_rate"] = float(pred.mean()) if len(pred) else np.nan

    if y.nunique() >= 2:
        metrics["auc"] = float(roc_auc_score(y, score))
    else:
        metrics["auc"] = np.nan

    metrics["accuracy"] = float(accuracy_score(y, pred)) if len(y) else np.nan
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y, pred)) if y.nunique() >= 2 else np.nan
    metrics["precision"] = float(precision_score(y, pred, zero_division=0)) if len(y) else np.nan
    metrics["recall"] = float(recall_score(y, pred, zero_division=0)) if len(y) else np.nan
    metrics["f1"] = float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan
    metrics["brier"] = float(brier_score_loss(y, score)) if len(y) else np.nan
    metrics["ece"] = expected_calibration_error(y, score)

    if y.nunique() >= 2:
        labels = [0, 1]
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=labels).ravel()
        metrics["false_positive_rate"] = safe_divide(fp, fp + tn)
        metrics["false_negative_rate"] = safe_divide(fn, fn + tp)
    else:
        metrics["false_positive_rate"] = np.nan
        metrics["false_negative_rate"] = np.nan

    return metrics


def find_fairness_group_columns(df: pd.DataFrame) -> List[str]:
    priority = [
        "race",
        "derived_race",
        "applicant_race_1",
        "ethnicity",
        "derived_ethnicity",
        "applicant_ethnicity_1",
        "sex",
        "derived_sex",
        "applicant_sex",
        "income_band",
        "minority_tract_band",
    ]
    found = [c for c in priority if c in df.columns]
    pattern_found = [
        c
        for c in df.columns
        if any(key in c for key in ["race", "ethnicity", "sex", "income_band", "minority_tract_band"])
        and c not in found
    ]
    cols = list(dict.fromkeys(found + pattern_found))

    usable = []
    for col in cols:
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 30:
            usable.append(col)
    return usable[:8]


def fairness_metrics_by_period(
    df_period: pd.DataFrame,
    target_col: str,
    score_col: str,
    group_cols: Sequence[str],
    threshold: float = 0.50,
) -> Dict[str, float]:
    results: Dict[str, float] = {
        "fairness_max_approval_gap": np.nan,
        "fairness_max_fnr_gap": np.nan,
        "fairness_max_fpr_gap": np.nan,
        "fairness_worst_group_column": None,
    }

    gaps = []
    fnr_gaps = []
    fpr_gaps = []

    if df_period.empty:
        return results

    temp = df_period.copy()
    temp["_pred"] = (temp[score_col].astype(float) >= threshold).astype(int)
    temp["_y"] = temp[target_col].astype(int)

    for group_col in group_cols:
        grouped = temp[[group_col, "_pred", "_y"]].copy()
        grouped[group_col] = grouped[group_col].astype("object").where(grouped[group_col].notna(), "Missing").astype(str)
        counts = grouped[group_col].value_counts()
        valid_groups = counts[counts >= max(20, int(0.01 * len(grouped)))].index.tolist()
        grouped = grouped[grouped[group_col].isin(valid_groups)]
        if grouped[group_col].nunique() < 2:
            continue

        approval_rates = grouped.groupby(group_col)["_pred"].mean()
        approval_gap = float(approval_rates.max() - approval_rates.min())
        gaps.append((group_col, approval_gap))

        fnrs = []
        fprs = []
        for _, g in grouped.groupby(group_col):
            y = g["_y"]
            p = g["_pred"]
            if y.nunique() < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            fnrs.append(safe_divide(fn, fn + tp))
            fprs.append(safe_divide(fp, fp + tn))

        fnrs = [x for x in fnrs if not pd.isna(x)]
        fprs = [x for x in fprs if not pd.isna(x)]
        if len(fnrs) >= 2:
            fnr_gaps.append((group_col, float(max(fnrs) - min(fnrs))))
        if len(fprs) >= 2:
            fpr_gaps.append((group_col, float(max(fprs) - min(fprs))))

    if gaps:
        worst = max(gaps, key=lambda x: x[1])
        results["fairness_max_approval_gap"] = worst[1]
        results["fairness_worst_group_column"] = worst[0]
    if fnr_gaps:
        results["fairness_max_fnr_gap"] = max(fnr_gaps, key=lambda x: x[1])[1]
    if fpr_gaps:
        results["fairness_max_fpr_gap"] = max(fpr_gaps, key=lambda x: x[1])[1]

    return results


# -----------------------------------------------------------------------------
# Monitoring summary, plotting, and report writing
# -----------------------------------------------------------------------------
def choose_csi_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    preferred_terms = [
        "loan_amount",
        "income",
        "debt_to_income",
        "loan_to_value",
        "ltv",
        "race",
        "ethnicity",
        "sex",
        "minority_tract",
        "tract",
        "property_value",
        "loan_purpose",
        "occupancy",
        "loan_type",
    ]
    preferred = [c for c in feature_cols if any(term in c for term in preferred_terms)]
    remaining = [c for c in feature_cols if c not in preferred]
    selected = list(dict.fromkeys(preferred + remaining))[:MAX_FEATURES_FOR_CSI]
    return selected


def build_monitoring_summary(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    split_info: PeriodSplitInfo,
    model_info: ModelInfo,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    X_all = align_features(df[feature_cols].copy(), model_info.feature_columns)
    df = df.copy()
    df["model_score"] = predict_scores(model_info.model, X_all)
    df["model_prediction"] = (df["model_score"] >= 0.50).astype(int)

    train_mask = df["monitoring_period"].astype(str).isin(split_info.train_periods)
    baseline = df[train_mask].copy()
    if baseline.empty:
        raise ValueError("Baseline training period is empty after period assignment.")

    csi_features = choose_csi_features(df, feature_cols)
    fairness_cols = find_fairness_group_columns(df)

    baseline_metrics = binary_classification_metrics(baseline[target_col], baseline["model_score"])
    baseline_fairness = fairness_metrics_by_period(
        baseline,
        target_col=target_col,
        score_col="model_score",
        group_cols=fairness_cols,
    )

    rows: List[Dict[str, object]] = []
    csi_all_periods: List[pd.DataFrame] = []
    periods = split_info.train_periods + split_info.monitor_periods

    for period in periods:
        period_df = df[df["monitoring_period"].astype(str) == str(period)].copy()
        if period_df.empty:
            continue

        metrics = binary_classification_metrics(period_df[target_col], period_df["model_score"])
        fairness = fairness_metrics_by_period(
            period_df,
            target_col=target_col,
            score_col="model_score",
            group_cols=fairness_cols,
        )
        csi_df = characteristic_stability(baseline, period_df, csi_features)
        if not csi_df.empty:
            csi_df.insert(0, "monitoring_period", period)
            csi_all_periods.append(csi_df)

        csi_mean = float(csi_df["csi"].mean()) if not csi_df.empty else np.nan
        csi_max = float(csi_df["csi"].max()) if not csi_df.empty else np.nan
        csi_max_feature = None if csi_df.empty else str(csi_df.iloc[0]["feature"])
        score_psi = psi_numeric(baseline["model_score"], period_df["model_score"])

        auc_change = metrics["auc"] - baseline_metrics["auc"] if not pd.isna(metrics["auc"]) and not pd.isna(baseline_metrics["auc"]) else np.nan
        brier_change = metrics["brier"] - baseline_metrics["brier"] if not pd.isna(metrics["brier"]) and not pd.isna(baseline_metrics["brier"]) else np.nan
        ece_change = metrics["ece"] - baseline_metrics["ece"] if not pd.isna(metrics["ece"]) and not pd.isna(baseline_metrics["ece"]) else np.nan
        approval_rate_change = metrics["approval_rate"] - baseline_metrics["approval_rate"] if not pd.isna(metrics["approval_rate"]) else np.nan
        fairness_approval_change = fairness["fairness_max_approval_gap"] - baseline_fairness["fairness_max_approval_gap"] if not pd.isna(fairness["fairness_max_approval_gap"]) and not pd.isna(baseline_fairness["fairness_max_approval_gap"]) else np.nan

        breaches: List[str] = []
        if classify_stability_index(score_psi) == "High":
            breaches.append("High prediction PSI")
        elif classify_stability_index(score_psi) == "Moderate":
            breaches.append("Moderate prediction PSI")

        if classify_stability_index(csi_max) == "High":
            breaches.append("High characteristic drift")
        elif classify_stability_index(csi_max) == "Moderate":
            breaches.append("Moderate characteristic drift")

        if not pd.isna(auc_change) and auc_change <= -0.05:
            breaches.append("High AUC deterioration")
        elif not pd.isna(auc_change) and auc_change <= -0.03:
            breaches.append("Moderate AUC deterioration")

        if not pd.isna(brier_change) and brier_change >= 0.03:
            breaches.append("High Brier deterioration")
        elif not pd.isna(brier_change) and brier_change >= 0.015:
            breaches.append("Moderate Brier deterioration")

        if not pd.isna(ece_change) and ece_change >= 0.04:
            breaches.append("High calibration drift")
        elif not pd.isna(ece_change) and ece_change >= 0.02:
            breaches.append("Moderate calibration drift")

        if not pd.isna(fairness["fairness_max_approval_gap"]) and fairness["fairness_max_approval_gap"] >= 0.15:
            breaches.append("High fairness approval-rate gap")
        elif not pd.isna(fairness["fairness_max_approval_gap"]) and fairness["fairness_max_approval_gap"] >= 0.10:
            breaches.append("Moderate fairness approval-rate gap")

        if not pd.isna(fairness_approval_change) and fairness_approval_change >= 0.05:
            breaches.append("Fairness drift versus baseline")

        period_role = "Baseline train period" if str(period) in split_info.train_periods else "Monitoring period"
        if period_role == "Baseline train period":
            # Baseline rows are shown for transparency but should not be escalated against themselves.
            breaches = []

        row = {
            "monitoring_period": period,
            "period_role": period_role,
            "n_records": int(len(period_df)),
            "model_source": model_info.source,
            "used_existing_model_artifact": model_info.used_existing_artifact,
            "event_rate": metrics["event_rate"],
            "score_mean": metrics["score_mean"],
            "approval_rate": metrics["approval_rate"],
            "approval_rate_change_vs_baseline": approval_rate_change,
            "auc": metrics["auc"],
            "auc_change_vs_baseline": auc_change,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "brier": metrics["brier"],
            "brier_change_vs_baseline": brier_change,
            "ece": metrics["ece"],
            "ece_change_vs_baseline": ece_change,
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
            "prediction_psi": score_psi,
            "prediction_drift_level": classify_stability_index(score_psi),
            "csi_mean": csi_mean,
            "csi_max": csi_max,
            "csi_max_feature": csi_max_feature,
            "data_drift_level": classify_stability_index(csi_max),
            "fairness_group_columns_tested": "; ".join(fairness_cols),
            "fairness_worst_group_column": fairness["fairness_worst_group_column"],
            "fairness_max_approval_gap": fairness["fairness_max_approval_gap"],
            "fairness_max_approval_gap_change_vs_baseline": fairness_approval_change,
            "fairness_max_fnr_gap": fairness["fairness_max_fnr_gap"],
            "fairness_max_fpr_gap": fairness["fairness_max_fpr_gap"],
            "monitoring_status": status_from_breaches(breaches),
            "breach_reasons": "; ".join(breaches) if breaches else "None",
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    csi_detail = pd.concat(csi_all_periods, ignore_index=True) if csi_all_periods else pd.DataFrame()
    return summary, csi_detail, fairness_cols


def create_dashboard_plot(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    plot_df = summary.copy()
    plot_df["monitoring_period"] = plot_df["monitoring_period"].astype(str)
    x = np.arange(len(plot_df))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(x, plot_df["prediction_psi"], marker="o")
    axes[0, 0].axhline(0.10, linestyle="--", linewidth=1)
    axes[0, 0].axhline(0.25, linestyle="--", linewidth=1)
    axes[0, 0].set_title("Prediction Drift: Score PSI")
    axes[0, 0].set_ylabel("PSI")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(plot_df["monitoring_period"], rotation=45, ha="right")

    axes[0, 1].plot(x, plot_df["csi_max"], marker="o")
    axes[0, 1].axhline(0.10, linestyle="--", linewidth=1)
    axes[0, 1].axhline(0.25, linestyle="--", linewidth=1)
    axes[0, 1].set_title("Data Drift: Maximum Characteristic Stability Index")
    axes[0, 1].set_ylabel("CSI")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(plot_df["monitoring_period"], rotation=45, ha="right")

    axes[1, 0].plot(x, plot_df["auc"], marker="o", label="AUC")
    axes[1, 0].plot(x, plot_df["brier"], marker="o", label="Brier")
    axes[1, 0].set_title("Performance Drift")
    axes[1, 0].set_ylabel("Metric value")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(plot_df["monitoring_period"], rotation=45, ha="right")
    axes[1, 0].legend()

    axes[1, 1].plot(x, plot_df["ece"], marker="o", label="ECE")
    if "fairness_max_approval_gap" in plot_df.columns:
        axes[1, 1].plot(x, plot_df["fairness_max_approval_gap"], marker="o", label="Fairness approval gap")
    axes[1, 1].set_title("Calibration and Fairness Drift")
    axes[1, 1].set_ylabel("Metric value")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(plot_df["monitoring_period"], rotation=45, ha="right")
    axes[1, 1].legend()

    fig.suptitle("AI Governance Monitoring and Drift Dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_number(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def write_monitoring_plan(
    summary: pd.DataFrame,
    split_info: PeriodSplitInfo,
    model_info: ModelInfo,
    target_col: str,
    feature_cols: Sequence[str],
    fairness_cols: Sequence[str],
    csi_detail: pd.DataFrame,
) -> None:
    monitoring_rows = summary[summary["period_role"] == "Monitoring period"].copy()
    red_count = int((monitoring_rows["monitoring_status"] == "Red").sum()) if not monitoring_rows.empty else 0
    amber_count = int((monitoring_rows["monitoring_status"] == "Amber").sum()) if not monitoring_rows.empty else 0
    green_count = int((monitoring_rows["monitoring_status"] == "Green").sum()) if not monitoring_rows.empty else 0

    worst_csi = None
    if not csi_detail.empty and csi_detail["csi"].notna().any():
        worst_csi = csi_detail.sort_values("csi", ascending=False).iloc[0]

    lines = []
    lines.append("# Monitoring and Drift Simulation Plan")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append(
        "This document defines an ongoing monitoring framework for the HMDA approval model used in this repository. "
        "The objective is to detect material deterioration after model development, including data drift, prediction drift, "
        "performance drift, fairness drift, and calibration drift."
    )
    lines.append("")
    lines.append("## 2. Monitoring Design")
    lines.append(f"- **Target variable:** `{target_col}`")
    lines.append(f"- **Model used for monitoring:** `{model_info.source}`")
    lines.append(f"- **Existing model artifact used:** `{model_info.used_existing_artifact}`")
    lines.append(f"- **Period source:** {split_info.period_source}")
    lines.append(f"- **Baseline train periods:** {', '.join(map(str, split_info.train_periods))}")
    lines.append(f"- **Monitoring periods:** {', '.join(map(str, split_info.monitor_periods))}")
    lines.append(f"- **Synthetic periods used:** `{split_info.synthetic_periods_used}`")
    if split_info.synthetic_periods_used:
        lines.append(
            "- **Limitation:** The processed public HMDA file did not provide a usable transaction-level timestamp. "
            "The script therefore created deterministic pseudo-periods. These periods support governance simulation, "
            "but they should be replaced with true monthly or quarterly production cohorts in a deployed system."
        )
    lines.append("")
    lines.append("## 3. Metrics")
    lines.append("The monitoring process computes the following controls:")
    lines.append("- **Population Stability Index:** drift in the model score distribution versus the baseline period.")
    lines.append("- **Characteristic Stability Index:** drift in selected input-feature distributions versus the baseline period.")
    lines.append("- **Data drift:** maximum and average CSI across monitored characteristics.")
    lines.append("- **Prediction drift:** PSI of predicted probabilities and movement in the model approval rate.")
    lines.append("- **Performance drift:** AUC, accuracy, balanced accuracy, precision, recall, F1, Brier score, false-positive rate, and false-negative rate.")
    lines.append("- **Fairness drift:** group-level approval-rate gap, false-negative-rate gap, and false-positive-rate gap.")
    lines.append("- **Calibration drift:** expected calibration error and Brier score movement versus baseline.")
    lines.append("")
    lines.append("## 4. Thresholds and Escalation")
    lines.append("| Metric family | Green | Amber | Red |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| PSI or CSI | < 0.10 | 0.10 to < 0.25 | >= 0.25 |")
    lines.append("| AUC change vs. baseline | > -0.03 | -0.05 to -0.03 | <= -0.05 |")
    lines.append("| Brier score increase | < 0.015 | 0.015 to < 0.030 | >= 0.030 |")
    lines.append("| ECE increase | < 0.020 | 0.020 to < 0.040 | >= 0.040 |")
    lines.append("| Group approval-rate gap | < 0.10 | 0.10 to < 0.15 | >= 0.15 |")
    lines.append("")
    lines.append("Recommended escalation:")
    lines.append("- **Green:** Continue scheduled monitoring.")
    lines.append("- **Amber:** Perform analyst review, feature-level diagnosis, and business-context assessment.")
    lines.append("- **Red:** Open a model-risk issue, notify the model owner and validator, assess customer impact, and consider recalibration, challenger review, policy override, or model retirement.")
    lines.append("")
    lines.append("## 5. Monitoring Results Summary")
    lines.append(f"- Green monitoring periods: **{green_count}**")
    lines.append(f"- Amber monitoring periods: **{amber_count}**")
    lines.append(f"- Red monitoring periods: **{red_count}**")
    if worst_csi is not None:
        lines.append(
            f"- Largest characteristic drift: **{worst_csi['feature']}** with CSI **{format_number(worst_csi['csi'])}** "
            f"during period **{worst_csi['monitoring_period']}**."
        )
    lines.append("")
    lines.append("## 6. Fairness Monitoring Scope")
    if fairness_cols:
        lines.append("The following group variables were tested for fairness drift:")
        for col in fairness_cols:
            lines.append(f"- `{col}`")
    else:
        lines.append("No usable group variables were available for fairness drift testing. Add race, ethnicity, sex, income band, and minority-tract band fields when available.")
    lines.append("")
    lines.append("## 7. Governance Alignment")
    lines.append(
        "This monitoring plan supports model lifecycle governance by defining measurable post-development controls, thresholds, "
        "escalation criteria, documentation artifacts, and review evidence. For high-risk AI use cases, this type of monitoring "
        "also supports post-market or post-deployment control expectations by collecting and analyzing performance and compliance evidence over time."
    )
    lines.append("")
    lines.append("## 8. Generated Evidence")
    lines.append(f"- `{SUMMARY_PATH.relative_to(ROOT_DIR)}`")
    lines.append(f"- `{FIGURE_PATH.relative_to(ROOT_DIR)}`")
    lines.append(f"- `{PLAN_PATH.relative_to(ROOT_DIR)}`")
    lines.append("")
    lines.append("## 9. Implementation Notes")
    lines.append(f"- Number of model features considered: **{len(feature_cols)}**")
    lines.append("- The baseline period is used as the reference distribution for PSI and CSI.")
    lines.append("- The script is intentionally conservative: when a champion artifact is incompatible, it trains a fallback monitoring model and records that fact.")
    lines.append("- Production implementation should replace pseudo-periods, if used, with actual monthly or quarterly production cohorts.")
    lines.append("")

    PLAN_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(summary: pd.DataFrame, csi_detail: pd.DataFrame) -> None:
    summary.to_csv(SUMMARY_PATH, index=False)
    if not csi_detail.empty:
        csi_detail_path = TABLE_DIR / "characteristic_stability_detail.csv"
        csi_detail.to_csv(csi_detail_path, index=False)
    create_dashboard_plot(summary)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_directories()

    print("Loading processed HMDA modeling dataset...")
    df = load_dataset()
    df, target_col = identify_or_create_target(df)
    df = add_governance_bands(df)

    print("Assigning time-based or simulated monitoring periods...")
    df, split_info = assign_monitoring_periods(df)

    feature_cols = select_feature_columns(df, target_col)
    print(f"Selected {len(feature_cols)} candidate monitoring features.")

    X_reference = df[feature_cols].head(min(len(df), 5000)).copy()
    model_info = try_load_existing_model(X_reference)
    if model_info is None:
        print("No compatible champion model artifact found. Training fallback monitoring model...")
        model_info = train_fallback_model(df, feature_cols, target_col, split_info.train_periods)
    else:
        print(f"Using existing model artifact: {model_info.source}")

    print("Computing drift, performance, fairness, and calibration monitoring metrics...")
    summary, csi_detail, fairness_cols = build_monitoring_summary(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        split_info=split_info,
        model_info=model_info,
    )

    print("Saving monitoring outputs...")
    save_outputs(summary, csi_detail)
    write_monitoring_plan(
        summary=summary,
        split_info=split_info,
        model_info=model_info,
        target_col=target_col,
        feature_cols=feature_cols,
        fairness_cols=fairness_cols,
        csi_detail=csi_detail,
    )

    print("\nStep M complete.")
    print(f"Created: {SUMMARY_PATH.relative_to(ROOT_DIR)}")
    print(f"Created: {FIGURE_PATH.relative_to(ROOT_DIR)}")
    print(f"Created: {PLAN_PATH.relative_to(ROOT_DIR)}")
    if not csi_detail.empty:
        print(f"Created: {(TABLE_DIR / 'characteristic_stability_detail.csv').relative_to(ROOT_DIR)}")

    status_counts = summary[summary["period_role"] == "Monitoring period"]["monitoring_status"].value_counts().to_dict()
    print(f"Monitoring status counts: {status_counts}")


if __name__ == "__main__":
    main()
