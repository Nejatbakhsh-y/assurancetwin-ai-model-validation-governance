"""
Step K — Explainability and Stability

Creates:
    reports/figures/feature_importance.png
    reports/figures/shap_summary.png
    reports/tables/explanation_stability.csv

Purpose:
    This script evaluates whether model explanations remain stable across:
        1. Overall feature importance
        2. Demographic groups
        3. Time-like splits
        4. Champion versus challenger models

Governance angle:
    A high-performing model can still be weak from a governance perspective
    if its explanations are unstable across groups, time periods, or model choices.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hmda_modeling_dataset.csv"

CHAMPION_MODEL_PATH = PROJECT_ROOT / "models" / "champion_model.pkl"
CHALLENGER_MODEL_PATH = PROJECT_ROOT / "models" / "challenger_model.pkl"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"
SHAP_SUMMARY_PATH = FIGURES_DIR / "shap_summary.png"
STABILITY_TABLE_PATH = TABLES_DIR / "explanation_stability.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

RANDOM_STATE = 42

MAX_FULL_IMPORTANCE_ROWS = 4000
MAX_SEGMENT_IMPORTANCE_ROWS = 1500
MAX_SHAP_ROWS = 500
MAX_SHAP_BACKGROUND_ROWS = 100

MIN_SEGMENT_ROWS = 100
TOP_N_FEATURES = 10

TARGET_CANDIDATES = [
    "approved",
    "target",
    "approval_target",
    "loan_approved",
    "y",
]

NON_FEATURE_OR_LEAKAGE_COLUMNS = [
    "approved",
    "target",
    "approval_target",
    "loan_approved",
    "y",
    "action_taken",
    "action_taken_name",
    "action_taken_code",
    "action_taken_label",
    "lei",
    "loan_id",
    "application_id",
    "respondent_id",
]

GROUP_COLUMN_CANDIDATES = {
    "race": [
        "derived_race",
        "race",
        "applicant_race",
        "applicant_race_1",
    ],
    "ethnicity": [
        "derived_ethnicity",
        "ethnicity",
        "applicant_ethnicity",
        "applicant_ethnicity_1",
    ],
    "sex": [
        "derived_sex",
        "sex",
        "applicant_sex",
    ],
    "income_band": [
        "income_band",
        "applicant_income_band",
        "income_category",
    ],
    "minority_tract_band": [
        "minority_tract_band",
        "minority_population_band",
        "tract_minority_population_percent_band",
    ],
}

NUMERIC_BAND_CANDIDATES = {
    "income_band": [
        "income",
        "applicant_income",
    ],
    "minority_tract_band": [
        "tract_minority_population_percent",
        "minority_population_percent",
    ],
}

TIME_COLUMN_CANDIDATES = [
    "activity_year",
    "year",
    "application_year",
    "loan_year",
    "date",
    "application_date",
    "submission_date",
]


# ---------------------------------------------------------------------
# Load data and models
# ---------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find modeling dataset:\n{DATA_PATH}\n\n"
            "Run the earlier clean-dataset step first."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(f"The modeling dataset is empty:\n{DATA_PATH}")

    return df


def unwrap_model_artifact(artifact, model_label: str):
    """
    The training script may save either:
        1. a fitted sklearn estimator directly, or
        2. a dictionary containing the fitted estimator under a key such as 'model'.

    sklearn functions require the estimator itself, not the metadata dictionary.
    """
    if isinstance(artifact, dict):
        possible_model_keys = [
            "model",
            "estimator",
            "pipeline",
            "fitted_model",
            "best_model",
            "classifier",
        ]

        for key in possible_model_keys:
            if key in artifact:
                model = artifact[key]
                print(f"{model_label} model loaded from dictionary key: '{key}'")
                return model, artifact

        for key, value in artifact.items():
            if hasattr(value, "predict"):
                print(f"{model_label} model inferred from dictionary key: '{key}'")
                return value, artifact

        raise ValueError(
            f"{model_label} artifact is a dictionary, but no fitted model was found. "
            f"Available keys: {list(artifact.keys())}"
        )

    print(f"{model_label} model loaded as a direct sklearn estimator.")
    return artifact, {}


def load_model(path: Path, model_label: str):
    if not path.exists():
        print(f"WARNING: {model_label} model not found: {path}")
        return None, {}

    artifact = joblib.load(path)
    model, metadata = unwrap_model_artifact(artifact, model_label)

    if not hasattr(model, "predict"):
        raise TypeError(
            f"{model_label} object does not look like a fitted sklearn estimator. "
            f"Loaded type: {type(model)}"
        )

    return model, metadata


# ---------------------------------------------------------------------
# Feature and target preparation
# ---------------------------------------------------------------------

def find_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a binary target column. Expected one of:\n"
        f"{TARGET_CANDIDATES}"
    )


def clean_binary_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return y.fillna(0).astype(int)

    positive_values = {
        "1",
        "yes",
        "y",
        "true",
        "approved",
        "approval",
        "originated",
        "loan originated",
    }

    return (
        y.astype(str)
        .str.strip()
        .str.lower()
        .map(lambda value: 1 if value in positive_values else 0)
        .astype(int)
    )


def get_expected_feature_columns_from_model(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def get_expected_feature_columns_from_metadata(metadata: Dict) -> Optional[List[str]]:
    possible_keys = [
        "feature_columns",
        "features",
        "model_features",
        "input_features",
        "selected_features",
    ]

    for key in possible_keys:
        if key in metadata and isinstance(metadata[key], list):
            return list(metadata[key])

    return None


def prepare_features(
    df: pd.DataFrame,
    model,
    metadata: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    metadata = metadata or {}

    target_col = find_target_column(df)
    y = clean_binary_target(df[target_col])

    expected_columns = get_expected_feature_columns_from_model(model)

    if expected_columns is None:
        expected_columns = get_expected_feature_columns_from_metadata(metadata)

    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]

        if missing:
            raise ValueError(
                "The saved model expects columns that are not present in the current dataset.\n"
                f"Missing columns: {missing[:30]}"
            )

        X = df[expected_columns].copy()
        return X, y, target_col

    drop_cols = [col for col in NON_FEATURE_OR_LEAKAGE_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    if target_col in X.columns:
        X = X.drop(columns=[target_col])

    if X.empty:
        raise ValueError("No feature columns remain after removing target/leakage columns.")

    return X, y, target_col


def sample_data(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X.copy(), y.copy()

    sample_index = X.sample(
        n=max_rows,
        random_state=random_state,
        replace=False,
    ).index

    return X.loc[sample_index].copy(), y.loc[sample_index].copy()


# ---------------------------------------------------------------------
# Prediction and scoring
# ---------------------------------------------------------------------

def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)

        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]

        if proba.ndim == 1:
            return proba

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))

    pred = np.asarray(model.predict(X)).astype(float)
    return pred


def safe_auc_or_accuracy_scorer(model, X_eval: pd.DataFrame, y_eval: pd.Series) -> float:
    y_eval = pd.Series(y_eval)

    if y_eval.nunique() >= 2:
        try:
            proba = predict_positive_probability(model, X_eval)
            return roc_auc_score(y_eval, proba)
        except Exception:
            pass

    try:
        pred = model.predict(X_eval)
        return accuracy_score(y_eval, pred)
    except Exception:
        proba = predict_positive_probability(model, X_eval)
        pred = (proba >= 0.5).astype(int)
        return accuracy_score(y_eval, pred)


# ---------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------

def compute_permutation_importance_df(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_repeats: int = 5,
    max_rows: int = MAX_FULL_IMPORTANCE_ROWS,
) -> pd.DataFrame:
    X_sample, y_sample = sample_data(X, y, max_rows=max_rows)

    result = permutation_importance(
        estimator=model,
        X=X_sample,
        y=y_sample,
        scoring=safe_auc_or_accuracy_scorer,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "model": model_name,
        }
    )

    importance_df["importance_abs"] = importance_df["importance_mean"].abs()

    importance_df = importance_df.sort_values(
        "importance_abs",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    top_df = importance_df.head(20).copy()
    top_df = top_df.sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance_mean"])
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.title("Champion Model Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------

def get_final_estimator(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def unwrap_calibrated_estimator(estimator):
    if hasattr(estimator, "calibrated_classifiers_"):
        calibrated_classifiers = estimator.calibrated_classifiers_

        if calibrated_classifiers:
            first = calibrated_classifiers[0]

            for attr in ["estimator", "base_estimator", "classifier"]:
                if hasattr(first, attr):
                    return getattr(first, attr)

    return estimator


def transform_for_final_estimator(
    model,
    X: pd.DataFrame,
) -> Tuple[object, List[str]]:
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        transformer = model[:-1]
        X_transformed = transformer.transform(X)

        try:
            feature_names = list(transformer.get_feature_names_out(X.columns))
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        return X_transformed, feature_names

    return X, list(X.columns)


def to_dense_if_small_enough(X_matrix, max_cells: int = 1_000_000):
    if hasattr(X_matrix, "toarray"):
        rows, cols = X_matrix.shape

        if rows * cols <= max_cells:
            return X_matrix.toarray()

    return X_matrix


def create_fallback_shap_plot(
    importance_df: pd.DataFrame,
    output_path: Path,
    reason: str,
) -> None:
    fallback_df = importance_df.head(20).copy()
    fallback_df = fallback_df.sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(fallback_df["feature"], fallback_df["importance_mean"])
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.title("SHAP Summary Fallback: Permutation-Based Explanation")
    plt.figtext(
        0.01,
        0.01,
        f"SHAP was not generated. Reason: {reason}",
        fontsize=8,
        ha="left",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_shap_summary_plot(
    model,
    X: pd.DataFrame,
    champion_importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    try:
        import shap
    except Exception as exc:
        create_fallback_shap_plot(
            champion_importance_df,
            output_path,
            reason=f"SHAP import failed: {exc}",
        )
        return

    try:
        y_dummy = pd.Series(np.zeros(len(X)), index=X.index)

        X_shap, _ = sample_data(
            X,
            y_dummy,
            max_rows=MAX_SHAP_ROWS,
            random_state=RANDOM_STATE,
        )

        X_background, _ = sample_data(
            X,
            y_dummy,
            max_rows=MAX_SHAP_BACKGROUND_ROWS,
            random_state=RANDOM_STATE + 1,
        )

        final_estimator = get_final_estimator(model)
        shap_estimator = unwrap_calibrated_estimator(final_estimator)

        X_transformed, feature_names = transform_for_final_estimator(model, X_shap)
        X_background_transformed, _ = transform_for_final_estimator(model, X_background)

        X_transformed = to_dense_if_small_enough(X_transformed)
        X_background_transformed = to_dense_if_small_enough(X_background_transformed)

        if hasattr(shap_estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(shap_estimator)
            shap_values = explainer.shap_values(X_transformed)

        elif hasattr(shap_estimator, "coef_"):
            explainer = shap.LinearExplainer(
                shap_estimator,
                X_background_transformed,
            )
            shap_values = explainer.shap_values(X_transformed)

        else:
            raise ValueError(
                "The final estimator is not tree-based or linear. "
                "A fallback explanation plot was created."
            )

        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values_to_plot = shap_values[1]
            else:
                shap_values_to_plot = shap_values[0]
        else:
            shap_values_to_plot = np.asarray(shap_values)

            if shap_values_to_plot.ndim == 3:
                shap_values_to_plot = shap_values_to_plot[:, :, -1]

        plt.figure()
        shap.summary_plot(
            shap_values_to_plot,
            X_transformed,
            feature_names=feature_names,
            plot_type="bar",
            max_display=20,
            show=False,
        )
        plt.title("Champion Model SHAP Summary")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as exc:
        create_fallback_shap_plot(
            champion_importance_df,
            output_path,
            reason=str(exc),
        )


# ---------------------------------------------------------------------
# Explanation stability
# ---------------------------------------------------------------------

def top_features_string(importance_df: pd.DataFrame, top_n: int = TOP_N_FEATURES) -> str:
    return "; ".join(importance_df.head(top_n)["feature"].astype(str).tolist())


def align_importance_vectors(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    reference = reference_df.set_index("feature")["importance_mean"]
    comparison = comparison_df.set_index("feature")["importance_mean"]

    features = sorted(set(reference.index).union(set(comparison.index)))

    ref_vector = reference.reindex(features).fillna(0.0).to_numpy()
    cmp_vector = comparison.reindex(features).fillna(0.0).to_numpy()

    return ref_vector, cmp_vector


def rank_correlation(reference_df: pd.DataFrame, comparison_df: pd.DataFrame) -> float:
    ref_vector, cmp_vector = align_importance_vectors(reference_df, comparison_df)

    if len(ref_vector) < 2:
        return np.nan

    if np.all(ref_vector == ref_vector[0]) or np.all(cmp_vector == cmp_vector[0]):
        return np.nan

    corr, _ = spearmanr(ref_vector, cmp_vector)
    return float(corr)


def top_k_jaccard(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    k: int = TOP_N_FEATURES,
) -> float:
    ref_top = set(reference_df.head(k)["feature"])
    cmp_top = set(comparison_df.head(k)["feature"])

    union = ref_top.union(cmp_top)

    if not union:
        return np.nan

    return len(ref_top.intersection(cmp_top)) / len(union)


def mean_abs_importance_delta(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> float:
    ref_vector, cmp_vector = align_importance_vectors(reference_df, comparison_df)
    return float(np.mean(np.abs(ref_vector - cmp_vector)))


def make_stability_row(
    comparison_type: str,
    dimension: str,
    segment: str,
    model_name: str,
    segment_df: pd.DataFrame,
    y_segment: pd.Series,
    reference_model: str,
    reference_segment: str,
    reference_importance_df: pd.DataFrame,
    comparison_importance_df: pd.DataFrame,
    notes: str,
) -> Dict[str, object]:
    return {
        "comparison_type": comparison_type,
        "dimension": dimension,
        "segment": segment,
        "model": model_name,
        "n_rows": int(len(segment_df)),
        "n_positive": int(y_segment.sum()) if len(y_segment) > 0 else 0,
        "positive_rate": float(y_segment.mean()) if len(y_segment) > 0 else np.nan,
        "reference_model": reference_model,
        "reference_segment": reference_segment,
        "rank_correlation_with_reference": rank_correlation(
            reference_importance_df,
            comparison_importance_df,
        ),
        "top10_jaccard_with_reference": top_k_jaccard(
            reference_importance_df,
            comparison_importance_df,
            k=TOP_N_FEATURES,
        ),
        "mean_abs_importance_delta": mean_abs_importance_delta(
            reference_importance_df,
            comparison_importance_df,
        ),
        "top_features": top_features_string(comparison_importance_df),
        "notes": notes,
    }


# ---------------------------------------------------------------------
# Group segments
# ---------------------------------------------------------------------

def find_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    return None


def create_numeric_band(series: pd.Series, labels: List[str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() < MIN_SEGMENT_ROWS:
        return pd.Series(["Missing"] * len(series), index=series.index)

    try:
        bands = pd.qcut(
            numeric.rank(method="first"),
            q=len(labels),
            labels=labels,
            duplicates="drop",
        )
        return bands.astype(str).replace("nan", "Missing")
    except Exception:
        median_value = numeric.median()
        return np.where(numeric <= median_value, labels[0], labels[-1])


def build_group_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    groups: Dict[str, pd.Series] = {}

    for group_name, candidates in GROUP_COLUMN_CANDIDATES.items():
        col = find_existing_column(df, candidates)

        if col is not None:
            groups[group_name] = (
                df[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna("Missing")
                .astype(str)
            )
            continue

        if group_name in NUMERIC_BAND_CANDIDATES:
            numeric_col = find_existing_column(df, NUMERIC_BAND_CANDIDATES[group_name])

            if numeric_col is not None:
                groups[group_name] = pd.Series(
                    create_numeric_band(
                        df[numeric_col],
                        labels=["low", "medium", "high"],
                    ),
                    index=df.index,
                ).fillna("Missing").astype(str)

    return groups


def compute_group_level_stability(
    model,
    model_name: str,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    reference_importance_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    rows = []
    group_series_dict = build_group_series(df)

    if not group_series_dict:
        rows.append(
            {
                "comparison_type": "group_vs_overall",
                "dimension": "no_group_columns_found",
                "segment": "not_applicable",
                "model": model_name,
                "n_rows": int(len(X)),
                "n_positive": int(y.sum()),
                "positive_rate": float(y.mean()),
                "reference_model": model_name,
                "reference_segment": "overall",
                "rank_correlation_with_reference": np.nan,
                "top10_jaccard_with_reference": np.nan,
                "mean_abs_importance_delta": np.nan,
                "top_features": "",
                "notes": "No usable demographic or band columns were found for group-level explanation comparison.",
            }
        )
        return rows

    for dimension, group_values in group_series_dict.items():
        group_values = group_values.loc[X.index]
        value_counts = group_values.value_counts(dropna=False)

        for group_value, count in value_counts.items():
            if count < MIN_SEGMENT_ROWS:
                continue

            segment_index = group_values[group_values == group_value].index
            segment_index = segment_index.intersection(X.index)

            if len(segment_index) < MIN_SEGMENT_ROWS:
                continue

            X_segment = X.loc[segment_index]
            y_segment = y.loc[segment_index]

            try:
                segment_importance_df = compute_permutation_importance_df(
                    model=model,
                    X=X_segment,
                    y=y_segment,
                    model_name=model_name,
                    n_repeats=3,
                    max_rows=MAX_SEGMENT_IMPORTANCE_ROWS,
                )

                rows.append(
                    make_stability_row(
                        comparison_type="group_vs_overall",
                        dimension=dimension,
                        segment=str(group_value),
                        model_name=model_name,
                        segment_df=X_segment,
                        y_segment=y_segment,
                        reference_model=model_name,
                        reference_segment="overall",
                        reference_importance_df=reference_importance_df,
                        comparison_importance_df=segment_importance_df,
                        notes="Group-level explanation comparison.",
                    )
                )

            except Exception as exc:
                rows.append(
                    {
                        "comparison_type": "group_vs_overall",
                        "dimension": dimension,
                        "segment": str(group_value),
                        "model": model_name,
                        "n_rows": int(len(X_segment)),
                        "n_positive": int(y_segment.sum()),
                        "positive_rate": float(y_segment.mean()),
                        "reference_model": model_name,
                        "reference_segment": "overall",
                        "rank_correlation_with_reference": np.nan,
                        "top10_jaccard_with_reference": np.nan,
                        "mean_abs_importance_delta": np.nan,
                        "top_features": "",
                        "notes": f"Group comparison failed: {exc}",
                    }
                )

    return rows


# ---------------------------------------------------------------------
# Time-like splits
# ---------------------------------------------------------------------

def find_time_split_column(df: pd.DataFrame) -> Optional[str]:
    return find_existing_column(df, TIME_COLUMN_CANDIDATES)


def create_time_split_labels(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    time_col = find_time_split_column(df)

    if time_col is not None:
        candidate = df[time_col]
        parsed_dates = pd.to_datetime(candidate, errors="coerce")

        if parsed_dates.notna().sum() >= MIN_SEGMENT_ROWS and parsed_dates.nunique() >= 3:
            ranks = parsed_dates.rank(method="first")
            labels = pd.qcut(
                ranks,
                q=3,
                labels=["early", "middle", "late"],
                duplicates="drop",
            )
            return labels.astype(str), f"calendar/date split using '{time_col}'"

        numeric_candidate = pd.to_numeric(candidate, errors="coerce")

        if numeric_candidate.notna().sum() >= MIN_SEGMENT_ROWS and numeric_candidate.nunique() >= 3:
            ranks = numeric_candidate.rank(method="first")
            labels = pd.qcut(
                ranks,
                q=3,
                labels=["early", "middle", "late"],
                duplicates="drop",
            )
            return labels.astype(str), f"numeric time split using '{time_col}'"

    row_order = pd.Series(np.arange(len(df)), index=df.index)

    labels = pd.qcut(
        row_order.rank(method="first"),
        q=3,
        labels=["early_proxy", "middle_proxy", "late_proxy"],
        duplicates="drop",
    )

    return (
        labels.astype(str),
        "row-order proxy split because no usable multi-period time column was available",
    )


def compute_time_split_stability(
    model,
    model_name: str,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    reference_importance_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    rows = []

    split_labels, split_note = create_time_split_labels(df)
    split_labels = split_labels.loc[X.index]

    for split_value in sorted(split_labels.dropna().unique()):
        segment_index = split_labels[split_labels == split_value].index

        if len(segment_index) < MIN_SEGMENT_ROWS:
            continue

        X_segment = X.loc[segment_index]
        y_segment = y.loc[segment_index]

        try:
            segment_importance_df = compute_permutation_importance_df(
                model=model,
                X=X_segment,
                y=y_segment,
                model_name=model_name,
                n_repeats=3,
                max_rows=MAX_SEGMENT_IMPORTANCE_ROWS,
            )

            rows.append(
                make_stability_row(
                    comparison_type="time_split_vs_overall",
                    dimension="time_split",
                    segment=str(split_value),
                    model_name=model_name,
                    segment_df=X_segment,
                    y_segment=y_segment,
                    reference_model=model_name,
                    reference_segment="overall",
                    reference_importance_df=reference_importance_df,
                    comparison_importance_df=segment_importance_df,
                    notes=split_note,
                )
            )

        except Exception as exc:
            rows.append(
                {
                    "comparison_type": "time_split_vs_overall",
                    "dimension": "time_split",
                    "segment": str(split_value),
                    "model": model_name,
                    "n_rows": int(len(X_segment)),
                    "n_positive": int(y_segment.sum()),
                    "positive_rate": float(y_segment.mean()),
                    "reference_model": model_name,
                    "reference_segment": "overall",
                    "rank_correlation_with_reference": np.nan,
                    "top10_jaccard_with_reference": np.nan,
                    "mean_abs_importance_delta": np.nan,
                    "top_features": "",
                    "notes": f"Time-split comparison failed: {exc}",
                }
            )

    return rows


# ---------------------------------------------------------------------
# Champion versus challenger
# ---------------------------------------------------------------------

def compute_model_level_stability(
    champion_importance_df: pd.DataFrame,
    challenger_importance_df: Optional[pd.DataFrame],
    X: pd.DataFrame,
    y: pd.Series,
) -> List[Dict[str, object]]:
    if challenger_importance_df is None:
        return []

    row = make_stability_row(
        comparison_type="challenger_vs_champion",
        dimension="model",
        segment="overall",
        model_name="challenger",
        segment_df=X,
        y_segment=y,
        reference_model="champion",
        reference_segment="overall",
        reference_importance_df=champion_importance_df,
        comparison_importance_df=challenger_importance_df,
        notes="Compares challenger explanation ranking against champion explanation ranking.",
    )

    return [row]


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    print("Step K — Explainability and Stability")
    print("Loading dataset and saved models...")

    df = load_dataset()

    champion_model, champion_metadata = load_model(CHAMPION_MODEL_PATH, "Champion")
    challenger_model, challenger_metadata = load_model(CHALLENGER_MODEL_PATH, "Challenger")

    if champion_model is None:
        raise FileNotFoundError(
            f"Champion model is required but was not found:\n{CHAMPION_MODEL_PATH}\n\n"
            "Run the champion/challenger training step first."
        )

    X, y, target_col = prepare_features(
        df=df,
        model=champion_model,
        metadata=champion_metadata,
    )

    print(f"Dataset rows: {len(df):,}")
    print(f"Target column: {target_col}")
    print(f"Feature columns used for champion model: {X.shape[1]:,}")

    print("Computing champion permutation importance...")
    champion_importance_df = compute_permutation_importance_df(
        model=champion_model,
        X=X,
        y=y,
        model_name="champion",
        n_repeats=5,
        max_rows=MAX_FULL_IMPORTANCE_ROWS,
    )

    plot_feature_importance(
        importance_df=champion_importance_df,
        output_path=FEATURE_IMPORTANCE_PATH,
    )

    print("Creating SHAP summary or fallback explanation plot...")
    create_shap_summary_plot(
        model=champion_model,
        X=X,
        champion_importance_df=champion_importance_df,
        output_path=SHAP_SUMMARY_PATH,
    )

    stability_rows: List[Dict[str, object]] = []

    stability_rows.append(
        {
            "comparison_type": "reference",
            "dimension": "overall",
            "segment": "overall",
            "model": "champion",
            "n_rows": int(len(X)),
            "n_positive": int(y.sum()),
            "positive_rate": float(y.mean()),
            "reference_model": "champion",
            "reference_segment": "overall",
            "rank_correlation_with_reference": 1.0,
            "top10_jaccard_with_reference": 1.0,
            "mean_abs_importance_delta": 0.0,
            "top_features": top_features_string(champion_importance_df),
            "notes": "Overall champion-model explanation reference.",
        }
    )

    print("Computing group-level explanation stability...")
    stability_rows.extend(
        compute_group_level_stability(
            model=champion_model,
            model_name="champion",
            df=df,
            X=X,
            y=y,
            reference_importance_df=champion_importance_df,
        )
    )

    print("Computing time-split explanation stability...")
    stability_rows.extend(
        compute_time_split_stability(
            model=champion_model,
            model_name="champion",
            df=df,
            X=X,
            y=y,
            reference_importance_df=champion_importance_df,
        )
    )

    if challenger_model is not None:
        try:
            X_challenger, y_challenger, _ = prepare_features(
                df=df,
                model=challenger_model,
                metadata=challenger_metadata,
            )

            common_index = X.index.intersection(X_challenger.index)
            X_challenger = X_challenger.loc[common_index]
            y_challenger = y_challenger.loc[common_index]

            print("Computing challenger permutation importance...")
            challenger_importance_df = compute_permutation_importance_df(
                model=challenger_model,
                X=X_challenger,
                y=y_challenger,
                model_name="challenger",
                n_repeats=5,
                max_rows=MAX_FULL_IMPORTANCE_ROWS,
            )

            stability_rows.extend(
                compute_model_level_stability(
                    champion_importance_df=champion_importance_df,
                    challenger_importance_df=challenger_importance_df,
                    X=X_challenger,
                    y=y_challenger,
                )
            )

        except Exception as exc:
            stability_rows.append(
                {
                    "comparison_type": "challenger_vs_champion",
                    "dimension": "model",
                    "segment": "overall",
                    "model": "challenger",
                    "n_rows": int(len(X)),
                    "n_positive": int(y.sum()),
                    "positive_rate": float(y.mean()),
                    "reference_model": "champion",
                    "reference_segment": "overall",
                    "rank_correlation_with_reference": np.nan,
                    "top10_jaccard_with_reference": np.nan,
                    "mean_abs_importance_delta": np.nan,
                    "top_features": "",
                    "notes": f"Challenger comparison failed: {exc}",
                }
            )

    stability_df = pd.DataFrame(stability_rows)

    required_columns = [
        "comparison_type",
        "dimension",
        "segment",
        "model",
        "n_rows",
        "n_positive",
        "positive_rate",
        "reference_model",
        "reference_segment",
        "rank_correlation_with_reference",
        "top10_jaccard_with_reference",
        "mean_abs_importance_delta",
        "top_features",
        "notes",
    ]

    for col in required_columns:
        if col not in stability_df.columns:
            stability_df[col] = np.nan

    stability_df = stability_df[required_columns]
    stability_df.to_csv(STABILITY_TABLE_PATH, index=False)

    print()
    print("Step K complete.")
    print(f"Created: {FEATURE_IMPORTANCE_PATH}")
    print(f"Created: {SHAP_SUMMARY_PATH}")
    print(f"Created: {STABILITY_TABLE_PATH}")

    print()
    print("Interpretation guide:")
    print("- rank_correlation_with_reference near 1.0 means explanation rankings are stable.")
    print("- top10_jaccard_with_reference near 1.0 means top features overlap strongly.")
    print("- larger mean_abs_importance_delta means larger explanation drift.")
    print("- low stability across groups, time splits, or challenger models is a governance concern.")


if __name__ == "__main__":
    main()