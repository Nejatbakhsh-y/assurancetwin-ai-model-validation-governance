"""
Step J — Calibration Analysis

Purpose:
    Evaluate whether the saved HMDA approval models produce reliable probability
    estimates, not only strong ranking performance.

Inputs:
    data/processed/hmda_modeling_dataset.csv
    models/champion_model.pkl
    models/challenger_model.pkl

Outputs:
    reports/tables/calibration_summary.csv
    reports/figures/calibration_curve.png

Why this matters:
    In financial decision models, probability quality matters. A model with strong
    AUC can still produce poor probability estimates, which can distort approval
    thresholds, risk tiering, monitoring, pricing, and governance decisions.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hmda_modeling_dataset.csv"

MODEL_PATHS = {
    "Champion Model": PROJECT_ROOT / "models" / "champion_model.pkl",
    "Challenger Model": PROJECT_ROOT / "models" / "challenger_model.pkl",
}

TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

SUMMARY_PATH = TABLE_DIR / "calibration_summary.csv"
FIGURE_PATH = FIGURE_DIR / "calibration_curve.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BINS = 10


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    """Create output directories if needed."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the modeling dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {path}\n"
            "Run Step E first to create data/processed/hmda_modeling_dataset.csv."
        )

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"The modeling dataset is empty: {path}")

    return df


def load_model_artifact(path: Path) -> Optional[Any]:
    """Load a saved model artifact if it exists."""
    if not path.exists():
        print(f"WARNING: Model file not found and will be skipped: {path}")
        return None

    return joblib.load(path)


def extract_estimator_and_metadata(artifact: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Extract estimator and metadata from a saved artifact.

    Supports either:
        1. A raw sklearn estimator or pipeline
        2. A dictionary artifact containing model/estimator plus metadata
    """
    metadata: Dict[str, Any] = {}

    if isinstance(artifact, dict):
        metadata = artifact.copy()

        possible_model_keys = [
            "model",
            "estimator",
            "pipeline",
            "best_model",
            "fitted_model",
            "champion_model",
            "challenger_model",
        ]

        for key in possible_model_keys:
            if key in artifact:
                return artifact[key], metadata

        raise ValueError(
            "The loaded artifact is a dictionary, but no estimator key was found. "
            "Expected one of: model, estimator, pipeline, best_model, fitted_model."
        )

    return artifact, metadata


def find_target_column(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """Find the binary target column."""
    metadata_target_keys = [
        "target_column",
        "target_col",
        "target",
        "y_column",
        "label_column",
    ]

    for key in metadata_target_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value in df.columns:
            return value

    preferred_targets = [
        "approved",
        "approval",
        "loan_approved",
        "target",
        "y",
    ]

    for col in preferred_targets:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find the target column. Expected a column such as "
        "'approved', 'approval', 'loan_approved', or 'target'."
    )


def coerce_binary_target(y: pd.Series) -> pd.Series:
    """Convert the target into a clean binary 0/1 series."""
    y_clean = y.copy()

    if pd.api.types.is_numeric_dtype(y_clean):
        y_clean = pd.to_numeric(y_clean, errors="coerce")
        unique_values = sorted(y_clean.dropna().unique().tolist())

        if set(unique_values).issubset({0, 1, 0.0, 1.0}):
            return y_clean.astype(int)

    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "approved": 1,
        "approval": 1,
        "originated": 1,
        "denied": 0,
        "not approved": 0,
        "not_approved": 0,
    }

    y_lower = y_clean.astype(str).str.strip().str.lower()
    y_mapped = y_lower.map(mapping)

    if y_mapped.isna().any():
        bad_values = sorted(y_lower[y_mapped.isna()].dropna().unique().tolist())[:20]
        raise ValueError(
            "Target column could not be converted to binary 0/1 values. "
            f"Problematic values include: {bad_values}"
        )

    return y_mapped.astype(int)


def infer_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    estimator: Any,
    metadata: Dict[str, Any],
) -> List[str]:
    """
    Infer the feature columns used by the model.

    Priority:
        1. Feature columns saved in artifact metadata
        2. sklearn feature_names_in_
        3. fallback: all columns except target and obvious leakage/ID/split fields
    """
    possible_feature_keys = [
        "feature_columns",
        "features",
        "x_columns",
        "X_columns",
        "model_features",
        "predictor_columns",
    ]

    for key in possible_feature_keys:
        value = metadata.get(key)
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            missing = [col for col in value if col not in df.columns]
            if not missing:
                return value

    if hasattr(estimator, "feature_names_in_"):
        value = list(estimator.feature_names_in_)
        missing = [col for col in value if col not in df.columns]
        if not missing:
            return value

    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                value = list(step.feature_names_in_)
                missing = [col for col in value if col not in df.columns]
                if not missing:
                    return value

    leakage_or_nonfeature_cols = {
        target_col,
        "approved",
        "approval",
        "loan_approved",
        "target",
        "y",
        "action_taken",
        "action_taken_name",
        "respondent_id",
        "lei",
        "loan_sequence_number",
        "application_id",
        "id",
        "split",
        "data_split",
        "sample_split",
    }

    feature_cols = [col for col in df.columns if col not in leakage_or_nonfeature_cols]

    if not feature_cols:
        raise ValueError("No feature columns could be inferred.")

    return feature_cols


def make_validation_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a validation sample.

    If a split column exists, use rows labeled test/validation/valid.
    Otherwise, reproduce a deterministic 80/20 stratified split.
    """
    y = coerce_binary_target(df[target_col])

    valid_mask = y.notna()
    df = df.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    split_cols = ["split", "data_split", "sample_split"]

    for split_col in split_cols:
        if split_col in df.columns:
            split_values = df[split_col].astype(str).str.lower().str.strip()
            test_mask = split_values.isin(
                ["test", "validation", "valid", "val", "holdout", "hold_out"]
            )

            if test_mask.sum() >= 25 and y.loc[test_mask].nunique() == 2:
                X_valid = df.loc[test_mask, feature_cols].copy()
                y_valid = y.loc[test_mask].copy()
                return X_valid, y_valid

    X = df[feature_cols].copy()

    if y.nunique() != 2:
        raise ValueError("The target must contain both classes 0 and 1.")

    _, X_valid, _, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_valid, y_valid


def get_positive_class_index(classes: Any) -> int:
    """Find the index of the positive class in predict_proba output."""
    if classes is None:
        return 1

    classes_list = list(classes)

    positive_labels = [1, "1", True, "true", "True", "yes", "Yes", "approved", "Approved"]

    for label in positive_labels:
        if label in classes_list:
            return classes_list.index(label)

    if len(classes_list) == 2:
        return 1

    raise ValueError(f"Could not identify the positive class from classes_: {classes_list}")


def predict_positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Return predicted probability for the positive class."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = getattr(estimator, "classes_", None)
        pos_idx = get_positive_class_index(classes)

        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba did not return a two-column probability matrix.")

        return np.asarray(proba[:, pos_idx], dtype=float)

    if hasattr(estimator, "decision_function"):
        warnings.warn(
            "Estimator does not expose predict_proba. Using sigmoid-transformed "
            "decision_function scores. This is not a true calibrated probability."
        )
        scores = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    raise ValueError(
        "Estimator does not support predict_proba or decision_function. "
        "Calibration analysis requires probability scores."
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_BINS,
) -> Tuple[float, float]:
    """
    Compute equal-width Expected Calibration Error and Maximum Calibration Error.

    ECE:
        Weighted average absolute difference between observed and predicted
        probabilities across bins.

    MCE:
        Maximum absolute bin-level calibration gap.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        count = int(mask.sum())

        if count == 0:
            continue

        observed_rate = float(y_true[mask].mean())
        mean_predicted = float(y_prob[mask].mean())
        gap = abs(observed_rate - mean_predicted)

        ece += (count / n) * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


def calibration_intercept_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate calibration intercept and slope.

    Ideal values:
        intercept = 0
        slope = 1
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    eps = 1e-6
    p = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)

    try:
        try:
            lr = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=1000,
            )
        except TypeError:
            lr = LogisticRegression(
                penalty="none",
                solver="lbfgs",
                max_iter=1000,
            )

        lr.fit(logit_p, y_true)

        intercept = float(lr.intercept_[0])
        slope = float(lr.coef_[0][0])

        return intercept, slope

    except Exception:
        return np.nan, np.nan


def summarize_calibration(
    model_name: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Create one model-level calibration summary row."""
    y_array = np.asarray(y_true).astype(int)
    p_array = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)

    ece, mce = expected_calibration_error(y_array, p_array, n_bins=N_BINS)
    intercept, slope = calibration_intercept_slope(y_array, p_array)

    try:
        auc = roc_auc_score(y_array, p_array)
    except Exception:
        auc = np.nan

    try:
        avg_precision = average_precision_score(y_array, p_array)
    except Exception:
        avg_precision = np.nan

    try:
        ll = log_loss(y_array, p_array, labels=[0, 1])
    except Exception:
        ll = np.nan

    row = {
        "model_name": model_name,
        "n_observations": int(len(y_array)),
        "observed_approval_rate": float(np.mean(y_array)),
        "mean_predicted_probability": float(np.mean(p_array)),
        "probability_min": float(np.min(p_array)),
        "probability_25pct": float(np.quantile(p_array, 0.25)),
        "probability_median": float(np.quantile(p_array, 0.50)),
        "probability_75pct": float(np.quantile(p_array, 0.75)),
        "probability_max": float(np.max(p_array)),
        "brier_score": float(brier_score_loss(y_array, p_array)),
        "log_loss": float(ll),
        "auc": float(auc),
        "average_precision": float(avg_precision),
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "calibration_intercept": float(intercept) if not pd.isna(intercept) else np.nan,
        "calibration_slope": float(slope) if not pd.isna(slope) else np.nan,
        "ideal_brier_score_direction": "lower_is_better",
        "ideal_ece_direction": "lower_is_better",
        "ideal_intercept": 0.0,
        "ideal_slope": 1.0,
    }

    return row


def get_curve_points(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int = N_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return calibration curve points."""
    y_array = np.asarray(y_true).astype(int)
    p_array = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_array,
        p_array,
        n_bins=n_bins,
        strategy="uniform",
    )

    return mean_predicted_value, fraction_of_positives


def plot_calibration_curves(
    curve_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Save calibration curve figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    markers = ["o", "s", "^", "D", "x"]

    for i, (model_name, (mean_pred, frac_pos)) in enumerate(curve_data.items()):
        ax.plot(
            mean_pred,
            frac_pos,
            marker=markers[i % len(markers)],
            linewidth=1.8,
            label=model_name,
        )

    ax.set_title("Calibration Curve: Predicted Approval Probability vs Observed Approval Rate")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed approval rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    ensure_directories()

    print("=" * 80)
    print("Step J — Calibration Analysis")
    print("=" * 80)

    df = load_dataset(DATA_PATH)
    print(f"Loaded modeling dataset: {DATA_PATH}")
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    summary_rows: List[Dict[str, Any]] = []
    curve_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    loaded_any_model = False

    for model_name, model_path in MODEL_PATHS.items():
        artifact = load_model_artifact(model_path)

        if artifact is None:
            continue

        loaded_any_model = True

        estimator, metadata = extract_estimator_and_metadata(artifact)

        target_col = find_target_column(df, metadata)
        feature_cols = infer_feature_columns(df, target_col, estimator, metadata)

        X_valid, y_valid = make_validation_split(df, feature_cols, target_col)

        print("-" * 80)
        print(f"Evaluating: {model_name}")
        print(f"Model path: {model_path}")
        print(f"Target column: {target_col}")
        print(f"Validation rows: {len(y_valid):,}")
        print(f"Feature columns used: {len(feature_cols):,}")

        try:
            y_prob = predict_positive_probability(estimator, X_valid)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate probabilities for {model_name}.\n"
                f"Reason: {exc}\n\n"
                "Most likely cause: the saved model expects a different feature set "
                "than the current modeling dataset. Re-run Step G before Step J."
            ) from exc

        if len(y_prob) != len(y_valid):
            raise ValueError(
                f"Probability length mismatch for {model_name}: "
                f"{len(y_prob)} probabilities vs {len(y_valid)} labels."
            )

        row = summarize_calibration(model_name, y_valid, y_prob)
        summary_rows.append(row)

        curve_data[model_name] = get_curve_points(y_valid, y_prob, n_bins=N_BINS)

        print(f"Brier score: {row['brier_score']:.6f}")
        print(f"Expected calibration error: {row['expected_calibration_error']:.6f}")
        print(f"Calibration intercept: {row['calibration_intercept']:.6f}")
        print(f"Calibration slope: {row['calibration_slope']:.6f}")

    if not loaded_any_model:
        raise FileNotFoundError(
            "No saved model files were found. Expected at least one of:\n"
            f"  {MODEL_PATHS['Champion Model']}\n"
            f"  {MODEL_PATHS['Challenger Model']}\n"
            "Run Step G first to train and save the champion/challenger models."
        )

    if not summary_rows:
        raise RuntimeError("No calibration results were generated.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    plot_calibration_curves(curve_data, FIGURE_PATH)

    print("=" * 80)
    print("Calibration analysis complete.")
    print(f"Saved summary table: {SUMMARY_PATH}")
    print(f"Saved calibration curve: {FIGURE_PATH}")
    print("=" * 80)

    print("\nInterpretation guide:")
    print("  - Lower Brier score is better.")
    print("  - Lower expected calibration error is better.")
    print("  - Calibration intercept close to 0 is better.")
    print("  - Calibration slope close to 1 is better.")
    print("  - A curve close to the diagonal line indicates better probability calibration.")


if __name__ == "__main__":
    main()