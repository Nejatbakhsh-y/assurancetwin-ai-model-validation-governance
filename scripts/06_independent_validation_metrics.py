"""
Step H — Independent Validation Metrics

Script:
    scripts/06_independent_validation_metrics.py

Purpose:
    Simulate the work of an independent model-validation team by loading
    saved champion and challenger models and evaluating them on a held-out
    validation sample.

Inputs:
    data/processed/hmda_modeling_dataset.csv
    models/champion_model.pkl
    models/challenger_model.pkl

Outputs:
    reports/validation/model_validation_report.md
    reports/tables/independent_validation_metrics.csv

Metrics:
    AUC
    Accuracy
    Precision
    Recall
    F1
    Balanced Accuracy
    Brier Score
    Calibration Error
    Confusion Matrix
    Approval Rate
    False Positive Rate
    False Negative Rate
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path("data/processed/hmda_modeling_dataset.csv")

CHAMPION_MODEL_PATH = Path("models/champion_model.pkl")
CHALLENGER_MODEL_PATH = Path("models/challenger_model.pkl")

TABLES_DIR = Path("reports/tables")
VALIDATION_DIR = Path("reports/validation")

METRICS_OUTPUT_PATH = TABLES_DIR / "independent_validation_metrics.csv"
REPORT_OUTPUT_PATH = VALIDATION_DIR / "model_validation_report.md"

TARGET_COLUMN = "approved"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CALIBRATION_BINS = 10
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50


# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------

@dataclass
class LoadedModel:
    """Container for a fitted model plus optional governance metadata."""
    model: Any
    source_path: Path
    wrapper_type: str
    wrapper_keys: list
    feature_columns: Optional[list] = None
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD


# ---------------------------------------------------------------------
# Directory and data functions
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    """Create required output directories."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Load the clean HMDA modeling dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}\n"
            "Expected file from Step E: data/processed/hmda_modeling_dataset.csv"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in {DATA_PATH}.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' has fewer than two classes. "
            "Independent validation metrics cannot be computed."
        )

    return df


def split_validation_sample(df: pd.DataFrame):
    """
    Create the independent validation sample.

    This script does not retrain or tune the model.
    It only evaluates previously saved models.
    """
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    _, X_validation, _, y_validation = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_validation, y_validation


# ---------------------------------------------------------------------
# Model loading and extraction
# ---------------------------------------------------------------------

def is_predictive_model(obj: Any) -> bool:
    """Return True if the object behaves like a fitted predictive model."""
    return hasattr(obj, "predict") or hasattr(obj, "predict_proba") or hasattr(obj, "decision_function")


def extract_threshold_from_dict(saved_object: dict) -> float:
    """Extract a classification threshold from a saved dictionary if available."""
    threshold_keys = [
        "threshold",
        "classification_threshold",
        "decision_threshold",
        "optimal_threshold",
    ]

    for key in threshold_keys:
        if key in saved_object:
            try:
                threshold = float(saved_object[key])
                if 0.0 <= threshold <= 1.0:
                    return threshold
            except Exception:
                pass

    return DEFAULT_CLASSIFICATION_THRESHOLD


def extract_feature_columns_from_dict(saved_object: dict) -> Optional[list]:
    """Extract feature columns from a saved dictionary if available."""
    feature_keys = [
        "feature_columns",
        "features",
        "model_features",
        "input_features",
        "training_features",
        "X_columns",
        "x_columns",
    ]

    for key in feature_keys:
        if key in saved_object:
            value = saved_object[key]
            if isinstance(value, (list, tuple, pd.Index)):
                return list(value)

    return None


def recursively_extract_model(obj: Any, depth: int = 0, max_depth: int = 5):
    """
    Extract a fitted model from common saved-object formats.

    Supports:
        1. Direct sklearn estimator or pipeline.
        2. Dictionary wrapper containing model/pipeline/estimator.
        3. Nested dictionary wrapper.
    """
    if depth > max_depth:
        return None

    if is_predictive_model(obj):
        return obj

    if isinstance(obj, dict):
        priority_keys = [
            "model",
            "estimator",
            "pipeline",
            "fitted_model",
            "trained_model",
            "best_model",
            "model_object",
            "sklearn_model",
            "classifier",
            "clf",
        ]

        for key in priority_keys:
            if key in obj:
                candidate = recursively_extract_model(
                    obj[key],
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                if candidate is not None:
                    return candidate

        for value in obj.values():
            candidate = recursively_extract_model(
                value,
                depth=depth + 1,
                max_depth=max_depth,
            )
            if candidate is not None:
                return candidate

    return None


def load_model_bundle(model_path: Path) -> LoadedModel:
    """
    Load a saved model.

    This function fixes the common error:

        AttributeError: 'dict' object has no attribute 'predict'

    That error occurs when the .pkl file contains a dictionary wrapper
    instead of the raw sklearn model object.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}\n"
            "Expected this file from Step G."
        )

    saved_object = joblib.load(model_path)

    wrapper_type = type(saved_object).__name__
    wrapper_keys = []
    feature_columns = None
    threshold = DEFAULT_CLASSIFICATION_THRESHOLD

    if isinstance(saved_object, dict):
        wrapper_keys = list(saved_object.keys())
        feature_columns = extract_feature_columns_from_dict(saved_object)
        threshold = extract_threshold_from_dict(saved_object)

    model = recursively_extract_model(saved_object)

    if model is None:
        if isinstance(saved_object, dict):
            raise ValueError(
                f"The file {model_path} contains a dictionary, but no fitted "
                f"model was found inside it.\n"
                f"Available dictionary keys: {list(saved_object.keys())}"
            )

        raise TypeError(
            f"The file {model_path} does not contain a valid sklearn model, "
            f"pipeline, or supported dictionary wrapper.\n"
            f"Loaded object type: {type(saved_object)}"
        )

    return LoadedModel(
        model=model,
        source_path=model_path,
        wrapper_type=wrapper_type,
        wrapper_keys=wrapper_keys,
        feature_columns=feature_columns,
        threshold=threshold,
    )


# ---------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------

def get_expected_feature_names(model: Any) -> Optional[list]:
    """
    Recover expected feature names from a fitted estimator or pipeline.

    Many fitted sklearn pipelines expose feature_names_in_ after fitting
    on a pandas DataFrame.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def prepare_features_for_model(
    model_bundle: LoadedModel,
    X_validation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align validation features to the saved model.

    Priority:
        1. Feature columns stored in the model dictionary wrapper.
        2. feature_names_in_ from the fitted sklearn model.
        3. Use validation data as-is.
    """
    expected_features = model_bundle.feature_columns

    if expected_features is None:
        expected_features = get_expected_feature_names(model_bundle.model)

    if expected_features is None:
        return X_validation.copy()

    missing_features = [col for col in expected_features if col not in X_validation.columns]

    if missing_features:
        raise ValueError(
            f"The model loaded from {model_bundle.source_path} expects features "
            f"that are not present in the validation dataset:\n"
            f"{missing_features}"
        )

    return X_validation[expected_features].copy()


# ---------------------------------------------------------------------
# Prediction and metrics
# ---------------------------------------------------------------------

def get_positive_class_probability(model: Any, X_model: pd.DataFrame) -> np.ndarray:
    """
    Return probability-like scores for the positive class.

    Preferred:
        predict_proba[:, 1]

    Fallback:
        decision_function converted through logistic sigmoid.

    Last fallback:
        binary predictions treated as probability-like scores.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_model)
        proba = np.asarray(proba)

        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]

        if proba.ndim == 1:
            return proba

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_model)
        scores = np.asarray(scores, dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    predictions = model.predict(X_model)
    return np.asarray(predictions, dtype=float)


def expected_calibration_error(
    y_true: pd.Series,
    y_probability: np.ndarray,
    n_bins: int = N_CALIBRATION_BINS,
) -> float:
    """
    Compute Expected Calibration Error.

    ECE compares the average predicted probability with the observed event
    rate inside probability bins.

    Lower is better.
    """
    y_true_array = np.asarray(y_true).astype(int)
    y_probability = np.asarray(y_probability).astype(float)
    y_probability = np.clip(y_probability, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        if i == n_bins - 1:
            in_bin = (y_probability >= lower) & (y_probability <= upper)
        else:
            in_bin = (y_probability >= lower) & (y_probability < upper)

        bin_count = int(np.sum(in_bin))

        if bin_count == 0:
            continue

        bin_observed_rate = float(np.mean(y_true_array[in_bin]))
        bin_predicted_rate = float(np.mean(y_probability[in_bin]))
        bin_weight = bin_count / len(y_true_array)

        ece += bin_weight * abs(bin_observed_rate - bin_predicted_rate)

    return float(ece)


def safe_auc(y_true: pd.Series, y_probability: np.ndarray) -> float:
    """Compute AUC safely."""
    if pd.Series(y_true).nunique() < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_probability))


def safe_brier_score(y_true: pd.Series, y_probability: np.ndarray) -> float:
    """Compute Brier Score safely."""
    y_probability = np.clip(y_probability, 0.0, 1.0)
    return float(brier_score_loss(y_true, y_probability))


def evaluate_model(
    model_name: str,
    model_bundle: LoadedModel,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    """Evaluate one saved model and return independent validation metrics."""
    X_model = prepare_features_for_model(model_bundle, X_validation)

    y_probability = get_positive_class_probability(model_bundle.model, X_model)
    y_probability = np.asarray(y_probability, dtype=float)
    y_probability = np.clip(y_probability, 0.0, 1.0)

    threshold = model_bundle.threshold
    y_pred = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        y_pred,
        labels=[0, 1],
    ).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    metrics = {
        "model_name": model_name,
        "model_file": str(model_bundle.source_path),
        "saved_object_type": model_bundle.wrapper_type,
        "saved_object_keys": "; ".join(model_bundle.wrapper_keys),
        "classification_threshold": float(threshold),
        "validation_n": int(len(y_validation)),
        "actual_approval_rate": float(np.mean(y_validation)),
        "predicted_approval_rate": float(np.mean(y_pred)),
        "auc": safe_auc(y_validation, y_probability),
        "accuracy": float(accuracy_score(y_validation, y_pred)),
        "precision": float(precision_score(y_validation, y_pred, zero_division=0)),
        "recall": float(recall_score(y_validation, y_pred, zero_division=0)),
        "f1": float(f1_score(y_validation, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_validation, y_pred)),
        "brier_score": safe_brier_score(y_validation, y_probability),
        "calibration_error": expected_calibration_error(
            y_validation,
            y_probability,
            n_bins=N_CALIBRATION_BINS,
        ),
        "confusion_matrix_tn": int(tn),
        "confusion_matrix_fp": int(fp),
        "confusion_matrix_fn": int(fn),
        "confusion_matrix_tp": int(tp),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
    }

    return metrics


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------

def format_metric(value) -> str:
    """Format values for Markdown tables."""
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (int, np.integer)):
        return f"{value:,}"

    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"

    return str(value)


def identify_best_models(metrics_df: pd.DataFrame) -> dict:
    """Identify best model under several validation criteria."""
    result = {}

    higher_is_better = [
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]

    lower_is_better = [
        "brier_score",
        "calibration_error",
        "false_positive_rate",
        "false_negative_rate",
    ]

    for metric in higher_is_better:
        if metric in metrics_df.columns and metrics_df[metric].notna().any():
            idx = metrics_df[metric].idxmax()
            result[metric] = metrics_df.loc[idx, "model_name"]

    for metric in lower_is_better:
        if metric in metrics_df.columns and metrics_df[metric].notna().any():
            idx = metrics_df[metric].idxmin()
            result[metric] = metrics_df.loc[idx, "model_name"]

    return result


def create_markdown_report(metrics_df: pd.DataFrame) -> str:
    """Create the independent validation Markdown report."""
    best_models = identify_best_models(metrics_df)

    lines = []

    lines.append("# Independent Model Validation Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report simulates the work of an independent model-validation team. "
        "The validation process loads the saved champion and challenger models, "
        "evaluates them on a held-out validation sample, and compares predictive, "
        "classification, calibration, and decision-risk metrics."
    )
    lines.append("")
    lines.append(
        "The validation step does not retrain the models and does not tune "
        "hyperparameters. It is intended to represent independent post-development "
        "review rather than model-development activity."
    )
    lines.append("")

    lines.append("## Validation Design")
    lines.append("")
    lines.append(f"- Source dataset: `{DATA_PATH}`")
    lines.append(f"- Target variable: `{TARGET_COLUMN}`")
    lines.append(f"- Validation split: `{int(TEST_SIZE * 100)}%`")
    lines.append(f"- Random state: `{RANDOM_STATE}`")
    lines.append(f"- Calibration bins: `{N_CALIBRATION_BINS}`")
    lines.append("")

    lines.append("## Model Loading Review")
    lines.append("")
    lines.append("| Model | Model File | Saved Object Type | Saved Object Keys | Threshold |")
    lines.append("|---|---|---|---|---:|")

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['model_name']} "
            f"| `{row['model_file']}` "
            f"| {row['saved_object_type']} "
            f"| {row['saved_object_keys'] if row['saved_object_keys'] else 'N/A'} "
            f"| {format_metric(row['classification_threshold'])} |"
        )

    lines.append("")

    lines.append("## Validation Metrics Summary")
    lines.append("")
    lines.append(
        "| Model | AUC | Accuracy | Precision | Recall | F1 | Balanced Accuracy | "
        "Brier Score | Calibration Error | Approval Rate | FPR | FNR |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['model_name']} "
            f"| {format_metric(row['auc'])} "
            f"| {format_metric(row['accuracy'])} "
            f"| {format_metric(row['precision'])} "
            f"| {format_metric(row['recall'])} "
            f"| {format_metric(row['f1'])} "
            f"| {format_metric(row['balanced_accuracy'])} "
            f"| {format_metric(row['brier_score'])} "
            f"| {format_metric(row['calibration_error'])} "
            f"| {format_metric(row['predicted_approval_rate'])} "
            f"| {format_metric(row['false_positive_rate'])} "
            f"| {format_metric(row['false_negative_rate'])} |"
        )

    lines.append("")

    lines.append("## Confusion Matrices")
    lines.append("")

    for _, row in metrics_df.iterrows():
        lines.append(f"### {row['model_name']}")
        lines.append("")
        lines.append("| Actual / Predicted | Predicted 0 | Predicted 1 |")
        lines.append("|---|---:|---:|")
        lines.append(
            f"| Actual 0 | {int(row['confusion_matrix_tn']):,} "
            f"| {int(row['confusion_matrix_fp']):,} |"
        )
        lines.append(
            f"| Actual 1 | {int(row['confusion_matrix_fn']):,} "
            f"| {int(row['confusion_matrix_tp']):,} |"
        )
        lines.append("")

    lines.append("## Independent Validation Findings")
    lines.append("")

    if "auc" in best_models:
        lines.append(
            f"- Best discriminatory performance by AUC: "
            f"**{best_models['auc']}**."
        )

    if "balanced_accuracy" in best_models:
        lines.append(
            f"- Best class-balance performance by balanced accuracy: "
            f"**{best_models['balanced_accuracy']}**."
        )

    if "brier_score" in best_models:
        lines.append(
            f"- Best probability accuracy by Brier Score: "
            f"**{best_models['brier_score']}**."
        )

    if "calibration_error" in best_models:
        lines.append(
            f"- Best calibration by Expected Calibration Error: "
            f"**{best_models['calibration_error']}**."
        )

    if "false_positive_rate" in best_models:
        lines.append(
            f"- Lowest false-positive rate: "
            f"**{best_models['false_positive_rate']}**."
        )

    if "false_negative_rate" in best_models:
        lines.append(
            f"- Lowest false-negative rate: "
            f"**{best_models['false_negative_rate']}**."
        )

    lines.append("")

    lines.append("## Governance Interpretation")
    lines.append("")
    lines.append(
        "A model with the highest predictive performance is not automatically "
        "the best governed model. Independent validation must also consider "
        "calibration quality, approval-rate behavior, false-positive risk, "
        "false-negative risk, operational consequences, and documentation quality."
    )
    lines.append("")
    lines.append(
        "For model-risk governance, the preferred model should be selected using "
        "a documented trade-off among predictive strength, calibration, stability, "
        "interpretability, and business-use risk. This is especially important "
        "for credit, insurance, lending, and other regulated decision environments."
    )
    lines.append("")

    lines.append("## Validation Conclusion")
    lines.append("")
    lines.append(
        "This independent validation step provides evidence that model selection "
        "should not rely only on AUC or accuracy. The challenger model may be "
        "preferable from a governance perspective if it has stronger calibration, "
        "lower error asymmetry, lower approval-rate distortion, or better documented "
        "risk controls. Conversely, the champion model remains supportable only if "
        "its predictive gains justify its validation, monitoring, and governance risks."
    )
    lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Metrics table: `{METRICS_OUTPUT_PATH}`")
    lines.append(f"- Validation report: `{REPORT_OUTPUT_PATH}`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    """Run Step H independent validation."""
    ensure_directories()

    print("Step H — Independent Validation Metrics")
    print("--------------------------------------")

    df = load_dataset()
    print(f"Loaded dataset: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")

    X_validation, y_validation = split_validation_sample(df)
    print(f"Validation sample size: {len(y_validation):,}")
    print(f"Validation approval rate: {y_validation.mean():.4f}")

    champion_bundle = load_model_bundle(CHAMPION_MODEL_PATH)
    challenger_bundle = load_model_bundle(CHALLENGER_MODEL_PATH)

    models_to_validate = {
        "Champion Model": champion_bundle,
        "Challenger Model": challenger_bundle,
    }

    all_metrics = []

    for model_name, model_bundle in models_to_validate.items():
        print(f"Validating: {model_name}")
        print(f"  Loaded from: {model_bundle.source_path}")
        print(f"  Saved object type: {model_bundle.wrapper_type}")

        if model_bundle.wrapper_keys:
            print(f"  Saved object keys: {model_bundle.wrapper_keys}")

        model_metrics = evaluate_model(
            model_name=model_name,
            model_bundle=model_bundle,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        all_metrics.append(model_metrics)

    metrics_df = pd.DataFrame(all_metrics)

    preferred_column_order = [
        "model_name",
        "model_file",
        "saved_object_type",
        "saved_object_keys",
        "classification_threshold",
        "validation_n",
        "actual_approval_rate",
        "predicted_approval_rate",
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "brier_score",
        "calibration_error",
        "confusion_matrix_tn",
        "confusion_matrix_fp",
        "confusion_matrix_fn",
        "confusion_matrix_tp",
        "false_positive_rate",
        "false_negative_rate",
    ]

    metrics_df = metrics_df[preferred_column_order]

    metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
    print(f"Saved metrics table: {METRICS_OUTPUT_PATH}")

    report_text = create_markdown_report(metrics_df)
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Saved validation report: {REPORT_OUTPUT_PATH}")

    print("")
    print("Step H complete.")
    print("Generated:")
    print(f"  - {METRICS_OUTPUT_PATH}")
    print(f"  - {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main() 