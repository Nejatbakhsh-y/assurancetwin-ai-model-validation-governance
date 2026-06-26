"""
Step G — Train Champion and Challenger Models

Purpose:
    Train multiple candidate models for HMDA approval prediction and select:

    1. Champion model:
       Best governance-aware model, not necessarily the highest raw AUC model.

    2. Challenger model:
       Strong alternative model, usually the strongest predictive model or the
       next-best governed model.

Inputs:
    data/processed/hmda_modeling_dataset.csv

Outputs:
    reports/tables/model_performance_summary.csv
    reports/figures/roc_curves.png
    reports/figures/precision_recall_curves.png
    models/champion_model.pkl
    models/challenger_model.pkl
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TARGET_COL = "approved"

INPUT_PATH = Path("data/processed/hmda_modeling_dataset.csv")

TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")
MODEL_DIR = Path("models")

PERFORMANCE_PATH = TABLE_DIR / "model_performance_summary.csv"
ROC_FIGURE_PATH = FIGURE_DIR / "roc_curves.png"
PR_FIGURE_PATH = FIGURE_DIR / "precision_recall_curves.png"
CHAMPION_MODEL_PATH = MODEL_DIR / "champion_model.pkl"
CHALLENGER_MODEL_PATH = MODEL_DIR / "challenger_model.pkl"

# Laptop-safe cap. If the clean HMDA file is larger, the script takes a
# stratified reproducible sample for model training.
MAX_TRAINING_ROWS = 150_000

# Direct protected attributes are excluded from model features.
# They can still be used later for fairness diagnostics if retained in the dataset.
PROTECTED_ATTRIBUTE_TERMS = [
    "race",
    "ethnicity",
    "sex",
    "gender",
    "age",
]

# Leakage / post-outcome columns that must not be used as predictors.
LEAKAGE_TERMS = [
    "action_taken",
    "denial",
    "denial_reason",
    "approved",
    "origination",
]

# Identifier-like columns that are not appropriate predictive features.
IDENTIFIER_TERMS = [
    "lei",
    "uli",
    "universal_loan_identifier",
    "loan_sequence_number",
    "respondent_id",
    "application_id",
]

HIGH_MISSING_THRESHOLD = 0.98
HIGH_CARDINALITY_THRESHOLD = 200


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def standardize_target(y: pd.Series) -> pd.Series:
    """
    Convert the approved target into clean binary 0/1 values.
    """
    if pd.api.types.is_numeric_dtype(y):
        y_clean = y.fillna(0).astype(int)
        return y_clean

    positive_values = {
        "1",
        "true",
        "yes",
        "y",
        "approved",
        "approve",
        "originated",
        "loan originated",
    }

    y_clean = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .map(lambda value: 1 if value in positive_values else 0)
        .astype(int)
    )

    return y_clean


def stratified_sample_if_needed(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Keep training manageable on a local laptop while preserving target balance.
    """
    if len(df) <= MAX_TRAINING_ROWS:
        return df.copy()

    sampled_parts = []
    frac = MAX_TRAINING_ROWS / len(df)

    for _, group in df.groupby(target_col):
        n_group = max(1, int(round(len(group) * frac)))
        sampled_parts.append(group.sample(n=n_group, random_state=RANDOM_STATE))

    sampled_df = pd.concat(sampled_parts, axis=0)
    sampled_df = sampled_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return sampled_df


def should_exclude_column(col: str) -> bool:
    """
    Governance-aware feature exclusion.
    """
    col_lower = col.lower()

    if col_lower == TARGET_COL.lower():
        return True

    for term in LEAKAGE_TERMS + IDENTIFIER_TERMS + PROTECTED_ATTRIBUTE_TERMS:
        if term in col_lower:
            return True

    return False


def coerce_mostly_numeric_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns to numeric when most non-missing values are numeric.
    """
    X_out = X.copy()

    for col in X_out.columns:
        if X_out[col].dtype == "object":
            candidate = (
                X_out[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            converted = pd.to_numeric(candidate, errors="coerce")

            non_missing_original = X_out[col].notna().sum()
            non_missing_converted = converted.notna().sum()

            if non_missing_original > 0:
                numeric_ratio = non_missing_converted / non_missing_original
                if numeric_ratio >= 0.85:
                    X_out[col] = converted

    return X_out


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Remove leakage, direct protected attributes, mostly missing columns,
    and high-cardinality categorical columns.
    """
    candidate_cols = [col for col in df.columns if not should_exclude_column(col)]

    X = df[candidate_cols].copy()
    X = coerce_mostly_numeric_columns(X)

    # Drop all-empty or near-empty columns.
    missing_rate = X.isna().mean()
    high_missing_cols = missing_rate[missing_rate >= HIGH_MISSING_THRESHOLD].index.tolist()
    X = X.drop(columns=high_missing_cols, errors="ignore")

    # Identify categorical columns.
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Drop high-cardinality categorical columns to reduce overfitting and improve governance.
    high_cardinality_cols = [
        col for col in categorical_cols
        if X[col].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
    ]
    X = X.drop(columns=high_cardinality_cols, errors="ignore")

    # Recompute column types after exclusions.
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].astype(object)

    removed_cols = sorted(
        set(df.columns)
        - set(X.columns)
        - {TARGET_COL}
    )

    return X, numeric_cols, categorical_cols, removed_cols


def make_one_hot_encoder():
    """
    Compatible OneHotEncoder across scikit-learn versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            min_frequency=0.01,
        )
    except TypeError:
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
                min_frequency=0.01,
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
            )


def build_linear_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable features remain after governance exclusions.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_tree_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable features remain after governance exclusions.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Lower is better.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0

    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

    return float(ece)


def make_calibrated_logistic_model(
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    base_lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_linear_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    try:
        return CalibratedClassifierCV(
            estimator=base_lr_pipeline,
            method="sigmoid",
            cv=3,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=base_lr_pipeline,
            method="sigmoid",
            cv=3,
        )


def build_model_specs(
    numeric_cols: list[str],
    categorical_cols: list[str],
    y_train: pd.Series,
):
    linear_preprocessor = build_linear_preprocessor(numeric_cols, categorical_cols)
    tree_preprocessor = build_tree_preprocessor(numeric_cols, categorical_cols)

    model_specs = []

    model_specs.append(
        (
            "Logistic Regression",
            Pipeline(
                steps=[
                    ("preprocessor", clone(linear_preprocessor)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "High interpretability; strong baseline; easier validation and documentation.",
            1.00,
            0.00,
        )
    )

    model_specs.append(
        (
            "Random Forest",
            Pipeline(
                steps=[
                    ("preprocessor", clone(tree_preprocessor)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            min_samples_leaf=50,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Nonlinear benchmark; stronger predictive capacity but lower transparency.",
            0.65,
            0.04,
        )
    )

    model_specs.append(
        (
            "Gradient Boosting",
            Pipeline(
                steps=[
                    ("preprocessor", clone(tree_preprocessor)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=250,
                            learning_rate=0.05,
                            max_leaf_nodes=31,
                            l2_regularization=0.10,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Strong nonlinear challenger; useful for accuracy comparison.",
            0.60,
            0.05,
        )
    )

    # Optional XGBoost or LightGBM model.
    # The script works even if neither package is installed.
    xgb_or_lgbm_added = False

    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    scale_pos_weight = n_negative / max(n_positive, 1)

    try:
        from xgboost import XGBClassifier

        model_specs.append(
            (
                "XGBoost",
                Pipeline(
                    steps=[
                        ("preprocessor", clone(tree_preprocessor)),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=300,
                                max_depth=4,
                                learning_rate=0.05,
                                subsample=0.90,
                                colsample_bytree=0.90,
                                objective="binary:logistic",
                                eval_metric="logloss",
                                tree_method="hist",
                                scale_pos_weight=scale_pos_weight,
                                n_jobs=-1,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
                "Advanced boosting challenger; high predictive capacity; higher validation burden.",
                0.50,
                0.07,
            )
        )
        xgb_or_lgbm_added = True

    except ImportError:
        try:
            from lightgbm import LGBMClassifier

            model_specs.append(
                (
                    "LightGBM",
                    Pipeline(
                        steps=[
                            ("preprocessor", clone(tree_preprocessor)),
                            (
                                "model",
                                LGBMClassifier(
                                    n_estimators=300,
                                    max_depth=-1,
                                    learning_rate=0.05,
                                    subsample=0.90,
                                    colsample_bytree=0.90,
                                    class_weight="balanced",
                                    n_jobs=-1,
                                    random_state=RANDOM_STATE,
                                    verbose=-1,
                                ),
                            ),
                        ]
                    ),
                    "Advanced boosting challenger; high predictive capacity; higher validation burden.",
                    0.50,
                    0.07,
                )
            )
            xgb_or_lgbm_added = True

        except ImportError:
            pass

    if not xgb_or_lgbm_added:
        print(
            "Optional model note: XGBoost/LightGBM is not installed. "
            "The script will continue without it. To include one, run: pip install xgboost"
        )

    model_specs.append(
        (
            "Calibrated Logistic Regression",
            make_calibrated_logistic_model(numeric_cols, categorical_cols),
            "Calibrated transparent model; strong governance candidate when probability quality matters.",
            0.95,
            0.01,
        )
    )

    return model_specs


def evaluate_model(name: str, model, X_test, y_test) -> tuple[dict, dict]:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

    metrics = {
        "model_name": name,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob, labels=[0, 1]),
        "expected_calibration_error": expected_calibration_error(y_test, y_prob),
    }

    curves = {
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }

    return metrics, curves


def calculate_governance_score(row: pd.Series) -> float:
    """
    Governance-aware score.

    The goal is not to reward raw discrimination only.
    This score balances:
        - ROC AUC
        - average precision
        - Brier score
        - calibration error
        - interpretability
        - model complexity penalty

    Higher is better.
    """
    score = (
        0.35 * row["roc_auc"]
        + 0.20 * row["average_precision"]
        + 0.20 * (1.00 - row["brier_score"])
        + 0.15 * (1.00 - row["expected_calibration_error"])
        + 0.10 * row["interpretability_score"]
        - row["complexity_penalty"]
    )

    return float(score)


def plot_roc_curves(curves_by_model: dict, performance_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 7))

    for model_name, curves in curves_by_model.items():
        auc_value = performance_df.loc[
            performance_df["model_name"] == model_name,
            "roc_auc",
        ].iloc[0]

        plt.plot(
            curves["fpr"],
            curves["tpr"],
            linewidth=2,
            label=f"{model_name} | AUC={auc_value:.3f}",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="No-skill baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Champion and Challenger Candidate Models")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(ROC_FIGURE_PATH, dpi=200)
    plt.close()


def plot_precision_recall_curves(
    curves_by_model: dict,
    performance_df: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    prevalence = float(np.mean(y_test))

    plt.figure(figsize=(9, 7))

    for model_name, curves in curves_by_model.items():
        ap_value = performance_df.loc[
            performance_df["model_name"] == model_name,
            "average_precision",
        ].iloc[0]

        plt.plot(
            curves["recall_curve"],
            curves["precision_curve"],
            linewidth=2,
            label=f"{model_name} | AP={ap_value:.3f}",
        )

    plt.axhline(
        y=prevalence,
        linestyle="--",
        linewidth=1,
        label=f"No-skill baseline | prevalence={prevalence:.3f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves: Champion and Challenger Candidate Models")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(PR_FIGURE_PATH, dpi=200)
    plt.close()


def main() -> None:
    ensure_directories()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. "
            "Run Step E first to create data/processed/hmda_modeling_dataset.csv."
        )

    print("Loading modeling dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            "Expected Step E to create a binary approved target."
        )

    df = df.copy()
    df[TARGET_COL] = standardize_target(df[TARGET_COL])

    target_counts = df[TARGET_COL].value_counts(dropna=False).to_dict()
    print(f"Original dataset shape: {df.shape}")
    print(f"Target distribution: {target_counts}")

    if df[TARGET_COL].nunique() != 2:
        raise ValueError(
            f"Target column '{TARGET_COL}' must contain both 0 and 1 classes."
        )

    df = stratified_sample_if_needed(df, TARGET_COL)
    print(f"Training dataset shape after optional sampling: {df.shape}")

    X, numeric_cols, categorical_cols, removed_cols = prepare_features(df)
    y = df[TARGET_COL].copy()

    print(f"Usable numeric features: {len(numeric_cols)}")
    print(f"Usable categorical features: {len(categorical_cols)}")
    print(f"Removed governance/leakage/high-risk columns: {len(removed_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    model_specs = build_model_specs(numeric_cols, categorical_cols, y_train)

    fitted_models = {}
    curves_by_model = {}
    performance_rows = []

    print("\nTraining candidate models...")

    for (
        model_name,
        model,
        governance_rationale,
        interpretability_score,
        complexity_penalty,
    ) in model_specs:

        print(f"Training: {model_name}")

        try:
            model.fit(X_train, y_train)

            metrics, curves = evaluate_model(model_name, model, X_test, y_test)

            metrics["interpretability_score"] = interpretability_score
            metrics["complexity_penalty"] = complexity_penalty
            metrics["governance_rationale"] = governance_rationale
            metrics["training_status"] = "trained"

            performance_rows.append(metrics)
            curves_by_model[model_name] = curves
            fitted_models[model_name] = model

        except Exception as exc:
            performance_rows.append(
                {
                    "model_name": model_name,
                    "training_status": "failed",
                    "failure_reason": str(exc),
                    "governance_rationale": governance_rationale,
                    "interpretability_score": interpretability_score,
                    "complexity_penalty": complexity_penalty,
                }
            )
            print(f"Failed: {model_name}. Reason: {exc}")

    performance_df = pd.DataFrame(performance_rows)

    trained_mask = performance_df["training_status"] == "trained"
    trained_df = performance_df.loc[trained_mask].copy()

    if trained_df.empty:
        raise RuntimeError("No models trained successfully.")

    trained_df["governance_score"] = trained_df.apply(calculate_governance_score, axis=1)

    trained_df["predictive_rank_by_auc"] = (
        trained_df["roc_auc"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    trained_df["governance_rank"] = (
        trained_df["governance_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    champion_name = (
        trained_df.sort_values(
            by=["governance_score", "roc_auc", "expected_calibration_error"],
            ascending=[False, False, True],
        )
        .iloc[0]["model_name"]
    )

    best_predictive_name = (
        trained_df.sort_values(
            by=["roc_auc", "average_precision"],
            ascending=[False, False],
        )
        .iloc[0]["model_name"]
    )

    if best_predictive_name != champion_name:
        challenger_name = best_predictive_name
    else:
        remaining = trained_df[trained_df["model_name"] != champion_name].copy()

        if remaining.empty:
            challenger_name = champion_name
        else:
            challenger_name = (
                remaining.sort_values(
                    by=["roc_auc", "average_precision"],
                    ascending=[False, False],
                )
                .iloc[0]["model_name"]
            )

    trained_df["selected_role"] = "candidate"
    trained_df.loc[trained_df["model_name"] == champion_name, "selected_role"] = (
        "champion_governance_selected"
    )
    trained_df.loc[trained_df["model_name"] == challenger_name, "selected_role"] = (
        "challenger_predictive_or_alternative"
    )

    # Merge trained rows back with any failed rows.
    failed_df = performance_df.loc[~trained_mask].copy()

    final_performance_df = pd.concat([trained_df, failed_df], axis=0, ignore_index=True)

    sort_cols = ["training_status", "governance_rank", "predictive_rank_by_auc"]
    existing_sort_cols = [col for col in sort_cols if col in final_performance_df.columns]

    final_performance_df = final_performance_df.sort_values(
        by=existing_sort_cols,
        ascending=[False, True, True][: len(existing_sort_cols)],
    )

    numeric_metric_cols = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "interpretability_score",
        "complexity_penalty",
        "governance_score",
    ]

    for col in numeric_metric_cols:
        if col in final_performance_df.columns:
            final_performance_df[col] = final_performance_df[col].round(6)

    final_performance_df.to_csv(PERFORMANCE_PATH, index=False)

    plot_roc_curves(curves_by_model, trained_df)
    plot_precision_recall_curves(curves_by_model, trained_df, y_test)

    champion_bundle = {
        "model": fitted_models[champion_name],
        "model_name": champion_name,
        "selection_role": "champion_governance_selected",
        "selection_logic": (
            "Selected using governance-aware score, balancing predictive performance, "
            "calibration, interpretability, and model complexity."
        ),
        "target_column": TARGET_COL,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "removed_columns": removed_cols,
        "performance_summary": trained_df[
            trained_df["model_name"] == champion_name
        ].to_dict(orient="records")[0],
    }

    challenger_bundle = {
        "model": fitted_models[challenger_name],
        "model_name": challenger_name,
        "selection_role": "challenger_predictive_or_alternative",
        "selection_logic": (
            "Selected as the strongest predictive alternative or next-best candidate "
            "for challenger comparison."
        ),
        "target_column": TARGET_COL,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "removed_columns": removed_cols,
        "performance_summary": trained_df[
            trained_df["model_name"] == challenger_name
        ].to_dict(orient="records")[0],
    }

    joblib.dump(champion_bundle, CHAMPION_MODEL_PATH)
    joblib.dump(challenger_bundle, CHALLENGER_MODEL_PATH)

    print("\nStep G complete.")
    print(f"Performance summary saved to: {PERFORMANCE_PATH}")
    print(f"ROC curves saved to: {ROC_FIGURE_PATH}")
    print(f"Precision-recall curves saved to: {PR_FIGURE_PATH}")
    print(f"Champion model saved to: {CHAMPION_MODEL_PATH}")
    print(f"Challenger model saved to: {CHALLENGER_MODEL_PATH}")

    print("\nModel selection result:")
    print(f"Champion model: {champion_name}")
    print(f"Challenger model: {challenger_name}")
    print(
        "\nGovernance interpretation: the champion is selected by a governance-aware "
        "score, not by raw predictive performance alone."
    )


if __name__ == "__main__":
    main()