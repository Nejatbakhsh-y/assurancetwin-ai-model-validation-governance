"""
Step E — Create the clean HMDA modeling dataset.

Input:
    data/raw/hmda_lar_nj_2024.csv

Outputs:
    data/processed/hmda_modeling_dataset.csv
    reports/tables/modeling_dataset_summary.csv
    reports/tables/modeling_target_distribution.csv

Purpose:
    Create a clean binary modeling dataset for HMDA model validation.

Target:
    approved = 1 for approval/origination-related action_taken codes:
        1 = Loan originated
        2 = Application approved but not accepted
        8 = Preapproval request approved but not accepted

    approved = 0 for non-approval action_taken codes:
        3 = Application denied
        4 = Application withdrawn by applicant
        5 = File closed for incompleteness
        7 = Preapproval request denied

    Dropped:
        6 = Purchased loan
"""

from pathlib import Path
from datetime import datetime
import re
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "hmda_lar_nj_2024.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

OUTPUT_MODELING_DATASET = PROCESSED_DIR / "hmda_modeling_dataset.csv"
OUTPUT_DATASET_SUMMARY = TABLES_DIR / "modeling_dataset_summary.csv"
OUTPUT_TARGET_DISTRIBUTION = TABLES_DIR / "modeling_target_distribution.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Target mapping
# ---------------------------------------------------------------------

ACTION_TAKEN_LABELS = {
    1: "Loan originated",
    2: "Application approved but not accepted",
    3: "Application denied",
    4: "Application withdrawn by applicant",
    5: "File closed for incompleteness",
    6: "Purchased loan",
    7: "Preapproval request denied",
    8: "Preapproval request approved but not accepted",
}

APPROVED_ACTION_CODES = {1, 2, 8}
NOT_APPROVED_ACTION_CODES = {3, 4, 5, 7}
DROPPED_ACTION_CODES = {6}

VALID_ACTION_CODES_FOR_TARGET = APPROVED_ACTION_CODES.union(NOT_APPROVED_ACTION_CODES)


# ---------------------------------------------------------------------
# Candidate columns
# ---------------------------------------------------------------------
# This column list is intentionally conservative.
# It avoids clear post-decision or pricing/outcome fields such as:
# interest_rate, rate_spread, purchaser_type, total_loan_costs,
# origination_charges, discount_points, and lender_credits.

ID_AND_AUDIT_COLUMNS = [
    "activity_year",
    "lei",
    "state_code",
    "county_code",
    "census_tract",
    "derived_msa_md",
]

CANDIDATE_FEATURE_COLUMNS = [
    # Loan/application structure
    "loan_type",
    "loan_purpose",
    "lien_status",
    "preapproval",
    "reverse_mortgage",
    "open_end_line_of_credit",
    "business_or_commercial_purpose",

    # Property / collateral
    "derived_loan_product_type",
    "derived_dwelling_category",
    "construction_method",
    "occupancy_type",
    "total_units",
    "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest",

    # Applicant / financial characteristics
    "loan_amount",
    "loan_to_value_ratio",
    "combined_loan_to_value_ratio",
    "property_value",
    "income",
    "debt_to_income_ratio",
    "applicant_credit_score_type",
    "co_applicant_credit_score_type",

    # Demographic fields retained for governance/fairness analysis.
    # Later modeling scripts can exclude these from predictive features.
    "derived_ethnicity",
    "derived_race",
    "derived_sex",
    "applicant_age",
    "co_applicant_age",
]

NUMERIC_COLUMNS = [
    "loan_amount",
    "loan_to_value_ratio",
    "combined_loan_to_value_ratio",
    "property_value",
    "income",
]

STANDARD_MISSING_VALUES = {
    "",
    " ",
    "NA",
    "N/A",
    "na",
    "n/a",
    "NaN",
    "nan",
    "NULL",
    "null",
    "None",
    "none",
    "Exempt",
    "exempt",
    "Not applicable",
    "Not Applicable",
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def normalize_column_name(col: str) -> str:
    """
    Convert HMDA column names into Python-friendly snake_case names.

    Examples:
        derived_msa-md -> derived_msa_md
        co-applicant_credit_score_type -> co_applicant_credit_score_type
        applicant_ethnicity-1 -> applicant_ethnicity_1
    """
    col = str(col).strip().lower()
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")
    return col


def make_unique_columns(columns):
    """
    Ensure normalized column names remain unique.
    """
    seen = {}
    unique_cols = []

    for col in columns:
        if col not in seen:
            seen[col] = 0
            unique_cols.append(col)
        else:
            seen[col] += 1
            unique_cols.append(f"{col}_{seen[col]}")

    return unique_cols


def clean_string_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and standardize obvious missing values.
    """
    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace(list(STANDARD_MISSING_VALUES), pd.NA)

    return df


def convert_numeric_columns(df: pd.DataFrame, numeric_columns) -> pd.DataFrame:
    """
    Convert selected modeling columns to numeric where present.
    """
    for col in numeric_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def require_file(path: Path) -> None:
    """
    Stop clearly if the raw HMDA data file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\nRaw HMDA file not found:\n{path}\n\n"
            "Expected file:\n"
            "data/raw/hmda_lar_nj_2024.csv\n\n"
            "Place the New Jersey HMDA LAR CSV there before running this script."
        )


def safe_action_label(value):
    """
    Convert action_taken_code to a human-readable label.
    """
    if pd.isna(value):
        return pd.NA

    try:
        return ACTION_TAKEN_LABELS.get(int(value), pd.NA)
    except Exception:
        return pd.NA


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Step E — Create clean HMDA modeling dataset")
    print("=" * 72)

    require_file(RAW_PATH)

    print(f"\nReading raw HMDA file:")
    print(RAW_PATH)

    raw_size_mb = RAW_PATH.stat().st_size / (1024 * 1024)

    df = pd.read_csv(
        RAW_PATH,
        dtype="string",
        low_memory=False,
        keep_default_na=False,
    )

    raw_rows, raw_cols = df.shape

    print(f"\nRaw rows: {raw_rows:,}")
    print(f"Raw columns: {raw_cols:,}")
    print(f"Raw file size: {raw_size_mb:,.2f} MB")

    # Normalize column names.
    normalized_columns = [normalize_column_name(c) for c in df.columns]
    df.columns = make_unique_columns(normalized_columns)

    # Clean string cells.
    df = clean_string_cells(df)

    if "action_taken" not in df.columns:
        available_preview = ", ".join(list(df.columns[:30]))
        raise KeyError(
            "The required column 'action_taken' was not found after column-name "
            "normalization.\n\n"
            f"First available columns are:\n{available_preview}"
        )

    # Convert action_taken to numeric code.
    df["action_taken_code"] = pd.to_numeric(
        df["action_taken"],
        errors="coerce"
    ).astype("Int64")

    # Count action codes before filtering.
    action_counts_before = (
        df["action_taken_code"]
        .value_counts(dropna=False)
        .rename_axis("action_taken_code")
        .reset_index(name="row_count")
        .sort_values("action_taken_code", na_position="last")
    )

    print("\nAction-taken distribution before target filtering:")
    print(action_counts_before.to_string(index=False))

    # Filtering diagnostics.
    rows_missing_action = int(df["action_taken_code"].isna().sum())

    rows_purchased = int(
        df["action_taken_code"].isin(DROPPED_ACTION_CODES).sum()
    )

    # Corrected block:
    # The .sum() must happen before int().
    rows_invalid_for_binary_target = int(
        (
            (~df["action_taken_code"].isin(VALID_ACTION_CODES_FOR_TARGET))
            & (~df["action_taken_code"].isin(DROPPED_ACTION_CODES))
            & (df["action_taken_code"].notna())
        ).sum()
    )

    # Keep only action_taken codes that belong to the binary approval target.
    df_model = df[
        df["action_taken_code"].isin(VALID_ACTION_CODES_FOR_TARGET)
    ].copy()

    if df_model.empty:
        raise ValueError(
            "After filtering action_taken, no rows remain for binary target creation. "
            "Check the raw HMDA file and action_taken coding."
        )

    # Create binary target.
    df_model["approved"] = np.where(
        df_model["action_taken_code"].isin(APPROVED_ACTION_CODES),
        1,
        0,
    ).astype("int8")

    df_model["action_taken_label"] = (
        df_model["action_taken_code"]
        .apply(safe_action_label)
        .astype("string")
    )

    # Convert numeric candidate features.
    df_model = convert_numeric_columns(df_model, NUMERIC_COLUMNS)

    # Create reproducible record identifier after filtering.
    if "record_id" in df_model.columns:
        df_model = df_model.drop(columns=["record_id"])

    df_model.insert(0, "record_id", range(1, len(df_model) + 1))

    # Select only available columns.
    available_id_cols = [c for c in ID_AND_AUDIT_COLUMNS if c in df_model.columns]
    available_feature_cols = [c for c in CANDIDATE_FEATURE_COLUMNS if c in df_model.columns]

    selected_columns = (
        ["record_id"]
        + available_id_cols
        + ["action_taken_code", "action_taken_label", "approved"]
        + available_feature_cols
    )

    # Remove accidental duplicates while preserving order.
    selected_columns = list(dict.fromkeys(selected_columns))

    df_model = df_model[selected_columns].copy()

    # Standardize column order.
    target_cols = (
        ["record_id"]
        + available_id_cols
        + ["action_taken_code", "action_taken_label", "approved"]
    )

    feature_cols_final = [c for c in df_model.columns if c not in target_cols]
    df_model = df_model[target_cols + feature_cols_final]

    # Save modeling dataset.
    df_model.to_csv(OUTPUT_MODELING_DATASET, index=False)

    # Target distribution.
    target_distribution = (
        df_model["approved"]
        .value_counts(dropna=False)
        .rename_axis("approved")
        .reset_index(name="row_count")
        .sort_values("approved")
    )

    target_distribution["target_label"] = target_distribution["approved"].map(
        {
            0: "not_approved_or_not_completed",
            1: "approved_or_origination_related",
        }
    )

    target_distribution["percent"] = (
        target_distribution["row_count"]
        / target_distribution["row_count"].sum()
        * 100
    ).round(4)

    target_distribution = target_distribution[
        ["approved", "target_label", "row_count", "percent"]
    ]

    target_distribution.to_csv(OUTPUT_TARGET_DISTRIBUTION, index=False)

    # Dataset-level summary.
    total_cells = int(df_model.shape[0] * df_model.shape[1])
    missing_cells = int(df_model.isna().sum().sum())
    missing_cell_rate = round(missing_cells / total_cells, 6) if total_cells else np.nan

    approved_rows = int((df_model["approved"] == 1).sum())
    not_approved_rows = int((df_model["approved"] == 0).sum())
    approval_rate = round(approved_rows / len(df_model), 6)

    summary = pd.DataFrame(
        [
            {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "script": "scripts/03_create_clean_hmda_dataset.py",
                "raw_file": str(RAW_PATH.relative_to(PROJECT_ROOT)),
                "raw_file_size_mb": round(raw_size_mb, 2),
                "raw_rows": raw_rows,
                "raw_columns": raw_cols,
                "modeling_rows": int(df_model.shape[0]),
                "modeling_columns": int(df_model.shape[1]),
                "approved_rows": approved_rows,
                "not_approved_rows": not_approved_rows,
                "approval_rate": approval_rate,
                "rows_missing_action_taken": rows_missing_action,
                "rows_dropped_purchased_loans_action_taken_6": rows_purchased,
                "rows_dropped_invalid_for_binary_target": rows_invalid_for_binary_target,
                "missing_cells_in_modeling_dataset": missing_cells,
                "missing_cell_rate": missing_cell_rate,
                "id_and_audit_columns_present": " | ".join(available_id_cols),
                "candidate_feature_columns_present": " | ".join(available_feature_cols),
                "approved_action_codes": "1, 2, 8",
                "not_approved_action_codes": "3, 4, 5, 7",
                "dropped_action_codes": "6",
                "target_definition": (
                    "approved=1 for action_taken in {1,2,8}; "
                    "approved=0 for action_taken in {3,4,5,7}; "
                    "action_taken=6 purchased loans dropped"
                ),
            }
        ]
    )

    summary.to_csv(OUTPUT_DATASET_SUMMARY, index=False)

    print("\nClean modeling dataset created successfully.")
    print(f"Modeling rows: {df_model.shape[0]:,}")
    print(f"Modeling columns: {df_model.shape[1]:,}")
    print(f"Approved rows: {approved_rows:,}")
    print(f"Not approved rows: {not_approved_rows:,}")
    print(f"Approval rate: {approval_rate:.4f}")

    print("\nRows removed before modeling dataset creation:")
    print(f"Missing action_taken: {rows_missing_action:,}")
    print(f"Purchased loans, action_taken=6: {rows_purchased:,}")
    print(f"Invalid for binary target: {rows_invalid_for_binary_target:,}")

    print("\nSaved outputs:")
    print(f"1. {OUTPUT_MODELING_DATASET.relative_to(PROJECT_ROOT)}")
    print(f"2. {OUTPUT_DATASET_SUMMARY.relative_to(PROJECT_ROOT)}")
    print(f"3. {OUTPUT_TARGET_DISTRIBUTION.relative_to(PROJECT_ROOT)}")

    print("\nTarget distribution:")
    print(target_distribution.to_string(index=False))

    print("\nStep E complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: Step E failed.")
        print(str(exc))
        sys.exit(1)