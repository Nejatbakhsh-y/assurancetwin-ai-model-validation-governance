from pathlib import Path
import zipfile

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAW_DIR = Path("data/raw")
TABLES_DIR = Path("reports/tables")
FIGURES_DIR = Path("reports/figures")

SCHEMA_OUTPUT = TABLES_DIR / "hmda_schema_summary.csv"
MISSING_OUTPUT = TABLES_DIR / "missing_value_summary.csv"
TARGET_OUTPUT = TABLES_DIR / "target_distribution.csv"
MISSINGNESS_PLOT = FIGURES_DIR / "missingness_plot.png"

CHUNKSIZE = 50_000

TARGET_CANDIDATES = [
    "action_taken",
    "loan_approved",
    "approval_status",
    "target",
    "label",
]

HMDA_ACTION_TAKEN_LABELS = {
    "1": "Loan originated",
    "2": "Application approved but not accepted",
    "3": "Application denied",
    "4": "Application withdrawn by applicant",
    "5": "File closed for incompleteness",
    "6": "Purchased loan",
    "7": "Preapproval request denied",
    "8": "Preapproval request approved but not accepted",
}

MISSING_TOKENS = [
    "",
    " ",
    "NA",
    "N/A",
    "NaN",
    "nan",
    "NULL",
    "null",
    "None",
    "none",
]


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def find_raw_data_file() -> Path:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            "The folder data/raw does not exist. Create data/raw and place the HMDA raw file there."
        )

    valid_suffixes = [".csv", ".gz", ".zip", ".parquet"]

    files = [
        path
        for path in RAW_DIR.iterdir()
        if path.is_file()
        and path.name != ".gitkeep"
        and any(path.name.lower().endswith(suffix) for suffix in valid_suffixes)
    ]

    if not files:
        raise FileNotFoundError(
            "No HMDA data file found in data/raw. Expected .csv, .csv.gz, .zip, or .parquet."
        )

    hmda_like_files = [
        path
        for path in files
        if "hmda" in path.name.lower()
        or "lar" in path.name.lower()
        or "loan" in path.name.lower()
    ]

    selected_file = sorted(hmda_like_files or files, key=lambda x: x.name.lower())[0]
    return selected_file


def get_csv_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as zipped:
        csv_members = [
            member
            for member in zipped.namelist()
            if member.lower().endswith(".csv")
            and not member.lower().startswith("__macosx/")
        ]

        if not csv_members:
            raise FileNotFoundError(f"No CSV file was found inside {zip_path.name}.")

        csv_members = sorted(
            csv_members,
            key=lambda member: zipped.getinfo(member).file_size,
            reverse=True,
        )

        return csv_members[0]


def read_csv_chunks(file_path_or_buffer, **extra_kwargs):
    common_kwargs = {
        "chunksize": CHUNKSIZE,
        "low_memory": False,
        "dtype": "string",
        "keep_default_na": True,
        "na_values": MISSING_TOKENS,
    }
    common_kwargs.update(extra_kwargs)

    try:
        yield from pd.read_csv(file_path_or_buffer, encoding="utf-8", **common_kwargs)
    except UnicodeDecodeError:
        yield from pd.read_csv(file_path_or_buffer, encoding="latin1", **common_kwargs)


def read_chunks(data_file: Path):
    file_name = data_file.name.lower()

    if file_name.endswith(".zip"):
        selected_member = get_csv_from_zip(data_file)
        print(f"CSV selected inside ZIP: {selected_member}")

        with zipfile.ZipFile(data_file, "r") as zipped:
            with zipped.open(selected_member) as file_handle:
                yield from read_csv_chunks(file_handle)

    elif file_name.endswith(".csv.gz"):
        yield from read_csv_chunks(data_file, compression="gzip")

    elif file_name.endswith(".csv"):
        yield from read_csv_chunks(data_file)

    elif file_name.endswith(".parquet"):
        df = pd.read_parquet(data_file)
        yield df.astype("string")

    else:
        raise ValueError(f"Unsupported file type: {data_file.name}")


def normalize_column_name(column_name: str) -> str:
    return str(column_name).strip().lower()


def find_target_column(columns) -> str | None:
    normalized_columns = {
        normalize_column_name(column): column
        for column in columns
    }

    for candidate in TARGET_CANDIDATES:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    return None


def example_values(series: pd.Series, max_values: int = 5) -> str:
    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    values = values[values != ""]

    unique_values = (
        values.drop_duplicates()
        .head(max_values)
        .tolist()
    )

    return " | ".join(unique_values)


def count_numeric_like_values(series: pd.Series) -> int:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return 0

    numeric_version = pd.to_numeric(cleaned, errors="coerce")
    return int(numeric_version.notna().sum())


def build_missingness_plot(missing_summary: pd.DataFrame) -> None:
    plot_data = missing_summary.head(30).copy()

    if plot_data.empty:
        return

    plot_data = plot_data.sort_values("missing_percent", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(plot_data["column_name"], plot_data["missing_percent"])
    plt.xlabel("Missing Percent")
    plt.ylabel("Column")
    plt.title("Top 30 HMDA Variables by Missingness")
    plt.tight_layout()
    plt.savefig(MISSINGNESS_PLOT, dpi=300)
    plt.close()


def main() -> None:
    ensure_output_dirs()

    data_file = find_raw_data_file()
    print(f"Using raw data file: {data_file}")

    total_rows = 0
    column_order = None
    sample_df = None
    missing_counts = None
    target_column = None
    target_counts = None

    for chunk_number, chunk in enumerate(read_chunks(data_file), start=1):
        if chunk.empty:
            continue

        chunk.columns = [str(column).strip() for column in chunk.columns]

        if sample_df is None:
            column_order = list(chunk.columns)
            sample_df = chunk.head(5_000).copy()
            missing_counts = pd.Series(0, index=column_order, dtype="int64")

            target_column = find_target_column(column_order)

            if target_column is not None:
                target_counts = pd.Series(dtype="int64")

        missing_columns = set(column_order) - set(chunk.columns)
        if missing_columns:
            raise ValueError(
                "Inconsistent columns across chunks. Missing columns in current chunk: "
                + ", ".join(sorted(missing_columns))
            )

        chunk = chunk[column_order]

        total_rows += len(chunk)

        chunk_missing_counts = chunk.isna().sum()
        missing_counts = (
            missing_counts.add(chunk_missing_counts, fill_value=0)
            .astype("int64")
        )

        if target_column is not None:
            chunk_target_counts = (
                chunk[target_column]
                .fillna("MISSING")
                .astype(str)
                .str.strip()
                .replace("", "MISSING")
                .value_counts()
            )

            target_counts = (
                target_counts.add(chunk_target_counts, fill_value=0)
                .astype("int64")
            )

        print(f"Processed chunk {chunk_number:,}; cumulative rows: {total_rows:,}")

    if total_rows == 0 or sample_df is None:
        raise ValueError("The selected HMDA file appears to be empty.")

    schema_records = []

    for column in column_order:
        missing_count = int(missing_counts[column])
        non_missing_count = int(total_rows - missing_count)
        missing_percent = round((missing_count / total_rows) * 100, 4)

        sample_non_missing = sample_df[column].dropna()
        sample_non_missing_count = int(sample_non_missing.shape[0])
        numeric_like_count = count_numeric_like_values(sample_df[column])

        if sample_non_missing_count > 0:
            numeric_like_percent_in_sample = round(
                (numeric_like_count / sample_non_missing_count) * 100,
                4,
            )
        else:
            numeric_like_percent_in_sample = 0.0

        schema_records.append(
            {
                "column_name": column,
                "sample_inferred_dtype": str(sample_df[column].dtype),
                "total_rows": total_rows,
                "non_missing_count": non_missing_count,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "unique_values_in_sample": int(sample_df[column].nunique(dropna=True)),
                "numeric_like_percent_in_sample": numeric_like_percent_in_sample,
                "example_values": example_values(sample_df[column]),
            }
        )

    schema_summary = pd.DataFrame(schema_records)
    schema_summary.to_csv(SCHEMA_OUTPUT, index=False)

    missing_summary = (
        schema_summary[
            [
                "column_name",
                "total_rows",
                "non_missing_count",
                "missing_count",
                "missing_percent",
            ]
        ]
        .sort_values(
            by=["missing_percent", "missing_count"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    missing_summary.to_csv(MISSING_OUTPUT, index=False)

    if target_column is not None:
        target_distribution = (
            target_counts
            .sort_index()
            .rename_axis("target_value")
            .reset_index(name="count")
        )

        target_distribution["target_variable"] = target_column
        target_distribution["percent"] = (
            target_distribution["count"]
            / target_distribution["count"].sum()
            * 100
        ).round(4)

        if normalize_column_name(target_column) == "action_taken":
            target_distribution["target_label"] = (
                target_distribution["target_value"]
                .astype(str)
                .map(HMDA_ACTION_TAKEN_LABELS)
                .fillna("Unmapped or missing value")
            )
        else:
            target_distribution["target_label"] = target_distribution["target_value"]

        target_distribution = target_distribution[
            [
                "target_variable",
                "target_value",
                "target_label",
                "count",
                "percent",
            ]
        ]

    else:
        target_distribution = pd.DataFrame(
            [
                {
                    "target_variable": "NOT_FOUND",
                    "target_value": "NOT_FOUND",
                    "target_label": (
                        "No target column found. Checked: "
                        + ", ".join(TARGET_CANDIDATES)
                    ),
                    "count": 0,
                    "percent": 0.0,
                }
            ]
        )

    target_distribution.to_csv(TARGET_OUTPUT, index=False)

    build_missingness_plot(missing_summary)

    print("")
    print("Step D completed successfully.")
    print(f"Rows inspected: {total_rows:,}")
    print(f"Columns inspected: {len(column_order):,}")
    print(f"Schema summary saved to: {SCHEMA_OUTPUT}")
    print(f"Missing value summary saved to: {MISSING_OUTPUT}")
    print(f"Target distribution saved to: {TARGET_OUTPUT}")
    print(f"Missingness plot saved to: {MISSINGNESS_PLOT}")

    if target_column is not None:
        print(f"Detected target column: {target_column}")
    else:
        print("No target column was detected automatically.")


if __name__ == "__main__":
    main()