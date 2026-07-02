"""Build the canonical casting master sample table from the Excel mapping file.

The resulting table has one row per cast part and is the central join point for
parameter sets, cavity-sensor data, and machine data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXCEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "master_excel"
    / "250929_Analysis_Casting_Trials_EpR_corrected260129_extend260402.xlsx"
)
DEFAULT_SHEET_NAME = "CSV EXPORT CLEANUP"
DEFAULT_OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "final_input_data" / "master_sample_table.parquet"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "final_input_data" / "master_sample_table.csv"
DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "data" / "processed" / "processing_info" / "master_sample_validation"
DEFAULT_CAVITY_ROOT = PROJECT_ROOT / "data" / "raw" / "cavity_sensors"
DEFAULT_DDM_ROOT = PROJECT_ROOT / "data" / "raw" / "ddm"


COLUMN_RENAMES = {
    "Parameter Set": "parameter_set",
    "Machine Cycle No": "machine_cycle_no",
    "Casting Part Label": "casting_part_label",
    "Datum": "casting_date",
    "Cavity Sensor Data File": "cavity_sensor_data_file",
    "Cavity Sensor Single Export File Path": "cavity_sensor_trial_folder",
    "Cavity Sensor File Cycle Nr": "cavity_sensor_file_cycle_nr",
    "Casting diagram File Path": "casting_diagram_file_path",
    "Shot info file path": "shot_info_file_path",
    "Diagnosis Diagram": "diagnosis_diagram",
    "Values corrected from DDM": "values_corrected_from_ddm",
}


@dataclass(frozen=True)
class BuildOutputs:
    """Paths written by the build command."""

    master_table: Path
    master_csv: Path | None
    validation_summary: Path
    validation_issues: Path


def _snake_case(name: object) -> str:
    """Convert an Excel header to a stable snake_case column name."""

    text = str(name).strip()
    text = text.replace("%", "percent")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "unnamed"


def _normalise_columns(columns: Iterable[object]) -> list[str]:
    """Apply explicit column names first, then snake_case the rest."""

    seen: dict[str, int] = {}
    normalised: list[str] = []
    for column in columns:
        new_name = COLUMN_RENAMES.get(str(column).strip(), _snake_case(column))
        seen[new_name] = seen.get(new_name, 0) + 1
        if seen[new_name] > 1:
            new_name = f"{new_name}_{seen[new_name]}"
        normalised.append(new_name)
    return normalised


def _clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip text cells and convert textual nan markers to missing values."""

    df = df.copy()
    for column in df.columns:
        if not (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue
        values = df[column].astype("string").str.strip()
        df[column] = values.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return df


def _to_nullable_int(series: pd.Series) -> pd.Series:
    """Convert mostly numeric identifiers to nullable integer dtype."""

    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _format_identifier(value: object) -> str | pd.NA:
    """Format numeric-like identifiers without a trailing decimal part."""

    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return text


def _extract_trial_folder(value: object) -> str | pd.NA:
    """Extract the cavity-sensor trial folder/day from the Excel cell."""

    if pd.isna(value):
        return pd.NA
    text = str(_format_identifier(value)).strip().replace("/", "\\")
    date_match = re.search(r"(?:^|\\)(\d{6})(?:\\|$)", text)
    if date_match:
        return date_match.group(1)
    loose_date_match = re.search(r"\b\d{6}\b", text)
    if loose_date_match:
        return loose_date_match.group(0)
    return Path(text).name or text


def _normalise_excel_date(series: pd.Series) -> pd.Series:
    """Convert Excel serials or parseable date strings to ISO date strings."""

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_dates = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    string_dates = pd.to_datetime(series, errors="coerce")
    dates = numeric_dates.where(numeric_dates.notna(), string_dates)
    return dates.dt.date.astype("string")


def read_master_excel(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> pd.DataFrame:
    """Read and normalise the Excel mapping sheet."""

    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    except ImportError as error:
        raise ImportError(
            "Reading .xlsx files requires openpyxl. Install project requirements first."
        ) from error

    df = df.dropna(how="all").copy()
    df.columns = _normalise_columns(df.columns)
    df = _clean_object_columns(df)

    required = [
        "parameter_set",
        "machine_cycle_no",
        "casting_part_label",
        "casting_date",
        "cavity_sensor_trial_folder",
        "cavity_sensor_file_cycle_nr",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Excel sheet {sheet_name!r} is missing required columns: {missing}")

    df["machine_cycle_no"] = _to_nullable_int(df["machine_cycle_no"])
    df["casting_part_label"] = _to_nullable_int(df["casting_part_label"])
    df["cavity_sensor_file_cycle_nr"] = _to_nullable_int(df["cavity_sensor_file_cycle_nr"])
    df["casting_date"] = _normalise_excel_date(df["casting_date"])
    df["cavity_sensor_trial_folder"] = (
        df["cavity_sensor_trial_folder"].map(_extract_trial_folder).astype("string")
    )
    df["casting_part_label_matches_machine_cycle"] = (
        df["casting_part_label"].notna()
        & df["machine_cycle_no"].notna()
        & (df["casting_part_label"] == df["machine_cycle_no"])
    )

    return df


def build_cavity_sensor_inventory(cavity_root: str | Path = DEFAULT_CAVITY_ROOT) -> pd.DataFrame:
    """Build an exact inventory of available cavity-sensor cycle numbers."""

    cavity_root = Path(cavity_root)
    rows: list[dict[str, object]] = []
    for csv_file in sorted(cavity_root.glob("*/06_Cavity_Sensors/*.csv")):
        trial_folder = csv_file.parent.parent.name
        try:
            cycles = pd.read_csv(
                csv_file,
                sep=";",
                usecols=["CycleNr"],
                low_memory=False,
            )["CycleNr"]
            unique_cycles = _to_nullable_int(cycles).dropna().drop_duplicates()
        except Exception as error:
            rows.append(
                {
                    "cavity_sensor_trial_folder": trial_folder,
                    "cavity_sensor_file_cycle_nr": pd.NA,
                    "cavity_source_file": csv_file.name,
                    "cavity_source_rel_path": str(csv_file.relative_to(cavity_root)),
                    "cavity_inventory_error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        for cycle_nr in unique_cycles:
            rows.append(
                {
                    "cavity_sensor_trial_folder": trial_folder,
                    "cavity_sensor_file_cycle_nr": int(cycle_nr),
                    "cavity_source_file": csv_file.name,
                    "cavity_source_rel_path": str(csv_file.relative_to(cavity_root)),
                    "cavity_inventory_error": pd.NA,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cavity_sensor_trial_folder",
                "cavity_sensor_file_cycle_nr",
                "cavity_source_file",
                "cavity_source_rel_path",
                "cavity_inventory_error",
                "cavity_match_file_count",
            ]
        )

    inventory = pd.DataFrame(rows)
    good_rows = inventory[inventory["cavity_sensor_file_cycle_nr"].notna()].copy()
    if good_rows.empty:
        inventory["cavity_match_file_count"] = 0
        return inventory

    grouped = (
        good_rows.groupby(
            ["cavity_sensor_trial_folder", "cavity_sensor_file_cycle_nr"],
            dropna=False,
        )
        .agg(
            cavity_source_file=("cavity_source_file", lambda values: "; ".join(sorted(set(values)))),
            cavity_source_rel_path=(
                "cavity_source_rel_path",
                lambda values: "; ".join(sorted(set(values))),
            ),
            cavity_match_file_count=("cavity_source_file", lambda values: len(set(values))),
        )
        .reset_index()
    )
    grouped["cavity_inventory_error"] = pd.NA
    return grouped


def build_ddm_inventory(ddm_root: str | Path = DEFAULT_DDM_ROOT) -> pd.DataFrame:
    """Build an inventory of DDM files keyed by the cycle number in the filename."""

    ddm_root = Path(ddm_root)
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"_(\d+)_([A-Za-z]+)\.xml(?:\.gz)?$", re.IGNORECASE)
    for xml_file in sorted(ddm_root.glob("*.xml*")):
        match = pattern.search(xml_file.name)
        if not match:
            continue
        rows.append(
            {
                "machine_cycle_no": int(match.group(1)),
                "ddm_source_file": xml_file.name,
                "ddm_source_rel_path": str(xml_file.relative_to(ddm_root)),
                "ddm_file_status": match.group(2).lower(),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "machine_cycle_no",
                "ddm_source_file",
                "ddm_source_rel_path",
                "ddm_file_status",
                "ddm_match_file_count",
            ]
        )

    inventory = pd.DataFrame(rows)
    return (
        inventory.groupby("machine_cycle_no", dropna=False)
        .agg(
            ddm_source_file=("ddm_source_file", lambda values: "; ".join(sorted(set(values)))),
            ddm_source_rel_path=("ddm_source_rel_path", lambda values: "; ".join(sorted(set(values)))),
            ddm_file_status=("ddm_file_status", lambda values: "; ".join(sorted(set(values)))),
            ddm_match_file_count=("ddm_source_file", lambda values: len(set(values))),
        )
        .reset_index()
    )


def enrich_with_data_sources(
    sample_table: pd.DataFrame,
    cavity_inventory: pd.DataFrame,
    ddm_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Attach cavity-sensor and DDM source-file metadata to the sample table."""

    enriched = sample_table.merge(
        cavity_inventory,
        on=["cavity_sensor_trial_folder", "cavity_sensor_file_cycle_nr"],
        how="left",
    )
    enriched = enriched.merge(ddm_inventory, on="machine_cycle_no", how="left")

    for column in ["cavity_match_file_count", "ddm_match_file_count"]:
        if column in enriched.columns:
            enriched[column] = enriched[column].fillna(0).astype("Int64")
    return enriched


def _issue_rows(issue_type: str, severity: str, rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Create detailed issue rows for a validation finding."""

    if rows.empty:
        return pd.DataFrame(columns=["issue_type", "severity", *columns])
    issue_df = rows.loc[:, [column for column in columns if column in rows.columns]].copy()
    issue_df.insert(0, "severity", severity)
    issue_df.insert(0, "issue_type", issue_type)
    return issue_df


def validate_master_sample_table(sample_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary and detailed validation issues for the master table."""

    details: list[pd.DataFrame] = []
    base_columns = [
        "parameter_set",
        "machine_cycle_no",
        "casting_part_label",
        "casting_date",
        "cavity_sensor_trial_folder",
        "cavity_sensor_file_cycle_nr",
        "cavity_source_file",
        "ddm_source_file",
    ]

    duplicate_machine = sample_table[
        sample_table["machine_cycle_no"].notna()
        & sample_table["machine_cycle_no"].duplicated(keep=False)
    ]
    details.append(_issue_rows("duplicate_machine_cycle_no", "error", duplicate_machine, base_columns))

    mismatch = sample_table[
        sample_table["machine_cycle_no"].notna()
        & sample_table["casting_part_label"].notna()
        & (sample_table["machine_cycle_no"] != sample_table["casting_part_label"])
    ]
    details.append(_issue_rows("casting_label_machine_cycle_mismatch", "error", mismatch, base_columns))

    parameter_counts = (
        sample_table.groupby("parameter_set", dropna=False)
        .size()
        .rename("part_count")
        .reset_index()
    )
    bad_parameter_counts = parameter_counts[parameter_counts["part_count"] != 3]
    details.append(
        _issue_rows(
            "parameter_set_part_count_not_3",
            "warning",
            bad_parameter_counts,
            ["parameter_set", "part_count"],
        )
    )

    has_cavity_key = (
        sample_table["cavity_sensor_trial_folder"].notna()
        & sample_table["cavity_sensor_file_cycle_nr"].notna()
    )
    duplicate_cavity_key = sample_table[
        has_cavity_key
        & sample_table.duplicated(
            ["cavity_sensor_trial_folder", "cavity_sensor_file_cycle_nr"],
            keep=False,
        )
    ]
    details.append(_issue_rows("duplicate_cavity_join_key_in_master", "error", duplicate_cavity_key, base_columns))

    missing_cavity = sample_table[has_cavity_key & sample_table["cavity_source_file"].isna()]
    details.append(_issue_rows("missing_cavity_sensor_match", "error", missing_cavity, base_columns))

    duplicate_cavity_match = sample_table[
        sample_table.get("cavity_match_file_count", pd.Series(dtype="Int64")).fillna(0) > 1
    ]
    details.append(_issue_rows("duplicate_cavity_sensor_match", "error", duplicate_cavity_match, base_columns))

    missing_ddm = sample_table[
        sample_table["machine_cycle_no"].notna() & sample_table["ddm_source_file"].isna()
    ]
    details.append(_issue_rows("missing_ddm_match", "warning", missing_ddm, base_columns))

    duplicate_ddm_match = sample_table[
        sample_table.get("ddm_match_file_count", pd.Series(dtype="Int64")).fillna(0) > 1
    ]
    details.append(_issue_rows("duplicate_ddm_match", "error", duplicate_ddm_match, base_columns))

    sample_or2509_001 = sample_table[
        (sample_table["parameter_set"] == "or2509-001")
        & sample_table["machine_cycle_no"].isin([12, 13, 14])
    ]
    bad_or2509_001 = sample_or2509_001[
        (sample_or2509_001["cavity_sensor_trial_folder"] != "250908")
        | (sample_or2509_001["cavity_sensor_file_cycle_nr"] != sample_or2509_001["machine_cycle_no"])
        | sample_or2509_001["cavity_source_file"].isna()
    ]
    details.append(_issue_rows("sample_check_or2509_001_failed", "error", bad_or2509_001, base_columns))

    high_cycle_rows = sample_table[
        sample_table["machine_cycle_no"].between(414392, 414423, inclusive="both")
    ]
    bad_high_cycle_rows = high_cycle_rows[
        (high_cycle_rows["cavity_sensor_trial_folder"] != "250905")
        | high_cycle_rows["cavity_source_file"].isna()
    ]
    details.append(_issue_rows("sample_check_414392_414423_failed", "error", bad_high_cycle_rows, base_columns))

    issue_details = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    if issue_details.empty:
        issue_details = pd.DataFrame(columns=["issue_type", "severity"])

    summary_rows = []
    for issue_type, severity in [
        ("duplicate_machine_cycle_no", "error"),
        ("casting_label_machine_cycle_mismatch", "error"),
        ("parameter_set_part_count_not_3", "warning"),
        ("duplicate_cavity_join_key_in_master", "error"),
        ("missing_cavity_sensor_match", "error"),
        ("duplicate_cavity_sensor_match", "error"),
        ("missing_ddm_match", "warning"),
        ("duplicate_ddm_match", "error"),
        ("sample_check_or2509_001_failed", "error"),
        ("sample_check_414392_414423_failed", "error"),
    ]:
        issue_count = int((issue_details["issue_type"] == issue_type).sum())
        summary_rows.append(
            {
                "issue_type": issue_type,
                "severity": severity,
                "issue_count": issue_count,
                "status": "OK" if issue_count == 0 else "CHECK",
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary, issue_details


def build_master_sample_table(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
    output_parquet: str | Path = DEFAULT_OUTPUT_PARQUET,
    output_csv: str | Path | None = DEFAULT_OUTPUT_CSV,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
    cavity_root: str | Path = DEFAULT_CAVITY_ROOT,
    ddm_root: str | Path = DEFAULT_DDM_ROOT,
) -> BuildOutputs:
    """Build, validate, and write the canonical master sample table."""

    sample_table = read_master_excel(excel_path=excel_path, sheet_name=sheet_name)
    cavity_inventory = build_cavity_sensor_inventory(cavity_root=cavity_root)
    ddm_inventory = build_ddm_inventory(ddm_root=ddm_root)
    sample_table = enrich_with_data_sources(sample_table, cavity_inventory, ddm_inventory)

    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_table.to_parquet(output_parquet, index=False)

    output_csv_path: Path | None = None
    if output_csv is not None:
        output_csv_path = Path(output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        sample_table.to_csv(output_csv_path, index=False)

    validation_summary, validation_issues = validate_master_sample_table(sample_table)
    validation_dir = Path(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_summary_path = validation_dir / "master_sample_validation_summary.txt"
    validation_issues_path = validation_dir / "master_sample_validation_issues.csv"
    validation_summary_path.write_text(
        validation_summary.to_string(index=False),
        encoding="utf-8",
    )
    validation_issues.to_csv(validation_issues_path, index=False)

    return BuildOutputs(
        master_table=output_parquet,
        master_csv=output_csv_path,
        validation_summary=validation_summary_path,
        validation_issues=validation_issues_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the casting master sample table.")
    parser.add_argument("--excel-path", default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--output-parquet", default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--no-csv", action="store_true", help="Do not write a CSV copy.")
    parser.add_argument("--validation-dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--cavity-root", default=DEFAULT_CAVITY_ROOT)
    parser.add_argument("--ddm-root", default=DEFAULT_DDM_ROOT)
    args = parser.parse_args()

    outputs = build_master_sample_table(
        excel_path=args.excel_path,
        sheet_name=args.sheet_name,
        output_parquet=args.output_parquet,
        output_csv=None if args.no_csv else args.output_csv,
        validation_dir=args.validation_dir,
        cavity_root=args.cavity_root,
        ddm_root=args.ddm_root,
    )

    print(f"Wrote master table: {outputs.master_table}")
    if outputs.master_csv is not None:
        print(f"Wrote CSV copy: {outputs.master_csv}")
    print(f"Wrote validation summary: {outputs.validation_summary}")
    print(f"Wrote validation issues: {outputs.validation_issues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
