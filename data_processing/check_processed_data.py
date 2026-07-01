"""Quick checks for processed pipeline datasets.

The checks read the real processed data first. Manifest files are only used as
secondary consistency checks, because a manifest can be stale or wrong.
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any

import pandas as pd

try:
    from data_processing.build_master_sample_table import (
        DEFAULT_EXCEL_PATH,
        DEFAULT_SHEET_NAME,
        read_master_excel,
    )
except ModuleNotFoundError:
    from build_master_sample_table import (
        DEFAULT_EXCEL_PATH,
        DEFAULT_SHEET_NAME,
        read_master_excel,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

DEFAULT_MASTER_PARQUET = PROCESSED_ROOT / "parquet" / "master_sample_table.parquet"
DEFAULT_MASTER_CSV = PROCESSED_ROOT / "csv" / "master_sample_table.csv"
DEFAULT_CAVITY_PARQUET = PROCESSED_ROOT / "parquet" / "cavity_sensor_data.parquet"
DEFAULT_CAVITY_MANIFEST = PROCESSED_ROOT / "cavity_sensors_manifest.parquet"
DEFAULT_REMOVED_CAVITY_CSV = PROCESSED_ROOT / "csv" / "removed_cavity_sensor_cycles.csv"
DEFAULT_DDM_PARQUET = PROCESSED_ROOT / "parquet" / "ddm_machine_data.parquet"
DEFAULT_DDM_MANIFEST = PROCESSED_ROOT / "ddm_manifest.parquet"
DEFAULT_REMOVED_DDM_CSV = PROCESSED_ROOT / "csv" / "removed_ddm_cycles.csv"
DEFAULT_UT_PARQUET = PROCESSED_ROOT / "fake_ut_test_results.parquet"
DEFAULT_UT_CSV = PROCESSED_ROOT / "fake_ut_test_results.csv"


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _read_first_existing(paths: list[Path]) -> tuple[pd.DataFrame, Path]:
    for path in paths:
        if path.exists():
            return _read_table(path), path
    raise FileNotFoundError(f"None of these files exist: {paths}")


def _base_result(dataset: str, path: Path) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "path": str(path),
        "path_exists": path.exists(),
        "ok": False,
        "issues": [],
    }


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    result["ok"] = not result["issues"]
    return result


def _missing_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [column for column in required_columns if column not in df.columns]


def _count_unique(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    return int(df[column].dropna().nunique())


def _row_count_from_manifest(manifest_path: Path) -> tuple[int | None, int | None]:
    if not manifest_path.exists():
        return None, None
    manifest = pd.read_parquet(manifest_path)
    manifest_rows = len(manifest)
    manifest_data_rows = (
        int(manifest["n_rows_long"].sum())
        if "n_rows_long" in manifest.columns
        else None
    )
    return manifest_rows, manifest_data_rows


def check_cavity_sensor_data(
    data_path: str | Path = DEFAULT_CAVITY_PARQUET,
    manifest_path: str | Path = DEFAULT_CAVITY_MANIFEST,
    removed_cycles_csv: str | Path = DEFAULT_REMOVED_CAVITY_CSV,
) -> dict[str, Any]:
    """Return basic health info for processed cavity sensor data."""

    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    removed_cycles_csv = Path(removed_cycles_csv)
    result = _base_result("cavity_sensor_data", data_path)

    if not data_path.exists():
        result["issues"].append("Processed cavity sensor parquet file is missing.")
        return _finish(result)

    df = pd.read_parquet(data_path)
    required = ["casting_part_label", "trial_folder", "cavity_sensor_file_cycle_nr", "time_s", "value"]
    missing = _missing_columns(df, required)
    if missing:
        result["issues"].append(f"Missing required columns: {missing}")

    result.update(
        {
            "n_rows": len(df),
            "n_casting_part_labels": _count_unique(df, "casting_part_label"),
            "n_trial_folders": _count_unique(df, "trial_folder"),
            "n_cycles": (
                int(df[["trial_folder", "cavity_sensor_file_cycle_nr"]].drop_duplicates().shape[0])
                if {"trial_folder", "cavity_sensor_file_cycle_nr"}.issubset(df.columns)
                else None
            ),
            "n_source_files": _count_unique(df, "source_file"),
            "n_removed_cycles": (
                len(pd.read_csv(removed_cycles_csv)) if removed_cycles_csv.exists() else None
            ),
        }
    )

    manifest_rows, manifest_data_rows = _row_count_from_manifest(manifest_path)
    result["manifest_exists"] = manifest_path.exists()
    result["manifest_rows"] = manifest_rows
    result["manifest_data_rows"] = manifest_data_rows
    result["manifest_row_count_matches_data"] = (
        manifest_data_rows == len(df) if manifest_data_rows is not None else None
    )
    if manifest_data_rows is not None and manifest_data_rows != len(df):
        result["issues"].append("Manifest n_rows_long sum does not match real cavity data row count.")

    if len(df) == 0:
        result["issues"].append("Processed cavity sensor data is empty.")

    return _finish(result)


def check_ddm_data(
    data_path: str | Path = DEFAULT_DDM_PARQUET,
    manifest_path: str | Path = DEFAULT_DDM_MANIFEST,
    removed_cycles_csv: str | Path = DEFAULT_REMOVED_DDM_CSV,
) -> dict[str, Any]:
    """Return basic health info for processed DDM machine data."""

    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    removed_cycles_csv = Path(removed_cycles_csv)
    result = _base_result("ddm_machine_data", data_path)

    if not data_path.exists():
        result["issues"].append("Processed DDM parquet file is missing.")
        return _finish(result)

    df = pd.read_parquet(data_path)
    required = ["casting_part_label", "machine_cycle_no", "time_s", "value", "short_name"]
    missing = _missing_columns(df, required)
    if missing:
        result["issues"].append(f"Missing required columns: {missing}")

    result.update(
        {
            "n_rows": len(df),
            "n_casting_part_labels": _count_unique(df, "casting_part_label"),
            "n_machine_cycles": _count_unique(df, "machine_cycle_no"),
            "n_curves": _count_unique(df, "short_name"),
            "n_source_files": _count_unique(df, "source_file"),
            "n_removed_cycles": (
                len(pd.read_csv(removed_cycles_csv)) if removed_cycles_csv.exists() else None
            ),
        }
    )

    manifest_rows, manifest_data_rows = _row_count_from_manifest(manifest_path)
    result["manifest_exists"] = manifest_path.exists()
    result["manifest_rows"] = manifest_rows
    result["manifest_data_rows"] = manifest_data_rows
    result["manifest_row_count_matches_data"] = (
        manifest_data_rows == len(df) if manifest_data_rows is not None else None
    )
    if manifest_data_rows is not None and manifest_data_rows != len(df):
        result["issues"].append("Manifest n_rows_long sum does not match real DDM data row count.")

    if len(df) == 0:
        result["issues"].append("Processed DDM data is empty.")

    return _finish(result)


def check_master_excel(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
    processed_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Return health info for the normalized master Excel and optional processed copy."""

    excel_path = Path(excel_path)
    result = _base_result("master_excel", excel_path)

    if not excel_path.exists():
        result["issues"].append("Master Excel file is missing.")
        return _finish(result)

    df = read_master_excel(excel_path=excel_path, sheet_name=sheet_name)
    required = [
        "parameter_set",
        "casting_part_label",
        "machine_cycle_no",
        "casting_date",
        "cavity_sensor_trial_folder",
        "cavity_sensor_file_cycle_nr",
    ]
    missing = _missing_columns(df, required)
    if missing:
        result["issues"].append(f"Missing required columns after normalization: {missing}")

    result.update(
        {
            "n_rows": len(df),
            "n_parameter_sets": _count_unique(df, "parameter_set"),
            "n_casting_part_labels": _count_unique(df, "casting_part_label"),
            "n_machine_cycles": _count_unique(df, "machine_cycle_no"),
            "missing_casting_part_labels": int(df["casting_part_label"].isna().sum()),
            "missing_machine_cycles": int(df["machine_cycle_no"].isna().sum()),
            "duplicate_casting_part_labels": int(df["casting_part_label"].dropna().duplicated().sum()),
            "duplicate_machine_cycles": int(df["machine_cycle_no"].dropna().duplicated().sum()),
        }
    )

    cavity_key_cols = ["cavity_sensor_trial_folder", "cavity_sensor_file_cycle_nr"]
    result["duplicate_cavity_sensor_keys"] = int(
        df.dropna(subset=cavity_key_cols).duplicated(cavity_key_cols).sum()
    )

    processed_paths = processed_paths or [DEFAULT_MASTER_PARQUET, DEFAULT_MASTER_CSV]
    processed_summary = {}
    for processed_path in processed_paths:
        processed_path = Path(processed_path)
        if processed_path.exists():
            processed = _read_table(processed_path)
            processed_summary[str(processed_path)] = {
                "exists": True,
                "n_rows": len(processed),
                "row_count_matches_excel": len(processed) == len(df),
            }
            if len(processed) != len(df):
                result["issues"].append(f"Processed master row count differs: {processed_path}")
        else:
            processed_summary[str(processed_path)] = {"exists": False}
    result["processed_outputs"] = processed_summary

    if result["duplicate_casting_part_labels"]:
        result["issues"].append("Duplicate casting_part_label values found.")
    if result["duplicate_machine_cycles"]:
        result["issues"].append("Duplicate machine_cycle_no values found.")
    if result["duplicate_cavity_sensor_keys"]:
        result["issues"].append("Duplicate cavity sensor join keys found.")

    return _finish(result)


def check_ut_test_data(
    paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Return basic health info for processed uniaxial tensile test data."""

    paths = paths or [DEFAULT_UT_PARQUET, DEFAULT_UT_CSV]
    result = _base_result("ut_test_data", paths[0])

    try:
        df, used_path = _read_first_existing(paths)
    except FileNotFoundError as error:
        result["issues"].append(str(error))
        return _finish(result)

    result["path"] = str(used_path)
    result["path_exists"] = True

    required_groups = {
        "casting_part_label": ["casting_part_label"],
        "parameter_set": ["parameter_set"],
        "sample_position": ["sample_position", "position_der_probe"],
        "yield_strength": ["yield_strength_mpa", "sigma_yield"],
        "yield_strain": ["yield_strain", "epsilon_yield"],
    }
    missing_groups = [
        group
        for group, options in required_groups.items()
        if not any(option in df.columns for option in options)
    ]
    if missing_groups:
        result["issues"].append(f"Missing expected UT data groups: {missing_groups}")

    result.update(
        {
            "n_rows": len(df),
            "n_casting_part_labels": _count_unique(df, "casting_part_label"),
            "n_parameter_sets": _count_unique(df, "parameter_set"),
            "columns": list(df.columns),
        }
    )

    if "casting_part_label" in df.columns:
        result["missing_casting_part_labels"] = int(df["casting_part_label"].isna().sum())
    if len(df) == 0:
        result["issues"].append("Processed UT test data is empty.")

    return _finish(result)


def check_all_processed_data() -> dict[str, dict[str, Any]]:
    """Run all processed-data checks."""

    return {
        "master_excel": check_master_excel(),
        "cavity_sensor_data": check_cavity_sensor_data(),
        "ddm_data": check_ddm_data(),
        "ut_test_data": check_ut_test_data(),
    }


if __name__ == "__main__":
    pprint(check_all_processed_data(), sort_dicts=False)
