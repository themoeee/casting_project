"""Check whether processed data is complete enough for ML datasets.

The master Excel is the reference list of relevant casting parts. Cavity
sensor data, DDM machine data, and UT test data are checked against that list
by ``casting_part_label``. The checks read the real processed data first;
manifest files are only used as secondary consistency checks.
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


def _label_set(df: pd.DataFrame, column: str = "casting_part_label") -> set[int]:
    if column not in df.columns:
        return set()
    values = pd.to_numeric(df[column], errors="coerce").dropna().astype("Int64")
    return set(values.astype(int).tolist())


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


def _coverage_summary(expected_labels: set[int], found_labels: set[int]) -> dict[str, Any]:
    missing = sorted(expected_labels - found_labels)
    unexpected = sorted(found_labels - expected_labels)
    expected_count = len(expected_labels)
    found_relevant_count = len(expected_labels & found_labels)
    return {
        "expected_casting_part_labels": expected_count,
        "found_relevant_casting_part_labels": found_relevant_count,
        "coverage_percent": (
            round(100 * found_relevant_count / expected_count, 2)
            if expected_count
            else None
        ),
        "missing_casting_part_labels": missing,
        "unexpected_casting_part_labels": unexpected,
    }


def _relevant_master_rows(master_df: pd.DataFrame) -> pd.DataFrame:
    return master_df.dropna(subset=["casting_part_label"]).copy()


def _required_ut_columns(df: pd.DataFrame) -> dict[str, str | None]:
    column_groups = {
        "casting_part_label": ["casting_part_label"],
        "parameter_set": ["parameter_set"],
        "sample_position": ["sample_position", "position_der_probe"],
        "yield_strength": ["yield_strength_mpa", "sigma_yield"],
        "yield_strain": ["yield_strain", "epsilon_yield"],
    }
    return {
        group: next((column for column in options if column in df.columns), None)
        for group, options in column_groups.items()
    }


def load_master_reference(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> tuple[pd.DataFrame, set[int]]:
    """Load the normalized master Excel and return relevant casting labels."""

    master = read_master_excel(excel_path=excel_path, sheet_name=sheet_name)
    relevant = _relevant_master_rows(master)
    labels = _label_set(relevant)
    return master, labels


def check_cavity_sensor_data(
    expected_labels: set[int] | None = None,
    data_path: str | Path = DEFAULT_CAVITY_PARQUET,
    manifest_path: str | Path = DEFAULT_CAVITY_MANIFEST,
    removed_cycles_csv: str | Path = DEFAULT_REMOVED_CAVITY_CSV,
) -> dict[str, Any]:
    """Return health and master-label coverage info for cavity sensor data."""

    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    removed_cycles_csv = Path(removed_cycles_csv)
    result = _base_result("cavity_sensor_data", data_path)

    if expected_labels is None:
        _, expected_labels = load_master_reference()

    if not data_path.exists():
        result["issues"].append("Processed cavity sensor parquet file is missing.")
        result.update(_coverage_summary(expected_labels, set()))
        return _finish(result)

    df = pd.read_parquet(data_path)
    required = [
        "casting_part_label",
        "trial_folder",
        "cavity_sensor_file_cycle_nr",
        "time_s",
        "value",
    ]
    missing = _missing_columns(df, required)
    if missing:
        result["issues"].append(f"Missing required columns: {missing}")

    found_labels = _label_set(df)
    result.update(
        {
            "n_rows": len(df),
            "n_casting_part_labels": len(found_labels),
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
            **_coverage_summary(expected_labels, found_labels),
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
    if result["missing_casting_part_labels"]:
        result["issues"].append("Cavity sensor data is missing master casting_part_labels.")

    return _finish(result)


def check_ddm_data(
    expected_labels: set[int] | None = None,
    data_path: str | Path = DEFAULT_DDM_PARQUET,
    manifest_path: str | Path = DEFAULT_DDM_MANIFEST,
    removed_cycles_csv: str | Path = DEFAULT_REMOVED_DDM_CSV,
) -> dict[str, Any]:
    """Return health and master-label coverage info for DDM machine data."""

    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    removed_cycles_csv = Path(removed_cycles_csv)
    result = _base_result("ddm_machine_data", data_path)

    if expected_labels is None:
        _, expected_labels = load_master_reference()

    if not data_path.exists():
        result["issues"].append("Processed DDM parquet file is missing.")
        result.update(_coverage_summary(expected_labels, set()))
        return _finish(result)

    df = pd.read_parquet(data_path)
    required = ["casting_part_label", "machine_cycle_no", "time_s", "value", "short_name"]
    missing = _missing_columns(df, required)
    if missing:
        result["issues"].append(f"Missing required columns: {missing}")

    found_labels = _label_set(df)
    result.update(
        {
            "n_rows": len(df),
            "n_casting_part_labels": len(found_labels),
            "n_machine_cycles": _count_unique(df, "machine_cycle_no"),
            "n_curves": _count_unique(df, "short_name"),
            "n_source_files": _count_unique(df, "source_file"),
            "n_removed_cycles": (
                len(pd.read_csv(removed_cycles_csv)) if removed_cycles_csv.exists() else None
            ),
            **_coverage_summary(expected_labels, found_labels),
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
    if result["missing_casting_part_labels"]:
        result["issues"].append("DDM machine data is missing master casting_part_labels.")

    return _finish(result)


def check_master_excel(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
    processed_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Return health info for the normalized master Excel and processed copy."""

    excel_path = Path(excel_path)
    result = _base_result("master_excel", excel_path)

    if not excel_path.exists():
        result["issues"].append("Master Excel file is missing.")
        return _finish(result)

    df = read_master_excel(excel_path=excel_path, sheet_name=sheet_name)
    relevant = _relevant_master_rows(df)
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
            "n_relevant_rows": len(relevant),
            "n_parameter_sets": _count_unique(df, "parameter_set"),
            "n_casting_part_labels": _count_unique(relevant, "casting_part_label"),
            "n_machine_cycles": _count_unique(relevant, "machine_cycle_no"),
            "missing_casting_part_labels": int(df["casting_part_label"].isna().sum()),
            "missing_machine_cycles_relevant": int(relevant["machine_cycle_no"].isna().sum()),
            "missing_cavity_keys_relevant": int(
                relevant[["cavity_sensor_trial_folder", "cavity_sensor_file_cycle_nr"]]
                .isna()
                .any(axis=1)
                .sum()
            ),
            "duplicate_casting_part_labels": int(
                relevant["casting_part_label"].dropna().duplicated().sum()
            ),
            "duplicate_machine_cycles": int(
                relevant["machine_cycle_no"].dropna().duplicated().sum()
            ),
        }
    )

    cavity_key_cols = ["cavity_sensor_trial_folder", "cavity_sensor_file_cycle_nr"]
    result["duplicate_cavity_sensor_keys"] = int(
        relevant.dropna(subset=cavity_key_cols).duplicated(cavity_key_cols).sum()
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
        result["issues"].append("Duplicate relevant casting_part_label values found.")
    if result["duplicate_machine_cycles"]:
        result["issues"].append("Duplicate relevant machine_cycle_no values found.")
    if result["duplicate_cavity_sensor_keys"]:
        result["issues"].append("Duplicate relevant cavity sensor join keys found.")
    if result["missing_machine_cycles_relevant"]:
        result["issues"].append("Relevant master rows without machine_cycle_no found.")
    if result["missing_cavity_keys_relevant"]:
        result["issues"].append("Relevant master rows without cavity sensor join key found.")

    return _finish(result)


def check_ut_test_data(
    expected_labels: set[int] | None = None,
    paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Return health and master-label coverage info for UT test data."""

    paths = paths or [DEFAULT_UT_PARQUET, DEFAULT_UT_CSV]
    result = _base_result("ut_test_data", paths[0])

    if expected_labels is None:
        _, expected_labels = load_master_reference()

    try:
        df, used_path = _read_first_existing(paths)
    except FileNotFoundError as error:
        result["issues"].append(str(error))
        result.update(_coverage_summary(expected_labels, set()))
        return _finish(result)

    result["path"] = str(used_path)
    result["path_exists"] = True

    selected_columns = _required_ut_columns(df)
    missing_groups = [
        group for group, column in selected_columns.items() if column is None
    ]
    if missing_groups:
        result["issues"].append(f"Missing expected UT data groups: {missing_groups}")

    found_labels = _label_set(df)
    result.update(
        {
            "n_rows": len(df),
            "n_casting_part_labels": len(found_labels),
            "n_parameter_sets": _count_unique(df, "parameter_set"),
            "selected_columns": selected_columns,
            "columns": list(df.columns),
            **_coverage_summary(expected_labels, found_labels),
        }
    )

    if "casting_part_label" in df.columns:
        result["missing_casting_part_labels_in_rows"] = int(df["casting_part_label"].isna().sum())
    if len(df) == 0:
        result["issues"].append("Processed UT test data is empty.")
    if result["missing_casting_part_labels"]:
        result["issues"].append("UT test data is missing master casting_part_labels.")

    return _finish(result)


def check_ml_input_readiness() -> dict[str, Any]:
    """Check whether every relevant master part has all ML input datasets."""

    master_df, expected_labels = load_master_reference()
    master = check_master_excel()
    cavity = check_cavity_sensor_data(expected_labels=expected_labels)
    ddm = check_ddm_data(expected_labels=expected_labels)
    ut = check_ut_test_data(expected_labels=expected_labels)

    dataset_checks = {
        "master_excel": master,
        "cavity_sensor_data": cavity,
        "ddm_machine_data": ddm,
        "ut_test_data": ut,
    }
    labels_by_dataset = {
        "cavity_sensor_data": expected_labels - set(cavity["missing_casting_part_labels"]),
        "ddm_machine_data": expected_labels - set(ddm["missing_casting_part_labels"]),
        "ut_test_data": expected_labels - set(ut["missing_casting_part_labels"]),
    }
    complete_labels = set(expected_labels)
    for labels in labels_by_dataset.values():
        complete_labels &= labels

    missing_by_label = []
    for label in sorted(expected_labels - complete_labels):
        missing_sources = [
            dataset for dataset, labels in labels_by_dataset.items() if label not in labels
        ]
        missing_by_label.append(
            {
                "casting_part_label": label,
                "missing_sources": missing_sources,
            }
        )

    issues = []
    for name, check in dataset_checks.items():
        for issue in check["issues"]:
            issues.append(f"{name}: {issue}")

    ready = not issues and len(complete_labels) == len(expected_labels)
    message = (
        "Hey, es ist alles gut mit den Daten. Du kannst sie in deine ML-Modelle fuettern."
        if ready
        else "Die ML-Eingangsdaten sind noch nicht vollstaendig. Siehe issues und missing_by_label."
    )

    return {
        "ok": ready,
        "message": message,
        "n_master_rows": len(master_df),
        "n_relevant_casting_part_labels": len(expected_labels),
        "n_complete_casting_part_labels": len(complete_labels),
        "complete_coverage_percent": (
            round(100 * len(complete_labels) / len(expected_labels), 2)
            if expected_labels
            else None
        ),
        "missing_by_label": missing_by_label,
        "issues": issues,
        "checks": dataset_checks,
    }


def check_all_processed_data() -> dict[str, Any]:
    """Run the full ML input readiness check."""

    return check_ml_input_readiness()


def format_ml_readiness_report(result: dict[str, Any]) -> str:
    """Return a compact human-readable ML readiness report."""

    lines = [
        result["message"],
        "",
        "ML input readiness",
        f"- Relevant casting parts: {result['n_relevant_casting_part_labels']}",
        f"- Complete casting parts: {result['n_complete_casting_part_labels']}",
        f"- Complete coverage: {result['complete_coverage_percent']}%",
        "",
        "Dataset coverage",
    ]

    for dataset_name in ["cavity_sensor_data", "ddm_machine_data", "ut_test_data"]:
        check = result["checks"][dataset_name]
        lines.append(
            "- "
            f"{dataset_name}: {check['found_relevant_casting_part_labels']}/"
            f"{check['expected_casting_part_labels']} labels "
            f"({check['coverage_percent']}%), rows={check.get('n_rows')}, ok={check['ok']}"
        )

    if result["issues"]:
        lines.extend(["", "Issues"])
        lines.extend(f"- {issue}" for issue in result["issues"])

    if result["missing_by_label"]:
        lines.extend(["", "Missing by casting_part_label"])
        for row in result["missing_by_label"][:25]:
            missing_sources = ", ".join(row["missing_sources"])
            lines.append(f"- {row['casting_part_label']}: {missing_sources}")
        remaining = len(result["missing_by_label"]) - 25
        if remaining > 0:
            lines.append(f"- ... {remaining} more")

    return "\n".join(lines)


if __name__ == "__main__":
    result = check_ml_input_readiness()
    print(format_ml_readiness_report(result))
    print()
    pprint(result, sort_dicts=False)
