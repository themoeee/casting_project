"""Inspect DDD and DDM Buehler XML/XML.GZ machine-data folders.

This module is a lightweight discovery/checking layer on top of
``read_xml_file.py``.  It cycles through the XML files in ``data/ddd`` and
``data/ddm`` separately, extracts the available metadata/curve/sample headers,
and performs a few small sanity checks.

Typical usage from the repository root::

    python -m data_processing.inspect_machine_xml_files

The command writes CSV summaries to ``data/processed/xml_inspection`` and also
prints the most important headers to the terminal.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Iterable

import pandas as pd

try:
    from data_processing.read_xml_file import load_xml_data, summarize_xml_curves
except ModuleNotFoundError:
    # Allow direct execution from inside data_processing, e.g.
    # ``py inspect_machine_xml_files.py``.
    sys.path.append(str(Path(__file__).resolve().parent))
    from read_xml_file import load_xml_data, summarize_xml_curves


XML_PATTERN = "*.xml*"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "processed" / "xml_inspection"


def _join_values(values: Iterable[object]) -> str:
    """Join non-empty unique values into a stable semicolon-separated string."""

    cleaned = {
        str(value).strip()
        for value in values
        if value is not None and str(value).strip() and str(value) != "nan"
    }
    return "; ".join(sorted(cleaned))


def _status_from_filename(path: Path) -> str:
    """Extract the trailing quality/status marker from DDM filenames if present."""

    name = path.name
    if name.lower().endswith(".xml.gz"):
        name = name[:-7]
    elif name.lower().endswith(".xml"):
        name = name[:-4]
    match = re.search(r"_([A-Za-z]+)$", name)
    return match.group(1).lower() if match else ""


def _duplicate_values(values: Iterable[str]) -> str:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return _join_values(duplicates)


def _inspect_xml_file(xml_file: Path, input_root: Path, collection: str) -> tuple[dict, list[dict]]:
    """Inspect one XML/XML.GZ file and return one file row plus curve rows."""

    xml_data = load_xml_data(xml_file)
    curve_summary = summarize_xml_curves(xml_data)

    curve_rows: list[dict] = []
    sample_columns: set[str] = set()
    curve_metadata_keys: set[str] = set()
    curve_labels: list[str] = []
    duplicate_short_names = _duplicate_values(curve.short_name for curve in xml_data.curves)

    checks: list[str] = []
    if not xml_data.curves:
        checks.append("no curveObject elements found")

    for curve_index, curve in enumerate(xml_data.curves):
        curve_sample_columns = list(curve.data.columns)
        sample_columns.update(curve_sample_columns)
        curve_metadata_keys.update(curve.metadata.keys())
        curve_labels.append(curve.label)

        if curve.sample_count == 0:
            checks.append(f"curve {curve_index} has no samples")
        elif "time_us" in curve.data and curve.data["time_us"].duplicated().any():
            checks.append(f"curve {curve.short_name or curve_index} has duplicate time_us values")
        elif "time_us" in curve.data and (curve.data["time_us"].diff().dropna() < 0).any():
            checks.append(f"curve {curve.short_name or curve_index} has decreasing time_us values")

        curve_rows.append(
            {
                "collection": collection,
                "source_rel_path": str(xml_file.relative_to(input_root)),
                "filename": xml_file.name,
                "file_status": _status_from_filename(xml_file),
                "curve_index": curve_index,
                "short_name": curve.short_name,
                "long_name": curve.long_name,
                "unit": curve.unit,
                "samples": curve.sample_count,
                "duration_s": (
                    float(curve.data["time_s"].iloc[-1])
                    if curve.sample_count and "time_s" in curve.data
                    else pd.NA
                ),
                "min": (
                    float(curve.data["value"].min())
                    if curve.sample_count and "value" in curve.data
                    else pd.NA
                ),
                "max": (
                    float(curve.data["value"].max())
                    if curve.sample_count and "value" in curve.data
                    else pd.NA
                ),
                "mean": (
                    float(curve.data["value"].mean())
                    if curve.sample_count and "value" in curve.data
                    else pd.NA
                ),
                "sample_columns": _join_values(curve_sample_columns),
                "curve_metadata_keys": _join_values(curve.metadata.keys()),
            }
        )

    if duplicate_short_names:
        checks.append(f"duplicate short_name values: {duplicate_short_names}")

    if not curve_summary.empty:
        if (curve_summary["samples"] <= 0).any():
            checks.append("one or more curves have zero samples")
        if curve_summary["duration_s"].dropna().le(0).any():
            checks.append("one or more curves have non-positive duration")

    file_row = {
        "collection": collection,
        "source_rel_path": str(xml_file.relative_to(input_root)),
        "filename": xml_file.name,
        "file_status": _status_from_filename(xml_file),
        "size_bytes": xml_file.stat().st_size,
        "metadata_key_count": len(xml_data.metadata),
        "metadata_keys": _join_values(xml_data.metadata.keys()),
        "curve_count": len(xml_data.curves),
        "total_samples": int(curve_summary["samples"].sum()) if not curve_summary.empty else 0,
        "min_duration_s": (
            float(curve_summary["duration_s"].min()) if not curve_summary.empty else pd.NA
        ),
        "max_duration_s": (
            float(curve_summary["duration_s"].max()) if not curve_summary.empty else pd.NA
        ),
        "curve_headers": "short_name; long_name; unit; samples; duration_s; min; max; mean",
        "sample_headers": _join_values(sample_columns),
        "curve_metadata_headers": _join_values(curve_metadata_keys),
        "available_curves": _join_values(curve_labels),
        "checks": "OK" if not checks else " | ".join(checks),
    }
    return file_row, curve_rows


def inspect_xml_folder(
    folder: str | Path,
    collection: str,
    input_root: str | Path | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cycle through all XML/XML.GZ files in one folder and inspect them."""

    folder = Path(folder)
    input_root = Path(input_root) if input_root is not None else folder.parent
    xml_files = sorted(folder.glob(XML_PATTERN))
    if limit is not None:
        xml_files = xml_files[:limit]

    file_rows: list[dict] = []
    curve_rows: list[dict] = []
    for index, xml_file in enumerate(xml_files, start=1):
        print(f"[{collection} {index}/{len(xml_files)}] {xml_file.name}")
        try:
            file_row, current_curve_rows = _inspect_xml_file(xml_file, input_root, collection)
        except Exception as error:  # keep discovery robust across many files
            file_row = {
                "collection": collection,
                "source_rel_path": str(xml_file.relative_to(input_root)),
                "filename": xml_file.name,
                "file_status": _status_from_filename(xml_file),
                "size_bytes": xml_file.stat().st_size,
                "metadata_key_count": pd.NA,
                "metadata_keys": "",
                "curve_count": pd.NA,
                "total_samples": pd.NA,
                "min_duration_s": pd.NA,
                "max_duration_s": pd.NA,
                "curve_headers": "",
                "sample_headers": "",
                "curve_metadata_headers": "",
                "available_curves": "",
                "checks": f"ERROR: {type(error).__name__}: {error}",
            }
            current_curve_rows = []
        file_rows.append(file_row)
        curve_rows.extend(current_curve_rows)

    return pd.DataFrame(file_rows), pd.DataFrame(curve_rows)


def inspect_ddd_folder(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect ``data/ddd`` files."""

    data_root = Path(data_root)
    return inspect_xml_folder(data_root / "ddd", collection="ddd", input_root=data_root, limit=limit)


def inspect_ddm_folder(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect ``data/ddm`` files."""

    data_root = Path(data_root)
    return inspect_xml_folder(data_root / "ddm", collection="ddm", input_root=data_root, limit=limit)


def inspect_machine_xml_files(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inspect both DDD and DDM collections and return summary tables."""

    ddd_files, ddd_curves = inspect_ddd_folder(data_root=data_root, limit=limit)
    ddm_files, ddm_curves = inspect_ddm_folder(data_root=data_root, limit=limit)

    file_summary = pd.concat([ddd_files, ddm_files], ignore_index=True)
    curve_summary = pd.concat([ddd_curves, ddm_curves], ignore_index=True)
    header_summary = summarize_headers(file_summary, curve_summary)
    return file_summary, curve_summary, header_summary


def summarize_headers(file_summary: pd.DataFrame, curve_summary: pd.DataFrame) -> pd.DataFrame:
    """Create one compact row per collection with the discovered headers."""

    rows: list[dict] = []
    for collection, file_group in file_summary.groupby("collection", dropna=False):
        curve_group = curve_summary[curve_summary["collection"] == collection]
        rows.append(
            {
                "collection": collection,
                "file_count": len(file_group),
                "error_count": int(file_group["checks"].astype(str).str.startswith("ERROR").sum()),
                "file_status_values": _join_values(file_group.get("file_status", [])),
                "metadata_headers": _join_values(
                    key
                    for keys in file_group.get("metadata_keys", [])
                    for key in str(keys).split("; ")
                ),
                "file_summary_headers": _join_values(file_summary.columns),
                "curve_summary_headers": _join_values(curve_summary.columns),
                "sample_headers": _join_values(
                    key
                    for keys in file_group.get("sample_headers", [])
                    for key in str(keys).split("; ")
                ),
                "curve_metadata_headers": _join_values(
                    key
                    for keys in file_group.get("curve_metadata_headers", [])
                    for key in str(keys).split("; ")
                ),
                "curve_names": _join_values(curve_group.get("short_name", [])),
                "curve_long_names": _join_values(curve_group.get("long_name", [])),
                "units": _join_values(curve_group.get("unit", [])),
            }
        )
    return pd.DataFrame(rows)


def write_inspection_outputs(
    file_summary: pd.DataFrame,
    curve_summary: pd.DataFrame,
    header_summary: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Write the inspection tables as CSV files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_summary.to_csv(output_dir / "machine_xml_file_summary.csv", index=False)
    curve_summary.to_csv(output_dir / "machine_xml_curve_summary.csv", index=False)
    header_summary.to_csv(output_dir / "machine_xml_header_summary.csv", index=False)


def print_header_summary(header_summary: pd.DataFrame) -> None:
    """Print a readable terminal summary of the discovered headers."""

    for row in header_summary.to_dict(orient="records"):
        print(f"\n=== {row['collection'].upper()} XML files ===")
        print(f"Files checked: {row['file_count']}  |  Parse errors: {row['error_count']}")
        if row.get("file_status_values"):
            print(f"Filename status values: {row['file_status_values']}")
        print(f"File summary headers: {row['file_summary_headers']}")
        print(f"Curve summary headers: {row['curve_summary_headers']}")
        print(f"XML metadata headers: {row['metadata_headers']}")
        print(f"Curve metadata headers: {row['curve_metadata_headers']}")
        print(f"Sample/data headers: {row['sample_headers']}")
        print(f"Curve short names: {row['curve_names']}")
        print(f"Curve long names: {row['curve_long_names']}")
        print(f"Units: {row['units']}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect DDD and DDM XML/XML.GZ files.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Repository data folder.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where CSV summaries are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional per-folder file limit for quick smoke tests.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary without writing CSV files.",
    )
    args = parser.parse_args()

    file_summary, curve_summary, header_summary = inspect_machine_xml_files(
        data_root=args.data_root,
        limit=args.limit,
    )
    print_header_summary(header_summary)

    problems = file_summary[file_summary["checks"] != "OK"]
    if not problems.empty:
        print("\nFiles with warnings/errors:")
        print(problems[["collection", "source_rel_path", "checks"]].to_string(index=False))
    else:
        print("\nMinor checks: OK for all inspected files.")

    if not args.no_write:
        write_inspection_outputs(file_summary, curve_summary, header_summary, args.output_dir)
        print(f"\nWrote inspection CSV files to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
