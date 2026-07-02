"""Preprocess Buehler DDM machine XML files into parquet.

The DDM files live in a flat raw folder, for example::

    data/raw/ddm/ddm_y_FliesslaengerformBuehler_0003_ok.xml.gz

This script mirrors the cavity-sensor preprocessing style: it discovers all
source files, writes one long-format parquet part per source file, and stores a
small manifest that can be used to trace rows back to their original XML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys

import pandas as pd

try:
    from data_processing.processing.build_master_sample_table import (
        DEFAULT_EXCEL_PATH,
        DEFAULT_SHEET_NAME,
        read_master_excel,
    )
    from data_processing.utils.read_xml_file import load_xml_data, xml_to_long_dataframe
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))
    from build_master_sample_table import (
        DEFAULT_EXCEL_PATH,
        DEFAULT_SHEET_NAME,
        read_master_excel,
    )
    from read_xml_file import load_xml_data, xml_to_long_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "ddm"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "long_form" / "ddm_parts"
DEFAULT_MERGED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_input_data" / "ddm_machine_data.parquet"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "processing_info" / "ddm_manifest.parquet"
DEFAULT_REMOVED_CYCLES_CSV = PROJECT_ROOT / "data" / "processed" / "processing_info" / "removed_ddm_cycles.csv"
DEFAULT_PATTERN = "*.xml*"


@dataclass(frozen=True)
class DDMPreprocessOutputs:
    """Paths written by the DDM preprocessing command."""

    parts_dir: Path
    merged_data: Path
    manifest: Path
    removed_cycles_csv: Path


def _safe_name(name: str) -> str:
    """Create safe file names for parquet parts."""

    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name)


def _trial_folder(xml_file: Path) -> str:
    """Return the trial folder name for the standard machine-data layout."""

    parts = xml_file.parts
    if "02_Machine data" in parts:
        machine_data_index = parts.index("02_Machine data")
        if machine_data_index > 0:
            return parts[machine_data_index - 1]
    if xml_file.parent.parent.parent != xml_file.parent:
        return xml_file.parent.parent.parent.name
    return ""


def _metadata_value(metadata: dict[str, str], key: str) -> str | pd.NA:
    value = metadata.get(key, "")
    return value if value != "" else pd.NA


def _machine_cycle_no(xml_file: Path) -> int | pd.NA:
    """Extract the DDM machine cycle number from the source filename."""

    match = re.search(r"_(\d+)_([A-Za-z]+)\.xml(?:\.gz)?$", xml_file.name, re.IGNORECASE)
    if not match:
        return pd.NA
    return int(match.group(1))


def _ddm_file_status(xml_file: Path) -> str | pd.NA:
    """Extract the DDM quality/status suffix from the source filename."""

    match = re.search(r"_(\d+)_([A-Za-z]+)\.xml(?:\.gz)?$", xml_file.name, re.IGNORECASE)
    if not match:
        return pd.NA
    return match.group(2).lower()


def build_ddm_casting_part_mapping(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> pd.DataFrame:
    """Build the unique machine cycle to casting part label mapping."""

    master = read_master_excel(excel_path=excel_path, sheet_name=sheet_name)
    mapping = master[
        ["machine_cycle_no", "casting_part_label"]
    ].dropna(subset=["machine_cycle_no", "casting_part_label"]).copy()
    mapping["machine_cycle_no"] = (
        pd.to_numeric(mapping["machine_cycle_no"], errors="coerce")
        .round()
        .astype("Int64")
    )

    duplicate_keys = mapping.duplicated(["machine_cycle_no"], keep=False)
    if duplicate_keys.any():
        duplicates = mapping.loc[
            duplicate_keys,
            ["machine_cycle_no", "casting_part_label"],
        ].sort_values("machine_cycle_no")
        raise ValueError(
            "Master Excel contains non-unique DDM machine cycle keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return mapping


def attach_casting_part_labels(
    long_df: pd.DataFrame,
    mapping: pd.DataFrame,
    xml_file: Path,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach casting_part_label and return irrelevant machine cycles removed."""

    machine_cycle_no = _machine_cycle_no(xml_file)
    source_file = xml_file.name
    relevant_cycle = (
        pd.notna(machine_cycle_no)
        and mapping["machine_cycle_no"].eq(machine_cycle_no).any()
    )

    long_df = long_df.copy()
    long_df["machine_cycle_no"] = (
        pd.to_numeric(long_df["machine_cycle_no"], errors="coerce")
        .round()
        .astype("Int64")
    )

    if long_df.empty:
        removed = pd.DataFrame(columns=["machine_cycle_no", "source_file"])
        if not relevant_cycle:
            removed = pd.DataFrame(
                [{"machine_cycle_no": machine_cycle_no, "source_file": source_file}]
            )
            if verbose:
                print(f"  - Removed DDM cycle without relevant master casting_part_label from {xml_file.name}")
        return long_df, removed

    labeled = long_df.merge(
        mapping,
        on="machine_cycle_no",
        how="left",
        validate="many_to_one",
    )

    removed = labeled.loc[
        labeled["casting_part_label"].isna(),
        ["machine_cycle_no", "source_file"],
    ].drop_duplicates()
    labeled = labeled.dropna(subset=["casting_part_label"]).copy()

    if verbose and not removed.empty:
        print(f"  - Removed DDM cycle without relevant master casting_part_label from {xml_file.name}")

    return labeled, removed


def validate_all_master_parts_have_ddm_data(
    mapping: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    """Verify every relevant master casting_part_label has DDM data."""

    expected = mapping[["machine_cycle_no", "casting_part_label"]].drop_duplicates()
    if merged.empty or "casting_part_label" not in merged.columns:
        missing = expected
        raise ValueError(
            "Some master casting_part_labels have no matching DDM data:\n"
            f"{missing.to_string(index=False)}"
        )

    found_labels = set(merged["casting_part_label"].dropna().astype("Int64").tolist())
    missing = expected[
        ~expected["casting_part_label"].astype("Int64").isin(found_labels)
    ]

    if not missing.empty:
        raise ValueError(
            "Some master casting_part_labels have no matching DDM data:\n"
            f"{missing.to_string(index=False)}"
        )


def merge_ddm_part_files(
    parts_dir: str | Path = DEFAULT_OUTPUT_PATH,
    merged_output_path: str | Path = DEFAULT_MERGED_OUTPUT_PATH,
    pattern: str = "part_*.parquet",
) -> pd.DataFrame:
    """Merge all DDM part parquet files into one parquet table.

    The part files are treated as the source of truth for the merged output so
    an interrupted run can be completed by calling this function directly.
    """

    parts_dir = Path(parts_dir)
    merged_output_path = Path(merged_output_path)
    part_files = sorted(parts_dir.glob(pattern))
    if not part_files:
        raise FileNotFoundError(f"No DDM parquet part files found with pattern: {parts_dir / pattern}")

    tables = [pd.read_parquet(part_file) for part_file in part_files]
    non_empty_tables = [table for table in tables if not table.empty]
    merged = (
        pd.concat(non_empty_tables, ignore_index=True)
        if non_empty_tables
        else pd.DataFrame()
    )

    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(merged_output_path, index=False)
    return merged


def read_ddm_xml(
    xml_file: Path,
    input_root: Path,
    source_file_id: int,
) -> pd.DataFrame:
    """Read one DDM XML/XML.GZ file and return a long-format DataFrame."""

    xml_data = load_xml_data(xml_file)
    long_df = xml_to_long_dataframe(xml_data)

    if long_df.empty:
        long_df = pd.DataFrame(
            columns=["time_us", "time_s", "value", "short_name", "long_name", "unit"]
        )

    metadata = xml_data.metadata
    file_date = pd.to_datetime(_metadata_value(metadata, "fileDate"), errors="coerce")
    ref_time = pd.to_datetime(_metadata_value(metadata, "refTime"), errors="coerce")

    long_df = long_df.copy()
    long_df["source_file_id"] = source_file_id
    long_df["source_file"] = xml_file.name
    long_df["source_rel_path"] = str(xml_file.relative_to(input_root))
    long_df["trial_folder"] = _trial_folder(xml_file)
    long_df["machine_cycle_no"] = _machine_cycle_no(xml_file)
    long_df["ddm_file_status"] = _ddm_file_status(xml_file)
    long_df["ddm_cycle_number"] = pd.to_numeric(
        _metadata_value(metadata, "cycleNumber"),
        errors="coerce",
    )
    long_df["ddm_cycle_number_short_text"] = _metadata_value(
        metadata,
        "cycleNumberShortText",
    )
    long_df["ddm_file_date"] = file_date
    long_df["ddm_date"] = (
        file_date.date().isoformat() if pd.notna(file_date) else pd.NA
    )
    long_df["ddm_ref_time"] = ref_time
    long_df["quality_attribute"] = _metadata_value(metadata, "qualityAttribute")
    long_df["machine_type"] = _metadata_value(metadata, "machineType")
    long_df["machine_id"] = _metadata_value(metadata, "machineId")
    long_df["casting_ident"] = _metadata_value(metadata, "castingIdent")
    long_df["form_ident"] = _metadata_value(metadata, "formIdent")
    long_df["metal_ident"] = _metadata_value(metadata, "metalIdent")

    return long_df


def find_ddm_files(input_path: str | Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """Find DDM XML/XML.GZ files in either a flat or trial-folder layout."""

    input_path = Path(input_path)
    xml_files = sorted(input_path.glob(pattern))
    if xml_files:
        return xml_files

    fallback_patterns = [
        "*.xml*",
        "ddm/*.xml*",
        "*/02_Machine data/ddm/*.xml*",
        "**/ddm/*.xml*",
    ]
    for fallback_pattern in fallback_patterns:
        xml_files = sorted(input_path.glob(fallback_pattern))
        if xml_files:
            return xml_files
    return []


def merge_ddm_to_parquet(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path | None = None,
    merged_output_path: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    removed_cycles_csv: str | Path = DEFAULT_REMOVED_CYCLES_CSV,
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
    pattern: str = DEFAULT_PATTERN,
    overwrite: bool = True,
    require_complete_master_coverage: bool = False,
    verbose: bool = False,
) -> DDMPreprocessOutputs:
    """Merge DDM XML/XML.GZ files into a long-format parquet dataset."""

    input_path = Path(input_path)
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    else:
        output_path = Path(output_path)
    if merged_output_path is None:
        merged_output_path = DEFAULT_MERGED_OUTPUT_PATH
    else:
        merged_output_path = Path(merged_output_path)
    manifest_path = Path(manifest_path)
    removed_cycles_csv = Path(removed_cycles_csv)

    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    removed_cycles_csv.parent.mkdir(parents=True, exist_ok=True)

    xml_files = find_ddm_files(input_path, pattern=pattern)
    if not xml_files:
        raise FileNotFoundError(f"No DDM XML files found with pattern: {input_path / pattern}")
    print(f"Found {len(xml_files)} DDM XML/XML.GZ files.")

    mapping = build_ddm_casting_part_mapping(
        excel_path=excel_path,
        sheet_name=sheet_name,
    )
    manifest_rows = []
    removed_cycles = []

    for i, xml_file in enumerate(xml_files):
        if verbose:
            print(f"\n[{i + 1}/{len(xml_files)}] Processing {xml_file.name}")
        elif i == 0 or (i + 1) % 25 == 0 or i + 1 == len(xml_files):
            print(f"Processing DDM files: {i + 1}/{len(xml_files)}")

        long_df = read_ddm_xml(
            xml_file=xml_file,
            input_root=input_path,
            source_file_id=i,
        )
        raw_long_rows = len(long_df)
        long_df, removed = attach_casting_part_labels(
            long_df=long_df,
            mapping=mapping,
            xml_file=xml_file,
            verbose=verbose,
        )
        if not removed.empty:
            removed_cycles.append(removed)

        if verbose:
            print(f"  - Read {len(long_df)} rows from {xml_file.name}")
        if verbose and not long_df.empty:
            print(f"  - Time range: {long_df['time_s'].min()}s to {long_df['time_s'].max()}s")

        part_name = f"part_{i:04d}_{_safe_name(xml_file.stem)}.parquet"
        part_path = output_path / part_name
        long_df.to_parquet(part_path, index=False)

        curve_counts = (
            long_df.groupby("short_name", dropna=False)
            .size()
            .rename("samples")
            .reset_index()
            if not long_df.empty and "short_name" in long_df
            else pd.DataFrame(columns=["short_name", "samples"])
        )

        manifest_rows.append(
            {
                "source_file_id": i,
                "source_file": xml_file.name,
                "source_rel_path": str(xml_file.relative_to(input_path)),
                "trial_folder": _trial_folder(xml_file),
                "n_rows_raw_long": raw_long_rows,
                "n_rows_long": len(long_df),
                "machine_cycle_no": _machine_cycle_no(xml_file),
                "ddm_file_status": _ddm_file_status(xml_file),
                "casting_part_label": (
                    long_df["casting_part_label"].dropna().iloc[0]
                    if not long_df.empty and long_df["casting_part_label"].notna().any()
                    else pd.NA
                ),
                "ddm_cycle_number": (
                    long_df["ddm_cycle_number"].dropna().iloc[0]
                    if not long_df.empty and long_df["ddm_cycle_number"].notna().any()
                    else pd.NA
                ),
                "ddm_file_date": (
                    long_df["ddm_file_date"].dropna().iloc[0]
                    if not long_df.empty and long_df["ddm_file_date"].notna().any()
                    else pd.NaT
                ),
                "quality_attribute": (
                    long_df["quality_attribute"].dropna().iloc[0]
                    if not long_df.empty and long_df["quality_attribute"].notna().any()
                    else pd.NA
                ),
                "curve_count": int(long_df["short_name"].nunique()) if not long_df.empty else 0,
                "curves": sorted(long_df["short_name"].dropna().unique().tolist())
                if not long_df.empty
                else [],
                "curve_sample_counts": curve_counts.to_dict(orient="records"),
                "min_time_s": long_df["time_s"].min() if not long_df.empty else pd.NA,
                "max_time_s": long_df["time_s"].max() if not long_df.empty else pd.NA,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(manifest_path, index=False)

    if removed_cycles:
        removed_manifest = pd.concat(removed_cycles, ignore_index=True).drop_duplicates()
    else:
        removed_manifest = pd.DataFrame(columns=["machine_cycle_no", "source_file"])
    removed_manifest = removed_manifest.sort_values(
        ["machine_cycle_no", "source_file"],
        ignore_index=True,
    )
    removed_manifest.to_csv(removed_cycles_csv, index=False)
    if not removed_manifest.empty:
        print(f"Removed {len(removed_manifest)} DDM cycles without relevant master casting_part_label.")

    merged = merge_ddm_part_files(
        parts_dir=output_path,
        merged_output_path=merged_output_path,
    )
    try:
        validate_all_master_parts_have_ddm_data(mapping=mapping, merged=merged)
    except ValueError as error:
        if require_complete_master_coverage:
            raise
        print(f"\nWARNING: {error}")

    print(f"\nSaved DDM parquet dataset to: {output_path}")
    print(f"Saved merged DDM data to: {merged_output_path}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Saved removed DDM cycles to: {removed_cycles_csv}")
    return DDMPreprocessOutputs(
        parts_dir=output_path,
        merged_data=merged_output_path,
        manifest=manifest_path,
        removed_cycles_csv=removed_cycles_csv,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess DDM XML/XML.GZ files into parquet.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--merged-output-path", default=DEFAULT_MERGED_OUTPUT_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--removed-cycles-csv", default=DEFAULT_REMOVED_CYCLES_CSV)
    parser.add_argument("--excel-path", default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing part files instead of recreating the parts directory.",
    )
    parser.add_argument(
        "--merge-existing-parts",
        action="store_true",
        help="Only merge existing part_*.parquet files into the merged parquet output.",
    )
    parser.add_argument(
        "--require-complete-master-coverage",
        action="store_true",
        help="Fail if any master casting_part_label has no DDM data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file processing details.",
    )
    args = parser.parse_args()

    if args.merge_existing_parts:
        merged = merge_ddm_part_files(
            parts_dir=args.output_path,
            merged_output_path=args.merged_output_path,
        )
        print(f"Wrote merged DDM data: {Path(args.merged_output_path)}")
        print(f"Rows: {len(merged)}")
    else:
        merge_ddm_to_parquet(
            input_path=args.input_path,
            output_path=args.output_path,
            merged_output_path=args.merged_output_path,
            manifest_path=args.manifest_path,
            removed_cycles_csv=args.removed_cycles_csv,
            excel_path=args.excel_path,
            sheet_name=args.sheet_name,
            pattern=args.pattern,
            overwrite=not args.no_overwrite,
            require_complete_master_coverage=args.require_complete_master_coverage,
            verbose=args.verbose,
        )
