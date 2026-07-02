# This file should be used to compare the raw DDM XML data and the processed
# DDM parquet data for one example index.

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
    from data_processing.processing.preprocess_ddm import (
        DEFAULT_INPUT_PATH,
        DEFAULT_MERGED_OUTPUT_PATH,
        DEFAULT_OUTPUT_PATH,
        find_ddm_files,
        load_xml_data,
        xml_to_long_dataframe,
    )
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))
    from data_processing.processing.preprocess_ddm import (
        DEFAULT_INPUT_PATH,
        DEFAULT_MERGED_OUTPUT_PATH,
        DEFAULT_OUTPUT_PATH,
        find_ddm_files,
        load_xml_data,
        xml_to_long_dataframe,
    )


def _get_file_for_index(files: list[Path], example_index: int, label: str) -> Path:
    if not files:
        raise FileNotFoundError(f"No {label} files found.")
    if example_index < 0 or example_index >= len(files):
        raise IndexError(
            f"{label} index {example_index} is outside 0..{len(files) - 1}"
        )
    return files[example_index]

def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def _print_dict(data: dict) -> None:
    if not data:
        print("  <empty>")
        return

    key_width = max(len(str(key)) for key in data)
    for key, value in data.items():
        print(f"  {key:<{key_width}} : {value}")

def _raw_ddm_file(example_index: int) -> Path:
    raw_files = find_ddm_files(DEFAULT_INPUT_PATH)
    return _get_file_for_index(raw_files, example_index, "raw DDM XML")

def _processed_ddm_data(
    example_index: int,
    raw_file: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    if DEFAULT_MERGED_OUTPUT_PATH.exists():
        processed_df = pd.read_parquet(DEFAULT_MERGED_OUTPUT_PATH)
        if "source_file_id" in processed_df.columns:
            processed_df = processed_df[processed_df["source_file_id"] == example_index]
        elif raw_file is not None and "source_file" in processed_df.columns:
            processed_df = processed_df[processed_df["source_file"] == raw_file.name]
        return DEFAULT_MERGED_OUTPUT_PATH, processed_df

    processed_files = sorted(DEFAULT_OUTPUT_PATH.glob("part_*.parquet"))
    processed_file = _get_file_for_index(
        processed_files,
        example_index,
        "processed DDM parquet",
    )
    return processed_file, pd.read_parquet(processed_file)

def show_raw_ddm_data(example_index: int) -> Path:
    """Show the raw DDM XML data information for one sorted file index."""

    raw_file = _raw_ddm_file(example_index)
    xml_data = load_xml_data(raw_file)

    _print_header(f"Raw DDM data for example index {example_index}")
    print(f"Raw file: {raw_file}")
    print(f"Available metadata fields: {len(xml_data.metadata)}")
    print(f"Available curves: {len(xml_data.curves)}")

    print(
        "\nThese are the metadata headers of the raw DDM data "
        f"for example index {example_index}:"
    )
    for key in sorted(xml_data.metadata):
        print(f"  - {key}")

    print("\nRaw metadata values:")
    _print_dict(dict(sorted(xml_data.metadata.items())))

    curve_rows = []
    for curve in xml_data.curves:
        curve_rows.append(
            {
                "short_name": curve.short_name,
                "long_name": curve.long_name,
                "unit": curve.unit,
                "samples": curve.sample_count,
                "data_columns": ", ".join(curve.data.columns),
            }
        )

    print("\nRaw curve information:")
    if curve_rows:
        curve_df = pd.DataFrame(curve_rows)
        print(curve_df.head(MAX_CURVES_TO_PRINT).to_string(index=False))
        if len(curve_df) > MAX_CURVES_TO_PRINT:
            print(f"... {len(curve_df) - MAX_CURVES_TO_PRINT} more curves")
    else:
        print("  <no curves>")

    return raw_file

def show_processed_ddm_data(example_index: int, raw_file: Path | None = None) -> Path:
    """Show the processed DDM parquet data information for one sorted file index."""

    processed_file, processed_df = _processed_ddm_data(example_index, raw_file=raw_file)

    _print_header(f"Processed DDM data for example index {example_index}")
    print(f"Processed file: {processed_file}")
    print(f"Rows for this index: {len(processed_df)}")
    print(f"Columns: {len(processed_df.columns)}")

    print(
        "\nThese are the headers of the processed DDM data "
        f"for example index {example_index}:"
    )
    for column in processed_df.columns:
        print(f"  - {column} ({processed_df[column].dtype})")

    if processed_df.empty:
        print("\nNo processed rows found for this index.")
        return processed_file

    metadata_columns = [
        "source_file_id",
        "source_file",
        "source_rel_path",
        "trial_folder",
        "machine_cycle_no",
        "ddm_file_status",
        "ddm_cycle_number",
        "ddm_cycle_number_short_text",
        "ddm_file_date",
        "ddm_date",
        "ddm_ref_time",
        "quality_attribute",
        "machine_type",
        "machine_id",
        "casting_ident",
        "form_ident",
        "metal_ident",
        "casting_part_label",
    ]
    metadata = {}
    for column in metadata_columns:
        if column in processed_df.columns:
            values = processed_df[column].dropna()
            metadata[column] = values.iloc[0] if not values.empty else "<empty>"

    print("\nProcessed metadata values:")
    _print_dict(metadata)

    if {"short_name", "long_name", "unit"}.issubset(processed_df.columns):
        curve_counts = (
            processed_df.groupby(["short_name", "long_name", "unit"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
        )
        print("\nProcessed curve information:")
        print(curve_counts.head(MAX_CURVES_TO_PRINT).to_string(index=False))
        if len(curve_counts) > MAX_CURVES_TO_PRINT:
            print(f"... {len(curve_counts) - MAX_CURVES_TO_PRINT} more curves")

    # print("\nFirst processed rows:")
    # print(processed_df.head(MAX_ROWS_TO_PRINT).to_string(index=False))

    return processed_file


if __name__ == "__main__":

    INDEX_RAN_DDM = 63
    MAX_CURVES_TO_PRINT = 50
    MAX_ROWS_TO_PRINT = 5


    if len(sys.argv) > 1:
        INDEX_RAN_DDM = int(sys.argv[1])

    raw_ddm_file = show_raw_ddm_data(INDEX_RAN_DDM)
    show_processed_ddm_data(INDEX_RAN_DDM, raw_file=raw_ddm_file)
