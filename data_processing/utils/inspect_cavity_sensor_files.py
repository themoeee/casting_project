# This file should be used to compare the raw cavity sensor CSV data and the
# processed cavity sensor parquet data for one example index.

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
    from data_processing.processing.preprocess_cavity_sensors import (
        DEFAULT_INPUT_PATH,
        DEFAULT_MERGED_OUTPUT_PATH,
        DEFAULT_OUTPUT_PATH,
        META_COLS,
        read_cavity_sensor_csv,
    )
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))
    from data_processing.processing.preprocess_cavity_sensors import (
        DEFAULT_INPUT_PATH,
        DEFAULT_MERGED_OUTPUT_PATH,
        DEFAULT_OUTPUT_PATH,
        META_COLS,
        read_cavity_sensor_csv,
    )


INDEX_RAN_CAVITY_SENSOR = 63
RAW_CAVITY_SENSOR_FILE = (
    DEFAULT_INPUT_PATH
    / "250909"
    / "06_Cavity_Sensors"
    / "cycle_data_1_78_shots_18-95.csv"
)
MAX_DESCRIPTIONS_TO_PRINT = 50
MAX_ROWS_TO_PRINT = 5


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

def _raw_cavity_sensor_file() -> Path:
    if not RAW_CAVITY_SENSOR_FILE.exists():
        raise FileNotFoundError(f"Raw cavity sensor CSV not found: {RAW_CAVITY_SENSOR_FILE}")
    return RAW_CAVITY_SENSOR_FILE

def _processed_cavity_sensor_data(
    casting_part_label: int,
) -> tuple[Path, pd.DataFrame]:
    if DEFAULT_MERGED_OUTPUT_PATH.exists():
        processed_df = pd.read_parquet(DEFAULT_MERGED_OUTPUT_PATH)
        if "casting_part_label" not in processed_df.columns:
            raise KeyError("Processed cavity sensor data has no casting_part_label column.")
        filtered_df = processed_df[
            processed_df["casting_part_label"] == casting_part_label
        ]
        return DEFAULT_MERGED_OUTPUT_PATH, filtered_df

    processed_files = sorted(DEFAULT_OUTPUT_PATH.glob("part_*.parquet"))
    if not processed_files:
        raise FileNotFoundError("No processed cavity sensor parquet files found.")

    filtered_parts = []
    for processed_file in processed_files:
        part_df = pd.read_parquet(processed_file)
        if "casting_part_label" not in part_df.columns:
            raise KeyError(f"{processed_file} has no casting_part_label column.")
        part_df = part_df[part_df["casting_part_label"] == casting_part_label]
        if not part_df.empty:
            filtered_parts.append(part_df)

    if not filtered_parts:
        return DEFAULT_OUTPUT_PATH, pd.DataFrame()
    return DEFAULT_OUTPUT_PATH, pd.concat(filtered_parts, ignore_index=True)

def show_raw_cavity_sensor_data() -> Path:
    """Show the hardcoded raw cavity sensor CSV data information."""

    raw_file = _raw_cavity_sensor_file()
    raw_df = pd.read_csv(raw_file, sep=";", low_memory=False)
    long_df = read_cavity_sensor_csv(
        csv_file=raw_file,
        input_root=DEFAULT_INPUT_PATH,
        source_file_id=0,
    )

    metadata_columns = [column for column in META_COLS if column in raw_df.columns]
    time_columns = [column for column in raw_df.columns if column not in META_COLS]

    _print_header("Raw cavity sensor data")
    print(f"Raw file: {raw_file}")
    print(f"Raw rows: {len(raw_df)}")
    print(f"Raw columns: {len(raw_df.columns)}")
    print(f"Metadata columns: {len(metadata_columns)}")
    print(f"Time columns: {len(time_columns)}")
    print(f"Long rows after raw melt: {len(long_df)}")

    print(
        "\nThese are the metadata headers of the raw cavity sensor data:"
    )
    for column in metadata_columns:
        print(f"  - {column}")

    print("\nRaw first-row metadata values:")
    if raw_df.empty:
        print("  <empty>")
    else:
        metadata = raw_df[metadata_columns].iloc[0].to_dict()
        _print_dict(metadata)

    if "Description" in raw_df.columns:
        descriptions = (
            raw_df["Description"]
            .dropna()
            .drop_duplicates()
            .head(MAX_DESCRIPTIONS_TO_PRINT)
            .tolist()
        )
        print("\nRaw descriptions:")
        for description in descriptions:
            print(f"  - {description}")

    return raw_file

def show_processed_cavity_sensor_data(
    casting_part_label: int,
) -> Path:
    """Show the processed cavity sensor parquet data for one casting_part_label."""

    processed_file, processed_df = _processed_cavity_sensor_data(
        casting_part_label,
    )

    _print_header(f"Processed cavity sensor data for casting_part_label {casting_part_label}")
    print(f"Processed file: {processed_file}")
    print(f"Rows for this casting_part_label: {len(processed_df)}")
    print(f"Columns: {len(processed_df.columns)}")

    print(
        "\nThese are the headers of the processed cavity sensor data "
        f"for casting_part_label {casting_part_label}:"
    )
    for column in processed_df.columns:
        print(f"  - {column} ({processed_df[column].dtype})")

    if processed_df.empty:
        print("\nNo processed rows found for this casting_part_label.")
        return processed_file

    metadata_columns = [
        "source_file_id",
        "source_file",
        "source_rel_path",
        "trial_folder",
        "cavity_sensor_file_cycle_nr",
        "cavity_sensor_datetime",
        "cavity_sensor_date",
        "casting_part_label",
        "CycleNr",
        "MachineCycleNr",
        "Timestamp",
        "ChannelNr",
        "PartNumber",
        "Description",
        "SensorSerialNumber",
        "YUnit",
        "Cavity",
        "Position",
        "SensorType",
    ]
    metadata = {}
    for column in metadata_columns:
        if column in processed_df.columns:
            values = processed_df[column].dropna()
            metadata[column] = values.iloc[0] if not values.empty else "<empty>"

    print("\nProcessed metadata values:")
    _print_dict(metadata)

    if {"Description", "YUnit", "Cavity", "Position", "SensorType"}.issubset(
        processed_df.columns
    ):
        sensor_counts = (
            processed_df.groupby(
                ["Description", "YUnit", "Cavity", "Position", "SensorType"],
                dropna=False,
            )
            .size()
            .rename("rows")
            .reset_index()
        )
        print("\nProcessed sensor information:")
        print(sensor_counts.head(MAX_DESCRIPTIONS_TO_PRINT).to_string(index=False))
        if len(sensor_counts) > MAX_DESCRIPTIONS_TO_PRINT:
            print(f"... {len(sensor_counts) - MAX_DESCRIPTIONS_TO_PRINT} more sensors")

    # print("\nFirst processed rows:")
    # print(processed_df.head(MAX_ROWS_TO_PRINT).to_string(index=False))

    return processed_file



if __name__ == "__main__":
    show_raw_cavity_sensor_data()
    show_processed_cavity_sensor_data(INDEX_RAN_CAVITY_SENSOR)
