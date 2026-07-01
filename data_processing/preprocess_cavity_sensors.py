# This file will serve as an first step to start out analysis of the data
# What I want to do is to read in all casting trials and see if all the data is in the correct format and available
# data lies in data folder

from pathlib import Path
import pandas as pd
from pathlib import Path
import shutil
import re
import pandas as pd


META_COLS = [
    "CycleNr", "MachineCycleNr", "Timestamp", "ChannelNr", "PartNumber",
    "Description", "SensorSerialNumber", "RangeYMin", "RangeYMax",
    "YMin", "TimeatYMin", "YMax", "TimeatYMax", "NumberOfPoints",
    "YUnit", "Cavity", "Position", "SensorType"
]


def merge_cavity_sensor(input_path: str):
    '''This function merges all cavity sensor files from different days into one csv file and saves it'''

    target_path = Path(__file__).resolve().parent.parent / "data"    
    
    csv_files = []

    for csv_file in input_path.glob("*/06_Cavity_Sensors/*.csv"):
        csv_files.append({
            "trial_folder": csv_file.parent.parent.name,
            "filename": csv_file.name,
            "path": str(csv_file),
        })

    print(f"Found {len(csv_files)} CSV files")
    print(f"\n Found CSV files {[file['trial_folder'] for file in csv_files]}")  # This line seems to have an error; trial_folder is not defined in this context
    
def _safe_name(name: str) -> str:
    '''Used to create safe file names for parquet files (e.g. avoid special characters)'''
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name)

def read_cavity_sensor_csv(csv_file: Path, input_root: Path, source_file_id: int) -> pd.DataFrame:
    """Read a cavity sensor CSV file and return a long-format DataFrame with metadata."""

    df = pd.read_csv(csv_file, sep=";", low_memory=False)

    # Check unnamed columns before dropping them
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]

    if unnamed_cols:
        for col in unnamed_cols:
            non_empty = df[col].dropna()

            if len(non_empty) > 0:
                print(f"\nWARNING: Unnamed column {col} in {csv_file.name} contains data!")
                print(non_empty.head(10))
                raise ValueError(f"Unnamed column {col} is not empty. Do not drop automatically.")

        print(f"Dropping empty unnamed columns in {csv_file.name}: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    #invalid_local_indices = [i for i, t in enumerate(time_values) if pd.isna(t)]

    # if invalid_local_indices:
    #     print(f"\nInvalid time columns in {csv_file.name}:")

    #     for local_idx in invalid_local_indices:
    #         original_col = time_cols[local_idx]
    #         global_idx = df.columns.get_loc(original_col)

    #         print("\n--- Invalid column diagnosis ---")
    #         print(f"Original column name: {repr(original_col)}")
    #         print(f"Local index in time_cols: {local_idx}")
    #         print(f"Global index in df.columns: {global_idx}")

    #         print("Previous 5 time columns:")
    #         print(time_cols[max(0, local_idx - 5):local_idx])

    #         print("Next 5 time columns:")
    #         print(time_cols[local_idx + 1:local_idx + 6])

    #         print("First 10 values in invalid column:")
    #         print(df[original_col].head(10))

    #         print("Non-NaN values in invalid column:")
    #         print(df[original_col].dropna().head(10))

    #     raise ValueError("Invalid time column found.")

    #print("Amount of NaN values in time_values: ", time_values.isna().sum())

    
    missing = [col for col in META_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_file} is missing metadata columns: {missing}")
    
    time_cols = [col for col in df.columns if col not in META_COLS] # All collumns not containing metadata are time columns

    # time is stored in the column names after the metadata columns
    time_values = pd.to_numeric(
        pd.Series(time_cols).astype(str).str.replace(",", ".", regex=False),
        errors="coerce" 
    )

    valid_time_cols = [col for col, t in zip(time_cols, time_values) if pd.notna(t)] 
    if len(valid_time_cols) != len(time_cols):
        invalid = [col for col, t in zip(time_cols, time_values) if pd.isna(t)]
        print(f"Warning: ignored invalid time columns in {csv_file.name}: {invalid}")
        invalid_index = [df.columns.get_loc(col) for col in invalid]
        print(f"  - Invalid time columns at indices: {invalid_index}")
      #  print(f"The values before conversion issue were: {col[]}")
       


    df = df[META_COLS + valid_time_cols].copy()

    # metadata for later joining
    df["source_file_id"] = source_file_id
    df["source_file"] = csv_file.name
    df["source_rel_path"] = str(csv_file.relative_to(input_root))
    df["trial_folder"] = csv_file.parent.parent.name

    # important later: this is only unique within one cavity sensor file/day
    df["cavity_sensor_file_cycle_nr"] = df["CycleNr"]

    # useful later for matching via date + cavity_sensor_file_cycle_nr
    parsed_timestamp = pd.to_datetime(
        df["Timestamp"],
        format="%Y%m%dT%H%M%S.%f",
        errors="coerce"
    )
    df["cavity_sensor_datetime"] = parsed_timestamp
    df["cavity_sensor_date"] = parsed_timestamp.dt.date.astype("string")

    id_cols = META_COLS + [
        "source_file_id",
        "source_file",
        "source_rel_path",
        "trial_folder",
        "cavity_sensor_file_cycle_nr",
        "cavity_sensor_datetime",
        "cavity_sensor_date",
    ]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=valid_time_cols,
        var_name="time_s",
        value_name="value"
    )

    long_df["time_s"] = pd.to_numeric(long_df["time_s"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    long_df = long_df.dropna(subset=["time_s", "value"])

    return long_df


def merge_cavity_sensors_to_parquet(
    input_path: str | Path,
    output_path: str | Path | None = None,
    pattern: str = "*/06_Cavity_Sensors/*.csv",
    overwrite: bool = True,
) -> None:
    
    '''Merge cavity sensor CSV files into a single parquet dataset.'''

    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / "processed" / "cavity_sensors_long"
    else:
        output_path = Path(output_path)

    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found with pattern: {input_path / pattern}")

    manifest_rows = []

    for i, csv_file in enumerate(csv_files):
        print(f"\n[{i + 1}/{len(csv_files)}] Processing {csv_file.name}")

        long_df = read_cavity_sensor_csv(
            csv_file=csv_file,
            input_root=input_path,
            source_file_id=i,
        )
        print(f"  - Read {len(long_df)} rows from {csv_file.name}")
        print(f"  - Time range: {long_df['time_s'].min()}s to {long_df['time_s'].max()}s")

        part_name = f"part_{i:04d}_{_safe_name(csv_file.stem)}.parquet"
        part_path = output_path / part_name

        long_df.to_parquet(part_path, index=False)

        manifest_rows.append({
            "source_file_id": i,
            "source_file": csv_file.name,
            "source_rel_path": str(csv_file.relative_to(input_path)),
            "trial_folder": csv_file.parent.parent.name,
            "n_rows_long": len(long_df),
            "descriptions": sorted(long_df["Description"].dropna().unique().tolist()),
            "min_cycle": long_df["CycleNr"].min(),
            "max_cycle": long_df["CycleNr"].max(),
            "min_time_s": long_df["time_s"].min(),
            "max_time_s": long_df["time_s"].max(),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_path.parent / "cavity_sensors_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    print(f"\nSaved cavity sensor parquet dataset to: {output_path}")
    print(f"Saved manifest to: {manifest_path}")
   

if __name__ == "__main__":

    data_path = Path(__file__).resolve().parent.parent / "data" / "cavity_sensors"


    merge_cavity_sensor(data_path)

    merge_cavity_sensors_to_parquet(
        input_path=data_path,
        output_path=data_path.parent / "processed" / "cavity_sensors_long",
    )
