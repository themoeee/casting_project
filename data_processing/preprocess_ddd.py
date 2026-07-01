"""Preprocess Buehler DDD machine XML files into parquet.

The DDD files live below the casting-trial folders, for example::

    data/cavity_sensors/250905/02_Machine data/ddd/ddd_y_250905095951.xml.gz

This script mirrors the cavity-sensor preprocessing style: it discovers all
source files, writes one long-format parquet part per source file, and stores a
small manifest that can be used to trace rows back to their original XML file.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys

import pandas as pd

try:
    from data_processing.read_xml_file import load_xml_data, xml_to_long_dataframe
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from read_xml_file import load_xml_data, xml_to_long_dataframe


DEFAULT_INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ddd"
DEFAULT_PATTERN = "*.xml*"


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


def read_ddd_xml(
    xml_file: Path,
    input_root: Path,
    source_file_id: int,
) -> pd.DataFrame:
    """Read one DDD XML/XML.GZ file and return a long-format DataFrame."""

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
    long_df["ddd_cycle_number"] = pd.to_numeric(
        _metadata_value(metadata, "cycleNumber"),
        errors="coerce",
    )
    long_df["ddd_cycle_number_short_text"] = _metadata_value(
        metadata,
        "cycleNumberShortText",
    )
    long_df["ddd_file_date"] = file_date
    long_df["ddd_date"] = (
        file_date.date().isoformat() if pd.notna(file_date) else pd.NA
    )
    long_df["ddd_ref_time"] = ref_time
    long_df["quality_attribute"] = _metadata_value(metadata, "qualityAttribute")
    long_df["machine_type"] = _metadata_value(metadata, "machineType")
    long_df["machine_id"] = _metadata_value(metadata, "machineId")
    long_df["casting_ident"] = _metadata_value(metadata, "castingIdent")
    long_df["form_ident"] = _metadata_value(metadata, "formIdent")
    long_df["metal_ident"] = _metadata_value(metadata, "metalIdent")

    return long_df


def find_ddd_files(input_path: str | Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """Find DDD XML/XML.GZ files in either a flat or trial-folder layout."""

    input_path = Path(input_path)
    xml_files = sorted(input_path.glob(pattern))
    if xml_files:
        return xml_files

    fallback_patterns = [
        "*.xml*",
        "ddd/*.xml*",
        "*/02_Machine data/ddd/*.xml*",
        "**/ddd/*.xml*",
    ]
    for fallback_pattern in fallback_patterns:
        xml_files = sorted(input_path.glob(fallback_pattern))
        if xml_files:
            return xml_files
    return []


def merge_ddd_to_parquet(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path | None = None,
    pattern: str = DEFAULT_PATTERN,
    overwrite: bool = True,
) -> None:
    """Merge DDD XML/XML.GZ files into a long-format parquet dataset."""

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / "processed" / "ddd_long"
    else:
        output_path = Path(output_path)

    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    xml_files = find_ddd_files(input_path, pattern=pattern)
    if not xml_files:
        raise FileNotFoundError(f"No DDD XML files found with pattern: {input_path / pattern}")

    manifest_rows = []

    for i, xml_file in enumerate(xml_files):
        print(f"\n[{i + 1}/{len(xml_files)}] Processing {xml_file.name}")

        long_df = read_ddd_xml(
            xml_file=xml_file,
            input_root=input_path,
            source_file_id=i,
        )

        print(f"  - Read {len(long_df)} rows from {xml_file.name}")
        if not long_df.empty:
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
                "n_rows_long": len(long_df),
                "ddd_cycle_number": (
                    long_df["ddd_cycle_number"].dropna().iloc[0]
                    if not long_df.empty and long_df["ddd_cycle_number"].notna().any()
                    else pd.NA
                ),
                "ddd_file_date": (
                    long_df["ddd_file_date"].dropna().iloc[0]
                    if not long_df.empty and long_df["ddd_file_date"].notna().any()
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
    manifest_path = output_path.parent / "ddd_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    print(f"\nSaved DDD parquet dataset to: {output_path}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    data_path = DEFAULT_INPUT_PATH
    merge_ddd_to_parquet(
        input_path=data_path,
        output_path=data_path.parent / "processed" / "ddd_long",
    )
