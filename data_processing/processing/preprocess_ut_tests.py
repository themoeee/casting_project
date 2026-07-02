"""Preprocess temporary UT test data into final input tables.

For now this is intentionally lightweight: the raw fake UT table is copied into
the processed final-input area after basic schema normalization and checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "ut_tests"
DEFAULT_INPUT_PARQUET = DEFAULT_INPUT_DIR / "fake_ut_test_results.parquet"
DEFAULT_INPUT_CSV = DEFAULT_INPUT_DIR / "fake_ut_test_results.csv"
DEFAULT_OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "final_input_data" / "fake_ut_test_results.parquet"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "final_input_data" / "fake_ut_test_results.csv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "processing_info" / "ut_tests_manifest.parquet"

REQUIRED_COLUMNS = [
    "parameter_set",
    "casting_part_label",
    "sample_position",
    "part_number",
    "yield_strength_mpa",
    "yield_strain",
]


@dataclass(frozen=True)
class UTPreprocessOutputs:
    """Paths written by the UT preprocessing command."""

    parquet_path: Path
    csv_path: Path | None
    manifest_path: Path
    row_count: int


def _read_first_existing(paths: list[Path]) -> tuple[pd.DataFrame, Path]:
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path), path
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path), path
        raise ValueError(f"Unsupported UT input format: {path}")
    raise FileNotFoundError(f"No UT input file found. Checked: {paths}")


def read_raw_ut_tests(
    input_parquet: str | Path = DEFAULT_INPUT_PARQUET,
    input_csv: str | Path = DEFAULT_INPUT_CSV,
) -> tuple[pd.DataFrame, Path]:
    """Read the raw UT test table, preferring parquet over CSV."""

    return _read_first_existing([Path(input_parquet), Path(input_csv)])


def preprocess_ut_tests_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the raw UT table enough for current ML inputs."""

    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"UT test data is missing required columns: {missing}")

    df["parameter_set"] = df["parameter_set"].astype("string").str.strip()
    df["casting_part_label"] = pd.to_numeric(
        df["casting_part_label"],
        errors="coerce",
    ).round().astype("Int64")
    df["sample_position"] = pd.to_numeric(
        df["sample_position"],
        errors="coerce",
    ).round().astype("Int64")
    df["part_number"] = pd.to_numeric(
        df["part_number"],
        errors="coerce",
    ).round().astype("Int64")
    df["yield_strength_mpa"] = pd.to_numeric(df["yield_strength_mpa"], errors="coerce")
    df["yield_strain"] = pd.to_numeric(df["yield_strain"], errors="coerce")

    return df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)


def write_ut_manifest(
    processed: pd.DataFrame,
    source_path: Path,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Write a small manifest for the processed UT table."""

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        [
            {
                "source_file": source_path.name,
                "source_path": str(source_path),
                "n_rows": len(processed),
                "n_casting_part_labels": processed["casting_part_label"].nunique(),
                "n_parameter_sets": processed["parameter_set"].nunique(),
                "n_sample_positions": processed["sample_position"].nunique(),
                "n_part_numbers": processed["part_number"].nunique(),
            }
        ]
    )
    manifest.to_parquet(manifest_path, index=False)
    return manifest_path


def preprocess_ut_tests(
    input_parquet: str | Path = DEFAULT_INPUT_PARQUET,
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_parquet: str | Path = DEFAULT_OUTPUT_PARQUET,
    output_csv: str | Path | None = DEFAULT_OUTPUT_CSV,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> UTPreprocessOutputs:
    """Read raw UT tests and write processed final-input files."""

    raw, source_path = read_raw_ut_tests(
        input_parquet=input_parquet,
        input_csv=input_csv,
    )
    processed = preprocess_ut_tests_table(raw)

    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_parquet, index=False)

    output_csv_path: Path | None = None
    if output_csv is not None:
        output_csv_path = Path(output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(output_csv_path, index=False)

    written_manifest = write_ut_manifest(
        processed=processed,
        source_path=source_path,
        manifest_path=manifest_path,
    )

    print(f"Processed UT test data from: {source_path}")
    print(f"Saved UT parquet: {output_parquet}")
    if output_csv_path is not None:
        print(f"Saved UT CSV: {output_csv_path}")
    print(f"Saved UT manifest: {written_manifest}")
    print(f"Rows: {len(processed)}")

    return UTPreprocessOutputs(
        parquet_path=output_parquet,
        csv_path=output_csv_path,
        manifest_path=written_manifest,
        row_count=len(processed),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess UT test data.")
    parser.add_argument("--input-parquet", default=DEFAULT_INPUT_PARQUET)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-parquet", default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--no-csv", action="store_true", help="Do not write a CSV copy.")
    args = parser.parse_args()

    preprocess_ut_tests(
        input_parquet=args.input_parquet,
        input_csv=args.input_csv,
        output_parquet=args.output_parquet,
        output_csv=None if args.no_csv else args.output_csv,
        manifest_path=args.manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
