"""Create a temporary fake tensile-test database for casting samples.

This is intentionally synthetic placeholder data. It uses the real parameter
sets and casting part labels from the master Excel file, then creates one row
for every sample position 1..27 and part number 1..3 per casting part.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXCEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "master_excel"
    / "250929_Analysis_Casting_Trials_EpR_corrected260129_extend260402.xlsx"
)
DEFAULT_SHEET_NAME = "or2509 Parameter Sets"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "ut_tests" / "fake_ut_test_results.csv"
DEFAULT_OUTPUT_PARQUET = PROJECT_ROOT / "data" / "raw" / "ut_tests" / "fake_ut_test_results.parquet"

SAMPLE_POSITIONS = range(1, 28)
PART_NUMBERS = range(1, 4)


@dataclass(frozen=True)
class BuildOutputs:
    """Paths written by the fake UT build command."""

    csv_path: Path
    parquet_path: Path | None
    row_count: int


def _stable_seed(*values: object) -> int:
    """Create a reproducible random seed from row identifiers."""

    text = "|".join(str(value) for value in values)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _to_int(value: object) -> int | None:
    """Convert Excel numeric labels to plain ints when possible."""

    if pd.isna(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        return int(numeric)
    return None


def read_casting_parts(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> pd.DataFrame:
    """Read the real parameter sets and casting part labels from Excel."""

    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    columns = ["Parameter Set", "Casting Part Label"]
    try:
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            usecols=columns,
            engine="openpyxl",
        )
    except ImportError as error:
        raise ImportError(
            "Reading .xlsx files requires openpyxl. Install project requirements first."
        ) from error

    df = df.rename(
        columns={
            "Parameter Set": "parameter_set",
            "Casting Part Label": "casting_part_label",
        }
    )
    df = df.dropna(subset=["parameter_set", "casting_part_label"]).copy()
    df["parameter_set"] = df["parameter_set"].astype("string").str.strip()
    df["casting_part_label"] = df["casting_part_label"].map(_to_int).astype("Int64")
    df = df.dropna(subset=["parameter_set", "casting_part_label"])
    return df.drop_duplicates().sort_values(
        ["parameter_set", "casting_part_label"],
        kind="stable",
    )


def _fake_mechanical_values(
    parameter_set: str,
    casting_part_label: int,
    sample_position: int,
    part_number: int,
) -> dict[str, float]:
    """Generate plausible, deterministic placeholder tensile-test values."""

    rng = random.Random(
        _stable_seed(parameter_set, casting_part_label, sample_position, part_number)
    )

    position_gradient = (sample_position - 14) * 0.35
    part_gradient = (part_number - 2) * 1.8
    label_gradient = (casting_part_label % 17) * 0.45
    parameter_gradient = (_stable_seed(parameter_set) % 1100) / 100.0

    yield_strength_mpa = (
        132.0
        + parameter_gradient
        + label_gradient
        + position_gradient
        + part_gradient
        + rng.uniform(-4.5, 4.5)
    )
    yield_strain = (
        0.0048
        + sample_position * 0.000055
        + part_number * 0.00012
        + rng.uniform(-0.00035, 0.00035)
    )

    return {
        "yield_strength_mpa": round(yield_strength_mpa, 3),
        "yield_strain": round(max(yield_strain, 0.001), 6),
        "whip_bezier_p0_x": 0.0,
        "whip_bezier_p0_y": 0.0,
        "whip_bezier_p1_x": round(0.24 + rng.uniform(-0.035, 0.035), 6),
        "whip_bezier_p1_y": round(yield_strength_mpa * rng.uniform(0.43, 0.58), 3),
        "whip_bezier_p2_x": round(yield_strain, 6),
        "whip_bezier_p2_y": round(yield_strength_mpa, 3),
    }


def build_fake_ut_database(casting_parts: pd.DataFrame) -> pd.DataFrame:
    """Create the fake UT database table."""

    rows: list[dict[str, object]] = []
    for part in casting_parts.itertuples(index=False):
        parameter_set = str(part.parameter_set)
        casting_part_label = int(part.casting_part_label)
        for sample_position in SAMPLE_POSITIONS:
            for part_number in PART_NUMBERS:
                row = {
                    "parameter_set": parameter_set,
                    "casting_part_label": casting_part_label,
                    "sample_position": sample_position,
                    "part_number": part_number,
                }
                row.update(
                    _fake_mechanical_values(
                        parameter_set=parameter_set,
                        casting_part_label=casting_part_label,
                        sample_position=sample_position,
                        part_number=part_number,
                    )
                )
                rows.append(row)

    return pd.DataFrame(rows)


def write_fake_ut_database(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    output_parquet: str | Path | None = DEFAULT_OUTPUT_PARQUET,
) -> BuildOutputs:
    """Build and write the temporary fake UT database."""

    casting_parts = read_casting_parts(excel_path=excel_path, sheet_name=sheet_name)
    fake_ut = build_fake_ut_database(casting_parts)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fake_ut.to_csv(output_csv, index=False)

    output_parquet_path: Path | None = None
    if output_parquet is not None:
        output_parquet_path = Path(output_parquet)
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        fake_ut.to_parquet(output_parquet_path, index=False)

    return BuildOutputs(
        csv_path=output_csv,
        parquet_path=output_parquet_path,
        row_count=len(fake_ut),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporary fake UT test results.")
    parser.add_argument("--excel-path", default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-parquet", default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--no-parquet", action="store_true", help="Do not write parquet output.")
    args = parser.parse_args()

    outputs = write_fake_ut_database(
        excel_path=args.excel_path,
        sheet_name=args.sheet_name,
        output_csv=args.output_csv,
        output_parquet=None if args.no_parquet else args.output_parquet,
    )

    print(f"Wrote fake UT CSV: {outputs.csv_path}")
    if outputs.parquet_path is not None:
        print(f"Wrote fake UT parquet: {outputs.parquet_path}")
    print(f"Rows: {outputs.row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
