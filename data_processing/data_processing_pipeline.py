"""Orchestrate the complete data processing pipeline.

This module runs the processing steps in the order required to create the
processed dataset:

1. build the master sample table,
2. preprocess cavity sensor CSV files,
3. preprocess DDM machine XML/XML.GZ files,
4. run the processed-data readiness check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.processing.build_master_sample_table import (  # noqa: E402
    BuildOutputs,
    build_master_sample_table,
)
from data_processing.processing.preprocess_cavity_sensors import (  # noqa: E402
    DEFAULT_INPUT_PATH as DEFAULT_CAVITY_INPUT_PATH,
    DEFAULT_MANIFEST_PATH as DEFAULT_CAVITY_MANIFEST_PATH,
    DEFAULT_MERGED_OUTPUT_PATH as DEFAULT_CAVITY_MERGED_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_CAVITY_OUTPUT_PATH,
    DEFAULT_REMOVED_CYCLES_CSV as DEFAULT_REMOVED_CAVITY_CYCLES_CSV,
    merge_cavity_sensors_to_parquet,
)
from data_processing.processing.preprocess_ddm import (  # noqa: E402
    DDMPreprocessOutputs,
    DEFAULT_INPUT_PATH as DEFAULT_DDM_INPUT_PATH,
    DEFAULT_MANIFEST_PATH as DEFAULT_DDM_MANIFEST_PATH,
    DEFAULT_MERGED_OUTPUT_PATH as DEFAULT_DDM_MERGED_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_DDM_OUTPUT_PATH,
    DEFAULT_REMOVED_CYCLES_CSV as DEFAULT_REMOVED_DDM_CYCLES_CSV,
    merge_ddm_part_files,
    merge_ddm_to_parquet,
)
from data_processing.processing.preprocess_ut_tests import (  # noqa: E402
    UTPreprocessOutputs,
    preprocess_ut_tests,
)
from data_processing.utils.check_processed_data import (  # noqa: E402
    check_ml_input_readiness,
    format_ml_readiness_report,
)


@dataclass(frozen=True)
class PipelineOutputs:
    """Outputs produced or checked by the pipeline."""

    master: BuildOutputs | None
    cavity_data: Path | None
    ddm: DDMPreprocessOutputs | Path | None
    ut: UTPreprocessOutputs | None
    readiness: dict


def _run_step(name: str, step: Callable[[], object]) -> object:
    print(f"\n=== {name} ===")
    return step()


def run_data_processing_pipeline(
    skip_master: bool = False,
    skip_cavity: bool = False,
    skip_ddm: bool = False,
    skip_ut: bool = False,
    merge_existing_ddm_parts: bool = False,
    overwrite: bool = True,
    strict_readiness: bool = False,
    require_complete_ddm_coverage: bool = False,
    verbose: bool = False,
) -> PipelineOutputs:
    """Run all processing steps and return the produced outputs."""

    master_outputs: BuildOutputs | None = None
    cavity_output: Path | None = None
    ddm_outputs: DDMPreprocessOutputs | Path | None = None
    ut_outputs: UTPreprocessOutputs | None = None

    if not skip_master:
        master_outputs = _run_step(
            "Build master sample table",
            build_master_sample_table,
        )

    if not skip_cavity:
        _run_step(
            "Preprocess cavity sensor data",
            lambda: merge_cavity_sensors_to_parquet(
                input_path=DEFAULT_CAVITY_INPUT_PATH,
                output_path=DEFAULT_CAVITY_OUTPUT_PATH,
                merged_output_path=DEFAULT_CAVITY_MERGED_OUTPUT_PATH,
                manifest_path=DEFAULT_CAVITY_MANIFEST_PATH,
                removed_cycles_csv=DEFAULT_REMOVED_CAVITY_CYCLES_CSV,
                overwrite=overwrite,
            ),
        )
        cavity_output = DEFAULT_CAVITY_MERGED_OUTPUT_PATH

    if not skip_ddm:
        if merge_existing_ddm_parts:
            _run_step(
                "Merge existing DDM part files",
                lambda: merge_ddm_part_files(
                    parts_dir=DEFAULT_DDM_OUTPUT_PATH,
                    merged_output_path=DEFAULT_DDM_MERGED_OUTPUT_PATH,
                ),
            )
            ddm_outputs = DEFAULT_DDM_MERGED_OUTPUT_PATH
        else:
            ddm_outputs = _run_step(
                "Preprocess DDM machine data",
                lambda: merge_ddm_to_parquet(
                    input_path=DEFAULT_DDM_INPUT_PATH,
                    output_path=DEFAULT_DDM_OUTPUT_PATH,
                    merged_output_path=DEFAULT_DDM_MERGED_OUTPUT_PATH,
                    manifest_path=DEFAULT_DDM_MANIFEST_PATH,
                    removed_cycles_csv=DEFAULT_REMOVED_DDM_CYCLES_CSV,
                    overwrite=overwrite,
                    require_complete_master_coverage=require_complete_ddm_coverage,
                    verbose=verbose,
                ),
            )

    if not skip_ut:
        ut_outputs = _run_step(
            "Preprocess UT test data",
            preprocess_ut_tests,
        )

    readiness = _run_step("Check processed data readiness", check_ml_input_readiness)
    print()
    print(format_ml_readiness_report(readiness))

    if strict_readiness and not readiness["ok"]:
        raise RuntimeError("Processed data readiness check failed.")

    return PipelineOutputs(
        master=master_outputs,
        cavity_data=cavity_output,
        ddm=ddm_outputs,
        ut=ut_outputs,
        readiness=readiness,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument("--skip-master", action="store_true")
    parser.add_argument("--skip-cavity", action="store_true")
    parser.add_argument("--skip-ddm", action="store_true")
    parser.add_argument("--skip-ut", action="store_true")
    parser.add_argument(
        "--merge-existing-ddm-parts",
        action="store_true",
        help="Merge existing DDM part parquet files instead of reprocessing XML files.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing part directories instead of recreating them.",
    )
    parser.add_argument(
        "--strict-readiness",
        action="store_true",
        help="Exit with an error if the final readiness check is not OK.",
    )
    parser.add_argument(
        "--require-complete-ddm-coverage",
        action="store_true",
        help="Fail DDM preprocessing if any master casting_part_label has no DDM data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file details from processing steps that support it.",
    )
    args = parser.parse_args()

    run_data_processing_pipeline(
        skip_master=args.skip_master,
        skip_cavity=args.skip_cavity,
        skip_ddm=args.skip_ddm,
        skip_ut=args.skip_ut,
        merge_existing_ddm_parts=args.merge_existing_ddm_parts,
        overwrite=not args.no_overwrite,
        strict_readiness=args.strict_readiness,
        require_complete_ddm_coverage=args.require_complete_ddm_coverage,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
