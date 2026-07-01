# Casting Project: Technical Documentation

This document describes the repository as it exists now: its purpose, the data it contains, the implemented workflows, and the role of the individual files. It is intended as the detailed companion to `README.md`. The README explains the project at a high level; this file records how the repository actually works.

## 1. Project purpose

This repository supports a semester project on process-property prediction in cold-chamber die casting. The long-term goal is to predict position-dependent material properties of cast aluminium parts from process settings, recorded machine data, specimen position, and potentially sensor time series.

The intended mapping is:

```text
process parameters + position along the casting + optional sensor data
    -> tensile properties or a representation of the stress-strain curve
```

The preferred full-curve target is currently a Whip-Bezier parameterization of the tensile stress-strain curve. Scalar targets such as yield strength, ultimate tensile strength, and elongation at failure may also be predicted or derived.

At the current stage, the repository is primarily concerned with:

- storing and understanding the experimental data;
- converting raw cavity-sensor CSV files into an analysis-friendly Parquet dataset;
- reading, inspecting, plotting, and exporting Buehler machine XML data;
- retaining existing utilities for design-of-experiments generation and shot-curve calculation;
- fitting stress-strain/yield curves in MATLAB; and
- preparing the data foundation for a later machine-learning pipeline.

Model training, feature assembly across all data sources, and a final joined sample-level dataset have not yet been implemented.

## 2. Experimental data model

According to the current project design, the experiment contains 42 process parameter settings, three cast parts per setting, and 27 tensile specimens per part, resulting in roughly 3,400 tensile tests.

The intended central identifier is `MachineCycleNr` (also referred to as `machine_cycle_no`). It should eventually connect:

- the parameter setting and its configured values;
- the type of first-phase casting curve;
- the trial date;
- cavity-sensor measurements;
- Buehler DDM machine data; and
- the tensile specimens and their measured properties.

This end-to-end join does not exist yet. In particular, cavity-sensor `CycleNr` values are only unique within an individual cavity-sensor file or trial day. The preprocessing code therefore preserves source-file and date information so that cycles can be matched safely later.

## 3. Current repository layout

```text
casting_project/
|-- README.md
|-- Documentation.md
|-- requirements.txt
|-- data/
|   |-- 250905/ ... 250912/       Raw data grouped by trial date
|   |-- cavity_sensor_file_list.csv
|   |-- ddm_y_FliesslaengerformBuehler_0080_ok.xml.gz
|   `-- 250908-250912_ParameterStudy.zip
|-- data_processing/
|   |-- preprocess_cavity_sensors.py
|   `-- read_xml_file.py
|-- processed/
|   |-- cavity_sensors_long/
|   `-- cavity_sensors_manifest.parquet
|-- MATLAB files/
|   |-- MATLAB_README.md
|   |-- fit/
|   `-- yield_curve/
`-- utils - bueler_nils/
    |-- DOE.py
    |-- DOE_generator.py
    |-- sampleGenerator.py
    |-- shot_curve_util.py
    |-- convert_xml_gz_to_csv.py
    |-- plot_xml_gz_file.py
    |-- merge_ts_tv_to_sv.py
    `-- exploratory notebooks
```

The layout currently differs from the aspirational `src/`, `notebooks/`, `configs/`, and `results/` structure described in the README. The tree above represents what is actually present and usable now.

## 4. Current workflow

### 4.1 Environment setup

From the repository root, create and activate a virtual environment and install the declared packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The current `requirements.txt` declares only:

- `pandas`
- `pyarrow`

These are sufficient for cavity-sensor preprocessing. The XML plotting and numerical utility modules additionally import packages such as `matplotlib`, `numpy`, and `scipy`; the DOE utilities may also require `seaborn`. These additional dependencies are not yet captured in `requirements.txt`.

### 4.2 Add raw trial data

Raw experiment data is stored below `data/`, grouped into date folders using the `YYMMDD` format. A typical trial folder contains some of the following:

```text
data/<trial-date>/
|-- 01_Alloy/
|-- 02_Machine data/
|   |-- ddd/                  General compressed machine XML files
|   |-- ddm/                  Per-cycle Buehler XML/XML.GZ files
|   |-- dtr/                  Trend data
|   |-- dwa/                  Per-cycle text exports
|   |-- err/                  Error/report archives
|   |-- pro/                  Machine/program configuration exports
|   `-- psc/                  Machine screenshots
|-- 03_Data analysis/
|-- 05_Protocol/
`-- 06_Cavity_Sensors/        Semicolon-separated cavity-sensor CSV files
```

Not every day necessarily contains every folder. Protocol spreadsheets and filenames vary between trial dates, so consumers should not assume a single exact workbook name.

### 4.3 Preprocess cavity-sensor data

Run the script from the repository root:

```powershell
python data_processing\preprocess_cavity_sensors.py
```

The active processing function searches for:

```text
data/*/06_Cavity_Sensors/*.csv
```

Each source CSV is semicolon-separated. It contains one row per sensor/channel/cycle and many time-sample columns whose headers encode time in seconds, sometimes using a decimal comma. `read_cavity_sensor_csv()` performs the following steps:

1. verifies that all expected metadata columns exist;
2. identifies sample columns by excluding the known metadata columns;
3. converts decimal-comma time headers to numeric seconds;
4. ignores and reports invalid time columns;
5. adds source-file, relative-path, source ID, and trial-folder provenance;
6. parses `Timestamp` values using `YYYYMMDDTHHMMSS.ffffff`;
7. preserves the file-local cycle as `cavity_sensor_file_cycle_nr`;
8. reshapes the wide sample table into long format; and
9. removes samples whose time or value cannot be parsed numerically.

The result is written as one Parquet part per source CSV beneath:

```text
processed/cavity_sensors_long/
```

The long table retains the original sensor metadata and adds these important columns:

- `source_file_id`: sequential ID assigned to the source CSV during a run;
- `source_file`: source filename;
- `source_rel_path`: path relative to `data/`;
- `trial_folder`: trial-date directory name;
- `cavity_sensor_file_cycle_nr`: explicit copy of file-local `CycleNr`;
- `cavity_sensor_datetime`: parsed measurement timestamp;
- `cavity_sensor_date`: date string for later matching;
- `time_s`: numeric time within the measurement;
- `value`: numeric sensor value.

A separate file, `processed/cavity_sensors_manifest.parquet`, contains one summary row per input CSV, including its source identity, row count, signal descriptions, cycle range, and time range.

The output directory is deleted and rebuilt by default (`overwrite=True`). Do not place manually edited files inside `processed/cavity_sensors_long/`.

The script also calls `merge_cavity_sensor()`, but that function currently only discovers and prints the input files; it does not create a merged CSV. The Parquet workflow in `merge_cavity_sensors_to_parquet()` is the implemented data product.

### 4.4 Inspect Buehler machine XML data

`data_processing/read_xml_file.py` reads both `.xml` and gzip-compressed `.xml.gz` Buehler files. It can be used as a module or command-line utility.

Example without opening a plot:

```powershell
python data_processing\read_xml_file.py "data\ddm_y_FliesslaengerformBuehler_0080_ok.xml.gz" --no-plot
```

Filter curves by a case-insensitive substring:

```powershell
python data_processing\read_xml_file.py <path-to-xml.gz> --pattern pressure
```

Without `--no-plot`, the command shows either the selected curves or a three-panel casting overview containing the available stroke, velocity, machine/metal pressure, and vacuum signals.

Typical notebook or Python usage:

```python
from data_processing.read_xml_file import (
    export_xml_to_csv,
    load_xml_data,
    plot_casting_overview,
    show_xml_contents,
)

xml_data = load_xml_data("data/example.xml.gz")
summary = show_xml_contents(xml_data)
plot_casting_overview(xml_data)
export_xml_to_csv(xml_data, "processed/example.csv")
```

The module represents a file as an `XMLData` object containing file metadata and a list of `XMLCurve` objects. Each curve contains names, units, metadata, and a pandas DataFrame with at least `time_us`, normalized `time_s`, and `value`.

Available operations include:

- selecting curves by substring or exact short/long name;
- printing curve counts, duration, minima, maxima, and means;
- converting curves to long format;
- aligning curves into a wide table using absolute microsecond timestamps;
- optionally interpolating gaps inside individual signal ranges;
- exporting a wide CSV;
- plotting arbitrary signals or the standard casting overview; and
- converting between stroke, velocity, time, and acceleration numerically.

This XML workflow currently operates on one file at a time. It is not yet connected to the cavity-sensor Parquet dataset and does not batch-extract features from all DDM files.

### 4.5 Fit and reconstruct stress-strain curves in MATLAB

`MATLAB files/` contains utilities for fitting Whip-Bezier curves to prepared stress-strain data and for reconstructing curves from fitted parameters. It also contains a combined Swift-Voce hardening-law implementation used as a reference or synthetic curve.

The main fitting entry points are:

- `fit/least_sq_main.m`: standard least-squares fitting workflow;
- `fit/least_sq_main_flex.m`: flexible variant that accepts a data matrix directly; and
- `fit/bezier_least_sq_export_iter.m`: helper intended for repeated fitting/export.

Supporting functions define initial parameters, bounds, the Swift-Voce curve, Bezier control points, interval subdivision, curve evaluation, and least-squares objectives. The fitted parameter vector is ordered as:

```text
[Rp, alpha, p_u, b_u]
```

where `Rp` is the proof/yield stress, `alpha` controls the initial tangent behavior, `p_u` is the plastic strain at the last point, and `b_u` is the stress at the last point.

The separate `yield_curve/` folder reconstructs a yield curve from parameters, including logic to divide the strain interval and calculate Whip-Bezier control points.

See `MATLAB files/MATLAB_README.md` for detailed MATLAB usage, parameter conventions, examples, and limitations. At present, fitting is an interactive or per-dataset workflow rather than an automated batch stage connected to Python.

## 5. File and directory reference

### Root files

#### `README.md`

High-level project description. It records the scientific motivation, intended modeling problem, possible model families, target representations, split strategies, and open research questions. It is forward-looking and should remain shorter than this technical document.

#### `Documentation.md`

This file. It describes the implemented state of the repository, concrete workflows, file responsibilities, data formats, and known limitations. Update it when repository behavior or structure changes.

#### `requirements.txt`

Python dependency list. It currently contains `pandas` and `pyarrow`, which support cavity CSV ingestion and Parquet output. It does not yet fully describe dependencies used by all utilities.

### `data/`

Holds raw or source experimental data. Most contents are organized by trial date. The data includes protocol workbooks, alloy information, machine exports, cavity-sensor tables, images, and archives.

`data/cavity_sensor_file_list.csv` is a simple inventory of the seven currently discovered cavity-sensor CSV files. Its `path` column contains machine-specific absolute paths, so it is an inventory rather than a portable pipeline input. The preprocessing script discovers files from the directory tree directly.

The root-level sample `ddm_y_FliesslaengerformBuehler_0080_ok.xml.gz` can be used to test XML parsing and plotting.

### `processed/`

Contains generated, analysis-friendly data. Currently this is the partitioned long-format cavity-sensor Parquet dataset and its manifest. These files can be loaded together with pandas/pyarrow as a Parquet dataset or one part at a time.

### `data_processing/preprocess_cavity_sensors.py`

Current cavity-sensor ingestion pipeline. Its reusable public functions are:

- `read_cavity_sensor_csv()`: read and reshape one CSV;
- `merge_cavity_sensors_to_parquet()`: discover all matching CSVs and build the partitioned output; and
- `merge_cavity_sensor()`: discovery/printing stub, not an actual merge operation yet.

The script currently contains duplicate `Path` and `pandas` imports and a hard-coded local path in its `__main__` block. It then replaces that path with a repository-relative path before performing the actual Parquet conversion, so the final conversion remains repository-relative. This should eventually be simplified into a command-line interface.

### `data_processing/read_xml_file.py`

General-purpose Buehler XML/XML.GZ reader and analysis module. This is the newer, more capable replacement for the small XML conversion/plotting scripts in `utils - bueler_nils/`.

### `utils - bueler_nils/`

Imported research and experiment-design utilities. They are useful references and some are executable, but they are not currently wired into a single package or the main preprocessing workflow.

#### `DOE.py`

Generates discrete split-plot designs. It separates hard-to-change parameters from free parameters, selects hard-to-change setups with maximin spacing, orders them to reduce setup changes, samples free variables with Sobol sequences, and supports uniform, log-uniform, and binned-normal parameter distributions. `extend_doe()` adds dependent columns from formulas.

#### `DOE_generator.py`

An alternative and more experiment-specific split-plot DOE generator. It uses maximin selection for hard-to-change variables, nearest-neighbor ordering, Sobol sampling for free variables, constant values, and formula-based dependent variables. Its executable example derives casting parameters with `fillingCurvePoints`.

#### `sampleGenerator.py`

Reusable Monte Carlo and Latin Hypercube samplers. `SampleGenerator` transforms unit-uniform samples into uniform or normal distributions and supports scalar or per-feature distribution parameters.

#### `shot_curve_util.py`

Defines `fillingCurvePoints`, a physical/numerical helper for creating plunger velocity profiles. It calculates shot-sleeve and die-filling positions from geometry and mass assumptions and supports three first-phase curve modes: Nogowizin, Buehler, and no optimization. It joins these to fast-shot and braking phases, calculates time from stroke and velocity, and reduces the curve using the Ramer-Douglas-Peucker algorithm.

The class assumes SI units internally, while some inputs can be recognized and converted from millimetres. Unit handling should be checked carefully when integrating it into a data pipeline.

#### `convert_xml_gz_to_csv.py`

Legacy single-file compressed-XML-to-CSV converter. It can filter curves by a text pattern. Prefer `data_processing/read_xml_file.py` for new work because the newer module preserves more structure and provides long/wide conversion, inspection, and plotting.

#### `plot_xml_gz_file.py`

Legacy single-file XML plotting helper. Prefer the plotting functions in `data_processing/read_xml_file.py` for new work.

#### `merge_ts_tv_to_sv.py`

Combines a time-stroke series and a time-velocity series to obtain a stroke-velocity representation, with optional axis swapping.

#### Notebooks

- `DoE_Cast.ipynb`, `DOE_script.ipynb`, and `DOE_generator.py`-related work explore experiment generation.
- `generate_shot_curve_points.ipynb` explores shot-curve construction.

Notebook outputs and assumptions may be exploratory; reusable behavior should ultimately be moved into maintained Python modules.

## 6. What is implemented and what is not

### Implemented

- Discovery of cavity-sensor CSVs across trial-date folders.
- Validation and long-format conversion of cavity-sensor signals.
- Provenance-aware, partitioned Parquet output plus a manifest.
- Parsing of individual Buehler XML and XML.GZ machine files.
- XML curve discovery, summaries, filtering, plotting, dataframe conversion, and CSV export.
- Numerical stroke/velocity/time conversion helpers.
- MATLAB Whip-Bezier fitting and curve reconstruction utilities.
- Several DOE and shot-curve generation approaches.

### Not yet implemented as an integrated workflow

- A single canonical table joining protocol, process-setting, machine, cavity-sensor, specimen-position, and tensile-test data.
- Reliable mapping from file-local cavity `CycleNr` to the global `MachineCycleNr` for every trial.
- Batch processing and feature extraction for all machine XML files.
- Automated extraction and ingestion of protocol spreadsheets.
- Automated Python-to-MATLAB or batch Whip-Bezier target generation.
- A finalized specimen-position representation.
- Train/validation/test dataset construction with leakage-safe grouping.
- Baseline or transformer model training and evaluation.
- Automated tests and continuous integration.
- A complete, pinned Python environment definition.

## 7. Data-handling conventions and cautions

- Treat raw files under `data/` as source material. Derived tables should go under `processed/` or a future generated-output directory.
- Preserve `MachineCycleNr` wherever it exists; it is intended to become the cross-source key.
- Never treat cavity `CycleNr` as globally unique. Pair it with the source/trial date until a verified global mapping exists.
- Retain provenance columns when reshaping or aggregating sensor data.
- Do not randomly split individual tensile specimens without considering their shared cast part and parameter setting. Such a split can leak closely related samples between training and evaluation.
- Be explicit about units. The repository contains millimetres and metres, seconds and microseconds, bar and mbar, and code that assumes SI units internally.
- Raw data naming is not completely uniform across days. Discover files by folder role and extension where possible, then validate their schemas.
- Some files originated as research utilities and contain local paths or executable examples. Review configuration before running them against new data.

## 8. Recommended next integration steps

The most natural continuation from the present state is:

1. clean the cavity preprocessing entry point and make input/output paths command-line arguments;
2. build inventories for protocol, DDM, and tensile-test files;
3. establish and validate the mapping to `MachineCycleNr` across trial dates;
4. batch-extract selected DDM metadata and signals into a provenance-aware dataset;
5. define a canonical sample table and specimen-position representation;
6. automate Whip-Bezier fitting or target import for all tensile tests;
7. add data-quality checks and tests around identifiers, expected cycles, missing signals, units, and joins; and
8. only then construct leakage-safe model datasets and baselines.

## 9. Documentation maintenance

When changing the repository:

- update `README.md` when the project goal, major capability, or intended architecture changes;
- update this file when concrete commands, file responsibilities, schemas, dependencies, or workflows change;
- use `running_changelog.md` for informal daily notes about work performed; and
- keep comments close to code for low-level implementation details that would become stale here.

This separation keeps the README approachable, this document operational, and the running changelog chronological.
