# Semester Project – Process-Property Prediction in Cold-Chamber Die Casting

This repository contains the code and documentation for my semester project on machine-learning-based prediction of material properties in cold-chamber die casting.

The goal is to develop a flexible modeling pipeline that predicts position-dependent tensile properties of cast aluminum parts from process parameters, machine data, and optionally sensor time series. The model should support a better understanding of process-property relationships and may later be used for process optimization.

## Project Context

The project is based on cold-chamber die-casting experiments with different machine parameter settings. For each parameter setting, cast parts are produced and tensile specimens are extracted along the runner.

The current experimental setup consists of:

- 42 process parameter settings
- 3 cast parts per setting
- 27 tensile specimens per part
- approximately 3400 tensile tests in total

Each sample may contain:

- process parameter setpoints
- recorded actual machine parameters
- tensile specimen position along the runner
- tensile-test results
- optionally: machine and sensor time-series data

The tensile-test result is a uniaxial stress-strain curve including the failure point. Instead of only predicting scalar material properties, the preferred output representation is currently the prediction of fitted Whip-Bezier parameters, since these encode the full stress-strain curve.

The pipeline should still remain flexible enough to also predict scalar target values such as:

- yield strength
- ultimate tensile strength
- elongation at failure
- other derived material properties

## Identifier

Each individual sample should be uniquely identifiable through its machine_cycle_no.
For each sample, we should know:
-- Machine_cycle_no  -> identifier through all datasets
-- Parameterset_no (with corresponding parameter values) -> excel
-- type_1st_phase   -> excel
-- datum    -> excel, cavity sensor csv, ddm xml
-- cavity sensor data   -> cavity sensor csv: file_cycle_number need to be 
-- machine data (ddm) -> ddm xml

## Main Modeling Idea

The core task is to learn a mapping of the form:

```text
process parameters + specimen position (+ sensor data)
        -> material properties / stress-strain curve parameters
```

Since the material properties are expected to vary along the flow path, the specimen position is an important input feature.

A simple first approach would be to use the specimen index from 1 to 27. However, a physically more meaningful representation may be the actual distance from the gate along the flow path, because the specimens are not necessarily equally spaced.

Possible position features:

- `sample_index`
- `distance_from_gate_mm`
- normalized relative distance from the gate
- optional segment or group ID along the runner

## Repository Structure

The repository is intended to follow this structure:

```text
.
├── data/                 # raw, interim, and processed data
├── notebooks/            # exploratory analysis and experiments
├── src/                  # main project source code
│   ├── data/             # loading, preprocessing, feature engineering
│   ├── models/           # model definitions and training logic
│   ├── evaluation/       # metrics, plots, validation logic
│   └── utils/            # helper functions
├── configs/              # experiment and model configs
├── results/              # generated predictions, metrics, plots
├── references/           # notes, papers, project documentation
└── README.md
```

## Planned Model Variants

The project should compare several model families instead of relying only on one architecture.

Possible models include:

- linear or polynomial regression baselines
- random forest
- XGBoost
- MLP-based tabular models
- sequence-to-sequence models along the runner positions
- transformer-based models
- transformer variants using sensor time-series data

The transformer-based approach is especially interesting because the output can be interpreted as a position-dependent sequence along the runner.

A possible simplified transformer setup is:

```text
input sequence:
    27 specimen positions
    + repeated process parameters at each position

output sequence:
    27 predicted material-property vectors
```

This would allow the model to predict the spatial variation of material properties along the casting.

## Sensor and Time-Series Data

The full dataset includes several time-dependent signals, for example:

- casting curve
- pressure curve
- intensification pressure
- vacuum pressure
- pressure sensors in the mold
- temperature sensor data

A likely first version of the pipeline may ignore sensor time-series data and focus only on process parameters and specimen position.

A later extension can include time-series data once the data structure and temporal alignment are better understood.

## Important Preprocessing Notes

Some process parameters are not simple scalar features. Instead, they define process curves.

Examples:

- start and end points of a linear casting curve
- coefficients or support points of a parabolic curve
- alternative parameter sets depending on the selected curve type

This means that different process settings may contain different parameters depending on the chosen curve representation.

A robust preprocessing pipeline should therefore convert heterogeneous curve-defining parameters into a unified representation before model training.

Possible approaches:

- sample every process curve onto a fixed grid
- extract curve descriptors such as mean, maximum, slope, area, or key transition points
- encode the curve type explicitly
- combine scalar parameters and sampled curves in a shared feature representation

## Target Representations

The main target representation is expected to be the Whip-Bezier parametrization of the stress-strain curve.

Advantages:

- compact representation of the full curve
- scalar tensile properties can later be derived from the reconstructed curve
- potentially more informative than predicting only yield strength or UTS

Alternative or additional targets:

- yield strength
- ultimate tensile strength
- elongation at failure
- full stress-strain curve sampled on a fixed strain grid
- failure strain
- curve-shape descriptors

## Planned Experiments

Possible experiments include:

1. process parameters only
2. process parameters + specimen index
3. process parameters + distance from gate
4. prediction of scalar tensile properties
5. prediction of Whip-Bezier parameters
6. models with and without sensor time-series data
7. transformer model compared against simpler tabular baselines
8. evaluation of generalization to unseen process parameter settings

## Evaluation and Splitting

The train/validation/test split should avoid data leakage.

Since multiple specimens come from the same process setting and cast part, random splitting over individual tensile specimens may overestimate model performance.

Possible split strategies:

- split by individual tensile specimens
- split by cast part
- split by process parameter setting
- leave-one-parameter-setting-out validation

The most relevant test is likely generalization to unseen process parameter settings.

## Current Status

The tensile-test data is still being generated. Until the full dataset is available, the focus is on:

- preparing a clean and flexible data pipeline
- defining input and output representations
- implementing baseline models
- preparing transformer-based model variants
- deciding how to represent position along the runner
- deciding how to represent curve-defining process parameters
- documenting assumptions and open questions

## Open Questions

Important open questions include:

- What is the exact file and folder structure of the final data?
- Which sensor signals are available for all experiments?
- Are machine data and mold sensor data already time-aligned?
- Should the first model version include sensor time series or only scalar process parameters?
- How should curve-defining process parameters be represented?
- Should the main model target be Whip-Bezier parameters, scalar tensile properties, or both?
- Which split strategy best reflects the intended use case?

## References

Relevant background references:

- Vaswani et al., *Attention Is All You Need*
- Grigsby et al., *Long-Range Transformers for Dynamic Spatiotemporal Forecasting*
- Sakaridis et al., Whip-Bezier stress-strain curve parametrization
- Caruso et al., *Not Another Imputation Method*, for transformer-based handling of missing tabular values

## Notes

This repository is under active development. The exact data format, available sensor signals, and final prediction targets may still change as the experimental data becomes available.
