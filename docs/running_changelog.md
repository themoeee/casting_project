# Running Changelog

Informal working log for the casting project. This does not need to be polished.
The point is simply to quickly write down what happened, what the project can
currently do, and what is still open.

## How I want to use this

After larger changes, after a commit, or at the end of a working day, add a new
entry at the top of the timeline.

A good entry can look like this:

```text
## YYYY-MM-DD - Short description

### What happened
- ...

### What the project can do now
- ...

### Open / next step
- ...

### Commit / files
- Commit: <hash or "not committed yet">
- Important: <files/folders>
```

## Current state in short

The project is currently mainly a data and preprocessing foundation for the
semester project on process-property prediction in cold-chamber die casting.

What already exists:

- README with the project idea, data model, model ideas, and open research questions.
- Technical documentation describing the actual repository state and usable workflows.
- Raw data structure under `data/` with trial days, protocols, machine data, and cavity-sensor data.
- Cavity-sensor preprocessing from CSV to long-format Parquet, including a manifest.
- Buehler XML/XML.GZ reader for machine data, including curve summaries, plotting, and CSV export.
- DDD preprocessing to long-format Parquet, including a manifest.
- XML inspection script for DDD/DDM files, writing file, curve, and header summaries.
- MATLAB reference code for Whip-Bezier fitting and reconstruction of stress-strain curves.
- DOE, sample, and shot-curve utilities from the Buehler/Nils context as reference material.

What the project does not yet provide as a finished end-to-end workflow:

- No central sample table yet combining protocol, parameters, `MachineCycleNr`, sensors, position, and tensile-test data.
- No robust join yet between file-local cavity `CycleNr` values and the global `MachineCycleNr`.
- No batch feature extraction yet for all DDM/DDD signals as ML features.
- No automated Whip-Bezier target generation yet for all tensile tests.
- No baseline or transformer models yet.
- No tests/CI yet, and no fully pinned Python environment yet.

## Next rough to-dos

- Complete `requirements.txt`, because some tools need more than `pandas` and `pyarrow`.
- Turn cavity preprocessing into a clean CLI instead of relying on implicit paths inside the script.
- Create inventories for protocols, DDM, DDD, cavity sensors, and later tensile-test files.
- Clarify and validate the mapping to `MachineCycleNr`.
- Plan DDM/DDD batch feature extraction: which curves, which features, which units.
- Define a canonical sample table.
- Decide on the specimen-position representation: index, distance from gate, normalized position.
- Automatically import or fit Whip-Bezier target values.
- Only then build ML datasets, splits, and baselines.

## Timeline

## 2026-06-26 - Running changelog created

### What happened

- Structured `running_changelog.md` as an informal working log.
- Collected the rough current project state from the README, technical documentation, and existing processing scripts.
- Added a simple to-do list as a reference for the next steps.

### What the project can do now

- Project progress can now be written down quickly and chronologically.
- There is now a fixed place to record "what was done", "what it can do now", and "what is still open" when committing or wrapping up a day.

### Open / next step

- Add older commits as individual historical entries if useful.
- Add the next changelog entry at the top of the timeline when making the next commit.

### Commit / files

- Commit: not committed yet
- Important: `running_changelog.md`

## Historical orientation from previous commits

### What happened so far

- Project initialized.
- Buehler/Nils utilities and MATLAB files added as reference material.
- First data preprocessing work for cavity-sensor CSV files started.
- Data, README/documentation, and useful Python files added.

### Known commits

- `b9a8cdc` / `c9d9f70`: initial project states.
- `30b250c`: files from the Buehler Other GitLab repository added.
- `d041db9`: MATLAB files added as reference material.
- `e62e09e`: start of CSV cavity-sensor preprocessing.
- `c2b9ad4`: first laptop commit with data preprocessing.
- `b21c861`: data, README/documentation, and useful Python files added.
