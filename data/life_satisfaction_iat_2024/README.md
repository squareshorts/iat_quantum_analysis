# Life Satisfaction IAT (2024)

This directory stores the independent Testable archive used for the manuscript's external-leverage analyses.

## Source archive

- `raw/187results.zip`
  - original export shared by the authors
  - contains 187 per-participant CSV files
  - each raw file includes the Testable metadata header and trial-level rows
  - contains direct identifiers in the Testable metadata header, including `Nome completo`
  - must not be uploaded to Zenodo or included in a public repository

## Task structure

The task has five phases:

1. `fase1`: 10 true/false practice items
2. `fase2`: 10 self/not-self practice items
3. `fase3`: 20 congruent combined trials
4. `fase4`: 10 self/not-self practice items with reversed response mapping
5. `fase5`: 20 incongruent combined trials

Incorrect responses are repeated until corrected in the raw logs. The processing pipeline collapses repeated attempts back to the original item and uses the summed latency across attempts as the item's total time cost.

## Cleaning rule used in the repo

`run_external_leverage_analysis.py` reconstructs the 180 valid participants described in the project notes by:

- keeping only the first attempt for repeated participant names
- excluding participants with more than 10% of raw attempt latencies below 300 ms

This removes:

- 1 repeated later attempt
- 6 fast-response invalid cases

Processed, anonymized outputs are written to `outputs/`:

- `external_life_satisfaction_participants_clean.csv`
- `external_life_satisfaction_trials_clean.csv`
- `external_life_satisfaction_excluded_cases.csv`

## Zenodo dataset release candidate

The raw Testable export is not publishable as-is. To publish the database used for the analyses, first build the de-identified release package:

```powershell
python scripts/build_life_satisfaction_zenodo_package.py
```

The script removes raw source filenames, exact timestamps, direct identifiers, and local participant IDs. It writes:

- `release/life_satisfaction_iat_database/`
- `release/life_satisfaction_iat_database_zenodo.zip`

Upload the ZIP to Zenodo as a dataset record. After Zenodo mints the DOI, add the dataset DOI to `DATA_AVAILABILITY.md`, `data_sources.yaml`, `paper/main_new.tex`, `CITATION.cff`, and `.zenodo.json`.
