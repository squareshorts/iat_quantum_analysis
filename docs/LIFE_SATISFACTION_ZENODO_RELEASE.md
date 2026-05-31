# Life-satisfaction IAT Zenodo Release

The database used for the life-satisfaction analyses is not the raw Testable export. The raw ZIP contains Testable metadata fields with direct identifiers, including `Nome completo`, plus raw source filenames and timestamps. It must not be uploaded publicly.

Use the de-identified release package instead:

```powershell
python scripts/build_life_satisfaction_zenodo_package.py
```

This creates:

```text
release/life_satisfaction_iat_database/
release/life_satisfaction_iat_database_zenodo.zip
```

The package contains:

- `life_satisfaction_iat_trials_deidentified.csv`
- `life_satisfaction_iat_participants_deidentified.csv`
- `life_satisfaction_iat_exclusions_deidentified.csv`
- `README.md`
- `CODEBOOK.md`
- `zenodo_metadata.json`

The de-identification removes:

- participant names
- raw Testable source filenames
- exact collection timestamps
- Testable links and browser metadata
- local participant IDs that could be mapped back to raw files

The package retains only the analysis database used by the manuscript workflow: participant placeholders, phase/block/trial structure, collapsed latencies, attempt/error counts, congruency labels, and derived participant-level summaries.

## Suggested Zenodo Record

Upload `release/life_satisfaction_iat_database_zenodo.zip` as a Zenodo **Dataset** record using the metadata in `release/life_satisfaction_iat_database/zenodo_metadata.json`.

Recommended title:

```text
De-identified life-satisfaction IAT database for angular profiling of implicit-association latency data
```

Related software archive:

```text
https://doi.org/10.5281/zenodo.19711302
```

After Zenodo mints the dataset DOI, replace the placeholders for the life-satisfaction dataset DOI in:

- `DATA_AVAILABILITY.md`
- `data_sources.yaml`
- `paper/main_new.tex`
- `CITATION.cff`
- `.zenodo.json`
- `docs/BRM_TOP_LEVEL2_AUDIT.md`

