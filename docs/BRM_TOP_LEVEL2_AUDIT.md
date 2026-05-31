# BRM TOP Level 2 Audit

Manuscript: BR-Org-26-540, "Angular profiling of contextual structure in implicit-association latency data"

Audit date: 2026-05-31

## Summary

The repository now distinguishes three kinds of materials:

- Public Project Implicit raw datasets hosted externally on OSF.
- The GitHub repository's Zenodo software/materials sister archive: https://doi.org/10.5281/zenodo.19711302.
- The independent life-satisfaction IAT database, which has been prepared as a de-identified Zenodo-ready dataset package. The raw Testable export is not publishable as-is because it contains direct identifiers.

## Editor Concerns And Fixes

| Concern | Resolution | Files changed |
| --- | --- | --- |
| Broken Gender--Science OSF parent link | Removed the broken `davke` node as a link and replaced the source with the verified public Raw Data component. | `README.md`, `DATA_AVAILABILITY.md`, `data_sources.yaml`, `paper/main_new.tex` |
| Direct Gender--Science data source required | Verified OSF Raw Data component and 2019 archive. | `README.md`, `DATA_AVAILABILITY.md`, `data_sources.yaml`, `paper/main_new.tex` |
| Sexuality IAT link missing from manuscript declarations | Added verified Sexuality OSF Raw Data component and 2019 archive to repository documentation and manuscript data statement. | `README.md`, `DATA_AVAILABILITY.md`, `data_sources.yaml`, `paper/main_new.tex` |
| Age IAT link missing | Added verified Age OSF Raw Data component and 2019 archive to repository documentation and manuscript data statement. | `README.md`, `DATA_AVAILABILITY.md`, `data_sources.yaml`, `paper/main_new.tex` |
| Need DATA_AVAILABILITY.md table | Replaced placeholder text with a dataset-by-dataset TOP Level 2 table. | `DATA_AVAILABILITY.md` |
| Life-satisfaction raw data decision | Raw Testable ZIP contains `Nome completo` and must not be published as-is. A de-identified Zenodo-ready database package is built from the analysis outputs. | `scripts/build_life_satisfaction_zenodo_package.py`, `docs/LIFE_SATISFACTION_ZENODO_RELEASE.md`, `data/life_satisfaction_iat_2024/README.md` |
| Synthetic data required for review | Added deterministic synthetic life-satisfaction data, generator, validator, and documentation. | `data/synthetic/life_satisfaction_iat_synthetic.csv`, `scripts/generate_synthetic_life_satisfaction_iat.py`, `scripts/validate_synthetic_life_satisfaction_iat.py`, `docs/SYNTHETIC_DATA.md` |
| Analysis scripts need public/synthetic option | Added `--use-synthetic-life-data` and `--synthetic-life-data` to the external life-satisfaction analysis script. | `scripts/run_external_leverage_analysis.py` |
| Smoke test required | Added a public/synthetic smoke test that does not require restricted raw life-satisfaction data. | `scripts/run_reproducibility_smoke_test.py` |
| Link checker required | Added link checker with OSF API validation for OSF node links. | `scripts/check_external_links.py` |
| README review section required | Added BRM review section with verified links, Zenodo software archive, synthetic workflow, and reviewer commands. | `README.md` |
| Manuscript declarations/open-practices updates required | Updated data availability, code availability, and Open Practices Statement. | `paper/main_new.tex` |
| Zenodo/CITATION metadata required | Added metadata for the repository/software archive and citation file. | `.zenodo.json`, `CITATION.cff` |

## Verified Dataset URLs

| Dataset | Verified URL | Type | Status |
| --- | --- | --- | --- |
| Gender--Science IAT | https://osf.io/cfvyj/ | OSF Raw Data component | Public |
| Gender--Science IAT 2019 archive | https://osf.io/download/7gb96/ | File-level OSF download | Public |
| Sexuality IAT | https://osf.io/5s9ty/ | OSF Raw Data component | Public |
| Sexuality IAT 2019 archive | https://osf.io/download/79ch3/ | File-level OSF download | Public |
| Age IAT | https://osf.io/9jvmk/ | OSF Raw Data component | Public |
| Age IAT 2019 archive | https://osf.io/download/34wsk/ | File-level OSF download | Public |
| Repository software archive | https://doi.org/10.5281/zenodo.19711302 | Zenodo software record | Public |

The editor-reported `davke` OSF node was checked through the OSF API and returned `404 Not found`; it is not used as a dataset link.

## Life-satisfaction Database Status

The raw local Testable export is not de-identified: its metadata header includes `Nome completo`, and the intermediate clean outputs included raw source filenames and exact timestamps. Those fields are excluded from the Zenodo-ready package.

Prepared release package:

- Directory: `release/life_satisfaction_iat_database/`
- Upload ZIP: `release/life_satisfaction_iat_database_zenodo.zip`
- Builder: `scripts/build_life_satisfaction_zenodo_package.py`
- Release documentation: `docs/LIFE_SATISFACTION_ZENODO_RELEASE.md`

The de-identified package contains:

- `life_satisfaction_iat_trials_deidentified.csv`
- `life_satisfaction_iat_participants_deidentified.csv`
- `life_satisfaction_iat_exclusions_deidentified.csv`
- `README.md`
- `CODEBOOK.md`
- `zenodo_metadata.json`

The package removes participant names, raw Testable source filenames, exact timestamps, Testable links, browser metadata, and local IDs that could map release rows back to raw files. After the Zenodo dataset DOI is minted, replace `TBD_AFTER_ZENODO_PUBLICATION` in the repository materials.

## Link-check Output

```text
URL | HTTP status | redirect target | pass/fail
--- | --- | --- | ---
https://doi.org/10.5281/zenodo.19711302 | 200 | https://zenodo.org/records/19711302 | PASS
https://github.com/squareshorts/iat_quantum_analysis | 200 | https://github.com/squareshorts/iat_quantum_analysis | PASS
https://osf.io/5s9ty/ | 200 | https://osf.io/5s9ty/ | PASS
https://osf.io/9jvmk/ | 200 | https://osf.io/9jvmk/ | PASS
https://osf.io/cfvyj/ | 200 | https://osf.io/cfvyj/ | PASS
https://osf.io/download/34wsk/ | 200 | https://files.osf.io/v1/resources/9jvmk/providers/osfstorage/62acb38854dfbe11ae5bbdb3 | PASS
https://osf.io/download/79ch3/ | 200 | https://files.osf.io/v1/resources/5s9ty/providers/osfstorage/62d3b5dec79a4c2af99e5ab5 | PASS
https://osf.io/download/7gb96/ | 200 | https://files.osf.io/v1/resources/cfvyj/providers/osfstorage/62bb6ff4cc0ff501758720d7 | PASS
https://zenodo.org/records/19711302 | 200 | https://zenodo.org/records/19711302 | PASS
```

## Verification Commands

These commands passed on 2026-05-31:

```powershell
python scripts/check_external_links.py
python scripts/validate_synthetic_life_satisfaction_iat.py
python scripts/run_reproducibility_smoke_test.py
```

Additional command used to prepare the de-identified life-satisfaction Zenodo package:

```powershell
python scripts/build_life_satisfaction_zenodo_package.py
```

## Reviewer Commands

Public/synthetic review path:

```powershell
python scripts/generate_synthetic_life_satisfaction_iat.py
python scripts/validate_synthetic_life_satisfaction_iat.py
python scripts/run_external_leverage_analysis.py --use-synthetic-life-data
```

Main public-domain analysis commands:

```powershell
python scripts/run_theta_grid_full_plus_baselines.py
python scripts/run_submission_evidence.py
python scripts/run_matched_public_domain_analysis.py
python -m src.pipeline.run_age_iat --osf-node cv7iq --year 2019
python -m src.analysis.generate_manuscript_outputs
```

