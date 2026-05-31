# Data Availability

Access date for all verified public links: 2026-05-31.

The GitHub repository and manuscript software/materials archive is published as a Zenodo software record at https://doi.org/10.5281/zenodo.19711302. This record is the sister publication for the GitHub repository, not a replacement for the external raw-data sources listed below.

| Dataset | Role in manuscript | Raw data source | Direct public link | Access status | Access date | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Gender--Science IAT | Primary public corpus | Project Implicit / OSF, Raw Data component | https://osf.io/cfvyj/ | Public | 2026-05-31 | Used critical blocks 3, 4, 6, and 7; 2019 raw archive is `GenderScience_iat_2019.zip` at https://osf.io/download/7gb96/. The broken `davke` node is not used. |
| Sexuality IAT | Matched public comparison | Project Implicit / OSF, Raw Data component | https://osf.io/5s9ty/ | Public | 2026-05-31 | Used 2019 raw archive `Sexuality_iat_2019.zip` at https://osf.io/download/79ch3/; `iat2.0003.txt` was excluded as malformed if applicable. |
| Age IAT | Additional public comparison | Project Implicit / OSF, Raw Data component | https://osf.io/9jvmk/ | Public | 2026-05-31 | Used 2019 raw archive `Age_iat_2019.zip` at https://osf.io/download/34wsk/; critical-block latencies were cleaned to 300--10,000 ms. |
| Life-satisfaction IAT | Independent external comparison | Original Testable study | Dataset DOI pending Zenodo publication | Pending public de-identified release | 2026-05-31 | The raw Testable export is not publishable as-is because it contains direct identifiers. A de-identified database package has been prepared at `release/life_satisfaction_iat_database_zenodo.zip`; replace this placeholder with the Zenodo dataset DOI after publication. Synthetic review data are provided in `data/synthetic/life_satisfaction_iat_synthetic.csv`. |

## Public And Synthetic Review Workflow

The public Project Implicit datasets are available without login through the OSF links above. The life-satisfaction participant-level raw data are not included for public release. Reviewers can run the life-satisfaction code path with synthetic data:

```powershell
python scripts/generate_synthetic_life_satisfaction_iat.py
python scripts/validate_synthetic_life_satisfaction_iat.py
python scripts/run_external_leverage_analysis.py --use-synthetic-life-data
```

The quick reviewer smoke test avoids restricted data and verifies the public/synthetic execution path:

```powershell
python scripts/check_external_links.py
python scripts/validate_synthetic_life_satisfaction_iat.py
python scripts/run_reproducibility_smoke_test.py
```

Build the de-identified life-satisfaction database package for Zenodo with:

```powershell
python scripts/build_life_satisfaction_zenodo_package.py
```

After the de-identified package is published on Zenodo, cite the minted dataset DOI here and in `paper/main_new.tex`.
