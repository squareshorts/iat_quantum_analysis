# IAT Quantum Analysis

This repository matches the manuscript project titled:

`Angular profiling of contextual structure in implicit-association latency data`

## Data, code, and materials for BRM review

This repository has been prepared for the Behavior Research Methods TOP Level 2 data/code/materials review for manuscript BR-Org-26-540.

Repository/software sister publication on Zenodo:

- Version DOI: https://doi.org/10.5281/zenodo.19711302
- Zenodo record: https://zenodo.org/records/19711302
- GitHub repository: https://github.com/squareshorts/iat_quantum_analysis

Verified public Project Implicit raw-data links, accessed 2026-05-31:

- Gender--Science IAT: OSF Raw Data component https://osf.io/cfvyj/; 2019 archive `GenderScience_iat_2019.zip` at https://osf.io/download/7gb96/.
- Sexuality IAT: OSF Raw Data component https://osf.io/5s9ty/; 2019 archive `Sexuality_iat_2019.zip` at https://osf.io/download/79ch3/.
- Age IAT: OSF Raw Data component https://osf.io/9jvmk/; 2019 archive `Age_iat_2019.zip` at https://osf.io/download/34wsk/.

The editor-reported Gender--Science parent node `davke` is not used because the OSF API reports that node as not found. The verified Gender--Science source is the public Raw Data component at https://osf.io/cfvyj/.

The participant-level raw data from the independent life-satisfaction IAT cannot be shared openly because the consent and ethics approval for the original study did not authorize unrestricted public release of participant-level behavioral data. For code execution and review, the repository provides:

- synthetic data: `data/synthetic/life_satisfaction_iat_synthetic.csv`
- generator: `scripts/generate_synthetic_life_satisfaction_iat.py`
- validator: `scripts/validate_synthetic_life_satisfaction_iat.py`
- documentation: `docs/SYNTHETIC_DATA.md`

The actual life-satisfaction database used for analysis must not be published from the raw Testable ZIP because the raw export includes direct identifiers. Build the de-identified Zenodo upload package with:

```powershell
python scripts/build_life_satisfaction_zenodo_package.py
```

This writes `release/life_satisfaction_iat_database_zenodo.zip` and a suggested `zenodo_metadata.json`. After the dataset record is published, replace the life-satisfaction DOI placeholder in `DATA_AVAILABILITY.md`, `data_sources.yaml`, `paper/main_new.tex`, `CITATION.cff`, and `.zenodo.json`.

Reviewer commands for the public/synthetic pipeline:

```powershell
python scripts/generate_synthetic_life_satisfaction_iat.py
python scripts/validate_synthetic_life_satisfaction_iat.py
python scripts/run_external_leverage_analysis.py --use-synthetic-life-data
```

Quick smoke test without restricted data:

```powershell
python scripts/check_external_links.py
python scripts/validate_synthetic_life_satisfaction_iat.py
python scripts/run_reproducibility_smoke_test.py
```

Main public-domain figure/table generation commands:

```powershell
python scripts/run_theta_grid_full_plus_baselines.py
python scripts/run_submission_evidence.py
python scripts/run_matched_public_domain_analysis.py
python -m src.pipeline.run_age_iat --osf-node cv7iq --year 2019
python -m src.analysis.generate_manuscript_outputs
```

`scripts/run_theta_grid_full_plus_baselines.py` regenerates the primary Gender--Science posterior, posterior predictive checks, and baseline comparison tables. `scripts/run_submission_evidence.py` regenerates robustness, calibration, null, WAIC, and blockwise diagnostics. `scripts/run_matched_public_domain_analysis.py` regenerates the matched Gender--Science/Sexuality public-domain outputs. `python -m src.pipeline.run_age_iat` downloads, preprocesses, analyzes, and exports the Age IAT outputs.

The main manuscript-aligned pipeline is the grid-OLS workflow in `run_theta_grid_full_plus_baselines.py`, which:

- loads `data/GenderScience_iat_2019/iat_2019/iat*.txt`
- filters to critical blocks `3, 4, 6, 7`
- builds 6-bin participant curves
- profiles a single interference angle `theta`
- writes the manuscript-facing figures and tables in `figures/` and `tables/`

## Verified manuscript match

These repo artifacts line up with the manuscript text you supplied:

- `tables/Table_theta_summary.tex`
  - `theta mean = 17.29`, `SD = 0.48`, `94% HDI = [16.50, 18.25]`
- `tables/Table_fit_quality.tex`
  - `N participants = 141329`, `theta_MAP = 17.250`, `Residual sigma = 0.630`
- `tables/Table_model_comparison.tex`
  - temporal hold-out RMSE comparison showing that local predictive rankings and structural inference can diverge
- `tables/Table_theta_bins_robustness.tex`
  - bin-robustness table for `4 / 6 / 8` bins
- `figures/theta_posterior_full.png`
- `figures/ppc_full.png`

The repo now also includes an independent 2024 life-satisfaction IAT archive plus an external-leverage pipeline:

- `data/life_satisfaction_iat_2024/raw/187results.zip`
- `scripts/run_external_leverage_analysis.py`
  - reconstructs the 180 valid participants
  - exports anonymized cleaned trials/participants
  - compares Gender--Science geometry against D-like strata and the independent task
  - writes `figures/external_leverage_summary.png`
  - writes `tables/Table_external_task_theta.tex`, `tables/Table_theta_dlike_quintiles.tex`, and `tables/Table_external_model_comparison.tex`

## Adding the Age IAT

We now support a conventional public Age IAT domain as the recommended replacement/supplement for the life-satisfaction task. Age was selected because it is a standard social-category IAT and the public OSF project exposes a dedicated `Raw Data` component with trial-level `iat.txt` archives, unlike participant-level D-score summary exports that are insufficient for the geometric pipeline.

Run the Age pipeline with:

```powershell
python -m src.data.download_age_iat --osf-node cv7iq --year 2019
python -m src.data.prepare_age_iat --critical-blocks 3 4 6 7
python -m src.analysis.run_domain --domain Age
python -m src.pipeline.run_age_iat
```

Outputs are written to:

- `data/raw/age_iat/`
- `data/interim/age_iat/`
- `data/processed/age_iat/`
- `results/age_iat/`
- `figures/age_iat/`
- `tables/age_iat/`
- `reports/age_iat_qc_report.md`

Important warning: the analysis requires trial-level files with session/block/trial/latency fields. The public participant-level `Age IAT.public.*` summary archives and analogous D-score files cannot be used directly for the manuscript’s geometric pipeline.

## Repository layout

- `data/`
  - public IAT datasets used by the analysis
  - includes the independent life-satisfaction IAT archive under `data/life_satisfaction_iat_2024/`
- `figures/`
  - manuscript-facing figure outputs
- `outputs/`
  - caches, intermediate results, diagnostics, and exported summaries
- `paper/`
  - manuscript source scaffold created from the current draft
- `src/`
  - small reusable utilities
- `tables/`
  - manuscript-facing LaTeX tables
- `scripts/`
  - top-level executable pipelines and analysis workflows
- `archive/`
  - legacy or non-manuscript material kept for reference

## Main scripts

- `scripts/run_theta_grid_full_plus_baselines.py`
  - primary manuscript-aligned analysis
- `scripts/run_theta_grid_full.py`
  - simpler posterior/PPC run without the baseline sweep
- `scripts/run_submission_evidence.py`
  - expanded scientific-audit pipeline with corrected full-sample hold-out, block-specific analyses, null checks, recovery simulations, and approximate WAIC
- `scripts/run_external_leverage_analysis.py`
  - external-task ingestion, cleaning, subgroup contrasts against a conventional D-like metric, and independent-task model comparison
- `scripts/run_objective_mismatch_analysis.py`
  - residual-structure, held-out distributional-fit, and blockwise interpretability diagnostics contrasting the interference and quadratic baselines
- `scripts/run_hierarchical_analysis.py`
  - exploratory PyMC workflow; not the primary manuscript result
- `scripts/run_grouped_loo.py`
  - exploratory information-criterion export; currently not submission-ready

## Reproducibility

Environment files live in `env/`:

- `env/requirements.txt`
- `env/requirements.lock.txt`

Typical run order:

```powershell
python scripts/run_theta_grid_full_plus_baselines.py
python scripts/run_export_participant_loglik.py
python scripts/run_export_baseline_loglik.py
python scripts/run_grouped_loo.py
python scripts/run_external_leverage_analysis.py
python scripts/run_objective_mismatch_analysis.py
```

## Submission status

See `SUBMISSION_AUDIT.md` for the current readiness check, known mismatches, and missing items to resolve before submission.

Supporting notes:

- `DATA_AVAILABILITY.md`
- `REPRODUCIBILITY.md`
- `SCIENTIFIC_AUDIT.md`
