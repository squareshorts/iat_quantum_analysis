# IAT Quantum Analysis

This repository matches the manuscript project titled:

`A single-parameter geometric model of contextual overlap in the Gender--Science IAT`

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
- `run_external_leverage_analysis.py`
  - reconstructs the 180 valid participants
  - exports anonymized cleaned trials/participants
  - compares Gender--Science geometry against D-like strata and the independent task
  - writes `figures/external_leverage_summary.png`
  - writes `tables/Table_external_task_theta.tex`, `tables/Table_theta_dlike_quintiles.tex`, and `tables/Table_external_model_comparison.tex`

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
- `archive/`
  - legacy or non-manuscript material kept for reference

## Main scripts

- `run_theta_grid_full_plus_baselines.py`
  - primary manuscript-aligned analysis
- `run_theta_grid_full.py`
  - simpler posterior/PPC run without the baseline sweep
- `run_submission_evidence.py`
  - expanded scientific-audit pipeline with corrected full-sample hold-out, block-specific analyses, null checks, recovery simulations, and approximate WAIC
- `run_external_leverage_analysis.py`
  - external-task ingestion, cleaning, subgroup contrasts against a conventional D-like metric, and independent-task model comparison
- `run_objective_mismatch_analysis.py`
  - residual-structure, held-out distributional-fit, and blockwise interpretability diagnostics contrasting the interference and quadratic baselines
- `run_hierarchical_analysis.py`
  - exploratory PyMC workflow; not the primary manuscript result
- `run_grouped_loo.py`
  - exploratory information-criterion export; currently not submission-ready

## Reproducibility

Environment files live in `env/`:

- `env/requirements.txt`
- `env/requirements.lock.txt`

Typical run order:

```powershell
python run_theta_grid_full_plus_baselines.py
python run_export_participant_loglik.py
python run_export_baseline_loglik.py
python run_grouped_loo.py
python run_external_leverage_analysis.py
python run_objective_mismatch_analysis.py
```

## Submission status

See `SUBMISSION_AUDIT.md` for the current readiness check, known mismatches, and missing items to resolve before submission.

Supporting notes:

- `DATA_AVAILABILITY.md`
- `REPRODUCIBILITY.md`
- `SCIENTIFIC_AUDIT.md`
