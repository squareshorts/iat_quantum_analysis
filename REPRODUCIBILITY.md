# Reproducibility

## Environment

Use the pinned environment files in `env/`:

- `env/requirements.txt`
- `env/requirements.lock.txt`

## Primary manuscript workflow

Run from the repository root:

```powershell
python run_theta_grid_full_plus_baselines.py
```

This produces the main manuscript-facing artifacts:

- `figures/theta_posterior_full.png`
- `figures/ppc_full.png`
- `tables/Table_theta_summary.tex`
- `tables/Table_fit_quality.tex`
- `tables/Table_model_comparison.tex`
- `tables/Table_theta_bins_robustness.tex`

## Optional exploratory workflows

These are not required for the main manuscript result and should be treated separately:

- `python run_hierarchical_analysis.py`
- `python run_export_participant_loglik.py`
- `python run_export_baseline_loglik.py`
- `python run_grouped_loo.py`

## Notes

- The main hold-out comparison uses temporal hold-out with leak-free participant-level standardization.
- The current RMSE sweep defaults to a participant subsample for speed (`EVAL_SUBSAMPLE_N = 5000` in `run_theta_grid_full_plus_baselines.py`).
