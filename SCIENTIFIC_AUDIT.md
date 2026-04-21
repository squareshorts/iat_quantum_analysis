# Scientific Audit

This file summarizes the additional numerical checks implemented in
`run_submission_evidence.py`.

## New analysis bundle

The script produces:

- corrected full-sample temporal hold-out comparisons across `4 / 6 / 8` bins
- block-specific pooled `theta` summaries
- a direct block-specific-theta competitor on participant-block curves
- negative controls
- simulation-based parameter recovery
- additive-null simulations
- an approximate pointwise WAIC comparison

Primary outputs are written to:

- `outputs/model_comparison_submission.csv`
- `outputs/block_theta_summary.csv`
- `outputs/block_model_comparison.csv`
- `outputs/block_phase_summary.csv`
- `outputs/negative_controls_summary.csv`
- `outputs/theta_recovery_simulation.csv`
- `outputs/theta_null_simulation.csv`
- `outputs/waic_comparison_submission.csv`
- `tables/Table_*_submission.tex`

## What now holds up

- The pooled six-bin `theta` result reproduces the manuscript-facing estimate:
  - mean `17.29`
  - SD `0.48`
  - `94%` HDI `[16.50, 18.25]`
  - MAP `17.25`
- Block-wise pooled angles differ substantially:
  - block 3: `21.0`
  - block 4: `29.0`
  - block 6: `26.5`
  - block 7: `26.25`
- Block-level phase dispersion is larger than its shuffle null:
  - observed phase-dispersion metric `0.00216`
  - null mean `7.64e-06`
  - permutation p-value `0.0476`

## What weakens the current argument

- Corrected temporal hold-out does not support the interference model as a predictive winner.
  - For six bins, `poly2` has slightly lower WAIC and slightly worse-to-tied RMSE than the interference model.
  - Simpler one-parameter baselines remain competitive.
- The direct falsification model for block-specific `theta` does not clearly rescue predictive performance.
  - On participant-block curves, the block-specific-theta model is not materially better than the global-theta model in RMSE.
  - Several block-specific fits land on search boundaries.
- The within-curve `x`-permutation null is not cleanly defeated.
  - Some permuted datasets still produce sharply concentrated posteriors at the boundary.
  - This means posterior concentration alone is not a sufficient falsification result.
- Simulation-based recovery is weak for low and moderate true angles.
  - Recovery is poor for true `theta` values near `10` and `17.25` degrees.
  - The additive null can still produce nontrivial apparent angles around `30` to `40` degrees.

## Submission implications

The strongest defensible claim remains the existence of a stable pooled
population-level angle under the original six-bin profile estimator.

The stronger mechanistic claims should be softened unless additional evidence is
added, especially:

- that poorer hold-out performance is specifically caused by block-dependent phase variation
- that the estimator is well-identified under realistic low-theta data-generating processes
- that narrow pooled posteriors are hard to obtain under reasonable nulls
