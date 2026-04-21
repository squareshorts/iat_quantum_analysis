# Submission Audit

## Verdict

Yes: this repo is the correct project for the manuscript's main analysis.

The strongest evidence is:

- the primary data source is `data/GenderScience_iat_2019/iat_2019/iat*.txt`
- the main pipeline reproduces the manuscript-facing participant count `141329`
- the key manuscript tables and figures are already present with matching values

## Present and aligned

- Main grid-OLS analysis script:
  - `run_theta_grid_full_plus_baselines.py`
- Expanded scientific-audit script:
  - `run_submission_evidence.py`
- Matching manuscript outputs:
  - `figures/theta_posterior_full.png`
  - `figures/ppc_full.png`
  - `tables/Table_theta_summary.tex`
  - `tables/Table_fit_quality.tex`
  - `tables/Table_model_comparison.tex`
  - `tables/Table_theta_bins_robustness.tex`
- Block-wise phase export:
  - `outputs/theta_phase_by_block.csv`
- Added manuscript-facing scientific-audit outputs:
  - `outputs/model_comparison_submission.csv`
  - `outputs/block_theta_summary.csv`
  - `outputs/block_model_comparison.csv`
  - `outputs/negative_controls_summary.csv`
  - `outputs/theta_recovery_simulation.csv`
  - `outputs/theta_null_simulation.csv`
  - `outputs/waic_comparison_submission.csv`
  - `SCIENTIFIC_AUDIT.md`

## Missing before submission

- Manuscript bibliography is not present.
  - `paper/main.tex` expects `paper/references.bib`, but that file still needs the real entries.
- No journal/package metadata yet.
  - There is no cover letter, title page variant, author contribution statement, conflict statement, or journal-specific checklist in the repo.
- No license file.
  - If this repo will be public, add a `LICENSE`.

## Inconsistencies to resolve

- The manuscript says ArviZ is used for HDIs and posterior-predictive checks in the main workflow.
  - The manuscript-aligned grid-OLS pipeline computes the discrete posterior and HDI directly in Python and uses Matplotlib for the PPC figure.
  - ArviZ appears in exploratory hierarchical scripts, not the main result path.
- The manuscript describes temporal hold-out evaluation generically.
  - The original script still defaults to a `5000`-participant subsample.
  - The new `run_submission_evidence.py` script performs a corrected full-sample re-analysis, but those results are not yet integrated into the manuscript text.
- The manuscript says robustness checks are repeated for `4, 6, 8` bins.
  - The new audit script does run hold-out comparisons across `4, 6, 8` bins.
  - However, those outputs show stronger preprocessing sensitivity and should be reflected honestly in the paper.
- The repo contains exploratory hierarchical outputs that disagree with the manuscript's main `theta ~ 17 deg` result.
  - Example files include `outputs/posterior_summary.csv` and `outputs/posterior_summary_GenderScience.csv`, which reflect a different PyMC workflow with `theta ~ 94 deg`.
  - Keep these clearly separated from submission artifacts to avoid confusion.
- `tables/Table_loo_comparison.tex` is not submission-ready.
  - It includes raw xarray object strings in the LaTeX output.
  - The underlying grouped-LOO workflow is exploratory and should be validated before citing it.

## Evidence gaps in the current paper package

- The new evidence package weakens several stronger claims rather than fully resolving them.
  - The pooled six-bin `theta` result reproduces.
  - The direct block-specific-theta falsification does not clearly rescue predictive performance.
  - The simulation-based recovery check is weak for low true angles.
  - The within-curve permutation null can still produce concentrated boundary-seeking posteriors.

## Recommended next actions

1. Add the real `paper/references.bib`.
2. Update the manuscript so its claims match `SCIENTIFIC_AUDIT.md` and the `*_submission` outputs.
3. Decide whether to keep or soften the mechanistic interpretation around block-dependent phase variation.
4. Add journal-specific submission files once the target venue is chosen.
