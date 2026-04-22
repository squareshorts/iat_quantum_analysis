#!/usr/bin/env python3
"""
run_cross_domain_expansion.py
==============================
Expands the matched public-domain comparison to additional Project Implicit
archives, with a focus on Race and Age IATs.

This script:
  1. Reads raw IAT log files for each domain using the same pipeline as
     run_matched_public_domain_analysis.py (critical blocks 3/4/6/7, six bins,
     leak-free standardisation).
  2. Profiles the pooled angular posterior for each domain.
  3. Computes a domain-transfer predictive criterion: θ trained on
     Gender–Science predicts held-out performance in each new domain (and
     vice-versa), testing whether the inferred geometry carries cross-task
     predictive leverage.
  4. Plots a multi-domain θ comparison showing whether the low-angle regime
     generalises or whether some consolidated social contrasts approach
     orthogonality under the same IAT architecture.

Data requirements
-----------------
The following directories must exist under <repo_root>/data/:

  data/Race_iat_2019/iat_<year>/     — Project Implicit Race IAT raw logs (*.txt)
  data/Age_iat_2019/iat_<year>/      — Project Implicit Age IAT raw logs (*.txt)

Each domain directory is expected to contain tab-separated files with columns:
  session_id, block_number, trial_number, trial_latency, task_name

If a domain's data directory is absent or empty the domain is skipped with a
warning; the script does not abort.  This allows running a partial analysis
when only some domain data are available.

Usage
-----
  python run_cross_domain_expansion.py [--domains race age] [--bins 6]

Outputs (in outputs/ and figures/ and tables/ next to this script)
------------------------------------------------------------------
  outputs/cross_domain_theta_summary.csv
  outputs/cross_domain_theta_bins.csv
  outputs/cross_domain_model_comparison.csv
  outputs/cross_domain_transfer_rmse.csv
  figures/cross_domain_theta_overlay.png
  figures/cross_domain_theta_bar.png
  tables/Table_cross_domain_theta.tex
  tables/Table_cross_domain_transfer.tex
"""

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import shared infrastructure from the main submission pipeline
# ---------------------------------------------------------------------------
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from run_submission_evidence import (
    CRITICAL_BLOCKS,
    PRIMARY_BINS,
    BIN_OPTIONS,
    HOLDOUT_FRAC,
    THETA_POSTERIOR_GRID,
    THETA_COARSE_GRID,
    THETA_REFINE_RADIUS,
    THETA_REFINE_STEP,
    K_COARSE_GRID,
    K_REFINE_RADIUS,
    K_REFINE_STEP,
    P_COARSE_GRID,
    P_REFINE_RADIUS,
    P_REFINE_STEP,
    OUT_DIR,
    FIG_DIR,
    TAB_DIR,
    arrays_from_curves,
    build_participant_curves_raw,
    fit_parameter_free_model,
    posterior_summary,
    profile_theta_posterior,
    row_standardize,
    search_best_param,
    stats_from_curves,
    temporal_holdout_mask,
    leak_free_standardize,
    design_matrix_from_param,
    compute_rss_and_beta,
    evaluate_on_mask,
    residual_sigma,
    HOLDOUT_RIDGE,
    PROFILE_RIDGE,
)

# ---------------------------------------------------------------------------
# Domain specifications
# All domains use the same critical blocks and preprocessing as Gender–Science.
# task_name must match the value in the task_name column of the raw log files.
# ---------------------------------------------------------------------------
ALL_DOMAIN_SPECS = {
    "gender_science": {
        "domain":    "Gender-Science",
        "short":     "gender_science_2019",
        "task_name": "scienceiat",
        "data_glob": BASE_DIR / "data" / "GenderScience_iat_2019" / "iat_2019",
        "file_glob": "iat*.txt",
        "color":     "#1f77b4",
    },
    "race": {
        "domain":    "Race",
        "short":     "race_2019",
        "task_name": "raceiat",
        "data_glob": BASE_DIR / "data" / "Race_iat_2019" / "iat_2019",
        "file_glob": "iat*.txt",
        "color":     "#d62728",
    },
    "age": {
        "domain":    "Age",
        "short":     "age_2019",
        "task_name": "ageiat",
        "data_glob": BASE_DIR / "data" / "Age_iat_2019" / "iat_2019",
        "file_glob": "iat*.txt",
        "color":     "#2ca02c",
    },
    "sexuality": {
        "domain":    "Sexuality",
        "short":     "sexuality_2019",
        "task_name": "sexualityiat",
        "data_glob": BASE_DIR / "data" / "sexuality_raw" / "Sexuality_iat_2019" / "iat",
        "file_glob": "iat*.txt",
        "color":     "#ff7f0e",
    },
}

RAW_COLS       = {"task_name", "block_number", "trial_number", "trial_latency", "session_id"}
PROCESSED_COLS = ["pid", "block", "trial_in_block", "rt"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PROCESSED_COLS)


def read_raw_file(path: Path, task_name: str) -> pd.DataFrame:
    """Read one raw IAT log file and return a standardised DataFrame."""
    try:
        df = pd.read_csv(path, sep="\t",
                         usecols=lambda c: c.strip() in RAW_COLS,
                         low_memory=False)
    except Exception as exc:
        print(f"    Warning: skipping {path.name} ({type(exc).__name__}: {exc})")
        return empty_frame()

    stripped = {str(c).strip() for c in df.columns}
    if RAW_COLS - stripped:
        print(f"    Warning: skipping {path.name}; missing {sorted(RAW_COLS - stripped)}")
        return empty_frame()

    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.rename(columns={
        "session_id":   "pid",
        "block_number": "block",
        "trial_number": "trial_in_block",
        "trial_latency":"rt",
    })
    keep = [*PROCESSED_COLS, "task_name"]
    df = df[keep].dropna(subset=["pid", "block", "trial_in_block", "rt", "task_name"])
    df["task_name"] = df["task_name"].astype(str).str.strip()
    df = df[df["task_name"] == task_name].copy()
    if df.empty:
        return empty_frame()
    for col in ["block", "trial_in_block", "rt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["block", "trial_in_block", "rt"])
    df["block"]        = df["block"].astype(int)
    df["trial_in_block"] = df["trial_in_block"].astype(int)
    df = df[df["block"].isin(CRITICAL_BLOCKS)].copy()
    return df[PROCESSED_COLS]


def cache_path(short: str, n_bins: int) -> Path:
    return OUT_DIR / f"cross_domain_{short}_curves_bins{n_bins}.pkl"


def build_domain_curves(spec: dict, n_bins: int) -> list[dict]:
    """Load or build participant curves for a domain at given bin count."""
    cp = cache_path(spec["short"], n_bins)
    if cp.exists():
        with open(cp, "rb") as fh:
            return pickle.load(fh)

    data_dir  = Path(spec["data_glob"])
    file_glob = spec["file_glob"]
    if not data_dir.exists():
        print(f"  [SKIP] {spec['domain']}: data dir not found ({data_dir})")
        return []
    paths = sorted(data_dir.glob(file_glob))
    if not paths:
        print(f"  [SKIP] {spec['domain']}: no files matching {file_glob} in {data_dir}")
        return []

    curves: list[dict] = []
    used, skipped = 0, 0
    for path in paths:
        df = read_raw_file(path, task_name=spec["task_name"])
        if df.empty:
            skipped += 1
            continue
        used += 1
        curves.extend(build_participant_curves_raw(df, n_bins=n_bins))
        del df
        gc.collect()

    print(f"  {spec['domain']} | bins={n_bins}: {len(curves)} curves "
          f"from {used} files" + (f"; {skipped} skipped" if skipped else ""))
    with open(cp, "wb") as fh:
        pickle.dump(curves, fh)
    return curves


# ---------------------------------------------------------------------------
# Profiling and model comparison
# ---------------------------------------------------------------------------

def profile_domain(curves: list[dict]) -> tuple[dict, np.ndarray, np.ndarray]:
    x, y_raw, mask, _ = arrays_from_curves(curves)
    y_std = row_standardize(y_raw, mask)
    posterior, rss = profile_theta_posterior(x, y_std, mask, THETA_POSTERIOR_GRID)
    summary = posterior_summary(THETA_POSTERIOR_GRID, posterior)
    return summary, posterior, rss


def model_comparison(curves: list[dict]) -> pd.DataFrame:
    x, y_raw, mask, _ = arrays_from_curves(curves)
    counts, sums, sumsqs = stats_from_curves(curves)
    specs = [
        ("interference", THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("cos_only",     THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("sin_only",     THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("poly2",        None,              None,                 None),
        ("exp",          K_COARSE_GRID,     K_REFINE_RADIUS,     K_REFINE_STEP),
        ("power",        P_COARSE_GRID,     P_REFINE_RADIUS,     P_REFINE_STEP),
    ]
    rows = []
    for model_name, coarse, rrad, rstep in specs:
        if coarse is None:
            res = fit_parameter_free_model(x, y_raw, mask, model_name,
                                           counts=counts, sums=sums, sumsqs=sumsqs)
        else:
            res = search_best_param(x, y_raw, mask, model_name,
                                    coarse, rrad, rstep,
                                    counts=counts, sums=sums, sumsqs=sumsqs)
        rows.append({k: res[k] for k in
                     ["model", "best_param", "train_rss", "rmse_test",
                      "mean_logscore_test", "sigma_train", "n_curves",
                      "n_test_obs", "boundary_hit"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-domain transfer criterion
# ---------------------------------------------------------------------------

def cross_domain_transfer_rmse(
    source_curves: list[dict],
    target_curves: list[dict],
    source_domain: str,
    target_domain: str,
) -> dict:
    """
    Train the interference model on source_curves (select θ by profile
    posterior MAP), then evaluate on the temporal hold-out split of
    target_curves.  Also evaluate the polynomial baseline on the same split.

    This criterion tests whether geometric structure discovered in one domain
    carries predictive leverage when transferred to another domain — a harder
    criterion than within-domain RMSE.

    Returns a dict with transfer RMSE for both models and the selected θ.
    """
    # --- Source: find θ MAP ---
    _, src_post, _ = profile_domain(source_curves)
    theta_transfer = float(THETA_POSTERIOR_GRID[np.argmax(src_post)])

    # --- Target: split and evaluate ---
    x_t, y_raw_t, mask_t, _ = arrays_from_curves(target_curves)
    counts_t, sums_t, sumsqs_t = stats_from_curves(target_curves)

    train_mask, test_mask, valid = temporal_holdout_mask(
        x_t, mask_t, HOLDOUT_FRAC, min_train_pts=3
    )
    if not valid.any():
        print(f"    [WARN] transfer {source_domain} → {target_domain}: no valid hold-out rows")
        return {"source": source_domain, "target": target_domain,
                "theta_source_map": theta_transfer,
                "interference_transfer_rmse": np.nan,
                "poly_transfer_rmse": np.nan}

    xv  = x_t[valid];   yv_raw = y_raw_t[valid]
    tv  = train_mask[valid]; tsv = test_mask[valid]
    sk  = {k: v[valid] for k, v in zip(
        ["counts", "sums", "sumsqs"],
        [counts_t[valid], sums_t[valid], sumsqs_t[valid]]
    )}
    yv = leak_free_standardize(yv_raw, tv, **sk)

    # Interference model at transferred θ
    des_int = design_matrix_from_param(xv, "interference", theta_transfer)
    _, beta_int, rows_int = compute_rss_and_beta(des_int, yv, tv, ridge=HOLDOUT_RIDGE)
    beta_full_int = np.zeros((xv.shape[0], 3))
    beta_full_int[rows_int] = beta_int
    sig_int = residual_sigma(des_int, beta_full_int, yv, tv)
    rmse_int, _ = evaluate_on_mask(des_int, beta_full_int, yv, tsv, sigma=sig_int)

    # Polynomial baseline (no transfer — fit directly on target train)
    des_poly = design_matrix_from_param(xv, "poly2", None)
    _, beta_poly, rows_poly = compute_rss_and_beta(des_poly, yv, tv, ridge=HOLDOUT_RIDGE)
    beta_full_poly = np.zeros((xv.shape[0], 3))
    beta_full_poly[rows_poly] = beta_poly
    sig_poly = residual_sigma(des_poly, beta_full_poly, yv, tv)
    rmse_poly, _ = evaluate_on_mask(des_poly, beta_full_poly, yv, tsv, sigma=sig_poly)

    return {
        "source":                        source_domain,
        "target":                        target_domain,
        "theta_source_map":              theta_transfer,
        "interference_transfer_rmse":    float(rmse_int),
        "poly_transfer_rmse":            float(rmse_poly),
        "transfer_advantage":            float(rmse_poly - rmse_int),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_theta_overlay(
    domain_summaries: list[dict],
    profile_map: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for spec in domain_summaries:
        domain  = spec["domain"]
        color   = spec["color"]
        post    = profile_map[domain]
        theta_m = spec["theta_map"]
        ax.plot(THETA_POSTERIOR_GRID, post / post.max(),
                lw=2.0, label=domain, color=color)
        ax.axvline(theta_m, ls="--", lw=1.0, color=color, alpha=0.85)
    ax.set_xlabel(r"$\theta$ (degrees)", fontsize=11)
    ax.set_ylabel("Normalised posterior",  fontsize=11)
    ax.set_title("Cross-domain pooled posterior comparison", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cross_domain_theta_overlay.png", dpi=300)
    plt.close()


def plot_theta_bar(domain_summaries: list[dict]) -> None:
    domains = [s["domain"]    for s in domain_summaries]
    means   = [s["theta_mean"] for s in domain_summaries]
    colors  = [s["color"]     for s in domain_summaries]
    lo_err  = [s["theta_mean"] - s["hdi_low"]  for s in domain_summaries]
    hi_err  = [s["hdi_high"]  - s["theta_mean"] for s in domain_summaries]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.bar(domains, means, color=colors, alpha=0.88)
    ax.errorbar(domains, means, yerr=[lo_err, hi_err],
                fmt="none", ecolor="black", capsize=5, lw=1.2)
    ax.set_ylabel(r"$\theta$ mean (degrees, 94\% HDI)", fontsize=11)
    ax.set_title("Cross-domain pooled θ comparison",     fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cross_domain_theta_bar.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(domain_keys: list[str], n_bins: int) -> None:
    ensure_dirs()
    t0 = time.time()

    specs_to_run = [ALL_DOMAIN_SPECS[k] for k in domain_keys
                    if k in ALL_DOMAIN_SPECS]

    # Build or load curves for each domain at n_bins
    domain_curves: dict[str, list[dict]] = {}
    for spec in specs_to_run:
        print(f"\n=== {spec['domain']} ===")
        curves = build_domain_curves(spec, n_bins=n_bins)
        if not curves:
            continue
        domain_curves[spec["domain"]] = curves

    if not domain_curves:
        print("\n[ERROR] No domain data could be loaded.  "
              "Check that data directories exist under data/.")
        return

    # Profile each domain
    summaries:   list[dict]          = []
    profile_map: dict[str, np.ndarray] = {}
    for spec in specs_to_run:
        domain = spec["domain"]
        if domain not in domain_curves:
            continue
        curves = domain_curves[domain]
        print(f"\n  Profiling {domain} ({len(curves):,} curves)...")
        summary, posterior, rss = profile_domain(curves)
        profile_map[domain] = posterior
        summaries.append({
            "domain":     domain,
            "year":       2019,
            "bins":       n_bins,
            "n_curves":   len(curves),
            "theta_mean": summary["mean"],
            "theta_sd":   summary["sd"],
            "theta_map":  summary["map"],
            "hdi_low":    summary["hdi_low"],
            "hdi_high":   summary["hdi_high"],
            "color":      spec["color"],
        })
        pd.DataFrame({
            "theta_deg": THETA_POSTERIOR_GRID,
            "posterior": posterior,
            "rss":       rss,
        }).to_csv(OUT_DIR / f"cross_domain_{spec['short']}_theta_profile.csv", index=False)
        print(f"    MAP={summary['map']:.2f}°, mean={summary['mean']:.2f}°, "
              f"HDI=[{summary['hdi_low']:.1f}°,{summary['hdi_high']:.1f}°]")

    summary_df = pd.DataFrame(summaries).drop(columns="color")
    summary_df.to_csv(OUT_DIR / "cross_domain_theta_summary.csv", index=False)

    # Model comparison for each domain
    mc_frames = []
    for spec in specs_to_run:
        domain = spec["domain"]
        if domain not in domain_curves:
            continue
        print(f"\n  Model comparison: {domain}")
        mc = model_comparison(domain_curves[domain])
        mc.insert(0, "domain", domain)
        mc_frames.append(mc)
    if mc_frames:
        mc_df = pd.concat(mc_frames, ignore_index=True)
        mc_df.to_csv(OUT_DIR / "cross_domain_model_comparison.csv", index=False)

    # Cross-domain transfer criterion
    transfer_rows = []
    domain_names  = [s["domain"] for s in summaries]
    for src in domain_names:
        for tgt in domain_names:
            if src == tgt:
                continue
            print(f"\n  Transfer: {src} → {tgt}")
            result = cross_domain_transfer_rmse(
                domain_curves[src], domain_curves[tgt], src, tgt
            )
            transfer_rows.append(result)
            print(f"    θ_source={result['theta_source_map']:.1f}°  "
                  f"int_RMSE={result['interference_transfer_rmse']:.4f}  "
                  f"poly_RMSE={result['poly_transfer_rmse']:.4f}  "
                  f"advantage={result['transfer_advantage']:+.4f}")

    if transfer_rows:
        transfer_df = pd.DataFrame(transfer_rows)
        transfer_df.to_csv(OUT_DIR / "cross_domain_transfer_rmse.csv", index=False)

    # Figures
    for s in summaries:
        s["color"] = ALL_DOMAIN_SPECS.get(
            next(k for k, v in ALL_DOMAIN_SPECS.items() if v["domain"] == s["domain"]),
            {}
        ).get("color", "gray")
    plot_theta_overlay(summaries, profile_map)
    plot_theta_bar(summaries)

    # LaTeX tables
    tex_df = summary_df.rename(columns={
        "domain": "Task", "year": "Year", "bins": "Bins",
        "n_curves": "N", "theta_mean": r"$\theta$ mean", "theta_sd": "SD",
        "theta_map": "MAP", "hdi_low": "HDI low", "hdi_high": "HDI high",
    })
    with open(TAB_DIR / "Table_cross_domain_theta.tex", "w", encoding="utf-8") as fh:
        fh.write(tex_df.to_latex(index=False, float_format="%.3f"))

    if transfer_rows:
        tex_transfer = transfer_df.rename(columns={
            "source": "Source", "target": "Target",
            "theta_source_map": r"$\theta$ (source)",
            "interference_transfer_rmse": "Int. transfer RMSE",
            "poly_transfer_rmse": "Poly RMSE",
            "transfer_advantage": "Advantage",
        })
        with open(TAB_DIR / "Table_cross_domain_transfer.tex", "w", encoding="utf-8") as fh:
            fh.write(tex_transfer.to_latex(index=False, float_format="%.4f"))

    elapsed = (time.time() - t0) / 60.0
    print(f"\nDone. Elapsed: {elapsed:.1f} min")
    with open(OUT_DIR / "cross_domain_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "domains":       [s["domain"] for s in summaries],
            "theta_summaries": [
                {k: v for k, v in s.items() if k != "color"} for s in summaries
            ],
            "elapsed_minutes": elapsed,
        }, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-domain IAT angular expansion analysis"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["gender_science", "sexuality", "race", "age"],
        choices=list(ALL_DOMAIN_SPECS.keys()),
        help="Domains to include (default: all four)",
    )
    parser.add_argument(
        "--bins", type=int, default=PRIMARY_BINS,
        help=f"Number of quantile bins (default: {PRIMARY_BINS})",
    )
    args = parser.parse_args()
    main(domain_keys=args.domains, n_bins=args.bins)
