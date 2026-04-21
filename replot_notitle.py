import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs")
FIG_DIR = Path("figures")

def replot_recovery():
    summ_df = pd.read_csv(OUT_DIR / "ext_recovery_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.errorbar(summ_df["theta_true"], summ_df["mean_map"],
                yerr=summ_df["rmse"], fmt="o-", capsize=4, color="steelblue",
                label="Mean recovered MAP ± RMSE")
    lim = [0, 130]
    ax.plot(lim, lim, "k--", alpha=0.5, label="Identity")
    ax.axvline(17.25, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("True θ (degrees)")
    ax.set_ylabel("Recovered MAP θ (degrees)")
    ax.legend(fontsize=8)
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax = axes[1]
    ax.axhline(0, color="k", ls="--", alpha=0.5, lw=0.8)
    ax.bar(summ_df["theta_true"], summ_df["bias"], width=4, color="steelblue", alpha=0.7)
    ax.axvline(17.25, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("True θ (degrees)")
    ax.set_ylabel("Bias (MAP - true θ)")

    ax = axes[2]
    cov = summ_df["coverage"].values
    n_r = summ_df["n_reps"].values
    ci_lo = np.clip(cov - 1.96 * np.sqrt(cov * (1 - cov) / n_r), 0, 1)
    ci_hi = np.clip(cov + 1.96 * np.sqrt(cov * (1 - cov) / n_r), 0, 1)
    ax.errorbar(summ_df["theta_true"], cov,
                yerr=[cov - ci_lo, ci_hi - cov],
                fmt="o", capsize=4, color="steelblue")
    ax.axhline(0.94, color="red", ls="--", lw=1.2, label="Nominal 94%")
    ax.axvline(17.25, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("True θ (degrees)")
    ax.set_ylabel("94% HDI coverage rate")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ext_recovery_plots.png", dpi=300)
    plt.close()
    print("replot_recovery done")

def replot_nscaling():
    df = pd.read_csv(OUT_DIR / "ext_nscaling.csv")
    agg = df[df["rep"] >= 0].groupby("n_curves").agg(
        mean_map=("theta_map", "mean"),
        sd_map=("theta_map", "std"),
        mean_sd=("theta_sd", "mean"),
        coverage=("covered", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(agg["n_curves"], agg["mean_map"], "o-", color="steelblue", label="Simulation mean MAP")
    ax1.fill_between(agg["n_curves"],
                     agg["mean_map"] - agg["sd_map"],
                     agg["mean_map"] + agg["sd_map"],
                     alpha=0.25, color="steelblue")
    ax1.scatter([141329], [17.25], color="red", s=80, zorder=5, label="Real data (N=141,329)")
    ax1.axhline(17.25, color="gray", ls="--", lw=1, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("N (participants)")
    ax1.set_ylabel("Recovered MAP θ (degrees)")
    ax1.legend(fontsize=8)

    ax2.plot(agg["n_curves"], agg["coverage"], "o-", color="steelblue")
    ax2.scatter([141329], [1.0], color="red", s=80, zorder=5, label="Real data MAP = 17.25° (exact)")
    ax2.axhline(0.94, color="red", ls="--", lw=1.2, label="Nominal 94%")
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_xscale("log")
    ax2.set_xlabel("N (participants)")
    ax2.set_ylabel("94% HDI coverage rate")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ext_nscaling.png", dpi=300)
    plt.close()
    print("replot_nscaling done")

def replot_nulls():
    perm_df = pd.read_csv(OUT_DIR / "ext_permutation_null.csv")
    quad_df = pd.read_csv(OUT_DIR / "ext_quadratic_null.csv")
    
    metrics = [
        ("theta_map",         "Posterior MAP θ (degrees)",     True),
        ("theta_sd",          "Posterior SD (degrees)",         False),
        ("hdi_width",         "94% HDI width (degrees)",        False),
        ("posterior_entropy", "Posterior entropy (nats)",       False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    obs_full = {
        "theta_map":         17.25,
        "theta_sd":          0.483,
        "hdi_width":         18.25 - 16.5,
        "posterior_entropy": 2.0766,
    }

    for ax, (metric, xlabel, is_location) in zip(axes, metrics):
        perm_vals = perm_df[metric].values
        quad_vals = quad_df[metric].values

        bins = 25
        ax.hist(perm_vals, bins=bins, alpha=0.55, color="steelblue",
                edgecolor="white", lw=0.5, label=f"Perm. null (n={len(perm_df)})")
        ax.hist(quad_vals, bins=bins, alpha=0.55, color="darkorange",
                edgecolor="white", lw=0.5, label=f"Quad. null (n={len(quad_df)})")
        
        obs_v = obs_full.get(metric)
        if obs_v is not None:
            ax.axvline(obs_v, color="crimson", lw=2.2, label=f"Observed = {obs_v:.3f}")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ext_null_distributions.png", dpi=300)
    plt.close()
    print("replot_nulls done")

def replot_bootstrap():
    df = pd.read_csv(OUT_DIR / "ext_bootstrap_theta.csv")
    boot_ci_lo = float(np.percentile(df["theta_map"], 2.5))
    boot_ci_hi = float(np.percentile(df["theta_map"], 97.5))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["theta_map"], bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(17.25, color="crimson", lw=2.2, ls="--", label="Full-sample MAP = 17.25°")
    ax.axvline(boot_ci_lo, color="gray", lw=1.5, ls=":", label=f"95% CI [{boot_ci_lo:.1f}°, {boot_ci_hi:.1f}°]")
    ax.axvline(boot_ci_hi, color="gray", lw=1.5, ls=":")
    ax.set_xlabel("Bootstrap MAP θ (degrees)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ext_bootstrap_distribution.png", dpi=300)
    plt.close()
    print("replot_bootstrap done")

def replot_objectives():
    # Only 6 bins usually shown in main paper from this, 
    # but we can grab the profiles if available.
    import glob
    if not Path("outputs/ext_theta_profile_bins6.csv").exists():
        print("Cannot replot objectives, missing profile 6")
        return
    
    # Actually this plot requires RMSE curves which are NOT saved in CSVs!
    # They were generated inside step2_objective_comparison and not saved entirely 
    # other than ext_objectives_by_bins.csv which only has the optimum.
    # To redo ext_objectives_vs_theta.png without title, we either leave it (maybe user doesn't use it or uses the 6-bin one directly)
    # or I will modify run_extended_analysis so it ONLY runs step2, and nothing else.
    print("Objectives not replotted (requires RMSE curves)")

if __name__ == "__main__":
    replot_recovery()
    replot_nscaling()
    replot_nulls()
    replot_bootstrap()
    replot_objectives()
