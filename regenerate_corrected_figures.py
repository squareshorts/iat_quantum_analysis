"""
Regenerate figures and table using display-order encoding for life-satisfaction IAT.

The key fix: use the FILE APPEARANCE POSITION (sequential order of rows in the raw
Testable CSV) as the within-block axis, NOT rowNo (which is the Testable script row
number and maps 1:1 to trialNo/item ID). The appearance position is the actual
randomized presentation order.

Outputs:
  - figures/domain_theta_contrast_social_vs_self.png  (no title, corrected LS values)
  - figures/external_leverage_summary.png             (no title, corrected LS values)
  - tables/Table_external_model_comparison.tex        (display-order model comparison)
"""
from __future__ import annotations
import csv, io, re, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
FIG_DIR  = BASE_DIR / "figures"
TAB_DIR  = BASE_DIR / "tables"

EXTERNAL_ZIP = DATA_DIR / "life_satisfaction_iat_2024" / "raw" / "187results.zip"
EXTERNAL_CRITICAL_PHASES = ["fase3", "fase5"]
DEFAULT_BINS = 6
HOLDOUT_FRAC = 0.20
HDI_PROB = 0.94

THETA_FINE_GRID   = np.arange(0.0, 180.0 + 0.25, 0.25)
THETA_SEARCH_GRID = np.arange(0.0, 180.0 + 2.0, 2.0)
THETA_COARSE_GRID = np.arange(0.0, 180.0 + 1.0, 1.0)
K_GRID = np.linspace(0.0, 12.0, 49)
P_GRID = np.linspace(0.05, 4.0, 40)

GS_POSTERIOR_CSV = OUT_DIR / "theta_grid_profile_submission.csv"


# ── helpers ──────────────────────────────────────────────────────────────

def hdi_from_discrete(grid_vals, probs, mass=HDI_PROB):
    order = np.argsort(probs)[::-1]
    chosen, total = [], 0.0
    for idx in order:
        chosen.append(int(idx)); total += float(probs[idx])
        if total >= mass: break
    sel = np.asarray(chosen, dtype=int)
    return float(grid_vals[sel].min()), float(grid_vals[sel].max())

def posterior_summary(grid_vals, posterior):
    mean = float((grid_vals * posterior).sum())
    sd   = float(np.sqrt(((grid_vals - mean)**2 * posterior).sum()))
    lo, hi = hdi_from_discrete(grid_vals, posterior)
    return {"theta_mean": mean, "theta_sd": sd,
            "theta_map": float(grid_vals[np.argmax(posterior)]),
            "hdi_low": lo, "hdi_high": hi}

def arrays_from_curves(curves):
    max_t = max(len(c["x"]) for c in curves)
    n = len(curves)
    x    = np.full((n, max_t), np.nan)
    y    = np.full((n, max_t), np.nan)
    mask = np.zeros((n, max_t), dtype=bool)
    for i, c in enumerate(curves):
        t = len(c["x"]); x[i,:t] = c["x"]; y[i,:t] = c["y"]; mask[i,:t] = True
    return x, y, mask

def profile_theta_posterior(x, y_std, mask, theta_grid_deg):
    n_obs = int(mask.sum()); y2_sum = float(np.sum((y_std**2)[mask]))
    ones = np.ones_like(x); post_log = []
    for theta_deg in theta_grid_deg:
        arg = np.deg2rad(theta_deg) * x
        c = np.cos(arg); s = np.sin(arg)
        c[~mask] = 0.0; s[~mask] = 0.0
        o = ones.copy(); o[~mask] = 0.0
        ym = np.where(mask, y_std, 0.0)
        s_o  = o.sum(1); s_c  = c.sum(1); s_s  = s.sum(1)
        s_cc = (c*c).sum(1); s_ss = (s*s).sum(1); s_cs = (c*s).sum(1)
        s_y  = ym.sum(1);  s_yc = (ym*c).sum(1);  s_ys = (ym*s).sum(1)
        a = np.stack([np.stack([s_o,s_c,s_s],-1),
                      np.stack([s_c,s_cc,s_cs],-1),
                      np.stack([s_s,s_cs,s_ss],-1)],-2)
        b = np.stack([s_y,s_yc,s_ys],-1)
        a[...,range(3),range(3)] += 1e-8
        proj = (b[...,None,:] @ np.linalg.solve(a, b[...,:,None])).squeeze(-1).squeeze(-1)
        rss = y2_sum - float(proj.sum())
        post_log.append(-(n_obs/2.0)*np.log(max(rss,1e-12)))
    pla = np.asarray(post_log); pla -= pla.max()
    posterior = np.exp(pla); posterior /= posterior.sum()
    return posterior

def temporal_holdout_mask(x, mask, frac, min_train_pts):
    n = x.shape[0]
    train_mask = np.zeros_like(mask, dtype=bool)
    test_mask  = np.zeros_like(mask, dtype=bool)
    valid      = np.zeros(n, dtype=bool)
    for i in range(n):
        idx = np.where(mask[i])[0]; n_obs = idx.size
        if n_obs < min_train_pts + 1: continue
        ordered = idx[np.argsort(x[i, idx])]
        k = max(1, int(np.ceil(frac * n_obs)))
        k = min(k, n_obs - min_train_pts)
        test_idx  = ordered[-k:]
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)
        if train_idx.size < min_train_pts: continue
        train_mask[i, train_idx] = True; test_mask[i, test_idx] = True; valid[i] = True
    return train_mask, test_mask, valid

def leak_free_standardize(y_raw, train_mask):
    yz = np.full_like(y_raw, np.nan)
    for i in range(y_raw.shape[0]):
        idx_tr  = np.where(train_mask[i])[0]
        idx_all = np.where(~np.isnan(y_raw[i]))[0]
        if idx_tr.size == 0 or idx_all.size == 0: continue
        mu = float(np.mean(y_raw[i, idx_tr]))
        sd = float(np.std(y_raw[i, idx_tr], ddof=1))
        if not np.isfinite(sd) or sd < 1e-9: sd = 1.0
        yz[i, idx_all] = (y_raw[i, idx_all] - mu) / sd
    return yz

def design_matrix(model_name, x, param=None):
    if model_name == "interference":
        arg = np.deg2rad(float(param)) * x
        return np.column_stack([np.ones(len(x)), np.cos(arg), np.sin(arg)])
    if model_name == "cos_only":
        arg = np.deg2rad(float(param)) * x
        return np.column_stack([np.ones(len(x)), np.cos(arg)])
    if model_name == "sin_only":
        arg = np.deg2rad(float(param)) * x
        return np.column_stack([np.ones(len(x)), np.sin(arg)])
    if model_name == "poly2":
        return np.column_stack([np.ones(len(x)), x, x**2])
    if model_name == "exp":
        return np.column_stack([np.ones(len(x)), np.exp(-float(param)*x)])
    if model_name == "power":
        return np.column_stack([np.ones(len(x)), np.power(np.clip(x,1e-6,None), float(param))])
    raise ValueError(model_name)

def evaluate_model_rmse(x, y_raw, mask, model_name, grid):
    train_mask, test_mask, valid = temporal_holdout_mask(x, mask, HOLDOUT_FRAC, 3)
    yz = leak_free_standardize(y_raw, train_mask)
    x = x[valid]; yz = yz[valid]
    train_mask = train_mask[valid]; test_mask = test_mask[valid]
    params = [None] if grid is None else list(grid)
    best_param, best_rmse = None, np.inf
    for param in params:
        sq_errors = []
        for i in range(x.shape[0]):
            tr = np.where(train_mask[i])[0]; te = np.where(test_mask[i])[0]
            a = design_matrix(model_name, x[i,tr], param)
            b = design_matrix(model_name, x[i,te], param)
            beta = np.linalg.lstsq(a, yz[i,tr], rcond=None)[0]
            sq_errors.extend((yz[i,te] - b@beta)**2)
        rmse = float(np.sqrt(np.mean(sq_errors)))
        if rmse < best_rmse:
            best_rmse = rmse; best_param = None if param is None else float(param)
    return best_param, best_rmse


# ── load raw archive with APPEARANCE ORDER ───────────────────────────────

def parse_testable_csv(text):
    parts = [p for p in text.split("\n\n") if p.strip()]
    if len(parts) < 2: raise ValueError("Bad format")
    meta = next(csv.DictReader(io.StringIO(parts[0])))
    attempts = list(csv.DictReader(io.StringIO(parts[1])))
    return meta, attempts


def load_display_order_trials():
    """Load life-satisfaction trials with APPEARANCE ORDER as display position.

    The key insight: in Testable CSVs, the rows appear in the order they were
    actually presented to the participant (randomized). The 'rowNo' field is
    the script row number, NOT the presentation position. We use enumerate()
    within each phase to get the true display position.
    """
    quality_rows = []
    all_rows = []

    with zipfile.ZipFile(EXTERNAL_ZIP) as zf:
        members = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for member in members:
            meta, attempts = parse_testable_csv(zf.read(member).decode("utf-8-sig"))
            participant_name = (meta.get("Nome completo") or "").strip()
            gmt_timestamp = meta.get("GMT_timestamp")

            rts = [float(r["RT"]) for r in attempts if r.get("RT")]
            fast_prop = float(np.mean(np.asarray(rts) < 300.0)) if rts else np.nan

            quality_rows.append({
                "source_file": member,
                "participant_name": participant_name,
                "gmt_timestamp": gmt_timestamp,
                "prop_fast": fast_prop,
            })

            # Track appearance position within each phase
            phase_counters = {}
            # Group attempts by (phase, base_trial) to collapse correction loops
            phase_item_first_appearance = {}

            for row in attempts:
                phase = row.get("condition1")
                if phase not in EXTERNAL_CRITICAL_PHASES:
                    continue
                trial_no_raw = row.get("trialNo", "")
                base_trial = str(trial_no_raw).split("_")[0]
                digits = re.sub(r"\D", "", base_trial)
                item_id = int(digits) if digits else 0
                rt_val = float(row["RT"]) if row.get("RT") else np.nan

                key = (phase, item_id)
                if key not in phase_item_first_appearance:
                    # First time seeing this item in this phase = its display position
                    if phase not in phase_counters:
                        phase_counters[phase] = 0
                    phase_item_first_appearance[key] = phase_counters[phase]
                    phase_counters[phase] += 1

                    all_rows.append({
                        "source_file": member,
                        "phase": phase,
                        "item_id": item_id,
                        "display_pos": phase_item_first_appearance[key],
                        "rt": rt_val,
                        "n_attempts": 1,
                    })
                else:
                    # Correction attempt: add RT to existing row
                    # Find the existing row and update
                    for r in reversed(all_rows):
                        if (r["source_file"] == member and r["phase"] == phase
                                and r["item_id"] == item_id):
                            r["rt"] += rt_val
                            r["n_attempts"] += 1
                            break

    quality_df = pd.DataFrame(quality_rows)
    trial_df = pd.DataFrame(all_rows)

    # Apply same exclusion as original: first attempt per name, <10% fast
    quality_df = quality_df.sort_values(["participant_name", "gmt_timestamp", "source_file"])
    quality_df["repeat_rank"] = quality_df.groupby("participant_name").cumcount() + 1
    quality_df["keep"] = (quality_df["repeat_rank"] == 1) & (quality_df["prop_fast"] <= 0.10)

    kept_files = set(quality_df.loc[quality_df["keep"], "source_file"])
    trial_df = trial_df[trial_df["source_file"].isin(kept_files)].copy()

    # Assign participant IDs
    kept_cases = quality_df.loc[quality_df["keep"], ["source_file", "gmt_timestamp"]].copy()
    kept_cases = kept_cases.sort_values(["gmt_timestamp", "source_file"]).reset_index(drop=True)
    kept_cases["participant_id"] = [f"lsiat_{i:03d}" for i in range(1, len(kept_cases)+1)]

    trial_df = trial_df.merge(kept_cases[["source_file","participant_id"]], on="source_file")

    print(f"  Loaded {len(kept_cases)} valid participants, {len(trial_df)} collapsed critical trials")

    # Verify display_pos is shuffled relative to item_id
    sample = trial_df[trial_df["participant_id"] == "lsiat_001"]
    sample_f3 = sample[sample["phase"] == "fase3"].sort_values("display_pos")
    print(f"  Verification — lsiat_001 fase3 item IDs in display order: "
          f"{sample_f3['item_id'].tolist()}")

    return trial_df


def build_curves_from_display_order(df, n_bins=DEFAULT_BINS, standardize=True):
    """Build binned curves using display_pos (appearance order) as within-block axis."""
    curves = []
    for pid, group in df.groupby("participant_id", sort=False):
        g = group.copy()
        # Normalize display_pos to [0,1] within each phase
        g["pos_norm"] = g.groupby("phase")["display_pos"].transform(
            lambda s: (s - s.min()) / max(1.0, float(s.max() - s.min()))
        )
        values = g[["pos_norm", "rt"]].to_numpy(dtype=float)
        if len(values) < n_bins:
            continue
        q = np.quantile(values[:,0], np.linspace(0,1,n_bins+1))
        x_bin, y_bin = [], []
        for i in range(n_bins):
            lo, hi = q[i], q[i+1]
            m = (values[:,0] >= lo) & (values[:,0] <= hi)
            chunk = values[m]
            if chunk.size == 0: continue
            x_bin.append(float(np.mean(chunk[:,0])))
            y_bin.append(float(np.mean(chunk[:,1])))
        if len(x_bin) < 3: continue
        y_arr = np.asarray(y_bin)
        if standardize:
            y_sd = float(np.std(y_arr, ddof=1))
            if not np.isfinite(y_sd) or y_sd < 1e-9: y_sd = 1.0
            y_arr = (y_arr - float(np.mean(y_arr))) / y_sd
        curves.append({"pid": pid, "x": np.asarray(x_bin), "y": y_arr})
    return curves


# ── 1. Forest plot ───────────────────────────────────────────────────────

def regenerate_forest_plot(ext_summary):
    data = [
        {"domain": "Gender-Science",    "mean": 17.29, "hdi_low": 16.5,  "hdi_high": 18.25, "type": "social"},
        {"domain": "Age",               "mean": 18.48, "hdi_low": 17.5,  "hdi_high": 19.25, "type": "social"},
        {"domain": "Sexuality",         "mean": 19.12, "hdi_low": 18.25, "hdi_high": 20.0,  "type": "social"},
        {"domain": "Life-Satisfaction\n(display order)",
         "mean": ext_summary["theta_mean"], "hdi_low": ext_summary["hdi_low"],
         "hdi_high": ext_summary["hdi_high"], "type": "self"},
    ]
    df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {"social": "#1f77b4", "self": "#ff7f0e"}
    y_pos = np.arange(len(df))
    for i, row in df.iterrows():
        c = colors[row["type"]]
        ax.plot([row["hdi_low"], row["hdi_high"]], [y_pos[i]]*2, color=c, lw=3, zorder=1)
        ax.scatter([row["mean"]], [y_pos[i]], color=c, s=80, zorder=2, edgecolor='k', alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["domain"], fontsize=11)
    ax.set_xlabel(r"$\theta$ Posterior Mean & 94% HDI (degrees)", fontsize=11)
    all_hi = max(df["hdi_high"].max(), df["mean"].max())
    ax.set_xlim(0, all_hi + 10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', label='Public Social-Category',
               markerfacecolor='#1f77b4', markersize=9, markeredgecolor='k', lw=3),
        Line2D([0],[0], marker='o', color='w', label='Self-Referential (display order)',
               markerfacecolor='#ff7f0e', markersize=9, markeredgecolor='k', lw=3),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=10)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, zorder=0)
    fig.tight_layout()
    out = FIG_DIR / "domain_theta_contrast_social_vs_self.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── 2. Table ─────────────────────────────────────────────────────────────

def regenerate_table(x, y_raw, mask):
    specs = [
        ("Interference",            "interference", THETA_SEARCH_GRID),
        ("Cosine only",             "cos_only",     THETA_SEARCH_GRID),
        ("Sine only",               "sin_only",     THETA_SEARCH_GRID),
        ("Polynomial (2nd order)",  "poly2",        None),
        ("Exponential",             "exp",          K_GRID),
        ("Power law",               "power",        P_GRID),
    ]
    rows = []
    for label, model_name, grid in specs:
        best_param, rmse = evaluate_model_rmse(x, y_raw, mask, model_name, grid)
        boundary_hit = bool(
            grid is not None and best_param is not None and
            (np.isclose(best_param, float(np.min(grid))) or np.isclose(best_param, float(np.max(grid))))
        )
        rows.append({"model": label, "best_param": best_param,
                      "rmse_test": rmse, "boundary_hit": boundary_hit})
        print(f"    {label}: param={best_param}, RMSE={rmse:.6f}, boundary={boundary_hit}")

    model_df = pd.DataFrame(rows).sort_values("rmse_test").reset_index(drop=True)
    model_df.to_csv(OUT_DIR / "external_life_satisfaction_model_comparison_display_order.csv", index=False)

    tab = model_df.copy()
    tab["best_param"] = tab["best_param"].map(lambda v: "---" if pd.isna(v) else f"{v:.2f}")
    tab["boundary_hit"] = tab["boundary_hit"].map(lambda f: "Yes" if f else "No")
    tab.columns = ["Model", "Best parameter", r"RMSE$_{test}$", "Boundary hit"]
    out = TAB_DIR / "Table_external_model_comparison.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write(tab.to_latex(index=False, escape=False))
    print(f"  Saved {out}")
    return model_df


# ── 3. Leverage figure ───────────────────────────────────────────────────

def regenerate_leverage_figure(ext_posterior, ext_summary, external_models):
    from run_external_leverage_analysis import (
        load_gender_science_curves, gender_science_dlike_quintiles,
        load_gender_science_full_summary,
    )
    gs_curves = load_gender_science_curves()
    gs_quintiles = gender_science_dlike_quintiles(gs_curves)
    gs_full = load_gender_science_full_summary()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    colors_q = ["#295c85","#5b8e7d","#c08a00","#c95d3a","#7b4d8d"]
    for color, (_, row) in zip(colors_q, gs_quintiles.sort_values("quintile").iterrows()):
        post = np.asarray(row["posterior"], dtype=float)
        axes[0].plot(THETA_COARSE_GRID, post/post.max(), lw=2, color=color,
                     label=f"{row['quintile']} (mean d={row['d_like_mean']:.2f})")
    axes[0].set_xlabel(r"$\theta$ (degrees)")
    axes[0].set_ylabel("Normalized posterior")
    axes[0].set_xlim(0, 40)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    gs_post_df = pd.read_csv(GS_POSTERIOR_CSV)
    gs_theta = gs_post_df["theta_deg"].to_numpy(dtype=float)
    gs_post  = gs_post_df["posterior"].to_numpy(dtype=float)
    axes[1].plot(gs_theta, gs_post/gs_post.max(), lw=2.2, color="#1f77b4", label="Gender-Science")
    axes[1].plot(THETA_FINE_GRID, ext_posterior/ext_posterior.max(), lw=2.2, color="#d55e00",
                 label="Life-satisfaction IAT\n(display order)")
    axes[1].axvline(gs_full["theta_map"], color="#1f77b4", ls="--", lw=1)
    axes[1].axvline(ext_summary["theta_map"], color="#d55e00", ls="--", lw=1)
    axes[1].set_xlabel(r"$\theta$ (degrees)")
    axes[1].set_ylabel("Normalized posterior")
    axes[1].set_xlim(0, 80)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    plot_df = external_models.copy()
    color_map = {"Interference":"#1f77b4","Cosine only":"#3a7ca5","Sine only":"#5fa8d3",
                 "Polynomial (2nd order)":"#8c8c8c","Exponential":"#a65e2e","Power law":"#c98544"}
    axes[2].bar(plot_df["model"], plot_df["rmse_test"],
                color=[color_map.get(m,"#999") for m in plot_df["model"]])
    axes[2].set_ylabel("RMSE (z-scored hold-out)")
    axes[2].tick_params(axis="x", labelrotation=35)
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = FIG_DIR / "external_leverage_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw archive with appearance-order encoding ...")
    trials = load_display_order_trials()

    print("\nBuilding display-order curves (standardized) ...")
    curves_std = build_curves_from_display_order(trials, standardize=True)
    x_s, y_s, mask_s = arrays_from_curves(curves_std)
    ext_posterior = profile_theta_posterior(x_s, y_s, mask_s, THETA_FINE_GRID)
    ext_summary = posterior_summary(THETA_FINE_GRID, ext_posterior)
    print(f"  Display-order posterior: MAP={ext_summary['theta_map']:.2f}, "
          f"mean={ext_summary['theta_mean']:.2f}, "
          f"HDI=[{ext_summary['hdi_low']:.2f},{ext_summary['hdi_high']:.2f}]")

    print("\nBuilding display-order curves (unstandardized) ...")
    curves_unstd = build_curves_from_display_order(trials, standardize=False)
    x_u, y_u, mask_u = arrays_from_curves(curves_unstd)

    print("\n[1/3] Regenerating domain_theta_contrast_social_vs_self.png ...")
    regenerate_forest_plot(ext_summary)

    print("\n[2/3] Regenerating Table_external_model_comparison.tex ...")
    ext_models = regenerate_table(x_u, y_u, mask_u)

    print("\n[3/3] Regenerating external_leverage_summary.png ...")
    regenerate_leverage_figure(ext_posterior, ext_summary, ext_models)

    print("\nDone.")


if __name__ == "__main__":
    main()
