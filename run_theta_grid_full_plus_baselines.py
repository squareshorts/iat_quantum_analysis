# ============================================================
# run_theta_grid_full_plus_baselines.py
# Interference θ profiling + baselines, ablations, robustness
# Fast OLS (no PyMC/JAX). Produces figures + LaTeX tables.
# ============================================================

import os, glob, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------- Config ---------------------
RANDOM_SEED = 123
DEFAULT_BINS = 6

# θ grid (degrees)
THETA_MIN, THETA_MAX, THETA_STEP = 0.0, 180.0, 0.25
HDI_PROB = 0.94

# Temporal hold-out evaluation
HOLDOUT_FRAC = 0.20       # cell-level holdout fraction
EVAL_SUBSAMPLE_N = 5000   # None = use all participants; else subsample for fast eval

# Baseline parameter grids
K_EXP_GRID = np.linspace(0.0, 6.0, 49)          # y ~ a + d * exp(-k x)
P_POW_GRID = np.linspace(0.25, 3.0, 56)         # y ~ a + d * x^p

# Optional robustness: run 4/6/8 bins summaries
RUN_BINS_SENSITIVITY = True
BINS_TO_TRY = [4, 6, 8]
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

# --------------------- Utilities ---------------------
def build_participant_curves(df: pd.DataFrame, n_bins=DEFAULT_BINS):
    need = {"pid", "block", "trial_in_block", "rt"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["block", "trial_in_block", "rt"])
    df = df[df["block"].isin([3,4,6,7])]

    out = []
    for pid, g in df.groupby("pid", sort=False):
        g = g.copy()
        g["pos_norm"] = g.groupby("block")["trial_in_block"].transform(
            lambda s: (s - s.min()) / max(1.0, (s.max() - s.min()))
        )
        vals = g[["pos_norm", "rt"]].to_numpy(dtype=float)
        if len(vals) < n_bins:
            continue
        q = np.quantile(vals[:,0], np.linspace(0,1,n_bins+1))
        xb, yb = [], []
        for i in range(n_bins):
            lo, hi = q[i], q[i+1]
            mask = (vals[:,0] >= lo) & (vals[:,0] <= (hi if i+1==n_bins else hi))
            chunk = vals[mask]
            if chunk.size == 0:
                continue
            xb.append(float(np.nanmean(chunk[:,0])))
            yb.append(float(np.nanmean(chunk[:,1])))
        if len(xb) < 3:
            continue
        y_mu, y_sd = np.mean(yb), np.std(yb, ddof=1) or 1.0
        y_std = (np.array(yb) - y_mu) / y_sd
        out.append(dict(pid=pid, x=np.array(xb, dtype=float), y=y_std, y_mu=y_mu, y_sd=y_sd))
    return out

def arrays_from_curves(curves):
    max_T = max(len(c["x"]) for c in curves)
    N = len(curves)
    X = np.full((N, max_T), np.nan)
    Y = np.full((N, max_T), np.nan)
    mask = np.zeros((N, max_T), dtype=bool)
    for i, c in enumerate(curves):
        t = len(c["x"])
        X[i,:t] = c["x"]
        Y[i,:t] = c["y"]
        mask[i,:t] = True
    return X, Y, mask

def hdi_from_discrete(grid_vals, probs, mass=HDI_PROB):
    idx = np.argsort(probs)[::-1]
    cum = 0.0
    chosen = []
    for i in idx:
        chosen.append(i)
        cum += probs[i]
        if cum >= mass:
            break
    chosen = np.array(chosen)
    lo = grid_vals[chosen].min()
    hi = grid_vals[chosen].max()
    return lo, hi

def posterior_summary(theta_grid, post):
    mean = float((theta_grid * post).sum())
    var = float(((theta_grid - mean)**2 * post).sum())
    sd = var**0.5
    lo, hi = hdi_from_discrete(theta_grid, post, mass=HDI_PROB)
    return mean, sd, lo, hi

# --------------------- Core: interference profile ---------------------
def grid_posterior_theta(X, Y, mask, theta_grid_deg):
    N, T = X.shape
    n_obs = mask.sum()
    y2_sum = np.sum((Y**2)[mask])
    post_log, rss_list = [], []
    ones = np.ones_like(X)

    for theta_deg in theta_grid_deg:
        arg = np.deg2rad(theta_deg) * X
        C = np.cos(arg); S = np.sin(arg)
        C[~mask] = 0.0; S[~mask] = 0.0
        O = ones.copy(); O[~mask] = 0.0
        Ym = np.where(mask, Y, 0.0)

        sO  = O.sum(axis=1)
        sC  = C.sum(axis=1)
        sS  = S.sum(axis=1)
        sCC = (C*C).sum(axis=1)
        sSS = (S*S).sum(axis=1)
        sCS = (C*S).sum(axis=1)
        sY  = Ym.sum(axis=1)
        sYC = (Ym*C).sum(axis=1)
        sYS = (Ym*S).sum(axis=1)

        A = np.stack([
            np.stack([sO, sC, sS], axis=-1),
            np.stack([sC, sCC, sCS], axis=-1),
            np.stack([sS, sCS, sSS], axis=-1)
        ], axis=-2)
        b = np.stack([sY, sYC, sYS], axis=-1)
        A[..., range(3), range(3)] += 1e-8
        proj = (b[...,None,:] @ np.linalg.solve(A, b[..., :, None])).squeeze(-1).squeeze(-1)
        rss_theta = y2_sum - proj.sum()
        rss_list.append(rss_theta)
        post_log.append(-(n_obs/2.0) * np.log(max(rss_theta, 1e-12)))

    post_log = np.array(post_log)
    post_log -= post_log.max()
    post = np.exp(post_log); post /= post.sum()
    return post, np.array(rss_list)

def coeffs_at_theta(X, Y, mask, theta_deg):
    arg = np.deg2rad(theta_deg) * X
    C = np.cos(arg); S = np.sin(arg)
    Cm = np.where(mask, C, 0.0)
    Sm = np.where(mask, S, 0.0)
    Om = np.where(mask, 1.0, 0.0)
    Ym = np.where(mask, Y, 0.0)

    sO  = Om.sum(axis=1)
    sC  = Cm.sum(axis=1)
    sS  = Sm.sum(axis=1)
    sCC = (Cm*Cm).sum(axis=1)
    sSS = (Sm*Sm).sum(axis=1)
    sCS = (Cm*Sm).sum(axis=1)
    sY  = Ym.sum(axis=1)
    sYC = (Ym*Cm).sum(axis=1)
    sYS = (Ym*Sm).sum(axis=1)

    A = np.stack([
        np.stack([sO, sC, sS], axis=-1),
        np.stack([sC, sCC, sCS], axis=-1),
        np.stack([sS, sCS, sSS], axis=-1)
    ], axis=-2)
    b  = np.stack([sY, sYC, sYS], axis=-1)
    A[..., range(3), range(3)] += 1e-8
    beta = np.linalg.solve(A, b[..., None])[...,0]
    return beta  # columns a,b,c

def coeffs_at_theta_by_block(df, theta_deg, n_bins=DEFAULT_BINS):
    """
    Compute (a,b,c) per participant *per block* at fixed theta.
    Returns DataFrame with pid, block, a, b, c.
    """
    rows = []
    for block in [3,4,6,7]:
        dfb = df[df["block"] == block]
        curves = build_participant_curves(dfb, n_bins=n_bins)
        if not curves:
            continue
        Xb, Yb, Mb = arrays_from_curves(curves)
        beta = coeffs_at_theta(Xb, Yb, Mb, theta_deg)
        for i, c in enumerate(curves):
            rows.append({
                "pid": c["pid"],
                "block": block,
                "a": beta[i,0],
                "b": beta[i,1],
                "c": beta[i,2],
            })
    out = pd.DataFrame(rows)
    out["phase"] = np.arctan2(out["c"], out["b"])
    out["phase_deg"] = np.rad2deg(out["phase"])
    return out

def make_ppc_plot(X, Y, mask, theta_deg, coeffs, out_png):
    a = coeffs[:,0][:,None]; b = coeffs[:,1][:,None]; c = coeffs[:,2][:,None]
    theta = np.deg2rad(theta_deg)
    mu = a + b*np.cos(theta*X) + c*np.sin(theta*X)
    resid = (Y - mu)[mask]
    sigma = float(np.std(resid, ddof=1))
    df = pd.DataFrame({"x":X[mask], "y":Y[mask], "mu":mu[mask]})
    df["bin"] = pd.cut(df["x"], bins=np.linspace(0,1,13), include_lowest=True, labels=False)
    agg = df.groupby("bin").agg(x=("x","mean"), y=("y","mean"), mu=("mu","mean")).reset_index(drop=True)

    plt.figure(figsize=(6.2,4.0))
    plt.plot(agg["x"], agg["y"], "o-", label="Observed (binned mean)")
    plt.plot(agg["x"], agg["mu"], "s--", label=f"Predicted @ θ={theta_deg:.1f}°")
    plt.xlabel("Within-block position (normalized)")
    plt.ylabel("Z-scored latency")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return sigma

# bump ridge globally
RIDGE = 1e-5

def compute_rss_and_beta(designs, Y, mask, min_obs_required=None):
    """
    designs: list of regressors (N,T)
    Only participants with >= min_obs_required training points are used.
    """
    N, T = Y.shape
    p = len(designs)
    need = min_obs_required if min_obs_required is not None else p
    nobs = mask.sum(axis=1)
    keep = nobs >= need
    if not keep.any():
        # no valid participants; return NaNs to signal skip
        return np.nan, np.full((0, p), np.nan), keep

    # slice to valid subset
    Yv   = Y[keep]
    mv   = mask[keep]
    R    = [np.where(mv, D[keep], 0.0) for D in designs]

    # build normal equations
    A = np.empty((keep.sum(), p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            A[:, i, j] = (R[i]*R[j]).sum(axis=1)
    b = np.empty((keep.sum(), p), dtype=float)
    for k in range(p):
        b[:, k] = (Yv*R[k]).sum(axis=1)

    A[..., range(p), range(p)] += RIDGE
    beta = np.linalg.solve(A, b[..., None])[..., 0]
    proj = (b[...,None,:] @ np.linalg.solve(A, b[..., :, None])).squeeze(-1).squeeze(-1)
    rss_total = float((Yv[mv]**2).sum() - proj.sum())
    return rss_total, beta, keep

def rmse_on_mask(designs, beta, Y, eval_mask, keep):
    Yv = Y[keep]
    mv = eval_mask[keep]
    if not mv.any():
        return np.nan
    mu = np.zeros_like(Yv)
    p  = len(designs)
    for j in range(p):
        mu += beta[:, j][:, None] * designs[j][keep]
    resid = (Yv - mu)[mv]
    return float(np.sqrt(np.mean(resid**2)))

# --------------------- Model builders ---------------------
def designs_interference(X, theta_deg):
    arg = np.deg2rad(theta_deg) * X
    return [np.ones_like(X), np.cos(arg), np.sin(arg)]

def designs_cos_only(X, theta_deg):
    arg = np.deg2rad(theta_deg) * X
    return [np.ones_like(X), np.cos(arg)]

def designs_sin_only(X, theta_deg):
    arg = np.deg2rad(theta_deg) * X
    return [np.ones_like(X), np.sin(arg)]

def designs_poly2(X):
    return [np.ones_like(X), X, X**2]

def designs_expdecay(X, k):
    return [np.ones_like(X), np.exp(-k*X)]

def designs_powerlaw(X, p):
    Xp = np.where(X>0, X**p, 0.0)
    return [np.ones_like(X), Xp]

# --------------------- Hold-out evaluation ---------------------
# --- NEW: temporal, per-participant holdout by x-quantile ---
def temporal_holdout_mask(X, mask, frac, min_train_pts):
    """
    Rank-based temporal holdout per participant (robust to ties).
    Holds out the highest-x bins.
    """
    N, T = X.shape
    train_mask = np.zeros_like(mask, dtype=bool)
    test_mask  = np.zeros_like(mask, dtype=bool)
    valid = np.zeros(N, dtype=bool)

    for i in range(N):
        mi = mask[i]
        idx = np.where(mi)[0]
        n = idx.size
        if n < min_train_pts + 1:
            continue

        # sort observed bins by x
        order = idx[np.argsort(X[i, idx])]

        k = max(1, int(np.ceil(frac * n)))
        if n - k < min_train_pts:
            k = n - min_train_pts

        test_idx = order[-k:]
        train_idx = order[:-k]

        train_mask[i, train_idx] = True
        test_mask[i, test_idx] = True
        valid[i] = True

    return train_mask, test_mask, valid

def leak_free_standardize(Y, train_mask, eps=1e-9):
    """
    Per-participant z-score using ONLY training bins.
    Y, train_mask: (N,T)
    Returns Yz (same shape), plus vectors of means/sds if you want to inspect.
    """
    N, T = Y.shape
    Yz = np.full_like(Y, np.nan, dtype=float)
    means = np.zeros(N); sds = np.ones(N)
    for i in range(N):
        mi = train_mask[i]
        if not mi.any():
            continue
        yi_tr = Y[i, mi]
        mu = float(np.mean(yi_tr))
        sd = float(np.std(yi_tr, ddof=1))
        if not np.isfinite(sd) or sd < eps:
            sd = 1.0
        means[i] = mu; sds[i] = sd
        sel = ~np.isnan(Y[i])  # standardize only observed cells
        Yz[i, sel] = (Y[i, sel] - mu) / sd
    return Yz, means, sds


def eval_model_rmse(X, Y, mask, builder, grid=None, label="model",
                    subsample_n=EVAL_SUBSAMPLE_N):
    # optional participant subsample
    if subsample_n is not None and subsample_n < X.shape[0]:
        idx = rng.choice(X.shape[0], size=subsample_n, replace=False)
        Xs, Ys, ms = X[idx], Y[idx], mask[idx]
    else:
        Xs, Ys, ms = X, Y, mask

    # number of regressors we will fit
    p = len(builder(Xs, (grid[0] if grid is not None else 0.0))
            if grid is not None else builder(Xs))

    # temporal holdout and min-train constraint
    MIN_TRAIN_PTS = p + 2

    train_mask, test_mask, keep_temporal = temporal_holdout_mask(
        Xs, ms, HOLDOUT_FRAC, min_train_pts=MIN_TRAIN_PTS
    )

    # ---- LEAK-FREE STANDARDIZATION (train-only) ----
    Yz, _, _ = leak_free_standardize(Ys, train_mask)

    curve = []
    if grid is None:  # parameter-free model
        designs = builder(Xs)
        rss, beta, keep_ols = compute_rss_and_beta(designs, Yz, train_mask, min_obs_required=p)
        keep_final = keep_temporal & keep_ols
        rmse = rmse_on_mask(designs, beta, Yz, test_mask, keep_final)
        return None, rmse, curve

    best_param, best_rmse = None, np.inf
    for param in grid:
        designs = builder(Xs, param)
        rss, beta, keep_ols = compute_rss_and_beta(designs, Yz, train_mask, min_obs_required=p)
        if np.isnan(rss):
            continue
        keep_final = keep_temporal & keep_ols
        rmse = rmse_on_mask(designs, beta, Yz, test_mask, keep_final)
        if not np.isfinite(rmse):
            continue
        curve.append((float(param), rmse))
        if rmse < best_rmse:

            best_rmse, best_param = rmse, float(param)
    return best_param, best_rmse, curve


# --------------------- Main ---------------------
if __name__ == "__main__":
    print("✅ Full run with baselines starting…")
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUT_DIR  = os.path.join(BASE_DIR, "outputs")
    FIG_DIR  = os.path.join(BASE_DIR, "figures")
    TAB_DIR  = os.path.join(BASE_DIR, "tables")
    for d in (OUT_DIR, FIG_DIR, TAB_DIR):
        os.makedirs(d, exist_ok=True)

    # ------------ Load data ------------
    gs_dir = os.path.join(DATA_DIR, "GenderScience_iat_2019", "iat_2019")
    paths = glob.glob(os.path.join(gs_dir, "iat*.txt"))
    if not paths:
        raise FileNotFoundError(f"No iat*.txt files found in {gs_dir}")
    dfs = [pd.read_csv(p, sep="\t", low_memory=False) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "session_id":"pid",
        "block_number":"block",
        "trial_number":"trial_in_block",
        "trial_latency":"rt",
    }).dropna(subset=["pid"])
    df = df[["pid","block","trial_in_block","rt"]]
    print(f"Loaded trial-level logs: {df.shape[0]:,} rows")

    # ------------ Curves (cache by bins) ------------
    cache_pkl = os.path.join(OUT_DIR, f"curves_cache_bins{DEFAULT_BINS}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            curves = pickle.load(f)
        print(f"Loaded cached curves ({len(curves):,} participants) for {DEFAULT_BINS} bins.")
    else:
        curves = build_participant_curves(df, n_bins=DEFAULT_BINS)
        with open(cache_pkl, "wb") as f:
            pickle.dump(curves, f)
        print(f"✅ Built and cached {len(curves):,} participant curves ({DEFAULT_BINS} bins).")

    X, Y, mask = arrays_from_curves(curves)
    N, T = X.shape
    print(f"Arrays: X={X.shape}, Y={Y.shape} (participants={N:,})")

    # ------------ θ posterior + refined grid ------------
    theta_grid = np.arange(THETA_MIN, THETA_MAX + THETA_STEP/2, THETA_STEP)
    t0 = time.time()
    post, rss = grid_posterior_theta(X, Y, mask, theta_grid)
    print(f"Computed θ posterior on grid of {len(theta_grid)} in {(time.time()-t0)/60:.2f} min.")
    pd.DataFrame({"theta_deg":theta_grid, "posterior":post, "rss":rss}).to_csv(
        os.path.join(OUT_DIR, "theta_grid_profile.csv"), index=False
    )

    mean, sd, lo, hi = posterior_summary(theta_grid, post)
    theta_map = float(theta_grid[np.argmax(post)])
    print(f"θ mean={mean:.2f}°, sd={sd:.2f}°, {int(HDI_PROB*100)}% HDI [{lo:.2f}°, {hi:.2f}°]; MAP={theta_map:.2f}°")

    # refined grid around MAP
    refine_lo = max(0.0, theta_map - 2.0)
    refine_hi = min(180.0, theta_map + 2.0)
    theta_refine = np.arange(refine_lo, refine_hi + 1e-6, 0.05)
    post_ref, rss_ref = grid_posterior_theta(X, Y, mask, theta_refine)
    pd.DataFrame({"theta_deg":theta_refine, "posterior":post_ref, "rss":rss_ref}).to_csv(
        os.path.join(OUT_DIR, "theta_grid_profile_refined.csv"), index=False
    )

    # figure (coarse)
    plt.figure(figsize=(6.2,4.0))
    plt.plot(theta_grid, post/post.max(), "-", lw=2)
    plt.axvline(mean, ls="--", alpha=0.6)
    plt.axvspan(lo, hi, color="gray", alpha=0.2, label=f"{int(HDI_PROB*100)}% HDI")
    plt.xlabel("θ (degrees)"); plt.ylabel("Posterior density (normalized)")
    plt.grid(alpha=0.25); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theta_posterior_full.png"), dpi=300); plt.close()

    # figure (refined)
    plt.figure(figsize=(6.2,4.0))
    plt.plot(theta_refine, post_ref/post_ref.max(), "-", lw=2)
    plt.xlabel("θ (degrees)"); plt.ylabel("Posterior density (normalized)")
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theta_posterior_refined.png"), dpi=300); plt.close()

    # ------------ Coeffs at MAP + PPC ------------
    beta = coeffs_at_theta(X, Y, mask, theta_map)
    coef_df = pd.DataFrame(beta, columns=["a","b","c"])
    coef_df.insert(0, "pid", [c["pid"] for c in curves])
    coef_df.to_parquet(os.path.join(OUT_DIR, "theta_map_coefficients.parquet"), index=False)
    # ------------ Phase analysis (Option C) ------------
    coef_df["phase"] = np.arctan2(coef_df["c"], coef_df["b"])  # radians in [-pi, pi]
    coef_df["phase_deg"] = np.rad2deg(coef_df["phase"])

    coef_df[["pid", "phase", "phase_deg"]].to_csv(
        os.path.join(OUT_DIR, "theta_phase_angles.csv"), index=False
    )

    sigma = make_ppc_plot(X, Y, mask, theta_map, beta, os.path.join(FIG_DIR, "ppc_full.png"))
    print(f"Estimated residual σ ≈ {sigma:.3f}")

    # ------------ Phase by block (Option C) ------------
    phase_block_df = coeffs_at_theta_by_block(df, theta_map, n_bins=DEFAULT_BINS)
    phase_block_df.to_csv(
        os.path.join(OUT_DIR, "theta_phase_by_block.csv"), index=False
    )
    print("Saved block-wise phase angles to outputs/theta_phase_by_block.csv")


    # ------------ Ablations + Baselines (hold-out RMSE) ------------
    print("Evaluating models on temporal hold-out… (using participant subsample for speed)")
    # Interference full (θ grid centered on MAP for evaluation)
    best_theta, rmse_int, curve_int = eval_model_rmse(
        X, Y, mask, builder=designs_interference,
        grid=np.arange(max(0, theta_map-2), min(180, theta_map+2)+1e-9, 0.1),
        label="interference"
    )
    # Cos-only / Sin-only (ablation)
    best_t_cos, rmse_cos, _ = eval_model_rmse(X, Y, mask, designs_cos_only,
                                              grid=np.arange(max(0, theta_map-2), min(180, theta_map+2)+1e-9, 0.1))
    best_t_sin, rmse_sin, _ = eval_model_rmse(X, Y, mask, designs_sin_only,
                                              grid=np.arange(max(0, theta_map-2), min(180, theta_map+2)+1e-9, 0.1))
    # Poly (no parameter)
    _, rmse_poly, _ = eval_model_rmse(X, Y, mask, designs_poly2, grid=None, label="poly2")
    # Exp decay / Power law
    best_k, rmse_exp, curve_exp = eval_model_rmse(X, Y, mask, designs_expdecay, grid=K_EXP_GRID, label="expdecay")
    best_p, rmse_pow, curve_pow = eval_model_rmse(X, Y, mask, designs_powerlaw, grid=P_POW_GRID, label="powerlaw")

    # Comparison table
    comp = pd.DataFrame([
        {"Model": "Interference (full)", "Best param": best_theta, "RMSE (holdout)": rmse_int},
        {"Model": "Interference (cos-only)", "Best param": best_t_cos, "RMSE (holdout)": rmse_cos},
        {"Model": "Interference (sin-only)", "Best param": best_t_sin, "RMSE (holdout)": rmse_sin},
        {"Model": r"Polynomial ($a + d_1 x + d_2 x^2$)", "Best param": np.nan, "RMSE (holdout)": rmse_poly},
        {"Model": r"Exponential decay ($a + d e^{-k x}$)", "Best param": best_k, "RMSE (holdout)": rmse_exp},
        {"Model": r"Power law ($a + d x^p$)", "Best param": best_p, "RMSE (holdout)": rmse_pow},
    ])
    comp_path = os.path.join(TAB_DIR, "Table_model_comparison.tex")
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(comp.to_latex(index=False, float_format="%.4f", na_rep=""))

    # Bar plot of RMSEs
    plt.figure(figsize=(7.2,4.2))
    labels = comp["Model"].tolist()
    vals = comp["RMSE (holdout)"].to_numpy()
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Hold-out RMSE")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "model_comparison_rmse.png"), dpi=300); plt.close()

    # Optional: parameter curves for exp/power (diagnostic)
    if curve_exp:
        k_vals, k_rmse = zip(*curve_exp)
        plt.figure(figsize=(6.2,4.0))
        plt.plot(k_vals, k_rmse, "-o", ms=3)
        plt.xlabel("k"); plt.ylabel("Hold-out RMSE")
        plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "baseline_exp_decay_curve.png"), dpi=300); plt.close()
    if curve_pow:
        p_vals, p_rmse = zip(*curve_pow)
        plt.figure(figsize=(6.2,4.0))
        plt.plot(p_vals, p_rmse, "-o", ms=3)
        plt.xlabel("p"); plt.ylabel("Hold-out RMSE")
        plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "baseline_power_curve.png"), dpi=300); plt.close()

    # ------------ LaTeX tables (θ + fit quality) ------------
    theta_tab = pd.DataFrame([{
        "Parameter": r"$\theta$ (deg)",
        "Mean": mean, "SD": sd,
        f"{int(HDI_PROB*100)}% HDI low": lo,
        f"{int(HDI_PROB*100)}% HDI high": hi
    }])
    with open(os.path.join(TAB_DIR, "Table_theta_summary.tex"), "w", encoding="utf-8") as f:
        f.write(theta_tab.to_latex(index=False, float_format="%.2f"))

    fit_tab = pd.DataFrame([{
        "N participants": X.shape[0],
        "Points/participant (max)": X.shape[1],
        r"$\theta_{\text{MAP}}$ (deg)": theta_map,
        r"Residual $\sigma$": sigma
    }])
    with open(os.path.join(TAB_DIR, "Table_fit_quality.tex"), "w", encoding="utf-8") as f:
        f.write(fit_tab.to_latex(index=False, float_format="%.3f"))

    # ------------ Robustness across bin counts (optional) ------------
    if RUN_BINS_SENSITIVITY:
        rows = []
        for nb in BINS_TO_TRY:
            cache_nb = os.path.join(OUT_DIR, f"curves_cache_bins{nb}.pkl")
            if os.path.exists(cache_nb):
                with open(cache_nb, "rb") as f:
                    curv_nb = pickle.load(f)
            else:
                curv_nb = build_participant_curves(df, n_bins=nb)
                with open(cache_nb, "wb") as f:
                    pickle.dump(curv_nb, f)
            Xb, Yb, Mb = arrays_from_curves(curv_nb)
            post_b, _ = grid_posterior_theta(Xb, Yb, Mb, theta_grid)
            mean_b, sd_b, lo_b, hi_b = posterior_summary(theta_grid, post_b)
            rows.append({"Bins": nb, "θ mean": mean_b, "θ sd": sd_b,
                         f"{int(HDI_PROB*100)}% HDI low": lo_b,
                         f"{int(HDI_PROB*100)}% HDI high": hi_b,
                         "Participants": Xb.shape[0]})
        bins_tab = pd.DataFrame(rows).sort_values("Bins")
        with open(os.path.join(TAB_DIR, "Table_theta_bins_robustness.tex"), "w", encoding="utf-8") as f:
            f.write(bins_tab.to_latex(index=False, float_format="%.2f"))
        # quick plot
        plt.figure(figsize=(6.2,4.0))
        plt.errorbar(bins_tab["Bins"], bins_tab["θ mean"],
                     yerr=bins_tab["θ sd"], fmt="o--")
        plt.xlabel("Bins per participant"); plt.ylabel("θ (mean ± sd)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "theta_bins_robustness.png"), dpi=300); plt.close()

    print("✅ Run complete: posterior, PPC, baselines, ablations, robustness.")
