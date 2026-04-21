# ============================================================
# run_theta_grid_full.py
# Fast full-run estimator for θ using closed-form OLS per participant
# Produces publication figures + LaTeX tables WITHOUT PyMC/JAX
# ============================================================

import os, glob, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------- Config ---------------------
RANDOM_SEED = 123
BINS = 6

# θ grid range (in degrees)
THETA_MIN, THETA_MAX, THETA_STEP = 0.0, 180.0, 0.25  # degrees
theta_grid = np.arange(THETA_MIN, THETA_MAX + THETA_STEP/2.0, THETA_STEP)
HDI_PROB = 0.94
# --------------------------------------------------


rng = np.random.default_rng(RANDOM_SEED)

def build_participant_curves(df: pd.DataFrame, n_bins=BINS):
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

def grid_posterior_theta(X, Y, mask, theta_grid_deg):
    """
    For each theta, fit y ~ a + b*cos(theta*x) + c*sin(theta*x) per participant in closed form
    and compute total RSS. Convert profile likelihood over theta to a normalized posterior
    using Jeffreys/reference prior for sigma (proportional to (RSS)^(-n/2)) or simply
    exp(-RSS/(2*sigma^2)) with sigma^2 profiled out.
    """
    N, T = X.shape
    n_obs = mask.sum()

    # y^T y term (constant over theta)
    y2_sum = np.sum((Y**2)[mask])

    post_log = []
    rss_list = []

    # Work buffers
    ones = np.ones_like(X)
    for theta_deg in theta_grid_deg:
        theta_rad = np.deg2rad(theta_deg)         # degrees -> radians
        arg = theta_rad * X
        C = np.cos(arg)
        S = np.sin(arg)

        C[~mask] = 0.0; S[~mask] = 0.0
        O = ones.copy(); O[~mask] = 0.0
        Ymask = Y.copy(); Ymask[mask==False] = 0.0

        # Sufficient stats per participant
        sO  = O.sum(axis=1)                       # Σ 1
        sC  = C.sum(axis=1)                       # Σ cos
        sS  = S.sum(axis=1)                       # Σ sin
        sCC = (C*C).sum(axis=1)                   # Σ cos^2
        sSS = (S*S).sum(axis=1)                   # Σ sin^2
        sCS = (C*S).sum(axis=1)                   # Σ cos·sin
        sY  = Ymask.sum(axis=1)                   # Σ y
        sYC = (Ymask*C).sum(axis=1)               # Σ y·cos
        sYS = (Ymask*S).sum(axis=1)               # Σ y·sin

        # Build (X'X) and (X'y) for each participant:
        # X'X = [[sO, sC, sS],
        #        [sC, sCC, sCS],
        #        [sS, sCS, sSS]]
        A = np.stack([
            np.stack([sO, sC, sS], axis=-1),
            np.stack([sC, sCC, sCS], axis=-1),
            np.stack([sS, sCS, sSS], axis=-1)
        ], axis=-2)                   # shape (N, 3, 3)
        b = np.stack([sY, sYC, sYS], axis=-1)     # shape (N, 3)

        # Solve A^{-1} b in batch; add tiny ridge to avoid singulars with T<3
        ridge = 1e-8
        A[..., range(3), range(3)] += ridge
        beta = np.linalg.solve(A, b[..., None])   # (N,3,1)
        beta = beta[...,0]                        # (N,3)

        # Compute projection term y'X (X'X)^{-1} X'y = b^T A^{-1} b
        proj = (b[...,None,:] @ np.linalg.solve(A, b[..., :, None])).squeeze(-1).squeeze(-1)  # (N,)

        # Total RSS(theta) = y'y - sum_i b_i^T A_i^{-1} b_i
        rss_theta = y2_sum - proj.sum()
        rss_list.append(rss_theta)

        # Using Gaussian model with sigma profiled out, the (unnormalized) log marginal likelihood
        # over θ is proportional to  - (n_obs/2) * log(RSS_theta)
        post_log.append(-(n_obs/2.0) * np.log(max(rss_theta, 1e-12)))

    post_log = np.array(post_log)
    # stabilize
    post_log -= post_log.max()
    post = np.exp(post_log)
    post /= post.sum()
    return post, np.array(rss_list)

def hdi_from_discrete(grid_vals, probs, mass=0.94):
    """Compute HDI on a discrete grid."""
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

def make_ppc_plot(X, Y, mask, theta_deg, coeffs, out_png):
    # coeffs: (N,3) for [a,b,c] at MAP theta
    a = coeffs[:,0][:,None]; b = coeffs[:,1][:,None]; c = coeffs[:,2][:,None]
    theta = np.deg2rad(theta_deg)
    mu = a + b * np.cos(theta * X) + c * np.sin(theta * X)
    resid = (Y - mu)[mask]
    sigma = float(np.std(resid, ddof=1))
    # Aggregate across participants to a mean curve over x
    x_flat = X[mask]; y_flat = Y[mask]; mu_flat = mu[mask]
    df = pd.DataFrame({"x":x_flat, "y":y_flat, "mu":mu_flat})
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

if __name__ == "__main__":
    print("✅ Fast publication run (grid OLS) starting…")
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUT_DIR  = os.path.join(BASE_DIR, "outputs")
    FIG_DIR  = os.path.join(BASE_DIR, "figures")
    TAB_DIR  = os.path.join(BASE_DIR, "tables")
    for d in (OUT_DIR, FIG_DIR, TAB_DIR):
        os.makedirs(d, exist_ok=True)

    # ------------ Load data or cached curves ------------
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

    cache_pkl = os.path.join(OUT_DIR, "curves_cache.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            curves = pickle.load(f)
        print(f"Loaded cached curves ({len(curves):,} participants).")
    else:
        curves = build_participant_curves(df, n_bins=BINS)
        with open(cache_pkl, "wb") as f:
            pickle.dump(curves, f)
        print(f"✅ Built and cached {len(curves):,} participant curves.")

    X, Y, mask = arrays_from_curves(curves)
    N, T = X.shape
    print(f"Arrays prepared: X={X.shape}, Y={Y.shape} (participants={N:,})")
    print("x range:", float(np.nanmin(X)), "to", float(np.nanmax(X)))  # should be 0..~1
    print("any NaNs in X?", np.isnan(X).any(), " any NaNs in Y?", np.isnan(Y).any())


    # ------------ θ grid posterior ------------
    theta_grid = np.arange(THETA_MIN, THETA_MAX + THETA_STEP/2, THETA_STEP)
    t0 = time.time()
    post, rss = grid_posterior_theta(X, Y, mask, theta_grid)
    elapsed = time.time() - t0
    print(f"Computed θ posterior on grid of {len(theta_grid)} in {elapsed/60:.2f} min.")

    # Save grid for reproducibility
    pd.DataFrame({"theta_deg":theta_grid, "posterior":post, "rss":rss}).to_csv(
        os.path.join(OUT_DIR, "theta_grid_profile.csv"), index=False
    )

    # Posterior summary + figure
    mean, sd, lo, hi = posterior_summary(theta_grid, post)
    print(f"θ mean={mean:.2f}°, sd={sd:.2f}°, {int(HDI_PROB*100)}% HDI [{lo:.2f}°, {hi:.2f}°]")

    plt.figure(figsize=(6.2,4.0))
    plt.plot(theta_grid, post/post.max(), "-", lw=2)
    plt.axvline(mean, ls="--", alpha=0.6)
    plt.axvspan(lo, hi, color="gray", alpha=0.2, label=f"{int(HDI_PROB*100)}% HDI")
    plt.xlabel("θ (degrees)")
    plt.ylabel("Posterior density (normalized)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theta_posterior_full.png"), dpi=300)
    plt.close()

    # ------------ Fit coefficients at MAP θ, PPC ------------
    theta_map = float(theta_grid[np.argmax(post)])
    print(f"θ MAP = {theta_map:.2f}°")

    # Closed-form per-participant coefficients at θ_MAP
    th = np.deg2rad(theta_map)
    arg = th * X                        # <- same as in grid
    C = np.cos(th * X); S = np.sin(th * X)
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
    ], axis=-2)  # (N,3,3)
    b  = np.stack([sY, sYC, sYS], axis=-1)  # (N,3)

    A[..., range(3), range(3)] += 1e-8
    beta = np.linalg.solve(A, b[..., None])[...,0]  # (N,3)
    coef_df = pd.DataFrame(beta, columns=["a","b","c"])
    coef_df.insert(0, "pid", [c["pid"] for c in curves])
    coef_df.to_parquet(os.path.join(OUT_DIR, "theta_map_coefficients.parquet"), index=False)

    sigma = make_ppc_plot(X, Y, mask, theta_map, beta, os.path.join(FIG_DIR, "ppc_full.png"))
    print(f"Estimated residual σ ≈ {sigma:.3f}")

    # ------------ LaTeX tables ------------
    # θ summary table
    theta_tab = pd.DataFrame([{
        "Parameter": r"$\theta$ (deg)",
        "Mean": mean, "SD": sd,
        f"{int(HDI_PROB*100)}% HDI low": lo,
        f"{int(HDI_PROB*100)}% HDI high": hi
    }])
    with open(os.path.join(TAB_DIR, "Table_theta_summary.tex"), "w", encoding="utf-8") as f:
        f.write(theta_tab.to_latex(index=False, float_format="%.2f"))

    # fit quality table (overall)
    fit_tab = pd.DataFrame([{
        "N participants": X.shape[0],
        "Points/participant (max)": X.shape[1],
        "θ MAP (deg)": theta_map,
        "Residual σ": sigma
    }])
    with open(os.path.join(TAB_DIR, "Table_fit_quality.tex"), "w", encoding="utf-8") as f:
        f.write(fit_tab.to_latex(index=False, float_format="%.3f"))

    print("✅ Fast publication run complete.")
