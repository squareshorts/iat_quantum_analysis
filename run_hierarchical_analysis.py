# ============================================================
# run_hierarchical_analysis.py
# Publication run — Hierarchical PyMC (Gender–Science IAT)
# Final figures + LaTeX tables; JAX/NumPyro (CPU) backend
# ============================================================

# ---- Minimal PyTensor flags (avoid slow pure-Python linker warnings) ----
import os
os.environ.setdefault("PYTENSOR_FLAGS", "mode=FAST_COMPILE,device=cpu,openmp=False")
# Optional: force JAX to CPU explicitly (harmless on Windows)
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import glob, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

# ----------------- Runtime knobs (publication) -----------------
RANDOM_SEED   = 123
BINS          = 6
CHAINS        = 4
DRAWS         = 2000         # robust posterior
TUNE          = 1500
TARGET_ACCEPT = 0.995        # reduce divergences
CORES         = CHAINS       # not used by JAX path, but harmless
# ---------------------------------------------------------------

# Prefer JAX/NumPyro if available (CPU JAX is fine)
USE_JAX = False
try:
    import pymc.sampling_jax as pmjax
    import jax
    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print(f"✅ JAX devices: {jax.devices()}")
    USE_JAX = True
except Exception:
    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print("ℹ️ JAX/NumPyro not available; using default PyMC NUTS.")

# ============================================================
# Helpers
# ============================================================

def build_participant_curves(df: pd.DataFrame, n_bins=BINS):
    """
    Build standardized x/y curves (length 3..n_bins) per participant.
    Keeps only critical IAT blocks (3,4,6,7).
    y is z-scored per participant to align with the Gaussian likelihood scale.
    """
    need = {"pid", "block", "trial_in_block", "rt"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["block", "trial_in_block", "rt"])
    df = df[df["block"].isin([3, 4, 6, 7])]

    out = []
    for pid, g in df.groupby("pid", sort=False):
        g = g.copy()
        # normalize position within block to [0,1]
        g["pos_norm"] = g.groupby("block")["trial_in_block"].transform(
            lambda s: (s - s.min()) / max(1.0, (s.max() - s.min()))
        )
        vals = g[["pos_norm", "rt"]].to_numpy(dtype=float)
        if len(vals) < n_bins:
            continue

        q = np.quantile(vals[:, 0], np.linspace(0, 1, n_bins + 1))
        x_bin, y_bin = [], []
        for i in range(n_bins):
            lo, hi = q[i], q[i + 1]
            # include upper bound in the last bin
            mask = (vals[:, 0] >= lo) & (vals[:, 0] <= hi if i + 1 == n_bins else vals[:, 0] < hi)
            chunk = vals[mask]
            if chunk.size == 0:
                continue
            x_bin.append(float(np.nanmean(chunk[:, 0])))
            y_bin.append(float(np.nanmean(chunk[:, 1])))

        if len(x_bin) < 3:
            continue

        y_mu, y_sd = np.mean(y_bin), np.std(y_bin, ddof=1) or 1.0
        y_std = (np.array(y_bin) - y_mu) / y_sd
        out.append(dict(pid=pid, x=np.array(x_bin), y=y_std, y_mu=y_mu, y_sd=y_sd))

    return out


def arrays_from_curves(curves):
    max_T = max(len(c["x"]) for c in curves)
    N = len(curves)
    # Use float32 to reduce memory and speed JAX
    X = np.full((N, max_T), np.nan, dtype=np.float32)
    Y = np.full((N, max_T), np.nan, dtype=np.float32)
    T = np.zeros(N, dtype=np.int32)
    for i, c in enumerate(curves):
        T[i] = len(c["x"])
        X[i, :T[i]] = c["x"].astype(np.float32)
        Y[i, :T[i]] = c["y"].astype(np.float32)
    return X, Y, T


def mu_quantum_x(x, a, b, theta_deg, f):
    # convert degrees → radians without pm.math.deg2rad
    theta_rad = (theta_deg * np.pi) / 180.0
    return a + b * pm.math.cos(theta_rad * x + f)


def build_model(X, Y):
    """Create hierarchical model; return model and mask."""
    N, _ = X.shape
    mask = ~np.isnan(Y)

    with pm.Model() as model:
        # Hyperpriors (tight-ish to tame divergences)
        a_mu = pm.Normal("a_mu", 0.0, 2.0)
        a_sd = pm.HalfNormal("a_sd", 0.75)

        b_mu = pm.Normal("b_mu", 0.0, 2.0)
        b_sd = pm.HalfNormal("b_sd", 0.75)

        f_mu = pm.Normal("f_mu", 0.0, 1.0)
        f_sd = pm.HalfNormal("f_sd", 0.35)

        # Center theta near mid-90s with wide support
        theta = pm.TruncatedNormal("theta", mu=95.0, sigma=20.0, lower=30.0, upper=150.0)

        # Random effects per participant
        a_i = pm.Normal("a_i", a_mu, a_sd, shape=N)
        b_i = pm.Normal("b_i", b_mu, b_sd, shape=N)
        f_i = pm.Normal("f_i", f_mu, f_sd, shape=N)

        sigma = pm.HalfNormal("sigma", 0.5)

        mu = mu_quantum_x(X, a_i[:, None], b_i[:, None], theta, f_i[:, None])

        pm.Normal("y_obs", mu=mu[mask], sigma=sigma, observed=Y[mask])

    return model, mask


def fit_with_jax(model):
    """NumPyro/JAX sampler (vectorized chains), with PPC and WAIC/LOO if possible."""
    with model:
        idata = pmjax.sample_numpyro_nuts(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            chain_method="vectorized",           # key speedup on CPU JAX
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            idata_kwargs={"log_likelihood": True},
        )
        ppc = pm.sample_posterior_predictive(
            idata, extend_inferencedata=True, random_seed=RANDOM_SEED, predictions=True
        )
        idata.extend(ppc)
        loo = waic = None
        try:
            waic = az.waic(idata, scale="deviance")
            loo  = az.loo(idata, pointwise=False)
        except Exception as e:
            print("⚠️ WAIC/LOO not computed (JAX):", e)
        return idata, loo, waic


def fit_with_default(model):
    """Fallback PyMC NUTS (not recommended for full N without JAX)."""
    with model:
        idata = pm.sample(
            draws=DRAWS, tune=TUNE,
            chains=CHAINS, cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            idata_kwargs={"log_likelihood": True},
        )
        ppc = pm.sample_posterior_predictive(
            idata, extend_inferencedata=True, random_seed=RANDOM_SEED, predictions=True
        )
        idata.extend(ppc)
        loo = waic = None
        try:
            waic = az.waic(idata, scale="deviance")
            loo  = az.loo(idata, pointwise=False)
        except Exception as e:
            print("⚠️ WAIC/LOO not computed:", e)
        return idata, loo, waic


def save_theta_hdi_table(idata, out_dir):
    th = idata.posterior["theta"].values.ravel()
    hdi = az.hdi(th, hdi_prob=0.94)
    df = pd.DataFrame([{
        "Parameter": "$\\theta$",
        "Mean": float(th.mean()),
        "94\\% HDI low": float(hdi[0]),
        "94\\% HDI high": float(hdi[1]),
    }])
    with open(os.path.join(out_dir, "Table_theta_hdi.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print("✅ Publication run starting…")

    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUT_DIR  = os.path.join(BASE_DIR, "outputs")
    FIG_DIR  = os.path.join(BASE_DIR, "figures")
    TAB_DIR  = os.path.join(BASE_DIR, "tables")
    for d in (OUT_DIR, FIG_DIR, TAB_DIR):
        os.makedirs(d, exist_ok=True)

    # ---- Load data (Gender–Science) ----
    gs_dir = os.path.join(DATA_DIR, "GenderScience_iat_2019", "iat_2019")
    paths = glob.glob(os.path.join(gs_dir, "iat*.txt"))
    if not paths:
        raise FileNotFoundError(f"No iat*.txt files found in {gs_dir}")

    dfs = [pd.read_csv(p, sep="\t", low_memory=False) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "session_id":     "pid",
        "block_number":   "block",
        "trial_number":   "trial_in_block",
        "trial_latency":  "rt",
    }).dropna(subset=["pid"])

    df = df[["pid", "block", "trial_in_block", "rt"]]
    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print(f"Loaded trial-level logs: {df.shape[0]:,} rows")

    # ---- Curves (cache) ----
    cache_pkl = os.path.join(OUT_DIR, "curves_cache.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            curves = pickle.load(f)
        if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
            print(f"Loaded cached curves ({len(curves):,} participants).")
    else:
        curves = build_participant_curves(df, n_bins=BINS)
        with open(cache_pkl, "wb") as f:
            pickle.dump(curves, f)
        if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
            print(f"✅ Built and cached {len(curves):,} participant curves.")

    X_all, Y_all, T_all = arrays_from_curves(curves)
    N_all = X_all.shape[0]
    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print(f"Arrays prepared: X={X_all.shape}, Y={Y_all.shape} (participants={N_all:,})")

    # ---- FINAL publication fit: full dataset ----
    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print(
            f"\n🚀 Fitting FINAL hierarchical model with all {N_all:,} participants "
            f"({CHAINS} chains × draws {DRAWS}, tune {TUNE}, target_accept {TARGET_ACCEPT})"
        )

    model, _ = build_model(X_all, Y_all)

    t0 = time.time()
    if USE_JAX:
        idata, loo, waic = fit_with_jax(model)
    else:
        idata, loo, waic = fit_with_default(model)
    elapsed = time.time() - t0

    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print(f"✅ Full run complete in {elapsed/3600:.2f} hours")

    # ---- Save model outputs ----
    az.to_netcdf(idata, os.path.join(OUT_DIR, "idata_theta_full.nc"))

    theta_mean = float(idata.posterior["theta"].mean())
    theta_sd   = float(idata.posterior["theta"].std())
    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print(f"✅ θ mean={theta_mean:.2f} (SD={theta_sd:.2f})")

    # ---- Posterior plot ----
    az.plot_posterior(idata, var_names=["theta"])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theta_posterior_full.png"), dpi=300)
    plt.close()

    # ---- Posterior predictive checks ----
    try:
        az.plot_ppc(idata, var_names=["y_obs"], num_pp_samples=50)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "ppc_full.png"), dpi=300)
        plt.close()
    except Exception as e:
        if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
            print("⚠️ PPC plot skipped:", e)

    # ---- Posterior summary (LaTeX + CSV) ----
    summ = az.summary(idata, var_names=["theta", "a_mu", "b_mu", "f_mu", "sigma"])
    summ_csv = os.path.join(OUT_DIR, "posterior_summary_full.csv")
    summ.to_csv(summ_csv)
    with open(os.path.join(TAB_DIR, "Table_posterior_summary_full.tex"), "w", encoding="utf-8") as f:
        f.write(summ.to_latex(float_format="%.3f"))

    # ---- HDI table (LaTeX) ----
    save_theta_hdi_table(idata, TAB_DIR)

    # ---- WAIC / LOO (if available) ----
    try:
        waic_val = waic.waic if waic is not None else np.nan
        loo_val  = loo.loo   if loo  is not None else np.nan
        if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
            print(f"📊 WAIC={waic_val:.2f}, LOOIC={loo_val:.2f}")
    except Exception:
        if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
            print("⚠️ WAIC/LOO not available.")

    if os.environ.get("PYMC_CHAIN_ID", "0") == "0":
        print("✅ Publication run finished successfully.")
