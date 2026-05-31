# ============================================================
# run_hierarchical_analysis.py
# Hierarchical PyMC analysis for IAT Quantum Project
# ============================================================

# ============================================================
# Imports and global setup
# ============================================================
import os
os.environ["PYTENSOR_FLAGS"] = "mode=FAST_COMPILE,device=cpu,openmp=False,cxx="

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import glob
import pickle

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def build_participant_curves(df, n_bins=6):
    """Build standardized x/y curves for each participant."""
    need = {"pid", "block", "trial_in_block", "rt"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["block", "trial_in_block", "rt"])
    df = df[df["block"].isin([3, 4, 6, 7])]

    out = []
    for pid, g in df.groupby("pid", sort=False):
        g["pos_norm"] = g.groupby("block")["trial_in_block"].transform(
            lambda s: (s - s.min()) / max(1.0, s.max() - s.min())
        )
        vals = g[["pos_norm", "rt"]].to_numpy(dtype=float)
        if len(vals) < n_bins:
            continue
        q = np.quantile(vals[:, 0], np.linspace(0, 1, n_bins + 1))
        x_bin, y_bin = [], []
        for i in range(n_bins):
            mask = (vals[:, 0] >= q[i]) & (vals[:, 0] <= q[i + 1])
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


def mu_quantum_x(x, a, b, theta_deg, f):
    return a + b * np.cos(np.deg2rad(theta_deg) * x + f)


def fit_hier_quantum(X, Y, T, draws=200, tune=200, target_accept=0.95, random_seed=123):
    """Hierarchical cosine model (MCMC)."""
    N, max_T = X.shape
    mask = ~np.isnan(Y)
    with pm.Model() as model:
        a_mu = pm.Normal("a_mu", 0.0, 2.0)
        a_sd = pm.HalfNormal("a_sd", 1.0)
        b_mu = pm.Normal("b_mu", 0.0, 2.0)
        b_sd = pm.HalfNormal("b_sd", 1.0)
        f_mu = pm.Normal("f_mu", 0.0, 1.0)
        f_sd = pm.HalfNormal("f_sd", 0.5)
        theta = pm.TruncatedNormal("theta", mu=85.0, sigma=15.0, lower=30.0, upper=120.0)

        a_i = pm.Normal("a_i", a_mu, a_sd, shape=N)
        b_i = pm.Normal("b_i", b_mu, b_sd, shape=N)
        f_i = pm.Normal("f_i", f_mu, f_sd, shape=N)
        sigma = pm.HalfNormal("sigma", 0.5)

        mu = mu_quantum_x(X, a_i[:, None], b_i[:, None], theta, f_i[:, None])
        pm.Normal("y_obs", mu=mu[mask], sigma=sigma, observed=Y[mask])

        idata = pm.sample(
            draws=150,
            tune=150,
            target_accept=0.95,
            chains=1,
            cores=1,
        )

        idata = pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        loo, waic = None, None
        try:
            loo = az.loo(idata)
            waic = az.waic(idata)
        except Exception as e:
            print(f"⚠️  LOO/WAIC not computed: {e}")

    return idata, loo, waic


def fit_variational_quantum(X, Y, T, n_iter=10000):
    """Fast ADVI approximation."""
    N, max_T = X.shape
    mask = ~np.isnan(Y)
    with pm.Model() as model:
        a_mu = pm.Normal("a_mu", 0.0, 2.0)
        a_sd = pm.HalfNormal("a_sd", 1.0)
        b_mu = pm.Normal("b_mu", 0.0, 2.0)
        b_sd = pm.HalfNormal("b_sd", 1.0)
        f_mu = pm.Normal("f_mu", 0.0, 1.0)
        f_sd = pm.HalfNormal("f_sd", 0.5)
        theta = pm.TruncatedNormal("theta", mu=85.0, sigma=15.0, lower=30.0, upper=120.0)

        a_i = pm.Normal("a_i", a_mu, a_sd, shape=N)
        b_i = pm.Normal("b_i", b_mu, b_sd, shape=N)
        f_i = pm.Normal("f_i", f_mu, f_sd, shape=N)
        sigma = pm.HalfNormal("sigma", 0.5)

        mu = a_i[:, None] + b_i[:, None] * pm.math.cos(np.deg2rad(theta) * X + f_i[:, None])
        pm.Normal("y_obs", mu=mu[mask], sigma=sigma, observed=Y[mask])

        advi_fit = pm.fit(n=n_iter, method="advi")
        idata = advi_fit.sample(draws=500)
    return idata


# ============================================================
# STEP C – Cross-domain comparison (Gender, Race, Age)
# ============================================================
if __name__ == "__main__":

    print("\n🌍 Running multi-domain comparison")

    domains = {
        "GenderScience": os.path.join(DATA_DIR, "GenderScience_iat_2019", "iat_2019"),
        "Race":          os.path.join(DATA_DIR, "race"),
        "Age":           os.path.join(DATA_DIR, "age_netherlands"),
    }

    all_results = []

    for name, folder in domains.items():
        print(f"\n🧠 Processing domain: {name}")

        # ---- Load & combine ----
        if name == "GenderScience":
            paths = glob.glob(os.path.join(folder, "iat*.txt"))
            dfs = [pd.read_csv(p, sep="\t", low_memory=False) for p in paths]
        else:
            paths = glob.glob(os.path.join(folder, "*.csv"))
            dfs = [pd.read_csv(p, sep=",", low_memory=False) for p in paths]
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(df)} rows")

        # ---- Clean & rename ----
        df.columns = df.columns.str.strip()
        rename_map = {}

        # Handle each schema separately
        if name == "GenderScience":
            rename_map = {
                "session_id": "pid",
                "block_number": "block",
                "trial_number": "trial_in_block",
                "trial_latency": "rt",
            }
        elif name == "Race":
            # Example mapping for Race CSVs (adjust if columns differ)
            for col in df.columns:
                cl = col.lower()
                if "session" in cl and "id" in cl:
                    rename_map[col] = "pid"
                elif "block" in cl:
                    rename_map[col] = "block"
                elif "trial" in cl:
                    rename_map[col] = "trial_in_block"
                elif "latency" in cl or "rt" in cl:
                    rename_map[col] = "rt"
        elif name == "Age":
            # Adjust if column names differ
            for col in df.columns:
                cl = col.lower()
                if "session" in cl and "id" in cl:
                    rename_map[col] = "pid"
                elif "block" in cl:
                    rename_map[col] = "block"
                elif "trial" in cl:
                    rename_map[col] = "trial_in_block"
                elif "latency" in cl or "rt" in cl:
                    rename_map[col] = "rt"

        df = df.rename(columns=rename_map).dropna(subset=["pid"])
        keep = [c for c in ["pid", "block", "trial_in_block", "rt", "trial_error"] if c in df.columns]
        df = df[keep]

        print(f"  Columns standardized: {df.columns.tolist()[:5]}")

        # ---- Build participant curves ----
        curves = build_participant_curves(df, n_bins=6)
        if len(curves) == 0:
            print(f"⚠️  No curves generated for {name}, skipping.")
            continue

        # ---- Prepare arrays ----
        max_T = max(len(c["x"]) for c in curves)
        N = len(curves)
        X = np.full((N, max_T), np.nan)
        Y = np.full((N, max_T), np.nan)
        T = np.zeros(N, dtype=int)
        for i, c in enumerate(curves):
            T[i] = len(c["x"])
            X[i, :T[i]] = c["x"]
            Y[i, :T[i]] = c["y"]

        # ---- Run short hierarchical fit (subset for speed) ----
        subset = min(300, N)
        idata_q, _, _ = fit_hier_quantum(X[:subset], Y[:subset], T[:subset])
        theta_mean = float(idata_q.posterior["theta"].mean())
        all_results.append({"domain": name, "theta_mean": theta_mean})
        print(f"✅ {name} θ mean: {theta_mean:.2f}")

        # Save per-domain posterior and plot
        az.plot_posterior(idata_q, var_names=["theta"])
        plt.title(f"Posterior θ – {name}")
        plt.savefig(os.path.join(FIG_DIR, f"theta_posterior_{name}.png"), dpi=300)
        plt.close()

        az.summary(idata_q, var_names=["theta"]).to_csv(
            os.path.join(OUT_DIR, f"posterior_summary_{name}.csv")
        )

    # ---- Save cross-domain summary ----
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT_DIR, "theta_domains_summary.csv"), index=False)
    print("\n✅ Saved domain comparison CSV:", os.path.join(OUT_DIR, "theta_domains_summary.csv"))

    # ---- Comparison plot ----
    plt.figure(figsize=(6, 4))
    plt.bar(results_df["domain"], results_df["theta_mean"], color=["#9ecae1", "#fdae6b", "#a1d99b"])
    plt.ylabel("θ mean (degrees)")
    plt.title("Contextual overlap across IAT domains")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theta_domains_comparison.png"), dpi=300)
    plt.close()
    print("✅ Saved comparison plot (figures/theta_domains_comparison.png)")
    os.system("shutdown /s /t 60")  # shutdown in 60 seconds

