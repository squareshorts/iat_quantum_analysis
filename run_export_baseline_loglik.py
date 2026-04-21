import pickle
import numpy as np
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
CURVES_PATH = "outputs/curves_cache_bins6.pkl"
OUT_DIR = Path("outputs/loglik")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters fixed from baseline sweep
# (see tables/Table_model_comparison.tex)
K_EXP = 0.0
P_POWER = 0.25

# ----------------------------
# Helpers
# ----------------------------
def fit_ols(X, y, ridge=1e-5):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def gaussian_loglik(resid, sigma):
    return -0.5 * (
        np.log(2 * np.pi * sigma**2) + (resid**2) / sigma**2
    )


def poly_design(x):
    return np.column_stack([np.ones_like(x), x, x**2])


def exp_design(x, k):
    return np.column_stack([np.ones_like(x), np.exp(-k * x)])


def power_design(x, p):
    return np.column_stack([np.ones_like(x), x**p])


# ----------------------------
# Load curves
# ----------------------------
with open(CURVES_PATH, "rb") as f:
    curves = pickle.load(f)

# ----------------------------
# Estimate sigma_hat (shared)
# ----------------------------
rss_all = []

for c in curves:
    x = c["x"]
    y = c["y"]
    X = poly_design(x)
    beta = fit_ols(X, y)
    rss_all.append(np.sum((y - X @ beta) ** 2))

rss_all = np.array(rss_all)
n_obs = len(curves) * len(curves[0]["y"])
sigma_hat = np.sqrt(rss_all.min() / n_obs)

print(f"Using σ̂ = {sigma_hat:.4f}")

# ----------------------------
# Compute log-likelihoods
# ----------------------------
loglik_poly = []
loglik_exp = []
loglik_power = []

for c in curves:
    x = c["x"]
    y = c["y"]

    # Polynomial
    Xp = poly_design(x)
    bp = fit_ols(Xp, y)
    rp = y - Xp @ bp
    loglik_poly.append(gaussian_loglik(rp, sigma_hat).sum())

    # Exponential
    Xe = exp_design(x, K_EXP)
    be = fit_ols(Xe, y)
    re = y - Xe @ be
    loglik_exp.append(gaussian_loglik(re, sigma_hat).sum())

    # Power law
    Xw = power_design(x, P_POWER)
    bw = fit_ols(Xw, y)
    rw = y - Xw @ bw
    loglik_power.append(gaussian_loglik(rw, sigma_hat).sum())

# ----------------------------
# Save
# ----------------------------
np.save(OUT_DIR / "loglik_poly.npy", np.array(loglik_poly))
np.save(OUT_DIR / "loglik_exp.npy", np.array(loglik_exp))
np.save(OUT_DIR / "loglik_power.npy", np.array(loglik_power))

print("Saved baseline participant-level log-likelihoods.")
