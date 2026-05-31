import pickle
import numpy as np
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
CURVES_PATH = "outputs/curves_cache_bins6.pkl"
THETA_GRID_PATH = "outputs/theta_grid_profile_refined.csv"
OUT_DIR = Path("outputs/loglik")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def design_matrix(x, theta_deg):
    w = np.pi * theta_deg / 180.0
    return np.column_stack([
        np.ones_like(x),
        np.cos(w * x),
        np.sin(w * x),
    ])


def fit_ols(X, y, ridge=1e-5):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def gaussian_loglik(resid, sigma):
    return -0.5 * (
        np.log(2 * np.pi * sigma**2) + (resid**2) / sigma**2
    )


# ----------------------------
# Load data
# ----------------------------
with open(CURVES_PATH, "rb") as f:
    curves = pickle.load(f)

import pandas as pd
theta_df = pd.read_csv(THETA_GRID_PATH)

thetas = theta_df["theta_deg"].values

# Use already-computed posterior weights
weights = theta_df["posterior"].values
weights = weights / weights.sum()

# Estimate sigma from minimum RSS
n_obs = len(curves) * len(curves[0]["y"])
rss_min = theta_df["rss"].min()
sigma_hat = np.sqrt(rss_min / n_obs)

print(f"Using σ̂ = {sigma_hat:.4f}")

# ----------------------------
# Compute per-participant log-likelihoods
# ----------------------------
loglik_participants = []

for c in curves:
    x = c["x"]
    y = c["y"]

    ll_theta = []

    for theta in thetas:
        X = design_matrix(x, theta)
        beta = fit_ols(X, y)
        resid = y - X @ beta
        ll = gaussian_loglik(resid, sigma_hat).sum()
        ll_theta.append(ll)

    ll_theta = np.array(ll_theta)
    ll_marginal = np.log(np.sum(weights * np.exp(ll_theta - ll_theta.max()))) + ll_theta.max()
    loglik_participants.append(ll_marginal)

loglik_participants = np.array(loglik_participants)

np.save(OUT_DIR / "loglik_interference.npy", loglik_participants)

print("Saved participant-level log-likelihoods for interference model.")
