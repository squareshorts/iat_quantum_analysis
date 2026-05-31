import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
CURVES_PATH = "outputs/curves_cache_bins6.pkl"
THETA_GRID = np.linspace(10, 25, 301)   # local grid around pooled θ
MAX_ITERS = 10
RIDGE = 1e-5
SIGMA_FLOOR = 0.1                       # minimum σθ for stability
TOL = 1e-3                              # convergence tolerance (deg)

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


def rss_for_theta(x, y, theta):
    X = design_matrix(x, theta)
    beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(3), X.T @ y)
    return np.sum((y - X @ beta) ** 2)


# ----------------------------
# Load curves
# ----------------------------
with open(CURVES_PATH, "rb") as f:
    curves_all = pickle.load(f)

# Subsample participants (OPTION A)
rng = np.random.default_rng(123)
curves = rng.choice(curves_all, size=10000, replace=False).tolist()

n = len(curves)
print(f"Using subsample of {n} participants")

# ----------------------------
# Initialization
# ----------------------------
theta_i = 17.25 + rng.normal(scale=0.3, size=n)   # jittered init
mu_theta = theta_i.mean()
sigma_theta = max(theta_i.std(), 0.2)             # stronger floor

prev_mu = mu_theta
print(f"Init: mu={mu_theta:.3f}, sigma={sigma_theta:.3f}")

# ----------------------------
# Iterative EB updates
# ----------------------------
for it in range(MAX_ITERS):
    sigma_theta = max(sigma_theta, SIGMA_FLOOR)
    lambda_pen = 1.0 / (sigma_theta ** 2)

    # --- Update individual θ_i ---
    for i, c in enumerate(curves):
        x, y = c["x"], c["y"]

        obj = [
            rss_for_theta(x, y, th) + lambda_pen * (th - mu_theta) ** 2
            for th in THETA_GRID
        ]
        theta_i[i] = THETA_GRID[np.argmin(obj)]

    # --- Update hyperparameters ---
    mu_theta = theta_i.mean()
    sigma_theta = theta_i.std()

    print(f"Iter {it+1}: mu={mu_theta:.3f}, sigma={sigma_theta:.3f}")

    # --- Convergence check ---
    if abs(mu_theta - prev_mu) < TOL:
        print("Converged.")
        break

    prev_mu = mu_theta

# ----------------------------
# Save outputs
# ----------------------------
Path("outputs").mkdir(exist_ok=True)

pd.DataFrame({
    "pid": [c["pid"] for c in curves],
    "theta_i": theta_i,
}).to_csv("outputs/theta_individual.csv", index=False)

pd.Series({
    "mu_theta": mu_theta,
    "sigma_theta": sigma_theta,
}).to_json("outputs/theta_hyperparams.json", indent=2)

print("Hierarchical θ estimation complete")
