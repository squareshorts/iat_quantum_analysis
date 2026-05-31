import pickle
import numpy as np
import pandas as pd
from pathlib import Path

CURVES_PATH = "outputs/curves_cache_bins6.pkl"
THETA_GRID = np.linspace(10, 25, 301)
RIDGE = 1e-5

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

with open(CURVES_PATH, "rb") as f:
    curves_all = pickle.load(f)

# Subsample for speed
rng = np.random.default_rng(123)
curves = rng.choice(curves_all, size=10000, replace=False).tolist()

theta_i = []
for c in curves:
    x, y = c["x"], c["y"]
    rss = [rss_for_theta(x, y, th) for th in THETA_GRID]
    theta_i.append(THETA_GRID[np.argmin(rss)])

theta_i = np.array(theta_i)

print(f"Individual θ: mean={theta_i.mean():.3f}, sd={theta_i.std():.4f}")

Path("outputs").mkdir(exist_ok=True)
pd.DataFrame({
    "pid": [c["pid"] for c in curves],
    "theta_i": theta_i
}).to_csv("outputs/theta_individual_nopenalty.csv", index=False)
