import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Load profile
# ----------------------------
df = pd.read_csv("outputs/theta_grid_profile_refined.csv")

theta = df["theta_deg"].values
posterior = df["posterior"].values

# Normalize for numerical stability
posterior = posterior / posterior.sum()
logp = np.log(posterior)

# ----------------------------
# Locate MAP
# ----------------------------
idx_map = np.argmax(logp)
theta_map = theta[idx_map]

# ----------------------------
# Local quadratic fit
# ----------------------------
# Use a small symmetric window around MAP
WINDOW = 5  # grid points on each side

i0 = max(idx_map - WINDOW, 0)
i1 = min(idx_map + WINDOW + 1, len(theta))

theta_local = theta[i0:i1] - theta_map
logp_local = logp[i0:i1]

# Fit: log p ≈ a + b θ + c θ^2
coef = np.polyfit(theta_local, logp_local, deg=2)
a, b, c = coef

# For a Gaussian: c = -1/(2*SE^2)
se_theta = np.sqrt(-1 / (2 * c))

# ----------------------------
# Save results
# ----------------------------
out = {
    "theta_map_deg": float(theta_map),
    "se_theta_deg": float(se_theta),
    "curvature_c": float(c),
}

Path("outputs").mkdir(exist_ok=True)
pd.Series(out).to_json("outputs/theta_curvature.json", indent=2)

print("θ curvature analysis complete")
print(out)
