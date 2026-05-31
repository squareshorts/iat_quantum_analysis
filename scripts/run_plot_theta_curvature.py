import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("outputs/theta_grid_profile_refined.csv")
curv = pd.read_json("outputs/theta_curvature.json", typ="series")

theta = df["theta_deg"].values
posterior = df["posterior"].values
posterior = posterior / posterior.sum()
logp = np.log(posterior)

theta_map = curv["theta_map_deg"]
se_theta = curv["se_theta_deg"]

# ----------------------------
# Quadratic approximation
# ----------------------------
WINDOW = 5
idx_map = np.argmin(np.abs(theta - theta_map))

i0 = max(idx_map - WINDOW, 0)
i1 = min(idx_map + WINDOW + 1, len(theta))

theta_local = theta[i0:i1] - theta_map
logp_local = logp[i0:i1]

coef = np.polyfit(theta_local, logp_local, deg=2)

theta_dense = np.linspace(theta_local.min(), theta_local.max(), 200)
logp_quad = np.polyval(coef, theta_dense)

# ----------------------------
# Plot
# ----------------------------
Path("figures").mkdir(exist_ok=True)

plt.figure(figsize=(7, 4.5))
plt.plot(theta, logp, color="black", lw=1.5, label="Profile log-posterior")
plt.plot(theta_dense + theta_map, logp_quad, "--", color="tab:blue", lw=2,
         label="Local quadratic fit")

plt.axvline(theta_map, color="tab:red", lw=2, label=r"$\theta^*$")
plt.axvline(theta_map - se_theta, color="tab:red", ls=":", lw=1.5)
plt.axvline(theta_map + se_theta, color="tab:red", ls=":", lw=1.5,
            label=r"$\theta^* \pm \mathrm{SE}$")

plt.xlabel(r"Interference angle $\theta$ (degrees)")
plt.ylabel(r"$\log p(\theta \mid \mathrm{data})$")
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("figures/theta_profile_curvature.png", dpi=300)
plt.close()

print("Saved figures/theta_profile_curvature.png")
