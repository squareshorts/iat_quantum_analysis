import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("outputs/theta_individual_nopenalty.csv")
theta_i = df["theta_i"].values

theta_pooled = 17.25  # pooled MAP estimate (from grid profiling)

# ----------------------------
# Output directory
# ----------------------------
Path("figures").mkdir(exist_ok=True)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(6.5, 4.5))

plt.hist(
    theta_i,
    bins=40,
    density=True,
    alpha=0.75,
    color="gray",
    edgecolor="black",
    linewidth=0.8,
)

# pooled theta
plt.axvline(
    theta_pooled,
    color="red",
    lw=2.5,
    label=r"Pooled $\theta^*$",
)

# annotation (no subtitle)
ymax = plt.ylim()[1]
plt.text(
    theta_pooled + 0.25,
    0.85 * ymax,
    r"$\theta^*$",
    color="red",
    fontsize=11,
    verticalalignment="center",
)

# axes
plt.xlabel(r"Individual interference angle $\theta_i$ (degrees)")
plt.ylabel("Density")

plt.xlim(10, 26)

plt.legend(frameon=False)
plt.tight_layout()

# ----------------------------
# Save
# ----------------------------
outpath = "figures/theta_individual_vs_pooled.png"
plt.savefig(outpath, dpi=300)
plt.close()

print(f"Saved {outpath}")
