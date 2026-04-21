import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Load pooled curve
# -----------------------
df = pd.read_csv("223251_curve.csv")

df_pooled = (
    df
    .groupby("bin", as_index=False)
    .agg(
        x=("x_mean", "mean"),
        y=("rt_mean", "mean")
    )
)

x = df_pooled["x"].values
y = df_pooled["y"].values
n = len(y)

# -----------------------
# Design matrix builder
# -----------------------
def design_matrix(x, theta_deg):
    omega = np.pi * theta_deg / 180.0
    return np.column_stack([
        np.ones_like(x),
        np.cos(omega * x),
        np.sin(omega * x)
    ])

# -----------------------
# RSS for fixed theta
# -----------------------
def rss_theta(theta_deg):
    X = design_matrix(x, theta_deg)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return np.sum(resid**2)

# -----------------------
# Grid profile
# -----------------------
theta_grid = np.linspace(0.5, 90.0, 718)
rss = np.array([rss_theta(t) for t in theta_grid])

# Normalize to minimum
rss_min = rss.min()
rss_rel = rss - rss_min

theta_hat = theta_grid[np.argmin(rss)]

print(f"θ_hat (deg) = {theta_hat:.2f}")

# -----------------------
# Plot RSS profile
# -----------------------
plt.figure()
plt.plot(theta_grid, rss_rel)
plt.axvline(theta_hat, linestyle="--")
plt.xlabel("θ (degrees)")
plt.ylabel("RSS − RSS_min")
plt.title("Profile RSS for θ (Brazilian IAT, pooled)")
plt.show()

theta_ref = 17.0  # from Gender–Science IAT

theta_ref = 17.0
rss_eps = rss_theta(0.5)
rss_ref = rss_theta(theta_ref)

print(f"RSS(theta=0.5°) = {rss_eps:.2f}")
print(f"RSS(theta=17°)  = {rss_ref:.2f}")
print(f"ΔRSS (17° - 0.5°) = {rss_ref - rss_eps:.2f}")


def profile_for_block(block, grid=np.linspace(0.5, 90.0, 718)):
    d = df[df["block_id"] == block].copy().sort_values("bin")
    xb = d["x_mean"].values
    yb = d["rt_mean"].values

    def rss_block(theta_deg):
        omega = np.pi * theta_deg / 180.0
        X = np.column_stack([np.ones_like(xb), np.cos(omega * xb), np.sin(omega * xb)])
        beta, *_ = np.linalg.lstsq(X, yb, rcond=None)
        resid = yb - X @ beta
        return np.sum(resid**2)

    rss = np.array([rss_block(t) for t in grid])
    return grid[np.argmin(rss)], rss.min()

for block in ["fase3", "fase5"]:
    t_hat_b, rssmin_b = profile_for_block(block)
    print(f"{block}: θ_hat={t_hat_b:.2f}°, RSS_min={rssmin_b:.2f}")
