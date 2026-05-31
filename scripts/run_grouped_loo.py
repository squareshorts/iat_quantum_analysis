import numpy as np
import arviz as az
import pandas as pd
from pathlib import Path

# ----------------------------
# Load log-likelihoods
# ----------------------------
ll = {
    "interference": np.load("outputs/loglik/loglik_interference.npy"),
    "poly": np.load("outputs/loglik/loglik_poly.npy"),
    "exp": np.load("outputs/loglik/loglik_exp.npy"),
    "power": np.load("outputs/loglik/loglik_power.npy"),
}

# Each array is shape (n_participants,)
n = len(next(iter(ll.values())))

# ----------------------------
# Build InferenceData
# Treat participants as draws
# ----------------------------
idata = {}
for k, v in ll.items():
    idata[k] = az.from_dict(
        posterior={"_dummy": np.zeros((1, n))},  # chain=1, draw=n
        log_likelihood={k: v.reshape(1, n, 1)},  # (chain, draw, obs=1)
    )

# ----------------------------
# Compute LOO
# ----------------------------
loos = {k: az.loo(v, scale="log") for k, v in idata.items()}

# ----------------------------
# ΔELPD table
# ----------------------------
ref = "interference"
rows = []

for k, loo in loos.items():
    rows.append({
        "model": k,
        "elpd_loo": loo.elpd_loo,
        "elpd_se": loo.se,
        "delta_elpd": loo.elpd_loo - loos[ref].elpd_loo,
        "delta_se": np.sqrt(loo.se**2 + loos[ref].se**2),
        "pareto_k_gt_07": np.mean(loo.pareto_k > 0.7),
    })

df = pd.DataFrame(rows).sort_values("delta_elpd", ascending=False)

# ----------------------------
# Save outputs
# ----------------------------
Path("tables").mkdir(exist_ok=True)
df.to_csv("outputs/loo_comparison.csv", index=False)

with open("tables/Table_loo_comparison.tex", "w") as f:
    f.write(df.to_latex(index=False, float_format="%.2f"))

print(df)
