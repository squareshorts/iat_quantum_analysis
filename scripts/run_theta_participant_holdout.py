import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
CURVES_PATH = "outputs/curves_cache_bins6.pkl"
SPLITS_PATH = "outputs/splits_participant_kfold.json"
THETA_DEG = 17.25  # pooled MAP estimate
K_FEWSHOT = 1

OUT_CSV = "outputs/participant_holdout_metrics.csv"
OUT_TEX = "tables/Table_participant_holdout.tex"

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


def predict(x, beta, theta_deg):
    X = design_matrix(x, theta_deg)
    return X @ beta


# ----------------------------
# Load data
# ----------------------------
with open(CURVES_PATH, "rb") as f:
    curves = pickle.load(f)

with open(SPLITS_PATH, "r", encoding="utf-8") as f:
    splits = json.load(f)["splits"]

print(f"Loaded {len(curves):,} curves and {len(splits)} folds")

# Index curves by participant ID
curve_by_pid = {c["pid"]: c for c in curves}

rows = []

# ----------------------------
# Main evaluation loop
# ----------------------------
for split in splits:
    fold = split["fold"]
    test_pids = set(split["test_participants"])
    train_pids = [pid for pid in curve_by_pid if pid not in test_pids]

    # --- Training data for global standardization ---
    y_train_all = []
    for pid in train_pids:
        y_train_all.extend(curve_by_pid[pid]["y"])

    mu_global = np.mean(y_train_all)
    sd_global = np.std(y_train_all)

    # --- Fit OLS on training participants ---
    X_all, y_all = [], []
    for pid in train_pids:
        c = curve_by_pid[pid]
        X_all.append(design_matrix(c["x"], THETA_DEG))
        y_all.append((c["y"] - mu_global) / sd_global)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    beta = fit_ols(X_all, y_all)

    # --- Evaluate on test participants ---
    errs_pure = []
    errs_fewshot = []

    for pid in test_pids:
        c = curve_by_pid[pid]
        x = c["x"]
        y = c["y"]

        # ---- PHO-pure ----
        y_std = (y - mu_global) / sd_global
        y_hat = predict(x, beta, THETA_DEG)
        errs_pure.extend(y_hat - y_std)

        # ---- PHO-fewshot ----
        x_cal = x[:K_FEWSHOT]
        y_cal = y[:K_FEWSHOT]

        mu_i = np.mean(y_cal)
        sd_i = np.std(y_cal) if np.std(y_cal) > 0 else sd_global

        x_test = x[K_FEWSHOT:]
        y_test = (y[K_FEWSHOT:] - mu_i) / sd_i
        y_hat_fs = predict(x_test, beta, THETA_DEG)

        errs_fewshot.extend(y_hat_fs - y_test)

    # --- Metrics ---
    def summarize(errs):
        errs = np.asarray(errs)
        return {
            "RMSE": np.sqrt(np.mean(errs ** 2)),
            "MAE": np.mean(np.abs(errs)),
            "N": len(errs),
        }

    rows.append({
        "fold": fold,
        "regime": "PHO-pure",
        **summarize(errs_pure),
    })

    rows.append({
        "fold": fold,
        "regime": f"PHO-{K_FEWSHOT}",
        **summarize(errs_fewshot),
    })

# ----------------------------
# Save results
# ----------------------------
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

# Aggregate over folds
agg = (
    df.groupby("regime")[["RMSE", "MAE"]]
    .mean()
    .reset_index()
)

Path("tables").mkdir(exist_ok=True)
with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(agg.to_latex(index=False, float_format="%.4f"))

print("Participant-held-out evaluation complete.")
print(agg)
