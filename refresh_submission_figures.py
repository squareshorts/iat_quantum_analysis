from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
TAB_DIR = BASE_DIR / "tables"
HDI_PROB = 0.94


def set_clean_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
    })


def style_axes(ax, grid_axis: str | None = "y"):
    if grid_axis:
        ax.grid(True, axis=grid_axis)
    ax.tick_params(length=3, width=0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def discrete_hdi(grid_vals: np.ndarray, probs: np.ndarray, mass: float = HDI_PROB):
    order = np.argsort(probs)[::-1]
    cum = 0.0
    chosen = []
    for idx in order:
        chosen.append(idx)
        cum += probs[idx]
        if cum >= mass:
            break
    chosen = np.asarray(chosen, dtype=int)
    return float(grid_vals[chosen].min()), float(grid_vals[chosen].max())


def profile_summary(csv_path: Path):
    df = pd.read_csv(csv_path)
    theta = df["theta_deg"].to_numpy(dtype=float)
    post = df["posterior"].to_numpy(dtype=float)
    post = post / post.sum()
    mean = float(np.sum(theta * post))
    lo, hi = discrete_hdi(theta, post, mass=HDI_PROB)
    theta_map = float(theta[np.argmax(post)])
    return df, mean, lo, hi, theta_map


def coeffs_at_theta(x: np.ndarray, y: np.ndarray, mask: np.ndarray, theta_deg: float):
    theta = np.deg2rad(theta_deg)
    arg = theta * x
    c = np.where(mask, np.cos(arg), 0.0)
    s = np.where(mask, np.sin(arg), 0.0)
    o = np.where(mask, 1.0, 0.0)
    ym = np.where(mask, y, 0.0)

    s_o = np.sum(o, axis=1)
    s_c = np.sum(c, axis=1)
    s_s = np.sum(s, axis=1)
    s_cc = np.sum(c * c, axis=1)
    s_ss = np.sum(s * s, axis=1)
    s_cs = np.sum(c * s, axis=1)
    s_y = np.sum(ym, axis=1)
    s_yc = np.sum(ym * c, axis=1)
    s_ys = np.sum(ym * s, axis=1)

    a = np.stack([
        np.stack([s_o, s_c, s_s], axis=-1),
        np.stack([s_c, s_cc, s_cs], axis=-1),
        np.stack([s_s, s_cs, s_ss], axis=-1),
    ], axis=-2)
    b = np.stack([s_y, s_yc, s_ys], axis=-1)
    a[..., range(3), range(3)] += 1e-8
    return np.linalg.solve(a, b[..., None])[..., 0]


def latex_table_to_df(path: Path):
    text = path.read_text(encoding="utf-8").splitlines()
    rows = []
    header = None
    for raw in text:
        line = raw.strip()
        if not line or line.startswith("\\"):
            continue
        if "&" not in line or not line.endswith("\\\\"):
            continue
        cells = [cell.strip() for cell in line[:-2].split("&")]
        if header is None:
            header = [clean_latex(cell) for cell in cells]
        else:
            rows.append([clean_latex(cell) for cell in cells])
    if header is None:
        raise ValueError(f"Could not parse LaTeX table: {path}")
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted.where(converted.notna(), df[col])
    return df


def clean_latex(text: str):
    cleaned = text.strip()
    cleaned = cleaned.replace(r"\%", "%")
    cleaned = cleaned.replace(r"\theta", "θ")
    cleaned = cleaned.replace(r"\sigma", "σ")
    cleaned = cleaned.replace(r"\text{MAP}", "MAP")
    cleaned = cleaned.replace(r"\mathrm{MAP}", "MAP")
    cleaned = cleaned.replace(r"\_", "_")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(r"\(", "").replace(r"\)", "")
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    return " ".join(cleaned.split())


def refresh_theta_posterior():
    df, mean, lo, hi, theta_map = profile_summary(OUT_DIR / "theta_grid_profile.csv")
    x = df["theta_deg"].to_numpy(dtype=float)
    y = df["posterior"].to_numpy(dtype=float)
    y = y / np.max(y)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(x, y, color="#1f4e79", linewidth=2.0)
    ax.axvspan(lo, hi, color="#9fb8cc", alpha=0.35)
    ax.axvline(theta_map, color="#5a5a5a", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Interference angle θ (degrees)")
    ax.set_ylabel("Normalized posterior density")
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "theta_posterior_full.png", bbox_inches="tight")
    plt.close(fig)


def refresh_theta_posterior_refined():
    path = OUT_DIR / "theta_grid_profile_refined.csv"
    if not path.exists():
        return
    df, _, lo, hi, theta_map = profile_summary(path)
    x = df["theta_deg"].to_numpy(dtype=float)
    y = df["posterior"].to_numpy(dtype=float)
    y = y / np.max(y)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(x, y, color="#1f4e79", linewidth=2.0)
    ax.axvspan(lo, hi, color="#9fb8cc", alpha=0.35)
    ax.axvline(theta_map, color="#5a5a5a", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Interference angle θ (degrees)")
    ax.set_ylabel("Normalized posterior density")
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "theta_posterior_refined.png", bbox_inches="tight")
    plt.close(fig)


def refresh_ppc():
    curves = pickle.load(open(OUT_DIR / "curves_cache.pkl", "rb"))
    max_t = max(len(c["x"]) for c in curves)
    n = len(curves)
    x = np.full((n, max_t), np.nan, dtype=float)
    y = np.full((n, max_t), np.nan, dtype=float)
    mask = np.zeros((n, max_t), dtype=bool)
    for i, curve in enumerate(curves):
        t = len(curve["x"])
        x[i, :t] = curve["x"]
        y[i, :t] = curve["y"]
        mask[i, :t] = True

    _, _, _, _, theta_map = profile_summary(OUT_DIR / "theta_grid_profile.csv")
    beta = coeffs_at_theta(x, y, mask, theta_map)
    theta = np.deg2rad(theta_map)
    mu = beta[:, 0][:, None] + beta[:, 1][:, None] * np.cos(theta * x) + beta[:, 2][:, None] * np.sin(theta * x)
    df = pd.DataFrame({"x": x[mask], "y": y[mask], "mu": mu[mask]})
    df["bin"] = pd.cut(df["x"], bins=np.linspace(0, 1, 13), include_lowest=True, labels=False)
    agg = df.groupby("bin").agg(x=("x", "mean"), y=("y", "mean"), mu=("mu", "mean")).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(agg["x"], agg["y"], "o-", color="#1f4e79", linewidth=1.8, markersize=4.5, label="Observed")
    ax.plot(agg["x"], agg["mu"], "s--", color="#b24a2f", linewidth=1.6, markersize=4.2, label="Predicted")
    ax.set_xlabel("Within-block position (normalized)")
    ax.set_ylabel("Z-scored latency")
    style_axes(ax, grid_axis="both")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ppc_full.png", bbox_inches="tight")
    plt.close(fig)


def refresh_model_comparison_legacy():
    path = TAB_DIR / "Table_model_comparison.tex"
    if not path.exists():
        return
    df = latex_table_to_df(path)
    labels = df["Model"].astype(str).tolist()
    labels = [label.replace("Interference ", "Interf. ").replace("Polynomial ", "Poly. ") for label in labels]
    values = pd.to_numeric(df["RMSE (holdout)"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(np.arange(len(values)), values, color="#5b8db8")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Hold-out RMSE")
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison_rmse.png", bbox_inches="tight")
    plt.close(fig)


def refresh_theta_bins_robustness():
    path = TAB_DIR / "Table_theta_bins_robustness.tex"
    if not path.exists():
        return
    df = latex_table_to_df(path)
    x = pd.to_numeric(df["Bins"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["θ mean"], errors="coerce").to_numpy(dtype=float)
    yerr = pd.to_numeric(df["θ sd"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", color="#1f4e79", linewidth=1.8, capsize=4, markersize=5)
    ax.set_xlabel("Bins per participant")
    ax.set_ylabel("θ (mean ± SD)")
    ax.set_xticks(x)
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "theta_bins_robustness.png", bbox_inches="tight")
    plt.close(fig)


def refresh_submission_model_comparison():
    path = OUT_DIR / "model_comparison_submission.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "bins" in df.columns:
        df = df[df["bins"] == 6].copy()
    labels = df["model"].astype(str).tolist()
    values = df["rmse_test"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.bar(np.arange(len(values)), values, color="#5b8db8")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Temporal hold-out RMSE")
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison_submission_bins6.png", bbox_inches="tight")
    plt.close(fig)


def refresh_block_theta_summary():
    path = OUT_DIR / "block_theta_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    x = np.arange(len(df))
    y = df["theta_mean"].to_numpy(dtype=float)
    yerr = np.vstack([
        y - df["theta_hdi_low"].to_numpy(dtype=float),
        df["theta_hdi_high"].to_numpy(dtype=float) - y,
    ])

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", color="#1f4e79", linewidth=1.8, capsize=4, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["block"].astype(str).tolist())
    ax.set_xlabel("Block")
    ax.set_ylabel("θ (degrees)")
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "block_theta_summary.png", bbox_inches="tight")
    plt.close(fig)


def refresh_negative_control():
    null_path = OUT_DIR / "theta_permutation_null.csv"
    summary_path = OUT_DIR / "negative_controls_summary.csv"
    if not null_path.exists() or not summary_path.exists():
        return
    null_df = pd.read_csv(null_path)
    summary_df = pd.read_csv(summary_path)
    observed = float(summary_df.loc[summary_df["control"] == "Within-curve x permutation", "observed_metric"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(null_df["theta_sd"].to_numpy(dtype=float), bins=10, color="#b8c7d6", edgecolor="#4d4d4d")
    ax.axvline(observed, color="#b24a2f", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Posterior SD under x-permutation null")
    ax.set_ylabel("Count")
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "theta_permutation_null.png", bbox_inches="tight")
    plt.close(fig)


def refresh_recovery():
    path = OUT_DIR / "theta_recovery_simulation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    agg = (
        df.groupby("theta_true")
        .agg(theta_map_mean=("theta_map", "mean"), theta_map_sd=("theta_map", "std"))
        .reset_index()
    )
    agg["theta_map_sd"] = agg["theta_map_sd"].fillna(0.0)
    lim_max = max(float(agg["theta_true"].max()), float(agg["theta_map_mean"].max())) + 5.0

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.errorbar(
        agg["theta_true"].to_numpy(dtype=float),
        agg["theta_map_mean"].to_numpy(dtype=float),
        yerr=agg["theta_map_sd"].to_numpy(dtype=float),
        fmt="o-",
        color="#1f4e79",
        linewidth=1.8,
        capsize=4,
        markersize=5,
    )
    ax.plot([0, lim_max], [0, lim_max], linestyle="--", color="#808080", linewidth=1.0)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("True θ (degrees)")
    ax.set_ylabel("Recovered θ MAP")
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "theta_recovery_simulation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    set_clean_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    refresh_theta_posterior()
    refresh_theta_posterior_refined()
    refresh_ppc()
    refresh_model_comparison_legacy()
    refresh_theta_bins_robustness()
    refresh_submission_model_comparison()
    refresh_block_theta_summary()
    refresh_negative_control()
    refresh_recovery()

    print("Refreshed clean submission figures:")
    for name in [
        "theta_posterior_full.png",
        "theta_posterior_refined.png",
        "ppc_full.png",
        "model_comparison_rmse.png",
        "theta_bins_robustness.png",
        "model_comparison_submission_bins6.png",
        "block_theta_summary.png",
        "theta_permutation_null.png",
        "theta_recovery_simulation.png",
    ]:
        path = FIG_DIR / name
        if path.exists():
            print(f" - {path}")


if __name__ == "__main__":
    main()
