from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_submission_evidence import (
    HOLDOUT_RIDGE,
    THETA_COARSE_GRID,
    THETA_REFINE_RADIUS,
    THETA_REFINE_STEP,
    arrays_from_curves,
    compute_rss_and_beta,
    design_matrix_from_param,
    fit_parameter_free_model,
    fit_poly_coefficients,
    fixed_theta_coefficients,
    leak_free_standardize,
    row_standardize,
    search_best_param,
    stats_from_curves,
    temporal_holdout_mask,
)


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
TAB_DIR = BASE_DIR / "tables"

BLOCKWISE_CRITICAL_BLOCKS = [3, 4, 6, 7]
QUANTILES = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=float)


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, TAB_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_cached_curves(name: str) -> list[dict[str, object]]:
    with open(OUT_DIR / name, "rb") as handle:
        return pickle.load(handle)


def theta_map_from_profile() -> float:
    profile = pd.read_csv(OUT_DIR / "theta_grid_profile_submission.csv")
    return float(profile.loc[profile["posterior"].idxmax(), "theta_deg"])


def block_theta_lookup() -> dict[int, float]:
    df = pd.read_csv(OUT_DIR / "block_theta_summary.csv")
    return {int(row.block): float(row.theta_map) for row in df.itertuples()}


def heldout_predictions(
    x: np.ndarray,
    y_raw: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
    sumsqs: np.ndarray,
    model_name: str,
    best_param: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = len(design_matrix_from_param(x, model_name, None if model_name == "poly2" else best_param))
    train_mask, test_mask, valid = temporal_holdout_mask(x, mask, 0.20, min_train_pts=p)
    xv = x[valid]
    yv_raw = y_raw[valid]
    train_v = train_mask[valid]
    test_v = test_mask[valid]
    yz = leak_free_standardize(
        yv_raw,
        train_v,
        counts=counts[valid],
        sums=sums[valid],
        sumsqs=sumsqs[valid],
    )
    designs = design_matrix_from_param(xv, model_name, best_param)
    _, beta, _ = compute_rss_and_beta(designs, yz, train_v, ridge=HOLDOUT_RIDGE)
    mu = np.zeros_like(yz)
    for j, design in enumerate(designs):
        mu += beta[:, j][:, None] * design
    return xv, yz, mu, train_v, test_v


def empirical_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size:
        raise ValueError("Empirical Wasserstein requires equal-length samples.")
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def quantile_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.quantile(a, QUANTILES) - np.quantile(b, QUANTILES))))


def iqr_error(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.quantile(a, [0.25, 0.75])
    qb = np.quantile(b, [0.25, 0.75])
    return float(abs((qa[1] - qa[0]) - (qb[1] - qb[0])))


def residual_and_distribution_analysis(
    block_curves: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    x, y_raw, mask, _, blocks = arrays_from_curves(block_curves, include_block=True)
    counts, sums, sumsqs = stats_from_curves(block_curves)
    theta_by_block = block_theta_lookup()

    best_int = search_best_param(
        x,
        y_raw,
        mask,
        "interference",
        THETA_COARSE_GRID,
        THETA_REFINE_RADIUS,
        THETA_REFINE_STEP,
        counts=counts,
        sums=sums,
        sumsqs=sumsqs,
    )
    best_poly = fit_parameter_free_model(
        x,
        y_raw,
        mask,
        "poly2",
        counts=counts,
        sums=sums,
        sumsqs=sumsqs,
    )

    x_valid, yz_int, mu_int, _, test_int = heldout_predictions(
        x, y_raw, mask, counts, sums, sumsqs, "interference", best_int["best_param"]
    )
    _, yz_poly, mu_poly, _, test_poly = heldout_predictions(
        x, y_raw, mask, counts, sums, sumsqs, "poly2", None
    )
    blocks_valid = blocks[test_int.any(axis=1)]

    res_int = np.where(test_int, yz_int - mu_int, np.nan)
    res_poly = np.where(test_poly, yz_poly - mu_poly, np.nan)

    residual_rows: list[dict[str, float | int]] = []
    distribution_rows: list[dict[str, float | int]] = []
    for block in BLOCKWISE_CRITICAL_BLOCKS:
        idx = np.where(blocks_valid == block)[0]
        for t in range(res_int.shape[1]):
            mask_cell = test_int[idx, t]
            if not np.any(mask_cell):
                continue
            cell_idx = idx[mask_cell]
            obs = yz_int[cell_idx, t]
            pred_int = mu_int[cell_idx, t]
            pred_poly = mu_poly[cell_idx, t]

            residual_rows.append(
                {
                    "block": int(block),
                    "theta_block": float(theta_by_block[block]),
                    "heldout_bin": int(t + 1),
                    "x_mean": float(np.mean(x_valid[cell_idx, t])),
                    "mean_residual_interference": float(np.mean(obs - pred_int)),
                    "mean_residual_poly": float(np.mean(obs - pred_poly)),
                    "abs_mean_residual_interference": float(abs(np.mean(obs - pred_int))),
                    "abs_mean_residual_poly": float(abs(np.mean(obs - pred_poly))),
                }
            )
            distribution_rows.append(
                {
                    "block": int(block),
                    "theta_block": float(theta_by_block[block]),
                    "heldout_bin": int(t + 1),
                    "n_points": int(obs.size),
                    "wasserstein_interference": empirical_wasserstein(obs, pred_int),
                    "wasserstein_poly": empirical_wasserstein(obs, pred_poly),
                    "quantile_error_interference": quantile_error(obs, pred_int),
                    "quantile_error_poly": quantile_error(obs, pred_poly),
                    "iqr_error_interference": iqr_error(obs, pred_int),
                    "iqr_error_poly": iqr_error(obs, pred_poly),
                }
            )

    residual_df = pd.DataFrame(residual_rows)
    distribution_df = pd.DataFrame(distribution_rows)
    residual_df.to_csv(OUT_DIR / "objective_mismatch_residual_cells.csv", index=False)
    distribution_df.to_csv(OUT_DIR / "objective_mismatch_distribution_cells.csv", index=False)

    high_blocks = residual_df["block"].isin([4, 7])
    final_bin = residual_df["heldout_bin"] == residual_df["heldout_bin"].max()
    metrics = {
        "block_curve_rmse_interference": float(best_int["rmse_test"]),
        "block_curve_rmse_poly": float(best_poly["rmse_test"]),
        "mean_abs_cell_residual_interference": float(residual_df["abs_mean_residual_interference"].mean()),
        "mean_abs_cell_residual_poly": float(residual_df["abs_mean_residual_poly"].mean()),
        "high_block_final_bin_abs_residual_interference": float(
            residual_df.loc[high_blocks & final_bin, "abs_mean_residual_interference"].mean()
        ),
        "high_block_final_bin_abs_residual_poly": float(
            residual_df.loc[high_blocks & final_bin, "abs_mean_residual_poly"].mean()
        ),
        "abs_residual_theta_corr_interference": float(
            np.corrcoef(
                residual_df["theta_block"],
                residual_df["abs_mean_residual_interference"],
            )[0, 1]
        ),
        "abs_residual_theta_corr_poly": float(
            np.corrcoef(
                residual_df["theta_block"],
                residual_df["abs_mean_residual_poly"],
            )[0, 1]
        ),
        "mean_wasserstein_interference": float(distribution_df["wasserstein_interference"].mean()),
        "mean_wasserstein_poly": float(distribution_df["wasserstein_poly"].mean()),
        "mean_quantile_error_interference": float(distribution_df["quantile_error_interference"].mean()),
        "mean_quantile_error_poly": float(distribution_df["quantile_error_poly"].mean()),
        "mean_iqr_error_interference": float(distribution_df["iqr_error_interference"].mean()),
        "mean_iqr_error_poly": float(distribution_df["iqr_error_poly"].mean()),
        "distribution_cells_wasserstein_win": float(
            (distribution_df["wasserstein_interference"] < distribution_df["wasserstein_poly"]).sum()
        ),
        "distribution_cells_quantile_win": float(
            (distribution_df["quantile_error_interference"] < distribution_df["quantile_error_poly"]).sum()
        ),
        "distribution_cells_iqr_win": float(
            (distribution_df["iqr_error_interference"] < distribution_df["iqr_error_poly"]).sum()
        ),
        "best_theta_block_holdout_interference": float(best_int["best_param"]),
    }
    return residual_df, distribution_df, metrics


def block_parameter_contrast(block_curves: list[dict[str, object]]) -> pd.DataFrame:
    x, y_raw, mask, _, blocks = arrays_from_curves(block_curves, include_block=True)
    y_std = row_standardize(y_raw, mask)
    designs = design_matrix_from_param(x, "poly2", None)
    _, beta_poly, _ = compute_rss_and_beta(designs, y_std, mask, ridge=HOLDOUT_RIDGE)

    rows: list[dict[str, float | int]] = []
    for block in BLOCKWISE_CRITICAL_BLOCKS:
        idx = np.where(blocks == block)[0]
        d1_mean = float(np.mean(beta_poly[idx, 1]))
        d2_mean = float(np.mean(beta_poly[idx, 2]))
        rows.append(
            {
                "block": int(block),
                "n_curves": int(idx.size),
                "d1_mean": d1_mean,
                "d1_sd": float(np.std(beta_poly[idx, 1], ddof=1)),
                "d2_mean": d2_mean,
                "d2_sd": float(np.std(beta_poly[idx, 2], ddof=1)),
                "coef_norm": float(np.hypot(d1_mean, d2_mean)),
                "coef_angle_deg": float(np.degrees(np.arctan2(d2_mean, d1_mean))),
            }
        )
    block_poly_df = pd.DataFrame(rows)
    block_poly_df.to_csv(OUT_DIR / "block_polynomial_coefficients.csv", index=False)
    return block_poly_df


def plot_residual_heatmaps(residual_df: pd.DataFrame) -> None:
    blocks = BLOCKWISE_CRITICAL_BLOCKS
    bins = sorted(residual_df["heldout_bin"].unique())
    int_mat = (
        residual_df.pivot(index="block", columns="heldout_bin", values="mean_residual_interference")
        .reindex(index=blocks, columns=bins)
        .to_numpy()
    )
    poly_mat = (
        residual_df.pivot(index="block", columns="heldout_bin", values="mean_residual_poly")
        .reindex(index=blocks, columns=bins)
        .to_numpy()
    )
    diff_mat = poly_mat - int_mat
    vmax = float(np.max(np.abs(np.concatenate([int_mat.ravel(), poly_mat.ravel()]))))
    diff_vmax = float(np.max(np.abs(diff_mat)))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, mat, cmap, limit in [
        (axes[0], int_mat, "coolwarm", vmax),
        (axes[1], poly_mat, "coolwarm", vmax),
        (axes[2], diff_mat, "PiYG", diff_vmax),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([str(v) for v in bins])
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels([str(v) for v in blocks])
        ax.set_xlabel("Held-out bin")
        for i in range(len(blocks)):
            for j in range(len(bins)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Block")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_structure_block_holdout.png", dpi=300)
    plt.close(fig)


def plot_distributional_fit(distribution_df: pd.DataFrame) -> None:
    metrics = [
        ("wasserstein_interference", "wasserstein_poly", "Wasserstein distance"),
        ("quantile_error_interference", "quantile_error_poly", "Quantile error"),
        ("iqr_error_interference", "iqr_error_poly", "IQR error"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    colors = ["#1f77b4", "#7f7f7f"]
    for ax, (col_int, col_poly, ylabel) in zip(axes, metrics):
        data = [distribution_df[col_int].to_numpy(), distribution_df[col_poly].to_numpy()]
        box = ax.boxplot(data, patch_artist=True, widths=0.55)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for idx, vals in enumerate(data, start=1):
            jitter = np.random.default_rng(123 + idx).normal(loc=idx, scale=0.04, size=len(vals))
            ax.scatter(jitter, vals, color="black", s=14, alpha=0.55)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Interference", "Polynomial"])
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "distributional_shape_holdout.png", dpi=300)
    plt.close(fig)


def plot_block_parameter_contrast(block_theta_df: pd.DataFrame, block_poly_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    x = np.arange(len(block_theta_df))
    y = block_theta_df["theta_mean"].to_numpy()
    yerr = np.vstack(
        [
            y - block_theta_df["theta_hdi_low"].to_numpy(),
            block_theta_df["theta_hdi_high"].to_numpy() - y,
        ]
    )
    axes[0].errorbar(x, y, yerr=yerr, fmt="o", capsize=4, color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(v) for v in block_theta_df["block"].tolist()])
    axes[0].set_xlabel("Block")
    axes[0].set_ylabel(r"$\theta$ (degrees)")

    axes[1].axhline(0, color="lightgray", linewidth=1)
    axes[1].axvline(0, color="lightgray", linewidth=1)
    axes[1].scatter(block_poly_df["d1_mean"], block_poly_df["d2_mean"], color="#7f7f7f", s=55)
    for row in block_poly_df.itertuples():
        axes[1].text(row.d1_mean + 0.05, row.d2_mean + 0.05, str(row.block), fontsize=9)
    axes[1].set_xlabel(r"$d_1$ mean")
    axes[1].set_ylabel(r"$d_2$ mean")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "block_parameter_contrast.png", dpi=300)
    plt.close(fig)


def write_tables(metrics: dict[str, float], block_poly_df: pd.DataFrame) -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "Block-curve RMSE",
                "Interference": metrics["block_curve_rmse_interference"],
                "Polynomial": metrics["block_curve_rmse_poly"],
            },
            {
                "Metric": "Mean abs. held-out cell residual",
                "Interference": metrics["mean_abs_cell_residual_interference"],
                "Polynomial": metrics["mean_abs_cell_residual_poly"],
            },
            {
                "Metric": "High-angle final-bin abs. residual",
                "Interference": metrics["high_block_final_bin_abs_residual_interference"],
                "Polynomial": metrics["high_block_final_bin_abs_residual_poly"],
            },
            {
                "Metric": r"Corr(abs. residual, block $\theta$)",
                "Interference": metrics["abs_residual_theta_corr_interference"],
                "Polynomial": metrics["abs_residual_theta_corr_poly"],
            },
            {
                "Metric": "Mean Wasserstein distance",
                "Interference": metrics["mean_wasserstein_interference"],
                "Polynomial": metrics["mean_wasserstein_poly"],
            },
            {
                "Metric": "Mean quantile error",
                "Interference": metrics["mean_quantile_error_interference"],
                "Polynomial": metrics["mean_quantile_error_poly"],
            },
            {
                "Metric": "Mean IQR error",
                "Interference": metrics["mean_iqr_error_interference"],
                "Polynomial": metrics["mean_iqr_error_poly"],
            },
        ]
    )
    with open(TAB_DIR / "Table_objective_mismatch_metrics.tex", "w", encoding="utf-8") as handle:
        handle.write(metrics_df.to_latex(index=False, escape=False, float_format="%.3f"))

    with open(TAB_DIR / "Table_block_polynomial_coefficients.tex", "w", encoding="utf-8") as handle:
        handle.write(block_poly_df.to_latex(index=False, float_format="%.3f"))


def main() -> None:
    ensure_dirs()
    block_curves = load_cached_curves("raw_block_curves_bins6_v2.pkl")

    residual_df, distribution_df, metrics = residual_and_distribution_analysis(block_curves)
    block_poly_df = block_parameter_contrast(block_curves)
    block_theta_df = pd.read_csv(OUT_DIR / "block_theta_summary.csv")

    plot_residual_heatmaps(residual_df)
    plot_distributional_fit(distribution_df)
    plot_block_parameter_contrast(block_theta_df, block_poly_df)
    write_tables(metrics, block_poly_df)

    summary = {
        "residual_metrics": metrics,
        "block_polynomial_coefficients": block_poly_df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "objective_mismatch_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
