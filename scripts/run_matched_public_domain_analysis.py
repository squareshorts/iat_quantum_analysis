import gc
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_submission_evidence import (
    BIN_OPTIONS,
    CRITICAL_BLOCKS,
    FIG_DIR,
    HOLDOUT_FRAC,
    K_COARSE_GRID,
    K_REFINE_RADIUS,
    K_REFINE_STEP,
    OUT_DIR,
    P_COARSE_GRID,
    P_REFINE_RADIUS,
    P_REFINE_STEP,
    PRIMARY_BINS,
    TAB_DIR,
    THETA_COARSE_GRID,
    THETA_POSTERIOR_GRID,
    THETA_REFINE_RADIUS,
    THETA_REFINE_STEP,
    arrays_from_curves,
    build_participant_curves_raw,
    fit_parameter_free_model,
    plot_model_comparison,
    posterior_summary,
    profile_theta_posterior,
    row_standardize,
    search_best_param,
    stats_from_curves,
)


BASE_DIR = Path(__file__).resolve().parent
RAW_COLS = {"task_name", "block_number", "trial_number", "trial_latency", "session_id"}
PROCESSED_COLS = ["pid", "block", "trial_in_block", "rt"]


def load_age_iat_df() -> pd.DataFrame:
    parquet_path = BASE_DIR / "data" / "processed" / "age_iat" / "age_iat_trials_standardized.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing standardized Age IAT parquet at {parquet_path}. "
            "Run `python -m src.data.download_age_iat` and `python -m src.data.prepare_age_iat` first."
        )
    df = pd.read_parquet(parquet_path)
    missing = [column for column in PROCESSED_COLS if column not in df.columns]
    if missing:
        raise RuntimeError(f"Standardized Age IAT parquet is missing required columns: {missing}")
    df = df[PROCESSED_COLS].dropna(subset=["pid", "block", "trial_in_block", "rt"]).copy()
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["trial_in_block"] = pd.to_numeric(df["trial_in_block"], errors="coerce")
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna(subset=["block", "trial_in_block", "rt"])
    df["block"] = df["block"].astype(int)
    df["trial_in_block"] = df["trial_in_block"].astype(int)
    return df[df["block"].isin(CRITICAL_BLOCKS)].copy()

DOMAIN_SPECS = [
    {
        "domain": "Gender-Science",
        "short": "gender_science_2019",
        "task_name": "scienceiat",
        "paths": sorted((BASE_DIR / "data" / "GenderScience_iat_2019" / "iat_2019").glob("iat*.txt")),
    },
    {
        "domain": "Sexuality",
        "short": "sexuality_2019",
        "task_name": "sexualityiat",
        "paths": sorted((BASE_DIR / "data" / "sexuality_raw" / "Sexuality_iat_2019" / "iat").glob("iat*.txt")),
    },
    {
        "domain": "Age",
        "short": "age_iat",
        "year": 2019,
        "loader": load_age_iat_df,
    },
]
DOMAIN_SPEC_BY_NAME = {spec["domain"].lower(): spec for spec in DOMAIN_SPECS}


def ensure_dirs():
    for path in [OUT_DIR, FIG_DIR, TAB_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def empty_domain_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PROCESSED_COLS)


def read_domain_raw_file(path: Path, task_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=lambda c: c.strip() in RAW_COLS,
            low_memory=False,
        )
    except Exception as exc:
        print(f"    Warning: skipping {path.name}; could not parse raw log ({type(exc).__name__}: {exc})")
        return empty_domain_frame()

    stripped_cols = {str(col).strip() for col in df.columns}
    missing_cols = sorted(RAW_COLS - stripped_cols)
    if missing_cols:
        print(f"    Warning: skipping {path.name}; missing required columns {missing_cols}")
        return empty_domain_frame()

    df = df.rename(columns={col: col.strip() for col in df.columns})
    df = df.rename(
        columns={
            "session_id": "pid",
            "block_number": "block",
            "trial_number": "trial_in_block",
            "trial_latency": "rt",
        }
    )
    keep = [*PROCESSED_COLS, "task_name"]
    df = df[keep].dropna(subset=["pid", "block", "trial_in_block", "rt", "task_name"])
    df["task_name"] = df["task_name"].astype(str).str.strip()
    df = df[df["task_name"] == task_name].copy()
    if df.empty:
        return empty_domain_frame()
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["trial_in_block"] = pd.to_numeric(df["trial_in_block"], errors="coerce")
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna(subset=["block", "trial_in_block", "rt"])
    df["block"] = df["block"].astype(int)
    df["trial_in_block"] = df["trial_in_block"].astype(int)
    df = df[df["block"].isin(CRITICAL_BLOCKS)].copy()
    return df[PROCESSED_COLS]


def cache_path_for(domain_short: str, n_bins: int) -> Path:
    return OUT_DIR / f"matched_public_{domain_short}_raw_curves_bins{n_bins}.pkl"


def load_domain_dataframe(spec: dict) -> pd.DataFrame:
    if "loader" in spec:
        return spec["loader"]()

    frames = []
    for path in spec["paths"]:
        df = read_domain_raw_file(path, task_name=spec["task_name"])
        if not df.empty:
            frames.append(df)
    if not frames:
        return empty_domain_frame()
    return pd.concat(frames, ignore_index=True)


def build_curves_for_domain(spec: dict, n_bins: int) -> list[dict]:
    cache_path = cache_path_for(spec["short"], n_bins)
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    curves: list[dict] = []
    if "loader" in spec:
        df = load_domain_dataframe(spec)
        curves.extend(build_participant_curves_raw(df, n_bins=n_bins))
        print(f"  Finished {spec['short']} | bins={n_bins}: {len(curves)} curves from standardized parquet")
    else:
        used_files = 0
        skipped_files = 0
        for path in spec["paths"]:
            print(f"  Reading {path.name} for {spec['short']} | bins={n_bins}")
            df = read_domain_raw_file(path, task_name=spec["task_name"])
            if df.empty:
                skipped_files += 1
                continue
            used_files += 1
            curves.extend(build_participant_curves_raw(df, n_bins=n_bins))
            del df
            gc.collect()

        print(
            f"  Finished {spec['short']} | bins={n_bins}: "
            f"{len(curves)} curves from {used_files} usable files"
            + (f"; skipped {skipped_files} malformed/empty files" if skipped_files else "")
        )
    with open(cache_path, "wb") as handle:
        pickle.dump(curves, handle)
    return curves


def profile_from_curves(curves: list[dict]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y_raw, mask, _ = arrays_from_curves(curves)
    y_std = row_standardize(y_raw, mask)
    posterior, rss = profile_theta_posterior(x, y_std, mask, THETA_POSTERIOR_GRID)
    summary = posterior_summary(THETA_POSTERIOR_GRID, posterior)
    return summary, posterior, rss, x, mask


def evaluate_models_for_curves(curves: list[dict]) -> pd.DataFrame:
    x, y_raw, mask, _ = arrays_from_curves(curves)
    counts, sums, sumsqs = stats_from_curves(curves)
    model_specs = [
        ("interference", THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("cos_only", THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("sin_only", THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("poly2", None, None, None),
        ("exp", K_COARSE_GRID, K_REFINE_RADIUS, K_REFINE_STEP),
        ("power", P_COARSE_GRID, P_REFINE_RADIUS, P_REFINE_STEP),
    ]
    rows = []
    for model_name, coarse_grid, refine_radius, refine_step in model_specs:
        print(f"    Hold-out model {model_name}")
        if coarse_grid is None:
            result = fit_parameter_free_model(
                x,
                y_raw,
                mask,
                model_name,
                counts=counts,
                sums=sums,
                sumsqs=sumsqs,
            )
        else:
            result = search_best_param(
                x,
                y_raw,
                mask,
                model_name,
                coarse_grid,
                refine_radius,
                refine_step,
                counts=counts,
                sums=sums,
                sumsqs=sumsqs,
            )
        rows.append(
            {
                "model": model_name,
                "best_param": result["best_param"],
                "train_rss": result["train_rss"],
                "rmse_test": result["rmse_test"],
                "mean_logscore_test": result["mean_logscore_test"],
                "sigma_train": result["sigma_train"],
                "n_curves": result["n_curves"],
                "n_test_obs": result["n_test_obs"],
                "boundary_hit": result["boundary_hit"],
            }
        )
    return pd.DataFrame(rows)


def plot_theta_overlay(summary_df: pd.DataFrame, profile_map: dict[str, np.ndarray], out_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.2))
    palette = {
        "Gender-Science": "#1f77b4",
        "Sexuality": "#d62728",
        "Age": "#2ca02c",
    }
    for _, row in summary_df.iterrows():
        domain = row["domain"]
        profile = profile_map[domain]
        color = palette.get(domain)
        plt.plot(THETA_POSTERIOR_GRID, profile / profile.max(), lw=2.2, label=domain, color=color)
        plt.axvline(row["theta_map"], ls="--", lw=1.2, color=color, alpha=0.9)
    plt.xlabel(r"$\theta$ (degrees)")
    plt.ylabel("Normalized posterior")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_theta_bar(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.8, 4.6))
    palette = {
        "Gender-Science": "#1f77b4",
        "Sexuality": "#d62728",
        "Age": "#2ca02c",
    }
    colors = [palette.get(domain, "#7f7f7f") for domain in summary_df["domain"]]
    y = summary_df["theta_mean"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - summary_df["hdi_low"].to_numpy(dtype=float),
            summary_df["hdi_high"].to_numpy(dtype=float) - y,
        ]
    )
    plt.bar(summary_df["domain"], y, color=colors, alpha=0.9)
    plt.errorbar(summary_df["domain"], y, yerr=yerr, fmt="none", ecolor="black", capsize=5, lw=1.2)
    plt.ylabel(r"$\theta$ mean (degrees)")
    plt.title("Matched public raw-domain theta summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def posterior_difference_summary(
    label_a: str,
    label_b: str,
    posterior_a: np.ndarray,
    posterior_b: np.ndarray,
) -> pd.DataFrame:
    theta_a = THETA_POSTERIOR_GRID[:, None]
    theta_b = THETA_POSTERIOR_GRID[None, :]
    delta = (theta_b - theta_a).ravel()
    weights = np.outer(posterior_a, posterior_b).ravel()
    delta_support, inverse = np.unique(delta, return_inverse=True)
    delta_posterior = np.zeros_like(delta_support, dtype=float)
    np.add.at(delta_posterior, inverse, weights)
    delta_posterior = delta_posterior / delta_posterior.sum()
    summary = posterior_summary(delta_support, delta_posterior)
    return pd.DataFrame(
        [
            {
                "contrast": f"{label_b} - {label_a}",
                "delta_mean": summary["mean"],
                "delta_sd": summary["sd"],
                "delta_map": summary["map"],
                "hdi_low": summary["hdi_low"],
                "hdi_high": summary["hdi_high"],
                "prob_gt_zero": float(delta_posterior[delta_support > 0].sum()),
            }
        ]
    )


def pairwise_posterior_difference_summaries(profile_map: dict[str, np.ndarray]) -> pd.DataFrame:
    labels = list(profile_map)
    frames = []
    for idx_a, label_a in enumerate(labels):
        for label_b in labels[idx_a + 1 :]:
            frames.append(
                posterior_difference_summary(
                    label_a=label_a,
                    label_b=label_b,
                    posterior_a=profile_map[label_a],
                    posterior_b=profile_map[label_b],
                )
            )
    return pd.concat(frames, ignore_index=True)


def main():
    ensure_dirs()
    t0 = time.time()

    primary_rows = []
    bins_rows = []
    profile_map: dict[str, np.ndarray] = {}
    model_frames = []

    for spec in DOMAIN_SPECS:
        if not spec.get("loader") and not spec["paths"]:
            raise FileNotFoundError(f"No raw files found for {spec['domain']}")

        print(f"\n=== {spec['domain']} ===")
        primary_curves = None
        for n_bins in BIN_OPTIONS:
            curves = build_curves_for_domain(spec=spec, n_bins=n_bins)
            if not curves:
                raise RuntimeError(f"No curves built for {spec['domain']} at {n_bins} bins")
            summary, posterior, rss, _, _ = profile_from_curves(curves)
            bins_rows.append(
                {
                    "domain": spec["domain"],
                    "bins": n_bins,
                    "n_curves": len(curves),
                    "theta_mean": summary["mean"],
                    "theta_sd": summary["sd"],
                    "theta_map": summary["map"],
                    "hdi_low": summary["hdi_low"],
                    "hdi_high": summary["hdi_high"],
                    "posterior_entropy": float(-np.sum(posterior * np.log(np.clip(posterior, 1e-12, None)))),
                    "rss_at_map": float(rss[np.argmax(posterior)]),
                }
            )
            if n_bins == PRIMARY_BINS:
                primary_curves = curves
                profile_map[spec["domain"]] = posterior
                primary_rows.append(
                    {
                        "domain": spec["domain"],
                        "year": spec.get("year", 2019),
                        "n_curves": len(curves),
                        "theta_mean": summary["mean"],
                        "theta_sd": summary["sd"],
                        "theta_map": summary["map"],
                        "hdi_low": summary["hdi_low"],
                        "hdi_high": summary["hdi_high"],
                    }
                )
                pd.DataFrame(
                    {
                        "theta_deg": THETA_POSTERIOR_GRID,
                        "posterior": posterior,
                        "rss": rss,
                    }
                ).to_csv(OUT_DIR / f"{spec['short']}_theta_profile.csv", index=False)

        if primary_curves is None:
            raise RuntimeError(f"Missing primary-bin curves for {spec['domain']}")
        comp_df = evaluate_models_for_curves(primary_curves)
        comp_df.insert(0, "domain", spec["domain"])
        comp_df.insert(1, "year", spec.get("year", 2019))
        comp_df.insert(2, "bins", PRIMARY_BINS)
        model_frames.append(comp_df)

    primary_df = pd.DataFrame(primary_rows).sort_values("theta_mean").reset_index(drop=True)
    bins_df = pd.DataFrame(bins_rows).sort_values(["domain", "bins"]).reset_index(drop=True)
    model_df = pd.concat(model_frames, ignore_index=True)
    delta_df = pairwise_posterior_difference_summaries(profile_map)

    primary_df.to_csv(OUT_DIR / "matched_public_domain_theta_summary.csv", index=False)
    bins_df.to_csv(OUT_DIR / "matched_public_domain_theta_bins.csv", index=False)
    model_df.to_csv(OUT_DIR / "matched_public_domain_model_comparison.csv", index=False)
    delta_df.to_csv(OUT_DIR / "matched_public_domain_theta_difference.csv", index=False)

    primary_table = primary_df.rename(
        columns={
            "domain": "Task",
            "year": "Year",
            "n_curves": "N",
            "theta_mean": "Theta mean",
            "theta_sd": "Theta SD",
            "theta_map": "Theta MAP",
            "hdi_low": "HDI low",
            "hdi_high": "HDI high",
        }
    )
    bins_table = bins_df.rename(
        columns={
            "domain": "Task",
            "bins": "Bins",
            "n_curves": "N",
            "theta_mean": "Theta mean",
            "theta_sd": "Theta SD",
            "theta_map": "Theta MAP",
            "hdi_low": "HDI low",
            "hdi_high": "HDI high",
            "posterior_entropy": "Posterior entropy",
            "rss_at_map": "RSS at MAP",
        }
    )
    model_table = model_df.rename(
        columns={
            "domain": "Task",
            "year": "Year",
            "bins": "Bins",
            "model": "Model",
            "best_param": "Best parameter",
            "train_rss": "Train RSS",
            "rmse_test": "RMSE test",
            "mean_logscore_test": "Mean log score test",
            "sigma_train": "Sigma train",
            "n_curves": "N",
            "n_test_obs": "Test observations",
            "boundary_hit": "Boundary hit",
        }
    )
    delta_table = delta_df.rename(
        columns={
            "contrast": "Contrast",
            "delta_mean": "Delta mean",
            "delta_sd": "Delta SD",
            "delta_map": "Delta MAP",
            "hdi_low": "HDI low",
            "hdi_high": "HDI high",
            "prob_gt_zero": "P(delta > 0)",
        }
    )

    with open(TAB_DIR / "Table_matched_public_domain_theta.tex", "w", encoding="utf-8") as handle:
        handle.write(primary_table.to_latex(index=False, float_format="%.3f"))
    with open(TAB_DIR / "Table_matched_public_domain_theta_bins.tex", "w", encoding="utf-8") as handle:
        handle.write(bins_table.to_latex(index=False, float_format="%.3f"))
    with open(TAB_DIR / "Table_matched_public_domain_model_comparison.tex", "w", encoding="utf-8") as handle:
        handle.write(model_table.to_latex(index=False, float_format="%.4f"))
    with open(TAB_DIR / "Table_matched_public_domain_theta_difference.tex", "w", encoding="utf-8") as handle:
        handle.write(delta_table.to_latex(index=False, float_format="%.3f"))

    plot_theta_overlay(primary_df, profile_map, FIG_DIR / "matched_public_domain_theta_overlay.png")
    plot_theta_bar(primary_df, FIG_DIR / "matched_public_domain_theta_bar.png")

    for domain_name in primary_df["domain"].tolist():
        plot_model_comparison(
            model_df[model_df["domain"] == domain_name].reset_index(drop=True),
            FIG_DIR / f"{domain_name.lower().replace('-', '_').replace(' ', '_')}_model_comparison_bins6.png",
        )

    summary = {
        "domains": primary_rows,
        "theta_difference": delta_df.to_dict(orient="records"),
        "elapsed_minutes": (time.time() - t0) / 60.0,
        "holdout_fraction": HOLDOUT_FRAC,
    }
    with open(OUT_DIR / "matched_public_domain_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
