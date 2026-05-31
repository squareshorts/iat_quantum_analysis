from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_matched_public_domain_analysis as matched_public
from run_submission_evidence import BIN_OPTIONS, PRIMARY_BINS, THETA_POSTERIOR_GRID


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"

DOMAIN_PALETTE = {
    "Gender-Science": "#1f77b4",
    "Sexuality": "#d62728",
    "Age": "#2ca02c",
}


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def get_domain_spec(domain: str) -> dict:
    spec = matched_public.DOMAIN_SPEC_BY_NAME.get(domain.strip().lower())
    if spec is None:
        supported = ", ".join(spec["domain"] for spec in matched_public.DOMAIN_SPECS)
        raise KeyError(f"Unknown domain '{domain}'. Supported domains: {supported}")
    return spec


def domain_slug(spec: dict) -> str:
    return spec["short"]


def ensure_domain_dirs(spec: dict) -> tuple[Path, Path, Path]:
    slug = domain_slug(spec)
    result_dir = RESULTS_DIR / slug
    figure_dir = FIGURES_DIR / slug
    table_dir = TABLES_DIR / slug
    for path in [result_dir, figure_dir, table_dir, RESULTS_DIR / "cross_domain", FIGURES_DIR / "cross_domain"]:
        path.mkdir(parents=True, exist_ok=True)
    return result_dir, figure_dir, table_dir


def _quick_qc_summary(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    summary = {
        "domain": spec["domain"],
        "raw_rows": len(df),
        "retained_trial_rows": len(df),
        "participants": df["pid"].nunique(),
        "available_blocks": ", ".join(str(v) for v in sorted(df["block"].dropna().astype(int).unique().tolist())),
        "retained_critical_blocks": ", ".join(str(v) for v in sorted(df["block"].dropna().astype(int).unique().tolist())),
        "rt_median": float(df["rt"].median()),
        "rt_mean": float(df["rt"].mean()),
        "rt_sd": float(df["rt"].std(ddof=1)),
        "prop_fast_lt300": float(df["rt"].lt(300).mean()),
        "prop_long_gt10000": float(df["rt"].gt(10000).mean()),
        "error_rate": float(df["trial_error"].fillna(0).mean()) if "trial_error" in df.columns else np.nan,
    }
    return pd.DataFrame([summary])


def load_qc_summary(spec: dict, df: pd.DataFrame) -> pd.DataFrame:
    if spec["domain"] == "Age":
        qc_path = BASE_DIR / "data" / "processed" / "age_iat" / "age_iat_qc_summary.csv"
        if qc_path.exists():
            return pd.read_csv(qc_path)
    return _quick_qc_summary(df, spec)


def _summary_table(df: pd.DataFrame, path: Path, float_format: str = "%.3f") -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(df.to_latex(index=False, float_format=float_format))


def plot_domain_main_figure(
    *,
    domain: str,
    posterior_df: pd.DataFrame,
    model_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    color = DOMAIN_PALETTE.get(domain, "#7f7f7f")

    axes[0].plot(posterior_df["theta_deg"], posterior_df["posterior"], color=color, lw=2)
    axes[0].set_xlabel(r"$\theta$ (degrees)")
    axes[0].set_ylabel("Posterior density")
    axes[0].set_title(f"{domain} posterior")

    model_df = model_df.sort_values("rmse_test").reset_index(drop=True)
    axes[1].bar(model_df["model"], model_df["rmse_test"], color=color, alpha=0.85)
    axes[1].set_ylabel("Temporal hold-out RMSE")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title(f"{domain} model comparison")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def analyze_domain(domain: str) -> dict[str, object]:
    spec = get_domain_spec(domain)
    result_dir, figure_dir, table_dir = ensure_domain_dirs(spec)
    slug = domain_slug(spec)

    LOGGER.info("Loading %s trials", spec["domain"])
    df = matched_public.load_domain_dataframe(spec)
    if df.empty:
        raise RuntimeError(f"No standardized trial-level rows are available for {spec['domain']}.")

    if "rt" not in df.columns:
        raise ValueError("Cleaned latency column 'rt' is missing from the analysis dataframe.")
    
    invalid_rt = ~np.isfinite(df["rt"])
    if invalid_rt.any():
        raise ValueError(f"Found {invalid_rt.sum()} non-finite 'rt' values in {spec['domain']} dataset.")
    
    fast_rt = df["rt"] < 300
    if fast_rt.any():
        raise ValueError(f"Found {fast_rt.sum()} 'rt' values < 300 ms in {spec['domain']} dataset. These must be excluded during preprocessing.")
        
    long_rt = df["rt"] > 10000
    if long_rt.any():
        raise ValueError(f"Found {long_rt.sum()} 'rt' values > 10000 ms in {spec['domain']} dataset. These must be excluded during preprocessing.")

    primary_curves: list[dict] | None = None
    bins_rows: list[dict[str, object]] = []
    posterior_df = pd.DataFrame()
    primary_summary: dict[str, object] | None = None

    for n_bins in BIN_OPTIONS:
        curves = matched_public.build_curves_for_domain(spec=spec, n_bins=n_bins)
        if not curves:
            raise RuntimeError(f"No participant curves were built for {spec['domain']} at {n_bins} bins.")
        summary, posterior, rss, _, _ = matched_public.profile_from_curves(curves)
        bins_rows.append(
            {
                "domain": spec["domain"],
                "year": spec.get("year", 2019),
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
            primary_summary = summary
            posterior_df = pd.DataFrame(
                {
                    "theta_deg": THETA_POSTERIOR_GRID,
                    "posterior": posterior,
                    "rss": rss,
                }
            )

    if primary_curves is None or primary_summary is None:
        raise RuntimeError(f"Primary-bin analysis did not complete for {spec['domain']}.")

    model_df = matched_public.evaluate_models_for_curves(primary_curves)
    model_df.insert(0, "domain", spec["domain"])
    model_df.insert(1, "year", spec.get("year", 2019))
    model_df.insert(2, "bins", PRIMARY_BINS)

    qc_df = load_qc_summary(spec, df)
    qc_out_path = result_dir / f"{slug}_qc_summary.csv"
    qc_df.to_csv(qc_out_path, index=False)

    domain_summary_df = pd.DataFrame(
        [
            {
                "domain": spec["domain"],
                "year": spec.get("year", 2019),
                "raw_rows": len(df),
                "participants": df["pid"].nunique(),
                "n_curves": len(primary_curves),
                "bins": PRIMARY_BINS,
                "theta_mean": primary_summary["mean"],
                "theta_sd": primary_summary["sd"],
                "theta_map": primary_summary["map"],
                "hdi_low": primary_summary["hdi_low"],
                "hdi_high": primary_summary["hdi_high"],
            }
        ]
    )
    posterior_summary_df = pd.DataFrame(
        [
            {
                "domain": spec["domain"],
                "year": spec.get("year", 2019),
                "theta_mean": primary_summary["mean"],
                "theta_sd": primary_summary["sd"],
                "theta_map": primary_summary["map"],
                "hdi_low": primary_summary["hdi_low"],
                "hdi_high": primary_summary["hdi_high"],
                "posterior_entropy": float(-np.sum(posterior_df["posterior"] * np.log(np.clip(posterior_df["posterior"], 1e-12, None)))),
            }
        ]
    )
    angle_overlap_df = pd.DataFrame(
        [
            {
                "domain": spec["domain"],
                "year": spec.get("year", 2019),
                "theta_mean_deg": primary_summary["mean"],
                "theta_map_deg": primary_summary["map"],
                "theta_hdi_low_deg": primary_summary["hdi_low"],
                "theta_hdi_high_deg": primary_summary["hdi_high"],
                "contextual_overlap_cos_theta_mean": float(np.cos(np.deg2rad(primary_summary["mean"]))),
                "contextual_overlap_cos_theta_map": float(np.cos(np.deg2rad(primary_summary["map"]))),
            }
        ]
    )

    domain_summary_path = result_dir / f"{slug}_domain_summary.csv"
    posterior_summary_path = result_dir / f"{slug}_posterior_summary.csv"
    angle_overlap_path = result_dir / f"{slug}_angle_or_overlap_summary.csv"
    theta_profile_path = result_dir / f"{slug}_theta_profile.csv"
    model_path = result_dir / f"{slug}_model_comparison.csv"
    bins_path = result_dir / f"{slug}_theta_bins.csv"
    figure_path = figure_dir / f"{slug}_main_figure.png"

    domain_summary_df.to_csv(domain_summary_path, index=False)
    posterior_summary_df.to_csv(posterior_summary_path, index=False)
    angle_overlap_df.to_csv(angle_overlap_path, index=False)
    posterior_df.to_csv(theta_profile_path, index=False)
    model_df.to_csv(model_path, index=False)
    pd.DataFrame(bins_rows).to_csv(bins_path, index=False)

    _summary_table(domain_summary_df, table_dir / f"{slug}_domain_summary.tex")
    _summary_table(posterior_summary_df, table_dir / f"{slug}_posterior_summary.tex")
    plot_domain_main_figure(
        domain=spec["domain"],
        posterior_df=posterior_df,
        model_df=model_df,
        out_path=figure_path,
    )

    LOGGER.info("Saved %s domain outputs under %s", spec["domain"], result_dir)
    return {
        "domain": spec["domain"],
        "slug": slug,
        "result_dir": str(result_dir),
        "figure_path": str(figure_path),
        "domain_summary_path": str(domain_summary_path),
        "posterior_summary_path": str(posterior_summary_path),
        "angle_overlap_path": str(angle_overlap_path),
        "theta_profile_path": str(theta_profile_path),
        "qc_summary_path": str(qc_out_path),
    }


def _load_profile_for_aggregate(domain: str, path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    profile = pd.read_csv(path)
    profile["domain"] = domain
    return profile


def refresh_public_domain_aggregate() -> dict[str, object]:
    cross_result_dir = RESULTS_DIR / "cross_domain"
    cross_figure_dir = FIGURES_DIR / "cross_domain"
    cross_result_dir.mkdir(parents=True, exist_ok=True)
    cross_figure_dir.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    legacy_summary_path = BASE_DIR / "outputs" / "matched_public_domain_theta_summary.csv"
    if legacy_summary_path.exists():
        legacy_df = pd.read_csv(legacy_summary_path)
        summary_frames.append(legacy_df[legacy_df["domain"].isin(["Gender-Science", "Sexuality"])])

    age_summary_path = RESULTS_DIR / "age_iat" / "age_iat_domain_summary.csv"
    if age_summary_path.exists():
        summary_frames.append(pd.read_csv(age_summary_path)[["domain", "year", "n_curves", "theta_mean", "theta_sd", "theta_map", "hdi_low", "hdi_high"]])

    if not summary_frames:
        raise RuntimeError("No public-domain summary files are available to aggregate.")

    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df = summary_df.drop_duplicates(subset=["domain"], keep="last").sort_values("theta_mean").reset_index(drop=True)

    summary_out = cross_result_dir / "public_domain_theta_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    summary_df.to_csv(BASE_DIR / "outputs" / "matched_public_domain_theta_summary.csv", index=False)

    plt.figure(figsize=(6.8, 4.6))
    colors = [DOMAIN_PALETTE.get(domain, "#7f7f7f") for domain in summary_df["domain"]]
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
    plt.savefig(cross_figure_dir / "public_domain_theta_bar.png", dpi=300)
    plt.savefig(BASE_DIR / "figures" / "matched_public_domain_theta_bar.png", dpi=300)
    plt.close()

    profile_map: dict[str, np.ndarray] = {}
    profile_frames = [
        _load_profile_for_aggregate("Gender-Science", BASE_DIR / "outputs" / "gender_science_2019_theta_profile.csv"),
        _load_profile_for_aggregate("Sexuality", BASE_DIR / "outputs" / "sexuality_2019_theta_profile.csv"),
        _load_profile_for_aggregate("Age", RESULTS_DIR / "age_iat" / "age_iat_theta_profile.csv"),
    ]
    profile_frames = [frame for frame in profile_frames if frame is not None]
    if profile_frames:
        plt.figure(figsize=(8.5, 5.2))
        for frame in profile_frames:
            domain = str(frame["domain"].iloc[0])
            profile_map[domain] = frame["posterior"].to_numpy(dtype=float)
            plt.plot(frame["theta_deg"], frame["posterior"] / frame["posterior"].max(), lw=2.2, label=domain, color=DOMAIN_PALETTE.get(domain, "#7f7f7f"))
            theta_map = summary_df.loc[summary_df["domain"].eq(domain), "theta_map"]
            if not theta_map.empty:
                plt.axvline(theta_map.iloc[0], ls="--", lw=1.2, color=DOMAIN_PALETTE.get(domain, "#7f7f7f"), alpha=0.9)
        plt.xlabel(r"$\theta$ (degrees)")
        plt.ylabel("Normalized posterior")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(cross_figure_dir / "public_domain_theta_overlay.png", dpi=300)
        plt.savefig(BASE_DIR / "figures" / "matched_public_domain_theta_overlay.png", dpi=300)
        plt.close()

    if len(profile_map) >= 2:
        delta_df = matched_public.pairwise_posterior_difference_summaries(profile_map)
        delta_df.to_csv(cross_result_dir / "public_domain_theta_difference.csv", index=False)
        delta_df.to_csv(BASE_DIR / "outputs" / "matched_public_domain_theta_difference.csv", index=False)

    table_df = summary_df.rename(
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
    with (BASE_DIR / "tables" / "Table_matched_public_domain_theta.tex").open("w", encoding="utf-8") as handle:
        handle.write(table_df.to_latex(index=False, float_format="%.3f"))

    return {
        "summary_path": str(summary_out),
        "domains": summary_df["domain"].tolist(),
    }
