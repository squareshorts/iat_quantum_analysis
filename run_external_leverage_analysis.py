from __future__ import annotations

import csv
import glob
import io
import json
import pickle
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
TAB_DIR = BASE_DIR / "tables"

EXTERNAL_ZIP = DATA_DIR / "life_satisfaction_iat_2024" / "raw" / "187results.zip"
EXTERNAL_PHASE_MAP = {"fase1": 1, "fase2": 2, "fase3": 3, "fase4": 4, "fase5": 5}
EXTERNAL_CRITICAL_PHASES = ["fase3", "fase5"]
DEFAULT_BINS = 6
HOLDOUT_FRAC = 0.20
HDI_PROB = 0.94

THETA_FINE_GRID = np.arange(0.0, 180.0 + 0.25, 0.25)
THETA_COARSE_GRID = np.arange(0.0, 180.0 + 1.0, 1.0)
THETA_SEARCH_GRID = np.arange(0.0, 180.0 + 2.0, 2.0)
K_GRID = np.linspace(0.0, 12.0, 49)
P_GRID = np.linspace(0.05, 4.0, 40)

RAW_CURVES_CACHE = OUT_DIR / "raw_curves_bins6_v2.pkl"
GS_POSTERIOR_CSV = OUT_DIR / "theta_grid_profile_submission.csv"


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, TAB_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def hdi_from_discrete(grid_vals: np.ndarray, probs: np.ndarray, mass: float = HDI_PROB) -> tuple[float, float]:
    order = np.argsort(probs)[::-1]
    chosen: list[int] = []
    total = 0.0
    for idx in order:
        chosen.append(int(idx))
        total += float(probs[idx])
        if total >= mass:
            break
    sel = np.asarray(chosen, dtype=int)
    return float(grid_vals[sel].min()), float(grid_vals[sel].max())


def posterior_summary(grid_vals: np.ndarray, posterior: np.ndarray) -> dict[str, float]:
    mean = float((grid_vals * posterior).sum())
    sd = float(np.sqrt(((grid_vals - mean) ** 2 * posterior).sum()))
    hdi_low, hdi_high = hdi_from_discrete(grid_vals, posterior, mass=HDI_PROB)
    theta_map = float(grid_vals[np.argmax(posterior)])
    return {
        "theta_mean": mean,
        "theta_sd": sd,
        "theta_map": theta_map,
        "hdi_low": hdi_low,
        "hdi_high": hdi_high,
    }


def arrays_from_curves(curves: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_t = max(len(curve["x"]) for curve in curves)
    n_curves = len(curves)
    x = np.full((n_curves, max_t), np.nan, dtype=float)
    y = np.full((n_curves, max_t), np.nan, dtype=float)
    mask = np.zeros((n_curves, max_t), dtype=bool)
    for i, curve in enumerate(curves):
        t = len(curve["x"])
        x[i, :t] = curve["x"]
        y[i, :t] = curve["y"]
        mask[i, :t] = True
    return x, y, mask


def profile_theta_posterior(x: np.ndarray, y_std: np.ndarray, mask: np.ndarray, theta_grid_deg: np.ndarray) -> np.ndarray:
    n_obs = int(mask.sum())
    y2_sum = float(np.sum((y_std ** 2)[mask]))
    ones = np.ones_like(x)
    post_log: list[float] = []

    for theta_deg in theta_grid_deg:
        arg = np.deg2rad(theta_deg) * x
        c = np.cos(arg)
        s = np.sin(arg)
        c[~mask] = 0.0
        s[~mask] = 0.0
        o = ones.copy()
        o[~mask] = 0.0
        ym = np.where(mask, y_std, 0.0)

        s_o = o.sum(axis=1)
        s_c = c.sum(axis=1)
        s_s = s.sum(axis=1)
        s_cc = (c * c).sum(axis=1)
        s_ss = (s * s).sum(axis=1)
        s_cs = (c * s).sum(axis=1)
        s_y = ym.sum(axis=1)
        s_yc = (ym * c).sum(axis=1)
        s_ys = (ym * s).sum(axis=1)

        a = np.stack([
            np.stack([s_o, s_c, s_s], axis=-1),
            np.stack([s_c, s_cc, s_cs], axis=-1),
            np.stack([s_s, s_cs, s_ss], axis=-1),
        ], axis=-2)
        b = np.stack([s_y, s_yc, s_ys], axis=-1)
        a[..., range(3), range(3)] += 1e-8
        proj = (b[..., None, :] @ np.linalg.solve(a, b[..., :, None])).squeeze(-1).squeeze(-1)
        rss_theta = y2_sum - float(proj.sum())
        post_log.append(-(n_obs / 2.0) * np.log(max(rss_theta, 1e-12)))

    post_log_arr = np.asarray(post_log, dtype=float)
    post_log_arr -= float(post_log_arr.max())
    posterior = np.exp(post_log_arr)
    posterior /= posterior.sum()
    return posterior


def temporal_holdout_mask(x: np.ndarray, mask: np.ndarray, frac: float, min_train_pts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, _ = x.shape
    train_mask = np.zeros_like(mask, dtype=bool)
    test_mask = np.zeros_like(mask, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        idx = np.where(mask[i])[0]
        n_obs = idx.size
        if n_obs < min_train_pts + 1:
            continue
        ordered = idx[np.argsort(x[i, idx])]
        k = max(1, int(np.ceil(frac * n_obs)))
        k = min(k, n_obs - min_train_pts)
        test_idx = ordered[-k:]
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)
        if train_idx.size < min_train_pts:
            continue
        train_mask[i, train_idx] = True
        test_mask[i, test_idx] = True
        valid[i] = True
    return train_mask, test_mask, valid


def leak_free_standardize(y_raw: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    yz = np.full_like(y_raw, np.nan, dtype=float)
    for i in range(y_raw.shape[0]):
        idx_train = np.where(train_mask[i])[0]
        idx_all = np.where(~np.isnan(y_raw[i]))[0]
        if idx_train.size == 0 or idx_all.size == 0:
            continue
        vals_train = y_raw[i, idx_train]
        mu = float(np.mean(vals_train))
        sd = float(np.std(vals_train, ddof=1))
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        yz[i, idx_all] = (y_raw[i, idx_all] - mu) / sd
    return yz


def design_matrix(model_name: str, x: np.ndarray, param: float | None = None) -> np.ndarray:
    if model_name == "interference":
        arg = np.deg2rad(float(param)) * x
        return np.column_stack([np.ones(len(x)), np.cos(arg), np.sin(arg)])
    if model_name == "cos_only":
        arg = np.deg2rad(float(param)) * x
        return np.column_stack([np.ones(len(x)), np.cos(arg)])
    if model_name == "sin_only":
        arg = np.deg2rad(float(param)) * x
        return np.column_stack([np.ones(len(x)), np.sin(arg)])
    if model_name == "poly2":
        return np.column_stack([np.ones(len(x)), x, x ** 2])
    if model_name == "exp":
        return np.column_stack([np.ones(len(x)), np.exp(-float(param) * x)])
    if model_name == "power":
        return np.column_stack([np.ones(len(x)), np.power(np.clip(x, 1e-6, None), float(param))])
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model_rmse(
    x: np.ndarray,
    y_raw: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    grid: np.ndarray | None,
) -> tuple[float | None, float]:
    min_train_pts = 3
    train_mask, test_mask, valid = temporal_holdout_mask(x, mask, HOLDOUT_FRAC, min_train_pts=min_train_pts)
    yz = leak_free_standardize(y_raw, train_mask)
    x = x[valid]
    yz = yz[valid]
    train_mask = train_mask[valid]
    test_mask = test_mask[valid]

    params = [None] if grid is None else list(grid)
    best_param = None
    best_rmse = np.inf

    for param in params:
        sq_errors: list[float] = []
        for i in range(x.shape[0]):
            tr = np.where(train_mask[i])[0]
            te = np.where(test_mask[i])[0]
            a = design_matrix(model_name, x[i, tr], param)
            b = design_matrix(model_name, x[i, te], param)
            beta = np.linalg.lstsq(a, yz[i, tr], rcond=None)[0]
            pred = b @ beta
            sq_errors.extend((yz[i, te] - pred) ** 2)
        rmse = float(np.sqrt(np.mean(sq_errors)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_param = None if param is None else float(param)
    return best_param, best_rmse


def parse_testable_csv(text: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    parts = [part for part in text.split("\n\n") if part.strip()]
    if len(parts) < 2:
        raise ValueError("Unexpected Testable export format.")
    meta = next(csv.DictReader(io.StringIO(parts[0])))
    attempts = list(csv.DictReader(io.StringIO(parts[1])))
    return meta, attempts


def collapse_attempts(file_name: str, meta: dict[str, str], attempts: list[dict[str, str]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    participant_name = (meta.get("Nome completo") or "").strip()
    gmt_timestamp = meta.get("GMT_timestamp")
    duration_s = float(meta.get("duration_s") or np.nan)
    rts = [float(row["RT"]) for row in attempts if row.get("RT")]
    fast_prop = float(np.mean(np.asarray(rts) < 300.0)) if rts else np.nan

    grouped: dict[tuple[str, str], dict[str, object]] = {}
    appearance_order: list[tuple[str, str]] = []

    for row in attempts:
        phase = row.get("condition1")
        trial_no = row.get("trialNo")
        if not phase or not trial_no:
            continue
        base_trial = str(trial_no).split("_")[0]
        key = (phase, base_trial)
        if key not in grouped:
            digits = re.sub(r"\D", "", base_trial)
            grouped[key] = {
                "phase": phase,
                "base_trial": int(digits) if digits else len(grouped) + 1,
                "rt_total": 0.0,
                "n_attempts": 0,
                "n_errors": 0,
            }
            appearance_order.append(key)
        grouped[key]["rt_total"] += float(row["RT"]) if row.get("RT") else np.nan
        grouped[key]["n_attempts"] += 1
        if row.get("correct") not in ("", None):
            grouped[key]["n_errors"] += int(int(row["correct"]) == 0)

    collapsed_rows = []
    for key in appearance_order:
        rec = grouped[key]
        collapsed_rows.append(
            {
                "source_file": file_name,
                "participant_name": participant_name,
                "gmt_timestamp": gmt_timestamp,
                "phase": rec["phase"],
                "base_trial": rec["base_trial"],
                "rt_total": rec["rt_total"],
                "n_attempts": rec["n_attempts"],
                "n_errors": rec["n_errors"],
            }
        )

    quality_row = {
        "source_file": file_name,
        "participant_name": participant_name,
        "gmt_timestamp": gmt_timestamp,
        "duration_s": duration_s,
        "prop_fast_lt300_attempts": fast_prop,
        "n_attempt_rows": len(attempts),
    }
    return quality_row, collapsed_rows


def load_external_archive(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_rows: list[dict[str, object]] = []
    collapsed_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(name for name in zf.namelist() if name.lower().endswith(".csv"))
        for member in members:
            meta, attempts = parse_testable_csv(zf.read(member).decode("utf-8-sig"))
            quality_row, rows = collapse_attempts(member, meta, attempts)
            quality_rows.append(quality_row)
            collapsed_rows.extend(rows)
    return pd.DataFrame(quality_rows), pd.DataFrame(collapsed_rows)


def clean_external_trials(
    quality_df: pd.DataFrame,
    collapsed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    quality_df = quality_df.sort_values(["participant_name", "gmt_timestamp", "source_file"]).copy()
    quality_df["repeat_rank"] = quality_df.groupby("participant_name").cumcount() + 1
    quality_df["exclude_fast"] = quality_df["prop_fast_lt300_attempts"] > 0.10
    quality_df["exclude_repeat"] = quality_df["repeat_rank"] > 1
    quality_df["keep"] = ~(quality_df["exclude_fast"] | quality_df["exclude_repeat"])

    kept_files = set(quality_df.loc[quality_df["keep"], "source_file"])
    clean_trials = collapsed_df[collapsed_df["source_file"].isin(kept_files)].copy()

    kept_cases = quality_df.loc[quality_df["keep"], ["source_file", "participant_name", "gmt_timestamp"]].copy()
    kept_cases = kept_cases.sort_values(["gmt_timestamp", "source_file"]).reset_index(drop=True)
    kept_cases["participant_id"] = [f"lsiat_{i:03d}" for i in range(1, len(kept_cases) + 1)]

    id_map = kept_cases[["source_file", "participant_id"]]
    clean_trials = clean_trials.merge(id_map, on="source_file", how="left")
    clean_trials["block"] = clean_trials["phase"].map(EXTERNAL_PHASE_MAP)
    clean_trials["trial_in_block"] = clean_trials["base_trial"].astype(int)
    clean_trials["rt"] = clean_trials["rt_total"].astype(float)
    clean_trials = clean_trials.drop(columns=["participant_name"]).sort_values(
        ["participant_id", "block", "trial_in_block"]
    )

    participant_summary = (
        clean_trials.groupby("participant_id")
        .agg(
            source_file=("source_file", "first"),
            total_items=("rt", "size"),
            total_attempts=("n_attempts", "sum"),
            total_errors=("n_errors", "sum"),
            mean_rt=("rt", "mean"),
            median_rt=("rt", "median"),
        )
        .reset_index()
    )
    participant_summary["error_rate"] = participant_summary["total_errors"] / participant_summary["total_attempts"]

    critical = clean_trials[clean_trials["phase"].isin(EXTERNAL_CRITICAL_PHASES)].copy()
    rt_by_phase = (
        critical.groupby(["participant_id", "phase"])["rt"]
        .mean()
        .unstack()
        .rename(columns={"fase3": "rt_congruent", "fase5": "rt_incongruent"})
    )
    rt_sd = critical.groupby("participant_id")["rt"].std()
    d_like = pd.DataFrame(index=rt_by_phase.index)
    d_like["rt_congruent"] = rt_by_phase["rt_congruent"]
    d_like["rt_incongruent"] = rt_by_phase["rt_incongruent"]
    d_like["sd_critical"] = rt_sd
    d_like["d_like"] = (d_like["rt_incongruent"] - d_like["rt_congruent"]) / d_like["sd_critical"]
    participant_summary = participant_summary.merge(d_like.reset_index(), on="participant_id", how="left")

    summary = {
        "raw_files": int(len(quality_df)),
        "valid_participants": int(quality_df["keep"].sum()),
        "excluded_fast": int(quality_df["exclude_fast"].sum()),
        "excluded_repeat": int(quality_df["exclude_repeat"].sum()),
    }
    return quality_df, participant_summary, clean_trials, summary


def build_curves_from_trials(
    df: pd.DataFrame,
    pid_col: str = "participant_id",
    block_col: str = "block",
    trial_col: str = "trial_in_block",
    rt_col: str = "rt",
    n_bins: int = DEFAULT_BINS,
    standardize: bool = True,
) -> list[dict[str, object]]:
    curves: list[dict[str, object]] = []
    for pid, group in df.groupby(pid_col, sort=False):
        g = group.copy()
        g["pos_norm"] = g.groupby(block_col)[trial_col].transform(
            lambda s: (s - s.min()) / max(1.0, float(s.max() - s.min()))
        )
        values = g[["pos_norm", rt_col]].to_numpy(dtype=float)
        if len(values) < n_bins:
            continue
        q = np.quantile(values[:, 0], np.linspace(0, 1, n_bins + 1))
        x_bin: list[float] = []
        y_bin: list[float] = []
        for i in range(n_bins):
            lo, hi = q[i], q[i + 1]
            mask = (values[:, 0] >= lo) & (values[:, 0] <= hi)
            chunk = values[mask]
            if chunk.size == 0:
                continue
            x_bin.append(float(np.mean(chunk[:, 0])))
            y_bin.append(float(np.mean(chunk[:, 1])))
        if len(x_bin) < 3:
            continue
        y_arr = np.asarray(y_bin, dtype=float)
        if standardize:
            y_sd = float(np.std(y_arr, ddof=1))
            if not np.isfinite(y_sd) or y_sd < 1e-9:
                y_sd = 1.0
            y_arr = (y_arr - float(np.mean(y_arr))) / y_sd
        curves.append(
            {
                "pid": pid,
                "x": np.asarray(x_bin, dtype=float),
                "y": y_arr,
            }
        )
    return curves


def load_gender_science_trials() -> pd.DataFrame:
    paths = sorted(glob.glob(str(DATA_DIR / "GenderScience_iat_2019" / "iat_2019" / "iat*.txt")))
    if not paths:
        raise FileNotFoundError("Could not find GenderScience IAT text files.")
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df[["block_number", "trial_number", "trial_latency", "session_id"]])
    trial_df = pd.concat(dfs, ignore_index=True)
    trial_df = trial_df.rename(
        columns={
            "block_number": "block",
            "trial_number": "trial_in_block",
            "trial_latency": "rt",
            "session_id": "pid",
        }
    )
    trial_df = trial_df.dropna(subset=["pid", "block", "trial_in_block", "rt"]).copy()
    trial_df["block"] = trial_df["block"].astype(int)
    trial_df["trial_in_block"] = trial_df["trial_in_block"].astype(int)
    trial_df["rt"] = pd.to_numeric(trial_df["rt"], errors="coerce")
    return trial_df.dropna(subset=["rt"])


def load_gender_science_curves(n_bins: int = DEFAULT_BINS) -> list[dict[str, object]]:
    if RAW_CURVES_CACHE.exists():
        with open(RAW_CURVES_CACHE, "rb") as handle:
            curves = pickle.load(handle)
        if curves and {"pid", "x", "y"}.issubset(curves[0].keys()):
            return curves
    df = load_gender_science_trials()
    df = df[df["block"].isin([3, 4, 6, 7])].copy()
    return build_curves_from_trials(
        df,
        pid_col="pid",
        block_col="block",
        trial_col="trial_in_block",
        rt_col="rt",
        standardize=False,
    )


def gender_science_dlike_quintiles(curves: list[dict[str, object]]) -> pd.DataFrame:
    trial_df = load_gender_science_trials()
    trial_df = trial_df[trial_df["block"].isin([3, 4, 6, 7])].copy()

    means = trial_df.groupby(["pid", "block"])["rt"].mean().unstack()
    all_sd = trial_df.groupby("pid")["rt"].std()
    d_like = pd.DataFrame(index=means.index)
    d_like["rt_congruent"] = means[[3, 4]].mean(axis=1)
    d_like["rt_incongruent"] = means[[6, 7]].mean(axis=1)
    d_like["sd_all"] = all_sd
    d_like["d_like"] = (d_like["rt_incongruent"] - d_like["rt_congruent"]) / d_like["sd_all"]
    d_like = d_like.replace([np.inf, -np.inf], np.nan).dropna(subset=["d_like"])

    curve_df = pd.DataFrame({"pid": [curve["pid"] for curve in curves], "curve": curves})
    merged = curve_df.merge(d_like[["d_like"]], left_on="pid", right_index=True, how="inner")

    quintiles = merged["d_like"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()

    def label_quintile(value: float) -> str:
        if value <= quintiles[0]:
            return "Q1"
        if value <= quintiles[1]:
            return "Q2"
        if value <= quintiles[2]:
            return "Q3"
        if value <= quintiles[3]:
            return "Q4"
        return "Q5"

    merged["quintile"] = merged["d_like"].map(label_quintile)

    rows: list[dict[str, object]] = []
    for quintile, group in merged.groupby("quintile", sort=True):
        subgroup_curves = [
            {"pid": curve["pid"], "x": curve["x"], "y": (curve["y"] - np.mean(curve["y"])) / max(np.std(curve["y"], ddof=1), 1e-9)}
            for curve in group["curve"].tolist()
        ]
        x, y, mask = arrays_from_curves(subgroup_curves)
        posterior = profile_theta_posterior(x, y, mask, THETA_COARSE_GRID)
        summary = posterior_summary(THETA_COARSE_GRID, posterior)
        rows.append(
            {
                "quintile": quintile,
                "n": int(len(group)),
                "d_like_mean": float(group["d_like"].mean()),
                "d_like_min": float(group["d_like"].min()),
                "d_like_max": float(group["d_like"].max()),
                "theta_mean": summary["theta_mean"],
                "theta_sd": summary["theta_sd"],
                "theta_map": summary["theta_map"],
                "hdi_low": summary["hdi_low"],
                "hdi_high": summary["hdi_high"],
                "posterior": posterior,
            }
        )
    return pd.DataFrame(rows)


def external_theta_summary(clean_trials: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    critical = clean_trials[clean_trials["phase"].isin(EXTERNAL_CRITICAL_PHASES)].copy()
    curves = build_curves_from_trials(critical, standardize=True)
    x, y, mask = arrays_from_curves(curves)
    posterior = profile_theta_posterior(x, y, mask, THETA_FINE_GRID)
    summary = posterior_summary(THETA_FINE_GRID, posterior)
    summary_df = pd.DataFrame(
        [
            {
                "task": "Life-satisfaction IAT",
                "n": int(len(curves)),
                **summary,
            }
        ]
    )
    return summary_df, posterior


def external_model_comparison(clean_trials: pd.DataFrame) -> pd.DataFrame:
    critical = clean_trials[clean_trials["phase"].isin(EXTERNAL_CRITICAL_PHASES)].copy()
    curves = build_curves_from_trials(critical, standardize=False)
    x, y_raw, mask = arrays_from_curves(curves)

    specs = [
        ("Interference", "interference", THETA_SEARCH_GRID),
        ("Cosine only", "cos_only", THETA_SEARCH_GRID),
        ("Sine only", "sin_only", THETA_SEARCH_GRID),
        ("Polynomial (2nd order)", "poly2", None),
        ("Exponential", "exp", K_GRID),
        ("Power law", "power", P_GRID),
    ]

    rows = []
    for label, model_name, grid in specs:
        best_param, rmse = evaluate_model_rmse(x, y_raw, mask, model_name, grid)
        boundary_hit = bool(
            grid is not None
            and best_param is not None
            and (np.isclose(best_param, float(np.min(grid))) or np.isclose(best_param, float(np.max(grid))))
        )
        rows.append(
            {
                "model": label,
                "best_param": best_param,
                "rmse_test": rmse,
                "boundary_hit": boundary_hit,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_test").reset_index(drop=True)


def load_gender_science_full_summary() -> dict[str, float]:
    if not GS_POSTERIOR_CSV.exists():
        return {
            "task": "Gender-Science IAT",
            "n": np.nan,
            "theta_mean": np.nan,
            "theta_sd": np.nan,
            "theta_map": np.nan,
            "hdi_low": np.nan,
            "hdi_high": np.nan,
        }
    posterior_df = pd.read_csv(GS_POSTERIOR_CSV)
    theta_grid = posterior_df["theta_deg"].to_numpy(dtype=float)
    posterior = posterior_df["posterior"].to_numpy(dtype=float)
    summary = posterior_summary(theta_grid, posterior)
    return {
        "task": "Gender-Science IAT",
        "n": 141329,
        **summary,
    }


def plot_external_leverage(gs_quintiles: pd.DataFrame, gs_full: dict[str, float], external_posterior: np.ndarray, external_models: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    colors = ["#295c85", "#5b8e7d", "#c08a00", "#c95d3a", "#7b4d8d"]
    for color, (_, row) in zip(colors, gs_quintiles.sort_values("quintile").iterrows()):
        posterior = np.asarray(row["posterior"], dtype=float)
        axes[0].plot(
            THETA_COARSE_GRID,
            posterior / posterior.max(),
            lw=2,
            color=color,
            label=f"{row['quintile']} (mean d={row['d_like_mean']:.2f})",
        )
    axes[0].set_xlabel(r"$\theta$ (degrees)")
    axes[0].set_ylabel("Normalized posterior")
    axes[0].set_xlim(0, 40)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    gs_post_df = pd.read_csv(GS_POSTERIOR_CSV)
    gs_theta = gs_post_df["theta_deg"].to_numpy(dtype=float)
    gs_post = gs_post_df["posterior"].to_numpy(dtype=float)
    axes[1].plot(gs_theta, gs_post / gs_post.max(), lw=2.2, color="#1f77b4", label="Gender-Science")
    axes[1].plot(
        THETA_FINE_GRID,
        external_posterior / external_posterior.max(),
        lw=2.2,
        color="#d55e00",
        label="Life-satisfaction IAT",
    )
    axes[1].axvline(gs_full["theta_map"], color="#1f77b4", ls="--", lw=1)
    axes[1].axvline(THETA_FINE_GRID[np.argmax(external_posterior)], color="#d55e00", ls="--", lw=1)
    axes[1].set_xlabel(r"$\theta$ (degrees)")
    axes[1].set_ylabel("Normalized posterior")
    axes[1].set_xlim(0, 180)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    plot_df = external_models.copy()
    color_map = {
        "Interference": "#1f77b4",
        "Cosine only": "#3a7ca5",
        "Sine only": "#5fa8d3",
        "Polynomial (2nd order)": "#8c8c8c",
        "Exponential": "#a65e2e",
        "Power law": "#c98544",
    }
    axes[2].bar(
        plot_df["model"],
        plot_df["rmse_test"],
        color=[color_map.get(model, "#999999") for model in plot_df["model"]],
    )
    axes[2].set_ylabel("RMSE (z-scored hold-out)")
    axes[2].tick_params(axis="x", labelrotation=35)
    axes[2].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "external_leverage_summary.png", dpi=300)
    plt.close(fig)


def write_tables(
    gs_full: dict[str, float],
    gs_quintiles: pd.DataFrame,
    external_theta: pd.DataFrame,
    external_models: pd.DataFrame,
) -> None:
    task_df = pd.DataFrame(
        [
            {
                "Task": "Gender-Science IAT",
                "N": int(gs_full["n"]),
                "Theta mean": gs_full["theta_mean"],
                "Theta SD": gs_full["theta_sd"],
                "Theta MAP": gs_full["theta_map"],
                "HDI low": gs_full["hdi_low"],
                "HDI high": gs_full["hdi_high"],
            },
            {
                "Task": "Life-satisfaction IAT",
                "N": int(external_theta.loc[0, "n"]),
                "Theta mean": external_theta.loc[0, "theta_mean"],
                "Theta SD": external_theta.loc[0, "theta_sd"],
                "Theta MAP": external_theta.loc[0, "theta_map"],
                "HDI low": external_theta.loc[0, "hdi_low"],
                "HDI high": external_theta.loc[0, "hdi_high"],
            },
        ]
    )
    with open(TAB_DIR / "Table_external_task_theta.tex", "w", encoding="utf-8") as handle:
        handle.write(task_df.to_latex(index=False, float_format="%.2f"))

    quintile_df = gs_quintiles[
        ["quintile", "n", "d_like_mean", "theta_mean", "theta_sd", "theta_map", "hdi_low", "hdi_high"]
    ].copy()
    quintile_df.columns = ["Quintile", "N", "Mean d-like", "Theta mean", "Theta SD", "Theta MAP", "HDI low", "HDI high"]
    with open(TAB_DIR / "Table_theta_dlike_quintiles.tex", "w", encoding="utf-8") as handle:
        handle.write(quintile_df.to_latex(index=False, float_format="%.3f"))

    model_df = external_models.copy()
    model_df["best_param"] = model_df["best_param"].map(lambda v: "---" if pd.isna(v) else f"{v:.2f}")
    model_df["boundary_hit"] = model_df["boundary_hit"].map(lambda flag: "Yes" if flag else "No")
    model_df.columns = ["Model", "Best parameter", "RMSE$_{test}$", "Boundary hit"]
    with open(TAB_DIR / "Table_external_model_comparison.tex", "w", encoding="utf-8") as handle:
        handle.write(model_df.to_latex(index=False, escape=False))


def main() -> None:
    ensure_dirs()
    if not EXTERNAL_ZIP.exists():
        raise FileNotFoundError(f"Missing external archive: {EXTERNAL_ZIP}")

    quality_df, collapsed_df = load_external_archive(EXTERNAL_ZIP)
    quality_df, participant_summary, clean_trials, external_summary = clean_external_trials(quality_df, collapsed_df)
    participant_summary.to_csv(OUT_DIR / "external_life_satisfaction_participants_clean.csv", index=False)
    clean_trials.to_csv(OUT_DIR / "external_life_satisfaction_trials_clean.csv", index=False)

    excluded = quality_df.loc[~quality_df["keep"]].copy()
    excluded["reason"] = np.where(
        excluded["repeat_rank"] > 1,
        "repeat_later_attempt",
        np.where(excluded["prop_fast_lt300_attempts"] > 0.10, "fast_response_exclusion", "other"),
    )
    excluded.drop(columns=["participant_name"], errors="ignore").to_csv(
        OUT_DIR / "external_life_satisfaction_excluded_cases.csv", index=False
    )

    external_theta_df, external_posterior = external_theta_summary(clean_trials)
    external_theta_df.to_csv(OUT_DIR / "external_life_satisfaction_theta_summary.csv", index=False)
    pd.DataFrame({"theta_deg": THETA_FINE_GRID, "posterior": external_posterior}).to_csv(
        OUT_DIR / "external_life_satisfaction_theta_profile.csv", index=False
    )

    external_models = external_model_comparison(clean_trials)
    external_models.to_csv(OUT_DIR / "external_life_satisfaction_model_comparison.csv", index=False)

    gs_curves = load_gender_science_curves()
    gs_quintiles = gender_science_dlike_quintiles(gs_curves)
    gs_quintiles_no_post = gs_quintiles.drop(columns=["posterior"]).copy()
    gs_quintiles_no_post.to_csv(OUT_DIR / "gender_science_theta_by_dlike_quintile.csv", index=False)

    gs_full = load_gender_science_full_summary()
    write_tables(gs_full, gs_quintiles, external_theta_df, external_models)
    plot_external_leverage(gs_quintiles, gs_full, external_posterior, external_models)

    summary = {
        "external_cleaning": external_summary,
        "external_theta": {
            "task": str(external_theta_df.loc[0, "task"]),
            "n": int(external_theta_df.loc[0, "n"]),
            "theta_mean": float(external_theta_df.loc[0, "theta_mean"]),
            "theta_sd": float(external_theta_df.loc[0, "theta_sd"]),
            "theta_map": float(external_theta_df.loc[0, "theta_map"]),
            "hdi_low": float(external_theta_df.loc[0, "hdi_low"]),
            "hdi_high": float(external_theta_df.loc[0, "hdi_high"]),
        },
        "external_best_rmse_model": {
            "model": str(external_models.loc[0, "model"]),
            "best_param": None if pd.isna(external_models.loc[0, "best_param"]) else float(external_models.loc[0, "best_param"]),
            "rmse_test": float(external_models.loc[0, "rmse_test"]),
            "boundary_hit": bool(external_models.loc[0, "boundary_hit"]),
        },
        "gender_science_dlike_theta_range": {
            "theta_map_min": float(gs_quintiles_no_post["theta_map"].min()),
            "theta_map_max": float(gs_quintiles_no_post["theta_map"].max()),
            "d_like_mean_min": float(gs_quintiles_no_post["d_like_mean"].min()),
            "d_like_mean_max": float(gs_quintiles_no_post["d_like_mean"].max()),
        },
    }
    with open(OUT_DIR / "external_leverage_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
