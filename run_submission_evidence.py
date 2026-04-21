import glob
import json
import os
import pickle
import time
from pathlib import Path

try:
    import arviz as az
except ImportError:  # Optional for utilities that do not touch WAIC.
    az = None
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

CRITICAL_BLOCKS = [3, 4, 6, 7]
PRIMARY_BINS = 6
BIN_OPTIONS = [4, 6, 8]
HOLDOUT_FRAC = 0.20
HOLDOUT_RIDGE = 1e-5
PROFILE_RIDGE = 1e-8
HDI_PROB = 0.94
RANDOM_SEED = 123

THETA_POSTERIOR_GRID = np.arange(0.0, 180.0 + 0.25, 0.25)
THETA_COARSE_GRID = np.arange(0.0, 180.0 + 1.0, 1.0)
THETA_REFINE_RADIUS = 2.0
THETA_REFINE_STEP = 0.1

K_COARSE_GRID = np.arange(0.0, 12.0 + 0.25, 0.25)
K_REFINE_RADIUS = 0.5
K_REFINE_STEP = 0.05

P_COARSE_GRID = np.arange(0.05, 4.0 + 0.1, 0.1)
P_REFINE_RADIUS = 0.25
P_REFINE_STEP = 0.02

PERMUTE_REPS = 8
BLOCK_SHUFFLE_REPS = 20

RECOVERY_THETAS = [10.0, 17.25, 30.0, 60.0]
RECOVERY_REPS = 3
SIM_N_CURVES = 5000
NULL_REPS = 3
SIM_THETA_GRID = np.arange(0.0, 90.0 + 0.5, 0.5)

WAIC_DRAW_COUNT = 25

rng = np.random.default_rng(RANDOM_SEED)


def ensure_dirs():
    for path in [OUT_DIR, FIG_DIR, TAB_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_gender_science_df():
    gs_dir = DATA_DIR / "GenderScience_iat_2019" / "iat_2019"
    paths = sorted(glob.glob(str(gs_dir / "iat*.txt")))
    if not paths:
        raise FileNotFoundError(f"No iat*.txt files found in {gs_dir}")
    dfs = [pd.read_csv(path, sep="\t", low_memory=False) for path in paths]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "session_id": "pid",
        "block_number": "block",
        "trial_number": "trial_in_block",
        "trial_latency": "rt",
    })
    keep = ["pid", "block", "trial_in_block", "rt"]
    df = df[keep].dropna(subset=["pid", "block", "trial_in_block", "rt"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["block"] = df["block"].astype(int)
    return df[df["block"].isin(CRITICAL_BLOCKS)].copy()


def _bin_curve(values, n_bins):
    if len(values) < n_bins:
        return None
    q = np.quantile(values[:, 0], np.linspace(0, 1, n_bins + 1))
    x_bin, y_bin, n_bin, sum_bin, sumsq_bin = [], [], [], [], []
    for i in range(n_bins):
        lo, hi = q[i], q[i + 1]
        mask = (values[:, 0] >= lo) & (values[:, 0] <= hi)
        chunk = values[mask]
        if chunk.size == 0:
            continue
        x_bin.append(float(np.mean(chunk[:, 0])))
        y_bin.append(float(np.mean(chunk[:, 1])))
        n_bin.append(int(chunk.shape[0]))
        sum_bin.append(float(np.sum(chunk[:, 1])))
        sumsq_bin.append(float(np.sum(chunk[:, 1] ** 2)))
    if len(x_bin) < 3:
        return None
    return {
        "x": np.asarray(x_bin, dtype=float),
        "y": np.asarray(y_bin, dtype=float),
        "n": np.asarray(n_bin, dtype=int),
        "sum_rt": np.asarray(sum_bin, dtype=float),
        "sumsq_rt": np.asarray(sumsq_bin, dtype=float),
    }


def build_participant_curves_raw(df, n_bins):
    out = []
    for pid, g in df.groupby("pid", sort=False):
        g = g.copy()
        g["pos_norm"] = g.groupby("block")["trial_in_block"].transform(
            lambda s: (s - s.min()) / max(1.0, float(s.max() - s.min()))
        )
        values = g[["pos_norm", "rt"]].to_numpy(dtype=float)
        binned = _bin_curve(values, n_bins=n_bins)
        if binned is None:
            continue
        out.append({
            "pid": pid,
            "x": binned["x"],
            "y": binned["y"],
            "n": binned["n"],
            "sum_rt": binned["sum_rt"],
            "sumsq_rt": binned["sumsq_rt"],
        })
    return out


def build_participant_block_curves_raw(df, n_bins):
    out = []
    for (pid, block), g in df.groupby(["pid", "block"], sort=False):
        g = g.copy()
        g["pos_norm"] = (g["trial_in_block"] - g["trial_in_block"].min()) / max(
            1.0, float(g["trial_in_block"].max() - g["trial_in_block"].min())
        )
        values = g[["pos_norm", "rt"]].to_numpy(dtype=float)
        binned = _bin_curve(values, n_bins=n_bins)
        if binned is None:
            continue
        out.append({
            "pid": pid,
            "block": int(block),
            "curve_id": f"{pid}_{int(block)}",
            "x": binned["x"],
            "y": binned["y"],
            "n": binned["n"],
            "sum_rt": binned["sum_rt"],
            "sumsq_rt": binned["sumsq_rt"],
        })
    return out


def load_or_build_curves(df, n_bins, blockwise=False):
    suffix = f"raw_block_curves_bins{n_bins}_v2.pkl" if blockwise else f"raw_curves_bins{n_bins}_v2.pkl"
    cache_path = OUT_DIR / suffix
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            curves = pickle.load(handle)
        if curves and {"n", "sum_rt", "sumsq_rt"}.issubset(curves[0].keys()):
            return curves
    curves = build_participant_block_curves_raw(df, n_bins) if blockwise else build_participant_curves_raw(df, n_bins)
    with open(cache_path, "wb") as handle:
        pickle.dump(curves, handle)
    return curves


def arrays_from_curves(curves, include_block=False):
    max_t = max(len(c["x"]) for c in curves)
    n = len(curves)
    x = np.full((n, max_t), np.nan, dtype=float)
    y = np.full((n, max_t), np.nan, dtype=float)
    mask = np.zeros((n, max_t), dtype=bool)
    ids = []
    blocks = []
    for i, curve in enumerate(curves):
        t = len(curve["x"])
        x[i, :t] = curve["x"]
        y[i, :t] = curve["y"]
        mask[i, :t] = True
        ids.append(curve.get("curve_id", curve["pid"]))
        if include_block:
            blocks.append(curve["block"])
    if include_block:
        return x, y, mask, ids, np.asarray(blocks, dtype=int)
    return x, y, mask, ids


def stats_from_curves(curves):
    max_t = max(len(c["x"]) for c in curves)
    n_curves = len(curves)
    counts = np.zeros((n_curves, max_t), dtype=float)
    sums = np.zeros((n_curves, max_t), dtype=float)
    sumsqs = np.zeros((n_curves, max_t), dtype=float)
    for i, curve in enumerate(curves):
        t = len(curve["x"])
        counts[i, :t] = curve["n"]
        sums[i, :t] = curve["sum_rt"]
        sumsqs[i, :t] = curve["sumsq_rt"]
    return counts, sums, sumsqs


def row_standardize(y, mask):
    yz = np.full_like(y, np.nan, dtype=float)
    for i in range(y.shape[0]):
        idx = np.where(mask[i])[0]
        if idx.size == 0:
            continue
        vals = y[i, idx]
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        yz[i, idx] = (vals - mu) / sd
    return yz


def leak_free_standardize(y_raw, train_mask, counts=None, sums=None, sumsqs=None):
    yz = np.full_like(y_raw, np.nan, dtype=float)
    for i in range(y_raw.shape[0]):
        idx_train = np.where(train_mask[i])[0]
        idx_all = np.where(~np.isnan(y_raw[i]))[0]
        if idx_train.size == 0 or idx_all.size == 0:
            continue
        if counts is not None and sums is not None and sumsqs is not None:
            n_total = float(np.sum(counts[i, idx_train]))
            if n_total <= 1:
                mu = float(np.mean(y_raw[i, idx_train]))
                sd = float(np.std(y_raw[i, idx_train], ddof=1))
            else:
                sum_total = float(np.sum(sums[i, idx_train]))
                sumsq_total = float(np.sum(sumsqs[i, idx_train]))
                mu = sum_total / n_total
                var_num = sumsq_total - n_total * (mu ** 2)
                sd = float(np.sqrt(max(var_num / (n_total - 1.0), 0.0)))
        else:
            vals_train = y_raw[i, idx_train]
            mu = float(np.mean(vals_train))
            sd = float(np.std(vals_train, ddof=1))
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        yz[i, idx_all] = (y_raw[i, idx_all] - mu) / sd
    return yz


def temporal_holdout_mask(x, mask, frac, min_train_pts):
    n, _ = x.shape
    train_mask = np.zeros_like(mask, dtype=bool)
    test_mask = np.zeros_like(mask, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        idx = np.where(mask[i])[0]
        n_obs = idx.size
        if n_obs < min_train_pts + 1:
            continue
        order = idx[np.argsort(x[i, idx])]
        k = max(1, int(np.ceil(frac * n_obs)))
        if n_obs - k < min_train_pts:
            k = n_obs - min_train_pts
        test_idx = order[-k:]
        train_idx = order[:-k]
        if train_idx.size < min_train_pts or test_idx.size == 0:
            continue
        train_mask[i, train_idx] = True
        test_mask[i, test_idx] = True
        valid[i] = True
    return train_mask, test_mask, valid


def hdi_from_discrete(grid_vals, probs, mass=HDI_PROB):
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


def posterior_summary(grid_vals, posterior):
    mean = float(np.sum(grid_vals * posterior))
    sd = float(np.sqrt(np.sum(((grid_vals - mean) ** 2) * posterior)))
    lo, hi = hdi_from_discrete(grid_vals, posterior, mass=HDI_PROB)
    map_val = float(grid_vals[np.argmax(posterior)])
    return {
        "mean": mean,
        "sd": sd,
        "hdi_low": lo,
        "hdi_high": hi,
        "map": map_val,
    }


def design_matrix_from_param(x, model_name, param=None):
    if model_name == "interference":
        theta = np.deg2rad(param)
        return [
            np.ones_like(x),
            np.cos(theta * x),
            np.sin(theta * x),
        ]
    if model_name == "cos_only":
        theta = np.deg2rad(param)
        return [
            np.ones_like(x),
            np.cos(theta * x),
        ]
    if model_name == "sin_only":
        theta = np.deg2rad(param)
        return [
            np.ones_like(x),
            np.sin(theta * x),
        ]
    if model_name == "poly2":
        return [
            np.ones_like(x),
            x,
            x ** 2,
        ]
    if model_name == "exp":
        return [
            np.ones_like(x),
            np.exp(-param * x),
        ]
    if model_name == "power":
        xp = np.where(x > 0.0, x ** param, 0.0)
        return [
            np.ones_like(x),
            xp,
        ]
    raise ValueError(f"Unknown model_name={model_name}")


def compute_rss_and_beta(designs, y, fit_mask, ridge=HOLDOUT_RIDGE):
    p = len(designs)
    rows = fit_mask.any(axis=1)
    if not rows.any():
        return np.nan, np.empty((0, p)), rows

    yv = y[rows]
    mv = fit_mask[rows]
    r = [np.where(mv, d[rows], 0.0) for d in designs]

    a = np.empty((rows.sum(), p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            a[:, i, j] = np.sum(r[i] * r[j], axis=1)
    b = np.empty((rows.sum(), p), dtype=float)
    for i in range(p):
        b[:, i] = np.sum(yv * r[i], axis=1)

    a[..., range(p), range(p)] += ridge
    beta = np.linalg.solve(a, b[..., None])[..., 0]
    proj = (b[..., None, :] @ np.linalg.solve(a, b[..., :, None])).squeeze(-1).squeeze(-1)
    rss = float(np.sum((yv[mv]) ** 2) - np.sum(proj))
    return rss, beta, rows


def residual_sigma(designs, beta, y, fit_mask):
    mu = np.zeros_like(y)
    for j, d in enumerate(designs):
        mu += beta[:, j][:, None] * d
    resid = (y - mu)[fit_mask]
    return float(np.sqrt(np.mean(resid ** 2)))


def evaluate_on_mask(designs, beta, y, eval_mask, sigma):
    mu = np.zeros_like(y)
    for j, d in enumerate(designs):
        mu += beta[:, j][:, None] * d
    resid = (y - mu)[eval_mask]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sigma = max(float(sigma), 1e-9)
    logscore = float(np.mean(-0.5 * (np.log(2.0 * np.pi * sigma ** 2) + (resid ** 2) / sigma ** 2)))
    return rmse, logscore


def search_best_param(x, y_raw, mask, model_name, coarse_grid, refine_radius, refine_step, counts=None, sums=None, sumsqs=None):
    p = len(design_matrix_from_param(x, model_name, coarse_grid[0]))
    train_mask, test_mask, valid = temporal_holdout_mask(x, mask, HOLDOUT_FRAC, min_train_pts=p)
    xv = x[valid]
    yv_raw = y_raw[valid]
    train_v = train_mask[valid]
    test_v = test_mask[valid]
    stats_kwargs = {}
    if counts is not None and sums is not None and sumsqs is not None:
        stats_kwargs = {
            "counts": counts[valid],
            "sums": sums[valid],
            "sumsqs": sumsqs[valid],
        }
    yv = leak_free_standardize(yv_raw, train_v, **stats_kwargs)

    curve = []
    best_param = None
    best_rss = np.inf
    best_beta = None
    best_designs = None

    for param in coarse_grid:
        designs = design_matrix_from_param(xv, model_name, param)
        rss, beta, _ = compute_rss_and_beta(designs, yv, train_v, ridge=HOLDOUT_RIDGE)
        curve.append((float(param), float(rss)))
        if np.isfinite(rss) and rss < best_rss:
            best_rss = rss
            best_param = float(param)
            best_beta = beta
            best_designs = designs

    if best_param is None:
        raise RuntimeError(f"No valid parameter found for {model_name}")

    grid_min = float(np.min(coarse_grid))
    grid_max = float(np.max(coarse_grid))
    lo = max(grid_min, best_param - refine_radius)
    hi = min(grid_max, best_param + refine_radius)
    refine_grid = np.arange(lo, hi + 0.5 * refine_step, refine_step)
    for param in refine_grid:
        designs = design_matrix_from_param(xv, model_name, param)
        rss, beta, _ = compute_rss_and_beta(designs, yv, train_v, ridge=HOLDOUT_RIDGE)
        curve.append((float(param), float(rss)))
        if np.isfinite(rss) and rss < best_rss:
            best_rss = rss
            best_param = float(param)
            best_beta = beta
            best_designs = designs

    sigma = residual_sigma(best_designs, best_beta, yv, train_v)
    rmse, logscore = evaluate_on_mask(best_designs, best_beta, yv, test_v, sigma=sigma)
    boundary = np.isclose(best_param, grid_min) or np.isclose(best_param, grid_max)
    return {
        "best_param": best_param,
        "train_rss": best_rss,
        "rmse_test": rmse,
        "mean_logscore_test": logscore,
        "sigma_train": sigma,
        "n_curves": int(xv.shape[0]),
        "n_test_obs": int(np.sum(test_v)),
        "boundary_hit": bool(boundary),
        "curve": sorted(curve),
    }


def fit_parameter_free_model(x, y_raw, mask, model_name, counts=None, sums=None, sumsqs=None):
    designs0 = design_matrix_from_param(x, model_name, None)
    p = len(designs0)
    train_mask, test_mask, valid = temporal_holdout_mask(x, mask, HOLDOUT_FRAC, min_train_pts=p)
    xv = x[valid]
    yv_raw = y_raw[valid]
    train_v = train_mask[valid]
    test_v = test_mask[valid]
    stats_kwargs = {}
    if counts is not None and sums is not None and sumsqs is not None:
        stats_kwargs = {
            "counts": counts[valid],
            "sums": sums[valid],
            "sumsqs": sumsqs[valid],
        }
    yv = leak_free_standardize(yv_raw, train_v, **stats_kwargs)
    designs = design_matrix_from_param(xv, model_name, None)
    rss, beta, _ = compute_rss_and_beta(designs, yv, train_v, ridge=HOLDOUT_RIDGE)
    sigma = residual_sigma(designs, beta, yv, train_v)
    rmse, logscore = evaluate_on_mask(designs, beta, yv, test_v, sigma=sigma)
    return {
        "best_param": np.nan,
        "train_rss": rss,
        "rmse_test": rmse,
        "mean_logscore_test": logscore,
        "sigma_train": sigma,
        "n_curves": int(xv.shape[0]),
        "n_test_obs": int(np.sum(test_v)),
        "boundary_hit": False,
        "curve": [],
    }


def profile_theta_posterior(x, y_std, mask, theta_grid_deg):
    n_obs = int(mask.sum())
    y2_sum = float(np.sum((y_std[mask]) ** 2))
    post_log = []
    rss_list = []
    for theta_deg in theta_grid_deg:
        arg = np.deg2rad(theta_deg) * x
        c = np.where(mask, np.cos(arg), 0.0)
        s = np.where(mask, np.sin(arg), 0.0)
        o = np.where(mask, 1.0, 0.0)
        ym = np.where(mask, y_std, 0.0)

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
        a[..., range(3), range(3)] += PROFILE_RIDGE
        proj = (b[..., None, :] @ np.linalg.solve(a, b[..., :, None])).squeeze(-1).squeeze(-1)
        rss_theta = y2_sum - float(np.sum(proj))
        rss_list.append(rss_theta)
        post_log.append(-(n_obs / 2.0) * np.log(max(rss_theta, 1e-12)))

    post_log = np.asarray(post_log, dtype=float)
    post_log -= np.max(post_log)
    posterior = np.exp(post_log)
    posterior /= np.sum(posterior)
    return posterior, np.asarray(rss_list, dtype=float)


def profile_theta_summary_from_arrays(x, y_std, mask, theta_grid):
    posterior, rss = profile_theta_posterior(x, y_std, mask, theta_grid)
    return posterior_summary(theta_grid, posterior), posterior, rss


def fixed_theta_coefficients(x, y_std, mask, theta_deg):
    arg = np.deg2rad(theta_deg) * x
    designs = [
        np.ones_like(x),
        np.cos(arg),
        np.sin(arg),
    ]
    _, beta, _ = compute_rss_and_beta(designs, y_std, mask, ridge=PROFILE_RIDGE)
    return beta


def circular_mean_deg(angles_deg):
    ang = np.deg2rad(np.asarray(angles_deg, dtype=float))
    c = np.mean(np.cos(ang))
    s = np.mean(np.sin(ang))
    return float(np.rad2deg(np.arctan2(s, c)))


def circular_variance(angles_deg):
    ang = np.deg2rad(np.asarray(angles_deg, dtype=float))
    r = float(np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang))))
    return float(1.0 - r)


def profile_theta_summary_from_curves(curves, theta_grid):
    x, y_raw, mask, _ = arrays_from_curves(curves)
    y_std = row_standardize(y_raw, mask)
    posterior, rss = profile_theta_posterior(x, y_std, mask, theta_grid)
    return posterior_summary(theta_grid, posterior), posterior, rss


def block_theta_summary(block_curves):
    rows = []
    for block in CRITICAL_BLOCKS:
        subset = [c for c in block_curves if c["block"] == block]
        summary, _, _ = profile_theta_summary_from_curves(subset, THETA_POSTERIOR_GRID)
        rows.append({
            "block": block,
            "n_curves": len(subset),
            "theta_mean": summary["mean"],
            "theta_sd": summary["sd"],
            "theta_hdi_low": summary["hdi_low"],
            "theta_hdi_high": summary["hdi_high"],
            "theta_map": summary["map"],
        })
    return pd.DataFrame(rows)


def block_phase_summary(block_curves, theta_map):
    x, y_raw, mask, _, blocks = arrays_from_curves(block_curves, include_block=True)
    y_std = row_standardize(y_raw, mask)
    beta = fixed_theta_coefficients(x, y_std, mask, theta_map)
    phase_deg = np.rad2deg(np.arctan2(beta[:, 2], beta[:, 1]))
    rows = []
    for block in sorted(np.unique(blocks)):
        vals = phase_deg[blocks == block]
        rows.append({
            "block": int(block),
            "n_curves": int(vals.size),
            "circular_mean_deg": circular_mean_deg(vals),
            "circular_variance": circular_variance(vals),
        })
    rows = pd.DataFrame(rows)
    mean_angles = rows["circular_mean_deg"].to_numpy()
    obs_dispersion = circular_variance(mean_angles)

    null_dispersion = []
    shuffled = blocks.copy()
    for _ in range(BLOCK_SHUFFLE_REPS):
        rng.shuffle(shuffled)
        means = []
        for block in sorted(np.unique(blocks)):
            vals = phase_deg[shuffled == block]
            means.append(circular_mean_deg(vals))
        null_dispersion.append(circular_variance(means))
    p_value = (1.0 + np.sum(np.asarray(null_dispersion) >= obs_dispersion)) / (len(null_dispersion) + 1.0)
    return rows, obs_dispersion, np.asarray(null_dispersion), p_value


def evaluate_block_specific_theta(block_curves):
    x, y_raw, mask, _, blocks = arrays_from_curves(block_curves, include_block=True)
    counts, sums, sumsqs = stats_from_curves(block_curves)
    rows = []
    sq_err = []
    log_terms = []

    for block in CRITICAL_BLOCKS:
        idx = np.where(blocks == block)[0]
        xb = x[idx]
        yb_raw = y_raw[idx]
        mb = mask[idx]
        best = search_best_param(
            xb,
            yb_raw,
            mb,
            "interference",
            THETA_COARSE_GRID,
            THETA_REFINE_RADIUS,
            THETA_REFINE_STEP,
            counts=counts[idx],
            sums=sums[idx],
            sumsqs=sumsqs[idx],
        )
        rows.append({
            "block": block,
            "theta_train_selected": best["best_param"],
            "rmse_test": best["rmse_test"],
            "mean_logscore_test": best["mean_logscore_test"],
            "boundary_hit": best["boundary_hit"],
        })

        p = 3
        train_mask, test_mask, valid = temporal_holdout_mask(xb, mb, HOLDOUT_FRAC, min_train_pts=p)
        xb = xb[valid]
        yb_raw = yb_raw[valid]
        train_mask = train_mask[valid]
        test_mask = test_mask[valid]
        yb = leak_free_standardize(
            yb_raw,
            train_mask,
            counts=counts[idx][valid],
            sums=sums[idx][valid],
            sumsqs=sumsqs[idx][valid],
        )
        designs = design_matrix_from_param(xb, "interference", best["best_param"])
        _, beta, _ = compute_rss_and_beta(designs, yb, train_mask, ridge=HOLDOUT_RIDGE)
        mu = np.zeros_like(yb)
        for j, d in enumerate(designs):
            mu += beta[:, j][:, None] * d
        resid = (yb - mu)[test_mask]
        sigma = max(best["sigma_train"], 1e-9)
        sq_err.append(resid ** 2)
        log_terms.append(-0.5 * (np.log(2.0 * np.pi * sigma ** 2) + (resid ** 2) / sigma ** 2))

    sq_err = np.concatenate(sq_err)
    log_terms = np.concatenate(log_terms)
    aggregate = {
        "model": "Interference (block-specific theta)",
        "best_param": json.dumps({row["block"]: row["theta_train_selected"] for row in rows}),
        "rmse_test": float(np.sqrt(np.mean(sq_err))),
        "mean_logscore_test": float(np.mean(log_terms)),
        "boundary_hit": any(row["boundary_hit"] for row in rows),
    }
    return aggregate, pd.DataFrame(rows)


def permutation_theta_null(x, y_std, mask):
    observed_summary, _, _ = profile_theta_summary_from_arrays(x, y_std, mask, THETA_COARSE_GRID)
    null_rows = []
    for rep in range(PERMUTE_REPS):
        y_perm = y_std.copy()
        for i in range(y_perm.shape[0]):
            idx = np.where(mask[i])[0]
            vals = y_perm[i, idx].copy()
            y_perm[i, idx] = vals[rng.permutation(len(vals))]
        summary, posterior, _ = profile_theta_summary_from_arrays(x, y_perm, mask, THETA_COARSE_GRID)
        null_rows.append({
            "rep": rep,
            "theta_map": summary["map"],
            "theta_sd": summary["sd"],
            "theta_hdi_width": summary["hdi_high"] - summary["hdi_low"],
            "posterior_peak": float(np.max(posterior)),
        })
    return observed_summary, pd.DataFrame(null_rows)


def fit_poly_coefficients(x, y_std, mask):
    designs = design_matrix_from_param(x, "poly2", None)
    _, beta, _ = compute_rss_and_beta(designs, y_std, mask)
    return beta


def simulate_interference_recovery(x_real, mask_real, beta_empirical, sigma_real):
    rows = []
    for theta_true in RECOVERY_THETAS:
        for rep in range(RECOVERY_REPS):
            idx = rng.choice(x_real.shape[0], size=SIM_N_CURVES, replace=True)
            xb = x_real[idx]
            mb = mask_real[idx]
            beta_idx = rng.choice(beta_empirical.shape[0], size=SIM_N_CURVES, replace=True)
            bb = beta_empirical[beta_idx]
            mu = bb[:, 0][:, None] + bb[:, 1][:, None] * np.cos(np.deg2rad(theta_true) * xb) + bb[:, 2][:, None] * np.sin(np.deg2rad(theta_true) * xb)
            y = mu + rng.normal(scale=sigma_real, size=mu.shape)
            y[~mb] = np.nan
            y_std = row_standardize(y, mb)
            summary, _, _ = profile_theta_summary_from_arrays(xb, y_std, mb, SIM_THETA_GRID)
            rows.append({
                "generator": "interference",
                "theta_true": theta_true,
                "rep": rep,
                "theta_map": summary["map"],
                "theta_mean": summary["mean"],
                "theta_sd": summary["sd"],
                "hdi_low": summary["hdi_low"],
                "hdi_high": summary["hdi_high"],
                "covered": summary["hdi_low"] <= theta_true <= summary["hdi_high"],
            })
    return pd.DataFrame(rows)


def simulate_additive_null(x_real, mask_real, poly_beta_empirical, sigma_real):
    rows = []
    for rep in range(NULL_REPS):
        idx = rng.choice(x_real.shape[0], size=SIM_N_CURVES, replace=True)
        xb = x_real[idx]
        mb = mask_real[idx]
        beta_idx = rng.choice(poly_beta_empirical.shape[0], size=SIM_N_CURVES, replace=True)
        bb = poly_beta_empirical[beta_idx]
        mu = bb[:, 0][:, None] + bb[:, 1][:, None] * xb + bb[:, 2][:, None] * (xb ** 2)
        y = mu + rng.normal(scale=sigma_real, size=mu.shape)
        y[~mb] = np.nan
        y_std = row_standardize(y, mb)
        summary, _, _ = profile_theta_summary_from_arrays(xb, y_std, mb, SIM_THETA_GRID)
        rows.append({
            "generator": "poly2_null",
            "rep": rep,
            "theta_map": summary["map"],
            "theta_mean": summary["mean"],
            "theta_sd": summary["sd"],
            "hdi_low": summary["hdi_low"],
            "hdi_high": summary["hdi_high"],
            "hdi_width": summary["hdi_high"] - summary["hdi_low"],
        })
    return pd.DataFrame(rows)


def draw_params_from_profile(grid, x, y_std, mask, model_name):
    if model_name == "poly2":
        draws = [np.nan] * WAIC_DRAW_COUNT
        return draws

    post_log = []
    for param in grid:
        designs = design_matrix_from_param(x, model_name, param)
        rss, _, _ = compute_rss_and_beta(designs, y_std, mask)
        post_log.append(-(int(mask.sum()) / 2.0) * np.log(max(rss, 1e-12)))
    post_log = np.asarray(post_log)
    post_log -= np.max(post_log)
    weights = np.exp(post_log)
    weights /= np.sum(weights)
    return rng.choice(grid, size=WAIC_DRAW_COUNT, replace=True, p=weights)


def approximate_waic_table(x, y_std, mask):
    if az is None:
        raise ImportError("arviz is required for approximate_waic_table")
    model_grids = {
        "interference": THETA_POSTERIOR_GRID,
        "exp": np.arange(0.0, 12.0 + 0.1, 0.1),
        "power": np.arange(0.05, 4.0 + 0.05, 0.05),
        "poly2": None,
    }
    obs_count = int(mask.sum())
    rows = []
    for model_name, grid in model_grids.items():
        param_draws = draw_params_from_profile(grid, x, y_std, mask, model_name) if grid is not None else [np.nan] * WAIC_DRAW_COUNT
        loglik = np.empty((1, WAIC_DRAW_COUNT, obs_count), dtype=np.float32)
        keep = mask.any(axis=1)
        xk = x[keep]
        yk = y_std[keep]
        mk = mask[keep]
        for draw_idx, param in enumerate(param_draws):
            designs = design_matrix_from_param(xk, model_name, None if np.isnan(param) else float(param))
            _, beta, _ = compute_rss_and_beta(designs, yk, mk)
            sigma = max(residual_sigma(designs, beta, yk, mk), 1e-9)
            mu = np.zeros_like(yk)
            for j, d in enumerate(designs):
                mu += beta[:, j][:, None] * d
            resid = (yk - mu)[mk]
            loglik[0, draw_idx, :] = -0.5 * (np.log(2.0 * np.pi * sigma ** 2) + (resid ** 2) / sigma ** 2)

        idata = az.from_dict(log_likelihood={"y": loglik})
        waic = az.waic(idata)
        rows.append({
            "model": model_name,
            "elpd_waic": float(waic.elpd_waic),
            "p_waic": float(waic.p_waic),
            "se": float(waic.se),
        })
    df = pd.DataFrame(rows).sort_values("elpd_waic", ascending=False).reset_index(drop=True)
    df["delta_elpd_waic"] = df["elpd_waic"] - df["elpd_waic"].iloc[0]
    return df


def plot_model_comparison(df, out_path):
    plt.figure(figsize=(8.5, 4.5))
    labels = [f"{row.model}\n({row.bins} bins)" for row in df.itertuples()]
    plt.bar(range(len(df)), df["rmse_test"].to_numpy())
    plt.xticks(range(len(df)), labels, rotation=25, ha="right")
    plt.ylabel("Temporal hold-out RMSE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_block_theta_summary(df, out_path):
    plt.figure(figsize=(6.4, 4.2))
    x = np.arange(len(df))
    y = df["theta_mean"].to_numpy()
    yerr = np.vstack([
        y - df["theta_hdi_low"].to_numpy(),
        df["theta_hdi_high"].to_numpy() - y,
    ])
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, [str(v) for v in df["block"].tolist()])
    plt.xlabel("Block")
    plt.ylabel("Theta (degrees)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_negative_control(null_df, observed_sd, out_path):
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(null_df["theta_sd"], bins=10, alpha=0.75, color="gray", edgecolor="black")
    plt.axvline(observed_sd, color="red", linewidth=2, label="Observed posterior SD")
    plt.xlabel("Posterior SD under x-permutation null")
    plt.ylabel("Count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_recovery(recovery_df, out_path):
    agg = (
        recovery_df.groupby("theta_true")
        .agg(theta_map_mean=("theta_map", "mean"), theta_map_sd=("theta_map", "std"))
        .reset_index()
    )
    agg["theta_map_sd"] = agg["theta_map_sd"].fillna(0.0)
    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(agg["theta_true"], agg["theta_map_mean"], yerr=agg["theta_map_sd"], fmt="o-", capsize=4)
    lims = [0, max(agg["theta_true"].max(), agg["theta_map_mean"].max()) + 5]
    plt.plot(lims, lims, "--", color="black", alpha=0.5)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True theta (degrees)")
    plt.ylabel("Recovered theta MAP")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ensure_dirs()
    t0 = time.time()
    print("Loading Gender-Science trial logs...")
    df = load_gender_science_df()
    print(f"Loaded {len(df):,} rows")

    curves_by_bins = {}
    arrays_by_bins = {}
    stats_by_bins = {}
    for n_bins in BIN_OPTIONS:
        curves = load_or_build_curves(df, n_bins=n_bins, blockwise=False)
        curves_by_bins[n_bins] = curves
        arrays_by_bins[n_bins] = arrays_from_curves(curves)
        stats_by_bins[n_bins] = stats_from_curves(curves)
        print(f"Prepared {len(curves):,} participant curves for {n_bins} bins")

    block_curves = load_or_build_curves(df, n_bins=PRIMARY_BINS, blockwise=True)
    print(f"Prepared {len(block_curves):,} participant-block curves")

    # Real-data pooled posterior on 6-bin curves.
    x6, y6_raw, mask6, _ = arrays_by_bins[PRIMARY_BINS]
    y6_std = row_standardize(y6_raw, mask6)
    posterior6, rss6 = profile_theta_posterior(x6, y6_std, mask6, THETA_POSTERIOR_GRID)
    pooled = posterior_summary(THETA_POSTERIOR_GRID, posterior6)
    pd.DataFrame({
        "theta_deg": THETA_POSTERIOR_GRID,
        "posterior": posterior6,
        "rss": rss6,
    }).to_csv(OUT_DIR / "theta_grid_profile_submission.csv", index=False)
    with open(TAB_DIR / "Table_theta_summary_submission.tex", "w", encoding="utf-8") as handle:
        handle.write(pd.DataFrame([{
            "Parameter": r"$\theta$ (deg)",
            "Mean": pooled["mean"],
            "SD": pooled["sd"],
            "94% HDI low": pooled["hdi_low"],
            "94% HDI high": pooled["hdi_high"],
            r"$\theta_{\mathrm{MAP}}$": pooled["map"],
        }]).to_latex(index=False, float_format="%.2f"))

    # Corrected full-sample model comparison across bins.
    print("Running corrected full-sample temporal hold-out comparison...")
    comp_rows = []
    model_specs = [
        ("interference", THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("cos_only", THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("sin_only", THETA_COARSE_GRID, THETA_REFINE_RADIUS, THETA_REFINE_STEP),
        ("poly2", None, None, None),
        ("exp", K_COARSE_GRID, K_REFINE_RADIUS, K_REFINE_STEP),
        ("power", P_COARSE_GRID, P_REFINE_RADIUS, P_REFINE_STEP),
    ]
    for n_bins in BIN_OPTIONS:
        x, y_raw, mask, _ = arrays_by_bins[n_bins]
        counts, sums, sumsqs = stats_by_bins[n_bins]
        for model_name, coarse_grid, refine_radius, refine_step in model_specs:
            print(f"  {model_name} | bins={n_bins}")
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
            comp_rows.append({
                "bins": n_bins,
                "model": model_name,
                "best_param": result["best_param"],
                "train_rss": result["train_rss"],
                "rmse_test": result["rmse_test"],
                "mean_logscore_test": result["mean_logscore_test"],
                "sigma_train": result["sigma_train"],
                "n_curves": result["n_curves"],
                "n_test_obs": result["n_test_obs"],
                "boundary_hit": result["boundary_hit"],
            })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "model_comparison_submission.csv", index=False)
    with open(TAB_DIR / "Table_model_comparison_submission.tex", "w", encoding="utf-8") as handle:
        handle.write(comp_df[comp_df["bins"] == PRIMARY_BINS].to_latex(index=False, float_format="%.4f"))
    with open(TAB_DIR / "Table_model_comparison_all_bins_submission.tex", "w", encoding="utf-8") as handle:
        handle.write(comp_df.to_latex(index=False, float_format="%.4f"))
    plot_model_comparison(comp_df[comp_df["bins"] == PRIMARY_BINS].reset_index(drop=True), FIG_DIR / "model_comparison_submission_bins6.png")

    # Block-specific theta and phase summaries.
    print("Profiling block-specific pooled theta...")
    block_theta_df = block_theta_summary(block_curves)
    block_theta_df.to_csv(OUT_DIR / "block_theta_summary.csv", index=False)
    with open(TAB_DIR / "Table_block_theta_summary.tex", "w", encoding="utf-8") as handle:
        handle.write(block_theta_df.to_latex(index=False, float_format="%.3f"))
    plot_block_theta_summary(block_theta_df, FIG_DIR / "block_theta_summary.png")

    print("Summarizing block-specific phase structure...")
    block_phase_df, obs_phase_dispersion, null_phase_dispersion, phase_p = block_phase_summary(block_curves, pooled["map"])
    block_phase_df["theta_map_global"] = pooled["map"]
    block_phase_df.to_csv(OUT_DIR / "block_phase_summary.csv", index=False)
    with open(TAB_DIR / "Table_block_phase_summary.tex", "w", encoding="utf-8") as handle:
        handle.write(block_phase_df.to_latex(index=False, float_format="%.4f"))

    # Direct competitor on participant-block curves.
    print("Evaluating block-specific theta competitor...")
    x_block, y_block_raw, mask_block, _ = arrays_from_curves(block_curves)
    counts_block, sums_block, sumsqs_block = stats_from_curves(block_curves)
    block_model_global = search_best_param(
        x_block,
        y_block_raw,
        mask_block,
        "interference",
        THETA_COARSE_GRID,
        THETA_REFINE_RADIUS,
        THETA_REFINE_STEP,
        counts=counts_block,
        sums=sums_block,
        sumsqs=sumsqs_block,
    )
    block_model_poly = fit_parameter_free_model(
        x_block,
        y_block_raw,
        mask_block,
        "poly2",
        counts=counts_block,
        sums=sums_block,
        sumsqs=sumsqs_block,
    )
    block_specific_agg, block_specific_details = evaluate_block_specific_theta(block_curves)
    block_model_df = pd.DataFrame([
        {
            "model": "Interference (global theta on block curves)",
            "best_param": block_model_global["best_param"],
            "rmse_test": block_model_global["rmse_test"],
            "mean_logscore_test": block_model_global["mean_logscore_test"],
            "boundary_hit": block_model_global["boundary_hit"],
        },
        block_specific_agg,
        {
            "model": "Polynomial (block curves)",
            "best_param": np.nan,
            "rmse_test": block_model_poly["rmse_test"],
            "mean_logscore_test": block_model_poly["mean_logscore_test"],
            "boundary_hit": False,
        },
    ])
    block_model_df.to_csv(OUT_DIR / "block_model_comparison.csv", index=False)
    block_specific_details.to_csv(OUT_DIR / "block_theta_holdout_details.csv", index=False)
    with open(TAB_DIR / "Table_block_model_comparison.tex", "w", encoding="utf-8") as handle:
        handle.write(block_model_df.to_latex(index=False, float_format="%.4f"))

    # Negative controls.
    print("Running negative controls...")
    perm_observed, perm_null_df = permutation_theta_null(x6, y6_std, mask6)
    perm_null_df.to_csv(OUT_DIR / "theta_permutation_null.csv", index=False)
    plot_negative_control(perm_null_df, perm_observed["sd"], FIG_DIR / "theta_permutation_null.png")
    neg_control_df = pd.DataFrame([
        {
            "control": "Within-curve x permutation",
            "observed_metric": perm_observed["sd"],
            "null_mean_metric": perm_null_df["theta_sd"].mean(),
            "null_mean_aux": perm_null_df["theta_hdi_width"].mean(),
            "p_value": (1.0 + np.sum(perm_null_df["theta_sd"].to_numpy() <= perm_observed["sd"])) / (len(perm_null_df) + 1.0),
        },
        {
            "control": "Block-label shuffle (phase dispersion)",
            "observed_metric": obs_phase_dispersion,
            "null_mean_metric": float(np.mean(null_phase_dispersion)),
            "null_mean_aux": np.nan,
            "p_value": phase_p,
        },
    ])
    neg_control_df.to_csv(OUT_DIR / "negative_controls_summary.csv", index=False)
    with open(TAB_DIR / "Table_negative_controls.tex", "w", encoding="utf-8") as handle:
        handle.write(neg_control_df.to_latex(index=False, float_format="%.4f"))

    # Simulation-based recovery and additive null.
    print("Running simulation-based recovery...")
    beta_emp = fixed_theta_coefficients(x6, y6_std, mask6, pooled["map"])
    poly_beta_emp = fit_poly_coefficients(x6, y6_std, mask6)
    sigma_real = 0.63
    recovery_df = simulate_interference_recovery(x6, mask6, beta_emp, sigma_real=sigma_real)
    null_sim_df = simulate_additive_null(x6, mask6, poly_beta_emp, sigma_real=sigma_real)
    recovery_df.to_csv(OUT_DIR / "theta_recovery_simulation.csv", index=False)
    null_sim_df.to_csv(OUT_DIR / "theta_null_simulation.csv", index=False)
    plot_recovery(recovery_df, FIG_DIR / "theta_recovery_simulation.png")

    recovery_summary = (
        recovery_df.groupby("theta_true")
        .agg(
            theta_map_mean=("theta_map", "mean"),
            theta_map_sd=("theta_map", "std"),
            theta_mean_mean=("theta_mean", "mean"),
            theta_sd_mean=("theta_sd", "mean"),
            coverage=("covered", "mean"),
        )
        .reset_index()
    )
    recovery_summary["theta_map_sd"] = recovery_summary["theta_map_sd"].fillna(0.0)
    with open(TAB_DIR / "Table_theta_recovery.tex", "w", encoding="utf-8") as handle:
        handle.write(recovery_summary.to_latex(index=False, float_format="%.4f"))

    null_summary = pd.DataFrame([{
        "generator": "poly2_null",
        "theta_map_mean": null_sim_df["theta_map"].mean(),
        "theta_map_sd": null_sim_df["theta_map"].std(ddof=1) if len(null_sim_df) > 1 else 0.0,
        "theta_sd_mean": null_sim_df["theta_sd"].mean(),
        "hdi_width_mean": null_sim_df["hdi_width"].mean(),
    }])
    with open(TAB_DIR / "Table_theta_null_simulation.tex", "w", encoding="utf-8") as handle:
        handle.write(null_summary.to_latex(index=False, float_format="%.4f"))

    print("Computing approximate pointwise WAIC table...")
    waic_df = approximate_waic_table(x6, y6_std, mask6)
    waic_df.to_csv(OUT_DIR / "waic_comparison_submission.csv", index=False)
    with open(TAB_DIR / "Table_waic_comparison_submission.tex", "w", encoding="utf-8") as handle:
        handle.write(waic_df.to_latex(index=False, float_format="%.4f"))

    summary_json = {
        "pooled_theta": pooled,
        "phase_dispersion_observed": float(obs_phase_dispersion),
        "phase_dispersion_null_mean": float(np.mean(null_phase_dispersion)),
        "phase_dispersion_p_value": float(phase_p),
        "elapsed_minutes": (time.time() - t0) / 60.0,
    }
    with open(OUT_DIR / "submission_evidence_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)

    print("Submission evidence analysis complete.")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
