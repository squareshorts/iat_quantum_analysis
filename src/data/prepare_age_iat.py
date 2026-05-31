from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PARQUET_PATH = BASE_DIR / "data" / "interim" / "age_iat" / "age_iat_trials_raw.parquet"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "age_iat"
REPORTS_DIR = BASE_DIR / "reports"
DOWNLOAD_METADATA_PATH = BASE_DIR / "data" / "raw" / "age_iat" / "age_iat_download_metadata.json"

STANDARDIZED_PARQUET_PATH = PROCESSED_DIR / "age_iat_trials_standardized.parquet"
QC_SUMMARY_PATH = PROCESSED_DIR / "age_iat_qc_summary.csv"
PARTICIPANT_QC_PATH = PROCESSED_DIR / "age_iat_participant_qc.csv"
RT_CLEANING_SUMMARY_PATH = PROCESSED_DIR / "age_iat_rt_cleaning_summary.csv"
QC_REPORT_PATH = REPORTS_DIR / "age_iat_qc_report.md"

DEFAULT_CRITICAL_BLOCKS = [3, 4, 6, 7]

TARGET_ALIASES = {
    "pid": [
        "session_id",
        "sessionid",
        "session",
        "user_id",
        "participant_id",
        "respondent_id",
    ],
    "block": [
        "block",
        "blocknum",
        "block_number",
        "trialblock",
        "trial_block",
        "block_number",
    ],
    "trial_in_block": [
        "trial",
        "trialnum",
        "trial_number",
        "trial_index",
    ],
    "rt": [
        "latency",
        "rt",
        "response_time",
        "latency_ms",
        "trial_latency",
    ],
    "trial_error": [
        "error",
        "correct",
        "accuracy",
        "trial_error",
    ],
    "stimulus": [
        "stimulus",
        "stim",
        "target",
        "trial_name",
    ],
    "category": [
        "category",
        "attribute",
        "left_category",
        "right_category",
        "trial_response",
        "block_pairing_definition",
    ],
}


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def find_matching_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {normalize_column_name(column): column for column in columns}
    for alias in aliases:
        alias_key = normalize_column_name(alias)
        if alias_key in normalized:
            return normalized[alias_key]
    for alias in aliases:
        alias_key = normalize_column_name(alias)
        for norm_name, original_name in normalized.items():
            if norm_name == alias_key or norm_name.endswith(f"_{alias_key}") or alias_key in norm_name:
                return original_name
    return None


def _coerce_numeric(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce")


def standardize_age_iat_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping: dict[str, str] = {}
    columns = list(df.columns)
    for target_name, aliases in TARGET_ALIASES.items():
        match = find_matching_column(columns, aliases)
        if match:
            mapping[target_name] = match

    required = {"pid", "block", "trial_in_block", "rt"}
    missing = sorted(required - set(mapping))
    if missing:
        raise RuntimeError(
            f"Could not map required Age IAT columns: {missing}. Available columns: {columns}"
        )

    standardized = pd.DataFrame()
    standardized["pid"] = df[mapping["pid"]].astype("string")
    standardized["block"] = _coerce_numeric(df[mapping["block"]])
    standardized["trial_in_block"] = _coerce_numeric(df[mapping["trial_in_block"]])
    standardized["rt_raw_ms"] = _coerce_numeric(df[mapping["rt"]])

    if "trial_error" in mapping:
        error_source = mapping["trial_error"]
        error_values = _coerce_numeric(df[error_source])
        norm_name = normalize_column_name(error_source)
        if "correct" in norm_name or "accuracy" in norm_name:
            standardized["trial_error"] = 1.0 - error_values
        else:
            standardized["trial_error"] = error_values

    for optional in ["stimulus", "category"]:
        if optional in mapping:
            standardized[optional] = df[mapping[optional]].astype("string")

    for passthrough in ["study_name", "task_name", "source_archive", "source_member", "source_year"]:
        if passthrough in df.columns:
            standardized[passthrough] = df[passthrough].astype("string")

    if "block_name" in df.columns:
        standardized["block_name"] = df["block_name"].astype("string")
    if "trial_response" in df.columns and "category" not in standardized.columns:
        standardized["category"] = df["trial_response"].astype("string")

    return standardized, mapping


def filter_critical_blocks(df: pd.DataFrame, critical_blocks: list[int]) -> pd.DataFrame:
    return df[df["block"].isin(critical_blocks)].copy()

import numpy as np
def compute_rt_stats(series: pd.Series, prefix: str = "") -> dict[str, object]:
    valid = series.dropna()
    n = len(valid)
    if n == 0:
        stats = {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "iqr": np.nan, "mad_scaled": np.nan, "min": np.nan, "max": np.nan, "p01": np.nan, "p05": np.nan, "p95": np.nan, "p99": np.nan}
    else:
        median = float(valid.median())
        mad = float(np.median(np.abs(valid - median)))
        stats = {
            "n": n,
            "mean": float(valid.mean()),
            "sd": float(valid.std(ddof=1)) if n > 1 else 0.0,
            "median": median,
            "iqr": float(valid.quantile(0.75) - valid.quantile(0.25)),
            "mad_scaled": mad * 1.4826,
            "min": float(valid.min()),
            "max": float(valid.max()),
            "p01": float(valid.quantile(0.01)),
            "p05": float(valid.quantile(0.05)),
            "p95": float(valid.quantile(0.95)),
            "p99": float(valid.quantile(0.99)),
        }
    if prefix:
        return {f"{prefix}_{k}": v for k, v in stats.items()}
    return stats


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_download_metadata() -> dict[str, object]:
    if not DOWNLOAD_METADATA_PATH.exists():
        return {}
    return json.loads(DOWNLOAD_METADATA_PATH.read_text(encoding="utf-8"))


def build_qc_report(
    *,
    summary_row: pd.Series,
    critical_blocks: list[int],
    download_metadata: dict[str, object],
) -> str:
    downloaded_files = [
        item.get("name", "")
        for item in download_metadata.get("selected_files", [])
        if isinstance(item, dict)
    ]
    source_nodes = sorted(
        {
            item.get("source_node_id", "")
            for item in download_metadata.get("selected_files", [])
            if isinstance(item, dict) and item.get("source_node_id")
        }
    )
    return "\n".join(
        [
            "# Age IAT QC report",
            "",
            f"- Raw rows: {int(summary_row['raw_rows'])}",
            f"- Retained trial rows: {int(summary_row['retained_trial_rows'])}",
            f"- Participants: {int(summary_row['participants'])}",
            f"- Available blocks: {summary_row['available_blocks']}",
            f"- Retained critical blocks: {', '.join(str(block) for block in critical_blocks)}",
            f"- Pre-cleaning RT median: {summary_row.get('pre_median', np.nan):.2f} ms",
            f"- Pre-cleaning RT mean: {summary_row.get('pre_mean', np.nan):.2f} ms",
            f"- Pre-cleaning RT SD: {summary_row.get('pre_sd', np.nan):.2f} ms",
            f"- Post-cleaning RT median: {summary_row.get('post_median', np.nan):.2f} ms",
            f"- Post-cleaning RT mean: {summary_row.get('post_mean', np.nan):.2f} ms",
            f"- Post-cleaning RT SD: {summary_row.get('post_sd', np.nan):.2f} ms",
            f"- Fast-trial proportion (<300 ms): {summary_row.get('pre_prop_fast_lt300', np.nan):.4f}",
            f"- Long-trial proportion (>10000 ms): {summary_row.get('pre_prop_long_gt10000', np.nan):.4f}",
            (
                f"- Error rate: {summary_row['error_rate']:.4f}"
                if pd.notna(summary_row.get("error_rate"))
                else "- Error rate: unavailable"
            ),
            f"- Exact files downloaded: {', '.join(downloaded_files) if downloaded_files else 'unavailable'}",
            (
                f"- Exact OSF node used: root {download_metadata.get('requested_osf_node', 'unknown')}; "
                f"selected source node(s) {', '.join(source_nodes) if source_nodes else 'unknown'}"
            ),
            "- File level used by preprocessing: trial-level",
        ]
    )


def prepare_age_iat(critical_blocks: list[int]) -> dict[str, object]:
    ensure_dirs()
    if not RAW_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw Age IAT parquet at {RAW_PARQUET_PATH}. Run `python -m src.data.download_age_iat` first."
        )

    LOGGER.info("Loading raw Age IAT parquet from %s", RAW_PARQUET_PATH)
    raw_df = pd.read_parquet(RAW_PARQUET_PATH)
    raw_rows = len(raw_df)
    standardized, column_mapping = standardize_age_iat_columns(raw_df)

    standardized["block"] = pd.to_numeric(standardized["block"], errors="coerce")
    standardized["trial_in_block"] = pd.to_numeric(standardized["trial_in_block"], errors="coerce")
    standardized["rt_raw_ms"] = pd.to_numeric(standardized["rt_raw_ms"], errors="coerce")
    if "trial_error" in standardized.columns:
        standardized["trial_error"] = pd.to_numeric(standardized["trial_error"], errors="coerce")

    available_blocks = sorted(
        int(block)
        for block in standardized["block"].dropna().astype(int).unique().tolist()
    )

    standardized["missing_rt"] = standardized["rt_raw_ms"].isna()
    standardized["nonpositive_rt"] = standardized["rt_raw_ms"].le(0).fillna(False)
    standardized["is_fast_lt300"] = standardized["rt_raw_ms"].lt(300).fillna(False)
    standardized["is_long_gt10000"] = standardized["rt_raw_ms"].gt(10000).fillna(False)
    standardized["is_critical_block"] = standardized["block"].isin(critical_blocks)

    cleaned = standardized[
        standardized["pid"].notna()
        & standardized["block"].notna()
        & standardized["trial_in_block"].notna()
    ].copy()
    cleaned = cleaned[~cleaned["missing_rt"]].copy()
    cleaned = cleaned[~cleaned["nonpositive_rt"]].copy()
    cleaned = filter_critical_blocks(cleaned, critical_blocks=critical_blocks)

    pre_stats = compute_rt_stats(cleaned["rt_raw_ms"], prefix="pre")
    pre_stats["pre_prop_fast_lt300"] = float(cleaned["is_fast_lt300"].mean())
    pre_stats["pre_prop_long_gt10000"] = float(cleaned["is_long_gt10000"].mean())

    cleaned = cleaned[~cleaned["is_fast_lt300"]].copy()
    cleaned = cleaned[~cleaned["is_long_gt10000"]].copy()
    
    cleaned["rt"] = cleaned["rt_raw_ms"].copy()

    post_stats = compute_rt_stats(cleaned["rt"], prefix="post")
    post_stats["post_prop_fast_lt300"] = float(cleaned["rt"].lt(300).mean())
    post_stats["post_prop_long_gt10000"] = float(cleaned["rt"].gt(10000).mean())

    if post_stats.get("post_sd", 0) > 3000:
        top_rts = cleaned["rt"].nlargest(20).tolist()
        raise ValueError(f"Cleaned RT SD is still implausibly large ({post_stats['post_sd']:.2f} > 3000). Top RTs: {top_rts}")

    cleaned["block"] = cleaned["block"].astype(int)
    cleaned["trial_in_block"] = cleaned["trial_in_block"].astype(int)
    cleaned = cleaned.sort_values(["pid", "block", "trial_in_block"]).reset_index(drop=True)

    participant_qc = (
        standardized.groupby("pid", dropna=True)
        .agg(
            total_trials=("pid", "size"),
            missing_rt_removed=("missing_rt", "sum"),
            nonpositive_rt_removed=("nonpositive_rt", "sum"),
            fast_lt300=("is_fast_lt300", "sum"),
            long_gt10000=("is_long_gt10000", "sum"),
            critical_block_trials=("is_critical_block", "sum"),
        )
        .reset_index()
    )
    
    crit_mask = standardized["is_critical_block"]
    crit_agg = standardized[crit_mask].groupby("pid", dropna=True).agg(
        fast_prop=("is_fast_lt300", "mean"),
        long_prop=("is_long_gt10000", "mean"),
    ).reset_index()
    if "trial_error" in standardized.columns:
        err_agg = standardized[crit_mask].groupby("pid", dropna=True)["trial_error"].mean().rename("error_rate").reset_index()
        crit_agg = crit_agg.merge(err_agg, on="pid", how="left")
    
    participant_qc = participant_qc.merge(crit_agg, on="pid", how="left")
    retained_counts = cleaned.groupby("pid").size().rename("retained_critical_trials")
    participant_qc = participant_qc.merge(retained_counts, on="pid", how="left").fillna({"retained_critical_trials": 0})
    participant_qc["retained_critical_trials"] = participant_qc["retained_critical_trials"].astype(int)
    participant_qc.to_csv(PARTICIPANT_QC_PATH, index=False)

    error_rate = pd.NA
    if "trial_error" in cleaned.columns:
        error_rate = float(cleaned["trial_error"].fillna(0).mean())

    summary_row = pd.Series(
        {
            "raw_rows": raw_rows,
            "retained_trial_rows": len(cleaned),
            "participants": cleaned["pid"].nunique(),
            "available_blocks": ", ".join(str(block) for block in available_blocks),
            "retained_critical_blocks": ", ".join(str(block) for block in critical_blocks),
            "error_rate": error_rate,
            **pre_stats,
            **post_stats,
            "participants_with_any_removed_rows": int(
                (participant_qc["missing_rt_removed"] + participant_qc["nonpositive_rt_removed"] > 0).sum()
            ),
            "column_mapping": json.dumps(column_mapping, sort_keys=True),
        }
    )
    pd.DataFrame([summary_row]).to_csv(QC_SUMMARY_PATH, index=False)
    
    cleaning_summary = pd.DataFrame([{"phase": "pre_cleaning", **pre_stats}, {"phase": "post_cleaning", **post_stats}])
    cleaning_summary.to_csv(RT_CLEANING_SUMMARY_PATH, index=False)
    cleaned.to_parquet(STANDARDIZED_PARQUET_PATH, index=False)

    download_metadata = load_download_metadata()
    report_text = build_qc_report(
        summary_row=summary_row,
        critical_blocks=critical_blocks,
        download_metadata=download_metadata,
    )
    QC_REPORT_PATH.write_text(report_text, encoding="utf-8")

    LOGGER.info("Saved standardized Age IAT parquet to %s", STANDARDIZED_PARQUET_PATH)
    LOGGER.info("Saved QC summary to %s", QC_SUMMARY_PATH)
    LOGGER.info("Saved QC report to %s", QC_REPORT_PATH)
    return {
        "standardized_path": str(STANDARDIZED_PARQUET_PATH),
        "qc_summary_path": str(QC_SUMMARY_PATH),
        "participant_qc_path": str(PARTICIPANT_QC_PATH),
        "rt_cleaning_summary_path": str(RT_CLEANING_SUMMARY_PATH),
        "qc_report_path": str(QC_REPORT_PATH),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare downloaded Age IAT raw data for the geometric pipeline.")
    parser.add_argument(
        "--critical-blocks",
        nargs="+",
        type=int,
        default=DEFAULT_CRITICAL_BLOCKS,
        help="Critical IAT blocks to retain.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return prepare_age_iat(critical_blocks=list(args.critical_blocks))


if __name__ == "__main__":
    main()
