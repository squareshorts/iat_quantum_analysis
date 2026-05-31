from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "synthetic" / "life_satisfaction_iat_synthetic.csv"
EXPECTED_PARTICIPANTS = 180
EXPECTED_BLOCKS = {3, 5}
EXPECTED_TRIALS_PER_BLOCK = 20
REQUIRED_COLUMNS = {
    "participant_id",
    "block",
    "trial_index",
    "trial_in_block",
    "display_order",
    "response_key",
    "response_label",
    "latency_ms",
    "rt",
    "accuracy",
    "correct",
    "stage",
    "congruency",
    "phase",
    "source_file",
    "n_attempts",
    "n_errors",
}


def validate(path: Path = DEFAULT_INPUT) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing synthetic dataset: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise AssertionError(f"Synthetic dataset is missing required columns: {missing}")

    n_participants = int(df["participant_id"].nunique())
    if n_participants != EXPECTED_PARTICIPANTS:
        raise AssertionError(f"Expected {EXPECTED_PARTICIPANTS} participants, found {n_participants}")

    participant_ids = df["participant_id"].astype(str)
    if not participant_ids.str.fullmatch(r"synthetic_lsiat_\d{3}").all():
        raise AssertionError("Participant IDs must be synthetic_lsiat_### placeholders only")

    blocks = set(df["block"].astype(int).unique())
    if blocks != EXPECTED_BLOCKS:
        raise AssertionError(f"Expected blocks {sorted(EXPECTED_BLOCKS)}, found {sorted(blocks)}")

    counts = df.groupby(["participant_id", "block"]).size()
    bad_counts = counts[counts != EXPECTED_TRIALS_PER_BLOCK]
    if not bad_counts.empty:
        raise AssertionError("Each participant must have 20 trials in each critical block")

    if df.duplicated(["participant_id", "block", "trial_in_block"]).any():
        raise AssertionError("Duplicate participant/block/trial records found")

    latency = pd.to_numeric(df["latency_ms"], errors="coerce")
    if latency.isna().any() or (latency < 300).any() or (latency > 10000).any():
        raise AssertionError("Latencies must be numeric and inside the 300--10,000 ms review window")
    if float(latency.mean()) <= float(latency.median()):
        raise AssertionError("Synthetic latency distribution should be broadly right-skewed")

    stages = set(df["stage"].astype(str).str.lower().unique())
    if not {"congruent", "incongruent"}.issubset(stages):
        raise AssertionError("Synthetic data must include congruent and incongruent stages")

    correctness = set(pd.to_numeric(df["correct"], errors="coerce").dropna().astype(int).unique())
    if not correctness.issubset({0, 1}) or correctness != {0, 1}:
        raise AssertionError("Synthetic correct flag must include binary 0/1 values")

    return {
        "path": str(path),
        "rows": int(len(df)),
        "participants": n_participants,
        "blocks": [int(block) for block in sorted(blocks)],
        "mean_latency_ms": float(latency.mean()),
        "median_latency_ms": float(latency.median()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic life-satisfaction IAT review data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Synthetic CSV to validate.")
    args = parser.parse_args(argv)

    summary = validate(args.input)
    print(
        "Synthetic life-satisfaction IAT validation passed: "
        f"{summary['rows']} rows, {summary['participants']} participants, "
        f"blocks {summary['blocks']}"
    )


if __name__ == "__main__":
    main()
