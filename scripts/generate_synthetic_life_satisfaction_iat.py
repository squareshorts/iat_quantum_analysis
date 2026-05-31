from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = BASE_DIR / "data" / "synthetic" / "life_satisfaction_iat_synthetic.csv"
DEFAULT_SEED = 20260531
N_PARTICIPANTS = 180
TRIALS_PER_CRITICAL_BLOCK = 20


def generate_synthetic_life_satisfaction_iat(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate synthetic review-only trial records for the life-satisfaction IAT.

    These records are simulated for code execution and review. They preserve the
    non-identifying structure required by the pipeline, but they are not real
    participant records and should not be used to reproduce empirical estimates.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    stage_by_block = {3: "congruent", 5: "incongruent"}

    for participant_index in range(1, N_PARTICIPANTS + 1):
        participant_id = f"synthetic_lsiat_{participant_index:03d}"
        participant_shift = rng.normal(0.0, 0.18)
        participant_accuracy_shift = rng.normal(0.0, 0.015)

        for block in (3, 5):
            stage = stage_by_block[block]
            phase = "fase3" if block == 3 else "fase5"
            stage_latency_shift = 0.0 if block == 3 else 0.16
            for trial_index in range(1, TRIALS_PER_CRITICAL_BLOCK + 1):
                stimulus_family = "logical_anchor" if trial_index <= 10 else "life_satisfaction_statement"
                family_shift = -0.06 if stimulus_family == "logical_anchor" else 0.08
                practice_trend = -0.10 * ((trial_index - 1) / (TRIALS_PER_CRITICAL_BLOCK - 1))
                log_latency = (
                    np.log(2450.0)
                    + participant_shift
                    + stage_latency_shift
                    + family_shift
                    + practice_trend
                    + rng.normal(0.0, 0.34)
                )
                latency_ms = int(np.clip(np.round(rng.lognormal(log_latency, 0.12)), 320, 10000))
                error_probability = np.clip(
                    0.055
                    + (0.025 if block == 5 else 0.0)
                    + (0.012 if stimulus_family == "life_satisfaction_statement" else 0.0)
                    + participant_accuracy_shift,
                    0.01,
                    0.22,
                )
                correct = int(rng.random() > error_probability)
                n_errors = int(not correct)
                response_key = "e" if (trial_index + block + participant_index) % 2 == 0 else "i"

                rows.append(
                    {
                        "participant_id": participant_id,
                        "source_file": f"{participant_id}.csv",
                        "phase": phase,
                        "block": block,
                        "trial_index": trial_index,
                        "trial_in_block": trial_index,
                        "display_order": trial_index,
                        "stimulus_id": f"{stimulus_family}_{trial_index:02d}",
                        "stimulus_family": stimulus_family,
                        "response_key": response_key,
                        "response_label": "left" if response_key == "e" else "right",
                        "latency_ms": latency_ms,
                        "rt": float(latency_ms),
                        "accuracy": correct,
                        "correct": correct,
                        "n_attempts": 1 + n_errors,
                        "n_errors": n_errors,
                        "stage": stage,
                        "congruency": stage,
                        "synthetic_seed": seed,
                    }
                )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic life-satisfaction IAT review data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed random seed.")
    args = parser.parse_args(argv)

    df = generate_synthetic_life_satisfaction_iat(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} synthetic rows for {df['participant_id'].nunique()} participants to {args.output}")


if __name__ == "__main__":
    main()
