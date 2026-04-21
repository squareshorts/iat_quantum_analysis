import json
import numpy as np
from pathlib import Path


def make_participant_kfold_splits(
    participant_ids,
    n_folds=5,
    seed=123,
    out_path="outputs/splits_participant_kfold.json",
):
    """
    Create participant-level K-fold splits.

    Each fold contains a list of participant IDs held out for testing.
    Training participants are implicitly all others.
    """
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(participant_ids))
    rng.shuffle(pids)

    folds = np.array_split(pids, n_folds)

    splits = []
    for k in range(n_folds):
        splits.append(
            {
                "fold": k,
                "test_participants": folds[k].tolist(),
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_folds": n_folds,
                "seed": seed,
                "splits": splits,
            },
            f,
            indent=2,
        )

    return splits


if __name__ == "__main__":
    raise RuntimeError(
        "This module is not meant to be executed directly. "
        "Import and call make_participant_kfold_splits()."
    )
