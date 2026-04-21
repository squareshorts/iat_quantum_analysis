import pickle
from src.tools.splits import make_participant_kfold_splits

# Load cached curves
with open("outputs/curves_cache_bins6.pkl", "rb") as f:
    curves = pickle.load(f)

print(f"Loaded curves cache of type: {type(curves)}")

# Extract participant IDs (curves is a list of dicts)
participant_ids = [c["pid"] for c in curves]

print(f"Found {len(participant_ids):,} participants")

make_participant_kfold_splits(
    participant_ids=participant_ids,
    n_folds=5,
    seed=123,
    out_path="outputs/splits_participant_kfold.json",
)

print("Participant-level K-fold splits written to outputs/")
