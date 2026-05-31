from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SYNTHETIC_CSV = BASE_DIR / "data" / "synthetic" / "life_satisfaction_iat_synthetic.csv"
RELEASE_ZIP = BASE_DIR / "release" / "life_satisfaction_iat_database_zenodo.zip"


def run_command(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=BASE_DIR, check=True)


def load_external_module():
    module_path = BASE_DIR / "scripts" / "run_external_leverage_analysis.py"
    spec = importlib.util.spec_from_file_location("run_external_leverage_analysis", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    run_command(["scripts/generate_synthetic_life_satisfaction_iat.py"])
    run_command(["scripts/validate_synthetic_life_satisfaction_iat.py"])
    run_command(["scripts/build_life_satisfaction_zenodo_package.py"])

    external = load_external_module()
    _, participants, clean_trials, summary = external.load_synthetic_external_trials(SYNTHETIC_CSV)
    theta_df, posterior = external.external_theta_summary(clean_trials)

    if int(summary["valid_participants"]) != 180:
        raise AssertionError("Synthetic life-satisfaction path did not preserve 180 participants")
    if int(theta_df.loc[0, "n"]) != 180:
        raise AssertionError("Synthetic theta summary did not produce 180 curves")
    if abs(float(posterior.sum()) - 1.0) > 1e-6:
        raise AssertionError("Synthetic theta posterior is not normalized")
    if not RELEASE_ZIP.exists() or RELEASE_ZIP.stat().st_size == 0:
        raise AssertionError("De-identified life-satisfaction Zenodo package was not created")
    if participants["participant_id"].nunique() != 180:
        raise AssertionError("Synthetic participant summary has unexpected participant count")

    print("Reproducibility smoke test passed without restricted life-satisfaction raw data.")


if __name__ == "__main__":
    main()
