from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
DEFAULT_RELEASE_DIR = BASE_DIR / "release" / "life_satisfaction_iat_database"
DEFAULT_ZIP = BASE_DIR / "release" / "life_satisfaction_iat_database_zenodo.zip"
SOFTWARE_ZENODO_DOI = "10.5281/zenodo.19711302"
GITHUB_REPOSITORY = "https://github.com/squareshorts/iat_quantum_analysis"


def load_analysis_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trials_path = OUTPUTS_DIR / "external_life_satisfaction_trials_clean.csv"
    participants_path = OUTPUTS_DIR / "external_life_satisfaction_participants_clean.csv"
    exclusions_path = OUTPUTS_DIR / "external_life_satisfaction_excluded_cases.csv"
    missing = [path for path in [trials_path, participants_path, exclusions_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing life-satisfaction analysis outputs: {missing_text}")
    return (
        pd.read_csv(trials_path),
        pd.read_csv(participants_path),
        pd.read_csv(exclusions_path),
    )


def deidentify(
    trials: pd.DataFrame,
    participants: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pid_order = sorted(participants["participant_id"].astype(str).unique())
    pid_map = {old: f"LSIAT{idx:03d}" for idx, old in enumerate(pid_order, start=1)}

    trials_public = trials.copy()
    trials_public["participant_id"] = trials_public["participant_id"].astype(str).map(pid_map)
    trials_public = trials_public.rename(
        columns={
            "base_trial": "trial_index",
            "rt_total": "latency_ms",
            "rt": "analysis_latency_ms",
        }
    )
    trials_public["stage"] = trials_public["phase"].map({"fase3": "congruent", "fase5": "incongruent"}).fillna("practice")
    trials_public["congruency"] = trials_public["stage"]
    trials_public["display_order"] = trials_public["trial_in_block"]
    trials_public = trials_public[
        [
            "participant_id",
            "phase",
            "block",
            "trial_index",
            "trial_in_block",
            "display_order",
            "latency_ms",
            "analysis_latency_ms",
            "n_attempts",
            "n_errors",
            "stage",
            "congruency",
        ]
    ].sort_values(["participant_id", "block", "trial_in_block"])

    participants_public = participants.copy()
    participants_public["participant_id"] = participants_public["participant_id"].astype(str).map(pid_map)
    participants_public = participants_public.drop(columns=["source_file"], errors="ignore")
    participants_public = participants_public.sort_values("participant_id")

    exclusions_public = exclusions.copy().reset_index(drop=True)
    exclusions_public.insert(0, "excluded_case_id", [f"EXCLUDED{idx:03d}" for idx in range(1, len(exclusions_public) + 1)])
    exclusions_public = exclusions_public.drop(columns=["source_file", "gmt_timestamp", "duration_s"], errors="ignore")

    return trials_public, participants_public, exclusions_public


def write_release_files(
    release_dir: Path,
    trials: pd.DataFrame,
    participants: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> list[Path]:
    release_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    trial_path = release_dir / "life_satisfaction_iat_trials_deidentified.csv"
    participant_path = release_dir / "life_satisfaction_iat_participants_deidentified.csv"
    exclusion_path = release_dir / "life_satisfaction_iat_exclusions_deidentified.csv"
    trials.to_csv(trial_path, index=False)
    participants.to_csv(participant_path, index=False)
    exclusions.to_csv(exclusion_path, index=False)
    paths.extend([trial_path, participant_path, exclusion_path])

    readme_path = release_dir / "README.md"
    readme_path.write_text(
        f"""# De-identified Life-satisfaction IAT Database

This package contains the de-identified life-satisfaction IAT database used as the independent external comparison in the manuscript `Angular profiling of contextual structure in implicit-association latency data`.

The original Testable exports are not included. Direct identifiers, raw source filenames, exact collection timestamps, and participant names were removed before packaging. Participant IDs are release-specific placeholders (`LSIAT001`--`LSIAT180`) and cannot be linked back to the local raw files from this package.

Files:

- `life_satisfaction_iat_trials_deidentified.csv`: collapsed trial-level analysis records for 180 participants across the five task phases.
- `life_satisfaction_iat_participants_deidentified.csv`: participant-level derived analysis summaries.
- `life_satisfaction_iat_exclusions_deidentified.csv`: de-identified exclusion summary for excluded local records.
- `CODEBOOK.md`: variable definitions and cleaning notes.
- `zenodo_metadata.json`: suggested Zenodo metadata for the dataset record.

Related software archive: https://doi.org/{SOFTWARE_ZENODO_DOI}
Related GitHub repository: {GITHUB_REPOSITORY}
""",
        encoding="utf-8",
    )
    paths.append(readme_path)

    codebook_path = release_dir / "CODEBOOK.md"
    codebook_path.write_text(
        """# Codebook

## life_satisfaction_iat_trials_deidentified.csv

- `participant_id`: release-specific placeholder identifier.
- `phase`: original task phase label (`fase1`--`fase5`).
- `block`: numeric task block mapped from the phase.
- `trial_index`: collapsed item index within phase.
- `trial_in_block`: trial order used by the analysis pipeline.
- `display_order`: within-block display order.
- `latency_ms`: collapsed latency in milliseconds after repeated correction attempts were summed back to the original item.
- `analysis_latency_ms`: latency value used in the analysis pipeline.
- `n_attempts`: number of attempts contributing to the collapsed item.
- `n_errors`: number of incorrect attempts before the final/correct response.
- `stage`: practice, congruent, or incongruent stage label.
- `congruency`: same coding as `stage`, retained for pipeline compatibility.

## life_satisfaction_iat_participants_deidentified.csv

- `participant_id`: release-specific placeholder identifier.
- `total_items`: number of collapsed task items.
- `total_attempts`: total attempts before collapsing correction loops.
- `total_errors`: total incorrect attempts.
- `mean_rt`: mean collapsed latency in milliseconds.
- `median_rt`: median collapsed latency in milliseconds.
- `error_rate`: `total_errors / total_attempts`.
- `rt_congruent`: mean latency in the congruent combined phase.
- `rt_incongruent`: mean latency in the incongruent combined phase.
- `sd_critical`: participant-level standard deviation across critical combined phases.
- `d_like`: D-like latency contrast used for analysis checks.

## life_satisfaction_iat_exclusions_deidentified.csv

- `excluded_case_id`: release-specific placeholder for excluded local record.
- `prop_fast_lt300_attempts`: proportion of raw attempts faster than 300 ms.
- `n_attempt_rows`: number of raw attempt rows in the local export.
- `repeat_rank`: repeat-attempt rank before de-identification.
- `exclude_fast`: whether the record exceeded the fast-response exclusion threshold.
- `exclude_repeat`: whether the record was a repeated later attempt.
- `keep`: retained flag; all records in this file are excluded.
- `reason`: exclusion reason.
""",
        encoding="utf-8",
    )
    paths.append(codebook_path)

    metadata_path = release_dir / "zenodo_metadata.json"
    metadata = {
        "title": "De-identified life-satisfaction IAT database for angular profiling of implicit-association latency data",
        "upload_type": "dataset",
        "description": (
            "De-identified life-satisfaction Implicit Association Test database used as the independent "
            "external comparison in the manuscript 'Angular profiling of contextual structure in "
            "implicit-association latency data'. The package excludes original Testable exports, participant "
            "names, source filenames, and exact collection timestamps."
        ),
        "creators": [
            {"name": "Matos, Felipe de Oliveira", "affiliation": "Universidade Estadual de Maringa", "orcid": "0000-0002-4926-4694"},
            {"name": "Andrade de Lima, Marlos", "affiliation": "Universidade Federal do Rio Grande do Sul", "orcid": "0000-0002-8901-2272"},
            {"name": "Zanon, Cristian", "affiliation": "Universidade Federal do Rio Grande do Sul", "orcid": "0000-0003-3822-5275"},
            {"name": "Pereira, Antonio", "affiliation": "Universidade Federal do Para", "orcid": "0000-0002-0808-1058"},
        ],
        "access_right": "open",
        "license": "cc-by-4.0",
        "keywords": [
            "Implicit Association Test",
            "life satisfaction",
            "response latency",
            "behavioral data",
            "de-identified data",
        ],
        "related_identifiers": [
            {
                "identifier": SOFTWARE_ZENODO_DOI,
                "relation": "isSupplementTo",
                "scheme": "doi",
                "resource_type": "software",
            },
            {
                "identifier": GITHUB_REPOSITORY,
                "relation": "isSupplementTo",
                "scheme": "url",
                "resource_type": "software",
            },
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths.append(metadata_path)
    return paths


def scan_for_direct_identifiers(paths: list[Path]) -> None:
    csv_paths = [path for path in paths if path.suffix.lower() == ".csv"]
    forbidden_column_tokens = {"source_file", "gmt_timestamp", "local_timestamp", "nome", "email", "link", "ip_address"}
    patterns = {
        "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "raw_source_path": re.compile(r"results/223251_[0-9_]+\.csv"),
        "testable_link": re.compile(r"https://www\.testable\.org/", re.IGNORECASE),
        "nome_completo": re.compile(r"Nome completo", re.IGNORECASE),
    }
    for path in csv_paths:
        df = pd.read_csv(path, dtype=str)
        bad_columns = [
            column for column in df.columns
            if any(token in column.lower() for token in forbidden_column_tokens)
        ]
        if bad_columns:
            raise AssertionError(f"{path.name} contains forbidden identifier-like columns: {bad_columns}")
        text = "\n".join(df.fillna("").astype(str).agg(" ".join, axis=1).tolist())
        hits = [name for name, pattern in patterns.items() if pattern.search(text)]
        if hits:
            raise AssertionError(f"{path.name} contains direct-identifier pattern(s): {hits}")


def write_zip(paths: list[Path], zip_path: Path, release_dir: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=path.relative_to(release_dir))
    return zip_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a de-identified Zenodo package for the life-satisfaction IAT database.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args(argv)

    trials, participants, exclusions = load_analysis_outputs()
    trials_public, participants_public, exclusions_public = deidentify(trials, participants, exclusions)
    paths = write_release_files(args.release_dir, trials_public, participants_public, exclusions_public)
    scan_for_direct_identifiers(paths)
    zip_path = write_zip(paths, args.zip_path, args.release_dir)
    print(f"Wrote de-identified release package to {args.release_dir}")
    print(f"Wrote Zenodo upload zip to {zip_path}")
    print(f"Trial rows: {len(trials_public)}; participants: {participants_public['participant_id'].nunique()}")


if __name__ == "__main__":
    main()
