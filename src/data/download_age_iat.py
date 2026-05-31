from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from tqdm import tqdm

from .osf_utils import classify_file_level, discover_osf_files_recursive, infer_file_type


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "age_iat"
INTERIM_DIR = BASE_DIR / "data" / "interim" / "age_iat"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "age_iat"
REPORTS_DIR = BASE_DIR / "reports"

DEFAULT_OSF_NODE = "cv7iq"
DEFAULT_YEAR = "2019"
MANIFEST_PATH = RAW_DIR / "osf_manifest_age_iat.csv"
DOWNLOAD_METADATA_PATH = RAW_DIR / "age_iat_download_metadata.json"
RAW_PARQUET_PATH = INTERIM_DIR / "age_iat_trials_raw.parquet"

SUPPORTED_TABULAR_TYPES = {"csv", "tsv", "txt", "sav", "dta"}
SUPPORTED_ARCHIVE_TYPES = {"zip", "csv.zip", "tsv.zip", "txt.zip", "sav.zip", "dta.zip"}
REQUIRED_TRIAL_COLUMNS = {"session_id", "block_number", "trial_number", "trial_latency"}


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def ensure_dirs() -> None:
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def infer_year_from_name(name: str) -> str | None:
    match = re.search(r"(20\d{2})", str(name))
    return match.group(1) if match else None


def normalize_columns(columns: list[str]) -> set[str]:
    return {
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in columns
        if str(column).strip()
    }


def print_manifest(manifest: pd.DataFrame) -> None:
    if manifest.empty:
        LOGGER.warning("No files were discovered from the requested OSF node.")
        return
    display_cols = [
        "name",
        "path",
        "size",
        "download_url",
        "inferred_file_type",
        "looks_like_level",
        "source_node_title",
    ]
    printable = manifest[display_cols].copy()
    printable["size"] = printable["size"].fillna(0).astype("Int64")
    print(printable.to_string(index=False, max_colwidth=80))


def discover_age_manifest(osf_node: str) -> pd.DataFrame:
    ensure_dirs()
    session = requests.Session()
    LOGGER.info("Querying OSF recursively from node %s", osf_node)
    manifest = discover_osf_files_recursive(root_node_id=osf_node, session=session)
    if manifest.empty:
        raise RuntimeError(f"No files were discovered under OSF node {osf_node}.")
    manifest["selected"] = False
    manifest["downloaded"] = False
    manifest["local_archive_path"] = ""
    manifest["extracted_dir"] = ""
    manifest["selected_year"] = ""
    print_manifest(manifest)
    manifest.to_csv(MANIFEST_PATH, index=False)
    return manifest


def _score_candidate(row: pd.Series, years: set[str] | None) -> int:
    name = str(row.get("name", ""))
    node_title = str(row.get("source_node_title", ""))
    level = str(row.get("looks_like_level", "unknown"))
    file_type = str(row.get("inferred_file_type", "unknown"))

    score = 0
    if "raw data" in node_title.lower():
        score += 100
    if level == "trial":
        score += 80
    elif level == "summary":
        score -= 50
    if "touch" in name.lower():
        score -= 25
    if infer_file_type(name) in SUPPORTED_ARCHIVE_TYPES | SUPPORTED_TABULAR_TYPES:
        score += 10
    if name.lower().startswith("age_iat_"):
        score += 15
    file_year = infer_year_from_name(name)
    if years:
        if file_year in years:
            score += 40
        else:
            score -= 20
    return score


def select_best_candidates(manifest: pd.DataFrame, year: str) -> pd.DataFrame:
    if manifest.empty:
        raise RuntimeError("The Age IAT OSF manifest is empty.")

    candidate_mask = manifest["kind"].eq("file") & manifest["download_url"].notna()
    candidate_mask &= manifest["inferred_file_type"].isin(SUPPORTED_ARCHIVE_TYPES | SUPPORTED_TABULAR_TYPES)
    candidates = manifest.loc[candidate_mask].copy()
    if candidates.empty:
        raise RuntimeError("No downloadable tabular or archive files were found in the Age IAT OSF tree.")

    years = None if year.lower() == "all" else {year}
    candidates["selection_score"] = candidates.apply(lambda row: _score_candidate(row, years), axis=1)
    candidates = candidates.sort_values(["selection_score", "size"], ascending=[False, False]).reset_index()

    if years is None:
        selected = candidates[
            candidates["source_node_title"].str.contains("Raw Data", case=False, na=False)
        ].copy()
    else:
        selected = candidates[
            candidates["source_node_title"].str.contains("Raw Data", case=False, na=False)
            & candidates["name"].str.contains(year, na=False)
        ].copy()

    if selected.empty:
        summary_like = candidates[candidates["looks_like_level"].eq("summary")]
        if not summary_like.empty:
            raise RuntimeError(
                "Age IAT discovery did not find a trial-level raw-data archive for the requested year. "
                "The remaining candidates look participant-level/public-summary only, which cannot feed the manuscript's trial-level geometry pipeline."
            )
        raise RuntimeError(
            "Age IAT discovery did not find a usable trial-level raw-data archive for the requested year."
        )

    selected = selected.sort_values(["selection_score", "size"], ascending=[False, False]).reset_index(drop=True)
    return selected


def download_file(session: requests.Session, url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Reusing existing download %s", destination)
        return destination
    response = session.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", "0"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=destination.name,
        leave=False,
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            progress.update(len(chunk))
    return destination


def extract_zip_archive(archive_path: Path, extract_root: Path) -> list[Path]:
    extract_dir = extract_root / archive_path.stem
    if extract_dir.exists() and any(extract_dir.rglob("*")):
        LOGGER.info("Reusing existing extraction %s", extract_dir)
        return [path for path in extract_dir.rglob("*") if path.is_file()]
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []
    with zipfile.ZipFile(archive_path) as zip_handle:
        for member in zip_handle.infolist():
            if member.is_dir():
                continue
            zip_handle.extract(member, extract_dir)
            extracted_paths.append(extract_dir / member.filename)
    return extracted_paths


def detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        header = handle.readline()
    tab_count = header.count("\t")
    comma_count = header.count(",")
    return "\t" if tab_count >= comma_count else ","


def preview_columns(path: Path, file_type: str) -> list[str]:
    if file_type in {"csv", "tsv", "txt"}:
        separator = "\t" if file_type in {"tsv", "txt"} else ","
        if file_type == "txt":
            separator = detect_separator(path)
        df = pd.read_csv(
            path,
            sep=separator,
            nrows=5,
            dtype=str,
            low_memory=False,
            encoding="utf-8-sig",
            encoding_errors="ignore",
        )
        return [str(column).strip() for column in df.columns]
    if file_type == "dta":
        df = pd.read_stata(path, convert_categoricals=False)
        return [str(column).strip() for column in df.columns]
    if file_type == "sav":
        try:
            import pyreadstat
        except ImportError as exc:  # pragma: no cover - exercised through explicit error handling
            raise RuntimeError(
                "Encountered an SPSS .sav file but pyreadstat is not installed. "
                "Install it with `pip install pyreadstat` to enable .sav support."
            ) from exc
        _, meta = pyreadstat.read_sav(path, metadataonly=True)
        return [str(column).strip() for column in meta.column_names]
    return []


def iter_dataframe_chunks(path: Path, file_type: str, chunksize: int = 250_000) -> Iterator[pd.DataFrame]:
    if file_type in {"csv", "tsv", "txt"}:
        separator = "\t" if file_type in {"tsv", "txt"} else ","
        if file_type == "txt":
            separator = detect_separator(path)
        for chunk in pd.read_csv(
            path,
            sep=separator,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
            encoding="utf-8-sig",
            encoding_errors="ignore",
        ):
            yield chunk
        return
    if file_type == "dta":
        df = pd.read_stata(path, convert_categoricals=False)
        yield df.astype("string")
        return
    if file_type == "sav":
        try:
            import pyreadstat
        except ImportError as exc:  # pragma: no cover - exercised through explicit error handling
            raise RuntimeError(
                "Encountered an SPSS .sav file but pyreadstat is not installed. "
                "Install it with `pip install pyreadstat` to enable .sav support."
            ) from exc
        df, _ = pyreadstat.read_sav(path)
        yield df.astype("string")
        return
    raise RuntimeError(f"Unsupported tabular file type: {file_type}")


def append_file_to_parquet(
    writer,
    tabular_path: Path,
    archive_name: str,
    year: str | None,
) -> tuple[object | None, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    file_type = infer_file_type(tabular_path.name)
    rows_written = 0
    for chunk in iter_dataframe_chunks(tabular_path, file_type=file_type):
        chunk = chunk.copy()
        chunk.columns = [str(column).strip() for column in chunk.columns]
        chunk["source_archive"] = archive_name
        chunk["source_member"] = tabular_path.name
        chunk["source_year"] = year
        chunk = chunk.astype("string")
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(RAW_PARQUET_PATH), table.schema, compression="snappy")
        else:
            writer_names = writer.schema.names
            chunk_names = list(chunk.columns)
            extra = [name for name in chunk_names if name not in writer_names]
            missing = [name for name in writer_names if name not in chunk_names]
            if extra:
                raise RuntimeError(
                    f"Unexpected columns {extra} in {tabular_path.name}; existing parquet schema is {writer_names}."
                )
            for name in missing:
                chunk[name] = pd.NA
            chunk = chunk[writer_names]
            table = pa.Table.from_pandas(chunk, preserve_index=False).cast(writer.schema)
        writer.write_table(table)
        rows_written += len(chunk)
    return writer, rows_written


def build_trial_level_raw_parquet(selected_files: list[dict[str, str]]) -> dict[str, object]:
    import pyarrow.parquet as pq

    if RAW_PARQUET_PATH.exists():
        RAW_PARQUET_PATH.unlink()

    writer = None
    downloaded_members: list[dict[str, str]] = []
    total_rows = 0
    try:
        for selected in selected_files:
            archive_path = Path(selected["local_archive_path"])
            archive_name = archive_path.name
            extracted_dir = Path(selected["extracted_dir"])
            tabular_files = sorted(
                path
                for path in extracted_dir.rglob("*")
                if path.is_file()
                and not any(part.startswith("__MACOSX") for part in path.parts)
                and not path.name.startswith(".")
                and not path.name.startswith("._")
            )
            trial_candidates: list[Path] = []
            summary_candidates: list[Path] = []
            for tabular_path in tabular_files:
                file_type = infer_file_type(tabular_path.name)
                if file_type not in SUPPORTED_TABULAR_TYPES:
                    continue
                try:
                    columns = preview_columns(tabular_path, file_type=file_type)
                except Exception as exc:
                    LOGGER.warning("Skipping %s during preview: %s", tabular_path.name, exc)
                    selected.setdefault("member_level_reasons", {})[tabular_path.name] = f"preview failed: {exc}"
                    continue
                level, reason = classify_file_level(
                    name=tabular_path.name,
                    path=str(tabular_path),
                    node_title=selected["source_node_title"],
                    columns=columns,
                )
                normalized_columns = normalize_columns(columns)
                has_required_columns = REQUIRED_TRIAL_COLUMNS.issubset(normalized_columns)
                if level == "trial" and has_required_columns:
                    trial_candidates.append(tabular_path)
                elif level == "summary":
                    summary_candidates.append(tabular_path)
                else:
                    LOGGER.info(
                        "Ignoring %s because required trial columns were not found. Columns=%s",
                        tabular_path.name,
                        columns[:10],
                    )
                suffix = ""
                if not has_required_columns:
                    suffix = f"; missing required columns {sorted(REQUIRED_TRIAL_COLUMNS - normalized_columns)}"
                selected.setdefault("member_level_reasons", {})[tabular_path.name] = reason + suffix

            if not trial_candidates:
                if summary_candidates:
                    raise RuntimeError(
                        "The selected Age IAT archive expanded into participant-level summary tables only. "
                        "This manuscript pipeline requires trial-level files with session/block/trial/latency fields."
                    )
                raise RuntimeError(
                    f"No supported trial-level tables were found inside {archive_name}."
                )

            for tabular_path in trial_candidates:
                writer, rows_written = append_file_to_parquet(
                    writer=writer,
                    tabular_path=tabular_path,
                    archive_name=archive_name,
                    year=selected.get("selected_year"),
                )
                total_rows += rows_written
                downloaded_members.append(
                    {
                        "archive": archive_name,
                        "member": tabular_path.name,
                        "path": str(tabular_path),
                    }
                )
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0 or not RAW_PARQUET_PATH.exists():
        raise RuntimeError("No trial-level rows were written to the interim Age IAT parquet.")

    return {
        "raw_parquet_path": str(RAW_PARQUET_PATH),
        "trial_members": downloaded_members,
        "trial_rows_written": total_rows,
    }


def discover_and_download_age_iat(osf_node: str, year: str) -> dict[str, object]:
    manifest = discover_age_manifest(osf_node)
    session = requests.Session()

    selected = select_best_candidates(manifest, year=year)
    LOGGER.info("Selected %d Age IAT archive(s) for year=%s", len(selected), year)

    selected_files: list[dict[str, str]] = []
    archives_dir = RAW_DIR / "archives"
    extracted_root = RAW_DIR / "extracted"

    for _, row in selected.iterrows():
        archive_name = str(row["name"])
        archive_path = archives_dir / archive_name
        LOGGER.info("Downloading %s", archive_name)
        download_file(session, str(row["download_url"]), archive_path)
        if infer_file_type(archive_name) in SUPPORTED_ARCHIVE_TYPES:
            extract_zip_archive(archive_path, extracted_root)
            extracted_dir = extracted_root / archive_path.stem
        else:
            extracted_dir = archive_path.parent
        manifest.loc[manifest["name"].eq(archive_name), "selected"] = True
        manifest.loc[manifest["name"].eq(archive_name), "downloaded"] = True
        manifest.loc[manifest["name"].eq(archive_name), "local_archive_path"] = str(archive_path)
        manifest.loc[manifest["name"].eq(archive_name), "extracted_dir"] = str(extracted_dir)
        manifest.loc[manifest["name"].eq(archive_name), "selected_year"] = year
        selected_files.append(
            {
                "name": archive_name,
                "source_node_id": str(row["source_node_id"]),
                "source_node_title": str(row["source_node_title"]),
                "download_url": str(row["download_url"]),
                "local_archive_path": str(archive_path),
                "extracted_dir": str(extracted_dir),
                "selected_year": year,
            }
        )

    manifest.to_csv(MANIFEST_PATH, index=False)
    parquet_info = build_trial_level_raw_parquet(selected_files=selected_files)

    metadata = {
        "domain": "Age",
        "requested_osf_node": osf_node,
        "selected_year": year,
        "manifest_path": str(MANIFEST_PATH),
        "selected_files": selected_files,
        **parquet_info,
    }
    DOWNLOAD_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved manifest to %s", MANIFEST_PATH)
    LOGGER.info("Saved trial-level raw parquet to %s", RAW_PARQUET_PATH)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Age IAT trial-level data from the public OSF API.")
    parser.add_argument("--osf-node", default=DEFAULT_OSF_NODE, help="Root OSF node to recurse from.")
    parser.add_argument(
        "--year",
        default=DEFAULT_YEAR,
        help="Year-specific raw archive to download (for example 2019), or 'all'.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only discover and save the OSF manifest without downloading any archives.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    if args.manifest_only:
        manifest = discover_age_manifest(osf_node=args.osf_node)
        return {
            "manifest_path": str(MANIFEST_PATH),
            "discovered_files": int(len(manifest)),
        }
    return discover_and_download_age_iat(osf_node=args.osf_node, year=str(args.year))


if __name__ == "__main__":
    main()
