from __future__ import annotations

from collections import deque
from typing import Any, Iterable
import re

import pandas as pd
import requests


OSF_API_ROOT = "https://api.osf.io/v2"

TRIAL_NAME_HINTS = {
    "raw data",
    "raw",
    "trial",
    "trials",
    "iat.txt",
}
SUMMARY_NAME_HINTS = {
    "public",
    "codebook",
    "summary",
    "score",
    "scores",
    "dscore",
}
TRIAL_COLUMN_HINTS = {
    "session_id",
    "sessionid",
    "participant_id",
    "participantid",
    "user_id",
    "user_id_num",
    "respondent_id",
    "block",
    "block_number",
    "blocknum",
    "trial",
    "trial_number",
    "trialnum",
    "trial_latency",
    "trial_error",
    "latency",
    "latency_ms",
    "rt",
    "response_time",
}
SUMMARY_COLUMN_HINTS = {
    "d_biep",
    "dscore",
    "mn_rt_all_3467",
    "pct_error_3467",
    "sd_all_3",
    "sd_all_4",
    "sd_all_6",
    "sd_all_7",
    "mn_rt_correct_3",
    "mn_rt_correct_4",
    "mn_rt_correct_6",
    "mn_rt_correct_7",
    "n_3467",
    "n_error_3",
    "n_error_4",
    "n_error_6",
    "n_error_7",
}


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def infer_file_type(name: str) -> str:
    lower_name = str(name).lower()
    if lower_name.endswith(".csv.zip"):
        return "csv.zip"
    if lower_name.endswith(".sav.zip"):
        return "sav.zip"
    if lower_name.endswith(".tsv.zip"):
        return "tsv.zip"
    if lower_name.endswith(".txt.zip"):
        return "txt.zip"
    parts = lower_name.rsplit(".", 1)
    return parts[-1] if len(parts) == 2 else "unknown"


def classify_file_level(
    *,
    name: str,
    path: str = "",
    node_title: str = "",
    member_names: Iterable[str] | None = None,
    columns: Iterable[str] | None = None,
) -> tuple[str, str]:
    text_chunks = [name, path, node_title]
    normalized_columns = {normalize_token(col) for col in (columns or []) if str(col).strip()}
    normalized_members = {normalize_token(member) for member in (member_names or []) if str(member).strip()}
    normalized_text = " ".join(normalize_token(chunk) for chunk in text_chunks if str(chunk).strip())

    trial_score = 0
    summary_score = 0
    reasons: list[str] = []

    if "raw_data" in normalize_token(node_title):
        trial_score += 5
        reasons.append("OSF node is labeled Raw Data")
    if "touch" in normalized_text:
        summary_score += 1
        reasons.append("touch-screen archive is deprioritized")

    if any(hint in normalized_text for hint in {normalize_token(v) for v in TRIAL_NAME_HINTS}):
        trial_score += 3
        reasons.append("filename/path looks raw or trial-level")
    if any(hint in normalized_text for hint in {normalize_token(v) for v in SUMMARY_NAME_HINTS}):
        summary_score += 2
        reasons.append("filename/path looks public-summary oriented")

    if any("iat_txt" in member or member.endswith("iat_txt") for member in normalized_members):
        trial_score += 5
        reasons.append("archive contains iat.txt")
    if any(member.endswith(("_csv", "_sav", "_dta")) and "public" in member for member in normalized_members):
        summary_score += 3
        reasons.append("archive contains public summary export")

    trial_col_hits = 0
    summary_col_hits = 0
    for col in normalized_columns:
        if col in TRIAL_COLUMN_HINTS or col.startswith("trial_") or col.startswith("block_"):
            trial_col_hits += 1
        if any(col.startswith(prefix) for prefix in SUMMARY_COLUMN_HINTS):
            summary_col_hits += 1
    if trial_col_hits:
        trial_score += 6 + trial_col_hits
        reasons.append(f"tabular columns include {trial_col_hits} trial-level hints")
    if summary_col_hits:
        summary_score += 6 + summary_col_hits
        reasons.append(f"tabular columns include {summary_col_hits} summary-level hints")

    if trial_score > summary_score and trial_score > 0:
        return "trial", "; ".join(reasons) or "trial-level heuristic matched"
    if summary_score > trial_score and summary_score > 0:
        return "summary", "; ".join(reasons) or "summary-level heuristic matched"
    return "unknown", "; ".join(reasons) or "insufficient evidence"


def _get_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def iter_paginated_items(session: requests.Session, url: str) -> Iterable[dict[str, Any]]:
    next_url = url
    while next_url:
        payload = _get_json(session, next_url)
        for item in payload.get("data", []):
            yield item
        next_url = payload.get("links", {}).get("next")


def get_node_title(session: requests.Session, node_id: str) -> str:
    payload = _get_json(session, f"{OSF_API_ROOT}/nodes/{node_id}/")
    return str(payload.get("data", {}).get("attributes", {}).get("title", node_id))


def list_osfstorage_files_for_node(
    session: requests.Session,
    node_id: str,
    node_title: str,
) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    folder_queue = deque([f"{OSF_API_ROOT}/nodes/{node_id}/files/osfstorage/"])
    visited_urls: set[str] = set()

    while folder_queue:
        folder_url = folder_queue.popleft()
        if folder_url in visited_urls:
            continue
        visited_urls.add(folder_url)
        for item in iter_paginated_items(session, folder_url):
            attributes = item.get("attributes", {})
            relationships = item.get("relationships", {})
            links = item.get("links", {})
            name = str(attributes.get("name", ""))
            path = str(attributes.get("materialized_path", attributes.get("path", "")))
            record = {
                "source_node_id": node_id,
                "source_node_title": node_title,
                "name": name,
                "path": path,
                "size": attributes.get("size"),
                "kind": attributes.get("kind"),
                "download_url": links.get("download"),
                "info_url": links.get("info"),
                "html_url": links.get("html"),
                "inferred_file_type": infer_file_type(name),
            }
            level, reason = classify_file_level(name=name, path=path, node_title=node_title)
            record["looks_like_level"] = level
            record["level_reason"] = reason
            discovered.append(record)

            if attributes.get("kind") == "folder":
                child_url = (
                    relationships.get("files", {})
                    .get("links", {})
                    .get("related", {})
                    .get("href")
                )
                if child_url:
                    folder_queue.append(child_url)
    return discovered


def discover_osf_files_recursive(root_node_id: str, session: requests.Session | None = None) -> pd.DataFrame:
    own_session = session or requests.Session()
    root_title = get_node_title(own_session, root_node_id)
    node_queue = deque([(root_node_id, root_title)])
    seen_nodes: set[str] = set()
    records: list[dict[str, Any]] = []

    while node_queue:
        node_id, node_title = node_queue.popleft()
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        records.extend(list_osfstorage_files_for_node(own_session, node_id=node_id, node_title=node_title))

        children_url = f"{OSF_API_ROOT}/nodes/{node_id}/children/"
        for child in iter_paginated_items(own_session, children_url):
            child_id = str(child.get("id"))
            child_title = str(child.get("attributes", {}).get("title", child_id))
            node_queue.append((child_id, child_title))

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        return manifest
    manifest = manifest.sort_values(
        ["source_node_title", "path", "name"],
        na_position="last",
    ).reset_index(drop=True)
    return manifest
