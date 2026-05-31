from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests


BASE_DIR = Path(__file__).resolve().parents[1]
FILES_TO_SCAN = [
    BASE_DIR / "README.md",
    BASE_DIR / "DATA_AVAILABILITY.md",
    BASE_DIR / "paper" / "main_new.tex",
    BASE_DIR / ".zenodo.json",
    BASE_DIR / "data_sources.yaml",
]
REQUIRED_DATASET_URLS = {
    "https://osf.io/cfvyj/",
    "https://osf.io/5s9ty/",
    "https://osf.io/9jvmk/",
    "https://osf.io/download/7gb96/",
    "https://osf.io/download/79ch3/",
    "https://osf.io/download/34wsk/",
}
URL_RE = re.compile(r"https?://[^\s\\{}\[\]()<>\"]+")
BAD_STATUSES = {403, 404}


@dataclass
class LinkResult:
    url: str
    status: str
    redirect_target: str
    ok: bool


def normalize_url(raw_url: str) -> str:
    return raw_url.rstrip(".,;:)]}'\"")


def extract_urls() -> list[str]:
    urls: set[str] = set()
    for path in FILES_TO_SCAN:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        urls.update(normalize_url(match.group(0)) for match in URL_RE.finditer(text))
    return sorted(url for url in urls if interesting_url(url))


def interesting_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return any(token in host for token in ("osf.io", "zenodo.org", "github.com", "doi.org"))


def osf_node_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "osf.io":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) == 1 and re.fullmatch(r"[A-Za-z0-9]{5}", parts[0]):
        return parts[0]
    return None


def check_osf_node(session: requests.Session, url: str, node_id: str) -> LinkResult:
    api_url = f"https://api.osf.io/v2/nodes/{node_id}/"
    try:
        response = session.get(api_url, timeout=30)
        redirect = url
        ok = response.status_code < 400 and response.json().get("data", {}).get("attributes", {}).get("public") is True
        return LinkResult(url=url, status=str(response.status_code), redirect_target=redirect, ok=ok)
    except Exception as exc:
        return LinkResult(url=url, status=f"ERROR {type(exc).__name__}", redirect_target="", ok=False)


def request_with_retries(session: requests.Session, url: str, retries: int = 3) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.head(url, allow_redirects=True, timeout=30)
            if response.status_code in {405} or response.status_code >= 500:
                response = session.get(url, stream=True, allow_redirects=True, timeout=30)
                response.close()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def check_url(session: requests.Session, url: str) -> LinkResult:
    node_id = osf_node_id(url)
    if node_id is not None:
        return check_osf_node(session, url, node_id)

    try:
        response = request_with_retries(session, url)
        status = response.status_code
        ok = status < 400 and status not in BAD_STATUSES
        return LinkResult(url=url, status=str(status), redirect_target=response.url, ok=ok)
    except Exception as exc:
        return LinkResult(url=url, status=f"ERROR {type(exc).__name__}", redirect_target="", ok=False)


def main() -> int:
    urls = set(extract_urls())
    missing_required = sorted(REQUIRED_DATASET_URLS - urls)
    urls.update(REQUIRED_DATASET_URLS)

    session = requests.Session()
    results = [check_url(session, url) for url in sorted(urls)]

    print("URL | HTTP status | redirect target | pass/fail")
    print("--- | --- | --- | ---")
    for result in results:
        print(f"{result.url} | {result.status} | {result.redirect_target} | {'PASS' if result.ok else 'FAIL'}")

    if missing_required:
        print("\nMissing required dataset URLs from scanned files:")
        print(json.dumps(missing_required, indent=2))

    failures = [result for result in results if not result.ok]
    return 1 if failures or missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
