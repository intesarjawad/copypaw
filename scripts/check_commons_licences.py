"""Fail unless every Wikimedia Commons motion source is CC0 or CC BY."""

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION = ROOT / "motions/ATTRIBUTION.md"
COMMONS_HOST = "commons.wikimedia.org"
API_URL = f"https://{COMMONS_HOST}/w/api.php"
USER_AGENT = "copypaw-licence-check (https://github.com/intesarjawad/copypaw)"
_TIMEOUT_S = 30
_CELLS = 7
_OWNER_RELEASE = re.compile(r"owner release \S+")
_OPEN_LICENCE = re.compile(r"CC0(?: 1\.0)?|CC[ -]BY(?:[ -][0-9]\.[0-9])?")
_URL = re.compile(r"https?://[^\s)>|]+")

type Fetch = Callable[[str], str]


@dataclass(frozen=True)
class Row:
    """One motion's row in the attribution table."""

    motion: str
    source: str
    licence: str
    author: str
    trim_start_s: float
    trim_end_s: float
    pet_name: str


def is_open_licence(licence: str) -> bool:
    """Accept CC0 and CC BY of any version; share-alike and non-commercial fail."""
    return _OPEN_LICENCE.fullmatch(licence.strip()) is not None


def is_owner_release(licence: str) -> bool:
    """Accept a licence cell naming an owner release identifier."""
    return _OWNER_RELEASE.fullmatch(licence.strip()) is not None


def parse_attribution(text: str) -> list[Row]:
    """Read the table rows of an attribution file, skipping the header and rule."""
    rows: list[Row] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: ") or cells[0].lower() == "motion":
            continue
        if len(cells) != _CELLS:
            raise ValueError(f"Attribution row needs {_CELLS} cells, found {len(cells)}: {line}")
        try:
            start, end = float(cells[4]), float(cells[5])
        except ValueError as error:
            raise ValueError(f"Trim start and end must be seconds: {line}") from error
        url = _URL.search(cells[1])
        rows.append(
            Row(
                motion=cells[0],
                source=url.group(0) if url else cells[1],
                licence=cells[2],
                author=cells[3],
                trim_start_s=start,
                trim_end_s=end,
                pet_name=cells[6],
            )
        )
    return rows


def commons_title(url: str) -> str | None:
    """Return the File: title of a Commons file page URL, or None for any other URL."""
    parts = urlsplit(url)
    if parts.hostname != COMMONS_HOST:
        return None
    title = parse_qs(parts.query).get("title", [None])[0]
    if title is None and parts.path.startswith("/wiki/"):
        title = unquote(parts.path.removeprefix("/wiki/"))
    if title is None or not title.startswith("File:"):
        return None
    return title.replace("_", " ")


def api_url(title: str) -> str:
    """Build the extmetadata query for one file title."""
    query = urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "extmetadata",
            "format": "json",
            "formatversion": "2",
        },
        quote_via=quote,
    )
    return f"{API_URL}?{query}"


def fetch_json(url: str) -> str:
    """Download a Commons API response."""
    if urlsplit(url).scheme != "https":
        raise ValueError(f"Commons API needs https: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.build_opener().open(request, timeout=_TIMEOUT_S) as response:
        return cast("bytes", response.read()).decode("utf-8")


def licence_short_name(response: str, title: str) -> str:
    """Pull LicenseShortName out of an extmetadata response."""
    data = cast("dict[str, object]", json.loads(response))
    query = cast("dict[str, object]", data.get("query", {}))
    for page in cast("list[dict[str, object]]", query.get("pages", [])):
        info = cast("list[dict[str, object]]", page.get("imageinfo", []))
        if not info:
            continue
        meta = cast("dict[str, dict[str, object]]", info[0].get("extmetadata", {}))
        value = meta.get("LicenseShortName", {}).get("value")
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"Commons reports no licence for {title}")


def check(rows: list[Row], fetch: Fetch = fetch_json) -> list[str]:
    """Return one problem line for every Commons source whose licence is not open."""
    problems: list[str] = []
    for row in rows:
        title = commons_title(row.source)
        if title is None:
            continue
        try:
            licence = licence_short_name(fetch(api_url(title)), title)
        except ValueError as error:
            problems.append(f"{row.motion}: {error}")
            continue
        print(f"{row.motion}: {licence}")
        if not is_open_licence(licence):
            problems.append(f"{row.motion}: {title} is {licence}, not CC0 or CC BY")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", type=Path, default=ATTRIBUTION)
    args = parser.parse_args()
    try:
        rows = parse_attribution(args.attribution.read_text(encoding="utf-8"))
        problems = check(rows)
    except (OSError, HTTPException, ValueError) as error:
        print(f"Licence check failed: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
