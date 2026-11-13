import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import check_commons_licences as licences
from scripts.check_commons_licences import (
    Row,
    api_url,
    check,
    commons_title,
    is_open_licence,
    is_owner_release,
    licence_short_name,
    parse_attribution,
)

TABLE = """# Motion attribution

| Motion | Source | Licence | Author | Trim start (s) | Trim end (s) | Pet |
| --- | --- | --- | --- | ---: | ---: | --- |
| dog_walk | [c](https://commons.wikimedia.org/wiki/File:Dog.webm) | CC BY 4.0 | A | 1.5 | 6.5 | |
| cat_trot | private submission | owner release CR-2 | B. Owner | 0 | 5 | Miso |
"""


OWNER_TABLE = """| Motion | Source | Licence | Author | Trim start (s) | Trim end (s) | Pet |
| --- | --- | --- | --- | ---: | ---: | --- |
| cat_trot | private submission | owner release CR-2 | B. Owner | 0 | 5 | Miso |
"""


def _row(motion: str, source: str, licence: str = "CC BY 4.0") -> Row:
    return Row(motion, source, licence, "Author", 0.0, 5.0, "")


def _response(licence: str | None) -> str:
    info: list[object] = []
    if licence is not None:
        info = [{"extmetadata": {"LicenseShortName": {"value": licence, "source": "commons"}}}]
    return json.dumps({"query": {"pages": [{"title": "File:X.webm", "imageinfo": info}]}})


def test_table_rows_are_read_without_the_header_or_rule() -> None:
    assert parse_attribution(TABLE) == [
        Row(
            "dog_walk",
            "https://commons.wikimedia.org/wiki/File:Dog.webm",
            "CC BY 4.0",
            "A",
            1.5,
            6.5,
            "",
        ),
        Row("cat_trot", "private submission", "owner release CR-2", "B. Owner", 0.0, 5.0, "Miso"),
    ]


def test_a_row_with_the_wrong_number_of_cells_is_refused() -> None:
    with pytest.raises(ValueError, match="needs 7 cells, found 3"):
        parse_attribution("| dog_walk | https://example.org | CC0 |\n")


def test_trim_times_must_be_numbers() -> None:
    with pytest.raises(ValueError, match="seconds"):
        parse_attribution("| dog_walk | x | CC0 | A | soon | 5 | |\n")


@pytest.mark.parametrize(
    "licence", ["CC0", "CC0 1.0", "CC BY 4.0", "CC BY 2.5", "CC-BY-4.0", "CC BY"]
)
def test_open_licences_pass(licence: str) -> None:
    assert is_open_licence(licence)


@pytest.mark.parametrize(
    "licence", ["CC BY-SA 4.0", "CC BY-NC 4.0", "CC BY-ND 3.0", "Public domain", "GFDL", ""]
)
def test_other_licences_fail(licence: str) -> None:
    assert not is_open_licence(licence)


def test_owner_release_needs_an_identifier() -> None:
    assert is_owner_release("owner release CR-2")
    assert not is_owner_release("owner release")
    assert not is_owner_release("CC0")


@pytest.mark.parametrize(
    ("url", "title"),
    [
        ("https://commons.wikimedia.org/wiki/File:Dog_walk.webm", "File:Dog walk.webm"),
        ("https://commons.wikimedia.org/wiki/File:Caf%C3%A9.ogv", "File:Café.ogv"),
        ("https://commons.wikimedia.org/w/index.php?title=File:A_b.ogv&oldid=2", "File:A b.ogv"),
        ("https://commons.wikimedia.org/wiki/Category:Dogs", None),
        ("https://example.org/wiki/File:Dog.webm", None),
        ("private submission", None),
    ],
)
def test_commons_title(url: str, title: str | None) -> None:
    assert commons_title(url) == title


def test_api_query_asks_for_extension_metadata_of_the_file() -> None:
    parts = urlsplit(api_url("File:Dog walk.webm"))

    assert (parts.scheme, parts.netloc, parts.path) == (
        "https",
        "commons.wikimedia.org",
        "/w/api.php",
    )
    query = parse_qs(parts.query)
    assert query["action"] == ["query"]
    assert query["titles"] == ["File:Dog walk.webm"]
    assert query["prop"] == ["imageinfo"]
    assert query["iiprop"] == ["extmetadata"]


def test_licence_short_name_is_read_from_the_response() -> None:
    assert licence_short_name(_response("CC BY-SA 3.0"), "File:X.webm") == "CC BY-SA 3.0"


def test_missing_licence_is_reported() -> None:
    with pytest.raises(ValueError, match=r"no licence for File:X\.webm"):
        licence_short_name(_response(None), "File:X.webm")


def test_check_flags_only_commons_sources_that_are_not_open() -> None:
    requested: list[str] = []

    def fetch(url: str) -> str:
        requested.append(url)
        return _response("CC BY-SA 4.0" if "Sa%20dog" in url else "CC BY 4.0")

    rows = [
        _row("dog_walk", "https://commons.wikimedia.org/wiki/File:Dog_walk.webm"),
        _row("dog_trot", "https://commons.wikimedia.org/wiki/File:Sa_dog.webm"),
        _row("cat_trot", "private submission", "owner release CR-2"),
    ]

    assert check(rows, fetch) == ["dog_trot: File:Sa dog.webm is CC BY-SA 4.0, not CC0 or CC BY"]
    assert len(requested) == 2


def test_check_reports_a_file_with_no_licence() -> None:
    rows = [_row("dog_walk", "https://commons.wikimedia.org/wiki/File:Dog_walk.webm")]

    def fetch(url: str) -> str:
        del url
        return _response(None)

    assert check(rows, fetch) == ["dog_walk: Commons reports no licence for File:Dog walk.webm"]


def _main(monkeypatch: pytest.MonkeyPatch, path: Path) -> int:
    monkeypatch.setattr(sys, "argv", ["check_commons_licences.py", "--attribution", str(path)])
    return licences.main()


def test_table_without_commons_sources_passes_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "ATTRIBUTION.md"
    path.write_text(OWNER_TABLE, encoding="utf-8")

    assert _main(monkeypatch, path) == 0


def test_missing_table_is_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _main(monkeypatch, tmp_path / "absent.md") == 2
