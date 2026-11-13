"""The field-test protocol must be registered before the first clip arrives.

The pre-registration is the commit that gave the protocol, the text outside the
results block, its current wording. Editing the protocol after a clip arrived
moves that commit past the arrival, which fails here.
"""

import csv
import re
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/field_test.md"
CLIP_TABLE = "docs/field_test.csv"
RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"
DISCLOSURE_FILES = (
    "docs/consent/artwork_release.md",
    "docs/consent/clip_release.md",
    "docs/call-for-clips.md",
)
DISCLOSURE_HEADING = "## Dropbox disclosure\n"
PRIVACY_LINK = "https://dropbox.com/privacy"


def protocol_text(document: str) -> str:
    """Return the document without the contents of its results block."""
    head, start, rest = document.partition(RESULTS_START)
    _, end, tail = rest.partition(RESULTS_END)
    if not (start and end) or RESULTS_START in tail or RESULTS_END in tail:
        raise ValueError(f"Expected one {RESULTS_START} ... {RESULTS_END} block")
    return head + tail


def registration_date(history: list[tuple[date, str]], current: str) -> date | None:
    """Date the current protocol was committed, or None while it has uncommitted edits.

    history lists (commit date, document) for every commit touching the file, newest first.
    """
    registered = None
    for committed_on, document in history:
        if protocol_text(document) != protocol_text(current):
            break
        registered = committed_on
    return registered


def committed_versions() -> list[tuple[date, str]]:
    """List the commit date and text of every committed version of the protocol."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")

    def run(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            [git, *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout

    commits = [line.split(",") for line in run("log", "--format=%H,%cI", "--", PROTOCOL).split()]
    return [
        (datetime.fromisoformat(committed_at).date(), run("show", f"{commit}:{PROTOCOL}"))
        for commit, committed_at in commits
    ]


def disclosure(document: str) -> str:
    """Return the Dropbox disclosure section, up to the next heading."""
    _, heading, rest = document.partition(DISCLOSURE_HEADING)
    if not heading:
        raise ValueError(f"Missing {DISCLOSURE_HEADING.strip()!r}")
    return re.split(r"^## ", rest, maxsplit=1, flags=re.MULTILINE)[0]


def test_no_clip_arrived_before_the_protocol_was_registered() -> None:
    registered = registration_date(committed_versions(), (ROOT / PROTOCOL).read_text("utf-8"))
    with (ROOT / CLIP_TABLE).open(newline="", encoding="utf-8") as handle:
        arrivals = [date.fromisoformat(row["arrival_date"]) for row in csv.DictReader(handle)]

    early = [arrival for arrival in arrivals if registered is None or arrival < registered]

    assert early == [], (
        f"The protocol in {PROTOCOL} changed after clips arrived (registered {registered}); "
        "restore its pre-registered text with git"
    )


@pytest.mark.parametrize("name", DISCLOSURE_FILES)
def test_disclosure_names_dropbox_and_links_its_privacy_policy(name: str) -> None:
    section = disclosure((ROOT / name).read_text(encoding="utf-8"))

    assert "Dropbox" in section
    assert PRIVACY_LINK in section


def test_disclosure_is_word_for_word_the_same_everywhere() -> None:
    sections = {disclosure((ROOT / name).read_text(encoding="utf-8")) for name in DISCLOSURE_FILES}

    assert len(sections) == 1


def _document(protocol: str, results: str = "") -> str:
    return f"# Protocol\n\n{protocol}\n\n{RESULTS_START}\n{results}{RESULTS_END}\n"


def test_results_do_not_move_the_registration() -> None:
    history = [
        (date(2026, 12, 3), _document("Twenty clips.", "kappa 0.8\n")),
        (date(2026, 11, 5), _document("Twenty clips.")),
    ]

    assert registration_date(history, _document("Twenty clips.", "kappa 0.9\n")) == date(
        2026, 11, 5
    )


def test_a_protocol_edit_moves_the_registration_to_the_edit() -> None:
    history = [
        (date(2026, 12, 3), _document("Thirty clips.")),
        (date(2026, 11, 5), _document("Twenty clips.")),
    ]

    assert registration_date(history, _document("Thirty clips.")) == date(2026, 12, 3)


def test_an_uncommitted_protocol_edit_has_no_registration() -> None:
    history = [(date(2026, 11, 5), _document("Twenty clips."))]

    assert registration_date(history, _document("Thirty clips.")) is None


@pytest.mark.parametrize(
    "document",
    [
        "# Protocol\n",
        f"# Protocol\n{RESULTS_END}\n{RESULTS_START}\n",
        _document("Twenty clips.") + RESULTS_START + "\n",
    ],
)
def test_a_missing_or_repeated_results_block_is_rejected(document: str) -> None:
    with pytest.raises(ValueError, match="Expected one"):
        protocol_text(document)
