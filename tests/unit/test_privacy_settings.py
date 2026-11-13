import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from copypaw import models

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("docs/site/index.html", "web/index.html")
STYLESHEETS = ("docs/site/style.css", "web/src/style.css")
BROWSER_SCRIPTS = ("docs/site/grid.js",)
SCRIPT_FOLDER = "web/src"
# Text-only links can name any host; every tag that makes the browser fetch something cannot.
LINK_TAGS = frozenset({"a"})
FETCHING_ATTRIBUTES = frozenset({"src", "href", "srcset", "poster", "data", "action"})
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
STORAGE_AND_BEACONS = re.compile(
    r"document\.cookie|localStorage|sessionStorage|indexedDB|sendBeacon"
    r"|XMLHttpRequest|WebSocket|EventSource"
)
ABSOLUTE_URL = re.compile(r"""["'`](?:https?:)?//[^"'`\s]+""")
CSS_EXTERNAL = re.compile(r"@import|url\(\s*[\"']?(?:https?:)?//", re.IGNORECASE)
MODEL_HOSTS = frozenset({"huggingface.co", "github.com"})


class _Fetches(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.external: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in LINK_TAGS:
            return
        for name, value in attrs:
            if name in FETCHING_ATTRIBUTES and value and _is_external(value):
                self.external.append(f"<{tag} {name}={value}>")


def _is_external(reference: str) -> bool:
    return bool(urlsplit(reference.strip()).netloc)


def external_fetches(html: str) -> list[str]:
    """Name every tag that would make a browser contact another host."""
    parser = _Fetches()
    parser.feed(html)
    return parser.external


def script_leaks(source: str) -> list[str]:
    """Name each storage, cookie or network-push API and each absolute address in a script."""
    found = STORAGE_AND_BEACONS.findall(source)
    found.extend(
        match.group(0).strip("\"'`")
        for match in ABSOLUTE_URL.finditer(source)
        if match.group(0).strip("\"'`") != SVG_NAMESPACE
    )
    return found


def browser_sources() -> list[Path]:
    """List the scripts that ship in the page, leaving out tests."""
    folder = ROOT / SCRIPT_FOLDER
    typed = [path for path in sorted(folder.rglob("*.ts")) if ".test." not in path.name]
    return [ROOT / name for name in BROWSER_SCRIPTS] + typed


def test_pages_load_nothing_from_another_host() -> None:
    for name in PAGES:
        html = (ROOT / name).read_text(encoding="utf-8")

        assert external_fetches(html) == [], name


def test_stylesheets_import_and_fetch_nothing_external() -> None:
    for name in STYLESHEETS:
        css = (ROOT / name).read_text(encoding="utf-8")

        assert CSS_EXTERNAL.search(css) is None, name


def test_browser_scripts_keep_data_in_the_tab() -> None:
    sources = browser_sources()

    assert len(sources) > 1
    assert {
        path.relative_to(ROOT).as_posix(): script_leaks(path.read_text(encoding="utf-8"))
        for path in sources
    } == {path.relative_to(ROOT).as_posix(): [] for path in sources}


def test_model_downloads_name_only_the_documented_hosts() -> None:
    for model in models.REGISTRY.values():
        address = urlsplit(model.url)

        assert address.scheme == "https"
        assert address.hostname in MODEL_HOSTS
        assert address.query == ""
        assert address.fragment == ""


def test_external_fetches_are_found_and_plain_links_are_not() -> None:
    html = (
        '<a href="https://example.org/docs">docs</a>'
        '<script src="https://cdn.example.org/a.js"></script>'
        '<link rel="stylesheet" href="//fonts.example.org/f.css">'
        '<img src="local.png"><script type="module" src="/src/main.ts"></script>'
    )

    assert external_fetches(html) == [
        "<script src=https://cdn.example.org/a.js>",
        "<link href=//fonts.example.org/f.css>",
    ]


def test_script_leaks_name_storage_beacons_and_addresses() -> None:
    source = (
        f'const ns = "{SVG_NAMESPACE}";\n'
        "document.cookie = 'a=1';\n"
        "navigator.sendBeacon('https://stats.example.org/hit');\n"
        "fetch('/api/health');\n"
    )

    assert script_leaks(source) == [
        "document.cookie",
        "sendBeacon",
        "https://stats.example.org/hit",
    ]


def test_stylesheet_pattern_catches_imports_and_remote_fonts() -> None:
    assert CSS_EXTERNAL.search("@import url(x.css);")
    assert CSS_EXTERNAL.search("src: url('https://fonts.example.org/a.woff2');")
    assert CSS_EXTERNAL.search("src: url(//fonts.example.org/a.woff2);")
    assert CSS_EXTERNAL.search("src: url(fonts/a.woff2);") is None
