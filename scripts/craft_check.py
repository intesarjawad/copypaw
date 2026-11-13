"""Check that the web page's colour pairs meet the WCAG contrast minimums."""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "web/src/style.css"
TEXT_MINIMUM = 4.5
NON_TEXT_MINIMUM = 3.0
_RULE = re.compile(r"(?P<selector>[^{}]+)\{(?P<body>[^{}]*)\}")
_HEX = re.compile(r"#(?P<digits>[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_VAR = re.compile(r"var\(\s*(?P<name>--[\w-]+)\s*\)")
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SHORT_HEX_DIGITS = 3
_HEX_BASE = 16
_LINEAR_KNEE = 0.04045
_LINEAR_SLOPE = 12.92
_GAMMA_OFFSET = 0.055
_GAMMA = 2.4
_LUMA = (0.2126, 0.7152, 0.0722)
_FLARE = 0.05
_MAX_CHANNEL = 255.0

type Colour = tuple[int, int, int]
type Source = tuple[str, str]


@dataclass(frozen=True)
class Pair:
    """A foreground and a background, each named by CSS selector and property."""

    name: str
    foreground: Source
    background: Source
    minimum: float


_PAGE = (":root", "background")
_ALERT = ('[role="alert"]:not(:empty)', "background")
_GLASS = (".sheets", "background")
PAIRS = (
    Pair("page text", (":root", "color"), _PAGE, TEXT_MINIMUM),
    Pair("secondary text", (".lede", "color"), _PAGE, TEXT_MINIMUM),
    Pair("focus ring", (":focus-visible", "outline"), _PAGE, NON_TEXT_MINIMUM),
    Pair("error text", ('[role="alert"]:not(:empty)', "color"), _ALERT, TEXT_MINIMUM),
    Pair("error marker", ('[role="alert"]:not(:empty)', "border-left"), _ALERT, NON_TEXT_MINIMUM),
    Pair("text on the glass", (".sheets", "color"), _GLASS, TEXT_MINIMUM),
    Pair("text at the glass's edge", (".sheets", "color"), (":root", "--glass-edge"), TEXT_MINIMUM),
    Pair(
        "focus ring on the glass",
        (".sheets :focus-visible", "outline-color"),
        _GLASS,
        NON_TEXT_MINIMUM,
    ),
    Pair("exposure sheet text", (".xsheet", "color"), (".xsheet", "background"), TEXT_MINIMUM),
    Pair("joint pin edge", (".dot", "stroke"), (".dot", "fill"), NON_TEXT_MINIMUM),
    Pair(
        "selected joint pin",
        (".dot.selected", "fill"),
        (".dot.selected", "stroke"),
        NON_TEXT_MINIMUM,
    ),
)


def parse_colour(value: str) -> Colour:
    """Return the first #rgb or #rrggbb colour in a CSS value."""
    match = _HEX.search(value)
    if match is None:
        raise ValueError(f"No hex colour in {value!r}")
    digits = match.group("digits")
    if len(digits) == _SHORT_HEX_DIGITS:
        digits = "".join(digit * 2 for digit in digits)
    red, green, blue = (int(digits[i : i + 2], _HEX_BASE) for i in (0, 2, 4))
    return red, green, blue


def relative_luminance(colour: Colour) -> float:
    """Return the WCAG relative luminance, 0 for black to 1 for white."""
    linear: list[float] = []
    for channel in colour:
        value = channel / _MAX_CHANNEL
        linear.append(
            value / _LINEAR_SLOPE
            if value <= _LINEAR_KNEE
            else ((value + _GAMMA_OFFSET) / (1 + _GAMMA_OFFSET)) ** _GAMMA
        )
    return sum(weight * value for weight, value in zip(_LUMA, linear, strict=True))


def contrast_ratio(first: Colour, second: Colour) -> float:
    """Return the WCAG contrast ratio, from 1 to 21, whichever colour is lighter."""
    lighter, darker = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (lighter + _FLARE) / (darker + _FLARE)


def declaration(css: str, selector: str, prop: str) -> str:
    """Return the value of the last `prop` declared in a rule for exactly `selector`."""
    found: str | None = None
    for rule in _RULE.finditer(_COMMENT.sub("", css)):
        selectors = [name.strip() for name in rule.group("selector").split(",")]
        if selector not in selectors:
            continue
        for line in rule.group("body").split(";"):
            name, _, value = line.partition(":")
            if name.strip() == prop:
                found = value.strip()
    if found is None:
        raise ValueError(f"No {prop} declared for {selector}")
    return found


def resolve(css: str, value: str) -> str:
    """Replace each var(--name) in a value with the custom property :root declares."""
    return _VAR.sub(lambda match: declaration(css, ":root", match.group("name")), value)


def measure(css: str, pair: Pair) -> float:
    """Return the contrast ratio of a pair, read from the stylesheet text."""
    foreground = parse_colour(resolve(css, declaration(css, *pair.foreground)))
    background = parse_colour(resolve(css, declaration(css, *pair.background)))
    return contrast_ratio(foreground, background)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--css", type=Path, default=STYLE, help="Stylesheet to read.")
    args = parser.parse_args()
    css = args.css.read_text(encoding="utf-8")
    failed = False
    for pair in PAIRS:
        ratio = measure(css, pair)
        passed = ratio >= pair.minimum
        failed = failed or not passed
        print(f"{'ok  ' if passed else 'FAIL'} {pair.name}: {ratio:.2f}:1 (needs {pair.minimum}:1)")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
