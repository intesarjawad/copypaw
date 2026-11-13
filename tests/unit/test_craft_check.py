import sys
from pathlib import Path

import pytest

from scripts.craft_check import (
    PAIRS,
    STYLE,
    contrast_ratio,
    declaration,
    main,
    measure,
    parse_colour,
)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def test_black_on_white_is_the_widest_ratio() -> None:
    assert contrast_ratio(BLACK, WHITE) == pytest.approx(21.0)


def test_a_colour_against_itself_is_one() -> None:
    assert contrast_ratio((29, 29, 27), (29, 29, 27)) == pytest.approx(1.0)


def test_ratio_does_not_depend_on_argument_order() -> None:
    assert contrast_ratio((118, 118, 118), WHITE) == contrast_ratio(WHITE, (118, 118, 118))


@pytest.mark.parametrize(("grey", "ratio"), [(0x76, 4.54), (0x77, 4.48)])
def test_grey_text_on_white_straddles_the_text_minimum(grey: int, ratio: float) -> None:
    assert contrast_ratio((grey, grey, grey), WHITE) == pytest.approx(ratio, abs=0.01)


@pytest.mark.parametrize(
    ("value", "colour"),
    [
        ("#fff", (255, 255, 255)),
        ("#1a5fb4", (26, 95, 180)),
        ("3px solid #A51D2D", (165, 29, 45)),
    ],
)
def test_colours_are_read_from_a_css_value(value: str, colour: tuple[int, int, int]) -> None:
    assert parse_colour(value) == colour


def test_a_value_without_a_hex_colour_is_rejected() -> None:
    with pytest.raises(ValueError, match="No hex colour"):
        parse_colour("rgb(0 0 0)")


def test_declaration_reads_grouped_nested_and_repeated_rules() -> None:
    css = """
    .a, .b { color: #111111; margin: 0 }
    @media (prefers-reduced-motion: no-preference) {
      .b { color: #222222 }
    }
    """

    assert declaration(css, ".a", "color") == "#111111"
    assert declaration(css, ".b", "color") == "#222222"
    with pytest.raises(ValueError, match=r"No margin declared for \.c"):
        declaration(css, ".c", "margin")


def test_comments_do_not_hide_a_rule() -> None:
    css = "/* the sheet, ruled */\n.a { color: #111111; /* why; and how */ }"

    assert declaration(css, ".a", "color") == "#111111"


def test_a_pair_is_measured_from_the_stylesheet_text() -> None:
    css = ":root { color: #000; background: #fff }"

    assert measure(css, PAIRS[0]) == pytest.approx(21.0)


def test_the_page_stylesheet_passes_every_pair(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["craft_check.py"])

    assert main() == 0
    assert capsys.readouterr().out.count("ok  ") == len(PAIRS)


def test_custom_properties_are_read_from_root() -> None:
    css = ":root { --ink: #000; --paper: #fff; color: var(--ink); background: var( --paper ) }"

    assert measure(css, PAIRS[0]) == pytest.approx(21.0)


def test_a_faint_stylesheet_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    css = STYLE.read_text(encoding="utf-8").replace(
        "--room-ink: #ece7de;", "--room-ink: #333333;", 1
    )
    faint = tmp_path / "style.css"
    faint.write_text(css, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["craft_check.py", "--css", str(faint)])

    assert main() == 1
    assert "FAIL page text" in capsys.readouterr().out
