from pathlib import Path

import pytest

from scripts.axe_check import (
    PROFILES,
    Finding,
    Focus,
    Profile,
    Report,
    blocking,
    focus_is_visible,
    focus_screenshot,
    slug,
)


def violation(rule: str, impact: str, nodes: int) -> dict[str, object]:
    return {"id": rule, "impact": impact, "nodes": [{}] * nodes, "help": f"{rule} help"}


def focus(**changes: object) -> Focus:
    fields: dict[str, object] = {
        "tag": "button",
        "label": "Animate",
        "key": "BUTTON|||",
        "focus_visible": True,
        "outline_style": "solid",
        "outline_width": "3px",
        "box_shadow": "none",
    }
    return Focus(**{**fields, **changes})  # type: ignore[arg-type]


def test_only_serious_and_critical_violations_block() -> None:
    found = blocking(
        "desktop: empty",
        [
            violation("color-contrast", "serious", 2),
            violation("region", "moderate", 1),
            violation("label", "critical", 1),
            violation("tabindex", "minor", 3),
        ],
    )

    assert found == [
        Finding("desktop: empty", "color-contrast", "serious", 2, "color-contrast help"),
        Finding("desktop: empty", "label", "critical", 1, "label help"),
    ]


def test_no_violations_means_nothing_blocks() -> None:
    assert blocking("try", []) == []


@pytest.mark.parametrize(
    ("changes", "visible"),
    [
        ({}, True),
        ({"outline_width": "2px"}, True),
        ({"outline_width": "1px"}, False),
        ({"outline_style": "none"}, False),
        ({"outline_style": "none", "box_shadow": "rgb(26, 95, 180) 0px 0px 0px 3px"}, True),
        ({"outline_width": "medium"}, False),
        ({"focus_visible": False}, False),
    ],
)
def test_a_focus_ring_needs_a_wide_outline_or_a_shadow(
    changes: dict[str, object], visible: bool
) -> None:
    assert focus_is_visible(focus(**changes)) is visible


def test_labels_become_file_name_parts() -> None:
    assert slug("Right back leg, top", "input") == "right-back-leg-top"
    assert slug("  ", "input") == "input"


def test_a_screenshot_is_stored_per_profile_and_stop() -> None:
    path = focus_screenshot(Path("build"), Profile("mobile", 390, 844), 7, focus())

    assert path == Path("build/focus/mobile/07-animate.png")


def test_a_report_passes_only_when_empty() -> None:
    failed = Report(failures=["desktop: flow did not finish"])

    assert Report().passed
    assert not failed.passed
    assert not Report(findings=[Finding("try", "label", "critical", 1, "")]).passed


def test_the_three_profiles_cover_both_sizes_and_touch() -> None:
    assert [(p.name, p.width, p.height, p.touch) for p in PROFILES] == [
        ("desktop", 1280, 800, False),
        ("mobile", 390, 844, True),
        ("keyboard", 1280, 800, False),
    ]
