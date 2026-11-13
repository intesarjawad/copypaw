import re
from importlib import import_module
from pathlib import Path

import pytest

API_DOC = Path(__file__).resolve().parents[2] / "docs/api.md"
PACKAGES = ("copypaw.capture", "copypaw.motion")


def documented_names(text: str, package: str) -> list[str]:
    """Read the bulleted names under the package's heading."""
    names: list[str] = []
    listing = False
    for line in text.splitlines():
        if line.startswith("## "):
            listing = line == f"## `{package}`"
        elif listing and (item := re.match(r"- `(\w+)`", line)):
            names.append(item.group(1))
    return names


def test_documented_names_stop_at_the_next_heading() -> None:
    text = "## `a.b`\n\n- `One`: x\n- `two`\n\n## `a.c`\n\n- `Three`\n"

    assert documented_names(text, "a.b") == ["One", "two"]
    assert documented_names(text, "a.c") == ["Three"]
    assert documented_names(text, "a.d") == []


@pytest.mark.parametrize("package", PACKAGES)
def test_all_equals_the_documented_names(package: str) -> None:
    exported: list[str] = import_module(package).__all__
    documented = documented_names(API_DOC.read_text(encoding="utf-8"), package)

    assert len(set(exported)) == len(exported)
    assert sorted(exported) == sorted(documented)


@pytest.mark.parametrize("package", PACKAGES)
def test_every_exported_name_exists(package: str) -> None:
    module = import_module(package)

    assert [name for name in module.__all__ if not hasattr(module, name)] == []
