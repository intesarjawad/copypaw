import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

ENGINE_MODULE = "copypaw.vendor.animated_drawings"
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "copypaw"


def engine_import_lines(source: str, package: str) -> list[int]:
    violations: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        else:
            continue
        if any(name == ENGINE_MODULE or name.startswith(f"{ENGINE_MODULE}.") for name in names):
            violations.append(node.lineno)
    return violations


@pytest.mark.parametrize(
    "source",
    [
        "import copypaw.vendor.animated_drawings",
        "import copypaw.vendor.animated_drawings.render as render",
        "import pathlib, copypaw.vendor.animated_drawings.config",
        "from copypaw.vendor.animated_drawings import render",
        "from copypaw.vendor.animated_drawings.config import Config",
        "from copypaw.vendor import animated_drawings as engine",
        "from .vendor.animated_drawings import render",
        "from .vendor import animated_drawings",
        "def delayed():\n    from copypaw.vendor.animated_drawings import render",
    ],
)
def test_detects_engine_imports(source: str) -> None:
    assert engine_import_lines(source, "copypaw")


@pytest.mark.parametrize(
    "source",
    [
        "from copypaw import doctor",
        "from copypaw.vendor import other_dependency",
        "import copypaw.vendor.animated_drawings_extra",
        "# import copypaw.vendor.animated_drawings",
        'name = "copypaw.vendor.animated_drawings"',
    ],
)
def test_allows_unrelated_source(source: str) -> None:
    assert engine_import_lines(source, "copypaw") == []


def test_engine_import_boundary() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] in {"animate", "vendor"}:
            continue
        package = ".".join(("copypaw", *relative.parts[:-1]))
        violations.extend(
            f"{path.relative_to(PACKAGE_ROOT.parent)}:{line}"
            for line in engine_import_lines(path.read_text(encoding="utf-8"), package)
        )
    assert not violations, "Engine imports must stay in copypaw/animate/:\n" + "\n".join(violations)
