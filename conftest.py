from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the option that points the gallery test at CI-built motion files."""
    parser.addoption(
        "--motions-extra",
        type=Path,
        default=None,
        help="Directory of generated community motion folders for the gallery test to check.",
    )
