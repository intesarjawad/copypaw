import tomllib
from pathlib import Path

import copypaw


def test_version_matches_project_metadata() -> None:
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with project_file.open("rb") as stream:
        project = tomllib.load(stream)

    assert copypaw.__version__ == project["project"]["version"]
