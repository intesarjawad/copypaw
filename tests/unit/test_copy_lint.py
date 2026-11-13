import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from scripts.copy_lint import main


def run_lint(cwd: Path, *paths: str) -> tuple[int, str]:
    output = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(output):
        patch.chdir(cwd)
        patch.setattr(sys, "argv", ["copy_lint.py", *paths])
        result = main()
    return result, output.getvalue()


def test_rejects_email_address(tmp_path: Path) -> None:
    (tmp_path / "copy.md").write_text("person@example.com", encoding="utf-8")
    result = run_lint(tmp_path, "copy.md")
    assert result[0] == 1
    assert "copy.md:1: email address" in result[1]


def test_skips_binary(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"\xff\xfe")
    assert run_lint(tmp_path, "clip.mp4")[0] == 0
