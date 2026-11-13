from pathlib import Path

import pytest
from typer.testing import CliRunner

from copypaw.cli import app


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "not-a-video.mp4"
    path.write_bytes(b"not a video")
    return path


def test_missing_clip_is_a_usage_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["capture", str(tmp_path / "absent.mp4"), "-o", str(tmp_path)])

    assert result.exit_code == 2
    assert "absent.mp4" in result.output


def test_unknown_mode_lists_the_choices(clip: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["capture", str(clip), "-o", str(tmp_path / "out"), "--mode", "extreme"]
    )

    assert result.exit_code == 2
    assert "balanced" in result.output


def test_unwritable_output_directory_fails_before_decoding(clip: Path) -> None:
    result = CliRunner().invoke(app, ["capture", str(clip), "-o", str(clip / "out")])

    assert result.exit_code == 2
    assert "error: Copypaw cannot write to the output folder" in result.stderr


def test_undecodable_clip_fails_without_models(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["capture", str(clip), "-o", str(out)])

    assert result.exit_code == 2
    assert f"error: Copypaw could not open the video {clip}." in result.stderr
    assert not (out / "motion.npz").exists()
