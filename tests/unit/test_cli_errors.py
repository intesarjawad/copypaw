from pathlib import Path

import pytest
from typer.testing import CliRunner

from copypaw import cli
from copypaw.capture.video import VideoReadError
from copypaw.cli import app

ISSUES_URL = "https://github.com/intesarjawad/copypaw/issues/new?template=bug.yml"


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "not-a-video.mp4"
    path.write_bytes(b"not a video")
    return path


def test_known_error_prints_two_lines_and_exits_2(clip: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["capture", str(clip), "-o", str(tmp_path / "out")])

    assert result.exit_code == 2
    assert result.stderr.splitlines() == [
        f"error: Copypaw could not open the video {clip}.",
        "next:  Re-export the clip as an H.264 MP4, then run `copypaw capture clip.mp4 --out out`.",
    ]
    assert result.stdout == ""


def test_unexpected_error_is_internal_and_exits_1(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(_path: Path) -> None:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(cli, "read_frames", explode)
    result = CliRunner().invoke(app, ["capture", str(clip), "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "error: Copypaw hit a bug it did not expect.",
        f"next:  Run `copypaw doctor` and open an issue with its output: {ISSUES_URL}",
    ]


def test_debug_adds_the_traceback_after_the_two_lines(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(_path: Path) -> None:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(cli, "read_frames", explode)
    result = CliRunner().invoke(app, ["--debug", "capture", str(clip), "-o", str(tmp_path / "o")])

    lines = result.stderr.splitlines()
    assert result.exit_code == 1
    assert lines[0] == "error: Copypaw hit a bug it did not expect."
    assert lines[1].startswith("next:  Run `copypaw doctor`")
    assert "Traceback (most recent call last):" in result.stderr
    assert "RuntimeError: secret detail" in result.stderr


def test_debug_does_not_change_a_known_error_exit_code(clip: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["--debug", "capture", str(clip), "-o", str(tmp_path / "o")])

    assert result.exit_code == 2
    assert result.stderr.startswith(f"error: Copypaw could not open the video {clip}.\n")


def test_multi_line_detail_stays_on_one_line(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_path: Path) -> None:
        raise VideoReadError("video_decode", path="a.mp4", reason="first line\n  second line")

    monkeypatch.setattr(cli, "read_frames", fail)
    result = CliRunner().invoke(app, ["capture", str(clip), "-o", str(tmp_path / "out")])

    assert result.stderr.splitlines()[0] == (
        "error: Copypaw could not decode a.mp4: first line second line"
    )
    assert len(result.stderr.splitlines()) == 2


def test_unwritable_output_folder_is_an_input_problem(clip: Path) -> None:
    result = CliRunner().invoke(app, ["capture", str(clip), "-o", str(clip / "out")])

    assert result.exit_code == 2
    assert result.stderr.startswith(
        f"error: Copypaw cannot write to the output folder {clip / 'out'}:"
    )
    assert "next:  Run the command again with `--out`" in result.stderr


def test_usage_errors_keep_their_own_format_and_exit_code(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["capture", str(tmp_path / "absent.mp4"), "-o", str(tmp_path)])

    assert result.exit_code == 2
    assert "error:" not in result.stderr.splitlines()[0]
