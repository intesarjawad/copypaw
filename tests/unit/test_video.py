from collections.abc import Generator
from contextlib import closing
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.capture.video import ANALYSIS_FPS, OUTPUT_FPS, VideoReadError, read_frames


@runtime_checkable
class _FFmpeg(Protocol):
    def write_frames(
        self,
        path: str,
        size: tuple[int, int],
        *,
        fps: int,
        codec: str,
        pix_fmt_out: str,
        input_params: list[str],
        output_params: list[str],
        ffmpeg_log_level: str,
        ffmpeg_timeout: float,
    ) -> Generator[None, bytes | None, None]: ...


def _clip(
    path: Path, fps: int, count: int, *, variable: bool = False, rotated: bool = False
) -> Path:
    rgb = np.zeros((count, 32, 48, 3), dtype=np.uint8)
    for index in range(count):
        rgb[index, :, :, :] = (index % 256, index // 256, 200)
    if rotated:
        rgb[:, :16, :24] = [240, 10, 20]
        rgb[:, :16, 24:] = [20, 200, 30]
        rgb[:, 16:, :24] = [230, 210, 60]
        rgb[:, 16:, 24:] = [40, 50, 220]
    filters = ["-vf", "setpts=if(lt(N\\,10)\\,N\\,10+(N-10)*3)"] if variable else []
    ffmpeg: object = imageio_ffmpeg
    assert isinstance(ffmpeg, _FFmpeg)
    with closing(
        ffmpeg.write_frames(
            str(path),
            (48, 32),
            fps=fps,
            codec="libx264rgb",
            pix_fmt_out="rgb24",
            input_params=["-display_rotation", "90", "-noautorotate"] if rotated else [],
            output_params=[*filters, "-fps_mode", "vfr", "-crf", "0"],
            ffmpeg_log_level="error",
            ffmpeg_timeout=30,
        )
    ) as writer:
        writer.send(None)
        for image in rgb:
            writer.send(image.tobytes())
    return path


@pytest.mark.parametrize(("target_fps", "count"), [(OUTPUT_FPS, 30), (ANALYSIS_FPS, 60)])
def test_analysis_and_output_sampling(tmp_path: Path, target_fps: int, count: int) -> None:
    frames = read_frames(_clip(tmp_path / "two_seconds.mp4", 30, 60), target_fps=target_fps)

    assert frames.rgb.shape == (count, 32, 48, 3)
    assert frames.rgb.dtype == np.uint8
    assert frames.source_fps == pytest.approx(30)
    assert frames.fps == target_fps
    assert not frames.trimmed
    np.testing.assert_allclose(frames.t_s, np.arange(count) / target_fps, atol=1e-6)
    np.testing.assert_array_equal(frames.rgb[:, 0, 0, 0], np.arange(0, 60, 30 // target_fps))
    np.testing.assert_array_equal(frames.rgb[0, 0, 0], [0, 0, 200])


def test_default_analysis_sampling_trims_at_eight_seconds(tmp_path: Path) -> None:
    frames = read_frames(_clip(tmp_path / "long.mp4", 30, 360))

    assert len(frames.rgb) == 240
    assert frames.trimmed
    np.testing.assert_allclose(frames.t_s, np.arange(240) / ANALYSIS_FPS, atol=1e-6)


def test_exact_duration_is_not_trimmed(tmp_path: Path) -> None:
    frames = read_frames(_clip(tmp_path / "exact.mp4", 30, 60), max_s=2)

    assert len(frames.rgb) == 60
    assert not frames.trimmed


def test_slow_source_keeps_each_frame_once(tmp_path: Path) -> None:
    frames = read_frames(_clip(tmp_path / "slow.mp4", 25, 25))

    assert frames.source_fps == pytest.approx(25)
    assert frames.fps == pytest.approx(25)
    assert len(frames.rgb) == 25
    np.testing.assert_array_equal(frames.rgb[:, 0, 0, 0], np.arange(25))
    np.testing.assert_allclose(frames.t_s, np.arange(25) / 25, atol=1e-6)


def test_rotation_metadata_is_applied_once(tmp_path: Path) -> None:
    frames = read_frames(_clip(tmp_path / "portrait.mp4", 30, 2, rotated=True))

    assert frames.rgb.shape == (2, 48, 32, 3)
    # Display rotation +90 maps the source's top-right corner to top-left.
    np.testing.assert_array_equal(frames.rgb[0, 0, 0], [20, 200, 30])
    np.testing.assert_array_equal(frames.rgb[0, 0, -1], [40, 50, 220])
    np.testing.assert_array_equal(frames.rgb[0, -1, 0], [240, 10, 20])
    np.testing.assert_array_equal(frames.rgb[0, -1, -1], [230, 210, 60])


@pytest.mark.parametrize("timestamps", [(0, 0, 0), (0, 0.1, 0.05)])
def test_non_increasing_decoder_timestamps_are_rejected(
    monkeypatch: pytest.MonkeyPatch, timestamps: tuple[float, ...]
) -> None:
    class Decoder:
        def __init__(self) -> None:
            self.index = -1
            self.released = False

        def is_opened(self) -> bool:
            return True

        def read(self) -> tuple[bool, NDArray[np.uint8]]:
            self.index += 1
            return self.index < len(timestamps), np.zeros((2, 3, 3), dtype=np.uint8)

        def get(self, prop: int) -> float:
            return 30 if prop == cv2.CAP_PROP_FPS else timestamps[self.index] * 1000

        def release(self) -> None:
            self.released = True

    decoder = Decoder()
    monkeypatch.setattr(decoder, "isOpened", decoder.is_opened, raising=False)

    def open_decoder(_path: str) -> Decoder:
        return decoder

    monkeypatch.setattr(cv2, "VideoCapture", open_decoder)

    with pytest.raises(VideoReadError, match=r"frame times.*timestamps\.mp4.*increasing order"):
        read_frames("timestamps.mp4")
    assert decoder.released


def test_variable_frame_rate_uses_nearest_presentation_times(tmp_path: Path) -> None:
    path = _clip(tmp_path / "variable.mp4", 30, 20, variable=True)
    frames = read_frames(path, target_fps=OUTPUT_FPS)
    true_times = np.concatenate((np.arange(10), 10 + np.arange(10) * 3)) / 30
    targets = np.arange(20) / OUTPUT_FPS
    nearest = np.abs(true_times[:, None] - targets).argmin(axis=0)
    expected = np.unique(nearest)

    np.testing.assert_array_equal(frames.rgb[:, 0, 0, 0], expected)
    np.testing.assert_allclose(frames.t_s, true_times[expected], atol=1e-6)
    assert np.max(np.min(np.abs(frames.t_s[:, None] - targets), axis=0)) <= 1 / OUTPUT_FPS


@pytest.mark.parametrize("kind", ["missing", "corrupt", "empty"])
def test_unreadable_input_names_file(tmp_path: Path, kind: str) -> None:
    path = tmp_path / f"{kind}.mp4"
    if kind != "missing":
        path.write_bytes(b"not a video" if kind == "corrupt" else b"")

    with pytest.raises(VideoReadError, match=path.name):
        read_frames(path)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("argument", ["max_s", "target_fps"])
def test_invalid_sampling_arguments(tmp_path: Path, argument: str, value: float) -> None:
    with pytest.raises(ValueError, match=argument):
        read_frames(tmp_path / "unused.mp4", **{argument: value})
