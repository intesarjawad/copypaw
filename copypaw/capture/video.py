"""Timestamp-based video sampling with metadata rotation left to OpenCV."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from copypaw.errors import CopypawError

ANALYSIS_FPS = 30
OUTPUT_FPS = 15
_TIME_EPSILON_S = 1e-9  # Keep exact sample boundaries stable under timestamp rounding.


class VideoReadError(CopypawError):
    """A video could not be opened or decoded into frames."""

    codes = (
        "video_open",
        "video_frame_rate",
        "video_timestamps",
        "video_no_frames",
        "video_decode",
    )


@dataclass(frozen=True)
class Frames:
    """Unique RGB frames and their source presentation times in seconds.

    rgb has shape (F, H, W, 3), t_s has shape (F,). fps is the target
    sampling rate capped at source_fps; irregular source times are preserved.
    """

    rgb: NDArray[np.uint8]
    t_s: NDArray[np.float64]
    source_fps: float
    fps: float
    trimmed: bool


def _decode(
    capture: cv2.VideoCapture, path: str | Path
) -> Iterator[tuple[cv2.typing.MatLike, float]]:
    previous_s: float | None = None
    while True:
        success, image = capture.read()
        if not success:
            return
        time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not math.isfinite(time_s) or time_s < 0:
            raise VideoReadError("video_timestamps", path=path)
        if previous_s is not None and time_s <= previous_s:
            raise VideoReadError("video_timestamps", path=path)
        previous_s = time_s
        yield image, time_s


def read_frames(path: str | Path, max_s: float = 8.0, target_fps: float = ANALYSIS_FPS) -> Frames:
    """Sample nearest presentation times before max_s, without duplicating frames.

    Sources slower than target_fps keep every frame. OpenCV applies display
    rotation metadata during decoding, so no additional rotation is applied.
    """
    for name, value in (("max_s", max_s), ("target_fps", target_fps)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    capture = cv2.VideoCapture(str(path))
    images: list[NDArray[np.uint8]] = []
    times: list[float] = []
    previous: cv2.typing.MatLike | None = None
    previous_s = 0.0
    next_sample = 0
    trimmed = False
    try:
        if not capture.isOpened():
            raise VideoReadError("video_open", path=path)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise VideoReadError("video_frame_rate", path=path)
        keep_all = source_fps <= target_fps
        for image, time_s in _decode(capture, path):
            if time_s >= max_s - _TIME_EPSILON_S:
                trimmed = True
                break
            if previous is not None:
                midpoint_s = (previous_s + time_s) / 2
                if keep_all or next_sample / target_fps <= midpoint_s + _TIME_EPSILON_S:
                    images.append(
                        np.asarray(cv2.cvtColor(previous, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                    )
                    times.append(previous_s)
                    # One frame can be nearest several targets in a VFR gap.
                    next_sample = math.floor((midpoint_s + _TIME_EPSILON_S) * target_fps) + 1
            previous, previous_s = image, time_s

        end_s = min(max_s, previous_s + 1 / source_fps)
        if previous is not None and (keep_all or next_sample / target_fps < end_s):
            images.append(np.asarray(cv2.cvtColor(previous, cv2.COLOR_BGR2RGB), dtype=np.uint8))
            times.append(previous_s)
        if not images:
            raise VideoReadError("video_no_frames", path=path)
    except cv2.error as error:
        raise VideoReadError("video_decode", path=path, reason=error) from error
    finally:
        capture.release()

    return Frames(
        rgb=np.stack(images),
        t_s=np.asarray(times, dtype=np.float64),
        source_fps=source_fps,
        fps=min(source_fps, target_fps),
        trimmed=trimmed,
    )
