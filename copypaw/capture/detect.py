"""Animal selection and IoU tracking over sampled RGB video frames."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from copypaw.capture.video import Frames
from copypaw.errors import CopypawError
from copypaw.vendor.rtmlib_onnx.yolox import YOLOX

__all__ = ["ANIMAL_CLASS_IDS", "YOLOX", "Detector", "NoAnimalFoundError", "Track", "track_animal"]

_BOX_ARRAY_DIMENSIONS = 2
_FRAME_ARRAY_DIMENSIONS = 4

# COCO's zero-based YOLOX order: cat through giraffe, excluding bird.
ANIMAL_CLASS_IDS = frozenset({15, 16, 17, 18, 19, 20, 21, 22, 23})
_MIN_TRACK_IOU = 0.3
_MAX_REUSED_FRAMES = 3
_MIN_PRESENT_FRACTION = 0.5


class NoAnimalFoundError(CopypawError):
    """Too few frames contain a detected animal to extract its motion."""

    codes = ("no_animal_found",)


class Detector(Protocol):
    """A BGR detector returning xyxy boxes and zero-based COCO class IDs."""

    def __call__(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Return boxes (N, 4) in source pixels and classes (N,)."""
        ...


@dataclass(frozen=True)
class Track:
    """Per-frame xyxy boxes (F, 4) and actual detection flags (F,).

    Undetected frames reuse the last box for at most three frames, with
    present false. Unavailable boxes contain NaN, including leading gaps.
    """

    boxes: NDArray[np.float32]
    present: NDArray[np.bool_]


def _animal_boxes(boxes: NDArray[np.float32], classes: NDArray[np.int64]) -> NDArray[np.float32]:
    if (
        boxes.ndim != _BOX_ARRAY_DIMENSIONS
        or boxes.shape[1:] != (4,)
        or classes.shape != (len(boxes),)
    ):
        raise ValueError("Detector must return boxes (N, 4) and classes (N,)")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Detector boxes must be finite xyxy coordinates with positive area")
    if not np.issubdtype(classes.dtype, np.integer):
        raise ValueError("Detector classes must be integer COCO IDs")
    return boxes[np.isin(classes, tuple(ANIMAL_CLASS_IDS))]


def _select_box(
    boxes: NDArray[np.float32], previous: NDArray[np.float32] | None
) -> NDArray[np.float32]:
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    if previous is not None:
        intersection = np.prod(
            np.maximum(
                0, np.minimum(boxes[:, 2:], previous[2:]) - np.maximum(boxes[:, :2], previous[:2])
            ),
            axis=1,
        )
        previous_area = np.prod(previous[2:] - previous[:2])
        overlaps = intersection / (areas + previous_area - intersection)
        best = int(np.argmax(overlaps))
        if overlaps[best] >= _MIN_TRACK_IOU:
            return boxes[best]
    return boxes[int(np.argmax(areas))]


def track_animal(frames: Frames, detector: Detector) -> Track:
    """Track one animal, requiring detections in at least half the frames."""
    if (
        frames.rgb.ndim != _FRAME_ARRAY_DIMENSIONS
        or frames.rgb.shape[-1:] != (3,)
        or frames.rgb.dtype != np.uint8
    ):
        raise ValueError("Expected uint8 RGB frames with shape (F, H, W, 3)")
    if frames.rgb.shape[1] == 0 or frames.rgb.shape[2] == 0:
        raise ValueError("Frame width and height must be positive")
    count = len(frames.rgb)
    boxes = np.full((count, 4), np.nan, dtype=np.float32)
    present = np.zeros(count, dtype=np.bool_)
    previous: NDArray[np.float32] | None = None
    missing = 0
    for index, rgb in enumerate(frames.rgb):
        candidates = _animal_boxes(*detector(rgb[..., ::-1]))
        if len(candidates):
            previous = _select_box(candidates, previous).copy()
            boxes[index] = previous
            present[index] = True
            missing = 0
        else:
            missing += 1
            if previous is not None and missing <= _MAX_REUSED_FRAMES:
                boxes[index] = previous
            else:
                previous = None
    if count == 0 or np.count_nonzero(present) / count < _MIN_PRESENT_FRACTION:
        raise NoAnimalFoundError()
    return Track(boxes, present)
