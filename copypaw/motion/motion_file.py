"""Save and load a captured animal motion as a single compressed archive."""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from copypaw.errors import CopypawError

__all__ = ["MOTION_FILE_VERSION", "MotionFile", "MotionFileError"]

MOTION_FILE_VERSION = 1
_KEYPOINT_COUNT = 17
_PAW_COUNT = 4  # Near-front, far-front, near-back, far-back.
_GROUND_LINE_SIZE = 2  # Slope and intercept of image y against image x.
_FACINGS = ("left", "right")


class MotionFileError(CopypawError):
    """A motion file is missing a key or holds data of the wrong kind."""

    codes = ("motion_invalid", "motion_missing_key", "motion_version", "motion_unreadable")


def _invalid(key: str, problem: str) -> MotionFileError:
    return MotionFileError("motion_invalid", key=key, problem=problem)


@dataclass(frozen=True)
class MotionFile:
    """One captured motion in source-pixel coordinates."""

    kp: NDArray[np.float32]
    scores: NDArray[np.float32]
    fps: float
    facing: Literal["left", "right"]
    contacts: NDArray[np.bool_]
    ground_line_px: NDArray[np.float64]
    travel_root_x: NDArray[np.float32]
    source: str
    version: int = MOTION_FILE_VERSION

    def save(self, path: Path) -> None:
        """Write every field to a compressed archive at path."""
        np.savez_compressed(
            path,
            version=np.int64(self.version),
            kp=self.kp,
            scores=self.scores,
            fps=np.float64(self.fps),
            facing=np.str_(self.facing),
            contacts=self.contacts,
            ground_line_px=self.ground_line_px,
            travel_root_x=self.travel_root_x,
            source=np.str_(self.source),
        )

    @classmethod
    def load(cls, path: Path) -> "MotionFile":
        """Read and validate an archive; raise MotionFileError naming the offending key."""
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, NDArray[np.generic]] = {
                    key: archive[key] for key in archive.files
                }
        except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
            raise MotionFileError("motion_unreadable", path=path) from error

        version = _scalar(arrays, "version", "i")
        if int(version) != MOTION_FILE_VERSION:
            raise MotionFileError(
                "motion_version", found=int(version), expected=MOTION_FILE_VERSION
            )
        kp = _array(arrays, "kp", np.float32, 3)
        frames = kp.shape[0]
        if kp.shape[1:] != (_KEYPOINT_COUNT, 2):
            raise _invalid("kp", f"must have shape (F, {_KEYPOINT_COUNT}, 2), got {kp.shape}")
        scores = _array(arrays, "scores", np.float32, 2)
        contacts = _array(arrays, "contacts", np.bool_, 2)
        travel = _array(arrays, "travel_root_x", np.float32, 1)
        ground = _array(arrays, "ground_line_px", np.float64, 1)
        for key, value in (("scores", scores), ("contacts", contacts), ("travel_root_x", travel)):
            if value.shape[0] != frames:
                raise _invalid(key, f"has {value.shape[0]} frames but kp has {frames}")
        if scores.shape[1] != _KEYPOINT_COUNT:
            raise _invalid("scores", f"must have shape (F, {_KEYPOINT_COUNT}), got {scores.shape}")
        if contacts.shape[1] != _PAW_COUNT:
            raise _invalid("contacts", f"must have shape (F, {_PAW_COUNT}), got {contacts.shape}")
        if ground.shape != (_GROUND_LINE_SIZE,):
            raise _invalid("ground_line_px", f"must have shape (2,), got {ground.shape}")

        fps = float(_scalar(arrays, "fps", "f"))
        facing = str(_scalar(arrays, "facing", "U"))
        if facing not in _FACINGS:
            raise _invalid("facing", f"must be 'left' or 'right', got {facing!r}")
        return cls(
            kp=kp,
            scores=scores,
            fps=fps,
            facing="left" if facing == "left" else "right",
            contacts=contacts,
            ground_line_px=ground,
            travel_root_x=travel,
            source=str(_scalar(arrays, "source", "U")),
            version=int(version),
        )


def _require(arrays: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.generic]:
    if key not in arrays:
        raise MotionFileError("motion_missing_key", key=key)
    return arrays[key]


def _array[T: np.generic](
    arrays: dict[str, NDArray[np.generic]], key: str, dtype: type[T], ndim: int
) -> NDArray[T]:
    value = _require(arrays, key)
    if value.dtype != np.dtype(dtype):
        raise _invalid(key, f"must have dtype {np.dtype(dtype)}, got {value.dtype}")
    if value.ndim != ndim:
        raise _invalid(key, f"must have {ndim} dimensions, got {value.ndim}")
    return value.astype(dtype, copy=False)


def _scalar(arrays: dict[str, NDArray[np.generic]], key: str, kind: str) -> int | float | str:
    value = _require(arrays, key)
    if value.dtype.kind != kind or value.ndim != 0:
        raise _invalid(key, f"must be a scalar of kind {kind!r}, got {value.dtype} {value.shape}")
    return value.item()
