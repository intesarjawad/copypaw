"""Clip quality measured from confidence scores before temporal cleaning."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from copypaw.capture.contacts import REQUIRED_LANDMARKS
from copypaw.capture.smoothing import smoothing_warning, validate_timestamps

__all__ = ["QualityReport", "assess_quality", "write_quality"]

_JOINT_COUNT = 17
_SCORE_DIMENSIONS = 2
_MIN_CONFIDENT_JOINTS = 12
_MAX_POOR_FRAME_FRACTION = 0.3
_FILMING_WARNING = (
    "Fewer than 12 of 17 joints are confident in over 30% of frames. "
    "Follow the filming guide: keep the whole pet visible in a steady, well-lit side view."
)


@dataclass(frozen=True)
class QualityReport:
    """Coverage in array joint order, before interpolation can hide missing data.

    frames_interpolated counts frames with any score below the threshold,
    including joints with no observations, which cleaning leaves missing.
    swaps_corrected counts pair-frame permutations from identity resolution.
    errors names required landmarks with no confident observation in the clip.
    """

    per_joint_confident_fraction: tuple[float, ...]
    frames_interpolated: int
    swaps_corrected: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def assess_quality(
    scores: NDArray[np.float32],
    timestamps_s: NDArray[np.float64],
    *,
    swaps_corrected: int = 0,
    min_score: float = 0.3,
) -> QualityReport:
    """Flag insufficient coverage using original scores, with inclusive confidence.

    Empty clips have zero coverage and an explicit warning instead of NaN
    fractions. Exactly 30% poor frames is allowed; more triggers the warning.
    """
    if scores.ndim != _SCORE_DIMENSIONS or scores.shape[1] != _JOINT_COUNT:
        raise ValueError("Expected scores shape (F, 17)")
    validate_timestamps(timestamps_s, len(scores))
    if not np.isfinite(min_score) or min_score <= 0:
        raise ValueError("min_score must be finite and positive")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Keypoint scores must be finite and nonnegative")
    if swaps_corrected < 0:
        raise ValueError("swaps_corrected must be nonnegative")
    confident = scores >= min_score
    errors = tuple(
        f"Missing required landmark: {name}."
        for joint, name in REQUIRED_LANDMARKS
        if not confident[:, joint].any()
    )
    if not len(scores):
        return QualityReport(
            (0.0,) * _JOINT_COUNT, 0, swaps_corrected, ("Clip has no frames.",), errors
        )
    poor_fraction = float(np.mean(confident.sum(axis=1) < _MIN_CONFIDENT_JOINTS))
    warnings = [_FILMING_WARNING] if poor_fraction > _MAX_POOR_FRAME_FRACTION else []
    timing_warning = smoothing_warning(timestamps_s)
    if timing_warning:
        warnings.append(timing_warning)
    return QualityReport(
        tuple(float(fraction) for fraction in confident.mean(axis=0)),
        int(np.count_nonzero(~confident.all(axis=1))),
        swaps_corrected,
        tuple(warnings),
        errors,
    )


def write_quality(report: QualityReport, out_dir: Path) -> Path:
    """Write quality.json in an existing output directory and return its path."""
    path = out_dir / "quality.json"
    path.write_text(json.dumps(asdict(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
