"""Confidence masking and temporal cleaning of source-pixel keypoints."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["CleanedKeypoints", "clean", "smoothing_warning", "validate_timestamps"]

_JOINT_COUNT = 17
_KEYPOINT_DIMENSIONS = 3
_POLYORDER = 2
_MIN_WINDOW_SAMPLES = 5
_WINDOW_RADIUS_SECONDS = 1 / 15  # Five samples at 30 Hz, with the same duration at other rates.
_GAIT_HZ = 4
_MAX_RESPONSE_ERROR = 0.1  # Bound distance from unity, guaranteeing at least 90% amplitude.


@dataclass(frozen=True)
class CleanedKeypoints:
    """Coordinates (F, 17, 2) and unavailable locations (F, 17).

    missing_mask is true throughout a joint's track if it was never confident.
    Those coordinates stay NaN; interpolation cannot supply anatomical evidence.
    """

    keypoints: NDArray[np.float32]
    missing_mask: NDArray[np.bool_]


def validate_timestamps(timestamps_s: NDArray[np.float64], frame_count: int) -> None:
    """Require one finite timestamp in seconds per frame, strictly increasing."""
    if timestamps_s.shape != (frame_count,) or not np.isfinite(timestamps_s).all():
        raise ValueError("Expected finite timestamps in seconds with shape (F,)")
    if np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("timestamps must be strictly increasing")


def smoothing_warning(timestamps_s: NDArray[np.float64]) -> str | None:
    """Explain reduced filtering when a five-sample time window is unavailable."""
    if np.any(np.diff(timestamps_s) > (_WINDOW_RADIUS_SECONDS / 2) * (1 + 1e-9)):
        return (
            "Smoothing is reduced or skipped at sampling rates below 30 Hz "
            "to preserve at least 90% of 4 Hz limb motion."
        )
    return None


def _smooth(tracks: NDArray[np.float64], timestamps_s: NDArray[np.float64]) -> NDArray[np.float64]:
    # Fit quadratics in elapsed seconds, including on irregular samples. Windows
    # with fewer than five points stay unchanged. Limit each filter's complex
    # 4 Hz response error to 0.1 so sparse or asymmetric windows cannot erase gait.
    result = tracks.copy()
    radius = _WINDOW_RADIUS_SECONDS * (1 + 1e-9)
    for frame, time_s in enumerate(timestamps_s):
        start, stop = np.searchsorted(timestamps_s, [time_s - radius, time_s + radius], side="left")
        if stop - start < _MIN_WINDOW_SAMPLES:
            continue
        offsets = timestamps_s[start:stop] - time_s
        design = np.vander(offsets / radius, N=_POLYORDER + 1, increasing=True)
        weights = np.linalg.lstsq(design.T, np.array([1.0, 0.0, 0.0]), rcond=None)[0]
        response_error = abs(np.dot(weights, np.exp(2j * np.pi * _GAIT_HZ * offsets)) - 1)
        blend = min(1.0, _MAX_RESPONSE_ERROR / max(response_error, np.finfo(float).eps))
        result[frame] = tracks[frame] + blend * (weights @ tracks[start:stop] - tracks[frame])
    return result


def clean(
    kp: NDArray[np.float32],
    scores: NDArray[np.float32],
    timestamps_s: NDArray[np.float64],
    min_score: float = 0.3,
) -> CleanedKeypoints:
    """Interpolate in seconds, hold boundary gaps, and smooth within 2/15 seconds.

    Scores at the threshold are confident. One observation gives a constant
    track; no observations leave NaN coordinates and a true missing mask.
    Boundary gaps hold the nearest observation even after smoothing. Inputs are
    preserved. Use original scores and timestamps for quality and ground fitting.
    """
    if (
        kp.ndim != _KEYPOINT_DIMENSIONS
        or kp.shape[1:] != (_JOINT_COUNT, 2)
        or scores.shape != kp.shape[:2]
    ):
        raise ValueError("Expected keypoint shape (F, 17, 2) and scores shape (F, 17)")
    validate_timestamps(timestamps_s, len(kp))
    if not np.isfinite(min_score) or min_score <= 0:
        raise ValueError("min_score must be finite and positive")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Keypoint scores must be finite and nonnegative")
    confident = scores >= min_score
    if not np.isfinite(kp[confident]).all():
        raise ValueError("Confident keypoint coordinates must be finite")

    result = np.full(kp.shape, np.nan, dtype=np.float64)
    missing = np.broadcast_to(~confident.any(axis=0), scores.shape).copy()
    for joint in range(_JOINT_COUNT):
        visible = confident[:, joint]
        if not visible.any():
            continue
        observed = timestamps_s[visible]
        for coordinate in range(2):
            result[:, joint, coordinate] = np.interp(
                timestamps_s, observed, kp[visible, joint, coordinate]
            )
    smoothed = _smooth(result.reshape(len(kp), _JOINT_COUNT * 2), timestamps_s).reshape(kp.shape)
    for joint in range(_JOINT_COUNT):
        observed_frames = np.flatnonzero(confident[:, joint])
        if len(observed_frames):
            first, last = observed_frames[0], observed_frames[-1]
            smoothed[:first, joint] = kp[first, joint]
            smoothed[last + 1 :, joint] = kp[last, joint]
    return CleanedKeypoints(smoothed.astype(np.float32), missing)
