"""Paw stance detection and image-space ground estimation from cleaned keypoints."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from copypaw.capture.smoothing import validate_timestamps

__all__ = [
    "CONTACT_HEIGHT_RATIO",
    "CONTACT_SPEED_RATIO",
    "PAW_INDICES",
    "REQUIRED_LANDMARKS",
    "ROOT_KP",
    "Contacts",
    "body_length_px",
    "detect_contacts",
]

CONTACT_HEIGHT_RATIO = 0.06  # Maximum paw clearance in body lengths.
CONTACT_SPEED_RATIO = 0.3  # Maximum vertical or forward speed in body lengths per second.
# Near front, far front, near back, far back after identity resolution.
PAW_INDICES = (7, 10, 13, 16)
ROOT_KP = 4
_NECK = 3
REQUIRED_LANDMARKS = (
    (_NECK, "Neck"),
    (ROOT_KP, "Root_of_tail"),
    (7, "Left_front_paw"),
    (10, "Right_front_paw"),
    (13, "Left_back_paw"),
    (16, "Right_back_paw"),
)
_JOINT_COUNT = 17
_KEYPOINT_DIMENSIONS = 3
_MAX_SLOPE_SAMPLES = 64  # Bound pairwise slope storage independently of clip duration.
_GROUND_MAD_LIMIT = 2
_MIN_CONTACT_FRAMES = 2


@dataclass(frozen=True)
class Contacts:
    """Stance flags (F, 4) and ground line (2,) in source-pixel coordinates.

    contacts orders paws near-front, far-front, near-back, far-back.
    ground_line_px contains slope and intercept for image y = slope * x + intercept.
    No stationary paw evidence gives no contacts and a NaN ground line.
    """

    contacts: NDArray[np.bool_]
    ground_line_px: NDArray[np.float64]


def body_length_px(kp: NDArray[np.float32]) -> float:
    """Measure median body length, rejecting unavailable required anatomy."""
    if kp.ndim != _KEYPOINT_DIMENSIONS or kp.shape[1:] != (_JOINT_COUNT, 2):
        raise ValueError("Expected keypoint shape (F, 17, 2)")
    if len(kp) == 0:
        raise ValueError("Keypoints must contain at least one frame")
    for joint, name in REQUIRED_LANDMARKS:
        if np.isnan(kp[:, joint]).all():
            raise ValueError(f"Missing required landmark: {name}")
        if not np.isfinite(kp[:, joint]).all():
            raise ValueError(f"Required landmark {name} coordinates must be finite")
    lengths = np.linalg.norm(kp[:, _NECK].astype(np.float64) - kp[:, ROOT_KP], axis=1)
    body_len = float(np.median(lengths))
    if body_len <= 0:
        raise ValueError("Median Neck to Root_of_tail body length must be positive")
    return body_len


def _ground_line(points: NDArray[np.float64]) -> NDArray[np.float64]:
    x = points[:, 0]
    y = points[:, 1]
    if np.ptp(x) == 0:
        return np.array([0, np.median(y)], dtype=np.float64)
    # A median of pairwise slopes and median intercept resist outliers on both
    # sides. Sample across sorted x for a bounded initialization, then use every
    # confident stance point in the MAD cut and centered least-squares refit.
    order = np.argsort(x)
    sample = order[np.linspace(0, len(x) - 1, min(len(x), _MAX_SLOPE_SAMPLES), dtype=int)]
    first, second = np.triu_indices(len(sample), k=1)
    delta_x = x[sample[second]] - x[sample[first]]
    distinct = delta_x != 0
    slope = float(np.median((y[sample[second]] - y[sample[first]])[distinct] / delta_x[distinct]))
    intercept = float(np.median(y - slope * x))
    center_x = float(np.mean(x))
    design = np.column_stack((x - center_x, np.ones(len(x))))
    residual = y - (slope * x + intercept)
    centered = np.abs(residual - np.median(residual))
    mad = float(np.median(centered))
    # Exact lines can have zero MAD; retain their roundoff-sized residuals.
    tolerance = np.finfo(np.float64).eps * max(1, float(np.max(np.abs(y)))) * len(y)
    inliers = centered <= max(_GROUND_MAD_LIMIT * mad, tolerance)
    if np.ptp(x[inliers]) == 0 or np.ptp(y[inliers]) == 0:
        return np.array([0, np.median(y[inliers])], dtype=np.float64)
    line = np.linalg.lstsq(design[inliers], y[inliers], rcond=None)[0]
    return np.array([line[0], line[1] - line[0] * center_x], dtype=np.float64)


def _clean_runs(contacts: NDArray[np.bool_]) -> None:
    for paw in range(contacts.shape[1]):
        track = contacts[:, paw]
        boundaries = np.flatnonzero(np.diff(np.r_[False, track, False]))
        for start, stop in boundaries.reshape(-1, 2):
            if stop - start < _MIN_CONTACT_FRAMES:
                track[start:stop] = False
        # Remove isolated detections before bridging only bounded one-frame gaps.
        track[1:-1] |= track[:-2] & track[2:]


def detect_contacts(
    kp: NDArray[np.float32],
    timestamps_s: NDArray[np.float64],
    *,
    scores: NDArray[np.float32],
    min_score: float = 0.3,
    height_ratio: float = CONTACT_HEIGHT_RATIO,
    speed_ratio: float = CONTACT_SPEED_RATIO,
) -> Contacts:
    """Find stance on finite, cleaned, right-facing AP-10K tracks.

    Body length is the clip's median Neck to Root_of_tail distance. Ground uses
    all confident, vertically quiet paws across the clip, with a median slope
    initialization and two-MAD residual cut. Only zero x span forces a horizontal
    line. Speeds use elapsed seconds with centered internal differences and
    one-sided endpoint differences;
    one frame has zero speed. Runs shorter than two frames are removed before
    internal one-frame gaps are filled. Inputs are preserved; empty clips and
    zero median body length are rejected because ground scale cannot be inferred.
    height_ratio and speed_ratio override the defaults for threshold tuning.
    """
    body_len = body_length_px(kp)
    validate_timestamps(timestamps_s, len(kp))
    if scores.shape != kp.shape[:2]:
        raise ValueError("Expected scores shape (F, 17)")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Keypoint scores must be finite and nonnegative")
    for name, value in (("min_score", min_score), ("height_ratio", height_ratio)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(speed_ratio) or speed_ratio < 0:
        raise ValueError("speed_ratio must be finite and nonnegative")
    confident = scores >= min_score
    for joint, name in REQUIRED_LANDMARKS:
        if not confident[:, joint].any():
            raise ValueError(f"Missing required landmark: {name}")
    paws = kp[:, PAW_INDICES].astype(np.float64)
    velocity_y = np.zeros(paws.shape[:2], dtype=np.float64)
    velocity_forward = np.zeros_like(velocity_y)
    if len(kp) > 1:
        velocity_y = np.gradient(paws[..., 1], timestamps_s, axis=0)
        relative_x = paws[..., 0] - kp[:, ROOT_KP, 0, None]
        velocity_forward = np.gradient(relative_x, timestamps_s, axis=0)
    speed_limit = speed_ratio * body_len
    ground_candidates = confident[:, PAW_INDICES] & (np.abs(velocity_y) < speed_limit)
    if not ground_candidates.any():
        return Contacts(np.zeros(paws.shape[:2], dtype=np.bool_), np.full(2, np.nan))
    ground = _ground_line(paws[ground_candidates])
    height = ground[0] * paws[..., 0] + ground[1] - paws[..., 1]
    contacts = (
        (height < height_ratio * body_len)
        & (np.abs(velocity_y) < speed_limit)
        & (velocity_forward < speed_limit)
    )
    _clean_runs(contacts)
    return Contacts(contacts, ground)
