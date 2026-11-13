"""Clip-wide limb identity and facing normalization in source-pixel coordinates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["IdentityResult", "resolve_identity"]

_LIMB_PAIRS = ((5, 6, 7, 8, 9, 10), (11, 12, 13, 14, 15, 16))
_SWAP_ORDER = (3, 4, 5, 0, 1, 2)
_JOINT_COUNT = 17
_NOSE = 2
_NECK = 3
_ROOT_OF_TAIL = 4
_FIT_WINDOW = 5  # Five samples reduce slope noise without spanning a whole gait phase.
_NORMAL_MAD = 0.6744897501960817
_NOISE_RADIUS = 3.0  # Three estimated standard deviations cover overlapping observations.
_MOTION_RADIUS = 2.0  # Include a frame of travel from each of the two chains.
_VARIANCE_FLOOR = float(np.finfo(np.float32).eps)
type _FloatArray = NDArray[np.float32] | NDArray[np.float64]


@dataclass(frozen=True)
class IdentityResult:
    """Relabelled float32 kp (F, 17, 2) and scores (F, 17).

    The first chain of each limb pair is Near, the higher-confidence chain
    over the clip. swaps_corrected counts pair-frame permutations, including
    the final Near/Far assignment. facing records the original direction;
    left-facing input is mirrored using frame_width - 1 - x. facing_observed
    distinguishes nose/tail evidence from the default right-facing result.
    """

    kp: NDArray[np.float32]
    scores: NDArray[np.float32]
    swaps_corrected: int
    facing: Literal["left", "right"]
    facing_observed: bool


def _validate_inputs(
    kp: _FloatArray,
    scores: _FloatArray,
    frame_width: float,
) -> None:
    if kp.shape[1:] != (_JOINT_COUNT, 2) or scores.shape != kp.shape[:2]:
        raise ValueError("Expected keypoint shape (F, 17, 2) and scores shape (F, 17)")
    if not np.isfinite(frame_width) or frame_width <= 0:
        raise ValueError("frame_width must be finite and positive")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Keypoint scores must be finite and nonnegative")
    if np.isinf(kp).any() or np.any(~np.isfinite(kp).all(axis=2) & (scores > 0)):
        raise ValueError("Keypoint coordinates must be finite, or NaN with zero confidence")


@dataclass(frozen=True)
class _Fit:
    velocity: NDArray[np.float64]
    intercept: NDArray[np.float64]
    confidence: NDArray[np.float64]
    score_variance: NDArray[np.float64]


@dataclass(frozen=True)
class IdentitySegment:
    """Endpoint costs and interior observation costs for one limb-pair interval."""

    start: int
    stop: int
    velocity: NDArray[np.float64]
    position: NDArray[np.float64]
    observation: NDArray[np.float64]


def _fit(
    coordinates: NDArray[np.float64], scores: NDArray[np.float64], times: NDArray[np.float64]
) -> _Fit:
    total = scores.sum(axis=0)
    divisor = np.where(total > 0, total, 1)
    mean_t = np.sum(scores * times[:, None], axis=0) / divisor
    mean_x = np.sum(coordinates * scores[..., None], axis=0) / divisor[:, None]
    centered_t = times[:, None] - mean_t
    denominator = np.sum(scores * centered_t**2, axis=0)
    velocity = np.sum(coordinates * (scores * centered_t)[..., None], axis=0)
    velocity /= np.where(denominator > 0, denominator, 1)[:, None]
    confidence = np.where(denominator > 0, scores.mean(axis=0), 0)
    return _Fit(
        velocity,
        mean_x - velocity * mean_t[:, None],
        confidence,
        np.var(scores, axis=0),
    )


def _evidence(coordinates: NDArray[np.float64], scores: NDArray[np.float64]) -> tuple[float, float]:
    midpoint = (coordinates[:, :3] + coordinates[:, 3:]) / 2
    visible = (scores[:, :3] > 0) & (scores[:, 3:] > 0)
    valid = visible[:-2] & visible[1:-1] & visible[2:]
    acceleration = np.diff(midpoint, n=2, axis=0)[valid].ravel()
    # Midpoint averaging halves variance; a second difference multiplies it by six.
    noise = (
        float(np.median(np.abs(acceleration - np.median(acceleration))))
        / (_NORMAL_MAD * np.sqrt(3))
        if len(acceleration)
        else 0.0
    )
    width = min(_FIT_WINDOW, len(coordinates))
    speeds: list[float] = []
    for start in range(len(coordinates) - width + 1):
        fit = _fit(
            coordinates[start : start + width],
            scores[start : start + width],
            np.arange(width, dtype=np.float64),
        )
        speeds.extend(np.linalg.norm(fit.velocity, axis=-1)[fit.confidence > 0].tolist())
    return noise, float(np.median(speeds)) if speeds else 0.0


def _intervals(ambiguous: NDArray[np.bool_]) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, ambiguous, False]))
    runs: list[tuple[int, int]] = []
    for start, stop in boundaries.reshape(-1, 2):
        if runs and start - runs[-1][1] < _FIT_WINDOW:
            runs[-1] = (runs[-1][0], int(stop))
        else:
            runs.append((int(start), int(stop)))
    return [(start - 1, stop) for start, stop in runs if start > 0 and stop < len(ambiguous)]


def _bridge(
    labelled: NDArray[np.float64],
    confidence: NDArray[np.float64],
    before: _Fit,
    after: _Fit,
    interval: tuple[int, int],
    *,
    noise: float,
) -> IdentitySegment:
    start, stop = interval
    velocity = np.zeros((2, 2), dtype=np.float64)
    observation = np.zeros((stop - start - 1, 2, 2, 2), dtype=np.float64)
    times = np.arange(start + 1, stop, dtype=np.float64)
    fraction = ((times - start) / (stop - start))[:, None]
    for last in range(2):
        right = np.arange(6) if last == 0 else np.array(_SWAP_ORDER)
        velocity[0, last] = np.sum(
            np.linalg.norm(before.velocity - after.velocity[right], axis=-1)
            * before.confidence
            * after.confidence[right]
        )
        predicted = (1 - fraction[..., None]) * (
            before.intercept + times[:, None, None] * before.velocity
        ) + fraction[..., None] * (
            after.intercept[right] + times[:, None, None] * after.velocity[right]
        )
        expected_score = (1 - fraction) * before.confidence + fraction * after.confidence[right]
        score_variance = (1 - fraction) * before.score_variance + fraction * after.score_variance[
            right
        ]
        residual = np.sum((labelled[start + 1 : stop] - predicted[:, None]) ** 2, axis=-1)
        score_residual = (confidence[start + 1 : stop] - expected_score[:, None]) ** 2
        observation[:, 0, last] = np.sum(
            (
                residual / max(noise**2, _VARIANCE_FLOOR)
                + score_residual / np.maximum(score_variance[:, None], _VARIANCE_FLOOR)
            )
            * confidence[start + 1 : stop]
            * expected_score[:, None],
            axis=-1,
        )
    # Complementing every label is exactly the same hypothesis, including in floating point.
    velocity[1] = velocity[0, ::-1]
    observation[:, 1] = observation[:, 0, ::-1, ::-1]
    return IdentitySegment(start, stop, velocity, observation.min(axis=-1).sum(axis=0), observation)


def identity_costs(kp: _FloatArray, scores: _FloatArray) -> list[IdentitySegment]:
    """Build additive path costs for one six-joint limb pair."""
    # Zero-confidence NaNs must contribute zero, rather than poison a whole path.
    coordinates = np.where(np.isfinite(kp), kp, 0).astype(np.float64)
    labelled = np.stack((coordinates, coordinates[:, _SWAP_ORDER]), axis=1)
    confidence = np.stack((scores, scores[:, _SWAP_ORDER]), axis=1).astype(np.float64)
    position = np.sum(
        np.linalg.norm(labelled[1:, None] - labelled[:-1, :, None], axis=-1)
        * confidence[1:, None]
        * confidence[:-1, :, None],
        axis=-1,
    )
    local_path = np.r_[0, np.cumsum(position[:, 0, 1] < position[:, 0, 0]) % 2]
    local = labelled[np.arange(len(kp)), local_path]
    weights = confidence[np.arange(len(kp)), local_path]
    noise, speed = _evidence(local, weights)
    weight = scores[:, :3] * scores[:, 3:]
    total = weight.sum(axis=-1)
    separation = np.sum(
        np.linalg.norm(coordinates[:, :3] - coordinates[:, 3:], axis=-1) * weight, axis=-1
    ) / np.where(total > 0, total, 1)
    ambiguous = (total == 0) | (separation <= _NOISE_RADIUS * noise + _MOTION_RADIUS * speed)
    bridges: dict[int, IdentitySegment] = {}
    for start, stop in _intervals(ambiguous):
        left_start, right_stop = max(0, start - _FIT_WINDOW + 1), min(len(kp), stop + _FIT_WINDOW)
        if start == left_start or stop + 1 == right_stop:
            continue
        before = _fit(
            local[left_start : start + 1, _SWAP_ORDER if local_path[start] else np.arange(6)],
            weights[left_start : start + 1, _SWAP_ORDER if local_path[start] else np.arange(6)],
            np.arange(left_start, start + 1, dtype=np.float64),
        )
        after = _fit(
            local[stop:right_stop, _SWAP_ORDER if local_path[stop] else np.arange(6)],
            weights[stop:right_stop, _SWAP_ORDER if local_path[stop] else np.arange(6)],
            np.arange(stop, right_stop, dtype=np.float64),
        )
        bridges[start] = _bridge(labelled, confidence, before, after, (start, stop), noise=noise)
    segments: list[IdentitySegment] = []
    frame = 0
    while frame < len(kp) - 1:
        segment = bridges.get(frame)
        if segment is None:
            segment = IdentitySegment(
                frame, frame + 1, np.zeros((2, 2)), position[frame], np.empty((0, 2, 2, 2))
            )
        segments.append(segment)
        frame = segment.stop
    return segments


def _pair_path(kp: _FloatArray, scores: _FloatArray) -> NDArray[np.int64]:
    path = np.zeros(len(kp), dtype=np.int64)
    if len(kp) <= 1:
        return path
    segments = identity_costs(kp, scores)
    # Lexicographic costs prevent positional evidence from reversing velocity pairing.
    costs = np.array([[0.0, 0.0], [np.inf, np.inf]])
    predecessors = np.zeros((len(segments), 2), dtype=np.int64)
    for index, segment in enumerate(segments):
        transitions = costs[:, None] + np.stack((segment.velocity, segment.position), axis=-1)
        previous = np.lexsort((transitions[..., 1], transitions[..., 0]), axis=0)[0]
        predecessors[index] = previous
        costs = transitions[previous, np.arange(2)]
    state = int(np.lexsort((costs[:, 1], costs[:, 0]))[0])
    for index in range(len(segments) - 1, -1, -1):
        segment = segments[index]
        previous = predecessors[index, state]
        path[segment.stop] = state
        if segment.stop > segment.start + 1:
            path[segment.start + 1 : segment.stop] = np.argmin(
                segment.observation[:, previous, state], axis=-1
            )
        state = previous
    return path


def _body_relative(kp: _FloatArray, scores: _FloatArray) -> NDArray[np.float64]:
    visible = scores > 0
    body = np.zeros_like(visible)
    body[:, (_NECK, _ROOT_OF_TAIL)] = visible[:, (_NECK, _ROOT_OF_TAIL)]
    # Visible-joint centroids preserve translation invariance when the torso is absent.
    selected = np.where(body.any(axis=1, keepdims=True), body, visible)
    coordinates = kp.astype(np.float64)
    center = np.where(selected[..., None], coordinates, 0).sum(axis=1)
    center /= np.maximum(selected.sum(axis=1, keepdims=True), 1)
    return coordinates - center[:, None]


def _facing(kp: _FloatArray, scores: _FloatArray) -> tuple[Literal["left", "right"], bool]:
    visible = (scores[:, _NOSE] > 0) & (scores[:, _ROOT_OF_TAIL] > 0)
    offset = kp[visible, _NOSE, 0].astype(np.float64) - kp[visible, _ROOT_OF_TAIL, 0]
    return ("left" if len(offset) and np.median(offset) < 0 else "right"), bool(visible.any())


def resolve_identity(
    kp: _FloatArray,
    scores: _FloatArray,
    *,
    frame_width: float,
) -> IdentityResult:
    """Resolve chains by an exact path with fitted-velocity crossing continuity.

    Ambiguity and velocities use body-relative coordinates to remove camera
    translation. Score-weighted five-frame fits pair chains across each
    interval before considering position or score
    residuals within it. Elsewhere, adjacent-frame distance applies. Missing
    observations contribute zero. Near/Far confidence ties keep the first
    frame's labels; absent or tied facing evidence defaults to right. Inputs
    are preserved, including zero-confidence NaN coordinates.
    """
    _validate_inputs(kp, scores, frame_width)
    resolved, confidence = kp.astype(np.float32), scores.astype(np.float32)
    relative = _body_relative(kp, scores)
    swaps_corrected = 0
    for pair in _LIMB_PAIRS:
        path = _pair_path(relative[:, pair], scores[:, pair])
        order = np.tile(np.arange(len(pair)), (len(kp), 1))
        order[path == 1] = _SWAP_ORDER
        pair_scores = np.take_along_axis(scores[:, pair], order, axis=1)
        if len(kp) and np.mean(pair_scores[:, :3]) < np.mean(pair_scores[:, 3:]):
            path = 1 - path
            order = order[:, _SWAP_ORDER]
            pair_scores = pair_scores[:, _SWAP_ORDER]
        resolved[:, pair] = np.take_along_axis(kp[:, pair], order[..., None], axis=1)
        confidence[:, pair] = pair_scores
        swaps_corrected += int(np.count_nonzero(path))
    facing, facing_observed = _facing(kp, scores)
    if facing == "left":
        resolved[..., 0] = frame_width - 1 - resolved[..., 0]
    return IdentityResult(resolved, confidence, swaps_corrected, facing, facing_observed)
