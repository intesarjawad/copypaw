from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from copypaw.capture.identity import IdentitySegment, identity_costs, resolve_identity

PAIRS = ((5, 6, 7, 8, 9, 10), (11, 12, 13, 14, 15, 16))
FRAME_WIDTH = 640.0


def _gait(count: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    kp = np.zeros((count, 17, 2), dtype=np.float32)
    kp[..., 0] = 300
    kp[:, 2, 0] = 400
    kp[:, 3, 0] = 400
    kp[:, 4, 0] = 300
    scores = np.full((count, 17), 0.8, dtype=np.float32)
    phase = np.arange(count) * 0.05
    for pair in PAIRS:
        for joint in range(3):
            kp[:, pair[joint], 0] = 350 + 10 * np.sin(phase) + joint * 3
            kp[:, pair[joint + 3], 0] = 350 - 10 * np.sin(phase) + joint * 3
            kp[:, pair[joint], 1] = 100 + joint * 20
            kp[:, pair[joint + 3], 1] = 120 + joint * 20
        scores[:, pair[:3]] = 0.95
        scores[:, pair[3:]] = 0.65
    return kp, scores


def _swap(
    kp: NDArray[np.float32],
    scores: NDArray[np.float32],
    frames: NDArray[np.int64],
    pair: tuple[int, ...],
) -> None:
    for frame in frames:
        kp[frame, pair] = kp[frame, (*pair[3:], *pair[:3])]
        scores[frame, pair] = scores[frame, (*pair[3:], *pair[:3])]


@settings(max_examples=60, deadline=None, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1), groups=st.integers(1, 20))
@pytest.mark.parametrize("separation", [15, 20])
def test_restores_twenty_percent_random_swaps_exactly(
    seed: int, groups: int, separation: int
) -> None:
    count = groups * 5
    expected, confidence = _gait(count)
    body_length = np.linalg.norm(expected[:, 3] - expected[:, 4], axis=1)
    np.testing.assert_allclose(body_length, 100)
    for pair in PAIRS:
        expected[:, pair[3:], 1] = expected[:, pair[:3], 1] + separation
    kp, scores = expected.copy(), confidence.copy()
    rng = np.random.default_rng(seed)
    for pair in PAIRS:
        frames = rng.choice(count, groups, replace=False)
        _swap(kp, scores, frames, pair)

    result = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)

    for pair in PAIRS:
        distance = np.linalg.norm(expected[:, pair[:3]] - expected[:, pair[3:]], axis=2)
        assert np.all(distance / body_length[:, None] >= separation / 100)
    np.testing.assert_array_equal(result.kp, expected)
    np.testing.assert_array_equal(result.scores, confidence)
    assert result.swaps_corrected == groups * 2
    assert result.facing == "right"
    assert result.kp.dtype == np.float32
    assert result.scores.dtype == np.float32


def _greedy(kp: NDArray[np.float32], scores: NDArray[np.float32]) -> NDArray[np.float32]:
    result, confidence = kp.copy(), scores.copy()
    for pair in PAIRS:
        for frame in range(1, len(kp)):
            costs: list[float] = []
            for order in (pair, (*pair[3:], *pair[:3])):
                distance = np.linalg.norm(result[frame - 1, pair] - kp[frame, order], axis=1)
                costs.append(
                    float(np.sum(distance * confidence[frame - 1, pair] * scores[frame, order]))
                )
            if costs[1] < costs[0]:
                _swap(result, confidence, np.array([frame], dtype=np.int64), pair)
    return result


def _noisy_crossing(seed: int, count: int = 81) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    crossing = count // 2
    expected, confidence = _gait(count)
    displacement = (np.arange(count) - crossing) * 2
    for pair in PAIRS:
        expected[:, pair[:3], 0] = 350 + displacement[:, None]
        expected[:, pair[3:], 0] = 350 - displacement[:, None]
        expected[:, pair[3:], 1] = expected[:, pair[:3], 1]
        np.testing.assert_array_equal(expected[crossing, pair[:3]], expected[crossing, pair[3:]])
    expected += np.random.default_rng(seed).normal(0, 2, expected.shape).astype(np.float32)
    return expected, confidence


@pytest.mark.parametrize("seed", range(10))
def test_velocity_continuity_tracks_noisy_crossings_better_than_greedy(seed: int) -> None:
    expected, confidence = _noisy_crossing(seed)
    crossing = len(expected) // 2
    kp, scores = expected.copy(), confidence.copy()
    for pair in PAIRS:
        _swap(kp, scores, np.array([crossing], dtype=np.int64), pair)

    result = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)
    greedy = _greedy(kp, scores)
    correct = np.mean(np.all(result.kp[:, 5:] == expected[:, 5:], axis=(1, 2)))
    greedy_correct = np.mean(np.all(greedy[:, 5:] == expected[:, 5:], axis=(1, 2)))

    assert correct >= 0.95, f"seed={seed}: Viterbi={correct:.3%}, greedy={greedy_correct:.3%}"
    assert correct > greedy_correct


@pytest.mark.parametrize("seed", range(10))
def test_crossing_pairing_does_not_require_different_chain_scores(seed: int) -> None:
    expected, confidence = _noisy_crossing(seed)
    confidence[:] = 0.8
    kp, scores = expected.copy(), confidence.copy()
    for pair in PAIRS:
        _swap(kp, scores, np.array([len(kp) // 2], dtype=np.int64), pair)

    result = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)

    correct = np.mean(np.all(result.kp[:, 5:] == expected[:, 5:], axis=(1, 2)))
    assert correct >= 0.95, f"seed={seed}: accuracy={correct:.3%}"


def test_left_facing_input_is_mirrored_about_the_frame_center() -> None:
    expected, confidence = _gait(20)
    expected[..., 0] = np.rint(expected[..., 0])
    expected[0, 0, 0] = 0
    expected[0, 1, 0] = 639
    # Each joint has its own row, including landmarks on both image boundaries.
    image = np.zeros((len(expected), 17, int(FRAME_WIDTH)), dtype=np.uint8)
    for frame, joint in np.ndindex(expected.shape[:2]):
        image[frame, joint, int(expected[frame, joint, 0])] = 1
    reversed_image = image[..., ::-1]
    kp = expected.copy()
    for frame, joint in np.ndindex(expected.shape[:2]):
        kp[frame, joint, 0] = np.flatnonzero(reversed_image[frame, joint])[0]
    _swap(kp, confidence, np.array([0, 5, 10, 15], dtype=np.int64), PAIRS[0])
    original_kp, original_scores = kp.copy(), confidence.copy()

    result = resolve_identity(kp, confidence, frame_width=FRAME_WIDTH)

    assert result.facing == "left"
    assert result.swaps_corrected == 4
    np.testing.assert_array_equal(result.kp, expected)
    np.testing.assert_array_equal(kp, original_kp)
    np.testing.assert_array_equal(confidence, original_scores)
    assert not np.shares_memory(result.kp, kp)
    assert not np.shares_memory(result.scores, confidence)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("body_visible", [True, False])
def test_camera_translation_preserves_crossing_recovery(seed: int, body_visible: bool) -> None:
    expected, confidence = _noisy_crossing(seed)
    if not body_visible:
        expected[:, 3:5] = np.nan
        confidence[:, 3:5] = 0
    kp, scores = expected.copy(), confidence.copy()
    for pair in PAIRS:
        _swap(kp, scores, np.array([len(kp) // 2], dtype=np.int64), pair)
    translation = np.zeros((len(kp), 1, 2), dtype=np.float32)
    translation[:, 0, 0] = np.arange(len(kp)) * 80

    stationary = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)
    moving = resolve_identity(kp + translation, scores, frame_width=FRAME_WIDTH)

    stationary_correct = np.all(stationary.kp[:, 5:] == expected[:, 5:], axis=(1, 2))
    moving_correct = np.all(moving.kp[:, 5:] == (expected + translation)[:, 5:], axis=(1, 2))
    assert stationary_correct.mean() >= 0.95
    np.testing.assert_array_equal(moving_correct, stationary_correct)
    np.testing.assert_array_equal(moving.scores, stationary.scores)
    assert moving.swaps_corrected == stationary.swaps_corrected


def test_float64_inputs_return_independent_float32_results() -> None:
    kp, scores = _gait(4)
    coordinates, confidence = kp.astype(np.float64), scores.astype(np.float64)
    result = resolve_identity(coordinates, confidence, frame_width=FRAME_WIDTH)

    assert result.kp.dtype == np.float32
    assert result.scores.dtype == np.float32
    np.testing.assert_array_equal(result.kp, kp)
    np.testing.assert_array_equal(result.scores, scores)
    assert not np.shares_memory(result.kp, coordinates)
    assert not np.shares_memory(result.scores, confidence)


@pytest.mark.parametrize("missing", [(2,), (4,), (2, 4)])
def test_facing_default_is_distinguishable_from_observed_right(missing: tuple[int, ...]) -> None:
    kp, scores = _gait(4)
    observed = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)
    kp[:, missing] = np.nan
    scores[:, missing] = 0
    defaulted = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)

    assert observed.facing == defaulted.facing == "right"
    assert observed.facing_observed
    assert not defaulted.facing_observed


def _path_cost(
    kp: NDArray[np.float64],
    scores: NDArray[np.float64],
    path: tuple[int, ...],
) -> float:
    finite = np.where(np.isfinite(kp), kp, 0)
    coordinates = np.stack(
        [finite[t, [0, 1] if state == 0 else [1, 0]] for t, state in enumerate(path)]
    )
    confidence = np.stack(
        [scores[t, [0, 1] if state == 0 else [1, 0]] for t, state in enumerate(path)]
    )
    position = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=-1)
    return float(np.sum(position * confidence[1:] * confidence[:-1]))


def _interval_path_cost(
    segments: list[IdentitySegment], path: tuple[int, ...]
) -> tuple[float, float]:
    velocity, observation = 0.0, 0.0
    for segment in segments:
        first, last = path[segment.start], path[segment.stop]
        velocity += float(segment.velocity[first, last])
        if segment.stop == segment.start + 1:
            observation += float(segment.position[first, last])
        else:
            for offset, frame in enumerate(range(segment.start + 1, segment.stop)):
                observation += float(segment.observation[offset, first, last, path[frame]])
    return velocity, observation


def _assert_exhaustive_minimum(
    kp: NDArray[np.float32], scores: NDArray[np.float32], resolved: NDArray[np.float32]
) -> None:
    # Enumeration checks the solver on body-relative costs; recovery fixtures
    # separately check the evidence against known joint identities.
    center = (kp[:, 3].astype(np.float64) + kp[:, 4]) / 2
    relative = kp.astype(np.float64) - np.nan_to_num(center)[:, None]
    for pair in PAIRS:
        segments = identity_costs(relative[:, pair], scores[:, pair])
        path = tuple(
            int(np.array_equal(resolved[t, pair[:3]], kp[t, pair[3:]])) for t in range(len(kp))
        )
        minimum = min(_interval_path_cost(segments, p) for p in product((0, 1), repeat=len(kp)))
        assert _interval_path_cost(segments, path) == pytest.approx(minimum)


@pytest.mark.parametrize("seed", [1, 5, 8])
def test_interval_objective_matches_exhaustive_paths(seed: int) -> None:
    expected, scores = _noisy_crossing(seed, count=13)
    scores[:, 5:] = np.random.default_rng(seed).uniform(0.4, 1, scores[:, 5:].shape)
    _swap(expected, scores, np.array([2, 6, 9], dtype=np.int64), PAIRS[0])
    result = resolve_identity(expected, scores, frame_width=FRAME_WIDTH)

    for pair in PAIRS:
        assert any(s.stop > s.start + 1 for s in identity_costs(expected[:, pair], scores[:, pair]))
    _assert_exhaustive_minimum(expected, scores, result.kp)


@pytest.mark.parametrize("count", [2, 3, 6])
def test_matches_exhaustive_minimum_with_confidence_weighted_costs(count: int) -> None:
    kp, scores = _gait(count)
    rng = np.random.default_rng(count)
    kp[:, 5:] = rng.normal(0, 10, kp[:, 5:].shape)
    scores[:, 5:] = rng.uniform(0.1, 1, scores[:, 5:].shape)
    result = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)

    for pair in PAIRS:
        chains = kp[:, pair].reshape(count, 2, 3, 2).astype(np.float64)
        confidence = scores[:, pair].reshape(count, 2, 3).astype(np.float64)
        path = tuple(
            int(np.array_equal(result.kp[t, pair[:3]], kp[t, pair[3:]])) for t in range(count)
        )
        minimum = min(_path_cost(chains, confidence, p) for p in product((0, 1), repeat=count))
        assert _path_cost(chains, confidence, path) == pytest.approx(minimum)
        assert np.mean(result.scores[:, pair[:3]]) >= np.mean(result.scores[:, pair[3:]])


@pytest.mark.parametrize("count", [0, 1, 2])
def test_short_clips_and_missing_observations(count: int) -> None:
    kp, scores = _gait(count)
    kp[:] = np.nan
    scores[:] = 0
    result = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)
    np.testing.assert_array_equal(result.kp, kp)
    np.testing.assert_array_equal(result.scores, scores)
    assert result.facing == "right"
    assert not result.facing_observed
    assert result.swaps_corrected == 0


def test_missing_frame_does_not_introduce_nan_transition_costs() -> None:
    kp, scores = _gait(5)
    kp[2] = np.nan
    scores[2] = 0
    _swap(kp, scores, np.array([4], dtype=np.int64), PAIRS[0])
    result = resolve_identity(kp, scores, frame_width=FRAME_WIDTH)
    assert np.isnan(result.kp[2]).all()
    np.testing.assert_array_equal(result.scores[2], 0)
    assert np.isfinite(result.kp[[0, 1, 3, 4]]).all()
    _assert_exhaustive_minimum(kp, scores, result.kp)


@pytest.mark.parametrize("width", [0, -1, np.inf, np.nan])
def test_invalid_frame_width_is_rejected(width: float) -> None:
    kp, scores = _gait(3)
    with pytest.raises(ValueError, match="frame_width"):
        resolve_identity(kp, scores, frame_width=width)


def test_invalid_shapes_are_rejected() -> None:
    kp, scores = _gait(3)
    with pytest.raises(ValueError, match="shape"):
        resolve_identity(kp[:, :16], scores, frame_width=FRAME_WIDTH)
    with pytest.raises(ValueError, match="shape"):
        resolve_identity(kp, scores[:2], frame_width=FRAME_WIDTH)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_scores_are_rejected(value: float) -> None:
    kp, scores = _gait(3)
    scores[0, 5] = value
    with pytest.raises(ValueError, match="scores"):
        resolve_identity(kp, scores, frame_width=FRAME_WIDTH)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_visible_coordinates_must_be_finite(value: float) -> None:
    kp, scores = _gait(3)
    kp[0, 5, 0] = value
    with pytest.raises(ValueError, match="coordinates"):
        resolve_identity(kp, scores, frame_width=FRAME_WIDTH)
