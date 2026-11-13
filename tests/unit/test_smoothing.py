import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from copypaw.capture.smoothing import clean


def _tracks(count: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    return np.zeros((count, 17, 2), dtype=np.float32), np.ones((count, 17), dtype=np.float32)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(
    seed=st.integers(0, 2**32 - 1),
    phase=st.floats(-np.pi, np.pi),
    amplitude=st.floats(5, 20),
)
def test_sine_with_twenty_percent_dropped_points_recovers_within_two_pixels(
    seed: int, phase: float, amplitude: float
) -> None:
    kp, scores = _tracks(150)
    times = np.arange(len(kp)) / 30
    kp[..., 0] = (100 + amplitude * np.sin(2 * np.pi * 0.25 * times + phase))[:, None]
    kp[..., 1] = (200 + amplitude * np.cos(2 * np.pi * 0.25 * times + phase))[:, None]
    expected = kp.copy()
    rng = np.random.default_rng(seed)
    for joint in range(17):
        missing = rng.choice(len(kp), len(kp) // 5, replace=False)
        kp[missing, joint] = np.nan
        scores[missing, joint] = 0
        observed = np.flatnonzero(scores[:, joint] >= 0.3)
        expected[: observed[0], joint] = expected[observed[0], joint]
        expected[observed[-1] + 1 :, joint] = expected[observed[-1], joint]

    result = clean(kp, scores, np.arange(len(kp)) / 30).keypoints

    assert np.isfinite(result).all()
    assert np.max(np.linalg.norm(result - expected, axis=-1)) <= 2


@pytest.mark.parametrize("fps", [12, 20, 24, 30, 60, 120])
def test_four_hz_gait_keeps_at_least_ninety_percent_amplitude(fps: int) -> None:
    kp, scores = _tracks(300)
    sine = np.sin(2 * np.pi * 4 * np.arange(len(kp)) / fps).astype(np.float32)
    kp[..., 0] = sine[:, None]
    result = clean(kp, scores, np.arange(len(kp)) / fps).keypoints
    # Project onto the known frequency; omit edge fits from the steady-state gain.
    gain = np.dot(result[2:-2, 0, 0], sine[2:-2]) / np.dot(sine[2:-2], sine[2:-2])
    assert 0.9 <= gain <= 1


def test_unobserved_joint_stays_missing() -> None:
    kp, scores = _tracks(20)
    kp[:, 3] = np.nan
    scores[:, 3] = 0
    result = clean(kp, scores, np.arange(len(kp)) / 30).keypoints
    assert np.isnan(result[:, 3]).all()
    cleaned = clean(kp, scores, np.arange(len(kp)) / 30)
    assert cleaned.missing_mask.shape == scores.shape
    assert cleaned.missing_mask[:, 3].all()
    assert not cleaned.missing_mask[:, [0, 1, 2, 4]].any()


@pytest.mark.parametrize("reverse", [False, True])
def test_boundary_gaps_hold_the_nearest_observation(reverse: bool) -> None:
    kp, scores = _tracks(4)
    kp[:, 0, 0] = [np.nan, np.nan, 10, 20]
    scores[:2, 0] = 0
    expected = np.array([10, 10, 10, 20])
    if reverse:
        kp, scores, expected = kp[::-1].copy(), scores[::-1].copy(), expected[::-1]
    result = clean(kp, scores, np.arange(len(kp)) / 30).keypoints
    np.testing.assert_array_equal(result[:, 0, 0], expected)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_short_clips_interpolate_each_joint_and_hold_endpoints(count: int) -> None:
    kp, scores = _tracks(count)
    kp[:] = np.nan
    scores[:] = 0
    kp[0, 0] = [10, 20]
    scores[0, 0] = 0.3
    kp[-1, 1] = [30, 40]
    scores[-1, 1] = 1
    result = clean(kp, scores, np.arange(len(kp)) / 30).keypoints
    np.testing.assert_array_equal(result[:, 0], np.tile([10, 20], (count, 1)))
    np.testing.assert_array_equal(result[:, 1], np.tile([30, 40], (count, 1)))
    assert np.isnan(result[:, 2:]).all()


def test_masks_low_scores_interpolates_and_preserves_inputs() -> None:
    kp, scores = _tracks(4)
    kp[:, 0, 0] = [0, 999, 999, 9]
    scores[1:3, 0] = 0.49
    original_kp, original_scores = kp.copy(), scores.copy()
    result = clean(kp, scores, np.arange(len(kp)) / 30, min_score=0.5).keypoints
    np.testing.assert_array_equal(result[:, 0, 0], [0, 3, 6, 9])
    np.testing.assert_array_equal(kp, original_kp)
    np.testing.assert_array_equal(scores, original_scores)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, kp)


def test_long_boundary_gaps_hold_endpoints_after_smoothing() -> None:
    kp, scores = _tracks(10)
    expected = np.arange(10, dtype=np.float32) * 3
    kp[..., 0] = expected[:, None]
    kp[[0, 1, 8, 9]] = np.nan
    scores[[0, 1, 8, 9]] = 0
    result = clean(kp, scores, np.arange(len(kp)) / 30).keypoints
    np.testing.assert_array_equal(result[:2, 0, 0], [6, 6])
    np.testing.assert_array_equal(result[-2:, 0, 0], [21, 21])


@pytest.mark.parametrize("count", [0, 5, 20])
def test_no_observations_returns_missing_tracks(count: int) -> None:
    kp, scores = _tracks(count)
    kp[:] = np.nan
    scores[:] = 0
    result = clean(kp, scores, np.arange(len(kp)) / 30).keypoints
    assert result.shape == kp.shape
    assert np.isnan(result).all()


def test_smoothing_reduces_alternating_jitter() -> None:
    kp, scores = _tracks(30)
    kp[..., 0] = (np.arange(30) % 2)[:, None]
    result = clean(kp, scores, np.arange(len(kp)) / 30).keypoints
    assert np.std(result[2:-2, 0, 0]) < np.std(kp[2:-2, 0, 0]) / 2


def test_interpolation_uses_elapsed_seconds() -> None:
    kp, scores = _tracks(4)
    kp[:, 0, 0] = [0, np.nan, np.nan, 12]
    scores[1:3, 0] = 0
    times = np.array([0, 0.1, 0.2, 1.2])
    np.testing.assert_allclose(clean(kp, scores, times).keypoints[:, 0, 0], [0, 1, 2, 12])


@pytest.mark.parametrize("fps", [12, 24, 30, 60, 120])
@pytest.mark.parametrize("phase", [0, 0.7, 1.8])
def test_irregular_samples_preserve_four_hz_amplitude(fps: int, phase: float) -> None:
    kp, scores = _tracks(300)
    rng = np.random.default_rng(42)
    times = np.cumsum(rng.uniform(0.7, 1.3, len(kp)) / fps)
    sine = np.sin(2 * np.pi * 4 * times + phase)
    kp[..., 0] = sine[:, None]
    result = clean(kp, scores, times).keypoints
    gain = np.dot(result[:, 0, 0], sine) / np.dot(sine, sine)
    assert gain >= 0.9


@pytest.mark.parametrize(
    "times", [[0, 0, 1], [0, 1, 0.5], [0, np.nan, 1], [0, np.inf, 1], [0, 1], [[0, 1, 2]]]
)
def test_invalid_timestamps_are_rejected(times: list[float] | list[list[int]]) -> None:
    kp, scores = _tracks(3)
    with pytest.raises(ValueError, match="timestamps"):
        clean(kp, scores, np.array(times))


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_threshold_is_rejected(value: float) -> None:
    kp, scores = _tracks(3)
    with pytest.raises(ValueError, match="min_score"):
        clean(kp, scores, np.arange(len(kp)) / 30, min_score=value)


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_invalid_scores_are_rejected(value: float) -> None:
    kp, scores = _tracks(3)
    scores[0, 0] = value
    with pytest.raises(ValueError, match="scores"):
        clean(kp, scores, np.arange(len(kp)) / 30)


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_confident_coordinates_must_be_finite(value: float) -> None:
    kp, scores = _tracks(3)
    kp[0, 0, 0] = value
    with pytest.raises(ValueError, match="coordinates"):
        clean(kp, scores, np.arange(len(kp)) / 30)


def test_invalid_shapes_are_rejected() -> None:
    kp, scores = _tracks(3)
    with pytest.raises(ValueError, match="shape"):
        clean(kp[:, :16], scores, np.arange(len(kp)) / 30)
    with pytest.raises(ValueError, match="shape"):
        clean(kp, scores[:2], np.arange(len(kp)) / 30)
