import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.capture.contacts import detect_contacts
from copypaw.capture.travel_root import travel_root_x

_PAWS = (7, 10, 13, 16)


def _tracks(count: int) -> NDArray[np.float32]:
    kp = np.zeros((count, 17, 2), dtype=np.float32)
    kp[:, 3, 0] = 100
    kp[:, _PAWS, 1] = 100
    return kp


def test_irregular_flight_intervals_carry_speed() -> None:
    times = np.array([0, 0.1, 0.2, 1.2])
    kp = _tracks(4)
    kp[:, [3, 4], 0] += (100 * times)[:, None]
    contacts = np.zeros((4, 4), dtype=np.bool_)
    contacts[:2, 0] = True
    result = travel_root_x(kp, contacts, times)
    np.testing.assert_allclose(result, [0, 0.1, 0.2, 1.2])


@pytest.mark.parametrize(
    "times", [[0, 0, 1], [0, 1, 0.5], [0, np.nan, 1], [0, np.inf, 1], [0, 1], [[0, 1, 2]]]
)
def test_invalid_timestamps_are_rejected(times: list[float] | list[list[int]]) -> None:
    with pytest.raises(ValueError, match="timestamps"):
        travel_root_x(_tracks(3), np.ones((3, 4), dtype=np.bool_), np.array(times))


@pytest.mark.parametrize("pan_px_s", [-200, 0, 150, 250])
def test_panning_walk_travel_distance_is_within_three_percent(pan_px_s: float) -> None:
    fps = 30
    count = 6 * fps
    speed_px_s = 150
    times = np.arange(count) / fps
    kp = _tracks(count)
    kp[:, :, 0] += ((speed_px_s - pan_px_s) * times)[:, None]
    for paw, (joint, offset) in enumerate(zip(_PAWS, (0, 15, 7, 22), strict=True)):
        phase = ((np.arange(count) + offset) % fps) / fps
        swing = np.maximum(phase - 0.6, 0) / 0.4
        kp[:, joint, 0] += (100 if paw < 2 else 0) + 25 - 150 * np.minimum(phase, 0.6) + 90 * swing
        kp[:, joint, 1] -= 30 * np.sin(np.pi * swing)
    original = kp.copy()
    contacts = detect_contacts(kp, times, scores=np.ones(kp.shape[:2], dtype=np.float32))
    result = travel_root_x(kp, contacts.contacts, times)
    truth = speed_px_s * times / 100
    assert result[0] == 0
    np.testing.assert_allclose(result[-1], truth[-1], rtol=0.03)
    np.testing.assert_allclose(result, truth, atol=0.03 * truth[-1])
    np.testing.assert_array_equal(kp, original)


def test_averages_only_paws_planted_on_both_frames() -> None:
    kp = _tracks(3)
    kp[:, 7, 0] = [0, -10, 999]
    kp[:, 10, 0] = [0, -30, -40]
    kp[:, 13, 0] = [999, 0, -30]
    kp[:, [3, 4], 0] += (np.arange(3) * 5)[:, None]
    contacts = np.array(
        [[True, True, False, False], [True, True, True, False], [False, True, True, False]]
    )
    original = contacts.copy()
    result = travel_root_x(kp, contacts, np.arange(len(kp)) / 30)
    np.testing.assert_allclose(result, [0, 0.25, 0.5])
    np.testing.assert_array_equal(contacts, original)


def test_flight_keeps_previous_speed_without_including_landing_jump() -> None:
    kp = _tracks(7)
    kp[:, 7, 0] = [0, -10, -100, 80, 200, 180, 160]
    contacts = np.zeros((7, 4), dtype=np.bool_)
    contacts[[0, 1, 4, 5, 6], 0] = True
    result = travel_root_x(kp, contacts, np.arange(len(kp)) / 30)
    np.testing.assert_allclose(result, [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])


@pytest.mark.parametrize("count", [1, 5])
def test_no_contacts_starts_with_zero_speed(count: int) -> None:
    kp = _tracks(count)
    kp[:, :, 0] += np.arange(count)[:, None] * 20
    np.testing.assert_array_equal(
        travel_root_x(kp, np.zeros((count, 4), dtype=np.bool_), np.arange(count) / 30), 0
    )


def test_travel_is_scale_and_translation_invariant_and_can_reverse() -> None:
    kp = _tracks(5)
    kp[:, _PAWS, 0] = -np.array([0, 10, 20, 10, 0])[:, None]
    contacts = np.ones((5, 4), dtype=np.bool_)
    expected = [0, 0.1, 0.2, 0.1, 0]
    np.testing.assert_allclose(
        travel_root_x(kp, contacts, np.arange(len(kp)) / 30), expected, atol=1e-7
    )
    np.testing.assert_allclose(
        travel_root_x(kp * 3 + 500, contacts, np.arange(len(kp)) / 30), expected, atol=1e-7
    )


@pytest.mark.parametrize("shape", [(4, 4), (5, 3), (5,)])
def test_invalid_contact_shape_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"contacts.*shape"):
        travel_root_x(_tracks(5), np.ones(shape, dtype=np.bool_), np.arange(5) / 30)


def test_zero_body_length_is_rejected() -> None:
    kp = _tracks(3)
    kp[:, 3] = kp[:, 4]
    with pytest.raises(ValueError, match="body length"):
        travel_root_x(kp, np.ones((3, 4), dtype=np.bool_), np.arange(3) / 30)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_keypoints_are_rejected(value: float) -> None:
    kp = _tracks(3)
    kp[1, 7, 0] = value
    with pytest.raises(ValueError, match="finite"):
        travel_root_x(kp, np.ones((3, 4), dtype=np.bool_), np.arange(3) / 30)
