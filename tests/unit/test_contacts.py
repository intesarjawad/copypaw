import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.capture.contacts import detect_contacts
from copypaw.capture.quality import assess_quality
from copypaw.capture.smoothing import clean
from copypaw.capture.travel_root import travel_root_x

_PAWS = (7, 10, 13, 16)


def _walk(
    fps: int, *, tilt_degrees: float = 0, clearance_px: float = 30
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    count = 6 * fps
    times = np.arange(count) / fps
    kp = np.zeros((count, 17, 2), dtype=np.float32)
    kp[:, 4, 0] = 100 + 150 * times
    kp[:, 4, 1] = 100
    kp[:, 3] = kp[:, 4] + [100, 0]
    expected = np.zeros((count, 4), dtype=np.bool_)
    for paw, (joint, offset) in enumerate(
        zip(_PAWS, (0, fps // 2, fps // 4, 3 * fps // 4), strict=True)
    ):
        phase = ((np.arange(count) + offset) % fps) / fps
        stance = phase < 0.6
        swing = np.maximum(phase - 0.6, 0) / 0.4
        kp[:, joint, 0] = (
            kp[:, 4, 0] + (100 if paw < 2 else 0) + 25 - 150 * np.minimum(phase, 0.6) + 90 * swing
        )
        kp[:, joint, 1] = 200 - clearance_px * np.sin(np.pi * swing)
        expected[:, paw] = stance
    angle = np.deg2rad(tilt_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (kp @ rotation.T).astype(np.float32), expected


def _standing(count: int) -> NDArray[np.float32]:
    kp = np.zeros((count, 17, 2), dtype=np.float32)
    kp[:, 4] = [0, 100]
    kp[:, 3] = [100, 100]
    kp[:, _PAWS, 0] = [80, 90, 0, 10]
    kp[:, _PAWS, 1] = 200
    return kp


def _onsets(contact: NDArray[np.bool_]) -> NDArray[np.int64]:
    return np.flatnonzero(np.diff(np.r_[False, contact].astype(np.int8)) == 1)


def test_rotated_standing_clip_keeps_all_four_contacts() -> None:
    kp = _standing(20)
    angle = np.deg2rad(5)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result = detect_contacts(
        (kp @ rotation.T).astype(np.float32),
        np.arange(len(kp)) / 30,
        scores=np.ones(kp.shape[:2], dtype=np.float32),
    )
    assert result.contacts.all()
    np.testing.assert_allclose(
        result.ground_line_px, [np.tan(angle), 200 / np.cos(angle)], atol=1e-4
    )


@pytest.mark.parametrize(
    ("joint", "name"),
    [
        (3, "Neck"),
        (4, "Root_of_tail"),
        (7, "Left_front_paw"),
        (10, "Right_front_paw"),
        (13, "Left_back_paw"),
        (16, "Right_back_paw"),
    ],
)
def test_unobserved_required_landmarks_block_geometry(joint: int, name: str) -> None:
    kp = _standing(20)
    kp[..., 0] += 300
    kp[:, 7, 1] -= 15
    scores = np.ones(kp.shape[:2], dtype=np.float32)
    kp[:, joint] = np.nan
    scores[:, joint] = 0
    times = np.arange(len(kp)) / 30
    cleaned = clean(kp, scores, times)
    with pytest.raises(ValueError, match=rf"Missing required landmark.*{name}"):
        detect_contacts(cleaned.keypoints, times, scores=scores)
    with pytest.raises(ValueError, match=rf"Missing required landmark.*{name}"):
        travel_root_x(cleaned.keypoints, np.ones((20, 4), dtype=np.bool_), times)
    assert any(name in error for error in assess_quality(scores, times).errors)


def test_missing_optional_landmarks_do_not_block_required_geometry() -> None:
    kp = _standing(20)
    kp[:, 0] = np.nan
    times = np.arange(len(kp)) / 30
    scores = np.ones(kp.shape[:2], dtype=np.float32)
    scores[:, 0] = 0
    assert detect_contacts(kp, times, scores=scores).contacts.all()
    np.testing.assert_array_equal(travel_root_x(kp, np.ones((20, 4), dtype=np.bool_), times), 0)


def test_ground_fit_uses_only_confident_paw_observations() -> None:
    kp = _standing(20)
    scores = np.ones(kp.shape[:2], dtype=np.float32)
    kp[10:, _PAWS, 1] = 400
    scores[10:, _PAWS] = 0
    result = detect_contacts(kp, np.arange(len(kp)) / 30, scores=scores)
    np.testing.assert_allclose(result.ground_line_px, [0, 200], atol=1e-5)


def test_contact_speeds_use_irregular_elapsed_seconds() -> None:
    kp = _standing(4)
    times = np.array([0, 0.1, 0.2, 1.2])
    kp[:, 7, 0] += 20 * times
    result = detect_contacts(kp, times, scores=np.ones(kp.shape[:2], dtype=np.float32))
    assert result.contacts.all()


def test_no_stationary_paw_evidence_leaves_ground_unavailable() -> None:
    kp = _standing(4)
    kp[:, _PAWS, 1] += np.arange(4)[:, None] * 20
    result = detect_contacts(kp, np.arange(4) / 30, scores=np.ones((4, 17), dtype=np.float32))
    assert not result.contacts.any()
    assert np.isnan(result.ground_line_px).all()


@pytest.mark.parametrize("fps", [20, 30, 60])
@pytest.mark.parametrize("tilt_degrees", [-5, 0, 5])
def test_walk_has_perfect_onset_f1_within_one_frame(fps: int, tilt_degrees: float) -> None:
    kp, expected = _walk(fps, tilt_degrees=tilt_degrees)
    original = kp.copy()
    result = detect_contacts(
        kp, np.arange(len(kp)) / fps, scores=np.ones(kp.shape[:2], dtype=np.float32)
    )
    for paw in range(4):
        actual_onsets = _onsets(result.contacts[:, paw])
        expected_onsets = _onsets(expected[:, paw])
        matches = np.abs(actual_onsets[:, None] - expected_onsets[None, :]) <= 1
        assert matches.sum(axis=0).tolist() == [1] * len(expected_onsets)
        assert matches.sum(axis=1).tolist() == [1] * len(actual_onsets)
        f1 = 2 * np.count_nonzero(matches) / (len(actual_onsets) + len(expected_onsets))
        assert f1 == 1
    np.testing.assert_allclose(
        result.ground_line_px[0], np.tan(np.deg2rad(tilt_degrees)), atol=1e-5
    )
    np.testing.assert_array_equal(kp, original)
    assert result.contacts.dtype == np.bool_


def test_low_clearance_forward_swing_is_not_contact() -> None:
    kp, expected = _walk(30, clearance_px=3)
    result = detect_contacts(
        kp, np.arange(len(kp)) / 30, scores=np.ones(kp.shape[:2], dtype=np.float32)
    )
    assert not result.contacts[~expected].any()
    assert result.contacts[expected].mean() > 0.9


def test_ground_fit_rejects_outliers_on_both_sides() -> None:
    kp = _standing(50)
    x = np.arange(len(kp)) * 4
    kp[:, _PAWS, 0] = x[:, None]
    kp[:, _PAWS, 1] = (0.08 * x + 200)[:, None]
    kp[[4, 15, 28], :, 1] += 80
    kp[[9, 22, 39], :, 1] -= 80
    result = detect_contacts(
        kp, np.arange(len(kp)) / 30, scores=np.ones(kp.shape[:2], dtype=np.float32)
    )
    np.testing.assert_allclose(result.ground_line_px, [0.08, 200], atol=1e-5)


def test_ground_without_any_horizontal_span_uses_horizontal_median() -> None:
    kp = _standing(10)
    kp[:, _PAWS, 0] = 0
    kp[:, _PAWS, 1] += np.arange(10)[:, None]
    result = detect_contacts(
        kp, np.arange(len(kp)) / 30, scores=np.ones(kp.shape[:2], dtype=np.float32)
    )
    np.testing.assert_array_equal(result.ground_line_px, [0, 204.5])


def test_removes_single_frame_contacts_then_fills_internal_single_frame_gaps() -> None:
    kp = _standing(20)
    kp[:, 7, 1] = 180
    kp[3:6, 7, 1] = 200  # Only the middle sample has low vertical speed.
    kp[9:19, 7, 1] = 194.5
    kp[14, 7, 1] = 193.5
    result = detect_contacts(
        kp, np.arange(len(kp)) / 30, scores=np.ones(kp.shape[:2], dtype=np.float32)
    )
    assert not result.contacts[:9, 0].any()
    assert result.contacts[10:18, 0].all()
    assert not result.contacts[19, 0]


@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize("step_px", [29, 30, 31])
@pytest.mark.parametrize("coordinate", [0, 1])
def test_speed_thresholds_are_strict_and_in_seconds(
    fps: int, step_px: float, coordinate: int
) -> None:
    kp = _standing(3)
    kp[:, 3, 0] = fps * 100
    # Integer coordinate steps keep the threshold equality exact in float32.
    kp[:, _PAWS, coordinate] += (np.arange(3) * step_px)[:, None]
    result = detect_contacts(
        kp, np.arange(len(kp)) / fps, scores=np.ones(kp.shape[:2], dtype=np.float32)
    )
    assert result.contacts.all() == (step_px < 30)


@pytest.mark.parametrize("height_px", [5, 6, 7])
def test_height_threshold_is_strict(height_px: float) -> None:
    kp = _standing(5)
    kp[:, 7, 1] -= height_px
    assert detect_contacts(
        kp, np.arange(len(kp)) / 30, scores=np.ones(kp.shape[:2], dtype=np.float32)
    ).contacts[:, 0].all() == (height_px < 6)


def test_ground_and_contacts_scale_with_the_image_and_use_median_body_length() -> None:
    kp, _ = _walk(30)
    # A binary-exact shear keeps the affine transform exact in float32.
    kp[..., 1] += kp[..., 0] / 8
    expected = detect_contacts(
        kp, np.arange(len(kp)) / 30, scores=np.ones(kp.shape[:2], dtype=np.float32)
    )
    kp[0, 3] = kp[0, 4]  # A degenerate frame must not change the clip's median scale.
    scaled = detect_contacts(
        (kp * 3 + [500, 300]).astype(np.float32),
        np.arange(len(kp)) / 30,
        scores=np.ones(kp.shape[:2], dtype=np.float32),
    )
    np.testing.assert_array_equal(scaled.contacts, expected.contacts)
    slope, intercept = expected.ground_line_px
    np.testing.assert_allclose(scaled.ground_line_px, [slope, intercept * 3 + 300 - slope * 500])


@pytest.mark.parametrize("count", [1, 2, 3])
def test_short_static_clips(count: int) -> None:
    result = detect_contacts(
        _standing(count), np.arange(count) / 30, scores=np.ones((count, 17), dtype=np.float32)
    )
    np.testing.assert_array_equal(result.contacts, np.full((count, 4), count >= 2))
    np.testing.assert_array_equal(result.ground_line_px, [0, 200])


@pytest.mark.parametrize("times", [[0, 0, 0.1], [0, 0.1, 0.05], [0, np.inf, 1], [0, np.nan, 1]])
def test_invalid_timestamps_are_rejected(times: list[float]) -> None:
    with pytest.raises(ValueError, match="timestamps"):
        detect_contacts(_standing(3), np.array(times), scores=np.ones((3, 17), dtype=np.float32))


@pytest.mark.parametrize("shape", [(3, 16, 2), (17, 2), (3, 17, 3)])
def test_invalid_shape_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        detect_contacts(
            np.zeros(shape, dtype=np.float32),
            np.arange(3) / 30,
            scores=np.ones((3, 17), dtype=np.float32),
        )


def test_empty_clip_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        detect_contacts(_standing(0), np.empty(0), scores=np.ones((0, 17), dtype=np.float32))


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_coordinates_are_rejected(value: float) -> None:
    kp = _standing(3)
    kp[1, 7, 0] = value
    with pytest.raises(ValueError, match="finite"):
        detect_contacts(kp, np.arange(len(kp)) / 30, scores=np.ones(kp.shape[:2], dtype=np.float32))


def test_zero_body_length_is_rejected() -> None:
    kp = _standing(3)
    kp[:, 3] = kp[:, 4]
    with pytest.raises(ValueError, match="body length"):
        detect_contacts(kp, np.arange(len(kp)) / 30, scores=np.ones(kp.shape[:2], dtype=np.float32))
