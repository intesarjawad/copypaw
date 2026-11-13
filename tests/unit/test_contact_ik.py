"""Tests for contact paw filtering."""

import math
from importlib import import_module
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from copypaw.animate.bake import BakeMode
from copypaw.animate.contact_ik import Filter, make_contact_filter

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CHARACTER = FIXTURES / "upstream/characters/char6/char_cfg.yaml"

# Depth-first order the engine reports for the upstream four-legged character.
JOINT_ORDER = (
    "root",
    "hip",
    "torso",
    "neck",
    "right_hip_b",
    "right_knee_b",
    "right_foot_b",
    "left_hip_b",
    "left_knee_b",
    "left_foot_b",
    "right_hip_f",
    "right_knee_f",
    "right_foot_f",
    "left_hip_f",
    "left_knee_f",
    "left_foot_f",
)
_HIP = JOINT_ORDER.index("left_hip_f")
_KNEE = JOINT_ORDER.index("left_knee_f")
_FOOT = JOINT_ORDER.index("left_foot_f")
_NEAR_FRONT = 0
_BASE = 0.5


def _frame(
    hip: tuple[float, float],
    knee: tuple[float, float],
    foot: tuple[float, float],
) -> npt.NDArray[np.float32]:
    positions = np.full((len(JOINT_ORDER), 2), _BASE, dtype=np.float32)
    positions[_HIP] = hip
    positions[_KNEE] = knee
    positions[_FOOT] = foot
    return positions


def _cross(
    hip: npt.NDArray[np.float32], knee: npt.NDArray[np.float32], foot: npt.NDArray[np.float32]
) -> float:
    first = np.asarray(knee, dtype=np.float64) - np.asarray(hip, dtype=np.float64)
    second = np.asarray(foot, dtype=np.float64) - np.asarray(knee, dtype=np.float64)
    return float(first[0] * second[1] - first[1] * second[0])


def _segment_lengths(
    hip: npt.NDArray[np.float32], knee: npt.NDArray[np.float32], foot: npt.NDArray[np.float32]
) -> tuple[float, float]:
    return (
        float(np.linalg.norm(np.asarray(knee, dtype=np.float64) - hip)),
        float(np.linalg.norm(np.asarray(foot, dtype=np.float64) - knee)),
    )


def _run(filter_: Filter, frames: list[npt.NDArray[np.float32]]) -> list[npt.NDArray[np.float32]]:
    return [filter_(index, frame) for index, frame in enumerate(frames)]


def test_joint_order_matches_engine_traversal() -> None:
    config = import_module("copypaw.vendor.animated_drawings.config")
    model = import_module("copypaw.vendor.animated_drawings.model.animated_drawing")
    character = config.CharacterConfig(str(CHARACTER))
    rig = model.AnimatedDrawingRig(character)
    assert tuple(rig.root_joint.get_chain_joint_names()) == JOINT_ORDER


def _median_ground_scene() -> tuple[list[npt.NDArray[np.float32]], npt.NDArray[np.bool_]]:
    frames = [
        _frame((0.15, 0.8), (0.2, 0.5), (0.2, 0.7)),
        _frame((0.15, 0.8), (0.2, 0.5), (0.2, 0.3)),
        _frame((0.15, 0.8), (0.2, 0.5), (0.2, 0.4)),
        _frame((0.15, 0.8), (0.2, 0.5), (0.2, 0.0)),
    ]
    contacts = np.zeros((len(frames), 4), dtype=np.bool_)
    contacts[:3, _NEAR_FRONT] = True
    return frames, contacts


def test_drawing_settles_until_its_contact_paw_stands_on_its_ground() -> None:
    frames, contacts = _median_ground_scene()
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(filter_, frames)
    filtered = _run(filter_, frames)

    # The root stays at 0.5 and the planted paw reaches down to 0.3 at its lowest.
    ground = 0.3
    assert filter_.lowest_ground() == pytest.approx(ground)
    for frame, row in zip(frames[:3], filtered[:3], strict=True):
        assert float(row[_FOOT, 1]) == pytest.approx(ground, abs=1e-6)
        # Every joint moves by the same amount, so no knee bends to plant the paw.
        np.testing.assert_allclose(
            row - frame, np.tile(row[_FOOT] - frame[_FOOT], (16, 1)), atol=1e-6
        )
    # Leaving contact, only that leg moves: its paw is halfway back to the pose.
    assert float(filtered[3][_FOOT, 1]) == pytest.approx((ground + frames[3][_FOOT, 1]) / 2)
    np.testing.assert_array_equal(filtered[3][:_KNEE], frames[3][:_KNEE])


def test_paws_drawn_at_different_heights_keep_their_own_ground() -> None:
    far_foot = JOINT_ORDER.index("right_foot_f")
    frames = [_frame((0.2, 0.8), (0.2, 0.5), (0.2, 0.2)) for _ in range(2)]
    for frame in frames:
        frame[JOINT_ORDER.index("right_hip_f")] = (0.3, 0.8)
        frame[JOINT_ORDER.index("right_knee_f")] = (0.3, 0.6)
        frame[far_foot] = (0.3, 0.3)
    contacts = np.zeros((2, 4), dtype=np.bool_)
    contacts[:, :2] = True
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(filter_, frames)
    filtered = _run(filter_, frames)

    for row in filtered:
        assert float(row[_FOOT, 1]) == pytest.approx(0.2, abs=1e-3)
        assert float(row[far_foot, 1]) == pytest.approx(0.3, abs=1e-3)
    assert filter_.lowest_ground() == pytest.approx(0.2)


def test_legs_at_one_end_slide_at_the_hip_onto_their_ground_without_bending() -> None:
    far = [JOINT_ORDER.index(f"right_{part}_f") for part in ("hip", "knee", "foot")]
    frames = [_frame((0.2, 0.8), (0.2, 0.5), (0.2, 0.2)) for _ in range(2)]
    for frame, far_height in zip(frames, (0.2, 0.3), strict=True):
        frame[far[0]] = (0.3, 0.8)
        frame[far[1]] = (0.3, 0.5)
        frame[far[2]] = (0.3, far_height)
    contacts = np.zeros((2, 4), dtype=np.bool_)
    contacts[:, :2] = True
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(filter_, frames)
    filtered = _run(filter_, frames)[1]

    # The far leg is posed 0.1 short of its ground; the body takes half of that
    # and each hip slides the rest, so both legs keep their posed shape.
    assert float(filtered[0, 1]) == pytest.approx(_BASE - 0.05, abs=1e-6)
    assert float(filtered[_HIP, 1]) == pytest.approx(0.8, abs=1e-6)
    assert float(filtered[far[0], 1]) == pytest.approx(0.7, abs=1e-6)
    assert float(filtered[_FOOT, 1]) == pytest.approx(0.2, abs=1e-3)
    assert float(filtered[far[2], 1]) == pytest.approx(0.2, abs=1e-3)
    for hip, foot in ((_HIP, _FOOT), (far[0], far[2])):
        posed = float(np.hypot(*(frames[1][foot] - frames[1][hip])))
        assert float(np.hypot(*(filtered[foot] - filtered[hip]))) == pytest.approx(posed, rel=2e-3)


def test_body_tips_so_a_paw_at_each_end_stands_without_bending_a_leg() -> None:
    back = [JOINT_ORDER.index(f"left_{part}_b") for part in ("hip", "knee", "foot")]
    frames = [_frame((0.8, 0.8), (0.8, 0.5), (0.8, 0.2)) for _ in range(2)]
    for frame, back_foot in zip(frames, ((0.2, 0.2), (0.5, 0.3)), strict=True):
        frame[JOINT_ORDER.index("right_hip_f")] = (0.8, 0.8)
        frame[JOINT_ORDER.index("right_hip_b")] = (0.2, 0.8)
        frame[back[0]] = (0.2, 0.8)
        frame[back[1]] = (0.2 + (back_foot[0] - 0.2) / 2, 0.8 + (back_foot[1] - 0.8) / 2)
        frame[back[2]] = back_foot
    contacts = np.zeros((2, 4), dtype=np.bool_)
    contacts[:, _NEAR_FRONT] = True
    contacts[:, 2] = True
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(filter_, frames)
    filtered = _run(filter_, frames)[1]

    # The back leg swings 0.1 off the ground; only the back end comes down to it.
    np.testing.assert_allclose(filtered[back[2]], (0.5, 0.2), atol=1e-3)
    np.testing.assert_allclose(filtered[back[0]], (0.2, 0.7), atol=1e-6)
    np.testing.assert_allclose(filtered[_HIP], frames[1][_HIP], atol=1e-6)
    np.testing.assert_allclose(filtered[_FOOT], frames[1][_FOOT], atol=1e-3)
    # Straight as posed, less the solver's 0.999 reach.
    posed = float(np.hypot(*(frames[1][back[2]] - frames[1][back[0]])))
    assert float(np.hypot(*(filtered[back[2]] - filtered[back[0]]))) == pytest.approx(
        posed, rel=2e-3
    )


def test_contact_segments_and_knee_side_are_preserved() -> None:
    frames, contacts = _median_ground_scene()
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(filter_, frames)
    filtered = _run(filter_, frames)

    for frame in range(3):
        before = _segment_lengths(frames[frame][_HIP], frames[frame][_KNEE], frames[frame][_FOOT])
        after = _segment_lengths(
            filtered[frame][_HIP], filtered[frame][_KNEE], filtered[frame][_FOOT]
        )
        np.testing.assert_allclose(after, before, atol=1e-4)
        assert np.sign(_cross(frames[frame][_HIP], frames[frame][_KNEE], frames[frame][_FOOT])) == (
            np.sign(_cross(filtered[frame][_HIP], filtered[frame][_KNEE], filtered[frame][_FOOT]))
        )


@pytest.mark.parametrize("mode", ["treadmill", "travel"])
def test_wrong_contact_labels_cannot_pull_a_leg_off_the_body(mode: BakeMode) -> None:
    rng = np.random.default_rng(7)
    frames: list[npt.NDArray[np.float32]] = []
    for step in range(40):
        swing = math.sin(step / 3)
        frame = _frame(
            (0.2, 0.8), (0.2 + 0.2 * swing, 0.5), (0.2 + 0.4 * swing, 0.2 + 0.3 * swing**2)
        )
        # The torso stands in for the body right where this leg joins it.
        frame[JOINT_ORDER.index("torso")] = frame[_HIP]
        frame[:, 0] += 0.05 * step
        frames.append(frame)
    # Contact flags unrelated to the motion: the paw is "planted" while it swings and lifts.
    contacts = rng.random((len(frames), 4)) < 0.6
    filter_ = make_contact_filter(JOINT_ORDER, contacts, mode)
    _run(filter_, frames)

    for frame, row in zip(frames, _run(filter_, frames), strict=True):
        leg = float(np.hypot(*(frame[_FOOT] - frame[_HIP])))
        body = row[JOINT_ORDER.index("torso")] - frame[JOINT_ORDER.index("torso")]
        slid = (row[_HIP] - frame[_HIP]) - body
        assert float(np.abs(slid).max()) <= 0.15 * leg + 1e-6
        # The leg is never stretched to reach a paw that was not really planted.
        assert (
            sum(_segment_lengths(row[_HIP], row[_KNEE], row[_FOOT]))
            <= sum(_segment_lengths(frame[_HIP], frame[_KNEE], frame[_FOOT])) + 1e-4
        )


def test_paw_out_of_reach_of_where_it_was_planted_slips_under_an_upright_leg() -> None:
    far = [JOINT_ORDER.index(f"right_{part}_f") for part in ("hip", "knee", "foot")]
    frames = [_frame((0.0, 2.0), (0.0, 1.0), (0.0, 0.0)) for _ in range(2)]
    # The near paw stays planted while the far one swings back and carries the body
    # forward; the near leg is then asked to stand 2.5 behind a hip that is 2 high.
    for frame, far_foot in zip(frames, (3.0, -2.0), strict=True):
        frame[far[0]] = (0.0, 2.0)
        frame[far[1]] = (far_foot / 2, 1.0)
        frame[far[2]] = (far_foot, 0.0)
    contacts = np.zeros((2, 4), dtype=np.bool_)
    contacts[:, :2] = True
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "travel")
    _run(filter_, frames)
    filtered = _run(filter_, frames)[1]

    # Both planted paws agree to carry the body 2.5; the near leg slides back 0.15 of
    # its length toward its spot and stays upright there.
    np.testing.assert_allclose(filtered[0], (_BASE + 2.5, _BASE), atol=1e-5)
    np.testing.assert_allclose(filtered[_HIP], (2.2, 2.0), atol=1e-5)
    np.testing.assert_allclose(filtered[_KNEE], (2.2, 1.0), atol=1e-5)
    np.testing.assert_allclose(filtered[_FOOT], (2.2, 0.0), atol=1e-5)


def test_travel_advances_by_the_swing_of_the_planted_paw() -> None:
    frames = [_frame((0.0, 2.0), (0.5 - step / 4, 1.0), (1.0 - step / 2, 0.0)) for step in range(5)]
    for frame in frames:
        frame[:, 0] += 7.0
    contacts = np.zeros((len(frames), 4), dtype=np.bool_)
    contacts[:, _NEAR_FRONT] = True
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "travel")
    _run(filter_, frames)
    filtered = _run(filter_, frames)

    # The paw swings back 0.5 a frame under a root the capture left in place.
    for step, row in enumerate(filtered):
        assert float(row[0, 0]) == pytest.approx(7.0 + _BASE + step / 2, abs=1e-5)
        assert float(row[_FOOT, 0]) == pytest.approx(8.0, abs=1e-5)


def test_leg_too_short_for_its_ground_reaches_0999_of_its_length() -> None:
    far = [JOINT_ORDER.index(f"right_{part}_f") for part in ("hip", "knee", "foot")]
    frames = [_frame((0.2, 0.8), (0.3, 0.5), (0.2, 0.2)) for _ in range(2)]
    for frame, far_foot in zip(frames, (0.2, 0.6), strict=True):
        frame[far[0]] = (0.3, 0.8)
        frame[far[1]] = (0.4, 0.7)
        frame[far[2]] = (0.3, far_foot)
    contacts = np.zeros((2, 4), dtype=np.bool_)
    contacts[:, :2] = True
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(filter_, frames)
    filtered = _run(filter_, frames)[1]

    # Folded, the far paw hangs 0.4 above its ground: more than the hip may slide,
    # so the leg straightens toward the ground and stops just short of full stretch.
    upper, lower = _segment_lengths(frames[1][far[0]], frames[1][far[1]], frames[1][far[2]])
    reach = float(np.hypot(*(filtered[far[2]] - filtered[far[0]])))
    assert reach == pytest.approx(0.999 * (upper + lower), abs=1e-4)
    np.testing.assert_allclose(
        _segment_lengths(filtered[far[0]], filtered[far[1]], filtered[far[2]]),
        (upper, lower),
        atol=1e-4,
    )


def test_adjacent_frames_blend_halfway_into_contact() -> None:
    frames = [
        _frame((0.0, 0.6), (0.0, 0.3), (0.0, 0.0)),
        _frame((0.0, 0.6), (0.15, 0.45), (0.2, 0.2)),
        _frame((0.0, 0.6), (0.15, 0.45), (0.2, 0.2)),
        _frame((0.0, 0.6), (0.0, 0.3), (0.0, -0.1)),
    ]
    contacts = np.zeros((len(frames), 4), dtype=np.bool_)
    contacts[1:3, _NEAR_FRONT] = True
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(filter_, frames)
    filtered = _run(filter_, frames)

    np.testing.assert_allclose(filtered[1][_FOOT, 1], 0.2, atol=1e-4)
    np.testing.assert_allclose(filtered[2][_FOOT, 1], 0.2, atol=1e-4)
    np.testing.assert_allclose(filtered[0][_KNEE], [0.1118, 0.35], atol=1e-4)
    np.testing.assert_allclose(filtered[0][_FOOT], [0.0, 0.1], atol=1e-4)
    np.testing.assert_allclose(filtered[3][_KNEE], [0.1391, 0.3938], atol=1e-4)
    np.testing.assert_allclose(filtered[3][_FOOT], [0.0, 0.05], atol=1e-4)


def test_treadmill_keeps_the_paw_under_the_leg_and_the_body_in_place() -> None:
    frames = [
        _frame((0.0, 0.6), (0.15, 0.45), (0.2, 0.2)),
        _frame((0.0, 0.6), (0.15, 0.45), (0.2, 0.2)),
        _frame((0.0, 0.6), (0.15, 0.45), (0.3, 0.2)),
        _frame((0.0, 0.6), (0.15, 0.45), (0.3, 0.2)),
    ]
    contacts = np.zeros((len(frames), 4), dtype=np.bool_)
    contacts[1:3, _NEAR_FRONT] = True
    treadmill = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(treadmill, frames)
    filtered = _run(treadmill, frames)

    np.testing.assert_allclose(filtered[2][_FOOT, 0], 0.3, atol=1e-4)
    np.testing.assert_allclose(filtered[2][0, 0], _BASE, atol=1e-6)


def test_first_pass_records_without_filtering() -> None:
    frames, contacts = _median_ground_scene()
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    first = _run(filter_, frames)
    for frame, row in zip(frames, first, strict=True):
        np.testing.assert_array_equal(row, frame)
    second = _run(filter_, frames)
    assert not np.array_equal(second[0], frames[0])


def test_no_contacts_leaves_positions_unchanged() -> None:
    frames = [_frame((0.0, 0.6), (0.15, 0.45), (0.2, 0.2))]
    contacts = np.zeros((1, 4), dtype=np.bool_)
    filter_ = make_contact_filter(JOINT_ORDER, contacts, "treadmill")
    _run(filter_, frames)
    filtered = _run(filter_, frames)
    np.testing.assert_array_equal(filtered[0], frames[0])
