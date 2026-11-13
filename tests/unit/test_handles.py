"""Tests for the extra mesh handles."""

import math

import numpy as np
import pytest

from copypaw.animate.handles import tail_handles, with_bone_handles

_NAMES = ("torso", "neck", "left_hip_f", "left_knee_f", "left_foot_f")
_POSITIONS = np.array([(0, 0), (7, 7), (0, -2), (2, -6), (2, -10)], dtype=np.float32)


def test_each_leg_bone_carries_handles_along_and_either_side_of_it() -> None:
    handles = with_bone_handles(_NAMES, _POSITIONS)

    np.testing.assert_array_equal(handles[: len(_NAMES)], _POSITIONS)
    # The lower bone runs from (2, -6) to (2, -10): straight down, four long.
    np.testing.assert_allclose(
        handles[len(_NAMES) + 7 : len(_NAMES) + 14],
        [(2, -7), (2.4, -7), (1.6, -7), (2, -8), (2, -9), (2.4, -9), (1.6, -9)],
        atol=1e-6,
    )
    np.testing.assert_allclose(handles[-1], (8.5, 8.5), atol=1e-6)


def test_side_handles_turn_with_their_bone() -> None:
    upright = with_bone_handles(_NAMES[2:4], np.array([(0, 0), (0, -10)], dtype=np.float32))
    level = with_bone_handles(_NAMES[2:4], np.array([(0, 0), (10, 0)], dtype=np.float32))

    turned = np.stack((-upright[:, 1], upright[:, 0]), axis=1)
    np.testing.assert_allclose(level, turned, atol=1e-6)


def test_bones_missing_a_joint_get_no_handle() -> None:
    handles = with_bone_handles(_NAMES[2:4], _POSITIONS[2:4])

    assert len(handles) == 2 + 7


def test_joint_count_must_match_the_names() -> None:
    with pytest.raises(ValueError, match="2 joints but 5 names"):
        with_bone_handles(_NAMES, _POSITIONS[:2])


def test_a_tail_above_the_ground_moves_with_the_body() -> None:
    tails = np.array([[(0, 5), (-4, 2)]], dtype=np.float32)

    handles = tail_handles(tails, np.array([1.0, -1.0]), ground=0.0)

    np.testing.assert_allclose(handles, [(1, 4), (-3, 1)])


@pytest.mark.parametrize(("tip_x", "side"), [(-4.0, -1), (4.0, 1)])
def test_a_lifted_tail_swings_away_from_the_legs_keeping_its_length(
    tip_x: float, side: int
) -> None:
    tails = np.array([[(0, 5), (tip_x, 1)]], dtype=np.float32)

    base, tip = tail_handles(tails, np.zeros(2), ground=None, lift=np.array([math.pi / 4]))

    assert tip[0] * side > abs(tip_x)
    assert tip[1] > 1
    assert np.linalg.norm(tip - base) == pytest.approx(math.hypot(tip_x, 4), abs=1e-5)


@pytest.mark.parametrize(("tip_x", "side"), [(-4.0, -1), (4.0, 1)])
def test_a_tip_that_would_sink_drags_on_the_ground_and_keeps_the_tail_length(
    tip_x: float, side: int
) -> None:
    tails = np.array([[(0, 5), (tip_x, 2)]], dtype=np.float32)

    base, tip = tail_handles(tails, np.array([0.0, -3.0]), ground=0.0)

    np.testing.assert_allclose(base, (0, 2))
    assert float(tip[1]) == pytest.approx(0.0)
    assert np.sign(tip[0]) == side
    assert np.linalg.norm(tip - base) == pytest.approx(5.0, abs=1e-5)


def test_without_a_ground_the_tail_only_follows_the_body() -> None:
    tails = np.array([[(0, 5), (-4, 2)]], dtype=np.float32)

    handles = tail_handles(tails, np.array([0.0, -9.0]), ground=None)

    np.testing.assert_allclose(handles, [(0, -4), (-4, -7)])


def test_tail_turns_about_its_base_and_keeps_its_length() -> None:
    tails = np.array([[(0, 5), (-3, 1)]], dtype=np.float32)

    base, tip = tail_handles(tails, np.zeros(2), None, turn=math.pi / 2)

    np.testing.assert_allclose(base, (0, 5))
    np.testing.assert_allclose(tip, (4, 2), atol=1e-6)


@pytest.mark.parametrize("turn", [-1.0, -0.3, 0.0, 0.3, 1.0])
def test_tail_tip_never_passes_the_ground_or_the_rearmost_leg(turn: float) -> None:
    tails = np.array([[(0, 5), (-3, 1)]], dtype=np.float32)

    base, tip = tail_handles(tails, np.zeros(2), 0.5, turn=turn, behind=np.array([-2.0]))

    assert float(tip[1]) >= 0.5 - 1e-6
    assert float(tip[0]) <= -2.0 + 1e-6
    assert np.linalg.norm(tip - base) == pytest.approx(5.0, abs=1e-5)
