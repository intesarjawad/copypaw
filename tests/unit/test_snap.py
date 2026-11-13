import math
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.rig.edits import MOVED_FRACTION, load_drawing, rig_drawing
from copypaw.rig.snap import detect_facing, snap
from copypaw.rig.template import fit_template


def _mask(height: int = 200, width: int = 200) -> NDArray[np.uint8]:
    return np.zeros((height, width), dtype=np.uint8)


def _fill(mask: NDArray[np.uint8], top: int, bottom: int, left: int, right: int) -> None:
    mask[top : bottom + 1, left : right + 1] = 255


def _four_leg_mask() -> NDArray[np.uint8]:
    mask = _mask()
    _fill(mask, 40, 110, 60, 140)  # body, above the leg region
    _fill(mask, 135, 189, 20, 41)  # back near
    _fill(mask, 125, 185, 50, 71)  # back far
    _fill(mask, 130, 190, 120, 141)  # front near
    _fill(mask, 120, 180, 150, 171)  # front far
    return mask


def test_four_legs_use_median_bottom_rows_and_55_percent_knees() -> None:
    mask = _four_leg_mask()
    joints = snap(mask, fit_template((20, 40, 171, 190)), "right")

    # Bottom three rows of the left back leg span x 20..41, so their median is 30.5.
    assert (joints["left_hip_b"].x, joints["left_hip_b"].y) == (30.5, 135)
    assert joints["left_foot_b"].x == pytest.approx(30.5)
    assert joints["left_foot_b"].y == pytest.approx(189)
    assert joints["left_knee_b"].x == pytest.approx(30.5)
    assert joints["left_knee_b"].y == pytest.approx(135 + 0.55 * (189 - 135))

    assert (joints["left_hip_f"].x, joints["left_hip_f"].y) == (130.5, 130)
    assert joints["left_foot_f"].y == pytest.approx(190)
    assert joints["left_knee_f"].y == pytest.approx(130 + 0.55 * (190 - 130))

    assert (joints["right_hip_f"].x, joints["right_hip_f"].y) == (160.5, 120)
    assert joints["right_foot_f"].y == pytest.approx(180)
    assert joints["right_knee_f"].y == pytest.approx(120 + 0.55 * (180 - 120))
    assert joints.legs == 4


def test_lower_component_of_each_pair_is_near_and_keeps_its_parents() -> None:
    joints = snap(_four_leg_mask(), fit_template((20, 40, 171, 190)), "right")

    assert joints["left_hip_f"].y == 130
    assert joints["right_hip_f"].y == 120
    assert joints["left_hip_b"].y == 135
    assert joints["right_hip_b"].y == 125
    assert joints["left_hip_f"].parent == "root"
    assert joints["left_knee_f"].parent == "left_hip_f"
    assert joints["left_foot_f"].parent == "left_knee_f"
    assert joints["right_hip_f"].parent == "root"


def test_left_facing_four_legs_take_the_leftmost_pair_as_front() -> None:
    mask = _four_leg_mask()
    joints = snap(mask, fit_template((20, 40, 171, 190)), "left")

    # The leftmost pair is front when the creature faces left.
    assert joints["left_hip_f"].y == 135
    assert joints["right_hip_f"].y == 125
    assert joints["left_hip_b"].y == 130
    assert joints["right_hip_b"].y == 120


def test_two_legs_share_positions_and_record_two_legs() -> None:
    mask = _mask()
    _fill(mask, 40, 110, 60, 140)
    _fill(mask, 130, 189, 30, 51)
    _fill(mask, 120, 190, 150, 171)

    joints = snap(mask, fit_template((30, 40, 171, 190)), "right")

    assert joints.legs == 2
    for end in ("f", "b"):
        for limb in ("hip", "knee", "foot"):
            left = joints[f"left_{limb}_{end}"]
            right = joints[f"right_{limb}_{end}"]
            assert (left.x, left.y) == (right.x, right.y)
    # Facing right, the rightmost component is the front leg.
    assert joints["left_hip_f"].x == pytest.approx(160.5)
    assert joints["left_hip_b"].x == pytest.approx(40.5)


def test_three_legs_keep_the_fitted_template_legs() -> None:
    mask = _mask()
    _fill(mask, 40, 110, 60, 140)
    _fill(mask, 140, 189, 20, 41)
    _fill(mask, 130, 189, 150, 171)
    _fill(mask, 120, 189, 175, 196)
    fitted = fit_template((20, 40, 196, 189))

    joints = snap(mask, fitted, "right")

    assert joints.legs == 4
    for name in fitted:
        if name.startswith(("left_", "right_")):
            assert (joints[name].x, joints[name].y) == (fitted[name].x, fitted[name].y)


def _headed_mask() -> NDArray[np.uint8]:
    mask = _mask()
    _fill(mask, 60, 110, 30, 150)  # body, above the leg region
    _fill(mask, 20, 80, 150, 170)  # neck, rising at the right end
    _fill(mask, 20, 30, 170, 190)  # head, jutting forward
    _fill(mask, 110, 189, 40, 61)
    _fill(mask, 110, 189, 120, 141)
    return mask


def _inside(mask: NDArray[np.uint8], x: float, y: float) -> bool:
    return bool(mask[round(y), round(x)])


def test_body_joints_stand_inside_the_outline_over_their_legs() -> None:
    mask = _headed_mask()

    joints = snap(mask, fit_template((30, 20, 190, 189)), "right")

    for name in ("root", "hip", "torso", "neck"):
        assert _inside(mask, joints[name].x, joints[name].y), name
    assert (joints["root"].x, joints["root"].y) == (joints["hip"].x, joints["hip"].y)
    # The hip is over the back leg and the torso over the front leg, inside the body.
    assert joints["hip"].x == pytest.approx(50.5, abs=3)
    assert joints["torso"].x == pytest.approx(130.5, abs=3)
    assert 60 <= joints["hip"].y <= 110
    assert 60 <= joints["torso"].y <= 110


def test_neck_follows_a_bent_neck_instead_of_cutting_the_corner() -> None:
    mask = _headed_mask()
    fitted = fit_template((30, 20, 190, 189))

    joints = snap(mask, fitted, "right")

    # Seventy percent of the straight line to the nose at (190, 20) leaves the outline.
    torso = joints["torso"]
    straight = (torso.x + 0.7 * (190 - torso.x), torso.y + 0.7 * (20 - torso.y))
    assert not _inside(mask, *straight)
    assert _inside(mask, joints["neck"].x, joints["neck"].y)
    assert joints["neck"].x >= 150
    assert joints["neck"].y < 60
    assert joints["neck"].parent == "torso"


def test_left_facing_neck_reaches_toward_the_leftmost_point() -> None:
    mask = np.ascontiguousarray(_headed_mask()[:, ::-1])

    joints = snap(mask, fit_template((9, 20, 169, 189), "left"), "left")

    assert _inside(mask, joints["neck"].x, joints["neck"].y)
    assert joints["neck"].x <= 49
    assert joints["neck"].y < 60
    assert joints["torso"].x < joints["hip"].x


def test_hips_sit_where_the_legs_join_the_body() -> None:
    mask = _headed_mask()

    joints = snap(mask, fit_template((30, 20, 190, 189)), "right")

    # The legs leave the body at row 110, above row 120 where the leg search starts.
    for name in ("left_hip_f", "left_hip_b"):
        assert joints[name].y == pytest.approx(110, abs=2)


def test_facing_follows_the_side_of_the_head() -> None:
    mask = _headed_mask()

    assert detect_facing(mask) == "right"
    assert detect_facing(np.ascontiguousarray(mask[:, ::-1])) == "left"


def test_a_drawing_without_a_head_faces_right() -> None:
    assert detect_facing(_four_leg_mask()) == "right"
    with pytest.raises(ValueError, match="drawing"):
        detect_facing(_mask())


def test_small_components_are_ignored() -> None:
    mask = _mask()
    _fill(mask, 40, 110, 60, 140)
    _fill(mask, 150, 152, 10, 12)  # nine pixels, well under 0.5% of the mask
    _fill(mask, 130, 189, 150, 171)

    joints = snap(mask, fit_template((10, 40, 171, 190)), "right")

    assert joints.legs == 4


def test_empty_mask_and_unknown_facing_are_rejected() -> None:
    with pytest.raises(ValueError, match="drawing"):
        snap(_mask(), fit_template((0, 0, 10, 10)), "right")
    with pytest.raises(ValueError, match="facing"):
        snap(_mask(), fit_template((0, 0, 10, 10)), cast(Literal["left", "right"], "up"))


def test_opening_removes_thin_bridges_between_legs() -> None:
    mask = _four_leg_mask()
    cv2.line(mask, (41, 170), (50, 170), 255, 1)

    joints = snap(mask, fit_template((20, 40, 171, 190)), "right")

    assert joints.legs == 4


# A right-facing creature on a 600 x 400 page; the body ends above row 240, where legs begin.
_PAGE = (400, 600)
_GROUND = 380
_LEG_WIDTH = 30


def _creature(legs_left: tuple[int, ...], *shapes: list[tuple[int, int]]) -> NDArray[np.uint8]:
    mask = np.zeros(_PAGE, dtype=np.uint8)
    cv2.ellipse(mask, (300, 170), (180, 60), 0, 0, 360, 255, -1)
    cv2.fillPoly(mask, [np.array([(440, 150), (520, 30), (560, 40), (480, 170)])], 255)  # head
    for left in legs_left:
        _fill(mask, 200, _GROUND, left, left + _LEG_WIDTH - 1)
    for shape in shapes:
        cv2.fillPoly(mask, [np.array(shape)], 255)
    return mask


_FOUR_LEGS = (155, 235, 345, 425)
_TAIL_BEHIND = [(118, 160), (138, 175), (82, _GROUND), (62, _GROUND)]


def _snap_on(mask: NDArray[np.uint8], facing: Literal["left", "right"]) -> dict[str, float]:
    rows, columns = np.nonzero(mask)
    box = (float(columns.min()), float(rows.min()), float(columns.max()), float(rows.max()))
    fitted = fit_template(box)
    joints = snap(mask, fitted, facing)
    assert joints.legs == 4
    return {
        name: joint.x
        for name, joint in joints.items()
        if name.endswith(("_foot_f", "_foot_b"))
        and (joint.x, joint.y) != (fitted[name].x, fitted[name].y)
    }


def _leg_centres(lefts: tuple[int, ...]) -> list[float]:
    return [left + (_LEG_WIDTH - 1) / 2 for left in lefts]


def test_a_tail_on_the_ground_behind_four_legs_is_not_a_leg() -> None:
    feet = _snap_on(_creature(_FOUR_LEGS, _TAIL_BEHIND), "right")

    assert sorted(feet.values()) == pytest.approx(_leg_centres(_FOUR_LEGS))


def test_a_mirrored_tail_is_found_at_the_other_end() -> None:
    mirrored = np.ascontiguousarray(_creature(_FOUR_LEGS, _TAIL_BEHIND)[:, ::-1])

    feet = _snap_on(mirrored, "left")

    expected = sorted(_PAGE[1] - 1 - centre for centre in _leg_centres(_FOUR_LEGS))
    assert sorted(feet.values()) == pytest.approx(expected)


def test_a_thick_tail_that_tapers_straight_down_is_not_a_leg() -> None:
    thick = [(95, 165), (150, 165), (128, _GROUND), (117, _GROUND)]

    feet = _snap_on(_creature(_FOUR_LEGS, thick), "right")

    assert sorted(feet.values()) == pytest.approx(_leg_centres(_FOUR_LEGS))


def test_a_tail_inside_the_hind_legs_that_leans_back_is_not_a_leg() -> None:
    legs = (125, 285, 385, 445)
    tail = [(245, 220), (262, 220), (190, _GROUND), (173, _GROUND)]

    feet = _snap_on(_creature(legs, tail), "right")

    assert sorted(feet.values()) == pytest.approx(_leg_centres(legs))


def test_five_upright_legs_stay_ambiguous_and_keep_the_template() -> None:
    mask = _creature((140, 210, 280, 350, 420))

    assert _snap_on(mask, "right") == {}


def test_the_sample_drawing_has_every_joint_inside_its_outline() -> None:
    drawing = Path(__file__).resolve().parents[2] / "copypaw/selftest/dragon.png"

    rigged = rig_drawing(load_drawing(drawing))

    for name, joint in rigged.snapped.items():
        assert _inside(rigged.mask, joint.x, joint.y), name


def test_a_mirrored_drawing_rigs_as_the_mirror_image() -> None:
    drawing = load_drawing(Path(__file__).resolve().parents[2] / "copypaw/selftest/dragon.png")
    width = drawing.shape[1]

    right = rig_drawing(drawing).snapped
    left = rig_drawing(np.ascontiguousarray(drawing[:, ::-1])).snapped

    assert left["neck"].x < left["hip"].x
    # Thinning is not mirror-symmetric, so medial joints may land a few pixels apart.
    tolerance = 0.01 * width
    for name, joint in right.items():
        assert left[name].x == pytest.approx(width - 1 - joint.x, abs=tolerance), name
        assert left[name].y == pytest.approx(joint.y, abs=tolerance), name


def test_the_sample_drawing_rigs_its_four_legs_and_leaves_its_tail() -> None:
    drawing = Path(__file__).resolve().parents[2] / "copypaw/selftest/dragon.png"

    rigged = rig_drawing(load_drawing(drawing))

    assert rigged.legs_snapped
    feet = sorted((joint.x, joint.y) for name, joint in rigged.snapped.items() if "_foot_" in name)
    # Bottom centres of the four drawn legs, measured on the sample's mask. The
    # tolerance is the share of the box height beyond which a dot counts as moved.
    expected = [(185, 619), (260, 639), (365, 629), (435, 609)]
    tolerance = MOVED_FRACTION * rigged.box_height
    for (x, y), (expected_x, expected_y) in zip(feet, expected, strict=True):
        assert math.hypot(x - expected_x, y - expected_y) < tolerance
