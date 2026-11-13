import pytest

from copypaw.rig.template import fit_template

UPSTREAM = {
    "root": (238, 264),
    "hip": (238, 264),
    "torso": (453, 181),
    "neck": (455, 121),
    "right_hip_b": (139, 266),
    "right_knee_b": (138, 290),
    "right_foot_b": (132, 324),
    "left_hip_b": (82, 249),
    "left_knee_b": (65, 279),
    "left_foot_b": (47, 308),
    "right_hip_f": (442, 245),
    "right_knee_f": (455, 284),
    "right_foot_f": (458, 320),
    "left_hip_f": (364, 266),
    "left_knee_f": (364, 305),
    "left_foot_f": (360, 336),
}
PARENTS = {
    "root": None,
    "hip": "root",
    "torso": "hip",
    "neck": "torso",
    "right_hip_b": "root",
    "right_knee_b": "right_hip_b",
    "right_foot_b": "right_knee_b",
    "left_hip_b": "root",
    "left_knee_b": "left_hip_b",
    "left_foot_b": "left_knee_b",
    "right_hip_f": "root",
    "right_knee_f": "right_hip_f",
    "right_foot_f": "right_knee_f",
    "left_hip_f": "root",
    "left_knee_f": "left_hip_f",
    "left_foot_f": "left_knee_f",
}


def test_template_keeps_upstream_positions_in_its_own_box() -> None:
    joints = fit_template((0, 0, 564, 352))

    assert len(joints) == 16
    assert joints.legs == 4
    for name, (x, y) in UPSTREAM.items():
        assert joints[name].x == pytest.approx(x)
        assert joints[name].y == pytest.approx(y)


def test_template_parents_match_upstream() -> None:
    joints = fit_template((0, 0, 564, 352))

    assert {name: joint.parent for name, joint in joints.items()} == PARENTS


def test_template_scales_from_its_box_into_the_mask_box() -> None:
    joints = fit_template((100, 50, 300, 250))

    # 238 / 564 * 200 + 100 and 264 / 352 * 200 + 50.
    assert joints["hip"].x == pytest.approx(184.39716312056737)
    assert joints["hip"].y == pytest.approx(200.0)
    # 455 / 564 * 200 + 100 and 121 / 352 * 200 + 50.
    assert joints["neck"].x == pytest.approx(261.34751773049647)
    assert joints["neck"].y == pytest.approx(118.75)


def test_every_template_point_stays_inside_the_box() -> None:
    left, top, right, bottom = 100, 50, 300, 250

    joints = fit_template((left, top, right, bottom))

    assert all(left <= joint.x <= right and top <= joint.y <= bottom for joint in joints.values())


@pytest.mark.parametrize("box", [(0, 0, 0, 10), (10, 0, 5, 10), (0, 10, 10, 5)])
def test_degenerate_boxes_are_rejected(box: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="positive area"):
        fit_template(box)


def test_non_finite_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_template((0, 0, float("inf"), 10))


def test_left_facing_template_is_the_mirror_image() -> None:
    right = fit_template((100, 50, 300, 250))
    left = fit_template((100, 50, 300, 250), "left")

    for name, joint in right.items():
        assert left[name].x == pytest.approx(400 - joint.x)
        assert left[name].y == joint.y
        assert left[name].parent == joint.parent
