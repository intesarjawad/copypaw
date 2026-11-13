import math
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.animate.bake import bake_bvh, pose_for_drawing
from copypaw.motion.motion_file import MotionFile

bvh_module = import_module("copypaw.vendor.animated_drawings.model.bvh")

# The video spans 200 px between the near shoulder and hip; the drawing's body
# spans 100 px between its left front and back hips.
JOINTS = {
    "left_hip_f": (0.0, 0.0),
    "left_knee_f": (30.0, 40.0),
    "left_foot_f": (30.0, 120.0),
    "left_hip_b": (-100.0, 0.0),
    "left_knee_b": (-100.0, 60.0),
    "left_foot_b": (-100.0, 140.0),
    "right_hip_f": (10.0, 0.0),
    "right_knee_f": (10.0, 70.0),
    "right_foot_f": (10.0, 190.0),
    "right_hip_b": (-90.0, 0.0),
    "right_knee_b": (-90.0, 20.0),
    "right_foot_b": (-90.0, 90.0),
}
# Hand-measured drawing lengths for each BVH leg bone.
DRAWING_LENGTHS = {
    "Near_Elbow": 50.0,
    "Near_F_Paw": 80.0,
    "Near_Knee": 60.0,
    "Near_B_Paw": 80.0,
    "Far_Elbow": 70.0,
    "Far_F_Paw": 120.0,
    "Far_Knee": 20.0,
    "Far_B_Paw": 70.0,
}
FRAME_COUNT = 3


def _write_motion(path: Path) -> None:
    kp = np.zeros((FRAME_COUNT, 17, 2), dtype=np.float32)
    kp[:, 0] = (110, -110)
    kp[:, 1] = (110, -90)
    kp[:, 2] = (120, -100)
    kp[:, 3] = (100, -100)
    kp[:, 4] = (100, 0)
    kp[:, 5] = (0, -100)
    kp[:, 6] = (0, 0)
    kp[:, 7] = (0, 100)
    kp[:, 8] = (-50, -100)
    kp[:, 9] = (-50, -20)
    kp[:, 10] = (-50, 40)
    kp[:, 11] = (200, -100)
    kp[:, 12] = (200, 0)
    kp[:, 13] = (200, 120)
    kp[:, 14] = (250, -100)
    kp[:, 15] = (250, -30)
    kp[:, 16] = (250, 40)
    MotionFile(
        kp=kp,
        scores=np.full((FRAME_COUNT, 17), 0.9, dtype=np.float32),
        fps=30.0,
        facing="right",
        contacts=np.ones((FRAME_COUNT, 4), dtype=np.bool_),
        ground_line_px=np.array([0.0, 120.0]),
        travel_root_x=np.linspace(0, 1, FRAME_COUNT, dtype=np.float32),
        source="clip.mp4",
    ).save(path)


def test_baked_leg_length_ratios_match_the_drawing(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    _write_motion(motion)

    baked = bake_bvh(motion, JOINTS, "treadmill")

    assert baked == tmp_path / "baked.bvh"
    skeleton = bvh_module.BVH.from_file(str(baked))
    baked_lengths: dict[str, float] = {}
    for bone, drawing_length in DRAWING_LENGTHS.items():
        joint = skeleton.root_joint.get_transform_by_name(bone)
        assert joint is not None
        local = np.asarray(joint.get_local_position(), dtype=np.float64).reshape(3)
        baked_lengths[bone] = float(np.linalg.norm(local[:2]))
        # The video body is twice the drawing body, so each leg doubles.
        assert baked_lengths[bone] == pytest.approx(drawing_length * 2, abs=1e-6)
    assert baked_lengths["Near_Elbow"] / baked_lengths["Near_F_Paw"] == pytest.approx(
        50 / 80, abs=1e-6
    )
    assert baked_lengths["Far_Knee"] / baked_lengths["Far_B_Paw"] == pytest.approx(
        20 / 70, abs=1e-6
    )


_NEAR_FRONT = (5, 6, 7)
_DRAWN_NEAR_FRONT = np.array([JOINTS["left_knee_f"], JOINTS["left_foot_f"]], dtype=np.float64)
# The pet's planted near-front leg spans 200 px; the drawn one spans its hip-to-foot distance.
_NEAR_FRONT_SCALE = 200 / math.hypot(30.0, 120.0)


def _near_front(posed: NDArray[np.float32]) -> NDArray[np.float32]:
    hip, knee, paw = _NEAR_FRONT
    return np.stack((posed[:, knee] - posed[:, hip], posed[:, paw] - posed[:, hip]), axis=1)


def test_a_still_pet_leaves_each_leg_as_it_was_drawn(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_motion(path)

    posed = pose_for_drawing(MotionFile.load(path), JOINTS)

    for leg in _near_front(posed):
        np.testing.assert_allclose(leg, _DRAWN_NEAR_FRONT * _NEAR_FRONT_SCALE, atol=1e-3)


def test_a_planted_leg_swings_from_the_hip_without_changing_shape(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_motion(path)
    motion = MotionFile.load(path)
    # The pet's leg shortens as it swings; the drawn leg must not.
    motion.kp[0, 7] = (-60, 60)
    motion.kp[2, 7] = (60, 60)

    legs = _near_front(pose_for_drawing(motion, JOINTS))

    assert float(legs[0, 1, 0]) < float(legs[1, 1, 0]) < float(legs[2, 1, 0])
    for knee, foot in legs:
        assert np.linalg.norm(knee) == pytest.approx(50 * _NEAR_FRONT_SCALE, abs=1e-3)
        assert np.linalg.norm(foot - knee) == pytest.approx(80 * _NEAR_FRONT_SCALE, abs=1e-3)
        assert np.linalg.norm(foot) == pytest.approx(200, abs=1e-3)


@pytest.mark.parametrize("paw_height", [80.0, 40.0, -90.0])
def test_a_lifted_paw_bends_the_knee_no_further_than_a_right_angle(
    tmp_path: Path, paw_height: float
) -> None:
    path = tmp_path / "motion.npz"
    _write_motion(path)
    motion = MotionFile.load(path)
    motion.kp[1, 7] = (0, paw_height)
    motion.contacts[1, 0] = False

    planted, lifted, _ = _near_front(pose_for_drawing(motion, JOINTS))

    assert float(lifted[1, 1]) < float(planted[1, 1])
    upper, lower = 50 * _NEAR_FRONT_SCALE, 80 * _NEAR_FRONT_SCALE
    assert np.linalg.norm(lifted[0]) == pytest.approx(upper, abs=1e-3)
    assert np.linalg.norm(lifted[1] - lifted[0]) == pytest.approx(lower, abs=1e-3)
    assert math.hypot(*lifted[1].tolist()) >= math.hypot(upper, lower) - 1e-3


def test_still_legs_keep_their_drawn_pose_whatever_the_pet_does(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_motion(path)
    motion = MotionFile.load(path)
    motion.kp[0, 7] = (-60, 60)
    motion.kp[1, 7] = (0, -90)
    motion.contacts[1, 0] = False

    for leg in _near_front(pose_for_drawing(motion, JOINTS, still_legs=True)):
        np.testing.assert_allclose(leg, _DRAWN_NEAR_FRONT * _NEAR_FRONT_SCALE, atol=1e-3)
