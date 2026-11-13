"""A treadmill render pins every reachable contact paw to the ground."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from copypaw.animate.render import render_character, resample_motion
from copypaw.motion.motion_file import MotionFile

pytestmark = pytest.mark.render

_ROOT = Path(__file__).resolve().parents[2]
_CHARACTER = _ROOT / "tests/fixtures/upstream/characters/char6"
_MOTION = _ROOT / "tests/golden/dog_walk.motion.npz"
_LEGS = (
    ("left_hip_f", "left_knee_f", "left_foot_f"),
    ("right_hip_f", "right_knee_f", "right_foot_f"),
    ("left_hip_b", "left_knee_b", "left_foot_b"),
    ("right_hip_b", "right_knee_b", "right_foot_b"),
)
_GROUND_TOLERANCE = 0.015


def _character_height() -> float:
    """The drawing's height in engine units, where the longer image side is 1."""
    config = yaml.safe_load((_CHARACTER / "char_cfg.yaml").read_text(encoding="utf-8"))
    return config["height"] / max(config["height"], config["width"])


def test_treadmill_contact_paws_stand_on_the_ground(tmp_path: Path) -> None:
    motion = MotionFile.load(_MOTION)
    contacts = resample_motion(motion).contacts

    result = render_character(_CHARACTER, motion, "treadmill", tmp_path / "out")

    assert result.joint_names is not None
    assert result.unfiltered_joints_per_frame is not None
    index = {name: position for position, name in enumerate(result.joint_names)}
    tolerance = _GROUND_TOLERANCE * _character_height()

    before: list[float] = []
    for paw, (hip, knee, foot) in enumerate(_LEGS):
        planted = np.flatnonzero(contacts[:, paw])
        assert planted.size > 1
        joints = result.joints_per_frame[planted]
        # Each paw has its own ground line: a far leg is often drawn shorter than a near one.
        heights = joints[:, index[foot], 1]
        assert float(np.ptp(heights)) < tolerance
        before.append(float(np.ptp(result.unfiltered_joints_per_frame[planted, index[foot], 1])))
        # Standing the paw never stretches or shrinks a bone.
        posed = result.unfiltered_joints_per_frame[planted]
        for near, far in ((hip, knee), (knee, foot)):
            np.testing.assert_allclose(
                np.linalg.norm(joints[:, index[near]] - joints[:, index[far]], axis=1),
                np.linalg.norm(posed[:, index[near]] - posed[:, index[far]], axis=1),
                atol=tolerance,
            )
    assert max(before) > tolerance, "the motion already planted every paw"
