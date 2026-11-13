"""A travelling drawing keeps its planted paws and its whole body in frame."""

from pathlib import Path

import numpy as np
import pytest

from copypaw.animate.render import render_character, resample_motion, travel_canvas
from copypaw.motion.motion_file import MotionFile

pytestmark = pytest.mark.render

_ROOT = Path(__file__).resolve().parents[2]
_CHARACTER = _ROOT / "tests/fixtures/upstream/characters/char6"
_MOTION = _ROOT / "tests/golden/dog_walk.motion.npz"
_PAWS = ("left_foot_f", "right_foot_f", "left_foot_b", "right_foot_b")
_MAX_DRIFT_RATIO = 0.02
_FRAMES = 30
_STANCE_FRAMES = 5
_STRIDE_FRAMES = 16


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    frames = np.flatnonzero(mask)
    if frames.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) > 1) + 1
    return list(np.split(frames, breaks))


def _gait_motion() -> MotionFile:
    """A diagonal-pair gait that walks a quarter of a body length per clip."""
    base = MotionFile.load(_MOTION)
    contacts = np.zeros((_FRAMES, 4), dtype=np.bool_)
    for frame in range(_FRAMES):
        near = frame % _STRIDE_FRAMES < _STANCE_FRAMES
        far = (frame + _STRIDE_FRAMES // 2) % _STRIDE_FRAMES < _STANCE_FRAMES
        contacts[frame] = [near, far, near, far]
    return MotionFile(
        kp=base.kp[:_FRAMES].copy(),
        scores=base.scores[:_FRAMES].copy(),
        fps=15.0,
        facing=base.facing,
        contacts=contacts,
        ground_line_px=base.ground_line_px,
        travel_root_x=np.linspace(0, 0.25, _FRAMES, dtype=np.float32),
        source=base.source,
    )


def test_travel_render_stays_in_frame_with_planted_paws(tmp_path: Path) -> None:
    motion = _gait_motion()
    resampled = resample_motion(motion)
    canvas = travel_canvas(_CHARACTER, resampled)

    result = render_character(_CHARACTER, motion, "travel", tmp_path / "out")

    assert len(result.frames_rgba) == canvas.frames
    height, width = result.frames_rgba.shape[1:3]
    assert width / height == pytest.approx(canvas.width / canvas.height, rel=0.02)

    ink = np.any(result.frames_rgba[..., :3] < 250, axis=-1)
    for index, frame in enumerate(ink):
        rows, columns = np.nonzero(frame)
        assert rows.size > 0, f"frame {index} is empty"
        assert rows.min() > 0, f"frame {index} touches the top edge"
        assert columns.min() > 0, f"frame {index} touches the left edge"
        assert rows.max() < height - 1, f"frame {index} touches the bottom edge"
        assert columns.max() < width - 1, f"frame {index} touches the right edge"

    assert result.joint_names is not None
    index = {name: position for position, name in enumerate(result.joint_names)}
    joints = result.joints_per_frame
    hips = joints[:, [index["left_hip_f"], index["left_hip_b"]]]
    body_length = float(np.median(np.linalg.norm(hips[:, 0] - hips[:, 1], axis=1)))
    contacts = resampled.contacts[: canvas.frames]
    for paw, name in enumerate(_PAWS):
        for contact in _runs(contacts[:, paw]):
            if len(contact) < 2:
                continue
            x = joints[contact, index[name], 0]
            assert np.max(np.abs(x - x[0])) < _MAX_DRIFT_RATIO * body_length
