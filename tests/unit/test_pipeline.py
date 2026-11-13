import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.capture.detect import Track
from copypaw.capture.pipeline import capture_motion
from copypaw.capture.pose import Keypoints
from copypaw.capture.video import Frames
from copypaw.errors import CopypawError
from copypaw.motion.motion_file import MotionFile

FRAME_COUNT = 10
FPS = 30.0
WIDTH = 160
# Right-facing standing dog, nose at x=120 and tail root at x=0, body length 100 px.
POSE = np.array(
    [
        (118, 35), (122, 35), (120, 40), (100, 50), (0, 50),
        (95, 60), (95, 80), (95, 100), (105, 60), (105, 80), (105, 100),
        (5, 60), (5, 80), (5, 100), (15, 60), (15, 80), (15, 100),
    ],
    dtype=np.float32,
)  # fmt: skip


def _frames() -> Frames:
    return Frames(
        rgb=np.zeros((FRAME_COUNT, 120, WIDTH, 3), dtype=np.uint8),
        t_s=np.arange(FRAME_COUNT, dtype=np.float64) / FPS,
        source_fps=FPS,
        fps=FPS,
        trimmed=False,
    )


def _scores() -> NDArray[np.float32]:
    scores = np.full((FRAME_COUNT, 17), 0.9, dtype=np.float32)
    scores[:, 8:11] = 0.8  # Far front limb is less confident than the near one.
    scores[:, 14:17] = 0.8
    scores[:, 0] = 0.1  # The left eye is never confident enough to keep.
    return scores


def _pose(frames: Frames, track: Track) -> Keypoints:
    assert track.present.all()
    return Keypoints(np.tile(POSE, (len(frames.rgb), 1, 1)), _scores())


def test_capture_writes_motion_skeleton_and_quality(tmp_path: Path) -> None:
    def detector(image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        assert image.shape == (120, WIDTH, 3)
        boxes = np.array([[0, 20, 130, 110]], dtype=np.float32)
        return boxes, np.array([16], dtype=np.int64)

    report = capture_motion(_frames(), "fake.mp4", tmp_path, detector=detector, pose=_pose)

    motion = MotionFile.load(tmp_path / "motion.npz")
    assert motion.source == "fake.mp4"
    assert motion.fps == FPS
    assert motion.facing == "right"
    expected = np.tile(POSE, (FRAME_COUNT, 1, 1))
    expected[:, 0] = np.nan
    np.testing.assert_allclose(motion.kp, expected, equal_nan=True)
    np.testing.assert_array_equal(motion.scores, _scores())
    assert motion.contacts.shape == (FRAME_COUNT, 4)
    assert motion.contacts.all()
    np.testing.assert_allclose(motion.travel_root_x, np.zeros(FRAME_COUNT), atol=1e-6)

    bvh = (tmp_path / "motion.bvh").read_text(encoding="utf-8")
    assert bvh.startswith("HIERARCHY")
    assert f"Frames: {FRAME_COUNT}" in bvh

    written = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert written["per_joint_confident_fraction"] == [0.0] + [1.0] * 16
    assert written["frames_interpolated"] == FRAME_COUNT
    assert written["swaps_corrected"] == 0
    assert written["warnings"] == []
    assert written["errors"] == []
    assert report.frames_interpolated == FRAME_COUNT


@pytest.mark.parametrize("joint", [3, 4, 7, 10, 13, 16])
def test_missing_anatomy_is_a_filming_error_before_motion_is_written(
    tmp_path: Path, joint: int
) -> None:
    def detector(image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        del image
        return np.array([[0, 20, 130, 110]], np.float32), np.array([16], np.int64)

    def pose(frames: Frames, track: Track) -> Keypoints:
        raw = _pose(frames, track)
        raw.scores[:, joint] = 0
        return raw

    with pytest.raises(CopypawError) as raised:
        capture_motion(_frames(), "missing.mp4", tmp_path, detector=detector, pose=pose)

    assert raised.value.code == "capture_anatomy"
    assert "Missing required landmark" in raised.value.what
    assert "whole pet" in raised.value.next
    assert not (tmp_path / "motion.npz").exists()
    assert json.loads((tmp_path / "quality.json").read_text())["errors"]
