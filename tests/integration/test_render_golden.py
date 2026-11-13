"""Rendered frames must match the committed golden drawing frame.

The golden was written on macOS arm64 with scripts/make_goldens.py --render.
OpenGL rasterisation can differ elsewhere, so a failure on another platform is
a measurement to record, not a cue to loosen the tolerance.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from copypaw.animate.render import render_character
from copypaw.capture.video import OUTPUT_FPS
from copypaw.motion.motion_file import MotionFile

pytestmark = pytest.mark.render

_ROOT = Path(__file__).resolve().parents[2]
_CHARACTER = _ROOT / "tests/fixtures/upstream/characters/char6"
_MOTION = _ROOT / "tests/golden/dog_walk.motion.npz"
_GOLDEN = _ROOT / "tests/golden/quadruped_trot_f20.png"
_GOLDEN_FRAME = 20
_MAX_ALPHA_DIFFERENCE = 0.02
_ANALYSIS_FRAMES = 60
_ANALYSIS_FPS = 12.0
_CHARACTER_JOINTS = 16
_EXPECTED_FRAMES = math.floor((_ANALYSIS_FRAMES - 1) / _ANALYSIS_FPS * OUTPUT_FPS) + 1


def test_rendered_frame_matches_golden(tmp_path: Path) -> None:
    if not _GOLDEN.is_file():
        pytest.skip("Render golden is absent; run scripts/make_goldens.py --render")

    motion = MotionFile.load(_MOTION)
    result = render_character(_CHARACTER, motion, "treadmill", tmp_path / "out")

    assert result.fps == OUTPUT_FPS
    assert len(result.frames_rgba) == _EXPECTED_FRAMES
    assert result.joints_per_frame.shape == (_EXPECTED_FRAMES, _CHARACTER_JOINTS, 2)

    with Image.open(_GOLDEN) as image:
        golden = np.asarray(image.convert("RGBA"))
    actual = result.frames_rgba[_GOLDEN_FRAME]
    # A windowed GL context on a high-DPI display renders at twice the requested
    # size, while OSMesa renders exactly what was asked, so compare at the render's size.
    height, width = actual.shape[:2]
    assert golden.shape[0] * width == golden.shape[1] * height
    golden = cv2.resize(golden, (width, height), interpolation=cv2.INTER_AREA)
    mismatched = (actual[:, :, 3] > 127) != (golden[:, :, 3] > 127)
    difference = np.count_nonzero(mismatched) / mismatched.size
    assert difference < _MAX_ALPHA_DIFFERENCE, f"Alpha masks differ by {difference:.2%}"
