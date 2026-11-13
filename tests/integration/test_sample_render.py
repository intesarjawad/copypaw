"""The sample drawing keeps its line, fills its panel and stays whole in every frame."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.animate.render import render_character
from copypaw.motion.motion_file import MotionFile
from copypaw.rig.char_cfg import write_character
from copypaw.rig.edits import load_drawing, rig_drawing

pytestmark = pytest.mark.render

_ROOT = Path(__file__).resolve().parents[2]
_DRAWING = _ROOT / "copypaw/selftest/dragon.png"
_MOTION = _ROOT / "tests/golden/dog_walk.motion.npz"
_INK_BELOW = 96
# The page behind the drawing is rendered pure white; the paper inside it is not.
_PAGE_FROM = 250
_MIN_FILL = 0.85


def _ink_share(grey: NDArray[np.uint8], drawing: NDArray[np.bool_]) -> float:
    """Return how much of the drawing's area its dark line covers."""
    return float(np.count_nonzero((grey < _INK_BELOW) & drawing) / np.count_nonzero(drawing))


def test_sample_keeps_its_line_and_fills_the_panel(tmp_path: Path) -> None:
    image = load_drawing(_DRAWING)
    rig = rig_drawing(image)
    character = write_character(image, rig.snapped, tmp_path / "drawing")
    mask = cv2.imread(str(character.mask), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    drawn = _ink_share(
        np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8),
        np.asarray(mask > 0, dtype=np.bool_),
    )

    result = render_character(
        character.directory, MotionFile.load(_MOTION), "treadmill", tmp_path / "out"
    )

    height, width = result.frames_rgba.shape[1:3]
    reach = np.zeros((height, width), dtype=np.bool_)
    for index, frame in enumerate(result.frames_rgba):
        shown = np.asarray(np.any(frame[:, :, :3] < _PAGE_FROM, axis=2), dtype=np.bool_)
        rows, columns = np.nonzero(shown)
        edges = (rows.min(), height - 1 - rows.max(), columns.min(), width - 1 - columns.max())
        assert min(edges) > 0, f"frame {index} is cropped"
        grey: NDArray[np.uint8] = frame[:, :, :3].max(axis=2)
        # The line is as dark and, for its size, as heavy as it was drawn.
        assert int(grey[shown].min()) <= int(image.min()) + 16
        assert _ink_share(grey, shown) == pytest.approx(drawn, rel=0.25), f"frame {index}"
        reach |= shown
    rows, columns = np.nonzero(reach)
    fill = max(float(np.ptp(rows) + 1) / height, float(np.ptp(columns) + 1) / width)
    assert fill > _MIN_FILL
