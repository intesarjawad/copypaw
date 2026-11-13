"""Captured motion on the sample clip must match the committed golden files.

The goldens were written on macOS arm64; onnxruntime CPU kernels can differ on
other platforms, so a failure elsewhere is a measurement to record, not a cue
to loosen the tolerances.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from copypaw.cli import app
from copypaw.models import MODELS, REGISTRY, Mode
from copypaw.motion.motion_file import MotionFile

pytestmark = pytest.mark.slow
_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE = _ROOT / "samples/pets/dog_walk.mp4"
_GOLDEN = _ROOT / "tests/golden"
_ANGLE_TOLERANCE_DEG = 0.5
_ROOT_Y_TOLERANCE = 1e-3  # Body lengths, the BVH unit.
_ROOT_CHANNELS = 6  # Root position then rotation.
_ROOT_Y_CHANNEL = 1
_JOINT_CHANNELS = 3


def _read_bvh(path: Path) -> tuple[list[str], NDArray[np.float64]]:
    """Return the driver joint names and a (frames, channels) table from a BVH file."""
    text = path.read_text(encoding="utf-8")
    header, motion = text.split("MOTION\n")
    names = [line.split()[1] for line in header.splitlines() if line.startswith("JOINT ")]
    rows = [line.split() for line in motion.splitlines()[2:] if line]
    return names, np.array(rows, dtype=np.float64)


def test_capture_matches_golden(tmp_path: Path) -> None:
    golden_motion = _GOLDEN / "dog_walk.motion.npz"
    golden_bvh = _GOLDEN / "dog_walk.bvh"
    if not (_SAMPLE.is_file() and golden_motion.is_file() and golden_bvh.is_file()):
        pytest.skip("Sample clip or golden files are absent; run scripts/make_goldens.py")
    cache = Path(os.environ.get("COPYPAW_CACHE", "~/.cache/copypaw/models")).expanduser()
    for key in MODELS[Mode.BALANCED]:
        if not (cache / REGISTRY[key].filename).is_file():
            pytest.skip(f"Cached model is absent: {key}; run copypaw models fetch")

    result = CliRunner().invoke(app, ["capture", str(_SAMPLE), "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output

    expected = MotionFile.load(golden_motion)
    actual = MotionFile.load(tmp_path / "motion.npz")
    assert len(actual.kp) == len(expected.kp)
    assert actual.fps == expected.fps
    np.testing.assert_array_equal(actual.contacts, expected.contacts)

    names, want = _read_bvh(golden_bvh)
    got_names, got = _read_bvh(tmp_path / "motion.bvh")
    assert got_names == names
    assert got.shape == want.shape

    root_y = np.abs(got[:, _ROOT_Y_CHANNEL] - want[:, _ROOT_Y_CHANNEL])
    frame = int(np.argmax(root_y))
    assert root_y[frame] <= _ROOT_Y_TOLERANCE, (
        f"Root y differs by {root_y[frame]:.6f} at frame {frame}"
    )
    for joint, name in enumerate(names):
        if not name.endswith("_drv"):
            continue
        column = _ROOT_CHANNELS + joint * _JOINT_CHANNELS
        delta = (got[:, column] - want[:, column] + 180) % 360 - 180
        frame = int(np.argmax(np.abs(delta)))
        assert abs(delta[frame]) <= _ANGLE_TOLERANCE_DEG, (
            f"{name} differs by {delta[frame]:.4f} degrees at frame {frame}"
        )
