from pathlib import Path
from typing import Any

import numpy as np
import pytest

from copypaw.motion.motion_file import MotionFile, MotionFileError

FRAMES = 3


def _arrays() -> dict[str, Any]:
    kp = np.arange(FRAMES * 17 * 2, dtype=np.float32).reshape(FRAMES, 17, 2)
    return {
        "version": np.int64(1),
        "kp": kp,
        "scores": np.full((FRAMES, 17), 0.5, dtype=np.float32),
        "fps": np.float64(30.0),
        "facing": np.str_("left"),
        "contacts": np.array([[True, False, True, False]] * FRAMES, dtype=np.bool_),
        "ground_line_px": np.array([0.25, 400.0], dtype=np.float64),
        "travel_root_x": np.array([0.0, 0.1, 0.2], dtype=np.float32),
        "source": np.str_("dog-trot"),
    }


def _write(tmp_path: Path, **overrides: Any) -> Path:
    arrays = {**_arrays(), **overrides}
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, **{k: v for k, v in arrays.items() if v is not None})
    return path


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    arrays = _arrays()
    motion = MotionFile(
        kp=arrays["kp"],
        scores=arrays["scores"],
        fps=30.0,
        facing="left",
        contacts=arrays["contacts"],
        ground_line_px=arrays["ground_line_px"],
        travel_root_x=arrays["travel_root_x"],
        source="dog-trot",
    )
    path = tmp_path / "motion.npz"
    motion.save(path)

    loaded = MotionFile.load(path)

    assert loaded.version == 1
    assert loaded.fps == 30.0
    assert loaded.facing == "left"
    assert loaded.source == "dog-trot"
    np.testing.assert_array_equal(loaded.kp, arrays["kp"])
    np.testing.assert_array_equal(loaded.scores, arrays["scores"])
    np.testing.assert_array_equal(loaded.contacts, arrays["contacts"])
    np.testing.assert_array_equal(loaded.ground_line_px, arrays["ground_line_px"])
    np.testing.assert_array_equal(loaded.travel_root_x, arrays["travel_root_x"])
    assert loaded.kp.dtype == np.float32
    assert loaded.contacts.dtype == np.bool_


def test_missing_contacts_names_the_key(tmp_path: Path) -> None:
    with pytest.raises(MotionFileError, match="contacts"):
        MotionFile.load(_write(tmp_path, contacts=None))


def test_wrong_dtype_names_the_key(tmp_path: Path) -> None:
    with pytest.raises(MotionFileError, match="contacts"):
        MotionFile.load(_write(tmp_path, contacts=np.ones((FRAMES, 4), dtype=np.int8)))


def test_wrong_shape_names_the_key(tmp_path: Path) -> None:
    with pytest.raises(MotionFileError, match="ground_line_px"):
        MotionFile.load(_write(tmp_path, ground_line_px=np.zeros(3, dtype=np.float64)))


def test_mismatched_frame_counts_name_the_key(tmp_path: Path) -> None:
    with pytest.raises(MotionFileError, match="travel_root_x"):
        MotionFile.load(_write(tmp_path, travel_root_x=np.zeros(FRAMES + 1, dtype=np.float32)))


def test_unknown_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MotionFileError, match="version 2"):
        MotionFile.load(_write(tmp_path, version=np.int64(2)))


def test_unknown_facing_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MotionFileError, match="facing"):
        MotionFile.load(_write(tmp_path, facing=np.str_("up")))


@pytest.mark.parametrize("content", [b"", b"plain text, not an archive"])
def test_file_that_is_not_an_archive_is_named(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "broken.npz"
    path.write_bytes(content)

    with pytest.raises(MotionFileError) as caught:
        MotionFile.load(path)

    assert caught.value.code == "motion_unreadable"
    assert caught.value.what == f"Copypaw could not read {path} as a motion file."
