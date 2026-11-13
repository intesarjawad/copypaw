import json
from pathlib import Path

import numpy as np

from copypaw.motion.motion_file import MotionFile
from scripts.make_goldens import write_web_motion


def _motion() -> MotionFile:
    kp = np.zeros((2, 17, 2), dtype=np.float32)
    kp[0, 3] = (10.5, 20.25)
    kp[1, 16] = (0.1, 7.0)
    contacts = np.zeros((2, 4), dtype=np.bool_)
    contacts[1, 2] = True
    return MotionFile(
        kp=kp,
        scores=np.full((2, 17), 0.75, dtype=np.float32),
        fps=30.0,
        facing="left",
        contacts=contacts,
        ground_line_px=np.array([0.5, 180.0]),
        travel_root_x=np.array([0.0, 2.5], dtype=np.float32),
        source="trot.mp4",
    )


def test_web_export_keeps_every_field_and_names_the_file_after_the_folder(tmp_path: Path) -> None:
    folder = tmp_path / "trot"
    folder.mkdir()
    _motion().save(folder / "motion.npz")

    out = write_web_motion(folder / "motion.npz", tmp_path / "web")

    assert out == tmp_path / "web" / "trot.motion.json"
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["version"] == 1
    assert record["fps"] == 30.0
    assert record["facing"] == "left"
    assert record["source"] == "trot.mp4"
    assert len(record["kp"]) == 2
    assert len(record["kp"][0]) == 17
    assert record["kp"][0][3] == [10.5, 20.25]
    assert record["kp"][0][0] == [0.0, 0.0]
    assert record["scores"][1][16] == 0.75
    assert record["contacts"][1] == [False, False, True, False]
    assert record["ground_line_px"] == [0.5, 180.0]
    assert record["travel_root_x"] == [0.0, 2.5]


def test_float32_values_are_written_exactly(tmp_path: Path) -> None:
    folder = tmp_path / "walk"
    folder.mkdir()
    _motion().save(folder / "motion.npz")

    record = json.loads(write_web_motion(folder / "motion.npz", tmp_path).read_text("utf-8"))

    # 0.1 is not a float32; the nearest float32 is 13421773 / 2**27.
    assert record["kp"][1][16][0] == 13421773 / 2**27


def test_an_archive_outside_a_gallery_folder_is_named_after_its_stem(tmp_path: Path) -> None:
    _motion().save(tmp_path / "dog_walk.motion.npz")

    out = write_web_motion(tmp_path / "dog_walk.motion.npz", tmp_path / "web")

    assert out == tmp_path / "web" / "dog_walk.motion.json"
