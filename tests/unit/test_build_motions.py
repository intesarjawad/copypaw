import json
import shutil
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from copypaw.motion.motion_file import MotionFile
from scripts import build_motions
from scripts.build_motions import (
    BVH,
    CONTACTS_AUTO,
    NPZ,
    THUMB,
    build_folder,
    check_folder,
    compare_motions,
    contact_stances,
    read_evaluation_set,
    with_contacts,
    write_thumbnail,
)
from scripts.label_contacts import Stance, load_labels

FPS = 30
PAWS = (7, 10, 13, 16)
LIMBS = ((5, 6, 7, 3), (8, 9, 10, 3), (11, 12, 13, 4), (14, 15, 16, 4))


def _walk() -> MotionFile:
    count = 6 * FPS
    times = np.arange(count) / FPS
    kp = np.zeros((count, 17, 2), dtype=np.float32)
    kp[:, 4, 0] = 100 + 150 * times
    kp[:, 4, 1] = 100
    kp[:, 3] = kp[:, 4] + [100, 0]
    kp[:, 2] = kp[:, 3] + [30, -10]
    for paw, (joint, offset) in enumerate(
        zip(PAWS, (0, FPS // 2, FPS // 4, 3 * FPS // 4), strict=True)
    ):
        phase = ((np.arange(count) + offset) % FPS) / FPS
        swing = np.maximum(phase - 0.6, 0) / 0.4
        kp[:, joint, 0] = (
            kp[:, 4, 0] + (100 if paw < 2 else 0) + 25 - 150 * np.minimum(phase, 0.6) + 90 * swing
        )
        kp[:, joint, 1] = 200 - 30 * np.sin(np.pi * swing)
    for shoulder, elbow, paw, top in LIMBS:
        kp[:, shoulder] = kp[:, top] + (kp[:, paw] - kp[:, top]) / 3 + [3, 0]
        kp[:, elbow] = kp[:, top] + 2 * (kp[:, paw] - kp[:, top]) / 3 - [3, 0]
    return MotionFile(
        kp=kp,
        scores=np.ones((count, 17), dtype=np.float32),
        fps=float(FPS),
        facing="right",
        contacts=np.zeros((count, 4), dtype=np.bool_),
        ground_line_px=np.array([0.0, 200.0]),
        travel_root_x=np.zeros(count, dtype=np.float32),
        source="clip.mp4",
    )


def _capture(clip: Path, out_dir: Path) -> None:
    del clip
    build_motions._save(_walk(), out_dir)  # pyright: ignore[reportPrivateUsage]


def _clip(path: Path, count: int = 5) -> None:
    ffmpeg: Any = imageio_ffmpeg
    with closing(
        ffmpeg.write_frames(
            str(path),
            (64, 32),
            fps=10,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=None,
            output_params=["-crf", "10"],
            ffmpeg_log_level="error",
        )
    ) as writer:
        writer.send(None)
        for index in range(count):
            writer.send(np.full((32, 64, 3), 50 * index, dtype=np.uint8).tobytes())


def _calibration(tmp_path: Path, height: object, speed: object) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"contact_height_ratio": height, "contact_speed_ratio": speed}),
        encoding="utf-8",
    )
    return path


def _thresholds(path: Path) -> tuple[float, float]:
    return build_motions._thresholds(path)  # pyright: ignore[reportPrivateUsage]


def test_untuned_calibration_falls_back_to_the_initial_thresholds(tmp_path: Path) -> None:
    assert _thresholds(_calibration(tmp_path, None, None)) == (0.06, 0.3)


def test_tuned_calibration_is_used(tmp_path: Path) -> None:
    assert _thresholds(_calibration(tmp_path, 0.08, 0.45)) == (0.08, 0.45)


def test_half_tuned_calibration_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no contact thresholds"):
        _thresholds(_calibration(tmp_path, 0.08, None))


def test_missing_calibration_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing calibration"):
        _thresholds(tmp_path / "absent.json")


def test_evaluation_set_ignores_blank_lines_and_notes(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_set.txt"
    path.write_text("# launch set\ndog_walk\n\ncat_trot  # owner clip\n", encoding="utf-8")

    assert read_evaluation_set(path) == {"dog_walk", "cat_trot"}


def test_contact_runs_become_onset_and_lift_off_frames() -> None:
    contacts = np.zeros((6, 4), dtype=np.bool_)
    contacts[0:2, 0] = True
    contacts[3:4, 0] = True
    contacts[2:6, 3] = True

    assert contact_stances(contacts) == {
        "near_front": (Stance(0.0, 2.0), Stance(3.0, 4.0)),
        "far_front": (),
        "near_back": (),
        "far_back": (Stance(2.0, 6.0),),
    }


def test_thresholds_decide_which_frames_are_planted() -> None:
    walk = _walk()

    assert with_contacts(walk, 100.0, 100.0).contacts.all()
    assert not with_contacts(walk, 0.06, 0.0).contacts.any()


def test_thumbnail_is_the_middle_frame(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _clip(clip)
    thumb = tmp_path / THUMB

    write_thumbnail(clip, thumb)

    image = cv2.imread(str(thumb))
    assert image is not None
    assert image.shape == (32, 64, 3)
    assert abs(float(image.mean()) - 100) < 8


def test_thumbnail_of_an_unreadable_clip_is_refused(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a video")

    with pytest.raises(ValueError, match="thumbnail"):
        write_thumbnail(clip, tmp_path / THUMB)


def _folder(parent: Path, name: str = "dog_walk") -> Path:
    folder = parent / name
    folder.mkdir(parents=True)
    _clip(folder / "clip.mp4")
    return folder


def test_build_writes_every_generated_file(tmp_path: Path) -> None:
    folder = _folder(tmp_path)

    build_folder(folder, _capture, 0.06, 0.3, contacts_auto=True)

    assert {path.name for path in folder.iterdir()} == {
        "clip.mp4",
        NPZ,
        BVH,
        THUMB,
        CONTACTS_AUTO,
    }
    labels = load_labels(folder / CONTACTS_AUTO)
    assert (labels.motion, labels.labeller, labels.fps) == ("dog_walk", "auto", 30.0)
    assert MotionFile.load(folder / NPZ).source == "clip.mp4"


def test_build_leaves_out_the_auto_contacts_for_launch_folders(tmp_path: Path) -> None:
    folder = _folder(tmp_path)

    build_folder(folder, _capture, 0.06, 0.3, contacts_auto=False)

    assert not (folder / CONTACTS_AUTO).exists()


def test_reproduced_folder_has_no_problems(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    build_folder(folder, _capture, 0.06, 0.3, contacts_auto=False)

    assert check_folder(folder, _capture, 0.06, 0.3) == []


def test_check_names_a_missing_committed_file(tmp_path: Path) -> None:
    folder = _folder(tmp_path)

    assert check_folder(folder, _capture, 0.06, 0.3) == [f"missing {NPZ}, {BVH}"]


def _copy_built(tmp_path: Path) -> tuple[Path, Path]:
    folder = _folder(tmp_path / "committed")
    build_folder(folder, _capture, 0.06, 0.3, contacts_auto=False)
    other = tmp_path / "actual"
    shutil.copytree(folder, other)
    return folder, other


def test_changed_contacts_are_reported_with_their_frame(tmp_path: Path) -> None:
    expected, actual = _copy_built(tmp_path)
    motion = MotionFile.load(actual / NPZ)
    contacts = motion.contacts.copy()
    contacts[7, 2] = not contacts[7, 2]
    MotionFile(
        kp=motion.kp,
        scores=motion.scores,
        fps=motion.fps,
        facing=motion.facing,
        contacts=contacts,
        ground_line_px=motion.ground_line_px,
        travel_root_x=motion.travel_root_x,
        source=motion.source,
    ).save(actual / NPZ)

    assert compare_motions(expected, actual) == ["contacts differ at frame 7"]


def _edit_first_row(path: Path, column: int, delta: float) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    first = next(index for index, line in enumerate(lines) if line.startswith("Frame Time")) + 1
    values = lines[first].split()
    values[column] = f"{float(values[column]) + delta:.6f}"
    lines[first] = " ".join(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_degree_of_driver_rotation_is_outside_the_tolerance(tmp_path: Path) -> None:
    expected, actual = _copy_built(tmp_path)
    _edit_first_row(actual / BVH, 6, 1.0)

    assert compare_motions(expected, actual) == ["Neck_drv differs by 1.0000 degrees at frame 0"]


def test_a_third_of_a_degree_is_inside_the_tolerance(tmp_path: Path) -> None:
    expected, actual = _copy_built(tmp_path)
    _edit_first_row(actual / BVH, 6, 0.3)

    assert compare_motions(expected, actual) == []


def test_root_height_drift_is_reported(tmp_path: Path) -> None:
    expected, actual = _copy_built(tmp_path)
    _edit_first_row(actual / BVH, 1, 0.01)

    assert compare_motions(expected, actual) == ["root y differs by 0.010000 at frame 0"]


def _run(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["build_motions.py", *arguments])
    monkeypatch.setattr(build_motions, "accurate_capture", lambda: _capture)
    return build_motions.main()


def _gallery(tmp_path: Path) -> Path:
    motions = tmp_path / "motions"
    for name in ("dog_walk", "cat_walk"):
        _folder(motions, name)
    (motions / "evaluation_set.txt").write_text("dog_walk\n", encoding="utf-8")
    return motions


def test_default_run_builds_only_launch_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    motions = _gallery(tmp_path)
    calibration = _calibration(tmp_path, None, None)

    code = _run(monkeypatch, "--motions", str(motions), "--calibration", str(calibration))

    assert code == 0
    assert (motions / "dog_walk" / NPZ).is_file()
    assert not (motions / "dog_walk" / CONTACTS_AUTO).exists()
    assert not (motions / "cat_walk" / NPZ).exists()


def test_new_run_builds_folders_without_an_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    motions = _gallery(tmp_path)
    calibration = _calibration(tmp_path, None, None)
    arguments = ("--motions", str(motions), "--calibration", str(calibration))
    _run(monkeypatch, *arguments)
    launch_archive = (motions / "dog_walk" / NPZ).stat().st_mtime_ns

    code = _run(monkeypatch, "--new", *arguments)

    assert code == 0
    assert (motions / "cat_walk" / CONTACTS_AUTO).is_file()
    assert (motions / "dog_walk" / NPZ).stat().st_mtime_ns == launch_archive


def test_check_run_passes_for_a_fresh_build_and_fails_after_an_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    motions = _gallery(tmp_path)
    calibration = _calibration(tmp_path, None, None)
    arguments = ("--motions", str(motions), "--calibration", str(calibration))
    _run(monkeypatch, *arguments)

    assert _run(monkeypatch, "--check", *arguments) == 0

    _edit_first_row(motions / "dog_walk" / BVH, 1, 0.01)
    assert _run(monkeypatch, "--check", *arguments) == 1


def test_contacts_run_rewrites_stored_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    motions = _gallery(tmp_path)
    folder = motions / "dog_walk"
    _capture(folder / "clip.mp4", folder)
    calibration = _calibration(tmp_path, 100.0, 100.0)
    arguments = ("--motions", str(motions), "--calibration", str(calibration))

    assert _run(monkeypatch, "--contacts", *arguments) == 0
    assert MotionFile.load(folder / NPZ).contacts.all()


def test_contacts_run_without_archives_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    motions = tmp_path / "motions"
    motions.mkdir()
    calibration = _calibration(tmp_path, None, None)
    arguments = ("--motions", str(motions), "--calibration", str(calibration))

    assert _run(monkeypatch, "--contacts", *arguments) == 1
    assert "No gallery motions" in capsys.readouterr().err
