from pathlib import Path

import numpy as np
import pytest

from copypaw.evaluate.fidelity import Fidelity, PawFidelity
from copypaw.motion.motion_file import MotionFile
from scripts.eval_motion import (
    GalleryMotion,
    Row,
    calibration_payload,
    gait_section,
    replace_section,
    run,
)
from scripts.label_contacts import (
    Labels,
    Stance,
    load_labels,
    merge_all,
    merge_labels,
    write_labels,
)

SLOTS = ("near_front", "far_front", "near_back", "far_back")
FPS = 30.0
FRAMES = 180
CYCLE = 60
NAMES = (
    "root",
    "hip",
    "neck",
    "torso",
    "left_hip_f",
    "left_knee_f",
    "left_foot_f",
    "right_hip_f",
    "right_knee_f",
    "right_foot_f",
    "left_hip_b",
    "left_knee_b",
    "left_foot_b",
    "right_hip_b",
    "right_knee_b",
    "right_foot_b",
)


def _labels(labeller: str, onsets: list[int], lift_offs: list[int]) -> Labels:
    paws = {
        slot: tuple(
            Stance(float(onset), float(lift_off))
            for onset, lift_off in zip(onsets, lift_offs, strict=True)
        )
        for slot in SLOTS
    }
    return Labels("dog_walk", labeller, 30.0, paws)


def test_merge_keeps_only_stances_both_labellers_placed_together() -> None:
    first = _labels("first", [10, 40], [20, 50])
    second = _labels("second", [11, 43], [21, 52])

    merged, agreement = merge_labels(first, second)

    assert agreement == 0.5
    assert merged.paws["near_front"] == (Stance(10.5, 20.5),)
    assert merged.labeller == "merged"


def test_label_files_round_trip_and_keep_integral_frames(tmp_path: Path) -> None:
    labels = _labels("first", [10, 40], [20, 50])
    path = tmp_path / "label.json"

    write_labels(path, labels)
    loaded = load_labels(path)

    assert loaded == labels
    assert '"onset": 10' in path.read_text(encoding="utf-8")


def test_merge_check_passes_for_a_fresh_merge(tmp_path: Path) -> None:
    motions = tmp_path / "motions"
    motion = motions / "dog_walk"
    motion.mkdir(parents=True)
    (motion / "motion.npz").touch()
    write_labels(motion / "contacts_reference.first.json", _labels("first", [10], [20]))
    write_labels(motion / "contacts_reference.second.json", _labels("second", [11], [21]))

    assert merge_all(motions, check=False) == 0
    assert merge_all(motions, check=True) == 0

    (motion / "contacts_reference.json").write_text("{}\n", encoding="utf-8")
    assert merge_all(motions, check=True) == 1


def test_merge_check_with_no_motions_is_a_no_op(tmp_path: Path) -> None:
    assert merge_all(tmp_path / "absent", check=True) == 0


def test_section_replacement_keeps_other_sections() -> None:
    document = "intro\n\n## Gait fidelity\n\nold\n\n## Other\n\nkeep\n"

    rewritten = replace_section(document, "## Gait fidelity\n\nnew\n")

    assert rewritten == "intro\n\n## Gait fidelity\n\nnew\n\n## Other\n\nkeep\n"


def _motion() -> MotionFile:
    return MotionFile(
        kp=np.zeros((10, 17, 2), dtype=np.float32),
        scores=np.ones((10, 17), dtype=np.float32),
        fps=30.0,
        facing="right",
        contacts=np.zeros((10, 4), dtype=np.bool_),
        ground_line_px=np.array([0.0, 100.0]),
        travel_root_x=np.zeros(10, dtype=np.float32),
        source="clip.mp4",
    )


def _fidelity(gait: float | None, contact: float, locomotion: bool = True):
    paw = PawFidelity(contact, 0.0 if locomotion else None, gait, 0.0 if gait is not None else None)
    return Fidelity(
        (paw,),
        contact,
        0.0 if locomotion else None,
        gait,
        0.0 if gait is not None else None,
        locomotion,
    )


def test_section_says_where_matched_is_not_higher() -> None:
    motion = GalleryMotion("dog_walk", "dog", "walk", _motion())
    row = Row(motion, "cat", _fidelity(0.5, 0.5), (_fidelity(0.7, 0.4),), (), None)

    section = gait_section((row,))

    assert "0.500" in section
    assert "matched gait r 0.500 is not higher than mismatched 0.700" in section


def test_section_prints_the_plain_words_for_non_locomotion() -> None:
    motion = GalleryMotion("dog_sit", "dog", "sit", _motion())
    row = Row(motion, "cat", _fidelity(None, 0.9, locomotion=False), (), (), None)

    section = gait_section((row,))

    assert "timing score not shown: no repeating steps found" in section
    assert "| dog_sit | cat | matched | n/a | 0.900 |" in section


def test_calibration_uses_the_95th_percentile_of_mismatched_pairs() -> None:
    motion = GalleryMotion("dog_walk", "dog", "walk", _motion())
    row = Row(
        motion,
        "cat",
        _fidelity(0.9, 0.9),
        tuple(_fidelity(value, 0.5) for value in (0.1, 0.2, 0.3, 0.4)),
        (),
        None,
    )

    payload = calibration_payload((row,), 0.06, 0.3, 0.8)

    assert payload["mismatched_pairs"] == 4
    assert payload["gait_r_p95"] == pytest.approx(0.385)


def test_calibration_refuses_an_empty_mismatched_distribution() -> None:
    motion = GalleryMotion("dog_walk", "dog", "walk", _motion())
    row = Row(motion, "cat", _fidelity(0.9, 0.9), (), (), None)

    with pytest.raises(ValueError, match="No mismatched"):
        calibration_payload((row,), 0.06, 0.3, 0.8)


def _walk_motion(source: str) -> MotionFile:
    frames = np.arange(FRAMES)
    phase = (frames % CYCLE) / CYCLE
    stance = phase < 0.6
    swing = np.clip((phase - 0.6) / 0.4, 0, 1)
    paw_y = 100 - 10 * np.sin(np.pi * swing)
    paw_x = np.linspace(50, 60, FRAMES)
    kp = np.zeros((FRAMES, 17, 2), dtype=np.float32)
    kp[:, 3] = [100, 0]
    kp[:, 4] = [0, 0]
    for paw, joint in enumerate((7, 10, 13, 16)):
        kp[:, joint, 0] = paw_x + paw
        kp[:, joint, 1] = paw_y
    for paw, joint in enumerate((5, 8, 11, 14)):
        kp[:, joint, 0] = paw_x + paw
        kp[:, joint, 1] = paw_y - 20
    return MotionFile(
        kp=kp,
        scores=np.ones((FRAMES, 17), dtype=np.float32),
        fps=FPS,
        facing="right",
        contacts=np.repeat(stance[:, None], 4, axis=1),
        ground_line_px=np.array([0.0, 100.0]),
        travel_root_x=(0.5 * frames / FPS).astype(np.float32),
        source=source,
    )


def _render(path: Path, motion: MotionFile) -> None:
    values: dict[str, np.ndarray] = {
        "root": np.column_stack([np.zeros(FRAMES), np.zeros(FRAMES)]),
        "neck": np.column_stack([np.full(FRAMES, 100.0), np.zeros(FRAMES)]),
        "torso": np.column_stack([np.full(FRAMES, 50.0), np.full(FRAMES, 50.0)]),
    }
    values["hip"] = values["root"]
    for paw, name in enumerate(("left_hip_f", "right_hip_f", "left_hip_b", "right_hip_b")):
        root = (5, 8, 11, 14)[paw]
        values[name] = motion.kp[:, root].astype(np.float64)
        paw_point = motion.kp[:, (7, 10, 13, 16)[paw]].astype(np.float64)
        values[name.replace("hip", "knee")] = paw_point + np.array([0.0, -10.0])
    for paw, name in enumerate(("left_foot_f", "right_foot_f", "left_foot_b", "right_foot_b")):
        values[name] = motion.kp[:, (7, 10, 13, 16)[paw]].astype(np.float64)
    points = np.stack([values[name] for name in NAMES], axis=1).astype(np.float32)
    # The engine reports world positions with y up.
    points[..., 1] *= -1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joints=points,
        joints_unfiltered=points,
        names=np.asarray(NAMES, dtype=np.str_),
        fps=np.float64(FPS),
    )


def _reference(motion: MotionFile, path: Path) -> None:
    onsets = [60, 120]
    paws = {slot: tuple(Stance(onset, onset + 36) for onset in onsets) for slot in SLOTS}
    write_labels(path, Labels(motion.source, "merged", FPS, paws))


def test_check_regenerates_the_section_and_calibration(tmp_path: Path) -> None:
    motions = tmp_path / "motions"
    drawings = tmp_path / "samples/drawings"
    renders = tmp_path / "evaluations/renders"
    upstream = tmp_path / "evaluations/upstream"
    docs = tmp_path / "docs/evaluation.md"
    adr = tmp_path / "docs/adr/ADR-005.md"
    calibration = tmp_path / "calibration.json"
    drawing = drawings / "cat"
    drawing.mkdir(parents=True)
    (drawing / "char_cfg.yaml").write_text("{}\n", encoding="utf-8")

    for name in ("dog_walk", "cat_walk"):
        motion = _walk_motion(name)
        directory = motions / name
        directory.mkdir(parents=True)
        motion.save(directory / "motion.npz")
        _render(renders / name / "cat/render_joints.npz", motion)
        _reference(motion, directory / "contacts_reference.json")
        write_labels(
            directory / "contacts_reference.first.json", _labels("first", [60, 120], [96, 156])
        )
        write_labels(
            directory / "contacts_reference.second.json", _labels("second", [61, 121], [97, 157])
        )
    _render(upstream / "zombie/cat/render_joints.npz", _walk_motion("zombie"))

    options = {
        "motions_dir": motions,
        "drawings_dir": drawings,
        "render_dir": renders,
        "upstream_dir": upstream,
        "docs_path": docs,
        "adr_path": adr,
        "calibration_path": calibration,
    }
    assert run(**options, check=False) == 0
    assert run(**options, check=True) == 0
    assert "dog_walk" in docs.read_text(encoding="utf-8")
    assert "Threshold sweep" in adr.read_text(encoding="utf-8")
    assert "gait_r_p95" in calibration.read_text(encoding="utf-8")

    calibration.write_text("{}\n", encoding="utf-8")
    assert run(**options, check=True) == 1
