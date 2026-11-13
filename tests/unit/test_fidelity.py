import json
import math
from pathlib import Path

import numpy as np
import pytest

from copypaw.evaluate.fidelity import (
    CalibrationError,
    DrawingJoints,
    autocorrelation_period,
    contact_f1,
    drawing_contacts,
    duty_factor_error,
    evaluate,
    is_locomotion,
    load_calibration,
    pearson_r,
    period_error_ratio,
    timing_words,
    write_fidelity,
)
from copypaw.motion.motion_file import MotionFile

FPS = 30.0
CYCLE = 60
FRAMES = 180
PAW_INDICES = (7, 10, 13, 16)
PAWS = PAW_INDICES
LEG_ROOTS = (5, 8, 11, 14)
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


def _walk_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.arange(FRAMES)
    phase = (frames % CYCLE) / CYCLE
    stance = phase < 0.6
    swing = np.clip((phase - 0.6) / 0.4, 0, 1)
    clearance = 10 * np.sin(np.pi * swing)
    paw_y = 100 - clearance
    paw_x = np.linspace(50, 60, FRAMES)
    contacts = np.repeat(stance[:, None], 4, axis=1)
    return paw_x, paw_y, contacts


def _clip() -> MotionFile:
    paw_x, paw_y, contacts = _walk_curves()
    kp = np.zeros((FRAMES, 17, 2), dtype=np.float32)
    kp[:, 3] = [100, 0]
    kp[:, 4] = [0, 0]
    for paw, joint in enumerate(PAWS):
        kp[:, joint, 0] = paw_x + paw
        kp[:, joint, 1] = paw_y
    for paw, joint in enumerate(LEG_ROOTS):
        kp[:, joint, 0] = paw_x + paw
        kp[:, joint, 1] = paw_y - 20
    return MotionFile(
        kp=kp,
        scores=np.ones((FRAMES, 17), dtype=np.float32),
        fps=FPS,
        facing="right",
        contacts=contacts,
        ground_line_px=np.array([0.0, 100.0]),
        travel_root_x=(0.5 * np.arange(FRAMES) / FPS).astype(np.float32),
        source="walk.mp4",
    )


def _drawing(clip: MotionFile | None = None) -> DrawingJoints:
    motion = _clip() if clip is None else clip
    values: dict[str, np.ndarray] = {
        "root": np.column_stack([np.zeros(FRAMES), np.zeros(FRAMES)]),
        "neck": np.column_stack([np.full(FRAMES, 100.0), np.zeros(FRAMES)]),
        "torso": np.column_stack([np.full(FRAMES, 50.0), np.full(FRAMES, 50.0)]),
    }
    values["hip"] = values["root"]
    for paw, name in enumerate(("left_hip_f", "right_hip_f", "left_hip_b", "right_hip_b")):
        values[name] = motion.kp[:, LEG_ROOTS[paw]].astype(np.float64)
        values[name.replace("hip", "knee")] = np.column_stack(
            [motion.kp[:, PAW_INDICES[paw], 0], motion.kp[:, PAW_INDICES[paw], 1] - 10.0]
        )
    for paw, name in enumerate(("left_foot_f", "right_foot_f", "left_foot_b", "right_foot_b")):
        values[name] = motion.kp[:, PAW_INDICES[paw]].astype(np.float64)
    points = np.stack([values[name] for name in NAMES], axis=1).astype(np.float32)
    # The engine reports world positions with y up.
    points[..., 1] *= -1
    return DrawingJoints(joints=points, joints_unfiltered=points, names=NAMES, fps=FPS)


def test_pearson_is_one_for_identical_curves_and_negative_for_half_a_cycle() -> None:
    frames = np.arange(120)
    curve = np.sin(2 * np.pi * frames / 20)

    assert pearson_r(curve, curve.copy()) == pytest.approx(1.0)
    assert pearson_r(curve, np.roll(curve, 10)) < 0


def test_contact_f1_counts_only_onsets_within_one_frame() -> None:
    clip = np.zeros((12, 4), dtype=np.bool_)
    drawing = np.zeros((12, 4), dtype=np.bool_)
    for paw in range(4):
        clip[[2, 8], paw] = True
        drawing[[3, 8], paw] = True

    assert contact_f1(clip, drawing) == 1.0

    drawing[:, :] = False
    for paw in range(4):
        drawing[[4, 8], paw] = True
    assert contact_f1(clip, drawing) == 0.5


def test_contact_f1_is_one_when_both_sides_have_no_onsets() -> None:
    empty = np.zeros((8, 4), dtype=np.bool_)

    assert contact_f1(empty, empty) == 1.0


def test_duty_factor_error_reads_the_stance_fraction_of_square_waves() -> None:
    clip = np.zeros((10, 4), dtype=np.bool_)
    drawing = np.zeros((10, 4), dtype=np.bool_)
    clip[:6] = True
    drawing[:3] = True

    assert duty_factor_error(clip, drawing) == pytest.approx(0.3)


def test_autocorrelation_period_finds_the_first_repeat() -> None:
    frames = np.arange(240)
    curve = np.sin(2 * np.pi * frames / 12)

    assert autocorrelation_period(curve, FPS) == pytest.approx(12 / FPS)
    assert period_error_ratio(np.sin(2 * np.pi * frames / 10), curve, FPS) == pytest.approx(0.2)


def test_locomotion_requires_two_paw_contacts_and_a_moving_root() -> None:
    contacts = np.zeros((90, 4), dtype=np.bool_)
    for paw in range(4):
        contacts[5 + paw : 15, paw] = True
        for start in range(35, 85, 30):
            contact_start = start + paw
            contacts[contact_start : contact_start + 10, paw] = True
    travel = np.arange(90, dtype=np.float32) / FPS * 0.5
    kp = np.zeros((90, 17, 2), dtype=np.float32)

    assert is_locomotion(kp, contacts, np.arange(90) / FPS, travel)


def test_motion_file_supplies_its_own_locomotion_evidence() -> None:
    assert is_locomotion(_clip())


def test_sit_down_is_not_locomotion_even_when_the_root_moves() -> None:
    contacts = np.ones((60, 4), dtype=np.bool_)
    travel = np.arange(60, dtype=np.float32) / FPS
    kp = np.zeros((60, 17, 2), dtype=np.float32)

    assert not is_locomotion(kp, contacts, np.arange(60) / FPS, travel)


def test_a_root_exactly_at_the_speed_limit_is_not_locomotion() -> None:
    contacts = np.zeros((60, 4), dtype=np.bool_)
    for paw in range(4):
        contacts[5 + paw : 15 + paw, paw] = True
        contacts[35 + paw : 45 + paw, paw] = True
    kp = np.zeros((60, 17, 2), dtype=np.float32)
    travel = (0.19 * np.arange(60) / FPS).astype(np.float32)

    assert not is_locomotion(kp, contacts, np.arange(60) / FPS, travel)


def test_matched_walk_drawing_correlates_with_the_clip() -> None:
    result = evaluate(_clip(), _drawing())

    assert result.locomotion
    assert result.contact_f1 == pytest.approx(1.0)
    assert result.gait_r == pytest.approx(1.0)
    assert result.period_error_ratio == pytest.approx(0.0)
    assert len(result.per_paw) == 4


def test_a_drawing_facing_left_matches_like_its_mirror() -> None:
    drawing = _drawing()
    mirrored = drawing.joints_unfiltered.copy()
    mirrored[..., 0] = 1 - mirrored[..., 0]

    facing_left = DrawingJoints(joints=mirrored, joints_unfiltered=mirrored, names=NAMES, fps=FPS)

    result = evaluate(_clip(), facing_left)

    assert result.contact_f1 == pytest.approx(1.0)
    assert result.gait_r == pytest.approx(1.0)
    np.testing.assert_array_equal(drawing_contacts(facing_left), drawing_contacts(drawing))


def test_non_locomotion_reports_only_contact_timing() -> None:
    clip = _clip()
    still = MotionFile(
        kp=clip.kp,
        scores=clip.scores,
        fps=clip.fps,
        facing=clip.facing,
        contacts=np.ones((FRAMES, 4), dtype=np.bool_),
        ground_line_px=clip.ground_line_px,
        travel_root_x=clip.travel_root_x,
        source=clip.source,
    )

    result = evaluate(still, _drawing(clip))

    assert not result.locomotion
    assert result.duty_factor_error is None
    assert result.gait_r is None
    assert result.period_error_ratio is None
    assert 0 <= result.contact_f1 <= 1


def test_drawing_archive_without_the_split_falls_back_to_joints(tmp_path: Path) -> None:
    points = np.zeros((2, 3, 2), dtype=np.float32)
    path = tmp_path / "render_joints.npz"
    np.savez_compressed(
        path,
        joints=points,
        names=np.asarray(["root", "neck", "left_foot_f"], dtype=np.str_),
    )

    loaded = DrawingJoints.load(path)

    assert loaded.fps == 15.0
    np.testing.assert_array_equal(loaded.joints_unfiltered, points)


def test_timing_words_use_the_stored_percentile() -> None:
    assert timing_words(0.9, locomotion=True, percentile=0.6) == "timing match: strong"
    assert timing_words(0.7, locomotion=True, percentile=0.6) == "timing match: fair"
    assert timing_words(0.5, locomotion=True, percentile=0.6) == "timing match: weak"
    assert (
        timing_words(0.9, locomotion=False, percentile=0.6)
        == "timing score not shown: no repeating steps found"
    )


def test_fidelity_json_holds_the_matched_metrics_and_wording(tmp_path: Path) -> None:
    result = evaluate(_clip(), _drawing())
    path = tmp_path / "fidelity.json"

    write_fidelity(path, result, percentile=0.6)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["stage"] == "uncorrected retarget joints"
    assert written["timing"] == "timing match: strong"
    assert written["contact_f1"] == pytest.approx(1.0)
    assert written["gait_r"] == pytest.approx(1.0)


def test_calibration_requires_a_finite_percentile(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    with pytest.raises(CalibrationError, match="Missing"):
        load_calibration(path)

    path.write_text(json.dumps({"gait_r_p95": None}), encoding="utf-8")
    with pytest.raises(CalibrationError, match="gait_r_p95"):
        load_calibration(path)

    path.write_text(json.dumps({"gait_r_p95": 0.25}), encoding="utf-8")
    assert load_calibration(path) == 0.25


def test_autocorrelation_rejects_a_constant_curve() -> None:
    assert math.isnan(autocorrelation_period(np.ones(20), FPS))
