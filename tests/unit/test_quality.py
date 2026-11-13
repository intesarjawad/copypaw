import json
from pathlib import Path

import numpy as np
import pytest

from copypaw.capture.quality import assess_quality, write_quality


def test_low_sampling_rate_reports_reduced_smoothing() -> None:
    report = assess_quality(np.ones((20, 17), dtype=np.float32), np.arange(20) / 12)
    assert any("4 Hz" in warning and "smooth" in warning.lower() for warning in report.warnings)


@pytest.mark.parametrize(
    ("joint", "name"),
    [
        (3, "Neck"),
        (4, "Root_of_tail"),
        (7, "Left_front_paw"),
        (10, "Right_front_paw"),
        (13, "Left_back_paw"),
        (16, "Right_back_paw"),
    ],
)
def test_missing_required_landmark_has_named_error(joint: int, name: str) -> None:
    scores = np.ones((20, 17), dtype=np.float32)
    scores[:, joint] = 0
    report = assess_quality(scores, np.arange(20) / 30)
    assert len(report.errors) == 1
    assert name in report.errors[0]


def test_intermittently_observed_required_landmark_is_available() -> None:
    scores = np.ones((20, 17), dtype=np.float32)
    scores[:, 3] = 0
    scores[10, 3] = 0.3
    assert assess_quality(scores, np.arange(20) / 30).errors == ()


def test_six_dead_joints_trigger_filming_guide_warning() -> None:
    scores = np.ones((10, 17), dtype=np.float32)
    scores[:, :6] = 0
    report = assess_quality(scores, np.arange(len(scores)) / 30, swaps_corrected=4)
    assert report.per_joint_confident_fraction == (0.0,) * 6 + (1.0,) * 11
    assert report.frames_interpolated == 10
    assert report.swaps_corrected == 4
    assert len(report.warnings) == 1
    assert "filming guide" in report.warnings[0]


@pytest.mark.parametrize("poor_frames", [0, 3, 4, 10])
def test_warning_requires_more_than_thirty_percent_poor_frames(poor_frames: int) -> None:
    scores = np.ones((10, 17), dtype=np.float32)
    scores[:poor_frames, :6] = 0.2
    report = assess_quality(scores, np.arange(len(scores)) / 30)
    assert bool(report.warnings) == (poor_frames > 3)
    assert report.frames_interpolated == poor_frames
    assert report.per_joint_confident_fraction[:6] == (1 - poor_frames / 10,) * 6


def test_twelve_confident_joints_are_sufficient_and_threshold_is_inclusive() -> None:
    scores = np.full((10, 17), 0.5, dtype=np.float32)
    scores[:, :5] = 0
    original = scores.copy()
    report = assess_quality(scores, np.arange(len(scores)) / 30, min_score=0.5)
    assert report.warnings == ()
    assert report.per_joint_confident_fraction == (0.0,) * 5 + (1.0,) * 12
    np.testing.assert_array_equal(scores, original)


def test_interpolated_frame_count_counts_each_frame_once() -> None:
    scores = np.ones((4, 17), dtype=np.float32)
    scores[0, :3] = 0
    scores[2, 3] = 0
    report = assess_quality(scores, np.arange(len(scores)) / 30)
    assert report.frames_interpolated == 2
    assert report.per_joint_confident_fraction == (0.75,) * 4 + (1.0,) * 13


def test_empty_clip_has_finite_zero_coverage_and_warning() -> None:
    report = assess_quality(np.empty((0, 17), dtype=np.float32), np.empty(0))
    assert report.per_joint_confident_fraction == (0.0,) * 17
    assert report.frames_interpolated == 0
    assert report.warnings == ("Clip has no frames.",)


def test_writes_quality_json(tmp_path: Path) -> None:
    scores = np.ones((5, 17), dtype=np.float32)
    scores[0, 0] = 0
    report = assess_quality(scores, np.arange(len(scores)) / 30, swaps_corrected=3)
    path = write_quality(report, tmp_path)
    assert path == tmp_path / "quality.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "per_joint_confident_fraction": [0.8] + [1.0] * 16,
        "frames_interpolated": 1,
        "swaps_corrected": 3,
        "warnings": [],
        "errors": [],
    }


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_scores_are_rejected(value: float) -> None:
    scores = np.ones((3, 17), dtype=np.float32)
    scores[0, 0] = value
    with pytest.raises(ValueError, match="scores"):
        assess_quality(scores, np.arange(len(scores)) / 30)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_threshold_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="min_score"):
        assess_quality(np.ones((3, 17), dtype=np.float32), np.arange(3) / 30, min_score=value)


@pytest.mark.parametrize("shape", [(3,), (3, 16), (3, 17, 1)])
def test_invalid_shapes_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        assess_quality(np.ones(shape, dtype=np.float32), np.arange(3) / 30)


def test_negative_swap_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="swaps_corrected"):
        assess_quality(np.ones((3, 17), dtype=np.float32), np.arange(3) / 30, swaps_corrected=-1)
