import math

import numpy as np
import pytest

from copypaw.evaluate.pck import Detection, GroundTruth, average_precision, oks, pck

AREA = 10000.0


def _visible(*joints: int) -> np.ndarray:
    visible = np.zeros(17, dtype=np.bool_)
    visible[list(joints)] = True
    return visible


def _pose(x: float) -> np.ndarray:
    return np.column_stack([np.full(17, x), np.arange(17, dtype=np.float64)])


def test_oks_matches_a_hand_computed_example() -> None:
    gt = _pose(0.0)
    pred = gt.copy()
    pred[0] += (3.0, 4.0)

    # Joint 0: sigma 0.025, d^2 25, exp(-25 / (2 * 10000 * 0.05^2)) = exp(-0.5); joint 5 exact.
    assert oks(pred, gt, _visible(0, 5), AREA) == pytest.approx(0.5 * (math.exp(-0.5) + 1))


def test_oks_ignores_unlabelled_joints_and_scores_missing_ones_zero() -> None:
    gt = _pose(0.0)
    pred = gt.copy()
    pred[3] += 1000.0
    pred[5] = np.nan

    assert oks(pred, gt, _visible(0), AREA) == 1.0
    assert oks(pred, gt, _visible(0, 3), AREA) == pytest.approx(0.5)
    assert oks(pred, gt, _visible(0, 5), AREA) == pytest.approx(0.5)
    assert oks(pred, gt, _visible(), AREA) == 0.0


def test_oks_refuses_a_non_positive_area() -> None:
    with pytest.raises(ValueError, match="area"):
        oks(_pose(0.0), _pose(0.0), _visible(0), 0.0)


def _truth(image: int, x: float, category: int = 1) -> GroundTruth:
    return GroundTruth(image, category, _pose(x), np.ones(17, dtype=np.bool_), AREA)


def _detection(image: int, x: float, score: float, category: int = 1) -> Detection:
    return Detection(image, category, _pose(x), score)


def test_ap_with_every_animal_found_is_one() -> None:
    truths = [_truth(0, 0.0), _truth(0, 1000.0)]
    detections = [_detection(0, 0.0, 0.9), _detection(0, 1000.0, 0.8)]

    assert average_precision(truths, detections) == pytest.approx(1.0)


def test_ap_samples_precision_at_101_recall_points() -> None:
    truths = [_truth(0, 0.0), _truth(0, 1000.0)]
    detections = [_detection(0, 0.0, 0.9), _detection(0, 5000.0, 0.8)]

    # Precision 1 up to recall 0.5, which is 51 of the 101 recall points; none beyond.
    assert average_precision(truths, detections) == pytest.approx(51 / 101)


def test_ap_ranks_by_score() -> None:
    truths = [_truth(0, 0.0), _truth(0, 1000.0)]
    detections = [_detection(0, 0.0, 0.8), _detection(0, 5000.0, 0.95)]

    # The false positive comes first, so precision is 0.5 wherever recall is reached.
    assert average_precision(truths, detections) == pytest.approx(0.5 * 51 / 101)


def test_ap_averages_species_and_scores_a_missed_species_zero() -> None:
    truths = [_truth(0, 0.0), _truth(1, 0.0, category=2)]

    assert average_precision(truths, [_detection(0, 0.0, 0.9)]) == pytest.approx(0.5)


def test_ap_does_not_match_across_species_or_images() -> None:
    truths = [_truth(0, 0.0)]

    assert average_precision(truths, [_detection(1, 0.0, 0.9)]) == 0.0
    assert average_precision(truths, [_detection(0, 0.0, 0.9, category=2)]) == 0.0


def test_ap_needs_something_labelled() -> None:
    with pytest.raises(ValueError, match="No labelled"):
        average_precision([], [])


def test_pck_uses_a_radius_of_five_percent_of_the_root_box_area() -> None:
    gt = np.zeros((2, 17, 2))
    pred = np.zeros((2, 17, 2))
    pred[0, 0] = (0.5, 0.0)
    pred[0, 1] = (3.0, 4.0)
    pred[1, 0] = (0.0, 4.9)
    visible = np.zeros((2, 17), dtype=np.bool_)
    visible[0, :2] = True
    visible[1, 0] = True

    # Box area 400 gives a radius of 1; area 10000 gives 5.
    result = pck(pred, gt, visible, np.array([400.0, 10000.0]))

    assert result.overall == pytest.approx(2 / 3)
    assert result.per_joint[0] == 1.0
    assert result.per_joint[1] == 0.0
    assert result.per_joint[2] is None


def test_pck_counts_a_missing_prediction_as_a_miss() -> None:
    gt = np.zeros((1, 17, 2))
    pred = np.zeros((1, 17, 2))
    pred[0, 1] = np.nan
    visible = np.zeros((1, 17), dtype=np.bool_)
    visible[0, :2] = True

    assert pck(pred, gt, visible, np.array([100.0])).overall == 0.5


def test_pck_with_nothing_labelled_is_none() -> None:
    result = pck(np.zeros((1, 17, 2)), np.zeros((1, 17, 2)), _visible()[None], np.array([1.0]))

    assert result.overall is None
