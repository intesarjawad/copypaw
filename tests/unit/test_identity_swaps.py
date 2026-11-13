import numpy as np

from copypaw.evaluate.identity_swaps import Flips, align_chains, count_flips

FRONT_LEFT = (5, 6, 7)
FRONT_RIGHT = (8, 9, 10)


def _truth(frames: int) -> tuple[np.ndarray, np.ndarray]:
    gt = np.zeros((frames, 17, 2))
    gt[:, FRONT_LEFT, 0] = 0.0
    gt[:, FRONT_RIGHT, 0] = 10.0
    gt[:, :, 1] = np.arange(17)
    visible = np.zeros((frames, 17), dtype=np.bool_)
    visible[:, 5:11] = True
    return gt, visible


def _exchange(pred: np.ndarray, frame: int) -> None:
    left = pred[frame, FRONT_LEFT].copy()
    pred[frame, FRONT_LEFT] = pred[frame, FRONT_RIGHT]
    pred[frame, FRONT_RIGHT] = left


def test_a_consistent_prediction_never_flips() -> None:
    gt, visible = _truth(4)

    assert count_flips(gt.copy(), gt, visible) == Flips(0, 3)


def test_a_clip_wide_exchange_is_not_a_flip() -> None:
    gt, visible = _truth(4)
    pred = gt.copy()
    for frame in range(4):
        _exchange(pred, frame)

    assert count_flips(pred, gt, visible) == Flips(0, 3)


def test_each_change_of_assignment_is_one_flip() -> None:
    gt, visible = _truth(4)
    pred = gt.copy()
    _exchange(pred, 1)
    _exchange(pred, 2)

    # Straight, exchanged, exchanged, straight: changes after frames 0 and 2.
    assert count_flips(pred, gt, visible) == Flips(2, 3)


def test_only_consecutive_labelled_frames_count() -> None:
    gt, visible = _truth(4)
    visible[2] = False
    pred = gt.copy()
    _exchange(pred, 3)

    assert count_flips(pred, gt, visible) == Flips(0, 1)


def test_a_chain_with_one_visible_joint_still_takes_part() -> None:
    gt, visible = _truth(2)
    visible[:, [6, 7, 9, 10]] = False
    pred = gt.copy()
    _exchange(pred, 1)

    assert count_flips(pred, gt, visible) == Flips(1, 1)


def test_a_missing_prediction_is_no_evidence() -> None:
    gt, visible = _truth(2)
    pred = gt.copy()
    pred[1] = np.nan

    assert count_flips(pred, gt, visible) == Flips(0, 0)


def test_a_gap_is_not_a_flip_or_an_opportunity() -> None:
    gt, visible = _truth(4)
    pred = gt.copy()
    for frame in range(4):
        _exchange(pred, frame)
    pred[2] = np.nan

    # Only frames 0 and 1 can be compared, and both are exchanged.
    assert count_flips(pred, gt, visible) == Flips(0, 1)


def test_back_legs_are_counted_separately() -> None:
    gt, visible = _truth(2)
    gt[:, 11:14, 0] = 0.0
    gt[:, 14:17, 0] = 10.0
    visible[:, 11:] = True
    pred = gt.copy()
    pred[1, 11:14], pred[1, 14:17] = gt[1, 14:17], gt[1, 11:14]

    assert count_flips(pred, gt, visible) == Flips(1, 2)


def test_flip_rate_is_none_without_opportunities() -> None:
    assert Flips(0, 0).rate is None
    assert Flips(1, 4).rate == 0.25
    assert Flips(1, 4) + Flips(2, 2) == Flips(3, 6)


def test_alignment_names_chains_after_the_labels_over_the_whole_clip() -> None:
    gt, visible = _truth(3)
    pred = gt.copy().astype(np.float32)
    scores = np.zeros((3, 17), dtype=np.float32)
    scores[:, FRONT_LEFT] = 0.9
    scores[:, FRONT_RIGHT] = 0.4
    for frame in range(3):
        _exchange(pred, frame)
        scores[frame, FRONT_LEFT], scores[frame, FRONT_RIGHT] = 0.4, 0.9

    aligned, aligned_scores = align_chains(pred, scores, gt, visible)

    np.testing.assert_array_equal(aligned, gt.astype(np.float32))
    assert np.all(aligned_scores[:, FRONT_LEFT] == np.float32(0.9))
    assert np.all(aligned_scores[:, FRONT_RIGHT] == np.float32(0.4))


def test_alignment_leaves_a_matching_clip_alone() -> None:
    gt, visible = _truth(3)
    pred = gt.copy()
    scores = np.ones((3, 17), dtype=np.float32)

    aligned, _ = align_chains(pred, scores, gt, visible)

    np.testing.assert_array_equal(aligned, gt)
