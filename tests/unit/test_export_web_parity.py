import numpy as np

from scripts.export_web_parity import mask_runs


def test_mask_runs_give_each_rows_filled_spans_as_half_open_pairs() -> None:
    mask = np.array([[0, 255, 255, 0, 255], [0, 0, 0, 0, 0], [255, 255, 255, 255, 255]], np.uint8)

    assert mask_runs(mask) == [[1, 3, 4, 5], [], [0, 5]]
