from typing import Any

import numpy as np
import pytest

from scripts.build_light_table import (
    CROP_MARGIN_PX,
    at_frames,
    atlas,
    dog_box,
    drawing_paws,
    exposure_sheet,
    facing_keypoints,
    ink_box,
    matte,
    smooth,
)

WHITE = 255


def _paper(count: int, height: int, width: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return np.full((count, height, width), WHITE, dtype=np.uint8)


def test_ink_box_spans_every_frame_with_a_margin() -> None:
    frames = _paper(2, 100, 100)
    frames[0, 40:50, 30:35] = 0
    frames[1, 60:70, 50:55] = 0
    assert ink_box(frames) == (
        30 - CROP_MARGIN_PX,
        40 - CROP_MARGIN_PX,
        55 + CROP_MARGIN_PX,
        70 + CROP_MARGIN_PX,
    )


def test_ink_box_rejects_a_blank_drawing() -> None:
    with pytest.raises(ValueError, match="no ink"):
        ink_box(_paper(1, 10, 10))


def test_atlas_lays_frames_row_by_row_and_pads_with_paper() -> None:
    frames = np.stack([np.full((2, 3), index, dtype=np.uint8) for index in range(5)])
    sheet = atlas(frames, columns=2)
    assert sheet.shape == (6, 6)
    assert sheet[0, 3] == 1
    assert sheet[2, 0] == 2
    assert sheet[4, 3] == WHITE


def test_facing_keypoints_mirror_a_dog_walking_left() -> None:
    keypoints = np.zeros((2, 17, 2), dtype=np.float32)
    keypoints[:, :, 0] = 10
    assert facing_keypoints(keypoints, "right", 100)[0, 0, 0] == 10
    assert facing_keypoints(keypoints, "left", 100)[0, 0, 0] == 90


def test_at_frames_picks_the_nearest_sample_in_time() -> None:
    samples = np.arange(10)
    # Ten samples a second read at the output's 15 frames a second.
    assert at_frames(samples, 10.0, 4).tolist() == [0, 1, 1, 2]


def test_smooth_holds_the_ends() -> None:
    track = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    assert smooth(track, 1).tolist() == pytest.approx([0, 1, 1, 1, 0])


def test_drawing_paws_follow_the_lowest_ink() -> None:
    drawing = _paper(2, 100, 100)
    drawing[:, 10:20, 70:90] = 0
    drawing[0, 90:100, 20:30] = 0
    drawing[1, 90:100, 40:50] = 0
    paws = drawing_paws(drawing)
    assert paws[1] > paws[0]
    assert 0.2 < paws[0] < 0.5


def test_matte_fills_the_dog_and_leaves_the_set() -> None:
    frames = np.full((3, 120, 200), 200, dtype=np.uint8)
    # A dark body with four legs on a pale wall, in the middle frame only.
    frames[1, 40:70, 60:140] = 60
    for x in (64, 84, 116, 132):
        frames[1, 70:100, x : x + 6] = 60
    keypoints = np.zeros((3, 17, 2), dtype=np.float32)
    points = {
        0: (55, 42), 1: (58, 42), 2: (56, 48), 3: (66, 50), 4: (136, 46),
        5: (70, 60), 6: (67, 80), 7: (67, 98), 8: (86, 60), 9: (87, 80), 10: (87, 98),
        11: (118, 60), 12: (119, 80), 13: (119, 98), 14: (134, 60), 15: (135, 80), 16: (135, 98),
    }  # fmt: skip
    for index, point in points.items():
        keypoints[:, index] = point
    mattes = matte(frames, keypoints, floor_y=100)
    assert mattes[1, 55, 100] > 200
    assert mattes[1, 10, 10] < 50
    assert mattes[1, 110, 100] < 50


def test_exposure_sheet_stands_the_drawing_on_the_clip_floor() -> None:
    drawing = _paper(3, 200, 100)
    drawing[:, 50:150, 20:80] = 0
    drawing[1, 150:160, 20:30] = 0
    box = ink_box(drawing)
    motion = {
        "kp": np.zeros((3, 17, 2), dtype=np.float32),
        "fps": 15.0,
        "ground_line_px": np.array([0.0, 80.0]),
        "facing": "right",
        "contacts": np.ones((3, 4), dtype=bool),
    }
    footfall = {"pet": ["111"] * 4, "drawing": ["101"] * 4}
    fidelity = {"contact_f1": 0.5, "locomotion": False}
    sheet: dict[str, Any] = exposure_sheet(
        footfall, fidelity, motion, (160, 100), drawing, box, [0.1, 0.2, 0.9, 0.8]
    )
    assert sheet["contactMatch"] == 0.0
    assert sheet["contactMatchStage"] == "corrected render footfalls"
    assert sheet["uncorrectedContactF1"] == 0.5
    assert sheet["frames"] == 3
    assert sheet["footfalls"]["drawing"] == ["101"] * 4
    assert sheet["landings"] == [0, 0, 0, 0]
    assert sheet["dog"]["floor"] == pytest.approx(0.8)
    top, bottom = box[1], box[3]
    # Each frame stands on its own lowest ink: the second frame's paw reaches lower.
    lows = [(row - top) / (bottom - top) for row in (150, 160, 150)]
    assert sheet["drawing"]["lows"] == pytest.approx(lows, abs=1e-4)
    assert sheet["dog"]["box"] == [0.1, 0.2, 0.9, 0.8]
    assert sheet["drawing"]["height"] == pytest.approx((bottom - top) / 200, abs=1e-4)
    assert len(sheet["dog"]["paws"]) == len(sheet["drawing"]["paws"]) == 3
    assert sheet["dog"]["span"] == 0


def test_exposure_sheet_refuses_short_footfall_tracks() -> None:
    drawing = _paper(3, 20, 20)
    drawing[:, 5:10, 5:10] = 0
    motion = {
        "kp": np.zeros((3, 17, 2), dtype=np.float32),
        "fps": 15.0,
        "ground_line_px": np.array([0.0, 8.0]),
        "facing": "right",
        "contacts": np.ones((3, 4), dtype=bool),
    }
    footfall = {"pet": ["11"] * 4, "drawing": ["11"] * 4}
    with pytest.raises(ValueError, match="fewer frames"):
        exposure_sheet(
            footfall,
            {"contact_f1": 0.0, "locomotion": False},
            motion,
            (20, 20),
            drawing,
            (0, 0, 20, 20),
            [0.0, 0.0, 1.0, 1.0],
        )


def test_dog_box_covers_the_dog_in_every_frame() -> None:
    mattes = np.zeros((2, 100, 200), dtype=np.uint8)
    mattes[0, 40:90, 20:60] = 255
    mattes[1, 30:90, 100:150] = 255
    assert dog_box(mattes) == [0.1, 0.3, 0.75, 0.9]


def test_landing_counts_use_only_the_displayed_pet_footfalls() -> None:
    drawing = _paper(3, 20, 20)
    drawing[:, 5:10, 5:10] = 0
    motion = {
        "kp": np.zeros((3, 17, 2), dtype=np.float32),
        "fps": 15.0,
        "ground_line_px": np.array([0.0, 8.0]),
        "facing": "right",
        "contacts": np.ones((3, 4), dtype=bool),
    }
    sheet = exposure_sheet(
        {"pet": ["011"] * 4, "drawing": ["011"] * 4},
        {"contact_f1": 0.0, "locomotion": False},
        motion,
        (20, 20),
        drawing,
        (0, 0, 20, 20),
        [0.0, 0.0, 1.0, 1.0],
    )
    assert sheet["landings"] == [1, 1, 1, 1]
