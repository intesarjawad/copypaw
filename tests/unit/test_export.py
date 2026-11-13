import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.animate import export
from copypaw.animate.export import (
    CREDIT_HEIGHT,
    FRAME_HEIGHT,
    align_facing,
    compose_frames,
    load_clip_frames,
    side_by_side,
    write_drawing,
    write_footfall,
)
from copypaw.animate.strip import ROW_HEIGHT
from copypaw.capture.video import Frames

FRAMES = 4
SIZE = (8, 8)
STRIP_HEIGHT = 4 * ROW_HEIGHT


def _clip_frames() -> NDArray[np.uint8]:
    frames = np.zeros((FRAMES, *SIZE, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(FRAMES, dtype=np.uint8)[:, None, None] * 20
    return frames


def _drawing_frames(red: int = 30) -> NDArray[np.uint8]:
    frames = np.zeros((FRAMES, *SIZE, 4), dtype=np.uint8)
    frames[..., 0] = red
    frames[..., 1] = 60
    frames[..., 2] = 90
    frames[..., 3] = 255
    return frames


def _compose(
    credit: bool,
    contacts: NDArray[np.bool_] | None = None,
    drawing_contacts: NDArray[np.bool_] | None = None,
) -> NDArray[np.uint8]:
    if contacts is None:
        contacts = np.zeros((FRAMES, 4), dtype=np.bool_)
    return compose_frames(
        _clip_frames(),
        _drawing_frames(),
        facing_clip="right",
        facing_drawing="right",
        keep_facing=False,
        credit=credit,
        contacts=contacts,
        drawing_contacts=drawing_contacts,
    )


def _frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _reader(clip: Frames) -> Callable[..., Frames]:
    def read(*_args: object, **_kwargs: object) -> Frames:
        return clip

    return read


def _numbered_clip(count: int, fps: float) -> Frames:
    rgb = np.zeros((count, 2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(count, dtype=np.uint8)[:, None, None]
    times_s = np.arange(count, dtype=np.float64) / fps
    return Frames(rgb=rgb, t_s=times_s, source_fps=fps, fps=min(fps, 15.0), trimmed=False)


def test_a_slower_clip_repeats_frames_to_keep_the_drawing_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(export, "read_frames", _reader(_numbered_clip(60, 12.0)))

    clip = load_clip_frames(Path("clip.mp4"))

    # The render of a 60-frame, 12 fps motion has floor(59 / 12 * 15) + 1 frames.
    assert len(clip) == 74
    nearest = np.floor(np.arange(74) * 12 / 15 + 0.5).astype(np.uint8)
    np.testing.assert_array_equal(clip[:, 0, 0, 0], nearest)


def test_a_clip_sampled_at_the_output_rate_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export, "read_frames", _reader(_numbered_clip(75, 15.0)))

    clip = load_clip_frames(Path("clip.mp4"))

    np.testing.assert_array_equal(clip[:, 0, 0, 0], np.arange(75))


def test_video_keeps_frame_dimensions_that_are_not_multiples_of_sixteen(tmp_path: Path) -> None:
    frames = np.zeros((FRAMES, 18, 40, 4), dtype=np.uint8)
    frames[..., 3] = 255
    result = write_drawing(frames, 15.0, tmp_path / "drawing.mp4", tmp_path / "drawing.gif")

    capture = cv2.VideoCapture(str(result.mp4))
    try:
        size = (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    assert size == (40, 18)


def test_frames_are_flipped_only_when_the_facings_differ() -> None:
    frames = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)

    np.testing.assert_array_equal(align_facing(frames, "left", "right", False), frames[:, :, ::-1])
    np.testing.assert_array_equal(align_facing(frames, "right", "right", False), frames)
    np.testing.assert_array_equal(align_facing(frames, "left", "right", True), frames)


def test_side_by_side_writes_both_files_with_every_frame(tmp_path: Path) -> None:
    result = side_by_side(
        _clip_frames(),
        _drawing_frames(),
        15.0,
        "right",
        "left",
        False,
        True,
        tmp_path / "pair.mp4",
        tmp_path / "pair.gif",
    )

    assert result.mp4.is_file()
    assert result.gif.is_file()
    assert result.frames == FRAMES
    assert _frame_count(result.mp4) == FRAMES


def test_drawing_video_is_written_over_the_ground(tmp_path: Path) -> None:
    result = write_drawing(
        _drawing_frames(), 15.0, tmp_path / "drawing.mp4", tmp_path / "drawing.gif"
    )

    assert result.gif.is_file()
    assert _frame_count(result.mp4) == FRAMES


def test_credit_strip_is_present_by_default_and_absent_when_disabled() -> None:
    credited = _compose(credit=True)
    plain = _compose(credit=False)

    assert credited.shape[1] == FRAME_HEIGHT + STRIP_HEIGHT + CREDIT_HEIGHT
    assert plain.shape[1] == FRAME_HEIGHT + STRIP_HEIGHT
    credit_rows = credited[0, -CREDIT_HEIGHT:]
    assert np.any(credit_rows != 255)
    assert np.all(credit_rows[:, -100:] == 255)


def test_drawing_is_composited_on_white() -> None:
    transparent = np.zeros((FRAMES, *SIZE, 4), dtype=np.uint8)
    composed = compose_frames(
        _clip_frames(),
        transparent,
        facing_clip="right",
        facing_drawing="right",
        keep_facing=False,
        credit=False,
    )

    assert np.all(composed[0, :FRAME_HEIGHT, FRAME_HEIGHT:] == 255)


def test_contacts_add_a_strip_row_for_every_paw() -> None:
    contacts = np.zeros((FRAMES, 4), dtype=np.bool_)
    contacts[1, 2] = True
    drawing_contacts = np.zeros((FRAMES, 4), dtype=np.bool_)
    drawing_contacts[2, 0] = True

    composed = _compose(credit=False, contacts=contacts, drawing_contacts=drawing_contacts)

    assert composed.shape[1] == FRAME_HEIGHT + 2 * STRIP_HEIGHT
    strip = composed[0, FRAME_HEIGHT:]
    # The second frame spans columns 256 to 511 of the 1024 px pair.
    np.testing.assert_array_equal(strip[2 * ROW_HEIGHT + 2, 256], (64, 64, 64))


def test_footfall_record_lists_one_string_per_paw_with_the_pet_first(tmp_path: Path) -> None:
    contacts = np.zeros((FRAMES, 4), dtype=np.bool_)
    contacts[1, 2] = True
    drawing_contacts = np.zeros((FRAMES, 4), dtype=np.bool_)
    drawing_contacts[2, 0] = True

    write_footfall(tmp_path / "footfall.json", contacts, drawing_contacts, FRAMES)

    record = json.loads((tmp_path / "footfall.json").read_text(encoding="utf-8"))
    assert [len(track) for track in record["pet"] + record["drawing"]] == [FRAMES] * 8
    assert record["pet"][2][1] == "1"
    assert record["drawing"][0][2] == "1"
    assert record["pet"][0] == "0" * FRAMES
