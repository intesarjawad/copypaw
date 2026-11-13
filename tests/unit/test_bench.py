from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from copypaw.animate.render import RenderResult
from copypaw.capture.detect import Track
from copypaw.capture.pipeline import PoseStage
from copypaw.capture.pose import Keypoints
from copypaw.capture.video import Frames
from copypaw.models import Mode
from copypaw.motion.motion_file import MotionFile
from scripts import bench
from scripts.bench import END_MARKER, START_MARKER, Row, format_table, paste_table


def test_table_formats_values_and_missing_stages() -> None:
    table = format_table([Row("Detect, per frame", 12.345, "ms"), Row("Render", None, "s")])

    assert table.splitlines() == [
        "| Measurement | Result |",
        "| --- | --- |",
        "| Detect, per frame | 12.3 ms |",
        "| Render | n/a |",
    ]


def test_paste_replaces_only_the_marked_section() -> None:
    readme = f"before\n{START_MARKER}\nold\n{END_MARKER}\nafter\n"

    assert paste_table(readme, "new") == f"before\n{START_MARKER}\nnew\n{END_MARKER}\nafter\n"


def test_pose_timing_divides_the_pose_stage_by_the_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = Frames(
        rgb=np.zeros((4, 2, 2, 3), dtype=np.uint8),
        t_s=np.arange(4, dtype=np.float64),
        source_fps=30.0,
        fps=30.0,
        trimmed=False,
    )
    clock = iter([10.0, 10.5])

    def read(path: Path) -> Frames:
        del path
        return frames

    def pose(frames: Frames, track: Track) -> Keypoints:
        del frames, track
        return Keypoints(np.zeros((4, 17, 2), np.float32), np.zeros((4, 17), np.float32))

    def stages(mode: Mode, flip_test: bool, models: Mapping[str, Path]) -> tuple[Any, PoseStage]:
        del mode, flip_test, models
        return None, pose

    def track_all(frames: Frames, detector: Any) -> Any:
        del frames, detector

    monkeypatch.setattr(bench, "SAMPLE", Path(__file__))
    monkeypatch.setattr(bench, "read_frames", read)
    monkeypatch.setattr(bench, "load_stages", stages)
    monkeypatch.setattr(bench, "track_animal", track_all)
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))

    assert bench.pose_ms_per_frame(Mode.BALANCED, True, {}) == 125.0


def test_pose_timing_needs_the_sample_clip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench, "SAMPLE", tmp_path / "absent.mp4")

    with pytest.raises(FileNotFoundError, match=r"absent\.mp4"):
        bench.pose_ms_per_frame(Mode.BALANCED, False, {})


def _frames(count: int, rgba: tuple[int, int, int, int]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return np.full((count, 16, 16, 4), rgba, dtype=np.uint8)


def test_sheet_tiles_every_fifth_frame_over_a_footfall_strip() -> None:
    contacts = np.zeros((12, 4), dtype=np.bool_)
    contacts[:6, 0] = True

    sheet = bench.compose_sheet(_frames(12, (255, 0, 0, 255)), contacts)

    assert sheet.shape == (128 + 8 + 4 * 12, 6 * 128, 3)
    assert sheet[64, 64].tolist() == [255, 0, 0]
    assert sheet[64, 2 * 128 + 64].tolist() == [255, 0, 0]
    assert sheet[64, 3 * 128 + 64].tolist() == [255, 255, 255]
    assert sheet[140, 10].tolist() == [64, 64, 64]
    assert sheet[140, 8 * 64 + 10].tolist() == [242, 242, 242]


def test_sheet_flattens_transparent_pixels_onto_white() -> None:
    contacts = np.zeros((1, 4), dtype=np.bool_)

    sheet = bench.compose_sheet(_frames(1, (255, 0, 0, 128)), contacts)

    assert sheet[64, 64].tolist() == [255, 127, 127]


def test_sheet_needs_one_contact_row_per_frame() -> None:
    with pytest.raises(ValueError, match="3 contact rows for 2 frames"):
        bench.compose_sheet(_frames(2, (0, 0, 0, 255)), np.zeros((3, 4), dtype=np.bool_))


def _motion_folder(directory: Path, name: str) -> Path:
    folder = directory / name
    folder.mkdir(parents=True)
    MotionFile(
        kp=np.zeros((2, 17, 2), np.float32),
        scores=np.ones((2, 17), np.float32),
        fps=30.0,
        facing="right",
        contacts=np.zeros((2, 4), dtype=np.bool_),
        ground_line_px=np.zeros(2),
        travel_root_x=np.zeros(2, np.float32),
        source="clip.mp4",
    ).save(folder / "motion.npz")
    return folder


def _fake_render(monkeypatch: pytest.MonkeyPatch) -> None:
    def render(char_dir: Path, motion: MotionFile, mode: str, out_dir: Path) -> RenderResult:
        del char_dir, motion, out_dir
        assert mode == "treadmill"
        return RenderResult(
            frames_rgba=_frames(7, (0, 0, 255, 255)),
            fps=15.0,
            joints_per_frame=np.zeros((7, 1, 2), np.float32),
        )

    def resample(motion: MotionFile) -> SimpleNamespace:
        del motion
        return SimpleNamespace(contacts=np.zeros((7, 4), dtype=np.bool_))

    monkeypatch.setattr(bench, "render_character", render)
    monkeypatch.setattr(bench, "resample_motion", resample)
    monkeypatch.setattr(bench.time, "perf_counter", iter([1.0, 3.5]).__next__)


def test_contact_sheet_is_written_beside_the_motion_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _motion_folder(tmp_path / "motions", "dog_walk")
    drawing = tmp_path / "dragon"
    drawing.mkdir()
    (drawing / "char_cfg.yaml").touch()
    _fake_render(monkeypatch)

    seconds = bench.contact_sheet(folder, drawing, tmp_path / "review")

    assert seconds == 2.5
    with Image.open(tmp_path / "review/dog_walk.png") as sheet:
        assert sheet.size == (6 * 128, 128 + 8 + 4 * 12)


def test_contact_sheet_needs_a_rigged_drawing(tmp_path: Path) -> None:
    folder = _motion_folder(tmp_path / "motions", "dog_walk")

    with pytest.raises(FileNotFoundError, match="Missing rigged drawing"):
        bench.contact_sheet(folder, tmp_path / "dragon", tmp_path / "review")


def test_contact_sheets_need_a_motion(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No gallery motions"):
        bench.contact_sheets(tmp_path, tmp_path, tmp_path)


def test_gallery_render_time_waits_for_motions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bench, "MOTIONS", tmp_path / "motions")
    monkeypatch.setattr(bench, "DRAGON", tmp_path)

    assert bench.gallery_render_seconds() is None


def test_default_benchmark_inputs_are_the_packaged_sample_pair() -> None:
    assert bench.SAMPLE.is_file()
    assert bench.DRAWING.is_file()


def test_full_pipeline_timing_runs_capture_rig_and_animation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> None:
        assert kwargs["check"]
        calls.append(command)

    monkeypatch.setattr(bench.subprocess, "run", run)
    monkeypatch.setattr(bench.time, "perf_counter", iter([1.0, 3.0, 8.0, 17.0]).__next__)
    render_s, total_s = bench.pipeline_seconds(tmp_path)
    assert render_s == 5.0
    assert total_s == 16.0
    assert [command[1] for command in calls] == ["capture", "rig", "animate"]
    assert "--clip" in calls[-1]
