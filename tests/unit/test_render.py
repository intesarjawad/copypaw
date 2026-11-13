import math
import shutil
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from scipy.spatial import Delaunay

import copypaw.animate.render as render_module
from copypaw.animate.render import (
    CharacterError,
    RenderResult,
    frame_camera,
    render_character,
    resample_motion,
    run_engine,
    travel_canvas,
)
from copypaw.motion.motion_file import MotionFile

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CHARACTER = FIXTURES / "upstream/characters/char6"
FRAME_SIZE = (4, 5)
KEYPOINTS = np.array(
    [
        [110, -110],
        [110, -90],
        [120, -100],
        [100, -100],
        [100, 0],
        [0, -100],
        [0, 0],
        [0, 100],
        [-50, -100],
        [-50, -20],
        [-50, 40],
        [200, -100],
        [200, 0],
        [200, 120],
        [250, -100],
        [250, -30],
        [250, 40],
    ],
    dtype=np.float32,
)


def _motion(frames: int = 3, fps: float = 10.0) -> MotionFile:
    kp = np.tile(KEYPOINTS, (frames, 1, 1))
    kp[:, :, 0] += np.arange(frames, dtype=np.float32)[:, None]
    kp[:, 0, 0] = np.arange(frames, dtype=np.float32)
    contacts = np.zeros((frames, 4), dtype=np.bool_)
    contacts[::2] = True
    return MotionFile(
        kp=kp,
        scores=np.linspace(0, 1, frames, dtype=np.float32)[:, None].repeat(17, axis=1),
        fps=fps,
        facing="right",
        contacts=contacts,
        ground_line_px=np.array([0.0, 120.0]),
        travel_root_x=np.linspace(0, 1, frames, dtype=np.float32),
        source="clip.mp4",
    )


def _character(directory: Path, *, mirrored: bool = False) -> Path:
    drawing = directory / "drawing"
    shutil.copytree(CHARACTER, drawing)
    manifest = "legs: 4\n" + ("mirrored: true\n" if mirrored else "")
    (drawing / "copypaw.yaml").write_text(manifest, encoding="utf-8")
    return drawing


def _fake_engine(mvc: Path, joints_path: Path) -> None:
    config = yaml.safe_load(mvc.read_text(encoding="utf-8"))
    gif = Path(config["controller"]["OUTPUT_VIDEO_PATH"])
    motion = yaml.safe_load((mvc.parent / "motion.yaml").read_text(encoding="utf-8"))
    count = int(motion["end_frame_idx"])

    frames: list[Image.Image] = []
    for index in range(count):
        pixels = np.zeros((*FRAME_SIZE, 4), dtype=np.uint8)
        pixels[:, index % FRAME_SIZE[1], 0] = 255
        pixels[..., 3] = 255
        frames.append(Image.fromarray(pixels, "RGBA"))
    if len(frames) > 1:
        frames[1] = frames[0].copy()
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / 15),
        disposal=2,
        loop=0,
    )

    joints = np.zeros((count, 3, 2), dtype=np.float32)
    np.savez_compressed(joints_path, joints=joints)


def _render(
    tmp_path: Path, mode: str = "treadmill", *, mirrored: bool = False
) -> tuple[RenderResult, Path]:
    character = _character(tmp_path, mirrored=mirrored)
    out = tmp_path / "out"
    result = render_character(character, _motion(), mode, out)
    return result, out


def test_keypoints_interpolate_and_contacts_take_the_nearest_frame() -> None:
    resampled = resample_motion(_motion())

    assert resampled.fps == 15.0
    assert len(resampled.kp) == 4
    assert resampled.kp[:, 0, 0] == pytest.approx([0, 2 / 3, 4 / 3, 2], abs=1e-5)
    assert resampled.contacts.tolist() == [
        [True, True, True, True],
        [False, False, False, False],
        [False, False, False, False],
        [True, True, True, True],
    ]
    assert resampled.travel_root_x == pytest.approx([0, 1 / 3, 2 / 3, 1], abs=1e-6)


def test_thirty_fps_five_seconds_becomes_seventy_five_output_frames() -> None:
    motion = _motion(frames=150, fps=30.0)

    resampled = resample_motion(motion)

    assert len(resampled.kp) == 75
    # Output frame 2 is at 2/15 s, exactly analysis frame 4.
    assert resampled.kp[2, 0, 0] == pytest.approx(motion.kp[4, 0, 0])
    # The final analysis timestamp is 149/30 s, so the last output frame is 148.
    assert resampled.kp[-1, 0, 0] == pytest.approx(motion.kp[148, 0, 0])


@pytest.mark.parametrize("fps", [0, -1])
def test_invalid_motion_and_output_rates_are_rejected(fps: int) -> None:
    motion = _motion()
    with pytest.raises(ValueError, match="fps"):
        resample_motion(motion, fps)
    with pytest.raises(ValueError, match="fps"):
        resample_motion(
            MotionFile(
                kp=motion.kp,
                scores=motion.scores,
                fps=float(fps),
                facing=motion.facing,
                contacts=motion.contacts,
                ground_line_px=motion.ground_line_px,
                travel_root_x=motion.travel_root_x,
                source=motion.source,
            )
        )


def test_rendered_frames_expand_merged_gif_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(render_module, "run_engine", _fake_engine)

    result, _ = _render(tmp_path)

    assert result.fps == 15.0
    assert result.frames_rgba.shape == (4, *FRAME_SIZE, 4)
    assert result.frames_rgba.dtype == np.uint8
    assert np.argmax(result.frames_rgba[0, 0, :, 0]) == 0
    assert np.argmax(result.frames_rgba[1, 0, :, 0]) == 0
    assert np.argmax(result.frames_rgba[2, 0, :, 0]) == 2
    assert np.argmax(result.frames_rgba[3, 0, :, 0]) == 3


def test_mirrored_drawing_is_flipped_back_with_its_joints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(render_module, "run_engine", _fake_engine)

    result, _ = _render(tmp_path, mirrored=True)

    assert np.argmax(result.frames_rgba[3, 0, :, 0]) == 1
    np.testing.assert_allclose(result.joints_per_frame[:, :, 0], 1, atol=0)


def test_treadmill_removes_root_travel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module, "run_engine", _fake_engine)

    _, out = _render(tmp_path, "treadmill")

    bvh = import_module("copypaw.vendor.animated_drawings.model.bvh").BVH.from_file(
        str(out / "baked.bvh")
    )
    np.testing.assert_allclose(bvh.pos_data[:, 0], 0, atol=0)


def test_travel_mode_keeps_the_resampled_root_travel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(render_module, "run_engine", _fake_engine)

    _, out = _render(tmp_path, "travel")

    bvh = import_module("copypaw.vendor.animated_drawings.model.bvh").BVH.from_file(
        str(out / "baked.bvh")
    )
    np.testing.assert_allclose(bvh.pos_data[:, 0], [0, 1 / 3, 2 / 3, 1], atol=1e-6)


@pytest.mark.parametrize("mode", ["treadmill", "travel"])
def test_every_mode_hands_the_contacts_and_mode_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(render_module, "run_engine", _fake_engine)

    _, out = _render(tmp_path, mode)

    with np.load(out / "render_contacts.npz", allow_pickle=False) as archive:
        assert str(archive["mode"]) == mode
        np.testing.assert_array_equal(archive["contacts"], resample_motion(_motion()).contacts)


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        render_character(_character(tmp_path), _motion(), "sprint", tmp_path / "out")


@pytest.mark.parametrize("missing", ["char_cfg.yaml", "texture.png", "mask.png"])
def test_a_folder_without_a_rigged_drawing_names_the_missing_file(
    tmp_path: Path, missing: str
) -> None:
    character = _character(tmp_path)
    (character / missing).unlink()

    with pytest.raises(CharacterError) as error:
        render_character(character, _motion(), "treadmill", tmp_path / "out")

    assert missing in error.value.what
    assert "copypaw rig" in error.value.next


def _travel_character(directory: Path) -> Path:
    """A 100 x 50 drawing whose root sits at x 40 and box at x 10..89."""
    drawing = directory / "travel"
    drawing.mkdir()
    skeleton = [
        {"name": "root", "loc": [40, 25], "parent": None},
        {"name": "left_hip_f", "loc": [70, 25], "parent": "root"},
        {"name": "left_hip_b", "loc": [10, 25], "parent": "root"},
    ]
    (drawing / "char_cfg.yaml").write_text(
        yaml.safe_dump({"height": 50, "width": 100, "skeleton": skeleton}, sort_keys=False),
        encoding="utf-8",
    )
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[:, 10:90] = 255
    Image.fromarray(mask, "L").save(drawing / "mask.png")
    return drawing


def _travel(frames: list[float]) -> MotionFile:
    motion = _motion()
    count = len(frames)
    return MotionFile(
        kp=np.tile(motion.kp[0], (count, 1, 1)).astype(np.float32),
        scores=np.ones((count, 17), dtype=np.float32),
        fps=10.0,
        facing="right",
        contacts=np.zeros((count, 4), dtype=np.bool_),
        ground_line_px=motion.ground_line_px,
        travel_root_x=np.asarray(frames, dtype=np.float32),
        source=motion.source,
    )


def test_travel_canvas_is_box_plus_path_plus_ten_percent_margins(tmp_path: Path) -> None:
    canvas = travel_canvas(_travel_character(tmp_path), _travel([0.0, 0.25, 0.5, 0.75, 1.0]))

    # The camera fits the 0.5 unit tall drawing, plus 5% above and below, to
    # height 512: 930.909 px per unit. Box 0.8 units and 0.6 units of path, plus 20%.
    assert (canvas.width, canvas.height, canvas.frames) == (1564, 512, 5)
    # Half the canvas, plus one margin, minus the box's offset from the root.
    assert canvas.start_x == pytest.approx(-1564 / (2 * 512 / 0.55) + 0.14 + 0.3)


def test_travel_canvas_keeps_the_longest_prefix_when_four_times_too_wide(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    canvas = travel_canvas(_travel_character(tmp_path), _travel(list(map(float, range(9)))))

    # Frame 2 spans one body length and fits under 2048; frame 3 would not.
    assert (canvas.width, canvas.frames) == (1564, 2)
    warning = capsys.readouterr().err
    assert "first 2 of 9 frames" in warning
    assert "--mode treadmill" in warning


def test_travel_canvas_without_travel_has_no_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    canvas = travel_canvas(_travel_character(tmp_path), _travel([0.0, 0.0, 0.0]))

    assert (canvas.width, canvas.frames) == (894, 3)
    assert capsys.readouterr().err == ""


def test_motion_config_end_index_covers_every_rendered_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(render_module, "run_engine", _fake_engine)

    _, out = _render(tmp_path)

    config = yaml.safe_load((out / "motion.yaml").read_text(encoding="utf-8"))
    assert config["end_frame_idx"] == 4
    assert config["forward_perp_joint_vectors"] == [["Fwd_Far", "Fwd_Near"]]
    assert "- Root" in (out / "retarget.yaml").read_text(encoding="utf-8")


class _Process:
    exitcode: int | None = 0
    timeout: float | None = None
    started = False
    terminated = False
    closed = False

    def __init__(self, target: object, args: tuple[Path, Path], *, alive: bool = False) -> None:
        self.target = target
        self.args = args
        self.alive = alive

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self.timeout = timeout

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True

    def close(self) -> None:
        self.closed = True


class _Context:
    def __init__(self) -> None:
        self.process: _Process | None = None
        self.next_process: _Process | None = None
        self.Process = self._create

    def _create(self, target: object, args: tuple[Path, Path]) -> _Process:
        self.process = self.next_process or _Process(target, args)
        return self.process


def test_engine_runs_in_a_spawned_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _Context()
    names: list[str] = []

    def get_context(name: str) -> _Context:
        names.append(name)
        return context

    monkeypatch.setattr(render_module.multiprocessing, "get_context", get_context)

    run_engine(tmp_path / "mvc.yaml", tmp_path / "joints.npz", timeout_s=12)

    assert names == ["spawn"]
    assert context.process is not None
    assert context.process.started
    assert context.process.args == (tmp_path / "mvc.yaml", tmp_path / "joints.npz")
    assert context.process.timeout == 12
    assert context.process.closed


def test_engine_failure_and_timeout_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _Context()

    def get_context(_name: str) -> _Context:
        return context

    monkeypatch.setattr(render_module.multiprocessing, "get_context", get_context)
    failing = _Process(lambda: None, (tmp_path / "a", tmp_path / "b"))
    failing.exitcode = 3
    context.next_process = failing
    with pytest.raises(RuntimeError, match="exited with 3"):
        run_engine(tmp_path / "mvc.yaml", tmp_path / "joints.npz")

    hanging = _Process(lambda: None, (tmp_path / "a", tmp_path / "b"), alive=True)
    context.next_process = hanging
    with pytest.raises(RuntimeError, match="timed out"):
        run_engine(tmp_path / "mvc.yaml", tmp_path / "joints.npz", timeout_s=0.01)
    assert hanging.terminated
    assert hanging.closed


def test_drawings_recorded_tail_reaches_the_engine_measured_up_from_the_foot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(render_module, "run_engine", _fake_engine)
    drawing = _character(tmp_path)
    with (drawing / "copypaw.yaml").open("a", encoding="utf-8") as manifest:
        manifest.write("tails:\n- - [10.0, 30.0]\n  - [2.0, 50.0]\n")
    height = yaml.safe_load((drawing / "char_cfg.yaml").read_text(encoding="utf-8"))["height"]

    render_character(drawing, _motion(), "treadmill", tmp_path / "out")

    with np.load(tmp_path / "out" / "render_contacts.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(
            archive["tails"], [[(10, height - 30), (2, height - 50)]], atol=1e-6
        )


@pytest.mark.parametrize(
    ("low", "high", "window"),
    [((0.0, 0.0), (2.0, 1.0), (512, 512)), ((-0.2, 0.1), (0.3, 1.4), (800, 400))],
)
def test_camera_shows_the_whole_reach_with_a_margin_and_fills_one_side(
    low: tuple[float, float], high: tuple[float, float], window: tuple[int, int]
) -> None:
    x, y, z = frame_camera(np.array(low), np.array(high), window)

    # The drawing stands at depth -0.5 and the view opens 35 degrees top to bottom.
    half_height = (z + 0.5) * math.tan(math.radians(35 / 2))
    half_width = half_height * window[0] / window[1]
    fill = ((high[0] - low[0]) / (2 * half_width), (high[1] - low[1]) / (2 * half_height))
    assert (x, y) == pytest.approx(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2))
    assert max(fill) == pytest.approx(1 / 1.1)
    assert min(fill) <= 1 / 1.1


def test_camera_rejects_a_drawing_with_no_reach() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        frame_camera(np.array((0.0, 0.0)), np.array((0.0, 1.0)), (512, 512))


def test_large_mesh_solver_does_not_overflow_its_singularity_check() -> None:
    arap = import_module("copypaw.vendor.animated_drawings.model.arap")
    x, y = np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 1, 12))
    vertices = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    triangles = [row.astype(np.int32) for row in Delaunay(vertices).simplices]
    pins = vertices[::4]
    with np.errstate(divide="raise", invalid="raise"):
        solver = arap.ARAP(pins, triangles, vertices, w=100)
        posed = solver.solve(pins)
    assert np.isfinite(posed).all()
    np.testing.assert_allclose(posed, vertices, atol=1e-3)
