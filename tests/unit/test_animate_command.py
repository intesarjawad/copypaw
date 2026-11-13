import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

import copypaw.cli as cli_module
from copypaw.animate.export import ExportResult
from copypaw.animate.render import RenderResult
from copypaw.cli import app
from copypaw.motion.motion_file import MotionFile

FRAMES = 3
SIZE = (8, 8)
JOINT_NAMES = (
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


def _joints() -> NDArray[np.float32]:
    points = np.zeros((FRAMES, len(JOINT_NAMES), 2), dtype=np.float32)
    points[:, 2] = [100, 0]
    for paw, name in enumerate(("left_foot_f", "right_foot_f", "left_foot_b", "right_foot_b")):
        points[:, JOINT_NAMES.index(name)] = [40 + paw * 20, 0]
    for paw, name in enumerate(("left_hip_f", "right_hip_f", "left_hip_b", "right_hip_b")):
        points[:, JOINT_NAMES.index(name)] = [40 + paw * 20, -40]
    return points


def _motion(path: Path) -> Path:
    kp = np.zeros((FRAMES, 17, 2), dtype=np.float32)
    MotionFile(
        kp=kp,
        scores=np.ones((FRAMES, 17), dtype=np.float32),
        fps=15.0,
        facing="right",
        contacts=np.zeros((FRAMES, 4), dtype=np.bool_),
        ground_line_px=np.array([0.0, 0.0]),
        travel_root_x=np.zeros(FRAMES, dtype=np.float32),
        source="clip.mp4",
    ).save(path)
    return path


def _run(tmp_path: Path) -> tuple[Path, Path, Path]:
    drawing = tmp_path / "drawing"
    drawing.mkdir()
    _motion(tmp_path / "motion.npz")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    out = tmp_path / "out"
    return drawing, out, clip


def test_unknown_mode_lists_the_choices(tmp_path: Path) -> None:
    drawing, out, clip = _run(tmp_path)
    motion = tmp_path / "motion.npz"

    result = CliRunner().invoke(
        app,
        [
            "animate",
            str(drawing),
            str(motion),
            "--clip",
            str(clip),
            "-o",
            str(out),
            "--mode",
            "sprint",
        ],
    )

    assert result.exit_code == 2
    assert "treadmill" in result.output
    assert "travel" in result.output


def test_animate_writes_every_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drawing, out, clip = _run(tmp_path)
    motion = tmp_path / "motion.npz"
    frames = np.zeros((FRAMES, *SIZE, 4), dtype=np.uint8)
    frames[..., 3] = 255

    def fake_render(
        _char_dir: Path, _captured: MotionFile, _mode: str, directory: Path
    ) -> RenderResult:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "render_joints.npz").write_bytes(b"joints")
        (directory / "baked.bvh").write_text("HIERARCHY\n", encoding="utf-8")
        (directory / "mvc.yaml").write_text("{}\n", encoding="utf-8")
        points = _joints()
        return RenderResult(
            frames_rgba=frames,
            fps=15.0,
            joints_per_frame=points,
            unfiltered_joints_per_frame=points,
            joint_names=JOINT_NAMES,
        )

    monkeypatch.setattr(cli_module, "render_character", fake_render)

    def fake_clip(_clip: Path) -> NDArray[np.uint8]:
        return np.zeros((FRAMES, *SIZE, 3), dtype=np.uint8)

    monkeypatch.setattr(cli_module, "load_clip_frames", fake_clip)
    captured: dict[str, object] = {}
    original = cli_module.side_by_side

    def recording_side_by_side(  # noqa: PLR0917
        clip_frames: NDArray[np.uint8],
        drawing_frames_rgba: NDArray[np.uint8],
        fps: float,
        facing_clip: Literal["left", "right"],
        facing_drawing: Literal["left", "right"],
        keep: bool,
        credit: bool,
        out_mp4: Path,
        out_gif: Path,
        *,
        contacts: NDArray[np.bool_] | None = None,
        drawing_contacts: NDArray[np.bool_] | None = None,
    ) -> ExportResult:
        captured["drawing_contacts"] = drawing_contacts
        return original(
            clip_frames,
            drawing_frames_rgba,
            fps,
            facing_clip,
            facing_drawing,
            keep,
            credit,
            out_mp4,
            out_gif,
            contacts=contacts,
            drawing_contacts=drawing_contacts,
        )

    monkeypatch.setattr(cli_module, "side_by_side", recording_side_by_side)

    result = CliRunner().invoke(
        app,
        ["animate", str(drawing), str(motion), "--clip", str(clip), "-o", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in out.iterdir()) == [
        "animation.gif",
        "animation.mp4",
        "fidelity.json",
        "footfall.json",
        "render_joints.npz",
        "side_by_side.gif",
        "side_by_side.mp4",
    ]
    written = json.loads((out / "fidelity.json").read_text(encoding="utf-8"))
    assert written["locomotion"] is False
    assert written["contact_f1"] == pytest.approx(1.0)
    assert written["gait_r"] is None
    drawing_contacts = cast("NDArray[np.bool_]", captured["drawing_contacts"])
    assert drawing_contacts.shape == (FRAMES, 4)
    assert "Wrote animation and side-by-side videos" in result.output
