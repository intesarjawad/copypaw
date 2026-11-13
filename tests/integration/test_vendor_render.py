import multiprocessing
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import GifImagePlugin, Image, ImageSequence

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/upstream"
RENDER_TIMEOUT_SECONDS = 180
MIN_OPAQUE_FRACTION = 0.05
BVH_FRAME_MS = 33


def render_four_legs(directory: str, use_mesa: bool) -> None:
    os.chdir(directory)
    if not use_mesa:
        glfw = import_module("glfw")
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize an OpenGL context")
        glfw.window_hint(glfw.VISIBLE, False)
    render = import_module("copypaw.vendor.animated_drawings.render")
    render.start(str(Path(directory) / "four_legs.yaml"))


@pytest.fixture
def four_legs_directory(tmp_path: Path) -> Path:
    (tmp_path / "examples").symlink_to(FIXTURES, target_is_directory=True)
    use_mesa = os.environ.get("PYOPENGL_PLATFORM") == "osmesa"
    mvc = (FIXTURES / "config/mvc/four_legs_example.yaml").read_text(encoding="utf-8")
    mvc += f"\nview:\n  USE_MESA: {str(use_mesa).lower()}\n"
    (tmp_path / "four_legs.yaml").write_text(mvc, encoding="utf-8")
    return tmp_path


def test_four_legs_deformation_without_context(
    four_legs_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(four_legs_directory)
    config = import_module("copypaw.vendor.animated_drawings.config")
    model = import_module("copypaw.vendor.animated_drawings.model.scene")
    cfg = config.Config(str(four_legs_directory / "four_legs.yaml"))
    scene = model.Scene(cfg.scene)
    drawing = scene.get_children()[0]
    initial_vertices = drawing.vertices.copy()
    scene.progress_time(100 * drawing.retargeter.bvh.frame_time)
    scene.update_transforms()
    assert np.isfinite(drawing.vertices).all()
    assert not np.array_equal(initial_vertices, drawing.vertices)


@pytest.mark.render
def test_four_legs_gif_frame_count_and_opacity(four_legs_directory: Path) -> None:
    use_mesa = os.environ.get("PYOPENGL_PLATFORM") == "osmesa"

    process = multiprocessing.get_context("spawn").Process(
        target=render_four_legs, args=(str(four_legs_directory), use_mesa)
    )
    process.start()
    try:
        process.join(RENDER_TIMEOUT_SECONDS)
        assert not process.is_alive(), "Four-legs render timed out"
        assert process.exitcode == 0, f"Render process exited with {process.exitcode}"
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()

    with (FIXTURES / "config/motion/zombie.yaml").open(encoding="utf-8") as stream:
        motion = yaml.safe_load(stream)
    expected_frames = motion["end_frame_idx"] - motion["start_frame_idx"]
    with Image.open(four_legs_directory / "video.gif") as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        # Pillow merges identical consecutive frames (the character walks out of view)
        # into one long frame, so count playback time rather than stored frames.
        frames = sum(
            round(frame.info["duration"] / BVH_FRAME_MS) for frame in ImageSequence.Iterator(gif)
        )
        assert frames == expected_frames
        gif.seek(100)
        alpha = np.asarray(gif.convert("RGBA"))[:, :, 3]
        assert np.count_nonzero(alpha == 255) / alpha.size >= MIN_OPAQUE_FRACTION
