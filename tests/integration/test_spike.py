import multiprocessing
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, ImageDraw, ImageSequence

from copypaw.animate.spike import (
    FRAME_COUNT,
    FRAME_MS,
    FRAMES_PER_CYCLE,
    NEAR_Z_SIGN,
    SpikeFiles,
    write_spike,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CHARACTER = FIXTURES / "upstream/characters/char6/char_cfg.yaml"
RENDER_TIMEOUT_SECONDS = 180
MIN_OVERLAP_PIXELS = 16


def render_spike(mvc_path: str) -> None:
    if os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
        glfw = import_module("glfw")
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize an OpenGL context")
        glfw.window_hint(glfw.VISIBLE, False)
    render = import_module("copypaw.vendor.animated_drawings.render")
    render.start(mvc_path)


def run_render(files: SpikeFiles) -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=render_spike, args=(str(files.mvc),)
    )
    process.start()
    try:
        process.join(RENDER_TIMEOUT_SECONDS)
        assert not process.is_alive(), "Planar trot render timed out"
        assert process.exitcode == 0, f"Render process exited with {process.exitcode}"
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()


def playback_frames(path: Path) -> list[Image.Image]:
    with Image.open(path) as gif:
        # GIF encoders merge identical frames. Recover the motion timeline from duration.
        return [
            frame.convert("RGBA").copy()
            for frame in ImageSequence.Iterator(gif)
            for _ in range(round(frame.info["duration"] / FRAME_MS))
        ]


def contact_sheet(frames: list[Image.Image], path: Path) -> None:
    indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
    width, height = frames[0].size
    label_height = 20
    sheet = Image.new("RGB", (4 * width, 2 * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for cell, index in enumerate(indices):
        x = (cell % 4) * width
        y = (cell // 4) * (height + label_height)
        frame = frames[int(index)]
        sheet.paste(frame, (x, y), frame)
        draw.text((x + 4, y + height + 2), f"Frame {index}", fill="black")
    sheet.save(path)


@pytest.fixture
def spike_files(tmp_path: Path) -> SpikeFiles:
    return write_spike(
        tmp_path, CHARACTER, use_mesa=os.environ.get("PYOPENGL_PLATFORM") == "osmesa"
    )


def assert_facing(files: SpikeFiles) -> None:
    config = import_module("copypaw.vendor.animated_drawings.config")
    bvh_module = import_module("copypaw.vendor.animated_drawings.model.bvh")
    model = import_module("copypaw.vendor.animated_drawings.model.retargeter")
    motion = config.MotionConfig(str(files.motion))
    retarget = config.RetargetConfig(str(files.retarget))
    bvh = bvh_module.BVH.from_file(str(files.bvh))
    # Check before Retargeter automatically rotates the skeleton to face +X.
    for index in range(FRAME_COUNT):
        bvh.apply_frame(index)
        np.testing.assert_allclose(
            bvh.get_skeleton_fwd(motion.forward_perp_joint_vectors).vs,
            [[1, 0, 0]],
            atol=1e-6,
            rtol=0,
        )
    retargeter = model.Retargeter(motion, retarget)
    np.testing.assert_allclose(
        retargeter.fwd_vectors, np.tile([1, 0, 0], (FRAME_COUNT, 1)), atol=1e-6, rtol=0
    )
    for limb in ("F", "B"):
        near = retargeter.bvh_joint_to_projection_depth[f"Near_{limb}_Paw"]
        far = retargeter.bvh_joint_to_projection_depth[f"Far_{limb}_Paw"]
        np.testing.assert_allclose(near, NEAR_Z_SIGN * 0.05, atol=1e-6, rtol=0)
        np.testing.assert_allclose(far, -near, atol=1e-6, rtol=0)


def test_facing_without_context(spike_files: SpikeFiles) -> None:
    assert_facing(spike_files)


def test_configs_and_deformation_without_context(spike_files: SpikeFiles) -> None:
    config = import_module("copypaw.vendor.animated_drawings.config")
    model = import_module("copypaw.vendor.animated_drawings.model.scene")
    cfg = config.Config(str(spike_files.mvc))
    scene = model.Scene(cfg.scene)
    drawing = scene.get_children()[0]
    assert drawing.retargeter.bvh.frame_max_num == FRAME_COUNT
    assert drawing.retargeter.bvh.frame_time == FRAME_MS / 1000
    initial = drawing.vertices.copy()
    for _ in range(FRAME_COUNT):
        scene.progress_time(FRAME_MS / 1000)
        scene.update_transforms()
        assert np.isfinite(drawing.vertices).all()
        np.testing.assert_allclose(drawing.retargeter.bvh_root_positions[:, [0, 2]], 0, atol=1e-6)
    scene.progress_time(FRAMES_PER_CYCLE / 4 * FRAME_MS / 1000)
    scene.update_transforms()
    assert not np.allclose(initial, drawing.vertices)
    positions = drawing.retargeter.joint_positions.reshape(FRAME_COUNT, -1, 3)
    names = drawing.retargeter.bvh_joint_names
    for limb in ("F", "B"):
        near = positions[:, names.index(f"Near_{limb}_Paw"), :2]
        far = positions[:, names.index(f"Far_{limb}_Paw"), :2]
        np.testing.assert_allclose(near, np.roll(far, FRAMES_PER_CYCLE // 2, axis=0), atol=1e-6)


def test_playback_and_contact_sheet_with_merged_frames(tmp_path: Path) -> None:
    gif_path = tmp_path / "merged.gif"
    Image.new("RGBA", (32, 32), "red").save(gif_path, duration=FRAME_MS * FRAME_COUNT)
    frames = playback_frames(gif_path)
    assert len(frames) == FRAME_COUNT
    path = tmp_path / "sheet.png"
    contact_sheet(frames, path)
    with Image.open(path) as sheet:
        assert sheet.size == (128, 104)


@pytest.mark.render
def test_trot_facing_and_playback(
    spike_files: SpikeFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_facing(spike_files)
    run_render(spike_files)
    frames = playback_frames(spike_files.gif)
    path = spike_files.gif.with_name("trot_contact_sheet.png")
    contact_sheet(frames, path)
    with capsys.disabled():
        print(f"Trot GIF: {spike_files.gif}\nContact sheet: {path}")
    assert len(frames) == FRAME_COUNT
    for frame in frames:
        alpha = np.asarray(frame)[:, :, 3]
        assert np.count_nonzero(alpha == 255) / alpha.size >= 0.05
    assert any(not np.array_equal(np.asarray(frames[0]), np.asarray(f)) for f in frames[1:])


@pytest.mark.render
def test_retarget_tuple_yaml(spike_files: SpikeFiles) -> None:
    config = import_module("copypaw.vendor.animated_drawings.config")
    retarget = config.RetargetConfig(str(spike_files.retarget))
    mapping: dict[str, tuple[str, str]] = retarget.char_joint_bvh_joints_mapping
    assert len(mapping) == 9
    assert all(type(pair) is tuple and len(pair) == 2 for pair in mapping.values())
    assert mapping["left_foot_f"] == ("Near_Elbow", "Near_F_Paw")
    assert mapping["right_foot_b"] == ("Far_Knee", "Far_B_Paw")


@pytest.fixture
def depth_character(tmp_path: Path) -> Path:
    config = import_module("copypaw.vendor.animated_drawings.config")
    directory = tmp_path / "depth_character"
    directory.mkdir()
    path = directory / "char_cfg.yaml"
    path.write_bytes((FIXTURES / "spike/depth_character.yaml").read_bytes())
    # CharacterConfig validates these paths before returning the normalized joints.
    texture = Image.new("RGBA", (256, 256))
    texture.save(directory / "texture.png")
    texture.getchannel("A").save(directory / "mask.png")
    char = config.CharacterConfig(str(path))
    joints = {joint["name"]: joint for joint in char.skeleton}
    draw = ImageDraw.Draw(texture)
    for joint in char.skeleton:
        name = joint["name"]
        if "knee" not in name and "foot" not in name:
            continue
        parent = joints[joint["parent"]]
        points = [tuple(round(v * 256) for v in j["loc"]) for j in (parent, joint)]
        draw.line(points, fill="red" if name.startswith("left_") else "blue", width=12)
    draw.rectangle((70, 60, 202, 100), fill="gray")
    draw.line((190, 80, 220, 48), fill="gray", width=18)
    texture.save(directory / "texture.png")
    texture.getchannel("A").save(directory / "mask.png")
    return path


def test_depth_fixture_without_context(tmp_path: Path, depth_character: Path) -> None:
    config = import_module("copypaw.vendor.animated_drawings.config")
    model = import_module("copypaw.vendor.animated_drawings.model.scene")
    vertices: list[npt.NDArray[np.float32]] = []
    for sign in (NEAR_Z_SIGN, -NEAR_Z_SIGN):
        files = write_spike(tmp_path / str(sign), depth_character, near_z_sign=sign)
        scene = model.Scene(config.Config(str(files.mvc)).scene)
        drawing = scene.get_children()[0]
        assert np.isfinite(drawing.vertices).all()
        vertices.append(drawing.vertices[:, :2].copy())
        names = drawing.rig.root_joint.get_chain_joint_names()
        positions = drawing.rig.get_joints_2D_positions()
        for limb in ("f", "b"):
            for joint in ("knee", "foot"):
                near = f"left_{joint}_{limb}"
                far = f"right_{joint}_{limb}"
                np.testing.assert_allclose(
                    positions[names.index(near)], positions[names.index(far)], atol=1e-6
                )
                assert drawing.joint_to_tri_v_idx[near].size > 0
                assert drawing.joint_to_tri_v_idx[far].size > 0
    # Changing only the depths must preserve the projection used by the overlap comparison.
    np.testing.assert_allclose(vertices[0], vertices[1], atol=1e-6)


@pytest.mark.render
def test_near_limbs_occlude_far_limbs(
    tmp_path: Path, depth_character: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_by_sign: dict[int, list[Image.Image]] = {}
    for sign in (NEAR_Z_SIGN, -NEAR_Z_SIGN):
        files = write_spike(
            tmp_path / f"depth_{sign}",
            depth_character,
            near_z_sign=sign,
            use_mesa=os.environ.get("PYOPENGL_PLATFORM") == "osmesa",
        )
        run_render(files)
        frames_by_sign[sign] = playback_frames(files.gif)
        path = files.gif.with_name("depth_contact_sheet.png")
        contact_sheet(frames_by_sign[sign], path)
        with capsys.disabled():
            print(f"Depth sign {sign} contact sheet: {path}")
    positive = frames_by_sign[NEAR_Z_SIGN]
    negative = frames_by_sign[-NEAR_Z_SIGN]
    assert len(positive) == len(negative) == FRAME_COUNT
    red_wins = blue_wins = 0
    for near_frame, far_frame in zip(positive, negative, strict=True):
        near = np.asarray(near_frame).astype(np.int16)
        far = np.asarray(far_frame).astype(np.int16)
        near_red = (near[:, :, 0] > near[:, :, 2] + 128) & (near[:, :, 3] == 255)
        near_blue = (near[:, :, 2] > near[:, :, 0] + 128) & (near[:, :, 3] == 255)
        far_red = (far[:, :, 0] > far[:, :, 2] + 128) & (far[:, :, 3] == 255)
        far_blue = (far[:, :, 2] > far[:, :, 0] + 128) & (far[:, :, 3] == 255)
        red_wins += np.count_nonzero(near_red & far_blue)
        blue_wins += np.count_nonzero(near_blue & far_red)
    assert red_wins >= MIN_OVERLAP_PIXELS, "No red-over-blue overlap when near depth is positive"
    assert blue_wins == 0, f"Blue won {blue_wins} overlap pixels with positive near depth"
