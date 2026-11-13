import shutil
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

import copypaw.animate.configs as configs_module
from copypaw.animate.configs import AnimationConfigs, write_configs
from copypaw.motion.bvh_writer import write_bvh
from copypaw.motion.skeleton import BONES, HELPER_JOINTS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CHARACTER = FIXTURES / "upstream/characters/char6"
FRAME_COUNT = 5

config = import_module("copypaw.vendor.animated_drawings.config")
bvh_module = import_module("copypaw.vendor.animated_drawings.model.bvh")

SKELETON_NAMES = {bone.name for bone in BONES} | {name for name, _ in HELPER_JOINTS}
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
EXPECTED_BODY_GROUPS = [
    {"bvh_depth_drivers": ["Root"], "char_joints": ["left_knee_f", "left_foot_f"]},
    {"bvh_depth_drivers": ["Root"], "char_joints": ["left_knee_b", "left_foot_b"]},
    {
        "bvh_depth_drivers": ["Root"],
        "char_joints": [
            "right_hip_b",
            "left_hip_b",
            "right_hip_f",
            "left_hip_f",
            "hip",
            "torso",
            "neck",
        ],
    },
    {"bvh_depth_drivers": ["Far_F_Paw"], "char_joints": ["right_knee_f", "right_foot_f"]},
    {"bvh_depth_drivers": ["Far_B_Paw"], "char_joints": ["right_knee_b", "right_foot_b"]},
]
EXPECTED_MAPPING = {
    "neck": ("Neck", "Nose"),
    "left_knee_f": ("Near_Shoulder", "Near_Elbow"),
    "left_foot_f": ("Near_Elbow", "Near_F_Paw"),
    "left_knee_b": ("Near_Hip", "Near_Knee"),
    "left_foot_b": ("Near_Knee", "Near_B_Paw"),
    "right_knee_f": ("Far_Shoulder", "Far_Elbow"),
    "right_foot_f": ("Far_Elbow", "Far_F_Paw"),
    "right_knee_b": ("Far_Hip", "Far_Knee"),
    "right_foot_b": ("Far_Knee", "Far_B_Paw"),
}


def _write_bvh(path: Path) -> Path:
    kp = np.tile(KEYPOINTS, (FRAME_COUNT, 1, 1))
    write_bvh(kp, 30.0, np.zeros(FRAME_COUNT, dtype=np.float32), out=path)
    return path


def _drawing(directory: Path, legs: int) -> Path:
    drawing = directory / "drawing"
    shutil.copytree(CHARACTER, drawing)
    (drawing / "copypaw.yaml").write_text(f"legs: {legs}\n", encoding="utf-8")
    return drawing


def _write(
    directory: Path,
    legs: int = 4,
    *,
    use_mesa: bool | None = False,
    output: Path | None = None,
) -> AnimationConfigs:
    bvh = _write_bvh(directory / "baked.bvh")
    drawing = _drawing(directory, legs)
    return write_configs(bvh, drawing, directory / "out", use_mesa=use_mesa, output=output)


def test_configs_load_through_the_animation_engine(tmp_path: Path) -> None:
    files = _write(tmp_path)

    motion = config.MotionConfig(str(files.motion))
    retarget = config.RetargetConfig(str(files.retarget))
    mvc = config.Config(str(files.mvc))

    assert motion.start_frame_idx == 0
    assert motion.end_frame_idx == FRAME_COUNT - 1
    assert motion.groundplane_joint == "Near_B_Paw"
    assert motion.forward_perp_joint_vectors == [["Fwd_Near", "Fwd_Far"]]
    assert motion.scale == 1.0
    assert motion.up == "+y"
    assert motion.bvh_p == tmp_path / "baked.bvh"

    assert retarget.char_start_loc == [0, 0, -0.5]
    assert len(retarget.bvh_projection_bodypart_groups) == 1
    group = retarget.bvh_projection_bodypart_groups[0]
    assert group["name"] == "all"
    assert group["method"] == "sagittal"
    assert set(group["bvh_joint_names"]) == SKELETON_NAMES
    assert retarget.char_bodypart_groups == EXPECTED_BODY_GROUPS
    assert retarget.char_bvh_root_offset == {
        "bvh_projection_bodypart_group_for_offset": "all",
        "bvh_joints": [
            ["Near_F_Paw", "Near_Elbow", "Near_Shoulder"],
            ["Far_F_Paw", "Far_Elbow", "Far_Shoulder"],
            ["Near_B_Paw", "Near_Knee", "Near_Hip"],
            ["Far_B_Paw", "Far_Knee", "Far_Hip"],
        ],
        "char_joints": [
            ["left_foot_f", "left_knee_f", "left_hip_f"],
            ["right_foot_f", "right_knee_f", "right_hip_f"],
            ["left_foot_b", "left_knee_b", "left_hip_b"],
            ["right_foot_b", "right_knee_b", "right_hip_b"],
        ],
    }
    assert retarget.char_joint_bvh_joints_mapping == EXPECTED_MAPPING
    assert retarget.char_runtime_checks == []

    assert len(mvc.scene.animated_characters) == 1
    assert mvc.controller.mode == "video_render"
    assert mvc.controller.output_video_path == str(files.gif)
    assert mvc.view.use_mesa is False
    assert mvc.view.window_dimensions == [512, 512]
    assert mvc.view.clear_color == [1.0, 1.0, 1.0, 0.0]


def test_retarget_names_exist_in_the_bvh_and_skeleton(tmp_path: Path) -> None:
    files = _write(tmp_path)
    retarget = config.RetargetConfig(str(files.retarget))
    bvh_names = set(bvh_module.BVH.from_file(str(tmp_path / "baked.bvh")).get_joint_names())

    referenced = {
        *(retarget.bvh_projection_bodypart_groups[0]["bvh_joint_names"]),
        *(
            driver
            for group in retarget.char_bodypart_groups
            for driver in group["bvh_depth_drivers"]
        ),
        *(name for chain in retarget.char_bvh_root_offset["bvh_joints"] for name in chain),
        *(name for pair in retarget.char_joint_bvh_joints_mapping.values() for name in pair),
    }
    assert referenced <= bvh_names
    assert referenced - {"Root"} <= SKELETON_NAMES


def test_two_leg_drawing_moves_both_chains_with_near_joints(tmp_path: Path) -> None:
    files = _write(tmp_path, legs=2)

    retarget = config.RetargetConfig(str(files.retarget))
    mapping = retarget.char_joint_bvh_joints_mapping
    pairs = [name for pair in mapping.values() for name in pair]

    assert not any(name.startswith("Far_") for name in pairs)
    assert mapping["right_knee_f"] == mapping["left_knee_f"]
    assert mapping["right_foot_f"] == mapping["left_foot_f"]
    assert mapping["right_knee_b"] == mapping["left_knee_b"]
    assert mapping["right_foot_b"] == mapping["left_foot_b"]
    drivers = [
        name for group in retarget.char_bodypart_groups for name in group["bvh_depth_drivers"]
    ]
    assert all(name.startswith("Near_") for name in drivers if name != "Root")
    root_legs = retarget.char_bvh_root_offset["bvh_joints"]
    assert len(root_legs) == len(retarget.char_bvh_root_offset["char_joints"])
    assert all(name.startswith("Near_") for chain in root_legs for name in chain)


def test_mesa_follows_the_doctor_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(configs_module, "render_backend", lambda: "osmesa")

    files = _write(tmp_path, use_mesa=None)

    assert config.Config(str(files.mvc)).view.use_mesa is True


def test_mp4_output_names_the_final_video(tmp_path: Path) -> None:
    files = _write(tmp_path, output=tmp_path / "final.mp4")

    mvc = config.Config(str(files.mvc))

    assert mvc.controller.output_video_path == str(tmp_path / "final.mp4")
    assert files.gif == tmp_path / "final.gif"
    assert files.mp4 == tmp_path / "final.mp4"


def test_travel_canvas_dimensions_and_starting_offset_are_written(tmp_path: Path) -> None:
    bvh = _write_bvh(tmp_path / "baked.bvh")
    drawing = _drawing(tmp_path, 4)

    files = write_configs(
        bvh,
        drawing,
        tmp_path / "out",
        use_mesa=False,
        window_dimensions=(900, 512),
        char_start_x=-0.25,
    )

    mvc = config.Config(str(files.mvc))
    retarget = config.RetargetConfig(str(files.retarget))
    assert mvc.view.window_dimensions == [900, 512]
    assert retarget.char_start_loc == [-0.25, 0, -0.5]
