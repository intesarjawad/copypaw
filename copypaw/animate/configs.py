"""Write the motion, retarget and view configs that drive one drawing."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from copypaw.doctor import render_backend
from copypaw.motion.skeleton import BONES, HELPER_JOINTS

__all__ = ["AnimationConfigs", "write_configs"]

_WINDOW_DIMENSIONS = (512, 512)
_CHARACTER_CFG = "char_cfg.yaml"
# Drawings may carry the manifest under either the current or the earlier name.
_MANIFEST_NAME = "copypaw.yaml"
_ANIMATION_GIF = "animation.gif"
_TWO_LEGS = 2
_FOUR_LEGS = 4
_ROOT_JOINT = "Root"
_SKELETON_JOINTS = tuple(bone.name for bone in BONES) + tuple(name for name, _ in HELPER_JOINTS)


@dataclass(frozen=True)
class AnimationConfigs:
    """Paths to the three generated configs and the videos they name."""

    motion: Path
    retarget: Path
    mvc: Path
    gif: Path
    mp4: Path


def _frame_count(bvh: Path) -> int:
    for line in bvh.read_text(encoding="utf-8").splitlines():
        if line.startswith("Frames:"):
            count = int(line.partition(":")[2])
            if count < 1:
                raise ValueError(f"BVH contains no frames: {bvh}")
            return count
    raise ValueError(f"BVH has no Frames header: {bvh}")


def _legs(character: Path, legs: int | None) -> int:
    """Use an explicit leg count, else the drawing manifest, else four legs."""
    if legs is None:
        legs = _FOUR_LEGS
        manifest = character / _MANIFEST_NAME
        if manifest.is_file():
            data = cast(
                "dict[str, object]",
                yaml.safe_load(manifest.read_text(encoding="utf-8")) or {},
            )
            legs = cast("int", data.get("legs", _FOUR_LEGS))
    if legs not in (_TWO_LEGS, _FOUR_LEGS):
        raise ValueError(f"legs must be {_TWO_LEGS} or {_FOUR_LEGS}, got {legs!r}")
    return legs


def _bodypart_groups(legs: int) -> list[dict[str, object]]:
    # A leg in front of the body tucks under it: drawn first at the body's own depth, which the
    # engine's stable sort keeps ahead of it, so the hip shows no seam. At that depth the engine
    # draws the foot before the knee, so the upper bone covers the knee as on a far leg.
    tucked: list[dict[str, object]] = [
        {
            "bvh_depth_drivers": [_ROOT_JOINT],
            "char_joints": [f"{prefix}_knee_{limb}", f"{prefix}_foot_{limb}"],
        }
        for prefix, near in (("left", True), ("right", False))
        for limb in ("f", "b")
        if near or legs == _TWO_LEGS
    ]
    groups: list[dict[str, object]] = [
        *tucked,
        {
            "bvh_depth_drivers": [_ROOT_JOINT],
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
    ]
    if legs == _FOUR_LEGS:
        groups.extend(
            {
                "bvh_depth_drivers": [f"Far_{limb.upper()}_Paw"],
                "char_joints": [f"right_knee_{limb}", f"right_foot_{limb}"],
            }
            for limb in ("f", "b")
        )
    return groups


def _mapping(legs: int) -> dict[str, tuple[str, str]]:
    far = "Near" if legs == _TWO_LEGS else "Far"
    return {
        "neck": ("Neck", "Nose"),
        "left_knee_f": ("Near_Shoulder", "Near_Elbow"),
        "left_foot_f": ("Near_Elbow", "Near_F_Paw"),
        "left_knee_b": ("Near_Hip", "Near_Knee"),
        "left_foot_b": ("Near_Knee", "Near_B_Paw"),
        "right_knee_f": (f"{far}_Shoulder", f"{far}_Elbow"),
        "right_foot_f": (f"{far}_Elbow", f"{far}_F_Paw"),
        "right_knee_b": (f"{far}_Hip", f"{far}_Knee"),
        "right_foot_b": (f"{far}_Knee", f"{far}_B_Paw"),
    }


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.dump(payload, sort_keys=False), encoding="utf-8")


def write_configs(
    bvh: Path,
    character: Path,
    directory: Path,
    *,
    legs: int | None = None,
    use_mesa: bool | None = None,
    output: Path | None = None,
    window_dimensions: tuple[int, int] = _WINDOW_DIMENSIONS,
    char_start_x: float = 0,
) -> AnimationConfigs:
    """Write motion, retarget and MVC YAML for a baked BVH beside a drawing.

    character is the drawing directory or its char_cfg.yaml. The manifest
    controls whether the far leg chains drive the same near joints; asking the
    doctor for the render backend avoids claiming mesa where it is unavailable.
    output defaults to a GIF in directory; its sibling MP4 names the final video.
    window_dimensions widens the canvas and char_start_x slides the drawing to
    the left edge, which travel mode needs so its whole path stays in frame.
    """
    drawing = character if character.is_dir() else character.parent
    character_cfg = (character if character.is_file() else drawing / _CHARACTER_CFG).resolve()
    bvh = bvh.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    frames = _frame_count(bvh)
    legs = _legs(drawing, legs)
    far = "Near" if legs == _TWO_LEGS else "Far"
    width, height = window_dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"window_dimensions must be positive, got {window_dimensions}")
    mesa = render_backend() == "osmesa" if use_mesa is None else use_mesa
    video = (directory / _ANIMATION_GIF if output is None else output).resolve()
    if video.suffix not in {".gif", ".mp4"}:
        raise ValueError(f"output must end in .gif or .mp4, got {video}")

    motion = directory / "motion.yaml"
    retarget = directory / "retarget.yaml"
    mvc = directory / "mvc.yaml"
    gif = video.with_suffix(".gif")
    mp4 = video.with_suffix(".mp4")

    _write_yaml(
        motion,
        {
            "filepath": str(bvh),
            "start_frame_idx": 0,
            "end_frame_idx": frames - 1,
            "groundplane_joint": "Near_B_Paw",
            "forward_perp_joint_vectors": [["Fwd_Near", "Fwd_Far"]],
            "scale": 1.0,
            "up": "+y",
        },
    )
    _write_yaml(
        retarget,
        {
            "char_starting_location": [char_start_x, 0, -0.5],
            "bvh_projection_bodypart_groups": [
                {
                    "name": "all",
                    "method": "sagittal",
                    "bvh_joint_names": list(_SKELETON_JOINTS),
                }
            ],
            "char_bodypart_groups": _bodypart_groups(legs),
            "char_bvh_root_offset": {
                "bvh_projection_bodypart_group_for_offset": "all",
                # The engine scales root motion by the summed drawn legs over the summed
                # BVH legs, so both lists name the same four legs in the same order.
                "bvh_joints": [
                    ["Near_F_Paw", "Near_Elbow", "Near_Shoulder"],
                    [f"{far}_F_Paw", f"{far}_Elbow", f"{far}_Shoulder"],
                    ["Near_B_Paw", "Near_Knee", "Near_Hip"],
                    [f"{far}_B_Paw", f"{far}_Knee", f"{far}_Hip"],
                ],
                "char_joints": [
                    ["left_foot_f", "left_knee_f", "left_hip_f"],
                    ["right_foot_f", "right_knee_f", "right_hip_f"],
                    ["left_foot_b", "left_knee_b", "left_hip_b"],
                    ["right_foot_b", "right_knee_b", "right_hip_b"],
                ],
            },
            "char_joint_bvh_joints_mapping": _mapping(legs),
            "char_runtime_checks": [],
        },
    )
    _write_yaml(
        mvc,
        {
            "scene": {
                "ANIMATED_CHARACTERS": [
                    {
                        "character_cfg": str(character_cfg),
                        "motion_cfg": str(motion.resolve()),
                        "retarget_cfg": str(retarget.resolve()),
                    }
                ]
            },
            "controller": {
                "MODE": "video_render",
                "OUTPUT_VIDEO_PATH": str(video),
            },
            "view": {
                "USE_MESA": mesa,
                "WINDOW_DIMENSIONS": [width, height],
                "CLEAR_COLOR": [1.0, 1.0, 1.0, 0.0],
            },
        },
    )
    return AnimationConfigs(motion=motion, retarget=retarget, mvc=mvc, gif=gif, mp4=mp4)
