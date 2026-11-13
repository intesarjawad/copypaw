"""Synthetic sagittal motion and configs for testing quadruped retargeting."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

FRAME_COUNT = 60
FRAME_MS = 50  # Exactly representable in GIF's 10 ms duration units.
FRAMES_PER_CYCLE = 20
NEAR_Z_SIGN = 1
LATERAL_OFFSET = 0.05
HIP_SWING_DEGREES = 25
KNEE_BEND_DEGREES = 18


@dataclass(frozen=True)
class SpikeFiles:
    """Paths to a self-contained synthetic motion and its render configs."""

    bvh: Path
    motion: Path
    retarget: Path
    mvc: Path
    gif: Path


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str | None
    offset: tuple[float, float, float]


def _skeleton(near_z_sign: int) -> tuple[_Joint, ...]:
    joints = [
        _Joint("Root", None, (0, 0.8, 0)),
        _Joint("Neck", "Root", (0.4, 0.15, 0)),
        _Joint("Nose", "Neck", (0.2, 0.05, 0)),
        # Helpers keep facing independent of the leg depth sign under test.
        _Joint("Fwd_Near", "Root", (0, 0, LATERAL_OFFSET)),
        _Joint("Fwd_Far", "Root", (0, 0, -LATERAL_OFFSET)),
    ]
    for side, sign in (("Near", near_z_sign), ("Far", -near_z_sign)):
        for anchor, middle, paw, x in (
            ("Shoulder", "Elbow", "F_Paw", 0.4),
            ("Hip", "Knee", "B_Paw", -0.4),
        ):
            joints.extend(
                (
                    _Joint(f"{side}_{anchor}", "Root", (x, 0, sign * LATERAL_OFFSET)),
                    _Joint(f"{side}_{middle}", f"{side}_{anchor}", (0, -0.3, 0)),
                    _Joint(f"{side}_{paw}", f"{side}_{middle}", (0, -0.3, 0)),
                )
            )
    return tuple(joints)


def _hierarchy(joint: _Joint, joints: tuple[_Joint, ...]) -> list[str]:
    kind = "ROOT" if joint.parent is None else "JOINT"
    channels = (
        "6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
        if joint.parent is None
        else "3 Zrotation Xrotation Yrotation"
    )
    lines = [
        f"{kind} {joint.name}",
        "{",
        "OFFSET " + " ".join(str(value) for value in joint.offset),
        f"CHANNELS {channels}",
    ]
    for child in joints:
        if child.parent == joint.name:
            lines.extend(_hierarchy(child, joints))
    return [*lines, "}"]


def _rotation_degrees(name: str, phase: float) -> float:
    if not name.startswith(("Near_", "Far_")):
        return 0
    if name.startswith("Far_"):
        phase += math.pi
    if name.endswith(("_Hip", "_Knee", "_B_Paw")):
        phase += math.pi
    if name.endswith(("_Shoulder", "_Hip")):
        return HIP_SWING_DEGREES * math.sin(phase)
    if name.endswith(("_Elbow", "_Knee")):
        return KNEE_BEND_DEGREES * (1 + math.sin(phase))
    return 0


def _write_bvh(path: Path, joints: tuple[_Joint, ...]) -> None:
    lines = [
        "HIERARCHY",
        *_hierarchy(joints[0], joints),
        "MOTION",
        f"Frames: {FRAME_COUNT}",
        f"Frame Time: {FRAME_MS / 1000}",
    ]
    for index in range(FRAME_COUNT):
        phase = 2 * math.pi * index / FRAMES_PER_CYCLE
        values = [0.0, 0.8, 0.0]
        for joint in joints:
            values.extend((_rotation_degrees(joint.name, phase), 0, 0))
        lines.append(" ".join(f"{value:.8f}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _retarget_config(joints: tuple[_Joint, ...]) -> dict[str, object]:
    mapping = {"neck": ("Neck", "Nose")}
    groups: list[dict[str, object]] = [
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
        }
    ]
    for prefix, side in (("left", "Near"), ("right", "Far")):
        for limb, anchor, middle in (("f", "Shoulder", "Elbow"), ("b", "Hip", "Knee")):
            paw = f"{side}_{limb.upper()}_Paw"
            knee_name, foot_name = f"{prefix}_knee_{limb}", f"{prefix}_foot_{limb}"
            mapping[knee_name] = (f"{side}_{anchor}", f"{side}_{middle}")
            mapping[foot_name] = (f"{side}_{middle}", paw)
            groups.append({"bvh_depth_drivers": [paw], "char_joints": [knee_name, foot_name]})
    return {
        "char_starting_location": [0, 0, -0.5],
        "bvh_projection_bodypart_groups": [
            {"name": "all", "method": "sagittal", "bvh_joint_names": [j.name for j in joints]}
        ],
        "char_bodypart_groups": groups,
        "char_bvh_root_offset": {
            "bvh_projection_bodypart_group_for_offset": "all",
            "bvh_joints": [
                [f"{side}_B_Paw", f"{side}_Knee", f"{side}_Hip"] for side in ("Near", "Far")
            ],
            "char_joints": [
                [f"{prefix}_foot_{limb}", f"{prefix}_knee_{limb}", f"{prefix}_hip_{limb}"]
                for limb in ("f", "b")
                for prefix in ("left", "right")
            ],
        },
        "char_joint_bvh_joints_mapping": mapping,
        "char_runtime_checks": [],
    }


def write_spike(
    directory: Path, character: Path, *, use_mesa: bool = False, near_z_sign: int = NEAR_Z_SIGN
) -> SpikeFiles:
    """Write a stationary trot and tuple-tagged YAML for the vendored renderer.

    Near and far legs are half a cycle apart; diagonal legs share a phase.
    Positive near depth is the candidate convention checked by the overlap render.
    """
    if near_z_sign not in (-1, 1):
        raise ValueError("near_z_sign must be +1 or -1")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    files = SpikeFiles(
        bvh=directory / "trot.bvh",
        motion=directory / "motion.yaml",
        retarget=directory / "retarget.yaml",
        mvc=directory / "mvc.yaml",
        gif=directory / "trot.gif",
    )
    joints = _skeleton(near_z_sign)
    _write_bvh(files.bvh, joints)
    files.motion.write_text(
        yaml.safe_dump(
            {
                "filepath": str(files.bvh),
                "start_frame_idx": 0,
                # The loader slices with an exclusive end index.
                "end_frame_idx": FRAME_COUNT,
                "groundplane_joint": "Near_B_Paw",
                # up cross (near - far) faces +X for the fixed positive-Z helpers.
                "forward_perp_joint_vectors": [["Fwd_Far", "Fwd_Near"]],
                "scale": 1.0,
                "up": "+y",
            }
        ),
        encoding="utf-8",
    )
    # FullLoader accepts !!python/tuple; safe_dump would lose the required tuple type.
    files.retarget.write_text(yaml.dump(_retarget_config(joints)), encoding="utf-8")
    files.mvc.write_text(
        yaml.safe_dump(
            {
                "scene": {
                    "ANIMATED_CHARACTERS": [
                        {
                            "character_cfg": str(character.resolve()),
                            "motion_cfg": str(files.motion),
                            "retarget_cfg": str(files.retarget),
                        }
                    ]
                },
                "controller": {"MODE": "video_render", "OUTPUT_VIDEO_PATH": str(files.gif)},
                "view": {
                    "USE_MESA": use_mesa,
                    "WINDOW_DIMENSIONS": [512, 512],
                    "CLEAR_COLOR": [1.0, 1.0, 1.0, 0.0],
                },
            }
        ),
        encoding="utf-8",
    )
    return files
