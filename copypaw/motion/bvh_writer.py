"""Export planar keypoints as driver-joint BVH motion."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from copypaw.motion.skeleton import BONES, HELPER_JOINTS, ROOT_KP, SIDE_OFFSET_RATIO, Bone

__all__ = ["NEAR_Z_SIGN", "BvhSummary", "write_bvh"]

NEAR_Z_SIGN = 1  # Positive z places near limbs in front in the animation engine.
_ROTATION_CHANNELS = "CHANNELS 3 Zrotation Xrotation Yrotation"
_KEYPOINT_SHAPE = (17, 2)
_KEYPOINT_DIMENSIONS = 3
_MEAN_DIRECTION_EPSILON = 1e-12  # Unit-vector cancellation tolerance.


@dataclass(frozen=True)
class BvhSummary:
    """Export dimensions and the pixel-to-body-length conversion used."""

    frame_count: int
    frame_time: float
    scale: float


def _hierarchy(
    parent: str | None,
    offsets: Mapping[str, NDArray[np.float64]],
    lines: list[str],
    order: list[str],
) -> None:
    for bone in BONES:
        if bone.parent != parent:
            continue
        lines.extend((f"JOINT {bone.name}_drv", "{", "OFFSET 0 0 0", _ROTATION_CHANNELS))
        order.extend((f"{bone.name}_drv", bone.name))
        offset = " ".join(f"{value:.6f}" for value in offsets[bone.name])
        lines.extend((f"JOINT {bone.name}", "{", f"OFFSET {offset}", _ROTATION_CHANNELS))
        if any(child.parent == bone.name for child in BONES):
            _hierarchy(bone.name, offsets, lines, order)
        else:
            lines.extend(("End Site", "{", "OFFSET 0 0.01 0", "}"))
        lines.extend(("}", "}"))


def _directions(
    kp: NDArray[np.float64],
    scale: float,
    bone_lengths: Mapping[str, float] | None,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    offsets: dict[str, NDArray[np.float64]] = {}
    deviations: dict[str, NDArray[np.float64]] = {}
    by_name = {bone.name: bone for bone in BONES}
    if bone_lengths is not None and set(bone_lengths) != set(by_name):
        raise ValueError("bone_lengths must contain exactly the skeleton bone names")
    for bone in BONES:
        vectors = kp[:, bone.dist_kp] - kp[:, bone.prox_kp]
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 0):
            raise ValueError(f"Bone {bone.name} must have positive length in every frame")
        mean = np.mean(vectors / lengths[:, None], axis=0)
        # A full turn can cancel the mean; its first direction fixes a stable rest pose.
        rest = vectors[0] / lengths[0] if np.linalg.norm(mean) < _MEAN_DIRECTION_EPSILON else mean
        rest = rest / np.linalg.norm(rest)
        length = (
            float(np.median(lengths)) * scale if bone_lengths is None else bone_lengths[bone.name]
        )
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"Bone length for {bone.name} must be finite and positive")
        offsets[bone.name] = np.array([rest[0] * length, rest[1] * length, _depth(bone)])
        deviations[bone.name] = np.arctan2(vectors[:, 1], vectors[:, 0]) - np.arctan2(
            rest[1], rest[0]
        )
    return offsets, deviations


def _depth(bone: Bone) -> float:
    if bone.side == "centre" or (
        bone.parent is not None
        and next(b for b in BONES if b.name == bone.parent).side == bone.side
    ):
        return 0.0
    return NEAR_Z_SIGN * SIDE_OFFSET_RATIO * (1 if bone.side == "near" else -1)


def write_bvh(
    kp_px: NDArray[np.float32] | NDArray[np.float64],
    fps: float,
    root_x: NDArray[np.float32] | NDArray[np.float64],
    bone_lengths: Mapping[str, float] | None = None,
    *,
    out: Path,
) -> BvhSummary:
    """Write cleaned image coordinates with lengths and root travel in body units.

    Missing bone_lengths uses median pixel lengths scaled by median Root to Neck
    length. Rest directions average per-frame unit vectors, so long observations
    do not outweigh short ones. Only first joints in side chains receive depth;
    descendants inherit it. Root y retains the image origin, with its sign flipped.
    """
    if kp_px.ndim != _KEYPOINT_DIMENSIONS or kp_px.shape[1:] != _KEYPOINT_SHAPE or len(kp_px) == 0:
        raise ValueError("Expected nonempty keypoint shape (F, 17, 2)")
    kp = kp_px.astype(np.float64, copy=True)
    if not np.isfinite(kp[:, 2:]).all():
        raise ValueError("Skeleton keypoints must be finite")
    if not np.isfinite(fps) or fps <= 0 or round(1 / fps, 6) <= 0:
        raise ValueError("fps must be positive with a nonzero six-decimal frame time")
    if root_x.shape != (len(kp),) or not np.isfinite(root_x).all():
        raise ValueError("root_x must contain one finite body-unit position per frame")
    body_length = float(np.median(np.linalg.norm(kp[:, 3] - kp[:, ROOT_KP], axis=1)))
    if body_length <= 0:
        raise ValueError("Median Root to Neck body length must be positive")
    scale = 1 / body_length
    kp[:, :, 1] *= -1
    offsets, deviations = _directions(kp, scale, bone_lengths)
    lines = [
        "HIERARCHY",
        "ROOT Root",
        "{",
        "OFFSET 0 0 0",
        "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
    ]
    order: list[str] = []
    _hierarchy(None, offsets, lines, order)
    for name, depth in HELPER_JOINTS:
        lines.extend(
            (
                f"JOINT {name}",
                "{",
                f"OFFSET 0 0 {depth:.6f}",
                _ROTATION_CHANNELS,
                "End Site",
                "{",
                "OFFSET 0 0.01 0",
                "}",
                "}",
            )
        )
        order.append(name)
    frame_time = round(1 / fps, 6)
    lines.extend(("}", "MOTION", f"Frames: {len(kp)}", f"Frame Time: {frame_time:.6f}"))
    rotations: dict[str, NDArray[np.float64]] = {}
    for bone in BONES:
        relative = deviations[bone.name].copy()
        if bone.parent is not None:
            relative -= deviations[bone.parent]
        rotations[f"{bone.name}_drv"] = np.rad2deg(relative)
    for frame in range(len(kp)):
        values = [float(root_x[frame]), float(kp[frame, ROOT_KP, 1] * scale), 0.0, 0.0, 0.0, 0.0]
        for name in order:
            angle = float(rotations[name][frame]) if name in rotations else 0.0
            values.extend((angle, 0.0, 0.0))
        lines.append(" ".join(f"{value:.6f}" for value in values))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BvhSummary(len(kp), frame_time, scale)
