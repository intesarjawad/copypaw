"""Bake captured motion onto a drawing's proportions as BVH."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from copypaw.motion.bvh_writer import write_bvh
from copypaw.motion.motion_file import MotionFile
from copypaw.motion.skeleton import BONES

__all__ = ["BakeMode", "bake_bvh", "pose_for_drawing"]

type BakeMode = Literal["treadmill", "travel"]

_OUTPUT_NAME = "baked.bvh"
_NEAR_SHOULDER_KP = 5
_NEAR_HIP_KP = 11
_MEAN_DIRECTION_EPSILON = 1e-12
_LEG_SEGMENTS: Mapping[str, tuple[str, str]] = {
    "Near_Elbow": ("left_hip_f", "left_knee_f"),
    "Near_F_Paw": ("left_knee_f", "left_foot_f"),
    "Near_Knee": ("left_hip_b", "left_knee_b"),
    "Near_B_Paw": ("left_knee_b", "left_foot_b"),
    "Far_Elbow": ("right_hip_f", "right_knee_f"),
    "Far_F_Paw": ("right_knee_f", "right_foot_f"),
    "Far_Knee": ("right_hip_b", "right_knee_b"),
    "Far_B_Paw": ("right_knee_b", "right_foot_b"),
}


# Each chain: its three keypoints, its contact column and its three drawing joints.
_LEG_CHAINS: tuple[tuple[tuple[int, int, int], int, tuple[str, str, str]], ...] = (
    ((5, 6, 7), 0, ("left_hip_f", "left_knee_f", "left_foot_f")),
    ((8, 9, 10), 1, ("right_hip_f", "right_knee_f", "right_foot_f")),
    ((11, 12, 13), 2, ("left_hip_b", "left_knee_b", "left_foot_b")),
    ((14, 15, 16), 3, ("right_hip_b", "right_knee_b", "right_foot_b")),
)
# Below this sine of the knee angle (about 6 degrees) a drawn leg counts as straight.
_STRAIGHT_LEG_SINE = 0.1
_NOSE_KP = 2
_NECK_KP = 3


def _cross(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])


def _turned(vector: NDArray[np.float64], angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the vector turned by each frame's angle."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.stack((cos * vector[0] - sin * vector[1], sin * vector[0] + cos * vector[1]), axis=1)


def _swing(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return each frame's angle from the mean direction of the vectors."""
    units = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    mean = units.mean(axis=0)
    # A full turn can cancel the mean; the first frame then fixes the reference.
    if np.linalg.norm(mean) < _MEAN_DIRECTION_EPSILON:
        mean = units[0]
    return np.arctan2(_cross(mean, units), units @ mean)


def _lift(
    paw: NDArray[np.float64],
    planted: NDArray[np.bool_],
    ground_line_px: NDArray[np.float64],
    full_length: float,
) -> NDArray[np.float64]:
    """Return how high the paw is off the ground per frame, in leg lengths.

    A planted paw is on the ground whatever its tracked height says, and its
    median tracked height is where the ground is for this paw.
    """
    slope, intercept = (float(value) for value in ground_line_px)
    clearance = slope * paw[:, 0] + intercept - paw[:, 1]
    level = float(np.median(clearance[planted])) if planted.any() else float(clearance.min())
    return np.where(planted, 0.0, np.maximum(clearance - level, 0.0)) / full_length


def _posed_leg(
    swing: NDArray[np.float64],
    lift: NDArray[np.float64],
    pet_bend: float,
    drawn: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return the drawn knee and foot per frame, relative to the hip.

    The leg turns about its hip by the pet leg's swing, keeping its drawn
    shape. Off the ground its foot clears the ground by at least the pet's
    lift: where the swing alone does not raise it enough the knee folds, up
    to a right angle, and the leg keeps the direction of the swing.
    """
    hip, knee, foot = drawn
    upper = float(np.linalg.norm(knee - hip))
    lower = float(np.linalg.norm(foot - knee))
    rest = foot - hip
    rest_length = float(np.linalg.norm(rest))
    if min(upper, lower, rest_length) <= 0:
        return None
    direction = _turned(rest / rest_length, swing)
    # A leg at full stretch straight down reaches the ground; the swing alone may clear it.
    # The paw is lifted along the swung leg, by folding the knee, so the leg never swings
    # further than the pet's did.
    drop = np.maximum(direction[:, 1], 1e-300)
    wanted = np.where(direction[:, 1] > 0, rest_length * (1 - lift) / drop, rest_length)
    # Past a right angle the lower leg folds over the upper one, so the knee stops there.
    reach = np.clip(wanted, min(math.hypot(upper, lower), rest_length), rest_length)
    normal = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
    along = (upper**2 + reach**2 - lower**2) / (2 * reach)
    across = np.sqrt(np.maximum(upper**2 - along**2, 0.0))
    drawn_bend = float(_cross(knee - hip, foot - knee)) / (upper * lower)
    # A leg drawn straight has no bend side of its own, so it takes the pet's.
    straight = abs(drawn_bend) < _STRAIGHT_LEG_SINE
    side = (np.sign(pet_bend) or 1.0) if straight else np.sign(drawn_bend)
    posed_knee = along[:, None] * direction - side * across[:, None] * normal
    return posed_knee, reach[:, None] * direction


def pose_for_drawing(
    motion: MotionFile,
    char_joints: Mapping[str, tuple[float, float]],
    *,
    still_legs: bool = False,
) -> NDArray[np.float32]:
    """Return keypoints whose legs and head move as the drawing's own would.

    The engine turns each drawn bone to the absolute direction of its motion
    bone, which would give a drawing the pet's anatomy as well as its gait.
    Here every drawn leg keeps its drawn shape while its paw is planted,
    swings from the hip by the pet's swing about its average direction, and
    bends at the knee only to lift its foot as high as the pet lifts its paw.
    The neck turns from its drawn direction by the turn of the pet's head.
    Joints the drawing lacks, or drew on one spot, stay as captured. With
    still_legs every leg keeps its drawn pose throughout, for a drawing whose
    leg joints are a guess: moving the mesh about joints that are not on its
    legs would shear it.
    """
    posed = motion.kp.astype(np.float64, copy=True)
    for (prox, mid, paw), column, names in _LEG_CHAINS:
        if not all(name in char_joints for name in names):
            continue
        hip, knee, foot = (_point(char_joints, name) for name in names)
        chord = posed[:, paw] - posed[:, prox]
        lengths = np.linalg.norm(chord, axis=1)
        if np.any(lengths <= 0):
            continue
        planted = motion.contacts[:, column]
        full = float(lengths[planted].max()) if planted.any() else float(lengths.max())
        pet_bend = float(
            _cross(posed[:, mid] - posed[:, prox], posed[:, paw] - posed[:, mid]).sum()
        )
        still = np.zeros(len(chord))
        leg = _posed_leg(
            still if still_legs else _swing(chord),
            still if still_legs else _lift(posed[:, paw], planted, motion.ground_line_px, full),
            pet_bend,
            (hip, knee, foot),
        )
        if leg is None:
            continue
        scale = full / float(np.linalg.norm(foot - hip))
        posed[:, mid] = posed[:, prox] + leg[0] * scale
        posed[:, paw] = posed[:, prox] + leg[1] * scale
    if "torso" in char_joints and "neck" in char_joints:
        drawn = _point(char_joints, "neck") - _point(char_joints, "torso")
        head = posed[:, _NOSE_KP] - posed[:, _NECK_KP]
        length = float(np.linalg.norm(drawn))
        if length > 0 and np.all(np.linalg.norm(head, axis=1) > 0):
            reach = float(np.median(np.linalg.norm(head, axis=1)))
            posed[:, _NOSE_KP] = posed[:, _NECK_KP] + _turned(drawn / length, _swing(head)) * reach
    return posed.astype(np.float32)


def _point(joints: Mapping[str, tuple[float, float]], name: str) -> NDArray[np.float64]:
    point = np.asarray(joints[name], dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"Character joint {name} must be a finite 2D point")
    return point


def _distance(joints: Mapping[str, tuple[float, float]], prox: str, dist: str) -> float:
    return float(np.linalg.norm(_point(joints, dist) - _point(joints, prox)))


def _video_bone_lengths(kp: NDArray[np.float32]) -> dict[str, float]:
    """Return the BVH lengths the video supplies, in its body-length units."""
    neck = kp[:, 3].astype(np.float64)
    root = kp[:, 4].astype(np.float64)
    body = float(np.median(np.linalg.norm(neck - root, axis=1)))
    if not np.isfinite(body) or body <= 0:
        raise ValueError("Median Neck to Root body length must be positive")
    scale = 1 / body
    points = kp.astype(np.float64)
    return {
        bone.name: float(
            np.median(np.linalg.norm(points[:, bone.dist_kp] - points[:, bone.prox_kp], axis=1))
        )
        * scale
        for bone in BONES
    }


def bake_bvh(motion: Path, char_joints: Mapping[str, tuple[float, float]], mode: BakeMode) -> Path:
    """Write a BVH with the drawing's leg proportions beside the motion archive.

    Leg bone lengths preserve the drawing's hip-to-knee and knee-to-foot lengths
    after converting them by the video-to-drawing body ratio. Bones outside the
    legs keep the lengths measured from the video. The legs and neck move from
    their drawn pose, as pose_for_drawing describes. Treadmill mode removes root travel; travel mode
    keeps the captured root movement.
    """
    if mode not in ("treadmill", "travel"):
        raise ValueError(f"Unknown bake mode {mode!r}")
    captured = MotionFile.load(motion)
    kp = captured.kp
    points = kp.astype(np.float64)
    video_body = float(
        np.median(np.linalg.norm(points[:, _NEAR_SHOULDER_KP] - points[:, _NEAR_HIP_KP], axis=1))
    )
    if not np.isfinite(video_body) or video_body <= 0:
        raise ValueError("Median Near_Shoulder to Near_Hip body length must be positive")
    drawing_body = _distance(char_joints, "left_hip_f", "left_hip_b")
    if drawing_body <= 0:
        raise ValueError("Drawing body length must be positive")
    ratio = video_body / drawing_body
    lengths = _video_bone_lengths(kp)
    for bone, (prox, dist) in _LEG_SEGMENTS.items():
        lengths[bone] = _distance(char_joints, prox, dist) * ratio
    root_x = np.zeros(len(kp)) if mode == "treadmill" else captured.travel_root_x
    out = motion.with_name(_OUTPUT_NAME)
    posed = pose_for_drawing(captured, char_joints)
    write_bvh(posed, captured.fps, root_x, lengths, out=out)
    return out
