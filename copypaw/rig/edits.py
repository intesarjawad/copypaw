"""Snap the template to a drawing and record how far a person moved its joints."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from copypaw.errors import CopypawError
from copypaw.rig.mask import Mask, segment
from copypaw.rig.snap import detect_facing, snap
from copypaw.rig.template import Joint, Joints, fit_template

__all__ = [
    "DrawingError",
    "JointsFileError",
    "RigResult",
    "joints_moved",
    "joints_to_json",
    "load_drawing",
    "load_joints",
    "rig_drawing",
    "write_edits",
]

# A joint that is further from its snapped position than this share of the box height was moved.
MOVED_FRACTION = 0.04
_PAIR = 2
# Snapping always moves the neck; every other joint off this list is a leg joint.
_BODY_JOINTS = frozenset({"root", "hip", "torso", "neck"})


class DrawingError(CopypawError):
    """The drawing file is not an image Copypaw can read."""

    codes = ("drawing_unreadable",)


class JointsFileError(CopypawError):
    """The joints file does not hold a position for every joint."""

    codes = ("joints_invalid",)


@dataclass(frozen=True)
class RigResult:
    """The drawing, its mask and the snapped joints.

    legs_snapped is False when no four or two separate legs were found, so the
    leg joints are the template's guess.
    """

    image: NDArray[np.uint8]
    mask: Mask
    snapped: Joints
    box_height: float
    legs_snapped: bool


def load_drawing(path: Path) -> NDArray[np.uint8]:
    """Read a drawing as uint8 BGR."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise DrawingError(path=path) from error
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise DrawingError(path=path)
    return np.asarray(image, dtype=np.uint8)


def rig_drawing(image: NDArray[np.uint8]) -> RigResult:
    """Fit the template to the drawing's mask and snap its joints.

    The drawing may face either way; the side its head is on decides.
    """
    mask = segment(image)
    if mask.shape != image.shape[:2]:
        mask = np.asarray(
            cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST),
            dtype=np.uint8,
        )
    rows, columns = np.nonzero(mask > 0)
    box = (
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    )
    facing = detect_facing(mask)
    template = fit_template(box, facing)
    snapped = snap(mask, template, facing)
    legs_snapped = any(
        snapped[name] != joint for name, joint in template.items() if name not in _BODY_JOINTS
    )
    snapped.legs_guessed = not legs_snapped
    return RigResult(image, mask, snapped, float(box[3] - box[1]), legs_snapped)


def joints_to_json(joints: Joints) -> str:
    """Write each joint as an [x, y] pair keyed by name."""
    return json.dumps({name: [joint.x, joint.y] for name, joint in joints.items()}, indent=2)


def _position(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(cast("list[object]", value)) != _PAIR:
        return None
    x, y = cast("list[object]", value)
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (x, y)):
        return None
    pair = (float(cast("float", x)), float(cast("float", y)))
    return pair if all(math.isfinite(v) for v in pair) else None


def load_joints(path: Path, template: Joints) -> Joints:
    """Read a joints file; parents, leg count and tails come from the snapped template."""
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise JointsFileError(path=path, problem=str(error)) from error
    if not isinstance(data, dict):
        raise JointsFileError(path=path, problem="expected an object keyed by joint name")
    positions = cast("dict[str, object]", data)
    if set(positions) != set(template):
        missing = sorted(set(template) - set(positions))
        extra = sorted(set(positions) - set(template))
        raise JointsFileError(
            path=path, problem=f"missing {missing or 'none'}, unknown {extra or 'none'}"
        )
    joints: dict[str, Joint] = {}
    for name, value in positions.items():
        position = _position(value)
        if position is None:
            raise JointsFileError(path=path, problem=f"{name} must be a finite [x, y] pair")
        joints[name] = Joint(*position, template[name].parent)
    return Joints(joints, legs=template.legs, tails=template.tails)


def joints_moved(snapped: Joints, final: Joints, box_height: float) -> int:
    """Count joints further than 4% of the box height from their snapped position."""
    limit = MOVED_FRACTION * box_height
    return sum(
        math.hypot(final[name].x - joint.x, final[name].y - joint.y) > limit
        for name, joint in snapped.items()
    )


def write_edits(path: Path, snapped: Joints, final: Joints, box_height: float) -> None:
    """Record the snapped joints, the final joints and how many were moved."""
    record = {
        "snapped": {name: [joint.x, joint.y] for name, joint in snapped.items()},
        "final": {name: [joint.x, joint.y] for name, joint in final.items()},
        "joints_moved": joints_moved(snapped, final, box_height),
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
