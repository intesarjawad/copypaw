"""Fit the upstream four-legged joint template to a drawing's bounding box."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = ["Joint", "Joints", "Tail", "fit_template"]

_TEMPLATE_WIDTH = 564.0
_TEMPLATE_HEIGHT = 352.0


@dataclass(frozen=True)
class Joint:
    """A named joint in source-pixel coordinates."""

    x: float
    y: float
    parent: str | None


# Where a tail leaves the body and where it ends, as (x, y) in source pixels.
type Tail = tuple[tuple[float, float], tuple[float, float]]


class Joints(dict[str, Joint]):
    """Joint positions by name, plus the number of drawn legs and any tails.

    legs is 2 for a side view with one visible leg per pair, so the animation
    can drive the far chain with the near chain's motion. tails holds the
    tails that reach down among the legs; the rig has no joints for them, so
    the animation needs to know where they are to keep them off the ground.
    legs_guessed marks leg joints that are the template's, not found on the
    drawing, so the animation leaves those legs as drawn.
    """

    legs: int
    tails: tuple[Tail, ...]
    legs_guessed: bool

    def __init__(
        self,
        joints: Mapping[str, Joint],
        *,
        legs: int = 4,
        tails: tuple[Tail, ...] = (),
        legs_guessed: bool = False,
    ) -> None:
        """Copy joints and record how many legs and which tails the drawing has."""
        super().__init__(joints)
        self.legs = legs
        self.tails = tails
        self.legs_guessed = legs_guessed


# name, parent, and pixel location in the upstream character's 564 x 352 box.
_TEMPLATE: tuple[tuple[str, str | None, float, float], ...] = (
    ("root", None, 238, 264),
    ("hip", "root", 238, 264),
    ("torso", "hip", 453, 181),
    ("neck", "torso", 455, 121),
    ("right_hip_b", "root", 139, 266),
    ("right_knee_b", "right_hip_b", 138, 290),
    ("right_foot_b", "right_knee_b", 132, 324),
    ("left_hip_b", "root", 82, 249),
    ("left_knee_b", "left_hip_b", 65, 279),
    ("left_foot_b", "left_knee_b", 47, 308),
    ("right_hip_f", "root", 442, 245),
    ("right_knee_f", "right_hip_f", 455, 284),
    ("right_foot_f", "right_knee_f", 458, 320),
    ("left_hip_f", "root", 364, 266),
    ("left_knee_f", "left_hip_f", 364, 305),
    ("left_foot_f", "left_knee_f", 360, 336),
)


def fit_template(mask_bbox: Sequence[float], facing: Literal["left", "right"] = "right") -> Joints:
    """Scale the template into an xyxy drawing box, mirrored for a left-facing drawing.

    Every template point lies strictly inside the box because its normalized
    coordinates stay below 1, so no point needs clamping.
    """
    left, top, right, bottom = (float(value) for value in mask_bbox)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError("mask_bbox must contain four finite coordinates")
    if right <= left or bottom <= top:
        raise ValueError("mask_bbox must be finite xyxy coordinates with positive area")

    width = right - left
    height = bottom - top
    return Joints(
        {
            name: Joint(
                left + (x if facing == "right" else _TEMPLATE_WIDTH - x) / _TEMPLATE_WIDTH * width,
                top + y / _TEMPLATE_HEIGHT * height,
                parent,
            )
            for name, parent, x, y in _TEMPLATE
        }
    )
