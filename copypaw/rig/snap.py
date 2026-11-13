"""Place the template's joints on the pixels of a segmented drawing."""

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize

from copypaw.rig.mask import Mask
from copypaw.rig.template import Joint, Joints, Tail

__all__ = ["detect_facing", "snap"]

_LEG_REGION_TOP = 0.6
_OPEN_KERNEL_PX = 5
_MIN_LEG_FRACTION = 0.005
_EDGE_ROWS = 3
_KNEE_FRACTION = 0.55
_NOSE_REGION = 0.5
_NECK_FRACTION = 0.7
# A head stands above the back, so the top quarter of the drawing leans to its side.
_HEAD_REGION = 0.25
# Share of the drawing's width the top quarter must lean before it counts as a head.
_FACING_DEADBAND = 0.02
_FOUR_LEGS = 4
_TWO_LEGS = 2
_MASK_DIMENSIONS = 2
# An appendage has joined the body where its run of mask pixels grows half again wider.
_BODY_RUN_FACTOR = 1.5
# Drawn legs stand within a few degrees of vertical; the sample's tail leans 14 degrees back.
_TAIL_LEAN_DEG = 8.0
# Leaning back this far, an appendage under the body is not standing on it either.
_STEEP_LEAN_DEG = 20.0
# A tail narrows towards the ground; a leg keeps its width or ends in a wider foot.
_TAIL_TAPER = 0.6


@dataclass(frozen=True)
class _Leg:
    """An appendage below the body: its left edge, joint positions and outline measures.

    The hip is where the appendage widens into the body, so the leg swings
    from the body and not from the row where the search began. taper is its
    median width over the bottom quarter divided by that over the top quarter.
    """

    left: int
    hip: tuple[float, float]
    knee: tuple[float, float]
    foot: tuple[float, float]
    taper: float


def _attachment(mask: Mask, x: float, top: int, width: float) -> tuple[float, float]:
    """Follow an appendage up through the mask until its run widens into the body."""
    centre = x
    for row in range(top - 1, -1, -1):
        filled = mask[row] > 0
        column = round(centre)
        if not filled[column]:
            return centre, float(row + 1)
        empty = np.flatnonzero(~filled)
        left = int(empty[empty < column].max(initial=-1)) + 1
        right = int(empty[empty > column].min(initial=len(filled)))
        if right - left > _BODY_RUN_FACTOR * width:
            return centre, float(row)
        centre = (left + right - 1) / 2
    return centre, 0.0


def _leg_geometry(component: NDArray[np.bool_], mask: Mask) -> _Leg:
    rows, columns = np.nonzero(component)
    top, bottom = int(rows.min()), int(rows.max())
    top_x = float(np.median(columns[rows < top + _EDGE_ROWS]))
    bottom_x = float(np.median(columns[rows > bottom - _EDGE_ROWS]))
    foot = (bottom_x, float(bottom))
    widths = np.bincount(rows - top)
    quarter = max(1, len(widths) // 4)
    taper = float(np.median(widths[-quarter:]) / max(1.0, float(np.median(widths[:quarter]))))
    hip = _attachment(mask, top_x, top, float(np.median(widths)))
    knee = (
        hip[0] + _KNEE_FRACTION * (foot[0] - hip[0]),
        hip[1] + _KNEE_FRACTION * (foot[1] - hip[1]),
    )
    return _Leg(int(columns.min()), hip, knee, foot, taper)


def _rear_lean_deg(leg: _Leg, facing: Literal["left", "right"]) -> float:
    """Degrees from vertical that the appendage leans back from its attachment to its foot."""
    rear = -1.0 if facing == "right" else 1.0
    (attach_x, attach_y), (foot_x, foot_y) = leg.hip, leg.foot
    return math.degrees(math.atan2(rear * (foot_x - attach_x), max(foot_y - attach_y, 1.0)))


def _without_tails(
    legs: list[_Leg], facing: Literal["left", "right"]
) -> tuple[list[_Leg], tuple[Tail, ...]]:
    """Split the tails from more than four appendages, if exactly four legs remain.

    A tail is the rearmost appendage, attached behind every other one, that
    leans back or narrows to the ground, or any appendage in the rear half
    that leans steeply back instead of standing under the body. Otherwise
    the appendages are returned unchanged and the caller keeps the template.
    """
    if len(legs) <= _FOUR_LEGS:
        return legs, ()
    rear = -1.0 if facing == "right" else 1.0
    attachments = [leg.hip[0] for leg in legs]
    middle = (min(attachments) + max(attachments)) / 2
    rearmost = max(legs, key=lambda leg: rear * leg.hip[0])
    tails = [
        leg
        for leg in legs
        if (
            leg is rearmost
            and (_rear_lean_deg(leg, facing) >= _TAIL_LEAN_DEG or leg.taper <= _TAIL_TAPER)
        )
        or (rear * (leg.hip[0] - middle) > 0 and _rear_lean_deg(leg, facing) >= _STEEP_LEAN_DEG)
    ]
    kept = [leg for leg in legs if leg not in tails]
    if len(kept) != _FOUR_LEGS:
        return legs, ()
    return kept, tuple((tail.hip, tail.foot) for tail in tails)


def _legs(mask: Mask) -> list[_Leg]:
    height = mask.shape[0]
    leg_top = int(height * _LEG_REGION_TOP)
    region = np.zeros_like(mask, dtype=np.uint8)
    region[leg_top:, :] = np.where(mask[leg_top:, :] > 0, 255, 0).astype(np.uint8)
    kernel = np.ones((_OPEN_KERNEL_PX, _OPEN_KERNEL_PX), dtype=np.uint8)
    opened = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    minimum_area = mask.size * _MIN_LEG_FRACTION
    components = [
        _leg_geometry(labels == label, mask)
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] > minimum_area
    ]
    return sorted(components, key=lambda leg: leg.left)


def _place_leg(joints: dict[str, Joint], leg: _Leg, side: str, end: str) -> None:
    hip_name = f"{side}_hip_{end}"
    knee_name = f"{side}_knee_{end}"
    foot_name = f"{side}_foot_{end}"
    joints[hip_name] = Joint(*leg.hip, "root")
    joints[knee_name] = Joint(*leg.knee, hip_name)
    joints[foot_name] = Joint(*leg.foot, knee_name)


def _place_pair(joints: dict[str, Joint], pair: list[_Leg], end: str) -> None:
    # A leg drawn closer to the viewer reaches lower on the page.
    near = pair[0] if pair[0].foot[1] >= pair[1].foot[1] else pair[1]
    far = pair[1] if near is pair[0] else pair[0]
    _place_leg(joints, near, "left", end)
    _place_leg(joints, far, "right", end)


def _place_side_view(joints: dict[str, Joint], leg: _Leg, end: str) -> None:
    _place_leg(joints, leg, "left", end)
    _place_leg(joints, leg, "right", end)


def _medial_axis(mask: Mask) -> NDArray[np.intp]:
    """Return the (x, y) pixels of the mask's medial axis."""
    rows, columns = np.nonzero(skeletonize(mask > 0))
    return np.column_stack((columns, rows))


def _on_axis(axis: NDArray[np.intp], x: float, y: float) -> tuple[float, float]:
    nearest = axis[np.argmin(np.hypot(axis[:, 0] - x, axis[:, 1] - y))]
    return float(nearest[0]), float(nearest[1])


def _above(mask: Mask, pair: list[_Leg]) -> tuple[float, float] | None:
    """Return the middle of the body over a pair of legs, or None if they float free."""
    column = round(sum(leg.hip[0] for leg in pair) / len(pair))
    filled = np.flatnonzero(mask[: round(min(leg.hip[1] for leg in pair)), column] > 0)
    if not filled.size:
        return None
    bottom = int(filled[-1])
    empty = np.flatnonzero(mask[:bottom, column] == 0)
    top = int(empty[-1]) + 1 if empty.size else 0
    return float(column), (top + bottom) / 2


def _body(
    joints: dict[str, Joint],
    mask: Mask,
    facing: Literal["left", "right"],
    front: list[_Leg],
    back: list[_Leg],
) -> None:
    """Move the hip, torso and neck onto the medial axis of the outline.

    The hip sits over the back legs and the torso over the front legs; without
    legs they start from the template. The neck starts 70% of the way from the
    torso to the nose. A joint outside the outline cannot move the drawing, so
    each one ends on the nearest medial pixel.
    """
    axis = _medial_axis(mask)
    hip = (_above(mask, back) if back else None) or (joints["hip"].x, joints["hip"].y)
    torso = (_above(mask, front) if front else None) or (joints["torso"].x, joints["torso"].y)
    hip = _on_axis(axis, *hip)
    torso = _on_axis(axis, *torso)

    rows = np.flatnonzero(mask.any(axis=1))
    nose_bottom = int(rows[0] + max(1, (rows[-1] + 1 - rows[0]) * _NOSE_REGION))
    nose_rows, nose_columns = np.nonzero(mask[:nose_bottom, :] > 0)
    index = int(np.argmax(nose_columns) if facing == "right" else np.argmin(nose_columns))
    nose_x, nose_y = float(nose_columns[index]), float(nose_rows[index])
    neck = _on_axis(
        axis,
        torso[0] + _NECK_FRACTION * (nose_x - torso[0]),
        torso[1] + _NECK_FRACTION * (nose_y - torso[1]),
    )
    joints["root"] = Joint(*hip, None)
    joints["hip"] = Joint(*hip, "root")
    joints["torso"] = Joint(*torso, "hip")
    joints["neck"] = Joint(*neck, "torso")


def detect_facing(mask: Mask) -> Literal["left", "right"]:
    """Return the side the head is on, or right when no head stands out.

    The head is the mass in the top quarter of the drawing. A raised tail is
    in that quarter too, but it is thinner than a head and neck.
    """
    rows, columns = np.nonzero(mask > 0)
    if not rows.size:
        raise ValueError("mask must contain the drawing")
    top, height = int(rows.min()), int(rows.max()) + 1 - int(rows.min())
    left, width = int(columns.min()), int(columns.max()) + 1 - int(columns.min())
    head = columns[rows < top + max(1, height * _HEAD_REGION)]
    lean = (float(head.mean()) - (left + (width - 1) / 2)) / width
    return "left" if lean < -_FACING_DEADBAND else "right"


def snap(mask: Mask, joints: Joints, facing: Literal["left", "right"]) -> Joints:
    """Move leg and body joints onto the drawing's mask.

    Four leg components become front and back pairs; the leg that reaches
    lower in each pair is near. With more than four, tails are set aside first
    and stay part of the body. Two components describe a side view, so both
    sides share one position per pair and the result records two legs. Any
    other component count keeps the fitted template legs. The hip, torso and
    neck always end inside the outline.
    """
    if facing not in {"left", "right"}:
        raise ValueError("facing must be 'left' or 'right'")
    if mask.ndim != _MASK_DIMENSIONS or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("Expected a nonempty two-dimensional mask")
    if not mask.any():
        raise ValueError("mask must contain the drawing")

    updated = dict(joints)
    legs, tails = _without_tails(_legs(mask), facing)
    result_legs = _FOUR_LEGS
    front: list[_Leg] = []
    back: list[_Leg] = []
    if len(legs) == _FOUR_LEGS:
        if facing == "right":
            front, back = legs[2:], legs[:2]
        else:
            front, back = legs[:2], legs[2:]
        _place_pair(updated, front, "f")
        _place_pair(updated, back, "b")
    elif len(legs) == _TWO_LEGS:
        front, back = ([legs[1]], [legs[0]]) if facing == "right" else ([legs[0]], [legs[1]])
        _place_side_view(updated, front[0], "f")
        _place_side_view(updated, back[0], "b")
        result_legs = _TWO_LEGS

    _body(updated, mask, facing, front, back)
    return Joints(updated, legs=result_legs, tails=tails)
