"""Extra mesh handles that keep a drawing's bones straight between its joints."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["tail_handles", "with_bone_handles"]

# The mesh is deformed as rigidly as its handles allow. A handle at each joint
# alone lets a limb curve along its whole length like a hose. Each leg bone
# therefore carries handles along it, and a pair either side of it near each
# end, which turn with the bone and hold its width, so the leg stays a
# straight drawn limb and the bend stays at the joint.
_LEG_BONES: tuple[tuple[str, str], ...] = tuple(
    (f"{side}_{upper}_{end}", f"{side}_{lower}_{end}")
    for side in ("left", "right")
    for end in ("f", "b")
    for upper, lower in (("hip", "knee"), ("knee", "foot"))
)
# Along the bone, then across it, in bone lengths. A tenth of the length either
# side stays inside all but stick-thin legs, whose side handles the solver ignores.
_BONE_HANDLES: tuple[tuple[float, float], ...] = (
    (0.25, 0.0),
    (0.25, 0.1),
    (0.25, -0.1),
    (0.5, 0.0),
    (0.75, 0.0),
    (0.75, 0.1),
    (0.75, -0.1),
)
_NECK_BONE = ("torso", "neck")
# The neck joint sits 70% of the way from the torso to the nose. A handle half
# of the rest of the way on turns the head with the neck instead of leaving it behind.
_HEAD_REACH = 0.5 * (1 - 0.7) / 0.7


def with_bone_handles(
    joint_names: Sequence[str], positions: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Return the joint positions followed by the handles of each leg bone and the head.

    Bones whose joints are not both named are left out, so the same names
    always give the same handles in the same order. A handle that falls
    outside the drawing is ignored by the mesh solver.
    """
    index = {name: row for row, name in enumerate(joint_names)}
    if positions.shape[0] != len(index):
        raise ValueError(
            f"positions has {positions.shape[0]} joints but {len(index)} names were given"
        )
    handles: list[NDArray[np.float32]] = []
    for upper, lower in _LEG_BONES:
        if upper not in index or lower not in index:
            continue
        bone = positions[index[lower]] - positions[index[upper]]
        across = np.array([-bone[1], bone[0]], dtype=np.float32)
        handles.extend(
            positions[index[upper]] + along * bone + aside * across
            for along, aside in _BONE_HANDLES
        )
    torso, neck = _NECK_BONE
    if torso in index and neck in index:
        bone = positions[index[neck]] - positions[index[torso]]
        handles.append(positions[index[neck]] + _HEAD_REACH * bone)
    return np.vstack((positions, *handles)).astype(np.float32, copy=False)


def tail_handles(
    tails: NDArray[np.float32],
    body_shift: NDArray[np.floating],
    ground: float | None,
    *,
    turn: float = 0.0,
    behind: NDArray[np.floating] | None = None,
    lift: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Return a base and a tip handle for each drawn tail, shape (2 * tails, 2).

    tails holds each tail's drawn base and tip, shape (tails, 2, 2). The base
    moves with the body and the tail turns about it by turn radians, keeping
    its length. behind gives, per tail, the x its tip must not pass on the way
    to the hind legs; the tail swings away from them instead. A tip that would
    pass through the ground the paws stand on stays on it and the tail swings
    out about its base, so the tip rests on the ground without being planted.
    lift, per tail, raises it by that many radians away from the legs.
    """
    base = (tails[:, 0] + body_shift).astype(np.float64)
    drawn = (tails[:, 1] - tails[:, 0]).astype(np.float64)
    length = np.linalg.norm(drawn, axis=1)
    side = np.where(drawn[:, 0] < 0, -1.0, 1.0)
    angle = turn + (0.0 if lift is None else side * lift)
    cos, sin = np.cos(angle), np.sin(angle)
    turned_x = cos * drawn[:, 0] - sin * drawn[:, 1]
    reach = np.stack((turned_x, sin * drawn[:, 0] + cos * drawn[:, 1]), axis=1)
    if behind is not None:
        # Measured away from the legs: the tip stays at least as far out as the rearmost leg.
        out = np.clip(np.maximum(reach[:, 0] * side, (behind - base[:, 0]) * side), -length, length)
        moved = out > reach[:, 0] * side
        reach[moved, 0] = (out * side)[moved]
        reach[moved, 1] = -np.sqrt(np.maximum(length**2 - out**2, 0.0))[moved]
    if ground is not None:
        drop = np.minimum(ground - base[:, 1], 0.0)
        dragging = base[:, 1] + reach[:, 1] < ground
        reach[dragging, 0] = (side * np.sqrt(np.maximum(length**2 - drop**2, 0.0)))[dragging]
        reach[dragging, 1] = drop[dragging]
    return np.stack((base, base + reach), axis=1).reshape(-1, 2).astype(np.float32)
