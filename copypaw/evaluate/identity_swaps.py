"""Temporal limb-identity flips between predicted and labelled leg chains."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["LIMB_PAIRS", "Flips", "align_chains", "count_flips"]

# Left and right chains of the front and back legs, as shoulder/hip, elbow/knee, paw.
LIMB_PAIRS = (((5, 6, 7), (8, 9, 10)), ((11, 12, 13), (14, 15, 16)))
_JOINT_COUNT = 17


@dataclass(frozen=True)
class Flips:
    """Identity changes between consecutive labelled frames, out of the pairs that could change."""

    flips: int
    opportunities: int

    def __add__(self, other: "Flips") -> "Flips":
        """Pool two counts."""
        return Flips(self.flips + other.flips, self.opportunities + other.opportunities)

    @property
    def rate(self) -> float | None:
        """Flips per opportunity, or None when no pair was ever labelled twice in a row."""
        return self.flips / self.opportunities if self.opportunities else None


def _swap_order(pair: tuple[tuple[int, ...], tuple[int, ...]]) -> NDArray[np.intp]:
    order = np.arange(_JOINT_COUNT)
    first, second = (np.array(chain) for chain in pair)
    order[first], order[second] = second, first
    return order


def _costs(
    pred: NDArray[np.float32] | NDArray[np.float64],
    gt: NDArray[np.float64],
    visible: NDArray[np.bool_],
    pair: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Per-frame cost with the chains kept, then exchanged, and where either has evidence."""
    joints = np.concatenate(pair)
    exchanged = _swap_order(pair)[joints]
    gaps = (
        np.linalg.norm(pred[:, source] - gt[:, joints], axis=-1) for source in (joints, exchanged)
    )

    def cost(gap: NDArray[np.float64]) -> NDArray[np.float64]:
        # A missing prediction is no evidence for either assignment.
        return np.sum(np.where(visible[:, joints] & np.isfinite(gap), gap, 0.0), axis=1)

    kept_gap, exchanged_gap = gaps
    evidence = np.any(visible[:, joints] & np.isfinite(kept_gap), axis=1)
    return cost(kept_gap), cost(exchanged_gap), evidence


def count_flips(
    pred: NDArray[np.float32] | NDArray[np.float64],
    gt: NDArray[np.float64],
    visible: NDArray[np.bool_],
) -> Flips:
    """Count assignment changes for one animal over consecutive frames.

    pred and gt have shape (F, 17, 2) and visible (F, 17). In each limb pair a
    frame is swapped when matching the predicted chains to the labelled ones
    crosswise is strictly cheaper than keeping them. A flip is a change of that
    state between frames t - 1 and t, counted only when each labelled chain has
    a visible joint in both frames and both frames carry a finite prediction.
    """
    total = Flips(0, 0)
    for pair in LIMB_PAIRS:
        kept, exchanged, evidence = _costs(pred, gt, visible, pair)
        swapped = exchanged < kept
        first, second = (list(chain) for chain in pair)
        seen = visible[:, first].any(axis=1) & visible[:, second].any(axis=1)
        comparable = seen[1:] & seen[:-1] & evidence[1:] & evidence[:-1]
        total += Flips(
            int(np.count_nonzero(comparable & (swapped[1:] != swapped[:-1]))),
            int(np.count_nonzero(comparable)),
        )
    return total


def align_chains(
    kp: NDArray[np.float32] | NDArray[np.float64],
    scores: NDArray[np.float32],
    gt: NDArray[np.float64],
    visible: NDArray[np.bool_],
) -> tuple[NDArray[np.float32] | NDArray[np.float64], NDArray[np.float32]]:
    """Name each resolved chain pair after the labels it follows best over the whole clip.

    Identity resolution keeps chains consistent but cannot know left from right,
    so its Near/Far labels are exchanged as a whole where that fits the labels
    better. Inputs have shapes (F, 17, 2), (F, 17), (F, 17, 2) and (F, 17).
    """
    order = np.arange(_JOINT_COUNT)
    for pair in LIMB_PAIRS:
        kept, exchanged, _ = _costs(kp, gt, visible, pair)
        if exchanged.sum() < kept.sum():
            order = order[_swap_order(pair)]
    return kp[:, order], scores[:, order]
