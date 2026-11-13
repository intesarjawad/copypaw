"""Keypoint accuracy against labelled poses: OKS average precision and PCK."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AP10K_SIGMAS",
    "Detection",
    "GroundTruth",
    "Pck",
    "average_precision",
    "oks",
    "pck",
]

# Per-joint falloff constants from ViTPose configs/_base_/datasets/ap10k.py,
# in the same joint order as capture.pose.KEYPOINT_NAMES.
AP10K_SIGMAS = np.array(
    [
        0.025,
        0.025,
        0.026,
        0.035,
        0.035,
        0.079,
        0.072,
        0.062,
        0.079,
        0.072,
        0.062,
        0.107,
        0.087,
        0.089,
        0.107,
        0.087,
        0.089,
    ]
)
_JOINT_COUNT = len(AP10K_SIGMAS)
_OKS_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_RECALL_POINTS = np.linspace(0.0, 1.0, 101)
# A match needs the threshold itself, but an OKS of exactly 1 must still match at 0.95.
_MATCH_EPSILON = 1e-10


@dataclass(frozen=True)
class GroundTruth:
    """One labelled animal: keypoints (17, 2), visibility (17,) and its box area."""

    image: int
    category: int
    kp: NDArray[np.float64]
    visible: NDArray[np.bool_]
    area: float


@dataclass(frozen=True)
class Detection:
    """One predicted animal, scored and tagged with the species of the box it came from."""

    image: int
    category: int
    kp: NDArray[np.float64]
    score: float


@dataclass(frozen=True)
class Pck:
    """Share of labelled joints within the threshold; None where nothing is labelled."""

    overall: float | None
    per_joint: tuple[float | None, ...]


def oks(
    pred: NDArray[np.float64], gt: NDArray[np.float64], visible: NDArray[np.bool_], area: float
) -> float:
    """Return object keypoint similarity over the labelled joints.

    Each joint contributes exp(-d^2 / (2 * area * (2 * sigma)^2)); a missing
    predicted joint (NaN) contributes zero rather than poisoning the mean.
    """
    if area <= 0:
        raise ValueError("Ground-truth area must be positive")
    if not visible.any():
        return 0.0
    squared = np.sum((pred - gt) ** 2, axis=-1)[visible]
    squared = np.where(np.isfinite(squared), squared, np.inf)
    exponent = squared / (2 * area * (2 * AP10K_SIGMAS[visible]) ** 2)
    return float(np.mean(np.exp(-exponent)))


def _matched(similarity: NDArray[np.float64], threshold: float) -> NDArray[np.bool_]:
    """Greedily give each detection, best score first, its most similar free ground truth."""
    taken = np.zeros(similarity.shape[1], dtype=np.bool_)
    matched = np.zeros(similarity.shape[0], dtype=np.bool_)
    if similarity.shape[1] == 0:
        return matched
    for index, row in enumerate(similarity):
        free = np.where(taken, -1.0, row)
        best = int(np.argmax(free))
        if free[best] >= min(threshold, 1 - _MATCH_EPSILON):
            taken[best] = True
            matched[index] = True
    return matched


def _category_ap(truths: Sequence[GroundTruth], detections: Sequence[Detection]) -> float:
    by_image: dict[int, tuple[list[GroundTruth], list[Detection]]] = defaultdict(lambda: ([], []))
    for truth in truths:
        by_image[truth.image][0].append(truth)
    for detection in detections:
        by_image[detection.image][1].append(detection)
    scores: list[float] = []
    hits: list[NDArray[np.bool_]] = []
    for image_truths, image_detections in by_image.values():
        ranked = sorted(image_detections, key=lambda item: -item.score)
        if not ranked:
            continue
        similarity = np.array(
            [[oks(det.kp, gt.kp, gt.visible, gt.area) for gt in image_truths] for det in ranked]
        ).reshape(len(ranked), len(image_truths))
        scores.extend(det.score for det in ranked)
        hits.append(np.stack([_matched(similarity, float(t)) for t in _OKS_THRESHOLDS]))
    if not scores:
        return 0.0
    order = np.argsort(-np.array(scores), kind="mergesort")
    true_positive = np.concatenate(hits, axis=1)[:, order]
    precisions: list[float] = []
    for row in true_positive:
        tp = np.cumsum(row)
        fp = np.cumsum(~row)
        recall = tp / len(truths)
        precision = np.maximum.accumulate((tp / (tp + fp))[::-1])[::-1]
        found = np.searchsorted(recall, _RECALL_POINTS, side="left")
        sampled = np.where(
            found < len(precision), precision[np.minimum(found, len(precision) - 1)], 0.0
        )
        precisions.append(float(np.mean(sampled)))
    return float(np.mean(precisions))


def average_precision(truths: Sequence[GroundTruth], detections: Sequence[Detection]) -> float:
    """Return OKS AP: 101-point precision, thresholds 0.5 to 0.95, mean over species.

    This follows xtcocotools as ViTPose runs it, with no pycocotools: detections
    are matched within an image and species, and species with no labelled animal
    are left out of the mean.
    """
    species = sorted({truth.category for truth in truths})
    if not species:
        raise ValueError("No labelled poses to evaluate")
    return float(
        np.mean(
            [
                _category_ap(
                    [truth for truth in truths if truth.category == category],
                    [det for det in detections if det.category == category],
                )
                for category in species
            ]
        )
    )


def pck(
    pred: NDArray[np.float64],
    gt: NDArray[np.float64],
    visible: NDArray[np.bool_],
    box_area: NDArray[np.float64],
    threshold: float = 0.05,
) -> Pck:
    """Return PCK with a per-animal radius of threshold * sqrt(box area).

    pred and gt have shape (N, 17, 2), visible (N, 17) and box_area (N,). A
    missing predicted joint counts as a miss.
    """
    radius = threshold * np.sqrt(box_area)[:, None]
    hit = np.linalg.norm(pred - gt, axis=-1) <= radius
    counted = hit & visible
    total = int(visible.sum())
    per_joint = tuple(
        float(counted[:, joint].sum() / visible[:, joint].sum())
        if visible[:, joint].any()
        else None
        for joint in range(_JOINT_COUNT)
    )
    return Pck(float(counted.sum() / total) if total else None, per_joint)
