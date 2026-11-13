"""Segment a single drawing from a photographed or scanned page."""

from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy import ndimage

from copypaw.errors import CopypawError

__all__ = ["Mask", "MaskError", "segment"]

type Mask = NDArray[np.uint8]

_BGR_DIMENSIONS = 3
_BGR_CHANNELS = 3
_MAX_SIDE_PX = 1000
_ADAPTIVE_BLOCK_PX = 115
_ADAPTIVE_C = 8
_MORPH_ITERATIONS = 2
_DILATE_SIZE_PX = 3
_SEED_STEP_PX = 10
_MIN_COVERAGE = 0.05
_MAX_COVERAGE = 0.90
_MIN_COMPONENT_FRACTION = 0.01
_CLOSE_SIZES_PX = (3, 5, 7, 9)
# Widest stroke restored around the enclosed region, at the working size.
_MAX_STROKE_REACH_PX = 24
# A ring of pixels around the enclosed region is stroke while this share of it is ink.
_STROKE_RING_SHARE = 0.5
# A stroke outside the outline is a leg when it hangs this close to the drawing, is at least
# this share of the drawing's height long, and starts in its lower half.
_LOOSE_REACH_PX = 24
_LOOSE_MIN_LENGTH_SHARE = 0.5
# Widest a leg of one pen line is thickened to, so the mesh has width to bend.
_LOOSE_WIDTH_PX = 21

type MaskErrorKind = Literal["too_small", "too_large", "multiple_creatures"]


class MaskError(CopypawError):
    """The page does not contain exactly one usable drawing."""

    codes = ("mask_too_small", "mask_too_large", "mask_multiple_creatures")

    def __init__(self, kind: MaskErrorKind) -> None:
        """Record the failure kind, which selects the message."""
        self.kind = kind
        super().__init__(f"mask_{kind}")


def _scaled(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= _MAX_SIDE_PX:
        return bgr
    scale = _MAX_SIDE_PX / longest
    size = (max(1, round(scale * width)), max(1, round(scale * height)))
    return np.asarray(cv2.resize(bgr, size), dtype=np.uint8)


def _ink(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    gray = np.min(bgr, axis=2)
    light = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        _ADAPTIVE_BLOCK_PX,
        _ADAPTIVE_C,
    )
    return np.asarray(cv2.bitwise_not(light), dtype=np.uint8)


def _enclosed(ink: NDArray[np.uint8], close_size_px: int) -> NDArray[np.uint8]:
    outline = ink
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size_px, close_size_px))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_DILATE_SIZE_PX, _DILATE_SIZE_PX))
    outline = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, close_kernel, iterations=_MORPH_ITERATIONS)
    outline = cv2.morphologyEx(
        outline, cv2.MORPH_DILATE, dilate_kernel, iterations=_MORPH_ITERATIONS
    )

    height, width = outline.shape
    blocking = np.zeros((height + 2, width + 2), np.uint8)
    blocking[1:-1, 1:-1] = outline
    # Paper ruling can touch the stroke, so only whitespace inside the stroke counts.
    enclosed = np.asarray(cv2.bitwise_not(outline), dtype=np.uint8)
    for x in range(0, width - 1, _SEED_STEP_PX):
        cv2.floodFill(enclosed, blocking, (x, 0), 0)
        cv2.floodFill(enclosed, blocking, (x, height - 1), 0)
    for y in range(0, height - 1, _SEED_STEP_PX):
        cv2.floodFill(enclosed, blocking, (0, y), 0)
        cv2.floodFill(enclosed, blocking, (width - 1, y), 0)
    enclosed[0, :] = 0
    enclosed[-1, :] = 0
    enclosed[:, 0] = 0
    enclosed[:, -1] = 0
    return enclosed


def _fill_largest(contours: Sequence[MatLike], shape: tuple[int, int]) -> Mask:
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros(shape, np.uint8)
    cv2.fillPoly(filled, [largest], 1)
    filled = ndimage.binary_fill_holes(filled)
    return (filled.astype(np.uint8) * 255).astype(np.uint8)


def _stroke_reach_px(filled: Mask, ink: NDArray[np.uint8]) -> int:
    """Distance from the enclosed region to the outer edge of the stroke around it.

    The enclosed region stops short of the stroke, so the rings of pixels
    around it are paper, then ink, then paper again. The stroke ends at the
    last ring that is mostly ink; paper ruling crosses the rings beyond it but
    never fills one.
    """
    distance = cv2.distanceTransform(cv2.bitwise_not(filled), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    rings = np.ceil(distance).astype(np.intp)
    near = (rings >= 1) & (rings <= _MAX_STROKE_REACH_PX)
    total = np.bincount(rings[near], minlength=_MAX_STROKE_REACH_PX + 1)
    inked = np.bincount(rings[near & (ink > 0)], minlength=_MAX_STROKE_REACH_PX + 1)
    stroke = np.flatnonzero(inked[1:] >= _STROKE_RING_SHARE * np.maximum(total[1:], 1))
    return int(stroke[-1]) + 1 if stroke.size else 0


def _with_stroke(filled: Mask, ink: NDArray[np.uint8]) -> Mask:
    reach = _stroke_reach_px(filled, ink)
    size = (2 * reach + 1, 2 * reach + 1)
    grown = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size))
    # A round kernel cuts the tip off a sharp corner, so ink within the square reach is kept.
    corners = cv2.bitwise_and(ink, cv2.dilate(filled, np.ones(size, np.uint8)))
    return np.asarray(cv2.bitwise_or(grown, corners), dtype=np.uint8)


def _bridge(mask: Mask, tip: tuple[int, int]) -> Mask:
    """A one-pixel line from a stroke's tip to the nearest pixel of the drawing."""
    rows, columns = np.nonzero(mask)
    nearest = int(np.argmin((columns - tip[0]) ** 2 + (rows - tip[1]) ** 2))
    dx = int(columns[nearest]) - tip[0]
    dy = int(rows[nearest]) - tip[1]
    steps = max(abs(dx), abs(dy), 1)
    along = np.arange(steps + 1)
    line = np.zeros_like(mask)
    line[
        np.floor(tip[1] + dy * along / steps + 0.5).astype(np.intp),
        np.floor(tip[0] + dx * along / steps + 0.5).astype(np.intp),
    ] = 255
    return line


def _with_legs(mask: Mask, ink: NDArray[np.uint8]) -> Mask:
    """Add single-line legs hanging off the drawing, thickened to a minimum width.

    The enclosed region leaves out strokes that enclose nothing, so a stick leg
    is found as ink outside the mask. Ruling is rejected by shape: it runs
    across the page or lies flat.
    """
    outside = np.where((ink > 0) & (mask == 0), 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(outside, connectivity=8)
    page_height, page_width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    body_top, body_height = int(rows[0]), int(rows[-1] - rows[0] + 1)
    legs = np.zeros_like(mask)
    for label in range(1, count):
        left, top, width, height, _ = (int(v) for v in stats[label])
        hangs = (
            height > width
            and height >= _LOOSE_MIN_LENGTH_SHARE * body_height
            and top >= body_top + body_height // 2
            and left > 0
            and top > 0
            and left + width < page_width
            and top + height < page_height
        )
        if not hangs:
            continue
        stroke = np.where(labels == label, 255, 0).astype(np.uint8)
        tip_row = top
        tip = (int(np.flatnonzero(stroke[tip_row])[0]), tip_row)
        bridge = _bridge(mask, tip)
        reach = int(np.count_nonzero(bridge))
        if reach > _LOOSE_REACH_PX + 1:
            continue
        legs = np.asarray(cv2.bitwise_or(legs, cv2.bitwise_or(stroke, bridge)), dtype=np.uint8)
    size = (_LOOSE_WIDTH_PX, _LOOSE_WIDTH_PX)
    grown = cv2.dilate(legs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size))
    return np.asarray(cv2.bitwise_or(mask, grown), dtype=np.uint8)


def segment(bgr: NDArray[np.uint8]) -> Mask:
    """Return the filled drawing on a light page as a 0/255 mask.

    The longest side is capped at 1000 px. The mask reaches the outer edge of
    the drawn stroke, so the outline is part of the drawing. A broken outline
    is retried with wider closing kernels; a page with no usable single
    creature raises MaskError with a fix for the drawing.
    """
    if bgr.ndim != _BGR_DIMENSIONS or bgr.shape[2] != _BGR_CHANNELS or bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image with shape (H, W, 3)")
    height, width = bgr.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Image width and height must be positive")

    page = _scaled(bgr)
    ink = _ink(page)
    minimum_area = page.shape[0] * page.shape[1] * _MIN_COMPONENT_FRACTION
    failure: MaskErrorKind = "too_small"
    for close_size_px in _CLOSE_SIZES_PX:
        enclosed = _enclosed(ink, close_size_px)
        contours, _ = cv2.findContours(enclosed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = tuple(
            contour for contour in contours if cv2.contourArea(contour) >= minimum_area
        )
        if not significant:
            continue
        mask = _fill_largest(significant, page.shape[:2])
        mask = _with_stroke(mask, ink)
        mask = _with_legs(mask, ink)
        coverage = np.count_nonzero(mask) / mask.size
        if coverage < _MIN_COVERAGE:
            continue
        if coverage > _MAX_COVERAGE:
            raise MaskError("too_large")
        if len(significant) > 1:
            if close_size_px == _CLOSE_SIZES_PX[0]:
                raise MaskError("multiple_creatures")
            failure = "multiple_creatures"
            continue
        return mask
    raise MaskError(failure)
