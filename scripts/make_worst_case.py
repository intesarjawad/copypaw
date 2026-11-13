"""Write the synthetic worst-case pages used by the rig segmentation tests."""

import argparse
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "samples/worst_case"

type Page = NDArray[np.uint8]
type Point = tuple[int, int]
type Box = tuple[int, int, int, int]

_PAGE_WIDTH = 600
_PAGE_HEIGHT = 800
_PAPER_BGR = (245, 245, 245)
_INK_BGR = (25, 25, 25)
_RULE_BGR = (205, 220, 240)
_RULE_STEP_PX = 40
_RULE_MARGIN_PX = 30
_RULE_WIDTH_PX = 2
_INK_WIDTH_PX = 2
_OUTLINE_PX = 4
_SKEW_DEGREES = 20.0
_CAMERA_DISTANCE_PX = _PAGE_WIDTH * 3.0
_PAGE_SHIFT_PX = -130
_COLOUR_PAPER_BGR = (205, 230, 245)
_GRADIENT_TOP = 245
_GRADIENT_BOTTOM = 210
_GLARE_CENTER = (470, 230)
_GLARE_RADIUS_PX = 220
_GLARE_STRENGTH = 55.0
_PENCIL_CONTRAST = 0.30
_FAINT_INK = round(_PAPER_BGR[0] * (1.0 - _PENCIL_CONTRAST))

_BODY_CENTER = (_PAGE_WIDTH // 2, 380)
_BODY_AXES = (170, 70)
_BASE_LEGS: tuple[Box, ...] = (
    (165, 400, 205, 620),
    (240, 400, 280, 640),
    (345, 400, 385, 630),
    (415, 400, 455, 610),
)
_STICK_LEG_POINTS: tuple[tuple[Point, Point], ...] = (
    ((185, 445), (195, 640)),
    ((260, 445), (270, 640)),
    ((365, 445), (375, 640)),
    ((435, 445), (445, 640)),
)
_TAIL: tuple[Point, ...] = ((125, 390), (72, 500), (60, 630), (92, 632), (100, 505), (160, 400))
_NECK: tuple[Point, ...] = ((440, 400), (500, 190), (535, 190), (500, 410))
_SCRIBBLE_Y = tuple(range(335, 430, 10))


def _blank(paper: tuple[int, int, int] = _PAPER_BGR) -> Page:
    return np.full((_PAGE_HEIGHT, _PAGE_WIDTH, 3), paper, np.uint8)


def _silhouette(
    *,
    center: Point = _BODY_CENTER,
    axes: Point = _BODY_AXES,
    legs: Iterable[Box] = (),
    additions: Iterable[Sequence[Point]] = (),
) -> NDArray[np.uint8]:
    mask = np.zeros((_PAGE_HEIGHT, _PAGE_WIDTH), np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    for left, top, right, bottom in legs:
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    for polygon in additions:
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
    return mask


def _draw(
    silhouette: NDArray[np.uint8],
    background: Page,
    *,
    fill: tuple[int, int, int] | None = None,
    ink: tuple[int, int, int] = _INK_BGR,
    details: Callable[[Page], None] | None = None,
) -> Page:
    page = background.copy()
    # An opaque drawing covers paper ruling; gradient and glare keep their local shade.
    if fill is not None:
        page[silhouette > 0] = fill
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(page, contours, -1, ink, _OUTLINE_PX)
    if details is not None:
        details(page)
    return page


def _ruled(*, grid: bool) -> Page:
    page = _blank()
    right = _PAGE_WIDTH - 1 - _RULE_MARGIN_PX
    bottom = _PAGE_HEIGHT - 1 - _RULE_MARGIN_PX
    for y in range(_RULE_MARGIN_PX + _RULE_STEP_PX, _PAGE_HEIGHT - _RULE_MARGIN_PX, _RULE_STEP_PX):
        cv2.line(page, (_RULE_MARGIN_PX, y), (right, y), _RULE_BGR, _RULE_WIDTH_PX)
    if grid:
        for x in range(
            _RULE_MARGIN_PX + _RULE_STEP_PX, _PAGE_WIDTH - _RULE_MARGIN_PX, _RULE_STEP_PX
        ):
            cv2.line(page, (x, _RULE_MARGIN_PX), (x, bottom), _RULE_BGR, _RULE_WIDTH_PX)
    return page


def _gradient() -> Page:
    shade = np.linspace(_GRADIENT_TOP, _GRADIENT_BOTTOM, _PAGE_HEIGHT).astype(np.uint8)
    page = np.empty((_PAGE_HEIGHT, _PAGE_WIDTH, 3), np.uint8)
    page[:] = shade[:, None, None]
    return page


def _glare() -> Page:
    yy, xx = np.mgrid[0:_PAGE_HEIGHT, 0:_PAGE_WIDTH]
    distance = np.hypot(xx - _GLARE_CENTER[0], yy - _GLARE_CENTER[1])
    glow = np.clip(
        _GLARE_STRENGTH * (1.0 - distance / _GLARE_RADIUS_PX), 0.0, _GLARE_STRENGTH
    ).astype(np.int16)
    page = np.full((_PAGE_HEIGHT, _PAGE_WIDTH, 3), _PAPER_BGR[0], np.int16)
    return np.clip(page + glow[..., None], 0, 255).astype(np.uint8)


def _project(x: float, y: float) -> tuple[float, float]:
    angle = math.radians(_SKEW_DEGREES)
    x_offset = x - _PAGE_WIDTH / 2
    y_offset = y - _PAGE_HEIGHT / 2
    depth = _CAMERA_DISTANCE_PX + x_offset * math.sin(angle)
    return (
        _PAGE_WIDTH / 2 + _CAMERA_DISTANCE_PX * x_offset * math.cos(angle) / depth,
        _PAGE_HEIGHT / 2 + _CAMERA_DISTANCE_PX * y_offset / depth,
    )


def _perspective(page: Page) -> Page:
    corners = (
        (0.0, 0.0),
        (float(_PAGE_WIDTH), 0.0),
        (float(_PAGE_WIDTH), float(_PAGE_HEIGHT)),
        (0.0, float(_PAGE_HEIGHT)),
    )
    source = np.asarray(corners, dtype=np.float32)
    target = np.asarray([_project(x, y) for x, y in corners], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(page, matrix, (_PAGE_WIDTH, _PAGE_HEIGHT), borderValue=_PAPER_BGR)
    return np.asarray(warped, dtype=np.uint8)


def _shift_left(page: Page, dx: int) -> Page:
    matrix = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, 0.0]], dtype=np.float32)
    shifted = cv2.warpAffine(page, matrix, (_PAGE_WIDTH, _PAGE_HEIGHT), borderValue=_PAPER_BGR)
    return np.asarray(shifted, dtype=np.uint8)


def _stick_legs(page: Page) -> None:
    for start, end in _STICK_LEG_POINTS:
        cv2.line(page, start, end, _INK_BGR, _INK_WIDTH_PX)


def _scribble(page: Page) -> None:
    points = [(200 if index % 2 == 0 else 400, y) for index, y in enumerate(_SCRIBBLE_Y)]
    cv2.polylines(page, [np.asarray(points, np.int32)], False, _INK_BGR, _INK_WIDTH_PX)


def build_pages() -> dict[str, Page]:
    """Return every generated page keyed by its file name."""
    base = _silhouette(legs=_BASE_LEGS)
    plain = _draw(base, _blank())
    stick = _draw(_silhouette(), _blank(), details=_stick_legs)
    scribble = _draw(base, _blank(), details=_scribble)
    return {
        "plain.png": plain,
        "lined_paper.png": _draw(base, _ruled(grid=False), fill=_PAPER_BGR),
        "graph_paper.png": _draw(base, _ruled(grid=True), fill=_PAPER_BGR),
        "coloured_paper.png": _draw(base, _blank(_COLOUR_PAPER_BGR), fill=_COLOUR_PAPER_BGR),
        "phone_shadow_gradient.png": _draw(base, _gradient()),
        "perspective_skew.png": _perspective(plain),
        "glare_blob.png": _draw(base, _glare()),
        "faint_pencil.png": _draw(base, _blank(), ink=(_FAINT_INK, _FAINT_INK, _FAINT_INK)),
        "edge_touching.png": _shift_left(plain, _PAGE_SHIFT_PX),
        "two_creatures.png": np.minimum(
            _draw(
                _silhouette(
                    center=(170, 380),
                    axes=(110, 55),
                    legs=(
                        (90, 405, 122, 620),
                        (140, 405, 172, 630),
                        (218, 405, 250, 615),
                    ),
                ),
                _blank(),
            ),
            _draw(
                _silhouette(
                    center=(445, 380),
                    axes=(110, 55),
                    legs=(
                        (360, 405, 392, 620),
                        (410, 405, 442, 630),
                        (488, 405, 520, 615),
                    ),
                ),
                _blank(),
            ),
        ),
        "stick_legs.png": stick,
        "two_legs.png": _draw(
            _silhouette(legs=((220, 400, 260, 620), (360, 400, 400, 640))), _blank()
        ),
        "four_legs_row.png": _draw(
            _silhouette(
                axes=(190, 70),
                legs=(
                    (140, 400, 174, 620),
                    (240, 400, 274, 640),
                    (340, 400, 374, 630),
                    (440, 400, 474, 610),
                ),
            ),
            _blank(),
        ),
        "tail_on_ground.png": _draw(_silhouette(legs=_BASE_LEGS, additions=(_TAIL,)), _blank()),
        "long_neck.png": _draw(_silhouette(legs=_BASE_LEGS, additions=(_NECK,)), _blank()),
        "short_legs.png": _draw(
            _silhouette(
                center=(300, 420),
                legs=(
                    (165, 470, 205, 530),
                    (240, 470, 280, 540),
                    (345, 470, 385, 535),
                    (415, 470, 455, 525),
                ),
            ),
            _blank(),
        ),
        "scribble_body.png": scribble,
    }


def _write(path: Path, page: Page) -> None:
    success, encoded = cv2.imencode(".png", page)
    if not success:
        raise OSError(f"Cannot encode PNG: {path}")
    path.write_bytes(np.asarray(encoded, dtype=np.uint8).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT, help="Directory for the pages.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, page in build_pages().items():
        _write(args.out / name, page)


if __name__ == "__main__":
    main()
