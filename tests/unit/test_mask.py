import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.rig.mask import MaskError, segment

PAGE_BGR = 245
INK_BGR = 20


def _page(height: int = 200, width: int = 300) -> NDArray[np.uint8]:
    return np.full((height, width, 3), PAGE_BGR, dtype=np.uint8)


def _component_count(mask: NDArray[np.uint8]) -> int:
    count, _, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return count - 1


def test_closed_outline_becomes_one_filled_region() -> None:
    page = _page()
    cv2.rectangle(page, (40, 50), (260, 150), (INK_BGR,) * 3, 3)

    mask = segment(page)

    assert mask.shape == page.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert _component_count(mask) == 1
    # The 220 x 100 outline encloses 37% of the page; dilation moves the edge outward.
    assert 0.30 <= np.count_nonzero(mask) / mask.size <= 0.45
    assert mask[100, 150] != 0
    assert mask[10, 10] == 0


@pytest.mark.parametrize("stroke_px", [2, 5, 9])
def test_mask_reaches_the_outer_edge_of_the_stroke(stroke_px: int) -> None:
    page = _page()
    cv2.rectangle(page, (40, 50), (260, 150), (INK_BGR,) * 3, stroke_px)

    mask = segment(page)

    ink = page[..., 0] < PAGE_BGR // 2
    assert np.all(mask[ink] != 0), "the mask cuts into the stroke"
    # The rectangle's outer edge is half a stroke outside its corner coordinates.
    reach = stroke_px // 2 + 3
    outside = np.ones(mask.shape, dtype=np.bool_)
    outside[50 - reach : 151 + reach, 40 - reach : 261 + reach] = False
    assert not np.any(mask[outside]), "the mask spills more than 3 px past the stroke"


def test_ruled_lines_do_not_widen_the_mask() -> None:
    page = _page()
    for y in range(10, 200, 20):
        cv2.line(page, (0, y), (299, y), (150,) * 3, 1)
    cv2.rectangle(page, (40, 50), (260, 150), (PAGE_BGR,) * 3, -1)
    cv2.rectangle(page, (40, 50), (260, 150), (INK_BGR,) * 3, 5)

    mask = segment(page)

    columns = np.flatnonzero(mask[100] != 0)
    assert 34 <= columns[0] <= 38
    assert 262 <= columns[-1] <= 266


def test_single_line_legs_join_the_mask_at_a_minimum_width() -> None:
    page = _page()
    cv2.rectangle(page, (40, 50), (260, 110), (INK_BGR,) * 3, 3)
    for x in (80, 150, 220):
        cv2.line(page, (x, 116), (x, 180), (INK_BGR,) * 3, 1)

    mask = segment(page)

    assert _component_count(mask) == 1
    for x in (80, 150, 220):
        assert np.count_nonzero(mask[170, x - 20 : x + 20]) >= 15
        assert mask[180, x] != 0
    assert mask[170, 115] == 0


def test_flat_ruling_below_the_drawing_is_not_a_leg() -> None:
    page = _page()
    cv2.rectangle(page, (40, 50), (260, 110), (INK_BGR,) * 3, 3)
    cv2.line(page, (60, 120), (240, 120), (INK_BGR,) * 3, 1)

    mask = segment(page)

    assert not mask[119:122].any()


def test_internal_hole_is_filled() -> None:
    page = _page()
    cv2.rectangle(page, (40, 50), (260, 150), (INK_BGR,) * 3, 3)
    cv2.circle(page, (150, 100), 25, (INK_BGR,) * 3, 2)

    mask = segment(page)

    assert _component_count(mask) == 1
    assert mask[100, 150] != 0


def test_six_pixel_gap_is_closed_by_a_wider_kernel() -> None:
    page = _page()
    cv2.circle(page, (150, 100), 35, (INK_BGR,) * 3, 2)
    cv2.rectangle(page, (147, 60), (152, 70), (PAGE_BGR,) * 3, -1)

    mask = segment(page)

    assert _component_count(mask) == 1
    # A 35 px circle plus the outline covers about 8% of the page.
    assert 0.05 <= np.count_nonzero(mask) / mask.size <= 0.15


def test_two_creatures_are_rejected() -> None:
    page = _page()
    cv2.rectangle(page, (20, 50), (120, 150), (INK_BGR,) * 3, 3)
    cv2.rectangle(page, (180, 50), (280, 150), (INK_BGR,) * 3, 3)

    with pytest.raises(MaskError) as error:
        segment(page)

    assert error.value.kind == "multiple_creatures"
    assert error.value.what == "The page contains more than one drawing."
    assert "one pet per page" in error.value.next


def test_blank_page_is_too_small() -> None:
    with pytest.raises(MaskError) as error:
        segment(_page())

    assert error.value.kind == "too_small"
    assert error.value.what == "The drawing covers too little of the page."
    assert "larger pet" in error.value.next


def test_page_filled_to_the_edges_is_too_large() -> None:
    page = _page()
    cv2.rectangle(page, (1, 1), (298, 198), (INK_BGR,) * 3, 2)

    with pytest.raises(MaskError) as error:
        segment(page)

    assert error.value.kind == "too_large"
    assert error.value.what == "The drawing covers too much of the page."
    assert "clear margin" in error.value.next


def test_longest_side_is_scaled_to_one_thousand_pixels() -> None:
    page = _page(800, 1200)
    cv2.rectangle(page, (100, 100), (1100, 700), (INK_BGR,) * 3, 3)

    mask = segment(page)

    assert mask.shape == (667, 1000)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_invalid_images_are_rejected(image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match=r"uint8 BGR|positive"):
        segment(image)
