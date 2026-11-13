import numpy as np
import pytest

from copypaw.animate.strip import StripColors, render_strip

BACKGROUND = (10, 20, 30)
MARK = (200, 100, 50)
HIGHLIGHT = (60, 70, 80)
COLORS = StripColors(background=BACKGROUND, mark=MARK, highlight=HIGHLIGHT)
CONTACTS = np.array(
    [
        [True, False, False, False],
        [False, True, False, False],
        [True, True, False, False],
        [False, False, True, False],
    ],
    dtype=np.bool_,
)
WIDTH = 8
COLUMNS = (0, 2, 4, 6, 8)


def test_planted_frames_are_the_only_marked_cells() -> None:
    image = render_strip(CONTACTS, 3, WIDTH, colors=COLORS, row_height=2)

    assert image.shape == (8, WIDTH, 3)
    for frame in range(3):
        columns = slice(COLUMNS[frame], COLUMNS[frame + 1])
        for paw in range(4):
            block = image[paw * 2, columns]
            expected = MARK if CONTACTS[frame, paw] else BACKGROUND
            np.testing.assert_array_equal(block, np.tile(expected, (2, 1)))
            np.testing.assert_array_equal(image[paw * 2 + 1, columns], np.tile(BACKGROUND, (2, 1)))


@pytest.mark.parametrize("current", range(4))
def test_highlight_column_is_the_current_frame(current: int) -> None:
    image = render_strip(CONTACTS, current, WIDTH, colors=COLORS, row_height=2)

    highlighted = np.flatnonzero(np.all(image[7] == HIGHLIGHT, axis=1))
    assert highlighted.tolist() == list(range(COLUMNS[current], COLUMNS[current + 1]))


def test_strip_rejects_geometry_it_cannot_draw() -> None:
    with pytest.raises(ValueError, match="current"):
        render_strip(CONTACTS, 4, WIDTH, colors=COLORS)
    with pytest.raises(ValueError, match="width"):
        render_strip(CONTACTS, 0, 3, colors=COLORS)
