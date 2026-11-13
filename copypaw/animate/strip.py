"""Draw a footfall strip: one row per paw and one column per frame."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ROW_HEIGHT", "StripColors", "render_strip"]

ROW_HEIGHT = 8
_ROW_GAP = 1
_PIXEL_CHANNELS = 3
_CONTACT_DIMENSIONS = 2


@dataclass(frozen=True)
class StripColors:
    """Background, planted-paw and current-frame colours."""

    background: tuple[int, int, int]
    mark: tuple[int, int, int]
    highlight: tuple[int, int, int]


def render_strip(
    contacts: NDArray[np.bool_],
    current: int,
    width: int,
    *,
    colors: StripColors,
    row_height: int = ROW_HEIGHT,
) -> NDArray[np.uint8]:
    """Return an RGB strip whose columns are frames and rows are paws.

    Each planted frame fills its paw's row, and the current frame's whole
    column is highlighted. Width is divided between frames as evenly as
    integer columns allow, so the strip always spans the paired frames.
    """
    track = np.asarray(contacts, dtype=np.bool_)
    if track.ndim != _CONTACT_DIMENSIONS or 0 in track.shape:
        raise ValueError(f"contacts must have shape (F, P) with frames, got {track.shape}")
    frames = len(track)
    if not 0 <= current < frames:
        raise ValueError(f"current must be in [0, {frames}), got {current}")
    if width < frames:
        raise ValueError(f"width must leave one column per frame, got {width} for {frames}")
    if row_height <= _ROW_GAP:
        raise ValueError(f"row_height must exceed {_ROW_GAP}, got {row_height}")

    height = track.shape[1] * row_height
    image = np.empty((height, width, _PIXEL_CHANNELS), dtype=np.uint8)
    image[:] = colors.background
    edges = np.arange(frames + 1, dtype=np.intp) * width // frames
    # Highlight first so a paw planted on the current frame keeps its mark.
    image[:, edges[current] : edges[current + 1]] = colors.highlight
    for frame in np.flatnonzero(track.any(axis=1)):
        columns = slice(edges[frame], edges[frame + 1])
        for paw in np.flatnonzero(track[frame]):
            top = paw * row_height
            image[top : top + row_height - _ROW_GAP, columns] = colors.mark
    return image
