"""Write a drawing, and the source clip beside it, as MP4 and GIF."""

import json
import math
from dataclasses import dataclass
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from copypaw.animate.strip import StripColors, render_strip
from copypaw.capture.video import OUTPUT_FPS, read_frames

__all__ = [
    "CREDIT_HEIGHT",
    "FRAME_HEIGHT",
    "ExportResult",
    "align_facing",
    "compose_frames",
    "drawing_facing",
    "load_clip_frames",
    "side_by_side",
    "write_drawing",
    "write_footfall",
]

FRAME_HEIGHT = 512
CREDIT_HEIGHT = 24
GIF_FPS = 15
PROJECT_NAME = "Copypaw"
REPO_URL = "https://github.com/intesarjawad/copypaw"
_RGB_CHANNELS = 3
_RGBA_CHANNELS = 4
_FRAME_DIMENSIONS = 3
_STACK_DIMENSIONS = 4
_ALPHA_MAX = 255
_PALETTE_COLORS = 256
_CREDIT_FONT_PX = 14
_CREDIT_PAD_PX = 8
_TOKENS_NAME = "media_tokens.json"
_MANIFEST = "copypaw.yaml"
_HEX_LENGTH = 6
_HEX_RADIX = 16
_RGB = tuple[int, int, int]
_TIME_EPSILON_S = 1e-6
_FFMPEG: Any = import_module("imageio_ffmpeg")


@dataclass(frozen=True)
class ExportResult:
    """Paths written and the number of frames they share."""

    mp4: Path
    gif: Path
    frames: int


@dataclass(frozen=True)
class _Tokens:
    ground: _RGB
    strip: StripColors
    credit_text: _RGB


def _colour(value: object) -> _RGB:
    text = str(value).removeprefix("#")
    if len(text) != _HEX_LENGTH:
        raise ValueError(f"Media colour must be #RRGGBB, got {value!r}")
    try:
        channels = tuple(
            int(text[index : index + 2], _HEX_RADIX) for index in range(0, _HEX_LENGTH, 2)
        )
    except ValueError as error:
        raise ValueError(f"Media colour must be #RRGGBB, got {value!r}") from error
    return cast("_RGB", channels)


def _load_tokens() -> _Tokens:
    text = resources.files("copypaw").joinpath(_TOKENS_NAME).read_text(encoding="utf-8")
    data = cast("dict[str, object]", json.loads(text))
    strip = cast("dict[str, object]", data["strip"])
    return _Tokens(
        ground=_colour(data["ground"]),
        strip=StripColors(
            background=_colour(strip["background"]),
            mark=_colour(strip["mark"]),
            highlight=_colour(strip["highlight"]),
        ),
        credit_text=_colour(data["credit_text"]),
    )


def load_clip_frames(clip: Path) -> NDArray[np.uint8]:
    """Read the source clip at the rate the drawing is rendered at.

    Output frame k shows the clip frame nearest k / OUTPUT_FPS seconds, so a
    clip slower than the output rate repeats frames instead of playing fast.
    """
    frames = read_frames(clip, target_fps=OUTPUT_FPS)
    times_s = frames.t_s - frames.t_s[0]
    count = math.floor(times_s[-1] * OUTPUT_FPS + _TIME_EPSILON_S) + 1
    targets_s = np.arange(count, dtype=np.float64) / OUTPUT_FPS
    nearest = np.abs(times_s[None, :] - targets_s[:, None]).argmin(axis=1)
    return np.asarray(frames.rgb[nearest], dtype=np.uint8)


def drawing_facing(char_dir: Path) -> Literal["left", "right"]:
    """Return the facing of the frames render_character produces for a drawing."""
    manifest = Path(char_dir) / _MANIFEST
    if not manifest.is_file():
        return "right"
    data = cast(
        "dict[str, object]",
        yaml.safe_load(manifest.read_text(encoding="utf-8")) or {},
    )
    return "left" if data.get("mirrored") is True else "right"


def align_facing(
    frames: NDArray[np.uint8],
    facing_clip: Literal["left", "right"],
    facing_drawing: Literal["left", "right"],
    keep_facing: bool,
) -> NDArray[np.uint8]:
    """Flip the drawing horizontally unless it already faces the clip's way."""
    array = np.asarray(frames, dtype=np.uint8)
    if keep_facing or facing_clip == facing_drawing:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array[:, :, ::-1])


def _rgb(frame: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != _FRAME_DIMENSIONS or array.shape[2] not in (
        _RGB_CHANNELS,
        _RGBA_CHANNELS,
    ):
        raise ValueError(f"Frame must have shape (H, W, 3 or 4), got {array.shape}")
    return array[..., :_RGB_CHANNELS]


def _rgba(frame: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != _FRAME_DIMENSIONS or array.shape[2] not in (
        _RGB_CHANNELS,
        _RGBA_CHANNELS,
    ):
        raise ValueError(f"Frame must have shape (H, W, 3 or 4), got {array.shape}")
    if array.shape[2] == _RGBA_CHANNELS:
        return array
    alpha = np.full((*array.shape[:2], 1), _ALPHA_MAX, dtype=np.uint8)
    return np.concatenate((array, alpha), axis=2)


def _resize(frame: NDArray[np.uint8], height: int) -> NDArray[np.uint8]:
    """Scale to a shared height, rounding the width up to an even H.264 size."""
    source_height, source_width = frame.shape[:2]
    width = max(2, round(source_width * height / source_height))
    width += width % 2
    return np.asarray(
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), dtype=np.uint8
    )


def _composite(frame_rgba: NDArray[np.uint8], ground: _RGB) -> NDArray[np.uint8]:
    colour = np.asarray(ground, dtype=np.uint16)
    rgb = frame_rgba[..., :_RGB_CHANNELS].astype(np.uint16)
    alpha = frame_rgba[..., _RGB_CHANNELS:_RGBA_CHANNELS].astype(np.uint16)
    blended = (rgb * alpha + colour * (_ALPHA_MAX - alpha) + _ALPHA_MAX // 2) // _ALPHA_MAX
    return blended.astype(np.uint8)


def _credit_strip(width: int, tokens: _Tokens) -> NDArray[np.uint8]:
    image = Image.new("RGB", (width, CREDIT_HEIGHT), tokens.ground)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=_CREDIT_FONT_PX)
    draw.text(
        (_CREDIT_PAD_PX, (CREDIT_HEIGHT - _CREDIT_FONT_PX) // 2),
        f"{PROJECT_NAME} - {REPO_URL}",
        fill=tokens.credit_text,
        font=font,
    )
    return np.asarray(image, dtype=np.uint8)


def _strip_contacts(
    contacts: NDArray[np.bool_] | None,
    drawing_contacts: NDArray[np.bool_] | None,
    count: int,
) -> NDArray[np.bool_] | None:
    if contacts is None:
        return None
    rows = np.asarray(contacts, dtype=np.bool_)[:count]
    if len(rows) != count:
        raise ValueError(f"contacts must cover {count} frames, got {len(rows)}")
    if drawing_contacts is not None:
        extra = np.asarray(drawing_contacts, dtype=np.bool_)[:count]
        if len(extra) != count:
            raise ValueError(f"drawing contacts must cover {count} frames, got {len(extra)}")
        rows = np.hstack((rows, extra))
    return rows


def write_footfall(
    path: Path,
    contacts: NDArray[np.bool_],
    drawing_contacts: NDArray[np.bool_] | None,
    frames: int,
) -> None:
    """Write the strip's rows, one string per paw with a "1" for each planted output frame.

    Pet rows come first, then the drawing's rows when there are any.
    """
    rows = _strip_contacts(contacts, drawing_contacts, frames)
    if rows is None:
        raise ValueError("contacts are required to write a footfall record")
    pet_paws = np.asarray(contacts).shape[1]
    tracks = ["".join("1" if planted else "0" for planted in column) for column in rows.T]
    record = {"pet": tracks[:pet_paws], "drawing": tracks[pet_paws:]}
    path.write_text(json.dumps(record), encoding="utf-8")


def compose_frames(
    clip_frames: NDArray[np.uint8],
    drawing_frames_rgba: NDArray[np.uint8],
    *,
    facing_clip: Literal["left", "right"],
    facing_drawing: Literal["left", "right"],
    keep_facing: bool,
    credit: bool,
    contacts: NDArray[np.bool_] | None = None,
    drawing_contacts: NDArray[np.bool_] | None = None,
) -> NDArray[np.uint8]:
    """Pair resized clip and drawing frames over their footfall and credit strips."""
    clip = np.asarray(clip_frames, dtype=np.uint8)
    drawing = align_facing(
        np.asarray(drawing_frames_rgba, dtype=np.uint8),
        facing_clip,
        facing_drawing,
        keep_facing,
    )
    count = min(len(clip), len(drawing))
    if count == 0:
        raise ValueError("The clip and the drawing must each contain a frame")
    tokens = _load_tokens()
    rows = _strip_contacts(contacts, drawing_contacts, count)
    frames: list[NDArray[np.uint8]] = []
    for index in range(count):
        pair = np.hstack(
            (
                _resize(_rgb(clip[index]), FRAME_HEIGHT),
                _composite(_resize(drawing[index], FRAME_HEIGHT), tokens.ground),
            )
        )
        parts = [pair]
        if rows is not None:
            parts.append(render_strip(rows, index, pair.shape[1], colors=tokens.strip))
        if credit:
            parts.append(_credit_strip(pair.shape[1], tokens))
        frames.append(np.vstack(parts))
    return np.stack(frames)


def _checked_rate(fps: float) -> float:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    return fps


def _at_rate(frames: NDArray[np.uint8], fps: float, target_fps: float) -> NDArray[np.uint8]:
    if fps == target_fps:
        return frames
    duration_s = (len(frames) - 1) / fps
    count = round(duration_s * target_fps) + 1
    indexes = np.minimum(
        np.floor(np.arange(count, dtype=np.float64) * fps / target_fps + 0.5).astype(np.intp),
        len(frames) - 1,
    )
    return frames[indexes]


def _write_mp4(frames: NDArray[np.uint8], fps: float, path: Path) -> None:
    height, width = frames.shape[1:3]
    if width % 2 or height % 2:
        raise ValueError(f"H.264 needs even frame dimensions, got {width}x{height}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = _FFMPEG.write_frames(
        str(path),
        size=(width, height),
        fps=_checked_rate(fps),
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def _write_gif(frames: NDArray[np.uint8], fps: float, path: Path) -> None:
    sampled = _at_rate(frames, _checked_rate(fps), GIF_FPS)
    images = [Image.fromarray(frame, "RGB") for frame in sampled]
    palette = images[0].quantize(colors=_PALETTE_COLORS, method=Image.Quantize.MEDIANCUT)
    indexed = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=round(1000 / GIF_FPS),
        loop=0,
        disposal=2,
        optimize=False,
    )


def _write_pair(
    frames: NDArray[np.uint8], fps: float, out_mp4: Path, out_gif: Path
) -> ExportResult:
    mp4 = Path(out_mp4)
    gif = Path(out_gif)
    _write_mp4(frames, fps, mp4)
    _write_gif(frames, fps, gif)
    return ExportResult(mp4=mp4, gif=gif, frames=len(frames))


def side_by_side(  # noqa: PLR0917
    clip_frames: NDArray[np.uint8],
    drawing_frames_rgba: NDArray[np.uint8],
    fps: float,
    facing_clip: Literal["left", "right"],
    facing_drawing: Literal["left", "right"],
    keep_facing: bool,
    credit: bool,
    out_mp4: Path,
    out_gif: Path,
    *,
    contacts: NDArray[np.bool_] | None = None,
    drawing_contacts: NDArray[np.bool_] | None = None,
) -> ExportResult:
    """Write the clip and drawing side by side with their footfall strip.

    Both halves are trimmed to the shorter frame count so each pair shares a
    timestamp. Contacts add one strip row per paw, pet rows before drawing rows.
    """
    frames = compose_frames(
        clip_frames,
        drawing_frames_rgba,
        facing_clip=facing_clip,
        facing_drawing=facing_drawing,
        keep_facing=keep_facing,
        credit=credit,
        contacts=contacts,
        drawing_contacts=drawing_contacts,
    )
    return _write_pair(frames, fps, out_mp4, out_gif)


def write_drawing(
    drawing_frames_rgba: NDArray[np.uint8],
    fps: float,
    out_mp4: Path,
    out_gif: Path,
) -> ExportResult:
    """Write the drawing alone over the media ground colour."""
    frames = np.asarray(drawing_frames_rgba, dtype=np.uint8)
    if frames.ndim != _STACK_DIMENSIONS or len(frames) == 0:
        raise ValueError(f"Frames must have shape (F, H, W, 3 or 4), got {frames.shape}")
    tokens = _load_tokens()
    composed = np.stack([_composite(_rgba(frame), tokens.ground) for frame in frames])
    return _write_pair(composed, fps, out_mp4, out_gif)
