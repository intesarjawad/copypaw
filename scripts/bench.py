"""Measure capture speed on the sample clip and print or paste a Markdown table."""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from copypaw.animate.render import render_character, resample_motion
from copypaw.animate.strip import StripColors, render_strip
from copypaw.capture.detect import track_animal
from copypaw.capture.identity import resolve_identity
from copypaw.capture.pipeline import capture_motion, load_stages
from copypaw.capture.smoothing import clean
from copypaw.capture.video import read_frames
from copypaw.models import MODELS, REGISTRY, Mode
from copypaw.motion.motion_file import MotionFile

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "copypaw/selftest/dog_walk.mp4"
DRAWING = ROOT / "copypaw/selftest/dragon.png"
README = ROOT / "README.md"
MOTIONS = ROOT / "motions"
REVIEW = ROOT / "build/motions"
DRAGON = ROOT / "samples/drawings/dragon"
START_MARKER = "<!-- bench:start -->"
END_MARKER = "<!-- bench:end -->"
MS_PER_S = 1000
SHEET_STEP = 5
_TILE_PX = 128
_SHEET_COLUMNS = 6
_STRIP_ROW_PX = 12
_STRIP_GAP_PX = 8
_WHITE = 255
_ALPHA_CHANNEL = 3
_RGB_CHANNELS = 3
_STRIP_COLORS = StripColors(
    background=(242, 242, 242), mark=(64, 64, 64), highlight=(242, 242, 242)
)


@dataclass(frozen=True)
class Row:
    """A measurement label and its value, or None when its inputs are absent."""

    label: str
    value: float | None
    unit: str


def format_table(rows: Sequence[Row]) -> str:
    """Render rows as a two-column Markdown table, with n/a for missing values."""
    lines = ["| Measurement | Result |", "| --- | --- |"]
    lines.extend(
        f"| {row.label} | {'n/a' if row.value is None else f'{row.value:.1f} {row.unit}'} |"
        for row in rows
    )
    return "\n".join(lines)


def paste_table(readme: str, table: str) -> str:
    """Replace the text between the benchmark markers."""
    start = readme.index(START_MARKER) + len(START_MARKER)
    return f"{readme[:start]}\n{table}\n{readme[readme.index(END_MARKER) :]}"


def _timed[T](work: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = work()
    return result, time.perf_counter() - started


def _capture_seconds(flip_test: bool, models: Mapping[str, Path]) -> float:
    """Time the capture pipeline, including decoding and model loading."""
    with tempfile.TemporaryDirectory() as out_dir:
        started = time.perf_counter()
        detector, pose = load_stages(Mode.BALANCED, flip_test, models)
        capture_motion(
            read_frames(SAMPLE), SAMPLE.name, Path(out_dir), detector=detector, pose=pose
        )
        return time.perf_counter() - started


def pose_ms_per_frame(mode: Mode, flip_test: bool, models: Mapping[str, Path]) -> float:
    """Time the pose stage alone on the sample clip, excluding opening the model."""
    if not SAMPLE.is_file():
        raise FileNotFoundError(f"Missing sample clip: {SAMPLE}")
    frames = read_frames(SAMPLE)
    detector, pose = load_stages(mode, flip_test, models)
    track = track_animal(frames, detector)
    _, seconds = _timed(lambda: pose(frames, track))
    return seconds * MS_PER_S / len(frames.rgb)


def _tile(frame_rgba: NDArray[np.uint8], index: int) -> NDArray[np.uint8]:
    """Flatten one frame onto white, shrink it and write its frame number in the corner."""
    alpha = frame_rgba[..., _ALPHA_CHANNEL:].astype(np.uint16)
    rgb = frame_rgba[..., :_RGB_CHANNELS].astype(np.uint16)
    flat = ((rgb * alpha + _WHITE * (_WHITE - alpha) + _WHITE // 2) // _WHITE).astype(np.uint8)
    small = cv2.resize(flat, (_TILE_PX, _TILE_PX), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(np.asarray(small, dtype=np.uint8), "RGB")
    ImageDraw.Draw(image).text((3, 3), str(index), fill=(0, 0, 0))
    return np.asarray(image, dtype=np.uint8)


def compose_sheet(
    frames_rgba: NDArray[np.uint8], contacts: NDArray[np.bool_], step: int = SHEET_STEP
) -> NDArray[np.uint8]:
    """Tile every step-th rendered frame, with the footfall strip of all frames beneath."""
    if len(contacts) != len(frames_rgba):
        raise ValueError(f"Got {len(contacts)} contact rows for {len(frames_rgba)} frames")
    tiles = [_tile(frames_rgba[index], index) for index in range(0, len(frames_rgba), step)]
    rows = -(-len(tiles) // _SHEET_COLUMNS)
    blank = np.full((_TILE_PX, _TILE_PX, _RGB_CHANNELS), _WHITE, dtype=np.uint8)
    tiles.extend([blank] * (rows * _SHEET_COLUMNS - len(tiles)))
    grid = np.vstack(
        [np.hstack(tiles[row * _SHEET_COLUMNS : (row + 1) * _SHEET_COLUMNS]) for row in range(rows)]
    )
    strip = render_strip(contacts, 0, grid.shape[1], colors=_STRIP_COLORS, row_height=_STRIP_ROW_PX)
    gap = np.full((_STRIP_GAP_PX, grid.shape[1], _RGB_CHANNELS), _WHITE, dtype=np.uint8)
    return np.vstack((grid, gap, strip))


def contact_sheet(motion_dir: Path, drawing: Path, review_dir: Path) -> float:
    """Render the drawing with one motion, write its sheet and return the render seconds."""
    if not (drawing / "char_cfg.yaml").is_file():
        raise FileNotFoundError(f"Missing rigged drawing: {drawing}. Run copypaw rig on it first.")
    motion = MotionFile.load(motion_dir / "motion.npz")
    with tempfile.TemporaryDirectory() as out_dir:
        result, seconds = _timed(
            lambda: render_character(drawing, motion, "treadmill", Path(out_dir))
        )
    sheet = compose_sheet(result.frames_rgba, resample_motion(motion).contacts)
    review_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, "RGB").save(review_dir / f"{motion_dir.name}.png")
    return seconds


def contact_sheets(motions: Path, drawing: Path, review_dir: Path) -> dict[str, float]:
    """Write a sheet per motion folder; map each name to its render seconds."""
    folders = sorted(path.parent for path in motions.glob("*/motion.npz"))
    if not folders:
        raise FileNotFoundError(f"No gallery motions under {motions}")
    return {folder.name: contact_sheet(folder, drawing, review_dir) for folder in folders}


def gallery_render_seconds() -> float | None:
    """Time the slowest gallery render, or None while there is nothing to render."""
    if not DRAGON.is_dir() or not any(MOTIONS.glob("*/motion.npz")):
        return None
    with tempfile.TemporaryDirectory() as review:
        return max(contact_sheets(MOTIONS, DRAGON, Path(review)).values())


def pipeline_seconds(work: Path) -> tuple[float, float]:
    """Time animate (including exports) and capture → rig → animate with cached models."""
    command = str(Path(sys.executable).with_name("copypaw"))
    started = time.perf_counter()
    for arguments in (
        ["capture", str(SAMPLE), "-o", str(work / "motion")],
        ["rig", str(DRAWING), "-o", str(work / "drawing")],
    ):
        subprocess.run([command, *arguments], check=True, stdout=subprocess.DEVNULL)  # noqa: S603
    _, render_s = _timed(
        lambda: subprocess.run(  # noqa: S603
            [
                command,
                "animate",
                str(work / "drawing"),
                str(work / "motion/motion.npz"),
                "--clip",
                str(SAMPLE),
                "-o",
                str(work / "result"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    )
    return render_s, time.perf_counter() - started


def measure() -> list[Row]:
    cache = Path(os.environ.get("COPYPAW_CACHE", "~/.cache/copypaw/models")).expanduser()
    models = {key: cache / REGISTRY[key].filename for key in MODELS[Mode.BALANCED]}
    missing = [path.name for path in models.values() if not path.is_file()]
    if missing or not SAMPLE.is_file():
        raise SystemExit(f"Missing inputs: {', '.join(missing) or SAMPLE}")

    frames = read_frames(SAMPLE)
    count = len(frames.rgb)
    detector, pose = load_stages(Mode.BALANCED, False, models)
    track, detect_s = _timed(lambda: track_animal(frames, detector))
    raw, pose_s = _timed(lambda: pose(frames, track))
    identity, identity_s = _timed(
        lambda: resolve_identity(raw.kp, raw.scores, frame_width=frames.rgb.shape[2])
    )
    _, smoothing_s = _timed(lambda: clean(identity.kp, identity.scores, frames.t_s))

    totals = {flip_test: _capture_seconds(flip_test, models) for flip_test in (False, True)}

    with tempfile.TemporaryDirectory() as work:
        render_s, total_s = pipeline_seconds(Path(work))

    per_frame = [
        ("Detect", detect_s),
        ("Pose", pose_s),
        ("Identity", identity_s),
        ("Smoothing", smoothing_s),
    ]
    return [
        *(
            Row(f"{name}, per frame", seconds * MS_PER_S / count, "ms")
            for name, seconds in per_frame
        ),
        Row("Capture, no flip test", totals[False], "s"),
        Row("Capture, flip test", totals[True], "s"),
        Row("Animate, including video/GIF exports", render_s, "s"),
        Row("Capture → rig → animate, cached models", total_s, "s"),
        Row("Gallery motion render", gallery_render_seconds(), "s"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-readme", action="store_true", help="paste the table between the README markers"
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="render the dragon sample with every gallery motion and write a review sheet",
    )
    parser.add_argument("--motions", type=Path, default=MOTIONS)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--drawing", type=Path, default=DRAGON)
    args = parser.parse_args()
    if args.contact_sheet:
        try:
            for name, seconds in contact_sheets(args.motions, args.drawing, args.review).items():
                print(f"{name}: rendered in {seconds:.1f} s")
        except FileNotFoundError as error:
            print(f"Contact sheet failed: {error}", file=sys.stderr)
            return 1
        return 0
    write_readme = args.write_readme
    table = format_table(measure())
    print(table)
    if write_readme:
        README.write_text(paste_table(README.read_text(encoding="utf-8"), table), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
