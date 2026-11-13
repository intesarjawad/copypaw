"""Render every sample drawing with every gallery motion and write the site's clips."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from copypaw.animate.export import compose_frames, drawing_facing, load_clip_frames
from copypaw.animate.render import RenderResult, render_character, resample_motion
from copypaw.capture.video import OUTPUT_FPS
from copypaw.errors import CopypawError
from copypaw.evaluate.fidelity import (
    CalibrationError,
    DrawingJoints,
    drawing_contacts,
    evaluate,
    is_locomotion,
    load_calibration,
    timing_words,
)

try:
    from scripts.eval_motion import (
        DRAWINGS,
        MOTIONS,
        GalleryMotion,
        discover_drawings,
        discover_motions,
    )
except ModuleNotFoundError:
    from eval_motion import DRAWINGS, MOTIONS, GalleryMotion, discover_drawings, discover_motions

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs/site"
CLIPS_NAME = "clips"
MANIFEST_NAME = "manifest.json"
CLIP = "clip.mp4"
HERO = "hero"
SECONDS = 4
WIDTH_PX = 512
PICTURE_PX = 256
START_CRF = 23
MAX_CRF = 51
CRF_STEP = 2
POSTER_QUALITY = 80
NOT_MEASURED = "timing match: not measured"
FONT_SUFFIXES = frozenset({".woff", ".woff2", ".ttf", ".otf"})
_RGB_CHANNELS = 3
_GROUND = 255
_FFMPEG: Any = import_module("imageio_ffmpeg")
# Drop the container and stream tags and the encoder's own SEI, so a clip holds
# only the picture. Type 6 is the H.264 SEI unit.
_STRIP_TAGS = [
    "-map_metadata",
    "-1",
    "-fflags",
    "+bitexact",
    "-flags:v",
    "+bitexact",
    "-metadata:s:v",
    "encoder=",
    "-bsf:v",
    "filter_units=remove_types=6",
    "-movflags",
    "+faststart",
]


@dataclass(frozen=True)
class Budget:
    """Byte limits the published site must stay under."""

    outside_clips: int = 5_000_000
    clip: int = 300_000
    first_view: int = 2_000_000
    site: int = 300_000_000


BUDGET = Budget()


@dataclass(frozen=True)
class Entry:
    """One rendered clip and what its caption says."""

    drawing: str
    motion: str
    animal: str
    gait: str
    timing: str
    crf: int
    file: str
    poster: str
    bytes: int

    def as_json(self) -> dict[str, object]:
        """Return the manifest fields in a stable order."""
        return {
            "drawing": self.drawing,
            "motion": self.motion,
            "animal": self.animal,
            "gait": self.gait,
            "timing": self.timing,
            "crf": self.crf,
            "file": self.file,
            "poster": self.poster,
            "bytes": self.bytes,
        }


def clip_id(drawing: str, motion: str) -> str:
    """Name a drawing and motion pair for file names."""
    return f"{drawing}__{motion}"


def _resize(frame: NDArray[np.uint8], size: tuple[int, int]) -> NDArray[np.uint8]:
    """Shrink or stretch a frame to (width, height), averaging when it shrinks."""
    return np.asarray(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def letterbox(frames: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Fit frames inside a square of the page's ground colour without stretching them."""
    height, width = frames.shape[1:3]
    scale = size / max(height, width)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    out = np.full((len(frames), size, size, _RGB_CHANNELS), _GROUND, dtype=np.uint8)
    for index, frame in enumerate(frames):
        out[index, top : top + new_height, left : left + new_width] = _resize(
            frame, (new_width, new_height)
        )
    return out


def compose_clip_frames(
    clip_frames: NDArray[np.uint8],
    drawing_frames_rgba: NDArray[np.uint8],
    *,
    facing_clip: Literal["left", "right"],
    facing_drawing: Literal["left", "right"],
    contacts: NDArray[np.bool_],
    drawing_stance: NDArray[np.bool_] | None,
    count: int = SECONDS * OUTPUT_FPS,
) -> NDArray[np.uint8]:
    """Pair the pet and the drawing, each in a square picture, over the footfall strip.

    The pair is looped from its start to fill count frames, and every input is
    cut to the shortest so a frame's two halves and its strip share a time.
    """
    available = min(
        len(clip_frames),
        len(drawing_frames_rgba),
        len(contacts),
        *([] if drawing_stance is None else [len(drawing_stance)]),
    )
    if available == 0:
        raise ValueError("The clip, the drawing and the contacts must each hold a frame")
    take = np.arange(count) % available
    pet = letterbox(np.asarray(clip_frames, dtype=np.uint8)[take], 2 * PICTURE_PX)
    drawing = np.asarray(drawing_frames_rgba, dtype=np.uint8)[take]
    full = compose_frames(
        pet,
        drawing,
        facing_clip=facing_clip,
        facing_drawing=facing_drawing,
        keep_facing=False,
        credit=False,
        contacts=np.asarray(contacts, dtype=np.bool_)[take],
        drawing_contacts=None if drawing_stance is None else drawing_stance[take],
    )
    size = (WIDTH_PX, full.shape[1] // 2)
    return np.stack([_resize(frame, size) for frame in full])


def encode(frames: NDArray[np.uint8], fps: float, path: Path, crf: int) -> int:
    """Write frames as a silent H.264 MP4 without metadata and return its size in bytes."""
    height, width = frames.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = _FFMPEG.write_frames(
        str(path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        # yuv420p needs even sides; a larger block would stretch the frame to pad it.
        macro_block_size=2,
        ffmpeg_log_level="error",
        output_params=["-crf", str(crf), "-preset", "slow", *_STRIP_TAGS],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()
    return path.stat().st_size


def encode_under_budget(
    frames: NDArray[np.uint8], fps: float, path: Path, limit: int
) -> tuple[int, int]:
    """Raise the CRF until the clip is under limit bytes; return the CRF and the size."""
    for crf in range(START_CRF, MAX_CRF + 1, CRF_STEP):
        size = encode(frames, fps, path, crf)
        if size < limit:
            return crf, size
    raise ValueError(f"{path.name} stays over {limit} bytes at CRF {MAX_CRF}")


def write_poster(frame: NDArray[np.uint8], path: Path) -> None:
    """Save a frame as a JPEG."""
    Image.fromarray(frame, "RGB").save(path, "JPEG", quality=POSTER_QUALITY)


def source_ref(env: Mapping[str, str]) -> str:
    """Name what the site was built from: the tag for a tag build, else the commit."""
    if env.get("GITHUB_REF_TYPE") == "tag" and env.get("GITHUB_REF_NAME"):
        return env["GITHUB_REF_NAME"]
    if env.get("GITHUB_SHA"):
        return env["GITHUB_SHA"]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def build_manifest(built_from: str, entries: list[Entry]) -> dict[str, object]:
    """Describe the clips; the first entry is the hero."""
    if not entries:
        raise ValueError("A gallery needs at least one clip")
    drawings = sorted({entry.drawing for entry in entries})
    motions = sorted({entry.motion for entry in entries})
    return {
        "built_from": built_from,
        "hero": {"file": f"{HERO}.mp4", "poster": f"{HERO}.jpg"},
        "drawings": drawings,
        "motions": motions,
        "clips": [entry.as_json() for entry in entries],
    }


def _drawing_joints(rendered: RenderResult) -> DrawingJoints | None:
    if rendered.joint_names is None:
        return None
    unfiltered = rendered.unfiltered_joints_per_frame
    return DrawingJoints(
        joints=rendered.joints_per_frame,
        joints_unfiltered=rendered.joints_per_frame if unfiltered is None else unfiltered,
        names=rendered.joint_names,
        fps=rendered.fps,
    )


def _timing(item: GalleryMotion, joints: DrawingJoints | None, percentile: float | None) -> str:
    if joints is None or percentile is None:
        return NOT_MEASURED
    result = evaluate(item.motion, joints)
    return timing_words(result.gait_r, locomotion=is_locomotion(item.motion), percentile=percentile)


def build_entry(
    item: GalleryMotion,
    *,
    folder: Path,
    drawing: str,
    drawing_dir: Path,
    clips: Path,
    percentile: float | None,
    limit: int,
) -> Entry:
    """Render one drawing with one motion beside the motion's own clip."""
    with tempfile.TemporaryDirectory() as tmp:
        rendered = render_character(drawing_dir, item.motion, "treadmill", Path(tmp))
    joints = _drawing_joints(rendered)
    frames = compose_clip_frames(
        load_clip_frames(folder / CLIP),
        rendered.frames_rgba,
        facing_clip=item.motion.facing,
        facing_drawing=drawing_facing(drawing_dir),
        contacts=resample_motion(item.motion).contacts,
        drawing_stance=None if joints is None else drawing_contacts(joints),
    )
    name = clip_id(drawing, item.name)
    crf, size = encode_under_budget(frames, OUTPUT_FPS, clips / f"{name}.mp4", limit)
    write_poster(frames[len(frames) // 2], clips / f"{name}.jpg")
    return Entry(
        drawing=drawing,
        motion=item.name,
        animal=item.animal,
        gait=item.gait,
        timing=_timing(item, joints, percentile),
        crf=crf,
        file=f"{name}.mp4",
        poster=f"{name}.jpg",
        bytes=size,
    )


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _first_view(site: Path, shell: list[Path]) -> list[Path]:
    """Pick the shell files a visitor loads before the hero clip."""
    return [
        path
        for path in shell
        if path == site / "index.html"
        or (path.parent == site and path.suffix in {".css", ".js"})
        or path.suffix in FONT_SUFFIXES
    ]


def budget_problems(site: Path, budget: Budget = BUDGET) -> list[str]:
    """List every size limit the site breaks; empty means it fits."""
    clips = site / CLIPS_NAME
    manifest = clips / MANIFEST_NAME
    if not manifest.is_file():
        return [f"missing {manifest}; run scripts/build_gallery.py first"]
    hero = cast(
        "dict[str, str]", cast("dict[str, object]", json.loads(manifest.read_text("utf-8")))["hero"]
    )
    files = _files(site)
    shell = [path for path in files if clips not in path.parents]
    videos = [path for path in files if clips in path.parents and path.suffix == ".mp4"]
    first_view = [*_first_view(site, shell), clips / hero["file"], clips / hero["poster"]]
    problems: list[str] = []
    shell_bytes = sum(path.stat().st_size for path in shell)
    if shell_bytes >= budget.outside_clips:
        problems.append(f"shell is {shell_bytes} bytes, limit {budget.outside_clips}")
    problems.extend(
        f"{path.name} is {path.stat().st_size} bytes, limit {budget.clip}"
        for path in videos
        if path.stat().st_size >= budget.clip
    )
    missing = [path.name for path in first_view if not path.is_file()]
    problems.extend(f"first view needs {name}, which is missing" for name in missing)
    first_bytes = sum(path.stat().st_size for path in first_view if path.is_file())
    if first_bytes >= budget.first_view:
        problems.append(f"first view is {first_bytes} bytes, limit {budget.first_view}")
    site_bytes = sum(path.stat().st_size for path in files)
    if site_bytes >= budget.site:
        problems.append(f"site is {site_bytes} bytes, limit {budget.site}")
    return problems


def build_site(
    motions: Path, drawings: Path, site: Path, built_from: str, limit: int = BUDGET.clip
) -> dict[str, object]:
    """Render every drawing with every motion into site/clips and write the manifest."""
    items = discover_motions(motions)
    names = discover_drawings(drawings)
    for item in items:
        if not (motions / item.name / CLIP).is_file():
            raise FileNotFoundError(f"Missing {CLIP} for motion {item.name}")
    try:
        percentile: float | None = load_calibration()
    except CalibrationError:
        percentile = None
    clips = site / CLIPS_NAME
    clips.mkdir(parents=True, exist_ok=True)
    for stale in (*clips.glob("*.mp4"), *clips.glob("*.jpg")):
        stale.unlink()
    entries: list[Entry] = []
    for drawing in names:
        for item in items:
            entry = build_entry(
                item,
                folder=motions / item.name,
                drawing=drawing,
                drawing_dir=drawings / drawing,
                clips=clips,
                percentile=percentile,
                limit=limit,
            )
            entries.append(entry)
            print(f"Rendered {entry.file}: {entry.bytes} bytes at CRF {entry.crf}")
    shutil.copyfile(clips / entries[0].file, clips / f"{HERO}.mp4")
    shutil.copyfile(clips / entries[0].poster, clips / f"{HERO}.jpg")
    manifest = build_manifest(built_from, entries)
    (clips / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, default=MOTIONS)
    parser.add_argument("--drawings", type=Path, default=DRAWINGS)
    parser.add_argument("--site", type=Path, default=SITE)
    parser.add_argument("--built-from", help="Tag or commit to record; default from the CI env")
    parser.add_argument(
        "--check-budget",
        action="store_true",
        help="Check the built site against the size limits instead of rendering",
    )
    args = parser.parse_args()
    if args.check_budget:
        problems = budget_problems(args.site)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print("Site is within budget")
        return 1 if problems else 0
    try:
        build_site(
            args.motions,
            args.drawings,
            args.site,
            args.built_from or source_ref(os.environ),
        )
    except (FileNotFoundError, ValueError, CopypawError, subprocess.CalledProcessError) as error:
        print(f"Gallery build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
