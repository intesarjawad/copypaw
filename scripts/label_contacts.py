"""Draw paw strips for contact labelling and merge independent label files."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw

from copypaw.capture.contacts import PAW_INDICES, body_length_px
from copypaw.capture.video import read_frames
from copypaw.motion.motion_file import MotionFile

ROOT = Path(__file__).resolve().parents[1]
MOTIONS = ROOT / "motions"
MERGED_NAME = "contacts_reference.json"
PAW_SLOTS = ("near_front", "far_front", "near_back", "far_back")
_CROP_SIZE = 64
_CROP_RADIUS_RATIO = 1.0
_AGREEMENT_FRAMES = 1
_LABELLERS = 2


@dataclass(frozen=True)
class Stance:
    """One labelled stance in analysis-frame numbers."""

    onset: float
    lift_off: float


@dataclass(frozen=True)
class Labels:
    """One labeller's stances for every paw of one motion."""

    motion: str
    labeller: str
    fps: float
    paws: dict[str, tuple[Stance, ...]]


def _stance(value: object) -> Stance:
    data = cast("dict[str, object]", value)
    onset = data.get("onset")
    lift_off = data.get("lift_off")
    if not isinstance(onset, (int, float)) or not isinstance(lift_off, (int, float)):
        raise ValueError("Each stance needs numeric onset and lift_off frames")
    if not np.isfinite(float(onset)) or not np.isfinite(float(lift_off)):
        raise ValueError("Stance frames must be finite")
    if onset < 0 or lift_off < onset:
        raise ValueError(f"Stance frames must satisfy 0 <= onset <= lift_off, got {value}")
    return Stance(float(onset), float(lift_off))


def load_labels(path: Path) -> Labels:
    """Read one contacts_reference.<labeller>.json file."""
    data = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    motion = data.get("motion")
    labeller = data.get("labeller")
    paws = data.get("paws")
    fps = data.get("fps")
    if not isinstance(motion, str) or not isinstance(labeller, str):
        raise ValueError(f"Label file needs motion and labeller strings: {path}")
    if not isinstance(fps, (int, float)) or not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError(f"Label file needs a positive fps: {path}")
    if not isinstance(paws, dict):
        raise ValueError(f"Label file needs a paws object: {path}")
    values = cast("dict[str, object]", paws)
    parsed: dict[str, tuple[Stance, ...]] = {}
    for slot in PAW_SLOTS:
        entries = values.get(slot, [])
        if not isinstance(entries, list):
            raise ValueError(f"Paw {slot} must hold a list of stances: {path}")
        parsed[slot] = tuple(_stance(entry) for entry in cast("list[object]", entries))
    return Labels(motion, labeller, float(fps), parsed)


def _payload(labels: Labels, labellers: tuple[str, ...] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "motion": labels.motion,
        "labeller": labels.labeller,
        "fps": labels.fps,
        "paws": {
            slot: [
                {
                    "onset": _frame_value(stance.onset),
                    "lift_off": _frame_value(stance.lift_off),
                }
                for stance in labels.paws.get(slot, ())
            ]
            for slot in PAW_SLOTS
        },
    }
    if labellers is not None:
        payload["labellers"] = list(labellers)
    return payload


def _frame_value(frame: float) -> int | float:
    return int(frame) if float(frame).is_integer() else frame


def _formatted(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def write_labels(path: Path, labels: Labels, labellers: tuple[str, ...] | None = None) -> None:
    """Write one label file with stable key and paw order."""
    path.write_text(_formatted(_payload(labels, labellers)), encoding="utf-8")


def _same_stance(first: Stance, second: Stance) -> bool:
    return (
        abs(first.onset - second.onset) <= _AGREEMENT_FRAMES
        and abs(first.lift_off - second.lift_off) <= _AGREEMENT_FRAMES
    )


def merge_labels(first: Labels, second: Labels) -> tuple[Labels, float]:
    """Keep only stances both labellers placed within one frame.

    The merged onset and lift-off are the mean of the two agreed frames. Onset
    agreement is the share of all labelled onsets that found a partner.
    """
    if first.motion != second.motion:
        raise ValueError(f"Cannot merge labels for {first.motion!r} and {second.motion!r}")
    if first.fps != second.fps:
        raise ValueError("Labels must use the same analysis rate")
    merged: dict[str, tuple[Stance, ...]] = {}
    agreed_onsets = 0
    total_onsets = 0
    for slot in PAW_SLOTS:
        left = list(first.paws.get(slot, ()))
        right = list(second.paws.get(slot, ()))
        total_onsets += len(left) + len(right)
        pairs: list[Stance] = []
        remaining = right.copy()
        for stance in left:
            match = next(
                (index for index, other in enumerate(remaining) if _same_stance(stance, other)),
                None,
            )
            if match is None:
                continue
            other = remaining.pop(match)
            agreed_onsets += 2
            pairs.append(
                Stance(
                    onset=(stance.onset + other.onset) / 2,
                    lift_off=(stance.lift_off + other.lift_off) / 2,
                )
            )
        merged[slot] = tuple(sorted(pairs, key=lambda stance: stance.onset))
    agreement = 0.0 if total_onsets == 0 else agreed_onsets / total_onsets
    return Labels(first.motion, "merged", first.fps, merged), agreement


def label_files(motion_dir: Path) -> list[Path]:
    return sorted(
        path for path in motion_dir.glob("contacts_reference.*.json") if path.name != MERGED_NAME
    )


def _label_motions(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.glob("*") if path.is_dir() and (path / "motion.npz").is_file()
    )


def merge_all(directory: Path, *, check: bool) -> int:
    """Merge every motion's two label files, or verify the committed result."""
    motions = _label_motions(directory)
    if not motions:
        print(f"No gallery motions under {directory}; nothing to merge.")
        return 0
    failures = 0
    for motion_dir in motions:
        files = label_files(motion_dir)
        if len(files) != _LABELLERS:
            print(f"{motion_dir.name}: expected two label files, found {len(files)}")
            failures += 1
            continue
        first, second = (load_labels(path) for path in files)
        merged, agreement = merge_labels(first, second)
        text = _formatted(_payload(merged, (first.labeller, second.labeller)))
        destination = motion_dir / MERGED_NAME
        print(f"{motion_dir.name}: onset agreement {agreement:.3f}")
        if check:
            if not destination.is_file() or destination.read_text(encoding="utf-8") != text:
                print(f"{motion_dir.name}: {destination} is out of date")
                failures += 1
        else:
            destination.write_text(text, encoding="utf-8")
    return 1 if failures else 0


def _clip_path(motion_dir: Path, motion: MotionFile) -> Path:
    candidates = (
        motion_dir / motion.source,
        motion_dir / Path(motion.source).name,
        ROOT / "samples/pets" / Path(motion.source).name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Clip {motion.source!r} is not beside {motion_dir / 'motion.npz'}")


def _cell(frame: np.ndarray, x: float, y: float, radius: int, number: int) -> Image.Image:
    height, width = frame.shape[:2]
    left = max(0, round(x - radius))
    top = max(0, round(y - radius))
    right = min(width, round(x + radius))
    bottom = min(height, round(y + radius))
    crop = Image.fromarray(np.asarray(frame[top:bottom, left:right], dtype=np.uint8), "RGB")
    crop = crop.resize((_CROP_SIZE, _CROP_SIZE), Image.Resampling.NEAREST)
    cell = Image.new("RGB", (_CROP_SIZE, _CROP_SIZE + 14), "white")
    cell.paste(crop, (0, 14))
    ImageDraw.Draw(cell).text((2, 1), str(number), fill="black")
    return cell


def render_strips(motion_dir: Path) -> list[Path]:
    """Write one numbered crop strip per paw beside the motion archive."""
    motion = MotionFile.load(motion_dir / "motion.npz")
    clip = _clip_path(motion_dir, motion)
    frames = read_frames(clip, max_s=1_000.0, target_fps=motion.fps)
    if len(frames.rgb) < len(motion.kp):
        raise ValueError(
            f"Clip has {len(frames.rgb)} frames but motion has {len(motion.kp)}: {clip}"
        )
    body = body_length_px(motion.kp)
    radius = max(1, round(_CROP_RADIUS_RATIO * body))
    out = motion_dir / "strips"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for slot, joint in zip(PAW_SLOTS, PAW_INDICES, strict=True):
        cells = [
            _cell(
                frame,
                float(motion.kp[index, joint, 0]),
                float(motion.kp[index, joint, 1]),
                radius,
                index,
            )
            for index, frame in enumerate(frames.rgb[: len(motion.kp)])
        ]
        strip = Image.new("RGB", (_CROP_SIZE * len(cells), _CROP_SIZE + 14), "white")
        for index, cell in enumerate(cells):
            strip.paste(cell, (index * _CROP_SIZE, 0))
        path = out / f"{slot}.png"
        strip.save(path)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", nargs="?", type=Path, help="Gallery motion directory")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge two independent label files for every gallery motion",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="With --merge, verify the merged files without writing them",
    )
    parser.add_argument(
        "--motions",
        type=Path,
        default=MOTIONS,
        help="Gallery motions directory (default: motions)",
    )
    args = parser.parse_args()
    if args.merge:
        return merge_all(args.motions, check=args.check)
    if args.motion is None:
        parser.error("give a motion directory, or use --merge")
    for path in render_strips(args.motion):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
