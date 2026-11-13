"""Build gallery motions from their clips: capture, contacts and thumbnail."""

import argparse
import json
import math
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from copypaw.capture.contacts import (
    CONTACT_HEIGHT_RATIO,
    CONTACT_SPEED_RATIO,
    detect_contacts,
)
from copypaw.capture.pipeline import capture_motion, load_stages
from copypaw.capture.travel_root import travel_root_x
from copypaw.capture.video import read_frames
from copypaw.errors import CopypawError
from copypaw.models import Mode, ensure_models
from copypaw.motion.bvh_writer import write_bvh
from copypaw.motion.motion_file import MotionFile

try:
    from scripts.label_contacts import PAW_SLOTS, Labels, Stance, write_labels
except ModuleNotFoundError:
    from label_contacts import PAW_SLOTS, Labels, Stance, write_labels

ROOT = Path(__file__).resolve().parents[1]
MOTIONS = ROOT / "motions"
CALIBRATION = ROOT / "copypaw/evaluate/calibration.json"
EVALUATION_SET = "evaluation_set.txt"
CLIP = "clip.mp4"
NPZ = "motion.npz"
BVH = "motion.bvh"
THUMB = "thumb.png"
CONTACTS_AUTO = "contacts_auto.json"
# Reproduction tolerances: stance flags match exactly, root height within a
# thousandth of a body length and driver joint angles within half a degree.
ANGLE_TOLERANCE_DEG = 0.5
ROOT_Y_TOLERANCE = 1e-3
_ROOT_CHANNELS = 6  # Root position then rotation.
_ROOT_Y_CHANNEL = 1
_JOINT_CHANNELS = 3

type Capture = Callable[[Path, Path], None]


def _thresholds(path: Path) -> tuple[float, float]:
    """Read the tuned contact thresholds, or the initial pair before the sweep."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing calibration: {path}. Run scripts/eval_motion.py on the gallery."
        )
    data = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    height = data.get("contact_height_ratio")
    speed = data.get("contact_speed_ratio")
    if height is None and speed is None:
        return CONTACT_HEIGHT_RATIO, CONTACT_SPEED_RATIO
    if not isinstance(height, (int, float)) or not isinstance(speed, (int, float)):
        raise ValueError(
            f"Calibration has no contact thresholds: {path}. Run scripts/eval_motion.py."
        )
    if not math.isfinite(float(height)) or float(height) <= 0:
        raise ValueError("contact_height_ratio must be finite and positive")
    if not math.isfinite(float(speed)) or float(speed) < 0:
        raise ValueError("contact_speed_ratio must be finite and nonnegative")
    return float(height), float(speed)


def read_evaluation_set(path: Path) -> frozenset[str]:
    """List the launch motions: one folder name per line, blank lines and # notes ignored."""
    lines = (
        line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
    )
    return frozenset(line for line in lines if line)


def with_contacts(motion: MotionFile, height_ratio: float, speed_ratio: float) -> MotionFile:
    """Re-detect contacts and travel on a motion's keypoints at the given thresholds."""
    times = np.arange(len(motion.kp), dtype=np.float64) / motion.fps
    contacts = detect_contacts(
        motion.kp,
        times,
        scores=motion.scores,
        height_ratio=height_ratio,
        speed_ratio=speed_ratio,
    )
    return MotionFile(
        kp=motion.kp,
        scores=motion.scores,
        fps=motion.fps,
        facing=motion.facing,
        contacts=contacts.contacts,
        ground_line_px=contacts.ground_line_px,
        travel_root_x=travel_root_x(motion.kp, contacts.contacts, times),
        source=motion.source,
    )


def _save(motion: MotionFile, directory: Path) -> None:
    motion.save(directory / NPZ)
    write_bvh(motion.kp, motion.fps, motion.travel_root_x, out=directory / BVH)


def rebuild_motion(path: Path, height_ratio: float, speed_ratio: float) -> MotionFile:
    """Re-detect contacts on a stored motion and write the archive and BVH back."""
    rebuilt = with_contacts(MotionFile.load(path), height_ratio, speed_ratio)
    _save(rebuilt, path.parent)
    return rebuilt


def accurate_capture() -> Capture:
    """Load the accurate models once and return a capture of one clip into a directory."""

    def progress(filename: str, downloaded: int, total: int) -> None:
        print(f"{filename}: {downloaded}/{total} bytes", file=sys.stderr)

    models = ensure_models(Mode.ACCURATE, progress)
    detector, pose = load_stages(Mode.ACCURATE, True, models)

    def capture(clip: Path, out_dir: Path) -> None:
        capture_motion(read_frames(clip), clip.name, out_dir, detector=detector, pose=pose)

    return capture


def capture_clip(
    clip: Path, out_dir: Path, capture: Capture, height_ratio: float, speed_ratio: float
) -> MotionFile:
    """Capture a clip into out_dir and apply the contact thresholds to the result."""
    capture(clip, out_dir)
    motion = with_contacts(MotionFile.load(out_dir / NPZ), height_ratio, speed_ratio)
    _save(motion, out_dir)
    return motion


def write_thumbnail(clip: Path, path: Path) -> None:
    """Cut the middle frame of a clip to a PNG with OpenCV."""
    video = cv2.VideoCapture(str(clip))
    try:
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.set(cv2.CAP_PROP_POS_FRAMES, max(count // 2, 0))
        found, frame = video.read()
        if not found:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            found, frame = video.read()
    finally:
        video.release()
    if not found or not cv2.imwrite(str(path), frame):
        raise ValueError(f"Could not cut a thumbnail from {clip}")


def contact_stances(contacts: NDArray[np.bool_]) -> dict[str, tuple[Stance, ...]]:
    """Turn stance flags into one onset and lift-off frame pair per planted run."""
    paws: dict[str, tuple[Stance, ...]] = {}
    for slot, track in zip(PAW_SLOTS, contacts.T, strict=True):
        edges = np.flatnonzero(np.diff(np.r_[False, track, False]))
        paws[slot] = tuple(
            Stance(float(start), float(stop)) for start, stop in edges.reshape(-1, 2)
        )
    return paws


def write_contacts_auto(folder: Path, motion: MotionFile) -> None:
    """Write the detected contacts in the label-file format."""
    labels = Labels(folder.name, "auto", motion.fps, contact_stances(motion.contacts))
    write_labels(folder / CONTACTS_AUTO, labels)


def build_folder(
    folder: Path,
    capture: Capture,
    height_ratio: float,
    speed_ratio: float,
    *,
    contacts_auto: bool,
) -> None:
    """Write motion.npz, motion.bvh and thumb.png for a folder's clip."""
    clip = folder / CLIP
    with TemporaryDirectory() as tmp:
        motion = capture_clip(clip, Path(tmp), capture, height_ratio, speed_ratio)
        shutil.copyfile(Path(tmp) / NPZ, folder / NPZ)
        shutil.copyfile(Path(tmp) / BVH, folder / BVH)
    write_thumbnail(clip, folder / THUMB)
    if contacts_auto:
        write_contacts_auto(folder, motion)


def _read_bvh(path: Path) -> tuple[list[str], NDArray[np.float64]]:
    """Return the joint names and a (frames, channels) table from a BVH file."""
    header, motion = path.read_text(encoding="utf-8").split("MOTION\n")
    names = [line.split()[1] for line in header.splitlines() if line.startswith("JOINT ")]
    rows = [line.split() for line in motion.splitlines()[2:] if line]
    return names, np.array(rows, dtype=np.float64)


def compare_motions(expected: Path, actual: Path) -> list[str]:
    """Compare two motion folders within the reproduction tolerances."""
    want = MotionFile.load(expected / NPZ)
    got = MotionFile.load(actual / NPZ)
    if len(got.kp) != len(want.kp):
        return [f"frame count {len(got.kp)} differs from {len(want.kp)}"]
    problems: list[str] = []
    if got.fps != want.fps:
        problems.append(f"fps {got.fps} differs from {want.fps}")
    if not np.array_equal(got.contacts, want.contacts):
        frame = int(np.flatnonzero((got.contacts != want.contacts).any(axis=1))[0])
        problems.append(f"contacts differ at frame {frame}")
    want_names, want_table = _read_bvh(expected / BVH)
    got_names, got_table = _read_bvh(actual / BVH)
    if got_names != want_names or got_table.shape != want_table.shape:
        return [*problems, "BVH joints or frame count differ"]
    root_y = np.abs(got_table[:, _ROOT_Y_CHANNEL] - want_table[:, _ROOT_Y_CHANNEL])
    frame = int(np.argmax(root_y))
    if root_y[frame] > ROOT_Y_TOLERANCE:
        problems.append(f"root y differs by {root_y[frame]:.6f} at frame {frame}")
    for joint, name in enumerate(want_names):
        if not name.endswith("_drv"):
            continue
        column = _ROOT_CHANNELS + joint * _JOINT_CHANNELS
        delta = (got_table[:, column] - want_table[:, column] + 180) % 360 - 180
        frame = int(np.argmax(np.abs(delta)))
        if abs(delta[frame]) > ANGLE_TOLERANCE_DEG:
            problems.append(f"{name} differs by {delta[frame]:.4f} degrees at frame {frame}")
    return problems


def check_folder(
    folder: Path, capture: Capture, height_ratio: float, speed_ratio: float
) -> list[str]:
    """Recapture a launch folder's clip and compare it with the committed files."""
    missing = [name for name in (NPZ, BVH) if not (folder / name).is_file()]
    if missing:
        return [f"missing {', '.join(missing)}"]
    with TemporaryDirectory() as tmp:
        capture_clip(folder / CLIP, Path(tmp), capture, height_ratio, speed_ratio)
        return compare_motions(folder, Path(tmp))


def _folders(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*") if (path / CLIP).is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, default=MOTIONS)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--new",
        action="store_true",
        help="Build only folders with no motion.npz and write contacts_auto.json for them",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Recapture every launch motion and compare it with the committed files",
    )
    mode.add_argument(
        "--contacts",
        action="store_true",
        help="Re-detect contacts on stored motions at the calibrated thresholds, without capture",
    )
    args = parser.parse_args()
    try:
        height, speed = _thresholds(args.calibration)
        if args.contacts:
            paths = sorted(args.motions.glob(f"*/{NPZ}"))
            if not paths:
                raise FileNotFoundError(f"No gallery motions under {args.motions}")
            for path in paths:
                rebuild_motion(path, height, speed)
                print(f"Rebuilt {path}")
            return 0
        folders = _folders(args.motions)
        if args.new:
            folders = [folder for folder in folders if not (folder / NPZ).is_file()]
        else:
            launch = read_evaluation_set(args.motions / EVALUATION_SET)
            folders = [folder for folder in folders if folder.name in launch]
        if not folders:
            print(f"No motions to build under {args.motions}")
            return 0
        capture = accurate_capture()
        failures = 0
        for folder in folders:
            if args.check:
                problems = check_folder(folder, capture, height, speed)
                failures += bool(problems)
                for problem in problems:
                    print(f"{folder.name}: {problem}", file=sys.stderr)
                print(f"{folder.name}: {'differs' if problems else 'reproduced'}")
            else:
                build_folder(folder, capture, height, speed, contacts_auto=args.new)
                print(f"Built {folder}")
        return 1 if failures else 0
    except (FileNotFoundError, ValueError, CopypawError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
