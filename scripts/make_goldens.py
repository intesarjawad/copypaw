"""Rewrite the golden motion and render files from the sample inputs."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from copypaw.animate.render import render_character
from copypaw.capture.pipeline import capture_motion, load_stages
from copypaw.capture.video import read_frames
from copypaw.models import Mode, ensure_models
from copypaw.motion.motion_file import MotionFile

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples/pets/dog_walk.mp4"
GOLDEN = ROOT / "tests/golden"
QUADRUPED = ROOT / "tests/fixtures/upstream/characters/char6"
RENDER_FRAME = "quadruped_trot_f20.png"
WEB_GOLDEN = ROOT / "web/e2e/golden"


def _report_progress(filename: str, downloaded: int, total: int) -> None:
    print(f"{filename}: {downloaded}/{total} bytes", file=sys.stderr)


def _render_golden() -> None:
    motion = MotionFile.load(GOLDEN / f"{SAMPLE.stem}.motion.npz")
    with TemporaryDirectory() as tmp:
        result = render_character(QUADRUPED, motion, "treadmill", Path(tmp))
    Image.fromarray(result.frames_rgba[20], "RGBA").save(GOLDEN / RENDER_FRAME)


def motion_record(motion: MotionFile) -> dict[str, Any]:
    """Return every field of a motion as JSON values the browser port can read.

    Float32 arrays become exact doubles, so Math.fround recovers the stored values.
    """
    return {
        "version": motion.version,
        "fps": motion.fps,
        "facing": motion.facing,
        "kp": motion.kp.tolist(),
        "scores": motion.scores.tolist(),
        "contacts": motion.contacts.tolist(),
        "ground_line_px": motion.ground_line_px.tolist(),
        "travel_root_x": motion.travel_root_x.tolist(),
        "source": motion.source,
    }


def write_web_motion(motion_path: Path, out_dir: Path = WEB_GOLDEN) -> Path:
    """Write a motion archive as JSON for the browser.

    A gallery motion.npz is named after its folder, any other archive after its own stem.
    """
    record = motion_record(MotionFile.load(motion_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = motion_path.parent.name if motion_path.name == "motion.npz" else motion_path.name
    out = out_dir / f"{name.split('.')[0]}.motion.json"
    out.write_text(json.dumps(record), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Write the rendered golden frame instead of the captured motion.",
    )
    parser.add_argument(
        "--web",
        type=Path,
        metavar="MOTION",
        help="Export a gallery motion.npz as JSON for the browser parity tests.",
    )
    parser.add_argument(
        "--web-dir",
        type=Path,
        default=WEB_GOLDEN,
        help="Directory the --web JSON is written to.",
    )
    args = parser.parse_args()
    if args.web:
        print(write_web_motion(args.web, args.web_dir))
        return
    if args.render:
        _render_golden()
        return

    models = ensure_models(Mode.BALANCED, _report_progress)
    detector, pose = load_stages(Mode.BALANCED, True, models)
    GOLDEN.mkdir(exist_ok=True)
    with TemporaryDirectory() as tmp:
        capture_motion(read_frames(SAMPLE), SAMPLE.name, Path(tmp), detector=detector, pose=pose)
        shutil.copyfile(Path(tmp) / "motion.npz", GOLDEN / f"{SAMPLE.stem}.motion.npz")
        shutil.copyfile(Path(tmp) / "motion.bvh", GOLDEN / f"{SAMPLE.stem}.bvh")


if __name__ == "__main__":
    main()
