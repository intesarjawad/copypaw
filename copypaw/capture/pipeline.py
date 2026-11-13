"""Turn sampled video frames into a saved motion, skeleton export and quality report."""

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

from copypaw.capture.contacts import detect_contacts
from copypaw.capture.detect import YOLOX, Detector, Track, track_animal
from copypaw.capture.identity import resolve_identity
from copypaw.capture.pose import Keypoints, estimate_keypoints
from copypaw.capture.quality import QualityReport, assess_quality, write_quality
from copypaw.capture.smoothing import clean
from copypaw.capture.travel_root import travel_root_x
from copypaw.capture.video import Frames
from copypaw.errors import CopypawError
from copypaw.models import MODELS, Mode
from copypaw.motion.bvh_writer import write_bvh
from copypaw.motion.motion_file import MotionFile

__all__ = ["PoseStage", "capture_motion", "load_stages"]


class CaptureQualityError(CopypawError):
    """Required anatomy was never observed in the source clip."""

    codes = ("capture_anatomy",)


type PoseStage = Callable[[Frames, Track], Keypoints]


def load_stages(
    mode: Mode, flip_test: bool, models: Mapping[str, Path]
) -> tuple[Detector, PoseStage]:
    """Open the detector for mode and bind the pose estimator to its model files."""
    detector_key, _ = MODELS[mode]
    return YOLOX(str(models[detector_key])), partial(
        estimate_keypoints, mode=mode, flip_test=flip_test, models=models
    )


def capture_motion(
    frames: Frames, source: str, out_dir: Path, *, detector: Detector, pose: PoseStage
) -> QualityReport:
    """Write motion.npz, motion.bvh and quality.json for frames into an existing out_dir.

    Quality is measured on the raw pose scores, before cleaning can hide gaps.
    Motion stores identity-resolved, cleaned, right-facing keypoints.
    """
    track = track_animal(frames, detector)
    raw = pose(frames, track)
    identity = resolve_identity(raw.kp, raw.scores, frame_width=frames.rgb.shape[2])
    report = assess_quality(raw.scores, frames.t_s, swaps_corrected=identity.swaps_corrected)
    if report.errors:
        write_quality(report, out_dir)
        raise CaptureQualityError(problem=" ".join(report.errors))
    cleaned = clean(identity.kp, identity.scores, frames.t_s)
    contacts = detect_contacts(cleaned.keypoints, frames.t_s, scores=identity.scores)
    root_x = travel_root_x(cleaned.keypoints, contacts.contacts, frames.t_s)
    MotionFile(
        kp=cleaned.keypoints,
        scores=identity.scores,
        fps=frames.fps,
        facing=identity.facing,
        contacts=contacts.contacts,
        ground_line_px=contacts.ground_line_px,
        travel_root_x=root_x,
        source=source,
    ).save(out_dir / "motion.npz")
    write_bvh(cleaned.keypoints, frames.fps, root_x, out=out_dir / "motion.bvh")
    write_quality(report, out_dir)
    return report
