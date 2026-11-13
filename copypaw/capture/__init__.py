"""Extract a pet's motion from a side-view clip."""

from copypaw.capture.detect import NoAnimalFoundError
from copypaw.capture.pipeline import PoseStage, capture_motion, load_stages
from copypaw.capture.quality import QualityReport
from copypaw.capture.video import Frames, VideoReadError, read_frames

__all__ = [
    "Frames",
    "NoAnimalFoundError",
    "PoseStage",
    "QualityReport",
    "VideoReadError",
    "capture_motion",
    "load_stages",
    "read_frames",
]
