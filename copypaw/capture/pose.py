"""RGB animal pose estimation with optional heatmap flip averaging."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from copypaw.capture.detect import Track
from copypaw.capture.video import Frames
from copypaw.models import MODELS, Mode
from copypaw.vendor.rtmlib_onnx.vitpose import ViTPose

__all__ = ["FLIP_PAIRS", "KEYPOINT_NAMES", "Keypoints", "estimate_keypoints"]

# AP-10K joint order, also used by the APT-36K ViTPose exports.
KEYPOINT_NAMES = (
    "L_Eye",
    "R_Eye",
    "Nose",
    "Neck",
    "Root_of_tail",
    "L_Shoulder",
    "L_Elbow",
    "L_F_Paw",
    "R_Shoulder",
    "R_Elbow",
    "R_F_Paw",
    "L_Hip",
    "L_Knee",
    "L_B_Paw",
    "R_Hip",
    "R_Knee",
    "R_B_Paw",
)
FLIP_PAIRS = ((0, 1), (5, 8), (6, 9), (7, 10), (11, 14), (12, 15), (13, 16))
_FLIP_ORDER = (1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13)
_FRAME_DIMENSIONS = 4
_JOINT_COUNT = len(KEYPOINT_NAMES)
_HEATMAP_SHAPE = (1, _JOINT_COUNT, 64, 48)  # 256x192 input at quarter resolution.


@dataclass(frozen=True)
class Keypoints:
    """Source-pixel coordinates kp (F, 17, 2) and confidence scores (F, 17).

    Frames without a track box have NaN coordinates and zero confidence.
    Reused track boxes are estimated even when the detection flag is false.
    """

    kp: NDArray[np.float32]
    scores: NDArray[np.float32]


def _heatmaps(
    outputs: list[NDArray[np.float32]],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    if len(outputs) != 1 or outputs[0].shape != _HEATMAP_SHAPE:
        raise ValueError("ViTPose must return one heatmap array with shape (1, 17, 64, 48)")
    maps = outputs[0]
    valid = np.isfinite(maps).all(axis=(-2, -1)) & (maps.max(axis=(-2, -1)) > 0)
    # Invalid joints cannot enter DARK's logarithm or contaminate other joints.
    return np.where(valid[..., None, None], maps, 0), valid


def _validate_inputs(frames: Frames, track: Track) -> None:
    if (
        frames.rgb.ndim != _FRAME_DIMENSIONS
        or frames.rgb.shape[-1:] != (3,)
        or frames.rgb.dtype != np.uint8
    ):
        raise ValueError("Expected uint8 RGB frames with shape (F, H, W, 3)")
    if frames.rgb.shape[1] == 0 or frames.rgb.shape[2] == 0:
        raise ValueError("Frame width and height must be positive")
    count = len(frames.rgb)
    if track.boxes.shape != (count, 4) or track.present.shape != (count,):
        raise ValueError("Track must contain boxes (F, 4) and detection flags (F,)")
    unavailable = np.isnan(track.boxes).all(axis=1)
    boxes = track.boxes[~unavailable]
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError(
            "Track boxes must be finite xyxy coordinates with positive area or all NaN"
        )
    if np.any(track.present & unavailable):
        raise ValueError("Detected frames must have a track box")


def estimate_keypoints(
    frames: Frames,
    track: Track,
    mode: str | Mode,
    flip_test: bool,
    models: Mapping[str, Path],
) -> Keypoints:
    """Estimate AP-10K joints using supplied model paths without fetching weights.

    ViTPose receives RGB crops with 1.25 box padding and ImageNet normalization.
    Flip testing restores heatmap width, joint pairs and MSRA alignment before
    averaging and DARK refinement. Invalid peaks become missing observations.
    """
    _validate_inputs(frames, track)
    model_key = MODELS[Mode(mode)][1]
    pose = ViTPose(str(models[model_key]))
    count = len(frames.rgb)
    kp = np.full((count, _JOINT_COUNT, 2), np.nan, dtype=np.float32)
    scores = np.zeros((count, _JOINT_COUNT), dtype=np.float32)
    for index, (rgb, box) in enumerate(zip(frames.rgb, track.boxes, strict=True)):
        if np.isnan(box).all():
            continue
        crop, center, scale = pose.preprocess(rgb, box)
        heatmaps, valid = _heatmaps(pose.inference(crop))
        if flip_test:
            flipped, flipped_valid = _heatmaps(pose.inference(crop[:, ::-1]))
            restored = flipped[:, _FLIP_ORDER, :, ::-1].copy()
            # MSRA rounds stride-four coordinates; reversal needs a one-bin shift.
            restored[..., 1:] = restored[..., :-1].copy()
            heatmaps = (heatmaps + restored) * 0.5
            valid &= flipped_valid[:, _FLIP_ORDER]
        coordinates, confidence = pose.postprocess([heatmaps], center, scale)
        if not np.isfinite(coordinates).all() or not np.isfinite(confidence).all():
            raise ValueError("ViTPose decoded keypoints and scores must be finite")
        valid &= confidence > 0
        kp[index] = np.where(valid[..., None], coordinates, np.nan)[0]
        scores[index] = np.where(valid, confidence, 0)[0]
    return Keypoints(kp, scores)
