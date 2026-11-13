"""Gait fidelity between a captured motion and the drawing it drives."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from copypaw.capture.contacts import (
    CONTACT_HEIGHT_RATIO,
    CONTACT_SPEED_RATIO,
    PAW_INDICES,
    Contacts,
    detect_contacts,
)
from copypaw.capture.travel_root import travel_root_x
from copypaw.capture.video import OUTPUT_FPS
from copypaw.motion.motion_file import MotionFile

__all__ = [
    "CALIBRATION_NAME",
    "CalibrationError",
    "DrawingJoints",
    "Fidelity",
    "PawFidelity",
    "autocorrelation_period",
    "contact_f1",
    "drawing_contacts",
    "duty_factor_error",
    "evaluate",
    "gait_r",
    "is_locomotion",
    "load_calibration",
    "onsets",
    "pearson_r",
    "timing_words",
    "write_fidelity",
]

CALIBRATION_NAME = "calibration.json"
_PAW_COUNT = 4
_JOINT_COUNT = 17
_ONSET_TOLERANCE_FRAMES = 1
_LOCOMOTION_BODY_LENGTHS_PER_SECOND = 0.2
_MIN_ONSETS_PER_PAW = 2
_STRONG_GAIT_R = 0.8
# The drawing's near and far chains share the clip's paw order.
_DRAWING_PAWS = ("left_foot_f", "right_foot_f", "left_foot_b", "right_foot_b")
_DRAWING_LEG_ROOTS = ("left_hip_f", "right_hip_f", "left_hip_b", "right_hip_b")
_CLIP_LEG_ROOTS = (5, 8, 11, 14)
_DRAWING_NECK = "neck"
_DRAWING_ROOT = "root"
_DRAWING_HIP = "hip"
_COORDINATE_DIMENSIONS = 3
_POINT_DIMENSIONS = 2
_CONTACT_DIMENSIONS = 2
_MIN_AUTOCORRELATION_FRAMES = 3
_MIN_FRAMES = 2


@dataclass(frozen=True)
class DrawingJoints:
    """Drawing joints per rendered frame, before and after contact correction.

    Positions are the engine's world coordinates, with y up.
    joints_unfiltered keeps the positions the retargeter produced, which is
    where paw clearance is measured. The corrected joints are conditioned on
    the clip's contacts and cannot independently evidence agreement.
    """

    joints: NDArray[np.float32]
    joints_unfiltered: NDArray[np.float32]
    names: tuple[str, ...]
    fps: float

    @classmethod
    def load(cls, path: Path) -> "DrawingJoints":
        """Read render_joints.npz, accepting archives written before the split."""
        with np.load(path, allow_pickle=False) as archive:
            if "joints" not in archive.files:
                raise ValueError(f"Render joints file is missing 'joints': {path}")
            joints = _joints(archive["joints"], "joints")
            unfiltered = _joints(
                archive["joints_unfiltered"]
                if "joints_unfiltered" in archive.files
                else archive["joints"],
                "joints_unfiltered",
            )
            if unfiltered.shape != joints.shape:
                raise ValueError(
                    "joints_unfiltered must match the shape of joints, "
                    f"got {unfiltered.shape} and {joints.shape}"
                )
            names = (
                tuple(str(name) for name in archive["names"])
                if "names" in archive.files
                else tuple(f"joint_{index}" for index in range(joints.shape[1]))
            )
            fps = float(archive["fps"]) if "fps" in archive.files else float(OUTPUT_FPS)

        if len(names) != joints.shape[1]:
            raise ValueError(
                f"joints has {joints.shape[1]} columns but names has {len(names)} entries"
            )
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Render joints fps must be finite and positive")
        return cls(joints=joints, joints_unfiltered=unfiltered, names=names, fps=fps)


def _joints(value: NDArray[np.generic], name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != _COORDINATE_DIMENSIONS or array.shape[2] != _POINT_DIMENSIONS:
        raise ValueError(f"{name} must have shape (F, J, 2), got {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return array


@dataclass(frozen=True)
class PawFidelity:
    """Metrics for one paw; gait fields are None outside locomotion."""

    contact_f1: float
    duty_factor_error: float | None
    gait_r: float | None
    period_error_ratio: float | None


@dataclass(frozen=True)
class Fidelity:
    """Mean per-paw metrics for one clip and one drawing run."""

    per_paw: tuple[PawFidelity, ...]
    contact_f1: float
    duty_factor_error: float | None
    gait_r: float | None
    period_error_ratio: float | None
    locomotion: bool

    def as_json(self, wording: str | None = None) -> dict[str, object]:
        """Return the stable fields written beside a rendered animation."""
        payload: dict[str, object] = {
            "stage": "uncorrected retarget joints",
            "contact_f1": _json_number(self.contact_f1),
            "duty_factor_error": _json_number(self.duty_factor_error),
            "gait_r": _json_number(self.gait_r),
            "period_error_ratio": _json_number(self.period_error_ratio),
            "locomotion": self.locomotion,
            "per_paw": [
                {
                    "contact_f1": _json_number(paw.contact_f1),
                    "duty_factor_error": _json_number(paw.duty_factor_error),
                    "gait_r": _json_number(paw.gait_r),
                    "period_error_ratio": _json_number(paw.period_error_ratio),
                }
                for paw in self.per_paw
            ],
        }
        if wording is not None:
            payload["timing"] = wording
        return payload


def _json_number(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def onsets(contacts: NDArray[np.bool_]) -> tuple[NDArray[np.int64], ...]:
    """Return rising-edge frame numbers for every paw."""
    if contacts.ndim != _CONTACT_DIMENSIONS or contacts.shape[1] != _PAW_COUNT:
        raise ValueError(f"contacts must have shape (F, {_PAW_COUNT}), got {contacts.shape}")
    return tuple(
        np.flatnonzero(~contacts[:-1, paw] & contacts[1:, paw]) + 1 for paw in range(_PAW_COUNT)
    )


def _onset_f1(expected: NDArray[np.int64], actual: NDArray[np.int64]) -> float:
    remaining = actual.tolist()
    matched = 0
    for frame in expected.tolist():
        best: int | None = None
        for index, candidate in enumerate(remaining):
            if abs(candidate - frame) <= _ONSET_TOLERANCE_FRAMES and (
                best is None or abs(candidate - frame) < abs(remaining[best] - frame)
            ):
                best = index
        if best is not None:
            matched += 1
            remaining.pop(best)
    total = len(expected) + len(actual)
    return 1.0 if total == 0 else 2 * matched / total


def contact_f1(
    clip_contacts: NDArray[np.bool_],
    drawing_contacts: NDArray[np.bool_],
) -> float:
    """F1 of contact onsets matched within one analysis frame, mean over paws."""
    if clip_contacts.shape != drawing_contacts.shape:
        raise ValueError("clip and drawing contacts must have the same shape")
    expected = onsets(clip_contacts)
    actual = onsets(drawing_contacts)
    return float(
        np.mean([_onset_f1(left, right) for left, right in zip(expected, actual, strict=True)])
    )


def duty_factor_error(
    clip_contacts: NDArray[np.bool_], drawing_contacts: NDArray[np.bool_]
) -> float:
    """Mean absolute stance-fraction difference over the paws."""
    if clip_contacts.shape != drawing_contacts.shape:
        raise ValueError("clip and drawing contacts must have the same shape")
    if clip_contacts.ndim != _CONTACT_DIMENSIONS or clip_contacts.shape[1] != _PAW_COUNT:
        raise ValueError(f"contacts must have shape (F, {_PAW_COUNT})")
    if len(clip_contacts) == 0:
        raise ValueError("contacts must contain at least one frame")
    clip_fraction = clip_contacts.mean(axis=0, dtype=np.float64)
    drawing_fraction = drawing_contacts.mean(axis=0, dtype=np.float64)
    return float(np.mean(np.abs(clip_fraction - drawing_fraction)))


def pearson_r(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Pearson correlation, NaN when either curve has no variation."""
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("curves must be one-dimensional and the same length")
    left = first - float(np.mean(first))
    right = second - float(np.mean(second))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0 or not math.isfinite(denominator):
        return math.nan
    return float(np.dot(left, right) / denominator)


def gait_r(clip_heights: NDArray[np.float64], drawing_heights: NDArray[np.float64]) -> float:
    """Correlate the clip's and drawing's leg-length-normalised paw heights."""
    return pearson_r(clip_heights, drawing_heights)


def autocorrelation_period(values: NDArray[np.float64], fps: float) -> float:
    """Period in seconds from the first positive autocorrelation peak."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    centered = values - float(np.mean(values))
    if len(centered) < _MIN_AUTOCORRELATION_FRAMES or not np.any(centered):
        return math.nan
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    if correlation[0] <= 0:
        return math.nan
    correlation = correlation / correlation[0]
    for lag in range(1, len(correlation) - 1):
        if (
            correlation[lag] > 0
            and correlation[lag] >= correlation[lag - 1]
            and correlation[lag] >= correlation[lag + 1]
        ):
            return lag / fps
    return math.nan


def period_error_ratio(
    first: NDArray[np.float64], second: NDArray[np.float64], fps: float
) -> float:
    """Absolute period difference relative to the clip's period."""
    if first.shape != second.shape:
        raise ValueError("curves must be the same length")
    first_period = autocorrelation_period(first, fps)
    second_period = autocorrelation_period(second, fps)
    if not math.isfinite(first_period) or not math.isfinite(second_period) or first_period <= 0:
        return math.nan
    return abs(second_period - first_period) / first_period


def is_locomotion(
    kp: NDArray[np.float32] | MotionFile,
    contacts: NDArray[np.bool_] | None = None,
    timestamps_s: NDArray[np.float64] | None = None,
    travel: NDArray[np.float32] | None = None,
) -> bool:
    """Whether every paw steps at least twice while the root advances fast enough.

    A MotionFile supplies its own keypoints, contacts, time base and root track.
    Speed is the mean absolute root displacement in body lengths per second, so
    a clip that reverses still counts as locomotion. Without a stored root track
    the travel-root integration is recomputed from contacts.
    """
    if isinstance(kp, MotionFile):
        if contacts is not None or timestamps_s is not None:
            raise ValueError("Do not pass contacts or timestamps with a MotionFile")
        motion = kp
        kp = motion.kp
        contacts = motion.contacts
        timestamps_s = np.arange(len(motion.kp), dtype=np.float64) / motion.fps
        travel = motion.travel_root_x if travel is None else travel
    if contacts is None or timestamps_s is None:
        raise ValueError("kp, contacts and timestamps are required")
    if contacts.ndim != _CONTACT_DIMENSIONS or contacts.shape[1] != _PAW_COUNT:
        raise ValueError(f"contacts must have shape (F, {_PAW_COUNT}), got {contacts.shape}")
    if len(kp) != len(contacts) or len(timestamps_s) != len(kp):
        raise ValueError("kp, contacts and timestamps must have the same frame count")
    for frames in onsets(contacts):
        if len(frames) < _MIN_ONSETS_PER_PAW:
            return False
    if len(kp) < _MIN_FRAMES:
        return False
    if travel is None:
        travel = travel_root_x(kp, contacts, timestamps_s)
    if travel.shape != (len(kp),) or not np.isfinite(travel).all():
        raise ValueError("travel must be finite with one value per frame")
    durations = np.diff(timestamps_s)
    if np.any(durations <= 0):
        raise ValueError("timestamps must be strictly increasing")
    speed = float(np.mean(np.abs(np.diff(travel)) / durations))
    return speed > _LOCOMOTION_BODY_LENGTHS_PER_SECOND


def _resample(
    joints: NDArray[np.float32], source_fps: float, target_fps: float, count: int
) -> NDArray[np.float32]:
    if count < 1:
        raise ValueError("count must be positive")
    if len(joints) == 1:
        return np.repeat(joints[:1], count, axis=0)
    times_s = np.arange(count, dtype=np.float64) / target_fps
    position = np.clip(times_s * source_fps, 0.0, len(joints) - 1)
    lower = np.minimum(np.floor(position).astype(np.intp), len(joints) - 1)
    upper = np.minimum(lower + 1, len(joints) - 1)
    weight = (position - lower).reshape(-1, 1, 1)
    values = joints.astype(np.float64)
    return np.asarray(values[lower] + (values[upper] - values[lower]) * weight, dtype=np.float32)


def _indexed(names: tuple[str, ...], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as error:
        raise ValueError(f"Drawing joints are missing {name!r}") from error


def _root_name(names: tuple[str, ...]) -> str:
    return _DRAWING_ROOT if _DRAWING_ROOT in names else _DRAWING_HIP


def _clip_axes(points: NDArray[np.float32], names: tuple[str, ...]) -> NDArray[np.float32]:
    """Turn the engine's joints into the clip's axes: y down and facing +x.

    The engine reports world positions with y up, and a drawing stored mirrored
    comes back facing left. Ground height and forward swing are read in the
    axes of the right-facing clip, so a paw lifted in the drawing is lifted here.
    """
    turned = np.array(points, dtype=np.float32)
    turned[..., 1] *= -1
    neck = turned[:, _indexed(names, _DRAWING_NECK), 0]
    root = turned[:, _indexed(names, _root_name(names)), 0]
    if float(np.median(neck - root)) < 0:
        turned[..., 0] *= -1
    return turned


def _drawing_frame(points: NDArray[np.float32], names: tuple[str, ...]) -> NDArray[np.float32]:
    root = points[:, _indexed(names, _root_name(names))]
    fake = np.repeat(root[:, None, :], _JOINT_COUNT, axis=1)
    fake[:, 3] = points[:, _indexed(names, _DRAWING_NECK)]
    fake[:, 4] = root
    fake[:, list(PAW_INDICES)] = np.stack(
        [points[:, _indexed(names, name)] for name in _DRAWING_PAWS], axis=1
    )
    return fake


def _drawing_stance(
    points: NDArray[np.float32],
    names: tuple[str, ...],
    fps: float,
    *,
    height_ratio: float,
    speed_ratio: float,
) -> Contacts:
    fake = _drawing_frame(points, names)
    return detect_contacts(
        np.asarray(fake, dtype=np.float32),
        np.arange(len(points), dtype=np.float64) / fps,
        scores=np.ones((len(points), _JOINT_COUNT), dtype=np.float32),
        height_ratio=height_ratio,
        speed_ratio=speed_ratio,
    )


def drawing_contacts(
    drawing: DrawingJoints,
    *,
    height_ratio: float = CONTACT_HEIGHT_RATIO,
    speed_ratio: float = CONTACT_SPEED_RATIO,
) -> NDArray[np.bool_]:
    """Detect stance on the drawing's unfiltered joints at its own frame rate.

    The footfall strip shows what the drawing did before contact correction;
    the corrected joints would only repeat the clip's contacts.
    """
    stance = _drawing_stance(
        _clip_axes(drawing.joints_unfiltered, drawing.names),
        drawing.names,
        drawing.fps,
        height_ratio=height_ratio,
        speed_ratio=speed_ratio,
    )
    return stance.contacts


def _leg_lengths(paws: NDArray[np.float64], roots: NDArray[np.float64]) -> NDArray[np.float64]:
    lengths = np.linalg.norm(paws - roots, axis=2)
    median = np.median(lengths, axis=0)
    if np.any(~np.isfinite(median)) or np.any(median <= 0):
        raise ValueError("Every paw needs a positive median leg length")
    return median


def _height_curves(
    paws: NDArray[np.float64],
    ground_line: NDArray[np.float64],
    leg_lengths: NDArray[np.float64],
) -> NDArray[np.float64]:
    if ground_line.shape != (2,) or not np.isfinite(ground_line).all():
        raise ValueError("A finite ground line is required for gait metrics")
    height = ground_line[0] * paws[..., 0] + ground_line[1] - paws[..., 1]
    return height / leg_lengths


def evaluate(
    clip: MotionFile,
    drawing: DrawingJoints,
    *,
    height_ratio: float = CONTACT_HEIGHT_RATIO,
    speed_ratio: float = CONTACT_SPEED_RATIO,
) -> Fidelity:
    """Measure matched fidelity between a clip and one drawing run.

    Drawing contacts and paw clearance come from the unfiltered joints at the
    clip's analysis rate. Gait correlation, period and duty factor are reported
    only when the clip is locomotion; contact F1 is always reported.
    """
    if not math.isfinite(clip.fps) or clip.fps <= 0:
        raise ValueError("Clip fps must be finite and positive")
    if len(clip.kp) == 0:
        raise ValueError("Clip must contain at least one frame")
    if len(clip.kp) != len(clip.contacts):
        raise ValueError("Clip kp and contacts must have the same frame count")
    available = (
        math.floor((len(drawing.joints) - 1) / drawing.fps * clip.fps) + 1
        if len(drawing.joints) > 1
        else 1
    )
    count = min(len(clip.kp), max(1, available))
    points = _clip_axes(
        _resample(drawing.joints_unfiltered, drawing.fps, clip.fps, count), drawing.names
    )
    clip_contacts = clip.contacts[:count]
    timestamps_s = np.arange(len(clip.kp), dtype=np.float64) / clip.fps
    locomotion = is_locomotion(clip.kp, clip.contacts, timestamps_s, clip.travel_root_x)
    stance = _drawing_stance(
        points, drawing.names, clip.fps, height_ratio=height_ratio, speed_ratio=speed_ratio
    )

    gait_values = _gait_values(
        clip, drawing, points, stance=stance, count=count, locomotion=locomotion
    )
    clip_onsets = onsets(clip_contacts)
    drawing_onsets = onsets(stance.contacts)
    per_paw = tuple(
        PawFidelity(
            contact_f1=_onset_f1(clip_onsets[paw], drawing_onsets[paw]),
            duty_factor_error=(
                float(
                    abs(
                        float(np.mean(clip_contacts[:, paw]))
                        - float(np.mean(stance.contacts[:, paw]))
                    )
                )
                if locomotion
                else None
            ),
            gait_r=gait_values[paw][0],
            period_error_ratio=gait_values[paw][1],
        )
        for paw in range(_PAW_COUNT)
    )
    return Fidelity(
        per_paw=per_paw,
        contact_f1=float(np.mean([paw.contact_f1 for paw in per_paw])),
        duty_factor_error=_mean_optional([paw.duty_factor_error for paw in per_paw]),
        gait_r=_mean_optional([paw.gait_r for paw in per_paw]),
        period_error_ratio=_mean_optional([paw.period_error_ratio for paw in per_paw]),
        locomotion=locomotion,
    )


def _gait_values(
    clip: MotionFile,
    drawing: DrawingJoints,
    points: NDArray[np.float32],
    *,
    stance: Contacts,
    count: int,
    locomotion: bool,
) -> tuple[tuple[float | None, float | None], ...]:
    if not locomotion:
        return tuple((None, None) for _ in range(_PAW_COUNT))
    clip_paws = clip.kp[:count, list(PAW_INDICES)].astype(np.float64)
    clip_roots = np.stack(
        [clip.kp[:count, joint].astype(np.float64) for joint in _CLIP_LEG_ROOTS], axis=1
    )
    drawing_paws = np.stack(
        [points[:, _indexed(drawing.names, name)] for name in _DRAWING_PAWS], axis=1
    ).astype(np.float64)
    drawing_roots = np.stack(
        [points[:, _indexed(drawing.names, name)] for name in _DRAWING_LEG_ROOTS], axis=1
    ).astype(np.float64)
    clip_heights = _height_curves(
        clip_paws, clip.ground_line_px, _leg_lengths(clip_paws, clip_roots)
    )
    drawing_heights = _height_curves(
        drawing_paws, stance.ground_line_px, _leg_lengths(drawing_paws, drawing_roots)
    )
    return tuple(
        (
            _finite_or_none(pearson_r(clip_heights[:, paw], drawing_heights[:, paw])),
            _finite_or_none(
                period_error_ratio(clip_heights[:, paw], drawing_heights[:, paw], clip.fps)
            ),
        )
        for paw in range(_PAW_COUNT)
    )


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _mean_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


class CalibrationError(RuntimeError):
    """The gallery calibration has not been written or holds no threshold."""


def calibration_path() -> Path:
    """Return the package's calibration file."""
    return Path(__file__).with_name(CALIBRATION_NAME)


def load_calibration(path: Path | None = None) -> float:
    """Read the mismatched-pair 95th percentile gait correlation."""
    source = calibration_path() if path is None else path
    if not source.is_file():
        raise CalibrationError(
            f"Missing gait calibration: {source}. Run scripts/eval_motion.py on the gallery."
        )
    data = cast("dict[str, object]", json.loads(source.read_text(encoding="utf-8")))
    value = data.get("gait_r_p95")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CalibrationError(
            f"Gait calibration has no gait_r_p95: {source}. Run scripts/eval_motion.py."
        )
    return float(value)


def timing_words(gait_r: float | None, *, locomotion: bool, percentile: float) -> str:
    """Turn a matched gait correlation into the app's plain timing words."""
    if not locomotion or gait_r is None or not math.isfinite(gait_r):
        return "timing score not shown: no repeating steps found"
    if gait_r > percentile and gait_r > _STRONG_GAIT_R:
        return "timing match: strong"
    if gait_r > percentile:
        return "timing match: fair"
    return "timing match: weak"


def write_fidelity(path: Path, result: Fidelity, *, percentile: float | None) -> None:
    """Write the matched metrics and, when calibrated, the plain timing words."""
    wording = (
        None
        if percentile is None
        else timing_words(result.gait_r, locomotion=result.locomotion, percentile=percentile)
    )
    path.write_text(
        json.dumps(result.as_json(wording), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
