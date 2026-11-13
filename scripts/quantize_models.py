"""Make fp16 and int8 ViTPose-S files for the browser and keep the smallest that holds accuracy.

Each variant is re-scored on APT-36K with the pose evaluation's conditions. A
variant is kept only when every condition loses under 1 point of AP and of
PCK against fp32, and it fits the browser budget beside YOLOX-tiny. YOLOX-tiny
stays fp32 because the pose evaluation scores ground-truth boxes, so nothing
here measures a quantised detector.
"""

import argparse
import importlib
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from copypaw.capture.detect import Track
from copypaw.capture.pose import estimate_keypoints
from copypaw.capture.video import Frames
from copypaw.models import MODELS, REGISTRY, Mode, ensure_models

try:
    from scripts.eval_pose import Split, find_splits, resolve_tracks, score_poses
except ModuleNotFoundError:
    from eval_pose import Split, find_splits, resolve_tracks, score_poses

PRECISIONS = ("fp32", "fp16", "int8")
MAX_DROP_POINTS = 1.0
BUDGET_BYTES = 70_000_000
MODE = Mode.LIGHTWEIGHT
DETECTOR_KEY, POSE_KEY = MODELS[MODE]
CONDITIONS = (
    ("raw", False, False),
    ("+ flip test", False, True),
    ("+ identity", True, False),
    ("+ identity + flip test", True, True),
)
_BATCH = 16

type Poses = tuple[NDArray[np.float32], NDArray[np.float32]]
type Predict = Callable[[Split, Path, bool], Poses]


@dataclass(frozen=True)
class Variant:
    """One precision of the pose model, its size, and AP and PCK in points per condition."""

    precision: str
    path: Path
    size_bytes: int
    scores: Mapping[str, tuple[float, float]]


def worst_drops(baseline: Variant, variant: Variant) -> tuple[float, float]:
    """Return the largest AP and PCK losses, in points, over every condition."""
    ap = max(baseline.scores[label][0] - variant.scores[label][0] for label in baseline.scores)
    pck = max(baseline.scores[label][1] - variant.scores[label][1] for label in baseline.scores)
    return ap, pck


def choose_variant(variants: Sequence[Variant], detector_bytes: int) -> Variant | None:
    """Return the smallest variant within the accuracy and size limits, or None."""
    baseline = next(variant for variant in variants if variant.precision == "fp32")
    kept = [
        variant
        for variant in variants
        if max(worst_drops(baseline, variant)) < MAX_DROP_POINTS
        and variant.size_bytes + detector_bytes < BUDGET_BYTES
    ]
    return min(kept, key=lambda variant: variant.size_bytes, default=None)


def make_variant(source: Path, precision: str, out_dir: Path) -> Path:
    """Write source at precision into out_dir; fp32 is the source itself."""
    if precision == "fp32":
        return source
    out = out_dir / f"{source.stem}-{precision}.onnx"
    # Both converters need the onnx package, which only this script uses.
    if precision == "fp16":
        onnx = importlib.import_module("onnx")
        float16 = importlib.import_module("onnxruntime.transformers.float16")
        onnx.save(float16.convert_float_to_float16(onnx.load(str(source))), str(out))
    elif precision == "int8":
        quantization = importlib.import_module("onnxruntime.quantization")
        quantization.quantize_dynamic(
            str(source), str(out), weight_type=quantization.QuantType.QInt8
        )
    else:
        raise ValueError(f"Unknown precision {precision!r}")
    return out


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def predict(split: Split, pose_path: Path, flip_test: bool) -> Poses:
    """Run one ViTPose file on every ground-truth box, batching crops of same-sized images."""
    models = {POSE_KEY: pose_path}
    kp = np.full((len(split.instances), 17, 2), np.nan, dtype=np.float32)
    scores = np.zeros((len(split.instances), 17), dtype=np.float32)
    by_size: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, instance in enumerate(split.instances):
        by_size[instance.size].append(index)
    for indices in by_size.values():
        for start in range(0, len(indices), _BATCH):
            chunk = indices[start : start + _BATCH]
            members = [split.instances[index] for index in chunk]
            frames = Frames(
                rgb=np.stack([_read_rgb(item.image) for item in members]),
                t_s=np.arange(len(chunk), dtype=np.float64),
                source_fps=1.0,
                fps=1.0,
                trimmed=False,
            )
            track = Track(
                boxes=np.stack([item.box for item in members]),
                present=np.ones(len(chunk), dtype=np.bool_),
            )
            result = estimate_keypoints(frames, track, MODE, flip_test, models)
            kp[chunk], scores[chunk] = result.kp, result.scores
    return kp, scores


def score_variant(
    split: Split, pose_path: Path, run: Predict = predict
) -> dict[str, tuple[float, float]]:
    """Return AP and overall PCK, in points, for each evaluation condition."""
    poses = {flip: run(split, pose_path, flip) for flip in (False, True)}
    resolved = {flip: resolve_tracks(split, *poses[flip]) for flip in (False, True)}
    scores: dict[str, tuple[float, float]] = {}
    for label, identity, flip in CONDITIONS:
        kp, conf = (resolved if identity else poses)[flip]
        result = score_poses(split, kp, conf, None)
        scores[label] = (100 * result.ap, 100 * (result.pck.overall or 0.0))
    return scores


def report(variants: Sequence[Variant], chosen: Variant | None, detector_bytes: int) -> str:
    """Describe each variant and the choice as plain text."""
    baseline = next(variant for variant in variants if variant.precision == "fp32")
    lines = [f"YOLOX-tiny fp32: {detector_bytes} bytes"]
    for variant in variants:
        ap, pck = worst_drops(baseline, variant)
        lines.append(
            f"ViTPose-S {variant.precision}: {variant.size_bytes} bytes, "
            f"worst AP drop {ap:.2f}, worst PCK drop {pck:.2f}"
        )
    lines.append(f"Kept: {chosen.precision if chosen else 'none'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apt36k", type=Path, required=True, help="Local APT-36K directory")
    parser.add_argument("--images", type=Path, help="Image root, default the APT-36K directory")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the variants")
    args = parser.parse_args()
    models = ensure_models(MODE)
    split = find_splits(args.apt36k, args.images or args.apt36k)[0]
    args.out.mkdir(parents=True, exist_ok=True)
    variants: list[Variant] = []
    for precision in PRECISIONS:
        path = make_variant(models[POSE_KEY], precision, args.out)
        variants.append(Variant(precision, path, path.stat().st_size, score_variant(split, path)))
    detector_bytes = REGISTRY[DETECTOR_KEY].size_bytes
    chosen = choose_variant(variants, detector_bytes)
    print(report(variants, chosen, detector_bytes))
    return 0 if chosen else 1


if __name__ == "__main__":
    sys.exit(main())
