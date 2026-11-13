"""Measure pose accuracy and limb identity on APT-36K and write the report."""

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from copypaw.capture.detect import Track
from copypaw.capture.identity import resolve_identity
from copypaw.capture.pose import KEYPOINT_NAMES, estimate_keypoints
from copypaw.capture.video import Frames
from copypaw.evaluate.identity_swaps import Flips, align_chains, count_flips
from copypaw.evaluate.pck import Detection, GroundTruth, Pck, average_precision, pck
from copypaw.models import Mode, ensure_models

try:
    from scripts.bench import pose_ms_per_frame
except ModuleNotFoundError:
    from bench import pose_ms_per_frame

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/evaluation.md"
ADR = ROOT / "docs/adr/ADR-001.md"
DOCS_HEADING = "## Pose"
ADR_HEADING = "## Pose evaluation"
MODES = (Mode.LIGHTWEIGHT, Mode.BALANCED, Mode.ACCURATE)
MODEL_NAMES = {
    Mode.LIGHTWEIGHT: "ViTPose-S",
    Mode.BALANCED: "ViTPose-B",
    Mode.ACCURATE: "ViTPose-L",
}
# APT-36K val AP from the ViTPose README, which evaluates with a flip test.
README_AP = {Mode.LIGHTWEIGHT: 74.2, Mode.BALANCED: 75.9, Mode.ACCURATE: 80.8}
TOLERANCE_AP = 1.5
_CONDITIONS = (
    ("raw", False, False),
    ("+ flip test", False, True),
    ("+ identity", True, False),
    ("+ identity + flip test", True, True),
)
_JOINT_COUNT = len(KEYPOINT_NAMES)
_VISIBILITY_THRESHOLD = 0.2
_DOG_CAT = {"dog", "cat"}
_BATCH = 16

type _Poses = tuple[NDArray[np.float32], NDArray[np.float32]]
type Predictor = Callable[["Split", Mode, bool], _Poses]
type Timer = Callable[[Mode, bool], float]


@dataclass(frozen=True)
class Instance:
    """One labelled animal in one image, with its cleaned box and ground-truth joints."""

    image: Path
    image_id: int
    size: tuple[int, int]
    category: int
    box: NDArray[np.float32]
    kp: NDArray[np.float64]
    visible: NDArray[np.bool_]
    area: float
    box_area: float


@dataclass(frozen=True)
class AnimalTrack:
    """One animal followed through a clip: its instances in frame order."""

    length: int
    width: int
    indices: NDArray[np.intp]
    frames: NDArray[np.intp]


@dataclass(frozen=True)
class Split:
    """The labelled animals of one annotation file and the species counted as dog or cat."""

    name: str
    instances: tuple[Instance, ...]
    tracks: tuple[AnimalTrack, ...]
    dog_cat: frozenset[int]


@dataclass(frozen=True)
class Scores:
    """Accuracy of one set of predictions on one species subset."""

    ap: float
    pck: Pck
    flips: Flips


@dataclass(frozen=True)
class Row:
    """One model and processing condition on both species subsets."""

    mode: Mode
    label: str
    ms_per_frame: float
    all_species: Scores
    dog_cat: Scores


@dataclass(frozen=True)
class SplitReport:
    """Every row measured on one split."""

    name: str
    animals: int
    images: int
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Reproduction:
    """A measured AP against the ViTPose README figure for the same model."""

    mode: Mode
    reference: float
    measured: float

    @property
    def gap(self) -> float:
        """Measured minus reference, in AP points."""
        return self.measured - self.reference

    @property
    def passed(self) -> bool:
        """Whether the pipeline reproduces the published number."""
        return abs(self.gap) <= TOLERANCE_AP


def _sequence(image: dict[str, Any]) -> str:
    for key in ("vid_id", "video_id"):
        if key in image:
            return str(image[key])
    return Path(str(image["file_name"])).parent.as_posix()


def _clean_box(box: list[float], width: int, height: int) -> list[float] | None:
    """Clip a COCO xywh box to the image the way the ViTPose dataset loader does."""
    x, y, w, h = box[:4]
    x1, y1 = max(0.0, x), max(0.0, y)
    x2 = min(width - 1.0, x1 + max(0.0, w - 1))
    y2 = min(height - 1.0, y1 + max(0.0, h - 1))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def load_split(path: Path, images_dir: Path, name: str) -> Split:
    """Read a COCO-style annotation file into labelled animals and per-animal tracks.

    Images carry an optional clip id (vid_id or video_id) and frame_id; without
    them the clip is the image's folder and frames follow file-name order.
    Animals are followed by track_id, or by their order within the image.
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]).lower() for item in data["categories"]}
    dog_cat = frozenset(key for key, label in categories.items() if label in _DOG_CAT)
    if {categories[key] for key in dog_cat} != _DOG_CAT:
        raise ValueError(f"{path} must name both a dog and a cat category")
    images = {int(item["id"]): item for item in data["images"]}
    clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images.values():
        clips[_sequence(image)].append(image)
    position: dict[int, tuple[str, int]] = {}
    for clip, members in clips.items():
        members.sort(key=lambda item: (item.get("frame_id", 0), str(item["file_name"])))
        position.update({int(item["id"]): (clip, rank) for rank, item in enumerate(members)})
    instances: list[Instance] = []
    keys: list[tuple[str, int]] = []
    frames: list[int] = []
    seen_in_image: dict[int, int] = defaultdict(int)
    for annotation in sorted(data["annotations"], key=lambda item: int(item["id"])):
        image_id = int(annotation["image_id"])
        order = seen_in_image[image_id]
        seen_in_image[image_id] += 1
        raw = np.array(annotation.get("keypoints", ()), dtype=np.float64)
        if annotation.get("iscrowd") or "bbox" not in annotation or raw.size != _JOINT_COUNT * 3:
            continue
        if raw.max() == 0 or annotation.get("num_keypoints", 1) == 0:
            continue
        image = images[image_id]
        width, height = int(image["width"]), int(image["height"])
        box = _clean_box(annotation["bbox"], width, height)
        area = float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]))
        if box is None or area <= 0:
            continue
        joints = raw.reshape(_JOINT_COUNT, 3)
        clip, frame = position[image_id]
        keys.append((clip, int(annotation.get("track_id", order))))
        frames.append(frame)
        instances.append(
            Instance(
                image=images_dir / str(image["file_name"]),
                image_id=image_id,
                size=(height, width),
                category=int(annotation["category_id"]),
                box=np.array(box, dtype=np.float32),
                kp=joints[:, :2],
                visible=joints[:, 2] > 0,
                area=area,
                box_area=float(annotation["bbox"][2] * annotation["bbox"][3]),
            )
        )
    if not instances:
        raise ValueError(f"No labelled animals in {path}")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        grouped[key].append(index)
    tracks: list[AnimalTrack] = []
    for (clip, _), members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda index: frames[index])
        tracks.append(
            AnimalTrack(
                length=len(clips[clip]),
                width=instances[ordered[0]].size[1],
                indices=np.array(ordered, dtype=np.intp),
                frames=np.array([frames[index] for index in ordered], dtype=np.intp),
            )
        )
    return Split(name, tuple(instances), tuple(tracks), dog_cat)


def find_splits(root: Path, images_dir: Path) -> tuple[Split, ...]:
    """Load the validation split, and the test split when present, refusing training overlap."""
    found: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*_annotations_1.json")):
        found[path.name].append(path)
    for name, paths in found.items():
        if len(paths) > 1:
            raise ValueError(f"Several files named {name} under {root}")
    if "val_annotations_1.json" not in found:
        raise FileNotFoundError(f"No val_annotations_1.json under {root}")
    splits = [
        load_split(found[file][0], images_dir, name)
        for name, file in (("val", "val_annotations_1.json"), ("test", "test_annotations_1.json"))
        if file in found
    ]
    if "train_annotations_1.json" in found:
        training = json.loads(found["train_annotations_1.json"][0].read_text(encoding="utf-8"))
        trained = {str(item["file_name"]) for item in training["images"]}
        for split in splits:
            overlap = trained & {
                str(inst.image.relative_to(images_dir)) for inst in split.instances
            }
            if overlap:
                raise ValueError(f"{len(overlap)} {split.name} images are also in the train split")
    return tuple(splits)


def _read_rgb(path: Path, size: tuple[int, int]) -> NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    if rgb.shape[:2] != size:
        raise ValueError(f"{path} is {rgb.shape[:2]}, annotations say {size}")
    return rgb


@cache
def _model_paths(mode: Mode) -> dict[str, Path]:
    return ensure_models(mode)


def predict_vitpose(split: Split, mode: Mode, flip_test: bool) -> _Poses:
    """Run ViTPose on every ground-truth box, batching crops from same-sized images."""
    models = _model_paths(mode)
    kp = np.full((len(split.instances), _JOINT_COUNT, 2), np.nan, dtype=np.float32)
    scores = np.zeros((len(split.instances), _JOINT_COUNT), dtype=np.float32)
    by_size: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, instance in enumerate(split.instances):
        by_size[instance.size].append(index)
    done = 0
    for size, indices in by_size.items():
        for start in range(0, len(indices), _BATCH):
            chunk = indices[start : start + _BATCH]
            members = [split.instances[index] for index in chunk]
            frames = Frames(
                rgb=np.stack([_read_rgb(member.image, size) for member in members]),
                t_s=np.arange(len(chunk), dtype=np.float64),
                source_fps=1.0,
                fps=1.0,
                trimmed=False,
            )
            track = Track(
                boxes=np.stack([member.box for member in members]),
                present=np.ones(len(chunk), dtype=np.bool_),
            )
            result = estimate_keypoints(frames, track, mode, flip_test, models)
            kp[chunk], scores[chunk] = result.kp, result.scores
            done += len(chunk)
            print(
                f"{split.name} {mode.value} flip_test={flip_test}: {done}/{len(kp)}",
                file=sys.stderr,
            )
    return kp, scores


def time_pose(mode: Mode, flip_test: bool) -> float:
    """Milliseconds per frame from the benchmark's sample clip."""
    return pose_ms_per_frame(mode, flip_test, _model_paths(mode))


def _dense(track: AnimalTrack, values: NDArray[Any], fill: float | bool) -> NDArray[Any]:
    dense = np.full((track.length, *values.shape[1:]), fill, dtype=values.dtype)
    dense[track.frames] = values[track.indices]
    return dense


def resolve_tracks(split: Split, kp: NDArray[np.float32], conf: NDArray[np.float32]) -> _Poses:
    """Apply identity resolution per animal, in the labels' own orientation and naming."""
    resolved_kp, resolved_conf = kp.copy(), conf.copy()
    truth = np.stack([instance.kp for instance in split.instances])
    visible = np.stack([instance.visible for instance in split.instances])
    for track in split.tracks:
        result = resolve_identity(
            _dense(track, kp, np.nan), _dense(track, conf, 0.0), frame_width=track.width
        )
        joints = result.kp.copy()
        if result.facing == "left":
            joints[..., 0] = track.width - 1 - joints[..., 0]
        joints, confidence = align_chains(
            joints, result.scores, _dense(track, truth, np.nan), _dense(track, visible, False)
        )
        resolved_kp[track.indices] = joints[track.frames]
        resolved_conf[track.indices] = confidence[track.frames]
    return resolved_kp, resolved_conf


def _pose_score(conf: NDArray[np.float32]) -> float:
    """Mean confidence of the joints above the visibility threshold, as ViTPose rescores."""
    confident = conf[conf > _VISIBILITY_THRESHOLD]
    return float(confident.mean()) if confident.size else 0.0


def score_poses(
    split: Split,
    kp: NDArray[np.float32],
    conf: NDArray[np.float32],
    categories: frozenset[int] | None,
) -> Scores:
    """Score predictions on the animals of categories, or on every animal when None."""
    chosen = [
        index
        for index, instance in enumerate(split.instances)
        if categories is None or instance.category in categories
    ]
    members = [split.instances[index] for index in chosen]
    truths = [
        GroundTruth(item.image_id, item.category, item.kp, item.visible, item.area)
        for item in members
    ]
    detections = [
        Detection(
            split.instances[index].image_id,
            split.instances[index].category,
            kp[index].astype(np.float64),
            _pose_score(conf[index]),
        )
        for index in chosen
    ]
    accuracy = pck(
        kp[chosen].astype(np.float64),
        np.stack([item.kp for item in members]),
        np.stack([item.visible for item in members]),
        np.array([item.box_area for item in members]),
    )
    truth = np.stack([instance.kp for instance in split.instances])
    visible = np.stack([instance.visible for instance in split.instances])
    flips = Flips(0, 0)
    for track in split.tracks:
        category = split.instances[int(track.indices[0])].category
        if categories is None or category in categories:
            flips += count_flips(
                _dense(track, kp.astype(np.float64), np.nan),
                _dense(track, truth, np.nan),
                _dense(track, visible, False),
            )
    return Scores(average_precision(truths, detections), accuracy, flips)


def measure_split(split: Split, predict: Predictor, timer: Timer) -> SplitReport:
    """Measure every model under the four conditions."""
    rows: list[Row] = []
    for mode in MODES:
        poses = {flip: predict(split, mode, flip) for flip in (False, True)}
        milliseconds = {flip: timer(mode, flip) for flip in (False, True)}
        resolved = {flip: resolve_tracks(split, *poses[flip]) for flip in (False, True)}
        for label, identity, flip in _CONDITIONS:
            kp, conf = (resolved if identity else poses)[flip]
            rows.append(
                Row(
                    mode,
                    label,
                    milliseconds[flip],
                    score_poses(split, kp, conf, None),
                    score_poses(split, kp, conf, split.dog_cat),
                )
            )
    return SplitReport(
        split.name,
        len(split.instances),
        len({instance.image_id for instance in split.instances}),
        tuple(rows),
    )


def reproduction(report: SplitReport) -> tuple[Reproduction, ...]:
    """Compare raw + flip test AP over all species with the ViTPose README."""
    return tuple(
        Reproduction(row.mode, README_AP[row.mode], 100 * row.all_species.ap)
        for row in report.rows
        if row.label == "+ flip test"
    )


def flip_default(report: SplitReport) -> tuple[bool, float, float]:
    """Choose the balanced flip_test default from the identity rows, which capture always runs."""
    overall = {
        row.label: row.all_species.pck.overall or 0.0
        for row in report.rows
        if row.mode is Mode.BALANCED
    }
    without, with_flip = overall["+ identity"], overall["+ identity + flip test"]
    return with_flip > without, without, with_flip


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def _flips(value: Flips) -> str:
    return "n/a" if value.rate is None else f"{value.flips}/{value.opportunities}"


def _table(lines: list[str], header: tuple[str, ...]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *lines,
    ]


def pose_section(reports: tuple[SplitReport, ...], cause: str | None) -> str:
    """Render the Pose section of docs/evaluation.md."""
    checks = reproduction(reports[0])
    passed = all(check.passed for check in checks)
    lines = [
        DOCS_HEADING,
        "",
        "Generated by `scripts/eval_pose.py`; edit the generator, not this section.",
        "",
        "APT-36K is in-distribution: the weights trained on its train split, so these",
        "rows measure the pipeline, not generalisation. Its 15-frame clips are subsampled",
        "at varying rates, so smoothing is not evaluated here. The [field test](field_test.md)",
        "is the out-of-distribution measure.",
        "",
        "Poses come from the ground-truth boxes. AP is OKS AP with the AP-10K sigmas,",
        "averaged over species and without OKS suppression. PCK@0.05 counts a joint as",
        "correct within 0.05 * sqrt(box area) of its label. An identity flip is a change",
        "between consecutive frames in which pairing the predicted limb chains with the",
        "labelled ones crosswise is cheaper, over front and back leg pairs. Identity rows",
        "resolve each animal's chains over its clip and name them after the labels they",
        "follow best. ms per frame times the pose stage alone on the benchmark sample clip,",
        "with and without the flip test.",
        "",
        "### Reproduction check",
        "",
        "Raw + flip test AP over all species on the validation split, against the ViTPose README.",
        "",
        *_table(
            [
                f"| {MODEL_NAMES[check.mode]} | {check.reference:.1f} | {check.measured:.1f} "
                f"| {check.gap:+.1f} | {'pass' if check.passed else 'fail'} |"
                for check in checks
            ],
            ("Model", "README AP", "Measured AP", "Gap", f"Within {TOLERANCE_AP} points"),
        ),
        "",
    ]
    if passed:
        lines.append(
            "The pipeline reproduces the published numbers; the rows below can be trusted."
        )
    elif cause:
        lines.append(f"The check failed. Cause, recorded in ADR-001: {cause}")
    else:
        lines.append(
            "The check failed and no cause is recorded yet, so no other row is trusted. "
            "Investigate colour order, crop padding, normalisation and DARK-UDP decoding."
        )
    for report in reports:
        lines.extend(
            [
                "",
                f"### {report.name.capitalize()} split",
                "",
                f"{report.animals} labelled animals in {report.images} images.",
                "",
                *_table(
                    [
                        f"| {MODEL_NAMES[row.mode]} | {row.label} "
                        f"| {_percent(row.all_species.ap)} | {_percent(row.dog_cat.ap)} "
                        f"| {_percent(row.all_species.pck.overall)} "
                        f"| {_percent(row.dog_cat.pck.overall)} "
                        f"| {_flips(row.all_species.flips)} | {_flips(row.dog_cat.flips)} "
                        f"| {row.ms_per_frame:.1f} |"
                        for row in report.rows
                    ],
                    (
                        "Model",
                        "Row",
                        "AP all",
                        "AP dog and cat",
                        "PCK@0.05 all",
                        "PCK@0.05 dog and cat",
                        "Flips all",
                        "Flips dog and cat",
                        "ms per frame",
                    ),
                ),
                "",
                "PCK@0.05 per joint, all species:",
                "",
                *_table(
                    [
                        f"| {MODEL_NAMES[row.mode]} | {row.label} | "
                        + " | ".join(_percent(value) for value in row.all_species.pck.per_joint)
                        + " |"
                        for row in report.rows
                    ],
                    ("Model", "Row", *KEYPOINT_NAMES),
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def adr_section(reports: tuple[SplitReport, ...], cause: str | None) -> str:
    """Render the pose evaluation record for ADR-001."""
    checks = reproduction(reports[0])
    measured = ", ".join(
        f"{MODEL_NAMES[check.mode]} {check.measured:.1f} against {check.reference:.1f}"
        for check in checks
    )
    if all(check.passed for check in checks):
        outcome = f"All three are within {TOLERANCE_AP} points, so the pipeline is trusted."
    elif cause:
        outcome = f"At least one is outside {TOLERANCE_AP} points. Cause: {cause}"
    else:
        outcome = (
            f"At least one is outside {TOLERANCE_AP} points and I have not found the cause. "
            "No other row is trusted until colour order, crop padding, normalisation and "
            "DARK-UDP decoding are cleared."
        )
    use_flip, without, with_flip = flip_default(reports[0])
    return (
        f"{ADR_HEADING}\n\n"
        "I evaluated the three pose models on the APT-36K validation split with "
        "ground-truth boxes. Raw + flip test AP over all species: "
        f"{measured}. {outcome}\n\n"
        "The flip_test default for balanced is "
        f"{'on' if use_flip else 'off'}: overall PCK@0.05 over all species with identity "
        f"resolution is {_percent(without)} without it and {_percent(with_flip)} with it. "
        "APT-36K is in-distribution, so the field test decides whether this holds on pets "
        "the weights never saw.\n"
    )


def replace_section(document: str, heading: str, section: str) -> str:
    """Replace the section under an exact heading line, or append it, keeping the rest."""
    start = re.search(f"^{re.escape(heading)}$", document, re.MULTILINE)
    if start is None:
        if not document:
            return section
        return document.rstrip("\n") + "\n\n" + section
    following = re.compile(r"^## ", re.MULTILINE).search(document, start.end())
    rest = "" if following is None else "\n" + document[following.start() :]
    return document[: start.start()] + section + rest


def _write_section(path: Path, heading: str, section: str) -> None:
    document = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replace_section(document, heading, section), encoding="utf-8")


def run(
    *,
    apt36k: Path,
    images: Path,
    docs_path: Path,
    adr_path: Path,
    cause: str | None,
    predict: Predictor = predict_vitpose,
    timer: Timer = time_pose,
) -> int:
    reports = tuple(measure_split(split, predict, timer) for split in find_splits(apt36k, images))
    _write_section(docs_path, DOCS_HEADING, pose_section(reports, cause))
    _write_section(adr_path, ADR_HEADING, adr_section(reports, cause))
    passed = all(check.passed for check in reproduction(reports[0]))
    if not (passed or cause):
        print("Reproduction check failed and no cause is recorded.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apt36k", type=Path, required=True, help="Local APT-36K directory")
    parser.add_argument("--images", type=Path, help="Image root, default the APT-36K directory")
    parser.add_argument("--cause", help="Cause of a failed reproduction check, for ADR-001")
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--adr", type=Path, default=ADR)
    args = parser.parse_args()
    try:
        return run(
            apt36k=args.apt36k,
            images=args.images or args.apt36k,
            docs_path=args.docs,
            adr_path=args.adr,
            cause=args.cause,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
