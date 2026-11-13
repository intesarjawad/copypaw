"""Measure gallery gait fidelity, tune contact thresholds and write the report."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from copypaw.capture.contacts import detect_contacts
from copypaw.evaluate.fidelity import (
    CALIBRATION_NAME,
    DrawingJoints,
    Fidelity,
    contact_f1,
    evaluate,
    is_locomotion,
)
from copypaw.motion.motion_file import MotionFile

try:
    from scripts.label_contacts import PAW_SLOTS, label_files, load_labels, merge_labels
except ModuleNotFoundError:
    from label_contacts import PAW_SLOTS, label_files, load_labels, merge_labels

ROOT = Path(__file__).resolve().parents[1]
MOTIONS = ROOT / "motions"
DRAWINGS = ROOT / "samples/drawings"
RENDERS = ROOT / "evaluations/renders"
UPSTREAM_RENDERS = ROOT / "evaluations/upstream"
DOCS = ROOT / "docs/evaluation.md"
ADR = ROOT / "docs/adr/ADR-005.md"
CALIBRATION = ROOT / "copypaw/evaluate" / CALIBRATION_NAME
SECTION = "## Gait fidelity"
_HEIGHT_SWEEP = (0.04, 0.06, 0.08)
_SPEED_SWEEP = (0.15, 0.3, 0.45)
_FIELD_TEST = "separate field test"
_TUNED_ON = "gallery motions"
_LABELLERS = 2


@dataclass(frozen=True)
class GalleryMotion:
    """One gallery clip with the animal and gait labels used for pairing."""

    name: str
    animal: str
    gait: str
    motion: MotionFile


@dataclass(frozen=True)
class Row:
    """Matched and baseline measurements for one motion on one drawing."""

    motion: GalleryMotion
    drawing: str
    matched: Fidelity
    mismatched: tuple[Fidelity, ...]
    same_gait: tuple[Fidelity, ...]
    upstream: Fidelity | None


def _labels(path: Path) -> tuple[str, str]:
    if path.is_file():
        data = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        animal = data.get("animal")
        gait = data.get("gait")
        if isinstance(animal, str) and isinstance(gait, str):
            return animal, gait
    animal, _, gait = path.parent.name.partition("_")
    return animal, gait or "unknown"


def discover_motions(directory: Path) -> tuple[GalleryMotion, ...]:
    """Load every gallery motion directory that holds motion.npz."""
    motions: list[GalleryMotion] = []
    for path in sorted(directory.glob("*/motion.npz")):
        motion = MotionFile.load(path)
        animal, gait = _labels(path.parent / "motion.json")
        motions.append(GalleryMotion(path.parent.name, animal, gait, motion))
    if not motions:
        raise FileNotFoundError(f"No gallery motions under {directory}")
    return tuple(motions)


def discover_drawings(directory: Path) -> tuple[str, ...]:
    """Return sample drawing directory names that contain a character config."""
    drawings = tuple(
        sorted(path.name for path in directory.glob("*") if (path / "char_cfg.yaml").is_file())
    )
    if not drawings:
        raise FileNotFoundError(f"No sample drawings under {directory}")
    return drawings


def render_path(directory: Path, motion: str, drawing: str) -> Path:
    """Name the stored render output for one motion and drawing."""
    return directory / motion / drawing / "render_joints.npz"


def _load_render(path: Path) -> DrawingJoints:
    if not path.is_file():
        raise FileNotFoundError(f"Missing rendered joints: {path}")
    return DrawingJoints.load(path)


def _locomotion(motion: MotionFile) -> bool:
    times = np.arange(len(motion.kp), dtype=np.float64) / motion.fps
    return is_locomotion(motion.kp, motion.contacts, times, motion.travel_root_x)


def build_rows(
    motions: tuple[GalleryMotion, ...],
    drawings: tuple[str, ...],
    render_dir: Path,
    upstream_dir: Path,
) -> tuple[Row, ...]:
    """Compare every gallery motion on every drawing with each baseline."""
    locomotion = {item.name: _locomotion(item.motion) for item in motions}
    rows: list[Row] = []
    for item in motions:
        for drawing in drawings:
            matched = evaluate(
                item.motion, _load_render(render_path(render_dir, item.name, drawing))
            )
            mismatched: list[Fidelity] = []
            same_gait: list[Fidelity] = []
            upstream: Fidelity | None = None
            if locomotion[item.name]:
                for other in motions:
                    if other.name == item.name or not locomotion[other.name]:
                        continue
                    candidate = evaluate(
                        item.motion,
                        _load_render(render_path(render_dir, other.name, drawing)),
                    )
                    mismatched.append(candidate)
                    if other.gait == item.gait and other.animal != item.animal:
                        same_gait.append(candidate)
                upstream = evaluate(
                    item.motion, _load_render(render_path(upstream_dir, "zombie", drawing))
                )
            rows.append(
                Row(
                    motion=item,
                    drawing=drawing,
                    matched=matched,
                    mismatched=tuple(mismatched),
                    same_gait=tuple(same_gait),
                    upstream=upstream,
                )
            )
    return tuple(rows)


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _metric_mean(items: tuple[Fidelity, ...], name: str) -> float | None:
    return _mean([getattr(item, name) for item in items])


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def gait_section(rows: tuple[Row, ...]) -> str:
    lines = [
        SECTION,
        "",
        "Generated by `scripts/eval_motion.py`; edit the generator, not this section.",
        "",
        "| Motion | Drawing | Condition | Gait r | Contact F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lower: list[str] = []
    for row in rows:
        matched_gait = row.matched.gait_r
        matched_contact = row.matched.contact_f1
        mismatched_gait = _metric_mean(row.mismatched, "gait_r")
        mismatched_contact = _metric_mean(row.mismatched, "contact_f1")
        same_gait = _metric_mean(row.same_gait, "gait_r")
        same_contact = _metric_mean(row.same_gait, "contact_f1")
        upstream_gait = None if row.upstream is None else row.upstream.gait_r
        upstream_contact = None if row.upstream is None else row.upstream.contact_f1
        comparisons = [
            (
                "mismatched",
                mismatched_gait,
                mismatched_contact,
                len(row.mismatched),
            ),
            (
                "same gait, other animal",
                same_gait,
                same_contact,
                len(row.same_gait),
            ),
            ("upstream", upstream_gait, upstream_contact, 1 if row.upstream else 0),
        ]
        lines.append(
            f"| {row.motion.name} | {row.drawing} | matched "
            f"| {_format(matched_gait)} | {_format(matched_contact)} |"
        )
        if not row.matched.locomotion:
            continue
        for label, gait, contact, count in comparisons:
            suffix = f" ({count})" if count > 1 else ""
            lines.append(
                f"| {row.motion.name} | {row.drawing} | {label}{suffix} "
                f"| {_format(gait)} | {_format(contact)} |"
            )
            if matched_gait is not None and gait is not None and matched_gait <= gait:
                lower.append(
                    f"{row.motion.name}/{row.drawing}: matched gait r "
                    f"{_format(matched_gait)} is not higher than {label} {_format(gait)}."
                )
            if contact is not None and matched_contact <= contact:
                lower.append(
                    f"{row.motion.name}/{row.drawing}: matched contact F1 "
                    f"{_format(matched_contact)} is not higher than {label} {_format(contact)}."
                )
    lines.append("")
    if lower:
        lines.append("Where matched is not higher:")
        lines.extend(f"- {item}" for item in lower)
    else:
        lines.append("Matched gait r and contact F1 were higher than every baseline.")
    notes = [row for row in rows if not row.matched.locomotion]
    if notes:
        lines.append("")
        lines.extend(
            f"{row.motion.name}/{row.drawing}: timing score not shown: "
            "no repeating steps found; contact F1 only."
            for row in notes
        )
    return "\n".join(lines) + "\n"


def replace_section(document: str, section: str) -> str:
    """Replace a Markdown section, preserving everything around it."""
    heading = document.find(SECTION)
    if heading < 0:
        if not document:
            return section
        separator = "" if document.endswith("\n") else "\n"
        return f"{document}{separator}\n{section}"
    following = document.find("\n## ", heading + len(SECTION))
    rest = "" if following < 0 else "\n" + document[following + 1 :]
    return document[:heading] + section + rest


def _reference_contacts(path: Path, motion: MotionFile) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing merged contact labels: {path}. Run label_contacts.py --merge."
        )
    labels = load_labels(path)
    contacts = np.zeros((len(motion.kp), 4), dtype=np.bool_)
    for paw, slot in enumerate(PAW_SLOTS):
        for stance in labels.paws.get(slot, ()):
            start = round(stance.onset)
            stop = round(stance.lift_off)
            contacts[start:stop, paw] = True
    return contacts


def sweep_contacts(
    motions: tuple[GalleryMotion, ...], directory: Path
) -> tuple[float, float, float, dict[str, float]]:
    """Pick the threshold pair with the best mean onset F1 on agreed labels."""
    best: tuple[float, float, float] | None = None
    for height in _HEIGHT_SWEEP:
        for speed in _SPEED_SWEEP:
            scores: list[float] = []
            for item in motions:
                reference = _reference_contacts(
                    directory / item.name / "contacts_reference.json", item.motion
                )
                times = np.arange(len(item.motion.kp), dtype=np.float64) / item.motion.fps
                detected = detect_contacts(
                    item.motion.kp,
                    times,
                    scores=item.motion.scores,
                    height_ratio=height,
                    speed_ratio=speed,
                )
                scores.append(contact_f1(detected.contacts, reference))
            value = float(np.mean(scores))
            if best is None or value > best[2]:
                best = (height, speed, value)
    if best is None:
        raise FileNotFoundError("No contact labels to sweep")
    agreements: dict[str, float] = {}
    for item in motions:
        files = label_files(directory / item.name)
        if len(files) == _LABELLERS:
            first, second = (load_labels(path) for path in files)
            _, agreements[item.name] = merge_labels(first, second)
    return best[0], best[1], best[2], agreements


def _mismatched_values(rows: tuple[Row, ...]) -> list[float]:
    return [
        value
        for row in rows
        for value in (item.gait_r for item in row.mismatched)
        if value is not None and math.isfinite(value)
    ]


def calibration_payload(
    rows: tuple[Row, ...],
    height: float,
    speed: float,
    onset_f1: float,
) -> dict[str, object]:
    """Build the package calibration data from the locomotion-only baseline."""
    values = _mismatched_values(rows)
    if not values:
        raise ValueError("No mismatched locomotion pairs to calibrate on")
    return {
        "contact_height_ratio": height,
        "contact_speed_ratio": speed,
        "field_test": f"the {_FIELD_TEST} is the held-out check",
        "gait_r_p95": float(np.percentile(values, 95)),
        "mismatched_pairs": len(values),
        "onset_f1": onset_f1,
        "tuned_on": _TUNED_ON,
    }


def _formatted_calibration(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _adr_section(
    height: float,
    speed: float,
    onset_f1: float,
    agreements: dict[str, float],
) -> str:
    agreement = (
        ", ".join(f"{name} {value:.3f}" for name, value in sorted(agreements.items()))
        if agreements
        else "not reported"
    )
    return (
        "## Threshold sweep\n\n"
        "I swept CONTACT_HEIGHT_RATIO over 0.04, 0.06 and 0.08 and "
        "CONTACT_SPEED_RATIO over 0.15, 0.3 and 0.45 per second against the "
        f"agreed gallery labels. The best mean onset F1 was {onset_f1:.3f} at "
        f"height {height:.2f} and speed {speed:.2f}. Onset agreement per motion: "
        f"{agreement}. The thresholds were tuned on the {_TUNED_ON}, and the "
        f"{_FIELD_TEST} is the held-out check.\n"
    )


def update_adr(document: str, section: str) -> str:
    """Replace or append the threshold note in the decision record."""
    heading = "## Threshold sweep"
    start = document.find(heading)
    if start < 0:
        return document.rstrip() + "\n\n" + section
    following = document.find("\n## ", start + len(heading))
    rest = "" if following < 0 else "\n" + document[following + 1 :]
    return document[:start] + section + rest


def run(
    *,
    motions_dir: Path,
    drawings_dir: Path,
    render_dir: Path,
    upstream_dir: Path,
    docs_path: Path,
    adr_path: Path,
    calibration_path: Path,
    check: bool,
) -> int:
    motions = discover_motions(motions_dir)
    drawings = discover_drawings(drawings_dir)
    rows = build_rows(motions, drawings, render_dir, upstream_dir)
    section = gait_section(rows)
    height, speed, onset_f1, agreements = sweep_contacts(motions, motions_dir)
    payload = calibration_payload(rows, height, speed, onset_f1)
    calibration = _formatted_calibration(payload)
    document = docs_path.read_text(encoding="utf-8") if docs_path.is_file() else ""
    rewritten = replace_section(document, section)
    if check:
        stale: list[str] = []
        if not docs_path.is_file() or docs_path.read_text(encoding="utf-8") != rewritten:
            stale.append(str(docs_path))
        if (
            not calibration_path.is_file()
            or calibration_path.read_text(encoding="utf-8") != calibration
        ):
            stale.append(str(calibration_path))
        if stale:
            print("Out of date: " + ", ".join(stale))
            return 1
        return 0
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(rewritten, encoding="utf-8")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(calibration, encoding="utf-8")
    adr = adr_path.read_text(encoding="utf-8") if adr_path.is_file() else ""
    adr_path.parent.mkdir(parents=True, exist_ok=True)
    adr_path.write_text(
        update_adr(adr, _adr_section(height, speed, onset_f1, agreements)),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing")
    parser.add_argument("--motions", type=Path, default=MOTIONS)
    parser.add_argument("--drawings", type=Path, default=DRAWINGS)
    parser.add_argument("--renders", type=Path, default=RENDERS)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM_RENDERS)
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--adr", type=Path, default=ADR)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    args = parser.parse_args()
    try:
        return run(
            motions_dir=args.motions,
            drawings_dir=args.drawings,
            render_dir=args.renders,
            upstream_dir=args.upstream,
            docs_path=args.docs,
            adr_path=args.adr,
            calibration_path=args.calibration,
            check=args.check,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
