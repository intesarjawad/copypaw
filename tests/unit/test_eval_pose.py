import json
from pathlib import Path

import numpy as np
import pytest

from copypaw.evaluate.identity_swaps import Flips
from copypaw.evaluate.pck import Pck
from copypaw.models import Mode
from scripts.eval_pose import (
    Predictor,
    Row,
    Scores,
    Split,
    SplitReport,
    find_splits,
    flip_default,
    load_split,
    replace_section,
    run,
)

FRONT_LEFT = (5, 6, 7)
FRONT_RIGHT = (8, 9, 10)
BACK_LEFT = (11, 12, 13)
BACK_RIGHT = (14, 15, 16)
LABELS = ("raw", "+ flip test", "+ identity", "+ identity + flip test")


def _keypoints(frame: int) -> list[float]:
    values: list[float] = []
    for joint in range(17):
        values.extend([10.0 + 5 * joint + 2 * frame, 20.0 + 3 * joint, 2.0])
    return values


def _annotation(
    identifier: int, image: int, category: int, frame: int, **extra: object
) -> dict[str, object]:
    return {
        "id": identifier,
        "image_id": image,
        "category_id": category,
        "bbox": [5, 5, 80, 80],
        "area": 6400,
        "num_keypoints": 17,
        "keypoints": _keypoints(frame),
        **extra,
    }


def _write_annotations(path: Path, extra_images: tuple[str, ...] = ()) -> None:
    names = ["clip0/000000.jpg", "clip0/000001.jpg", "clip0/000002.jpg", "clip1/000000.jpg"]
    names.extend(extra_images)
    images = [
        {"id": index, "file_name": name, "width": 100, "height": 100}
        for index, name in enumerate(names)
    ]
    annotations = [
        _annotation(1, 0, 1, 0),
        _annotation(2, 1, 1, 1),
        _annotation(3, 2, 1, 2),
        _annotation(4, 3, 2, 0),
        _annotation(5, 3, 3, 0),
    ]
    categories = [{"id": 1, "name": "Dog"}, {"id": 2, "name": "cat"}, {"id": 3, "name": "horse"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )


def _exact(exchange_frame: int | None = None) -> Predictor:
    def predict(split: Split, mode: Mode, flip_test: bool) -> tuple[np.ndarray, np.ndarray]:
        del mode, flip_test
        kp = np.stack([instance.kp for instance in split.instances]).astype(np.float32)
        if exchange_frame is not None:
            for chain, other in ((FRONT_LEFT, FRONT_RIGHT), (BACK_LEFT, BACK_RIGHT)):
                kp[exchange_frame, chain], kp[exchange_frame, other] = (
                    kp[exchange_frame, other].copy(),
                    kp[exchange_frame, chain].copy(),
                )
        return kp, np.ones((len(kp), 17), dtype=np.float32)

    return predict


def _timer(mode: Mode, flip_test: bool) -> float:
    base = {Mode.LIGHTWEIGHT: 10.0, Mode.BALANCED: 20.0, Mode.ACCURATE: 30.0}[mode]
    return base * (2 if flip_test else 1)


def _run(tmp_path: Path, cause: str | None = None, exchange_frame: int | None = None) -> int:
    root = tmp_path / "apt36k"
    _write_annotations(root / "annotations/val_annotations_1.json")
    return run(
        apt36k=root,
        images=root,
        docs_path=tmp_path / "evaluation.md",
        adr_path=tmp_path / "ADR-001.md",
        cause=cause,
        predict=_exact(exchange_frame),
        timer=_timer,
    )


def test_loading_follows_clips_frames_and_species(tmp_path: Path) -> None:
    path = tmp_path / "val.json"
    _write_annotations(path)

    split = load_split(path, tmp_path, "val")

    assert len(split.instances) == 5
    assert split.dog_cat == frozenset({1, 2})
    assert sorted(len(track.indices) for track in split.tracks) == [1, 1, 3]
    dog = next(track for track in split.tracks if len(track.indices) == 3)
    assert dog.length == 3
    assert dog.frames.tolist() == [0, 1, 2]
    assert split.instances[0].image == tmp_path / "clip0/000000.jpg"
    assert split.instances[0].visible.all()
    np.testing.assert_array_equal(split.instances[0].box, [5, 5, 84, 84])


def test_loading_needs_both_dog_and_cat(tmp_path: Path) -> None:
    path = tmp_path / "val.json"
    _write_annotations(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["categories"] = [{"id": 1, "name": "dog"}]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="dog and a cat"):
        load_split(path, tmp_path, "val")


def test_validation_images_may_not_be_training_images(tmp_path: Path) -> None:
    _write_annotations(tmp_path / "annotations/val_annotations_1.json")
    _write_annotations(tmp_path / "annotations/train_annotations_1.json")

    with pytest.raises(ValueError, match="also in the train split"):
        find_splits(tmp_path, tmp_path)


def test_test_split_is_read_when_present_and_validation_is_required(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"val_annotations_1\.json"):
        find_splits(tmp_path, tmp_path)

    _write_annotations(tmp_path / "annotations/val_annotations_1.json")
    _write_annotations(tmp_path / "annotations/test_annotations_1.json")

    assert [split.name for split in find_splits(tmp_path, tmp_path)] == ["val", "test"]


def test_exact_predictions_score_perfectly_on_every_row(tmp_path: Path) -> None:
    assert _run(tmp_path, cause="exact predictions do not match the README") == 0

    section = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    rows = [line for line in section.splitlines() if line.startswith("| ViTPose-B | ")]
    rows = [row for row in rows if row.split("|")[2].strip() in LABELS]
    assert [row.split("|")[2].strip() for row in rows[:4]] == [
        "raw",
        "+ flip test",
        "+ identity",
        "+ identity + flip test",
    ]
    for row in rows[:4]:
        cells = [cell.strip() for cell in row.split("|")[3:-1]]
        # AP and PCK are perfect; the dog's three frames give two chances per leg pair.
        assert cells[:4] == ["100.0"] * 4
        assert cells[4:6] == ["0/4", "0/4"]
    assert "| ViTPose-B | raw | 100.0 | 100.0 | 100.0 | 100.0 | 0/4 | 0/4 | 20.0 |" in section
    assert (
        "| ViTPose-B | + flip test | 100.0 | 100.0 | 100.0 | 100.0 | 0/4 | 0/4 | 40.0 |" in section
    )
    assert "5 labelled animals in 4 images." in section
    assert "in-distribution" in section
    assert "field test" in section


def test_an_exchanged_frame_is_two_flips_per_leg_pair(tmp_path: Path) -> None:
    assert _run(tmp_path, cause="synthetic", exchange_frame=1) == 0

    section = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    assert "| ViTPose-S | raw |" in section
    row = next(line for line in section.splitlines() if line.startswith("| ViTPose-S | raw |"))
    assert row.split("|")[7].strip() == "4/4"


def test_a_failed_reproduction_check_needs_its_cause(tmp_path: Path) -> None:
    assert _run(tmp_path) == 1

    section = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    assert "| ViTPose-B | 75.9 | 100.0 | +24.1 | fail |" in section
    assert "no cause is recorded yet" in section
    assert "I have not found the cause" in (tmp_path / "ADR-001.md").read_text(encoding="utf-8")


def test_a_recorded_cause_goes_to_both_documents(tmp_path: Path) -> None:
    assert _run(tmp_path, cause="the crop was BGR") == 0

    assert "Cause, recorded in ADR-001: the crop was BGR" in (tmp_path / "evaluation.md").read_text(
        encoding="utf-8"
    )
    assert "Cause: the crop was BGR" in (tmp_path / "ADR-001.md").read_text(encoding="utf-8")


def test_rerunning_replaces_only_the_pose_sections(tmp_path: Path) -> None:
    (tmp_path / "evaluation.md").write_text(
        "# Evaluation\n\n## Gait fidelity\n\nkeep\n\n## Pose\n\nold\n\n## Later\n\nkeep too\n",
        encoding="utf-8",
    )

    _run(tmp_path, cause="synthetic")
    _run(tmp_path, cause="synthetic")

    document = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    assert document.count("## Pose\n") == 1
    assert "old" not in document
    assert "## Gait fidelity\n\nkeep\n" in document
    assert document.endswith("## Later\n\nkeep too\n")


def test_a_tie_between_the_identity_rows_leaves_the_flip_test_off(tmp_path: Path) -> None:
    _run(tmp_path, cause="synthetic")

    adr = (tmp_path / "ADR-001.md").read_text(encoding="utf-8")
    assert "The flip_test default for balanced is off: overall PCK@0.05" in adr


def test_section_replacement_matches_whole_heading_lines() -> None:
    document = "## Pose evaluation\n\nkeep\n"

    assert replace_section(document, "## Pose", "## Pose\n\nnew\n") == (
        "## Pose evaluation\n\nkeep\n\n## Pose\n\nnew\n"
    )
    assert replace_section("", "## Pose", "## Pose\n\nnew\n") == "## Pose\n\nnew\n"


def _report(without: float, with_flip: float) -> SplitReport:
    def row(label: str, overall: float) -> Row:
        scores = Scores(0.0, Pck(overall, ()), Flips(0, 0))
        return Row(Mode.BALANCED, label, 0.0, scores, scores)

    return SplitReport(
        "val",
        1,
        1,
        (
            row("raw", 0.1),
            row("+ flip test", 0.2),
            row("+ identity", without),
            row("+ identity + flip test", with_flip),
        ),
    )


def test_the_flip_test_is_on_only_when_it_raises_pck() -> None:
    assert flip_default(_report(0.80, 0.85)) == (True, 0.80, 0.85)
    assert flip_default(_report(0.85, 0.80)) == (False, 0.85, 0.80)
    assert flip_default(_report(0.80, 0.80))[0] is False
