import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from copypaw.cli import app

PAGE_BGR = 245
INK_BGR = 20
JOINT_COUNT = 16
MOVED_FRACTION = 0.10
SMALL_FRACTION = 0.02


@pytest.fixture
def drawing(tmp_path: Path) -> Path:
    page = np.full((200, 300, 3), PAGE_BGR, dtype=np.uint8)
    cv2.rectangle(page, (40, 30), (260, 170), (INK_BGR,) * 3, 3)
    path = tmp_path / "drawing.png"
    assert cv2.imwrite(str(path), page)
    return path


def _box_height(out: Path) -> int:
    mask = cv2.imread(str(out / "mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    rows = np.nonzero(mask.max(axis=1))[0]
    return int(rows.max() - rows.min() + 1)


def _rig(drawing: Path, out: Path, *extra: str) -> tuple[int, str, dict[str, list[float]]]:
    result = CliRunner().invoke(app, ["rig", str(drawing), "-o", str(out), *extra])
    printed: dict[str, list[float]] = json.loads(result.stdout) if result.exit_code == 0 else {}
    return result.exit_code, result.output, printed


def test_rig_writes_the_character_and_prints_sixteen_joints(drawing: Path, tmp_path: Path) -> None:
    out = tmp_path / "rig"

    code, output, printed = _rig(drawing, out)

    assert code == 0, output
    for name in ("texture.png", "mask.png", "char_cfg.yaml"):
        assert (out / name).is_file()
    assert not (out / "rig_edits.json").exists()
    assert len(printed) == JOINT_COUNT
    assert all(len(position) == 2 for position in printed.values())


def test_drawing_without_separate_legs_warns_that_the_leg_dots_are_guesses(
    drawing: Path, tmp_path: Path
) -> None:
    code, output, _ = _rig(drawing, tmp_path / "rig")

    assert code == 0, output
    assert "Warning: no four or two separate legs were found" in output
    assert "--joints" in output


def test_drawing_with_four_separate_legs_rigs_without_a_warning(tmp_path: Path) -> None:
    plain = Path(__file__).resolve().parents[2] / "samples/worst_case/plain.png"

    code, output, _ = _rig(plain, tmp_path / "rig")

    assert code == 0, output
    assert "Warning" not in output


def test_given_joints_replace_the_guess_without_a_warning(drawing: Path, tmp_path: Path) -> None:
    _, _, printed = _rig(drawing, tmp_path / "first")
    joints = tmp_path / "joints.json"
    joints.write_text(json.dumps(printed), encoding="utf-8")

    code, output, _ = _rig(drawing, tmp_path / "second", "--joints", str(joints))

    assert code == 0, output
    assert "Warning" not in output


def test_printed_joints_round_trip_with_no_edits(drawing: Path, tmp_path: Path) -> None:
    _, _, printed = _rig(drawing, tmp_path / "first")
    joints = tmp_path / "joints.json"
    joints.write_text(json.dumps(printed), encoding="utf-8")
    out = tmp_path / "second"

    code, output, again = _rig(drawing, out, "--joints", str(joints))

    assert code == 0, output
    assert again == printed
    edits = json.loads((out / "rig_edits.json").read_text(encoding="utf-8"))
    assert edits["snapped"] == printed
    assert edits["final"] == printed
    assert edits["joints_moved"] == 0


def test_one_joint_moved_by_ten_percent_of_the_box_counts_once(
    drawing: Path, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    _, _, printed = _rig(drawing, first)
    height = _box_height(first)
    edited = {name: list(position) for name, position in printed.items()}
    edited["left_foot_f"][1] -= MOVED_FRACTION * height
    joints = tmp_path / "joints.json"
    joints.write_text(json.dumps(edited), encoding="utf-8")
    out = tmp_path / "second"

    code, output, _ = _rig(drawing, out, "--joints", str(joints))

    assert code == 0, output
    edits = json.loads((out / "rig_edits.json").read_text(encoding="utf-8"))
    assert edits["joints_moved"] == 1
    assert edits["snapped"] == printed
    assert edits["final"]["left_foot_f"] == edited["left_foot_f"]


def test_a_small_nudge_is_not_a_move(drawing: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    _, _, printed = _rig(drawing, first)
    height = _box_height(first)
    edited = {name: list(position) for name, position in printed.items()}
    edited["neck"][0] += SMALL_FRACTION * height
    joints = tmp_path / "joints.json"
    joints.write_text(json.dumps(edited), encoding="utf-8")
    out = tmp_path / "second"

    _rig(drawing, out, "--joints", str(joints))

    edits = json.loads((out / "rig_edits.json").read_text(encoding="utf-8"))
    assert edits["joints_moved"] == 0


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"hip": [1, 2]}', json.dumps({"hip": "x"})],
)
def test_bad_joints_file_reports_a_fix(drawing: Path, tmp_path: Path, content: str) -> None:
    joints = tmp_path / "joints.json"
    joints.write_text(content, encoding="utf-8")

    result = CliRunner().invoke(
        app, ["rig", str(drawing), "-o", str(tmp_path / "out"), "--joints", str(joints)]
    )

    assert result.exit_code == 2
    assert f"error: The joints file {joints} is not usable" in result.stderr
    assert "next:  Write a fresh one with `copypaw rig" in result.stderr


def test_a_joint_with_a_non_finite_position_is_rejected(drawing: Path, tmp_path: Path) -> None:
    _, _, printed = _rig(drawing, tmp_path / "first")
    joints = tmp_path / "joints.json"
    joints.write_text(json.dumps({**printed, "hip": [float("nan"), 1]}), encoding="utf-8")

    result = CliRunner().invoke(
        app, ["rig", str(drawing), "-o", str(tmp_path / "out"), "--joints", str(joints)]
    )

    assert result.exit_code == 2
    assert "hip must be a finite [x, y] pair" in result.stderr


def test_unreadable_drawing_reports_a_fix(tmp_path: Path) -> None:
    text = tmp_path / "drawing.png"
    text.write_text("not an image", encoding="utf-8")

    result = CliRunner().invoke(app, ["rig", str(text), "-o", str(tmp_path / "out")])

    assert result.exit_code == 2
    assert f"error: Copypaw could not read {text} as an image." in result.stderr
