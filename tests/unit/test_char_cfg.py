from importlib import import_module
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
import pytest
import yaml
from numpy.typing import NDArray
from PIL import Image

from copypaw.rig.char_cfg import write_character
from copypaw.rig.template import Joint, Joints, fit_template

PAGE_BGR = 245
INK_BGR = 20
WIDTH = 300
HEIGHT = 200


def _page() -> NDArray[np.uint8]:
    page = np.full((HEIGHT, WIDTH, 3), PAGE_BGR, dtype=np.uint8)
    cv2.rectangle(page, (40, 30), (260, 170), (INK_BGR,) * 3, 3)
    return page


class _SkeletonJoint(TypedDict):
    name: str
    loc: list[int]
    parent: str | None


class _CharacterConfig(TypedDict):
    height: int
    width: int
    skeleton: list[_SkeletonJoint]


def _config(path: Path) -> _CharacterConfig:
    return cast(_CharacterConfig, yaml.safe_load(path.read_text(encoding="utf-8")))


def _metadata(path: Path) -> dict[str, object]:
    return cast(dict[str, object], yaml.safe_load(path.read_text(encoding="utf-8")))


def test_writes_engine_files_and_metadata(tmp_path: Path) -> None:
    result = write_character(_page(), fit_template((40, 30, 260, 170)), tmp_path / "char")

    assert result.texture == tmp_path / "char/texture.png"
    assert result.mask.exists()
    assert result.char_cfg.exists()
    assert result.manifest.exists()
    assert _metadata(result.manifest) == {"legs": 4}

    texture = np.asarray(Image.open(result.texture))
    assert texture.shape == (HEIGHT, WIDTH, 4)
    assert np.all(texture[..., 3] == 255)
    mask = np.asarray(Image.open(result.mask))
    assert mask.shape == (HEIGHT, WIDTH)
    assert set(np.unique(mask).tolist()) <= {0, 255}

    config = _config(result.char_cfg)
    assert config["height"] == HEIGHT
    assert config["width"] == WIDTH
    skeleton = {joint["name"]: joint for joint in config["skeleton"]}
    assert len(skeleton) == 16
    assert skeleton["root"]["loc"] == skeleton["hip"]["loc"]
    assert skeleton["hip"]["parent"] == "root"
    assert skeleton["torso"]["parent"] == "hip"
    assert skeleton["neck"]["parent"] == "torso"
    assert skeleton["left_knee_f"]["parent"] == "left_hip_f"
    assert skeleton["right_foot_b"]["parent"] == "right_knee_b"


def test_vendored_character_config_loads_the_output(tmp_path: Path) -> None:
    result = write_character(_page(), fit_template((40, 30, 260, 170)), tmp_path / "char")
    character_config = import_module("copypaw.vendor.animated_drawings.config").CharacterConfig

    loaded = character_config(str(result.char_cfg))

    assert (loaded.img_height, loaded.img_width) == (HEIGHT, WIDTH)
    assert len(loaded.skeleton) == 16
    assert loaded.mask_p == result.mask
    assert loaded.txtr_p == result.texture


def test_mirroring_round_trip_records_the_flip_and_restores_positions(tmp_path: Path) -> None:
    page = _page()
    right = fit_template((40, 30, 260, 170))
    left = Joints(
        {name: Joint(WIDTH - 1 - joint.x, joint.y, joint.parent) for name, joint in right.items()},
        legs=right.legs,
    )

    right_result = write_character(page, right, tmp_path / "right")
    left_result = write_character(np.ascontiguousarray(page[:, ::-1]), left, tmp_path / "left")

    assert _metadata(left_result.manifest)["mirrored"] is True
    assert _metadata(right_result.manifest).get("mirrored") is None
    right_joints = {
        joint["name"]: joint["loc"] for joint in _config(right_result.char_cfg)["skeleton"]
    }
    left_joints = {
        joint["name"]: joint["loc"] for joint in _config(left_result.char_cfg)["skeleton"]
    }
    assert left_joints == right_joints
    assert left_joints["neck"][0] > left_joints["hip"][0]


def test_tail_is_recorded_and_mirrored_with_the_joints(tmp_path: Path) -> None:
    page = _page()
    right = fit_template((40, 30, 260, 170))
    tail = ((50.0, 90.0), (20.0, 140.0))
    left = Joints(
        {name: Joint(WIDTH - 1 - joint.x, joint.y, joint.parent) for name, joint in right.items()},
        legs=right.legs,
        tails=(((WIDTH - 1 - 50.0, 90.0), (WIDTH - 1 - 20.0, 140.0)),),
    )

    plain = write_character(page, right, tmp_path / "plain")
    tailed = write_character(
        page, Joints(dict(right.items()), legs=right.legs, tails=(tail,)), tmp_path / "tailed"
    )
    mirrored = write_character(np.ascontiguousarray(page[:, ::-1]), left, tmp_path / "mirrored")

    assert "tails" not in _metadata(plain.manifest)
    assert _metadata(tailed.manifest)["tails"] == [[list(tail[0]), list(tail[1])]]
    assert _metadata(mirrored.manifest)["tails"] == [[list(tail[0]), list(tail[1])]]


def test_guessed_legs_are_recorded_for_the_animation(tmp_path: Path) -> None:
    joints = fit_template((40, 30, 260, 170))
    found = write_character(_page(), joints, tmp_path / "found")
    joints.legs_guessed = True
    guessed = write_character(_page(), joints, tmp_path / "guessed")

    assert "legs_guessed" not in _metadata(found.manifest)
    assert _metadata(guessed.manifest)["legs_guessed"] is True


def test_two_leg_sidecar_is_written_from_the_joints(tmp_path: Path) -> None:
    page = _page()
    joints = Joints(fit_template((40, 30, 260, 170)), legs=2)

    result = write_character(page, joints, tmp_path / "char")

    assert _metadata(result.manifest) == {"legs": 2}


def test_png_outputs_carry_no_exif_or_text_metadata(tmp_path: Path) -> None:
    result = write_character(_page(), fit_template((40, 30, 260, 170)), tmp_path / "char")

    for path in (result.texture, result.mask):
        data = path.read_bytes()
        assert b"Exif" not in data
        assert b"tEXt" not in data


def test_missing_hip_is_rejected(tmp_path: Path) -> None:
    joints = Joints(
        {name: joint for name, joint in fit_template((40, 30, 260, 170)).items() if name != "hip"}
    )

    with pytest.raises(KeyError):
        write_character(_page(), joints, tmp_path / "char")
