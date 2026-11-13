"""Every generated worst-case page must segment to the outcome recorded for it."""

import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from copypaw.rig.mask import MaskError, segment
from copypaw.rig.snap import snap
from copypaw.rig.template import fit_template

_ROOT = Path(__file__).resolve().parents[2]
_SAMPLES = _ROOT / "samples/worst_case"
_GENERATOR = _ROOT / "scripts/make_worst_case.py"
_LEG_SIDES = ("left_", "right_")
_MAX_PIXEL_DRIFT = 1


class _Required(TypedDict):
    outcome: str


class _Expected(_Required, total=False):
    coverage: list[float]
    legs: int
    fallback: bool


_EXPECTED = cast(
    dict[str, _Expected],
    yaml.safe_load((_SAMPLES / "expected.yaml").read_text(encoding="utf-8")),
)


def _page(name: str) -> NDArray[np.uint8]:
    image = cv2.imread(str(_SAMPLES / name), cv2.IMREAD_COLOR)
    assert image is not None, f"Missing worst-case page: {name}"
    return np.asarray(image, dtype=np.uint8)


def _decode(path: Path) -> NDArray[np.int16]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, f"Unreadable worst-case page: {path}"
    return image.astype(np.int16)


def _generate(out: Path) -> None:
    subprocess.run(  # noqa: S603 - the generator is the artifact under test.
        [sys.executable, str(_GENERATOR), "--out", str(out)],
        check=True,
    )


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_recorded_outcome(name: str) -> None:
    expected = _EXPECTED[name]
    image = _page(name)

    if expected["outcome"] != "ok":
        with pytest.raises(MaskError) as error:
            segment(image)
        assert error.value.kind == expected["outcome"]
        return

    mask = segment(image)
    coverage_range = expected.get("coverage")
    assert coverage_range is not None
    low, high = coverage_range
    coverage = np.count_nonzero(mask) / mask.size
    assert low <= coverage <= high

    legs = expected.get("legs")
    if legs is None:
        return
    rows, columns = np.nonzero(mask)
    template = fit_template(
        (float(columns.min()), float(rows.min()), float(columns.max()), float(rows.max()))
    )
    joints = snap(mask, template, "right")
    assert joints.legs == legs
    leg_names = [name for name in template if name.startswith(_LEG_SIDES)]
    fallback = all(joints[name] == template[name] for name in leg_names)
    expected_fallback = expected.get("fallback")
    assert expected_fallback is not None
    assert fallback is expected_fallback
    for joint_name in ("root", "hip", "torso", "neck") if fallback else tuple(joints):
        joint = joints[joint_name]
        assert mask[round(joint.y), round(joint.x)], f"{joint_name} is outside the outline"


def test_generator_output_is_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _generate(first)
    _generate(second)

    generated = sorted(path.name for path in first.glob("*.png"))
    assert generated == sorted(_EXPECTED)
    for name in generated:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        if sys.platform == "darwin":
            assert (first / name).read_bytes() == (_SAMPLES / name).read_bytes()
            continue
        # The committed pages were generated on macOS. OpenCV's Linux build rounds
        # warpPerspective differently, by at most one intensity level.
        difference = np.abs(_decode(first / name) - _decode(_SAMPLES / name))
        assert difference.max() <= _MAX_PIXEL_DRIFT, name
