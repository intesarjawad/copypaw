"""Over a whole gait the sample's legs stay straight limbs, its tail a tail and its head a head."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from shapely import Polygon, union_all

from copypaw.animate.render import posed_scene, render_character, resample_motion
from copypaw.motion.motion_file import MotionFile
from copypaw.rig.char_cfg import write_character
from copypaw.rig.edits import load_drawing, rig_drawing

pytestmark = pytest.mark.render

_ROOT = Path(__file__).resolve().parents[2]
_DRAWING = _ROOT / "copypaw/selftest/dragon.png"
_SHORT_LEGS = _ROOT / "samples/worst_case/short_legs.png"
_MOTION = _ROOT / "tests/golden/dog_walk.motion.npz"
_LEGS = ("left_{}_f", "right_{}_f", "left_{}_b", "right_{}_b")
_NECK_KP = 3
_NOSE_KP = 2


@dataclass(frozen=True)
class _Posed:
    names: list[str]
    rest: NDArray[np.float64]
    vertices: NDArray[np.float64]
    joints: NDArray[np.float64]
    drawn: NDArray[np.float64]
    triangles: dict[str, NDArray[np.int64]]
    depths: list[dict[str, float]]
    ground: float | None
    tails: NDArray[np.float64]

    def joint(self, name: str) -> NDArray[np.float64]:
        return self.joints[:, self.names.index(name)]


def _pose(drawing: Path, directory: Path) -> _Posed:
    image = load_drawing(drawing)
    character = write_character(image, rig_drawing(image).snapped, directory / "drawing")
    render_character(character.directory, MotionFile.load(_MOTION), "treadmill", directory / "out")
    _, scene, handles, names = posed_scene(directory / "out" / "mvc.yaml")
    model = cast("Any", scene.get_children()[0])
    vertices: list[NDArray[np.float64]] = []
    depths: list[dict[str, float]] = []
    for frame in range(model.retargeter.bvh.frame_max_num):
        model.set_time(frame * model.retargeter.bvh.frame_time)
        model.update()
        vertices.append(np.asarray(model.vertices, dtype=np.float64)[:, :2].copy())
        depths.append(dict(model.retargeter.get_retargeted_frame_data(model.get_time())[1]))
    with np.load(directory / "out" / "render_contacts.npz", allow_pickle=False) as archive:
        tails = np.asarray(archive["tails"], dtype=np.float64) / model.img_dim
    drawn = handles.drawn_pose()[: len(names)].astype(np.float64)
    still = np.tile(drawn, (len(vertices), 1, 1))
    joints = np.stack(handles.filtered[-len(vertices) :]) if handles.filtered else still
    return _Posed(
        names=names,
        rest=np.asarray(model.mesh["vertices"], dtype=np.float64)[:, :2],
        vertices=np.stack(vertices),
        joints=np.asarray(joints, dtype=np.float64),
        drawn=drawn,
        triangles={
            name: np.asarray(indices, dtype=np.int64).reshape(-1, 3)
            for name, indices in model.joint_to_tri_v_idx.items()
        },
        depths=depths,
        ground=None if handles.contact_filter is None else handles.contact_filter.lowest_ground(),
        tails=tails,
    )


def _norms(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.sqrt((vectors**2).sum(axis=-1)), dtype=np.float64)


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> _Posed:
    return _pose(_DRAWING, tmp_path_factory.mktemp("sample"))


def _signed_areas(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> NDArray[np.float64]:
    first, second, third = (points[triangles[:, corner]] for corner in range(3))
    edge, other = second - first, third - first
    return np.asarray(0.5 * (edge[:, 0] * other[:, 1] - edge[:, 1] * other[:, 0]))


@pytest.mark.parametrize("leg", _LEGS)
def test_a_leg_never_folds_over_itself(sample: _Posed, leg: str) -> None:
    triangles = np.vstack([sample.triangles[leg.format(part)] for part in ("knee", "foot")])
    for frame, points in enumerate(sample.vertices):
        areas = _signed_areas(points, triangles)
        covered = float(union_all([Polygon(points[corners]) for corners in triangles]).area)
        # A clean bend pinches the inside of the knee; a leg lying over itself hides far more.
        assert 1 - covered / float(np.abs(areas).sum()) < 0.06, f"frame {frame}"


@pytest.mark.parametrize("leg", _LEGS)
@pytest.mark.parametrize("bone", [("hip", "knee"), ("knee", "foot")])
def test_each_leg_bone_stays_a_straight_drawn_limb(
    sample: _Posed, leg: str, bone: tuple[str, str]
) -> None:
    upper, lower = (sample.names.index(leg.format(part)) for part in bone)
    axis = sample.drawn[lower] - sample.drawn[upper]
    length = float(np.hypot(*axis))
    along = (sample.rest - sample.drawn[upper]) @ axis / length**2
    across = (sample.rest - sample.drawn[upper]) @ np.array([-axis[1], axis[0]]) / length**2
    middle = (along > 0.25) & (along < 0.75) & (np.abs(across) < 0.1)
    assert np.count_nonzero(middle) > 3

    for frame, (points, joints) in enumerate(zip(sample.vertices, sample.joints, strict=True)):
        posed = joints[lower] - joints[upper]
        normal = np.array([-posed[1], posed[0]])
        expected = joints[upper] + np.outer(along[middle], posed) + np.outer(across[middle], normal)
        drift = float(_norms(points[middle] - expected).max())
        assert drift < 0.05 * length, f"frame {frame}"


def test_far_legs_are_drawn_behind_the_body_and_near_legs_in_front(sample: _Posed) -> None:
    for depths in sample.depths:
        for end in ("F", "B"):
            assert depths[f"Far_{end}_Paw"] < depths["Root"] < depths[f"Near_{end}_Paw"]


def test_tail_sways_behind_the_hind_legs_and_stays_on_or_above_the_ground(sample: _Posed) -> None:
    assert len(sample.tails) == 1
    assert sample.ground is not None
    base, tip = sample.tails[0]
    tip_vertex = int(np.argmin(_norms(sample.rest - tip)))
    track = sample.vertices[:, tip_vertex]
    away = np.sign(base[0] - sample.drawn[sample.names.index("left_hip_b"), 0])
    length = float(np.hypot(*(tip - base)))

    assert float(np.ptp(track[:, 0])) > 0.05 * length
    assert float(track[:, 1].min()) > sample.ground - 0.02 * length
    for foot in ("left_foot_b", "right_foot_b"):
        assert np.all((track[:, 0] - sample.joint(foot)[:, 0]) * away > -0.02 * length)


def test_neck_turns_as_far_as_the_pets_head_without_stretching_the_head(sample: _Posed) -> None:
    motion = resample_motion(MotionFile.load(_MOTION))
    head = (motion.kp[:, _NOSE_KP] - motion.kp[:, _NECK_KP]).astype(np.float64)
    pet_turn = float(np.ptp(np.unwrap(np.arctan2(head[:, 1], head[:, 0]))))
    neck = sample.joint("neck") - sample.joint("torso")
    turn = float(np.ptp(np.unwrap(np.arctan2(neck[:, 1], neck[:, 0]))))
    assert turn == pytest.approx(pet_turn, rel=0.15)

    # The head is everything past the neck joint along the drawn neck.
    torso, joint = (sample.drawn[sample.names.index(name)] for name in ("torso", "neck"))
    beyond = (sample.rest - joint) @ (joint - torso) > 0
    crown = int(np.argmax(np.where(beyond, _norms(sample.rest - joint), 0)))
    drawn_reach = float(_norms(sample.rest[crown] - joint))
    reach = _norms(sample.vertices[:, crown] - sample.joint("neck"))
    np.testing.assert_allclose(reach, drawn_reach, rtol=0.05)


def test_guessed_legs_ride_with_the_body_without_shearing(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    posed = _pose(_SHORT_LEGS, tmp_path_factory.mktemp("short_legs"))
    assert posed.ground is None
    triangles = np.vstack(
        [
            posed.triangles[leg.format(part)]
            for leg in _LEGS
            for part in ("knee", "foot")
            if leg.format(part) in posed.triangles
        ]
    )
    drawn = np.abs(_signed_areas(posed.rest, triangles))
    for frame, points in enumerate(posed.vertices):
        np.testing.assert_allclose(
            np.abs(_signed_areas(points, triangles)).sum(),
            drawn.sum(),
            rtol=0.06,
            err_msg=f"frame {frame}",
        )
    assert float(np.ptp(posed.vertices[:, :, 1].mean(axis=1))) > 0
