from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from copypaw.motion.bvh_writer import write_bvh

bvh_module = import_module("copypaw.vendor.animated_drawings.model.bvh")
# Input topology is independent of the exported hierarchy.
SEGMENTS = (
    ("Neck", 4, 3),
    ("Nose", 3, 2),
    ("Near_Shoulder", 3, 5),
    ("Near_Elbow", 5, 6),
    ("Near_F_Paw", 6, 7),
    ("Far_Shoulder", 3, 8),
    ("Far_Elbow", 8, 9),
    ("Far_F_Paw", 9, 10),
    ("Near_Hip", 4, 11),
    ("Near_Knee", 11, 12),
    ("Near_B_Paw", 12, 13),
    ("Far_Hip", 4, 14),
    ("Far_Knee", 14, 15),
    ("Far_B_Paw", 15, 16),
)


def trajectory(phase: float = 0.0, amplitude: float = 0.6) -> NDArray[np.float64]:
    kp = np.zeros((5, 17, 2), dtype=np.float64)
    kp[:, 4] = np.column_stack((np.arange(5) * 7 + 200, np.arange(5) * 3 + 150))
    time = np.linspace(0, 2 * np.pi, 5)
    for index, (_, prox, dist) in enumerate(SEGMENTS):
        angle = phase + index * 0.65 + amplitude * np.sin(time + index)
        length = 100 if dist == 3 else 20 + index
        kp[:, dist] = kp[:, prox] + length * np.column_stack((np.cos(angle), np.sin(angle)))
    return kp


def assert_round_trip(
    kp: NDArray[np.float64], path: Path, lengths: dict[str, float] | None = None
) -> None:
    root_x = np.linspace(0, 2, 5)
    summary = write_bvh(kp, 30, root_x, lengths, out=path)
    motion = bvh_module.BVH.from_file(str(path))
    assert motion.frame_max_num == summary.frame_count == 5
    assert motion.frame_time == summary.frame_time == 0.033333
    assert summary.scale == pytest.approx(0.01)
    names = [name for name in motion.get_joint_names() if name != "End Site"]
    assert len(names) == len(set(names))
    assert {name for name, _, _ in SEGMENTS}.issubset(names)
    for frame in range(5):
        motion.apply_frame(frame)
        motion.root_joint.update_transforms(update_ancestors=True)
        positions: dict[str, NDArray[np.float64]] = {}
        for name in ("Root", "Fwd_Near", "Fwd_Far", *(s[0] for s in SEGMENTS)):
            joint = motion.root_joint.get_transform_by_name(name)
            assert joint is not None
            positions[name] = np.asarray(joint.get_world_position(), dtype=np.float64).reshape(3)
        root = np.array([root_x[frame], -kp[frame, 4, 1] / 100, 0])
        np.testing.assert_allclose(positions["Root"], root, atol=1e-6, rtol=0)
        for name, z in (("Fwd_Near", 0.5), ("Fwd_Far", -0.5)):
            np.testing.assert_allclose(positions[name] - root, [0, 0, z], atol=1e-6, rtol=0)
        for name, prox, dist in SEGMENTS:
            parent_name = "Root" if prox == 4 else next(n for n, _, d in SEGMENTS if d == prox)
            actual = (positions[name] - positions[parent_name])[:2]
            expected: NDArray[np.float64] = kp[frame, dist] - kp[frame, prox]
            expected[1] *= -1
            actual /= np.linalg.norm(actual)
            expected /= np.linalg.norm(expected)
            error = np.arctan2(
                actual[0] * expected[1] - actual[1] * expected[0], np.dot(actual, expected)
            )
            assert abs(error) < 1e-3
            expected_length = (
                lengths[name]
                if lengths is not None
                else (
                    100
                    if dist == 3
                    else 20 + next(i for i, segment in enumerate(SEGMENTS) if segment[0] == name)
                )
                / 100
            )
            np.testing.assert_allclose(
                np.linalg.norm((positions[name] - positions[parent_name])[:2]),
                expected_length,
                atol=2e-6,
                rtol=0,
            )
            expected_z = (
                0.08 if name.startswith("Near_") else -0.08 if name.startswith("Far_") else 0
            )
            assert positions[name][2] == pytest.approx(expected_z, abs=1e-6)
    text = path.read_text()
    frames = text.split("Frame Time: 0.033333\n")[1].splitlines()
    assert len(frames) == 5
    for line in frames:
        assert "  " not in line
        assert len(line.split(" ")) == 96
        assert all(len(value.split(".")[1]) == 6 for value in line.split(" "))


def test_five_frames_round_trip(tmp_path: Path) -> None:
    assert_round_trip(trajectory(), tmp_path / "motion.bvh")


def test_custom_drawing_lengths(tmp_path: Path) -> None:
    assert_round_trip(
        trajectory(),
        tmp_path / "drawing.bvh",
        {name: (index + 1) / 10 for index, (name, _, _) in enumerate(SEGMENTS)},
    )


# Each example writes and parses a file, which outruns the 200 ms default under load.
@settings(
    max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    phase=st.floats(min_value=-np.pi, max_value=np.pi),
    amplitude=st.floats(min_value=0, max_value=2.5),
)
def test_smooth_trajectories_round_trip(tmp_path: Path, phase: float, amplitude: float) -> None:
    assert_round_trip(trajectory(phase, amplitude), tmp_path / "smooth.bvh")


def test_stationary_root_and_full_turn(tmp_path: Path) -> None:
    kp = trajectory()
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    kp[:, 2] = kp[:, 3] + 20 * np.column_stack((np.cos(angles), np.sin(angles)))
    path = tmp_path / "turn.bvh"
    write_bvh(kp, 15, np.zeros(5), out=path)
    motion = bvh_module.BVH.from_file(str(path))
    np.testing.assert_array_equal(motion.pos_data[:, 0], 0)
    for frame, angle in enumerate(angles):
        motion.apply_frame(frame)
        motion.root_joint.update_transforms(update_ancestors=True)
        nose = motion.root_joint.get_transform_by_name("Nose")
        neck = motion.root_joint.get_transform_by_name("Neck")
        assert nose is not None
        assert neck is not None
        delta = np.asarray(nose.get_world_position() - neck.get_world_position()).reshape(3)
        np.testing.assert_allclose(
            delta[:2], 0.2 * np.array([np.cos(angle), -np.sin(angle)]), atol=1e-6
        )


@pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf, 3e6])
def test_invalid_frame_rate(tmp_path: Path, fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        write_bvh(trajectory(), fps, np.zeros(5), out=tmp_path / "invalid.bvh")


@pytest.mark.parametrize("kind", ["empty", "shape", "nan", "zero_body", "zero_bone", "root"])
def test_invalid_tracks(tmp_path: Path, kind: str) -> None:
    kp = trajectory()
    root = np.zeros(5)
    if kind == "empty":
        kp = kp[:0]
    elif kind == "shape":
        kp = kp[:, :16]
    elif kind == "nan":
        kp[0, 7, 0] = np.nan
    elif kind == "zero_body":
        kp[:, 3] = kp[:, 4]
    elif kind == "zero_bone":
        kp[0, 7] = kp[0, 6]
    else:
        root = np.zeros(4)
    path = tmp_path / "invalid.bvh"
    with pytest.raises(ValueError, match=r"shape|finite|positive|root_x"):
        write_bvh(kp, 30, root, out=path)
    assert not path.exists()


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_custom_lengths(tmp_path: Path, value: float) -> None:
    lengths = dict.fromkeys((name for name, _, _ in SEGMENTS), 1.0)
    lengths["Neck"] = value
    with pytest.raises(ValueError, match="length"):
        write_bvh(trajectory(), 30, np.zeros(5), lengths, out=tmp_path / "invalid.bvh")


def test_incomplete_custom_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly"):
        write_bvh(trajectory(), 30, np.zeros(5), {}, out=tmp_path / "invalid.bvh")


def test_variable_pixel_lengths_use_clip_medians(tmp_path: Path) -> None:
    kp = trajectory(amplitude=0)
    kp *= np.array([1, 2, 3, 4, 10])[:, None, None]
    path = tmp_path / "scaled.bvh"
    summary = write_bvh(kp, 24, np.zeros(5), out=path)
    assert summary.scale == pytest.approx(1 / 300)
    motion = bvh_module.BVH.from_file(str(path))
    motion.root_joint.update_transforms(update_ancestors=True)
    for index, (name, prox, dist) in enumerate(SEGMENTS):
        parent_name = "Root" if prox == 4 else next(n for n, _, d in SEGMENTS if d == prox)
        joint = motion.root_joint.get_transform_by_name(name)
        parent = motion.root_joint.get_transform_by_name(parent_name)
        assert joint is not None
        assert parent is not None
        delta = np.asarray(joint.get_world_position() - parent.get_world_position()).reshape(3)
        expected = 1.0 if dist == 3 else (20 + index) / 100
        assert np.linalg.norm(delta[:2]) == pytest.approx(expected, abs=1e-6)
    np.testing.assert_allclose(motion.pos_data[:, 1], -kp[:, 4, 1] / 300, atol=1e-6)
