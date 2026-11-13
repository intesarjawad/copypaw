from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.capture.detect import Track
from copypaw.capture.pose import FLIP_PAIRS, KEYPOINT_NAMES, estimate_keypoints
from copypaw.capture.video import Frames
from copypaw.models import MODELS, Mode


def _frames(count: int = 1) -> Frames:
    rgb = np.zeros((count, 320, 320, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(320, dtype=np.uint16) % 256
    rgb[..., 1] = 64
    rgb[..., 2] = 128
    return Frames(rgb, np.arange(count, dtype=np.float64) / 30, 30, 30, False)


def _track(count: int = 1) -> Track:
    return Track(
        np.tile(np.array([80, 40, 240, 200], dtype=np.float32), (count, 1)),
        np.ones(count, dtype=np.bool_),
    )


def _peaks() -> NDArray[np.float32]:
    return np.column_stack((8 + np.arange(17) * 2, 12 + np.arange(17) * 2)).astype(np.float32)


def _heatmaps() -> NDArray[np.float32]:
    y, x = np.mgrid[:64, :48]
    maps = np.empty((1, 17, 64, 48), dtype=np.float32)
    for joint, (px, py) in enumerate(_peaks()):
        maps[0, joint] = np.exp(-((x - px) ** 2 + (y - py) ** 2) / 8) * (0.4 + joint / 50)
    return maps


def _flipped(heatmaps: NDArray[np.float32]) -> NDArray[np.float32]:
    result = heatmaps[..., ::-1].copy()
    for left, right in [(0, 1), (5, 8), (6, 9), (7, 10), (11, 14), (12, 15), (13, 16)]:
        result[:, [left, right]] = result[:, [right, left]]
    # Quantized MSRA targets move one bin right after image reversal.
    result[..., 1:] = result[..., :-1].copy()
    result[..., 0] = 0
    return result


@dataclass
class FakeInput:
    name: str
    shape: list[int | str | None]


class FakeSession:
    def __init__(self, outputs: list[NDArray[np.float32]]) -> None:
        self.outputs = iter(outputs)
        self.inputs: list[NDArray[np.float32]] = []
        self.paths: list[str] = []

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("images", [1, 3, 256, 192])]

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]:
        assert output_names is None
        self.inputs.append(input_feed["images"].copy())
        return [next(self.outputs)]


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> FakeSession:
    heatmaps = _heatmaps()
    fake = FakeSession([heatmaps.copy(), _flipped(heatmaps) * 0.5])

    def make_session(path_or_bytes: str, *, providers: list[str]) -> FakeSession:
        assert providers == ["CPUExecutionProvider"]
        fake.paths.append(path_or_bytes)
        return fake

    monkeypatch.setattr("onnxruntime.InferenceSession", make_session)
    return fake


def _models() -> dict[str, Path]:
    return {pair[1]: Path(pair[1] + ".onnx") for pair in MODELS.values()}


def test_flip_restores_every_joint_and_averages_heatmaps_before_decoding(
    session: FakeSession,
) -> None:
    heatmaps = _heatmaps()
    session.outputs = iter([heatmaps.copy()])
    reference = estimate_keypoints(_frames(), _track(), "balanced", False, _models())
    session.outputs = iter([heatmaps.copy(), _flipped(heatmaps) * 0.5])
    result = estimate_keypoints(_frames(), _track(), "balanced", True, _models())

    np.testing.assert_allclose(result.kp, reference.kp, atol=0.05, rtol=0)
    np.testing.assert_allclose(result.scores[0], (0.4 + np.arange(17) / 50) * 0.75, atol=1e-6)
    assert len(session.inputs) == 3
    assert result.kp.shape == (1, 17, 2)
    assert result.scores.shape == (1, 17)
    assert result.kp.dtype == np.float32
    assert result.scores.dtype == np.float32


@pytest.mark.parametrize("flip_test", [False, True])
@pytest.mark.parametrize(
    ("source", "crop_point"),
    [((160, 120), (96, 128)), ((135, 145), (72, 152)), ((185, 95), (120, 104))],
)
def test_crop_and_training_targets_reconstruct_source_landmark(
    session: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
    flip_test: bool,
    source: tuple[int, int],
    crop_point: tuple[int, int],
) -> None:
    matrices: list[NDArray[np.float64]] = []
    warp_affine = cv2.warpAffine

    def record_crop(
        image: NDArray[np.uint8],
        matrix: NDArray[np.float64],
        size: tuple[int, int],
        *,
        flags: int,
    ) -> NDArray[np.uint8]:
        matrices.append(matrix.copy())
        return np.asarray(warp_affine(image, matrix, size, flags=flags), dtype=np.uint8)

    def ideal_outputs() -> Iterator[NDArray[np.float32]]:
        crop_xy = matrices[0] @ [*source, 1]
        np.testing.assert_allclose(crop_xy, crop_point, atol=1e-5, rtol=0)
        y, x = np.mgrid[:64, :48]
        for flipped in (False, True) if flip_test else (False,):
            point = crop_xy.copy()
            if flipped:
                point[0] = 191 - point[0]
            # APT-36K's MSRA encoder rounds input coordinates at stride four.
            # The chosen source landmarks land exactly on the unflipped target grid.
            px, py = np.floor(point / 4 + 0.5)
            gaussian = np.exp(-((x - px) ** 2 + (y - py) ** 2) / 8)
            gaussian[(np.abs(x - px) > 6) | (np.abs(y - py) > 6)] = 0
            yield np.tile(gaussian.astype(np.float32), (1, 17, 1, 1))

    monkeypatch.setattr(cv2, "warpAffine", record_crop)
    session.outputs = ideal_outputs()

    result = estimate_keypoints(_frames(), _track(), "balanced", flip_test, _models())

    np.testing.assert_allclose(result.kp[0], np.tile(source, (17, 1)), atol=0.05, rtol=0)


def test_vitpose_receives_normalized_rgb_and_a_flipped_crop(session: FakeSession) -> None:
    estimate_keypoints(_frames(), _track(), Mode.LIGHTWEIGHT, True, _models())

    original, flipped = session.inputs
    assert original.shape == (1, 3, 256, 192)
    assert original.dtype == np.float32
    assert original.flags.c_contiguous
    assert flipped.flags.c_contiguous
    expected_rgb = (np.array([160, 64, 128]) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
    np.testing.assert_allclose(original[0, :, 128, 96], expected_rgb, atol=1e-6)
    np.testing.assert_array_equal(flipped, original[..., ::-1])


def test_disagreeing_passes_average_heatmaps_instead_of_decoded_poses(session: FakeSession) -> None:
    original = _heatmaps()
    session.outputs = iter([original.copy()])
    reference = estimate_keypoints(_frames(), _track(), "balanced", False, _models())
    displaced = np.roll(original, 12, axis=2) * 0.5
    displaced[:, :, :12] = 0
    session.outputs = iter([original.copy(), _flipped(displaced)])

    result = estimate_keypoints(_frames(), _track(), "balanced", True, _models())

    # Separated weaker peaks halve the confidence, rather than moving joints halfway.
    np.testing.assert_allclose(result.kp, reference.kp, atol=0.1, rtol=0)
    np.testing.assert_allclose(result.scores, reference.scores * 0.5, atol=1e-6)


@pytest.mark.parametrize("mode", list(Mode))
def test_no_flip_runs_once_with_the_selected_model(session: FakeSession, mode: Mode) -> None:
    result = estimate_keypoints(_frames(), _track(), mode, False, _models())

    assert session.paths == [str(_models()[MODELS[mode][1]])]
    assert len(session.inputs) == 1
    np.testing.assert_allclose(result.scores[0], 0.4 + np.arange(17) / 50, atol=1e-6)


def test_missing_boxes_are_skipped_and_reused_boxes_are_estimated(session: FakeSession) -> None:
    session.outputs = iter([_heatmaps(), _heatmaps()])
    track = _track(3)
    track.boxes[0] = np.nan
    track.present[:] = [False, True, False]
    result = estimate_keypoints(_frames(3), track, "balanced", False, _models())

    assert len(session.inputs) == 2
    assert np.isnan(result.kp[0]).all()
    np.testing.assert_array_equal(result.scores[0], 0)
    assert np.isfinite(result.kp[1:]).all()
    np.testing.assert_array_equal(result.kp[1], result.kp[2])
    np.testing.assert_array_equal(result.scores[1], result.scores[2])


def test_empty_frames_return_shaped_arrays(session: FakeSession) -> None:
    result = estimate_keypoints(_frames(0), _track(0), "balanced", False, _models())
    assert result.kp.shape == (0, 17, 2)
    assert result.scores.shape == (0, 17)
    assert not session.inputs


def test_joint_names_and_flip_pairs_follow_ap10k() -> None:
    assert KEYPOINT_NAMES == (
        "L_Eye",
        "R_Eye",
        "Nose",
        "Neck",
        "Root_of_tail",
        "L_Shoulder",
        "L_Elbow",
        "L_F_Paw",
        "R_Shoulder",
        "R_Elbow",
        "R_F_Paw",
        "L_Hip",
        "L_Knee",
        "L_B_Paw",
        "R_Hip",
        "R_Knee",
        "R_B_Paw",
    )
    assert FLIP_PAIRS == ((0, 1), (5, 8), (6, 9), (7, 10), (11, 14), (12, 15), (13, 16))


@pytest.mark.parametrize("shape", [(1, 16, 64, 48), (2, 17, 64, 48), (1, 17, 48, 64)])
@pytest.mark.parametrize("flip_test", [False, True])
def test_invalid_heatmap_shape_fails_loudly(
    session: FakeSession, shape: tuple[int, int, int, int], flip_test: bool
) -> None:
    invalid = np.zeros(shape, dtype=np.float32)
    session.outputs = iter([_heatmaps(), invalid] if flip_test else [invalid])
    with pytest.raises(ValueError, match="heatmap array"):
        estimate_keypoints(_frames(), _track(), "balanced", flip_test, _models())


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1, 0])
@pytest.mark.parametrize("flip_test", [False, True])
def test_invalid_heatmap_peak_is_a_missing_observation(
    session: FakeSession, value: float, flip_test: bool
) -> None:
    invalid = _heatmaps()
    invalid[0, 3] = value
    if not np.isfinite(value):
        invalid[0, 3] = _heatmaps()[0, 3]
        invalid[0, 3, 12, 20] = value
    session.outputs = iter([invalid.copy(), _flipped(invalid)] if flip_test else [invalid])

    result = estimate_keypoints(_frames(), _track(), "balanced", flip_test, _models())

    assert result.scores[0, 3] == 0
    assert np.isnan(result.kp[0, 3]).all()
    assert np.all(result.scores[0, np.arange(17) != 3] > 0)
    assert np.isfinite(result.kp[0, np.arange(17) != 3]).all()


@pytest.mark.parametrize("box", [(0, 0, 0, 1), (2, 0, 1, 1), (0, 0, np.nan, 1), (0, 0, np.inf, 1)])
def test_invalid_boxes_are_rejected(box: tuple[float, float, float, float]) -> None:
    track = Track(np.array([box], dtype=np.float32), np.ones(1, dtype=np.bool_))
    with pytest.raises(ValueError, match="positive area or all NaN"):
        estimate_keypoints(_frames(), track, "balanced", False, _models())


def test_detected_frame_requires_a_box() -> None:
    track = _track()
    track.boxes[:] = np.nan
    with pytest.raises(ValueError, match="Detected frames"):
        estimate_keypoints(_frames(), track, "balanced", False, _models())


def test_track_length_must_match_frames() -> None:
    with pytest.raises(ValueError, match="Track must contain"):
        estimate_keypoints(_frames(), _track(2), "balanced", False, _models())
