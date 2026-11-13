import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from copypaw.capture.detect import YOLOX, NoAnimalFoundError, track_animal
from copypaw.capture.video import Frames
from copypaw.models import REGISTRY
from copypaw.vendor.rtmlib_onnx.vitpose import ViTPose


def _frames(count: int, height: int = 24, width: int = 32) -> Frames:
    rgb = np.zeros((count, height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    return Frames(rgb, np.arange(count, dtype=np.float64) / 30, 30, 30, False)


def _detections(
    *rows: tuple[float, float, float, float, int],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    data = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    return data[:, :4], data[:, 4].astype(np.int64)


class FakeDetector:
    def __init__(self, *results: tuple[NDArray[np.float32], NDArray[np.int64]]) -> None:
        self.results = iter(results)

    def __call__(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        assert image.shape[-1] == 3
        return next(self.results)


@dataclass
class FakeInput:
    name: str
    shape: list[int | str | None]


class FakeSession:
    def __init__(self) -> None:
        self.inputs: list[NDArray[np.float32]] = []

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("images", [1, 3, 32, 32])]

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]:
        assert output_names is None
        self.inputs.append(input_feed["images"].copy())
        # 32x32 has 16 + 4 + 1 anchors over strides 8, 16, 32.
        output = np.zeros((1, 21, 85), dtype=np.float32)
        output[0, 0, :4] = [2, 2, np.log(2), np.log(2)]
        output[0, 0, 4] = 1
        output[0, 0, 5 + 16] = 0.9
        return [output]


@pytest.fixture
def fake_yolox(monkeypatch: pytest.MonkeyPatch) -> tuple[YOLOX, FakeSession]:
    session = FakeSession()

    def make_session(path_or_bytes: str, *, providers: list[str]) -> FakeSession:
        assert path_or_bytes == "test.onnx"
        assert providers == ["CPUExecutionProvider"]
        return session

    monkeypatch.setattr("onnxruntime.InferenceSession", make_session)
    return YOLOX("test.onnx"), session


def test_rgb_red_reaches_yolox_session_as_bgr(fake_yolox: tuple[YOLOX, FakeSession]) -> None:
    detector, session = fake_yolox
    track = track_animal(_frames(1), detector)

    tensor = session.inputs[0]
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor.shape == (1, 3, 32, 32)
    np.testing.assert_array_equal(tensor[0, :, 0, 0], [0, 0, 255])
    np.testing.assert_array_equal(tensor[0, :, 31, 0], [114, 114, 114])
    np.testing.assert_allclose(track.boxes, [[8, 8, 24, 24]])
    assert track.present.tolist() == [True]


def test_yolox_restores_source_coordinates(fake_yolox: tuple[YOLOX, FakeSession]) -> None:
    detector, _ = fake_yolox
    boxes, classes = detector(np.zeros((16, 16, 3), dtype=np.uint8))
    np.testing.assert_allclose(boxes, [[4, 4, 12, 12]])
    np.testing.assert_array_equal(classes, [16])


def test_yolox_class_aware_nms_suppresses_only_same_class(
    fake_yolox: tuple[YOLOX, FakeSession],
) -> None:
    detector, _ = fake_yolox
    output = np.zeros((1, 21, 85), dtype=np.float32)
    for anchor, class_id, confidence in [(0, 16, 0.9), (1, 16, 0.8), (2, 15, 0.85)]:
        output[0, anchor, :4] = [2 - anchor, 2, np.log(2), np.log(2)]
        output[0, anchor, 4] = 1
        output[0, anchor, 5 + class_id] = confidence
    boxes, classes = detector.postprocess(output)
    np.testing.assert_allclose(boxes, [[8, 8, 24, 24], [8, 8, 24, 24]])
    np.testing.assert_array_equal(classes, [15, 16])


def test_yolox_no_scores_returns_shaped_empty_arrays(
    fake_yolox: tuple[YOLOX, FakeSession],
) -> None:
    detector, _ = fake_yolox
    boxes, classes = detector.postprocess(np.zeros((1, 21, 85), dtype=np.float32))
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)
    assert boxes.dtype == np.float32
    assert classes.dtype == np.int64


@pytest.mark.parametrize("shape", [(1, 21, 5), (2, 21, 85), (1, 20, 85)])
def test_yolox_rejects_incompatible_outputs(
    fake_yolox: tuple[YOLOX, FakeSession], shape: tuple[int, int, int]
) -> None:
    detector, _ = fake_yolox
    with pytest.raises(ValueError, match="YOLOX"):
        detector.postprocess(np.zeros(shape, dtype=np.float32))


def test_person_larger_than_dog_is_ignored() -> None:
    detector = FakeDetector(_detections((0, 0, 100, 100, 0), (5, 6, 15, 16, 16)))
    np.testing.assert_array_equal(track_animal(_frames(1), detector).boxes, [[5, 6, 15, 16]])


@pytest.mark.parametrize("class_id", range(80))
def test_only_the_nine_animal_classes_are_accepted(class_id: int) -> None:
    detector = FakeDetector(_detections((0, 0, 10, 10, class_id)))
    # Zero-based order published in COCO's category listing:
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
    # Cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe; bird is excluded.
    if class_id in {15, 16, 17, 18, 19, 20, 21, 22, 23}:
        assert track_animal(_frames(1), detector).present.all()
    else:
        with pytest.raises(NoAnimalFoundError, match="half the frames"):
            track_animal(_frames(1), detector)


def test_tracks_highest_iou_even_when_other_dog_grows() -> None:
    detector = FakeDetector(
        _detections((0, 0, 10, 10, 16), (20, 20, 25, 25, 16)),
        _detections((20, 20, 100, 100, 16), (1, 0, 11, 10, 16)),
        _detections((0, 0, 15, 15, 16), (2, 0, 12, 10, 16)),
    )
    np.testing.assert_array_equal(
        track_animal(_frames(3), detector).boxes,
        [[0, 0, 10, 10], [1, 0, 11, 10], [2, 0, 12, 10]],
    )


@pytest.mark.parametrize(("offset", "expected"), [(7, [7, 0, 20, 10]), (8, [30, 0, 60, 20])])
def test_iou_threshold_is_inclusive_and_otherwise_uses_largest(
    offset: int, expected: list[int]
) -> None:
    detector = FakeDetector(
        _detections((0, 0, 13, 10, 16)),
        _detections((offset, 0, offset + 13, 10, 16), (30, 0, 60, 20, 16)),
    )
    np.testing.assert_array_equal(track_animal(_frames(2), detector).boxes[1], expected)


def test_fourth_missing_frame_has_no_box_and_reacquires_largest() -> None:
    dog = _detections((0, 0, 10, 10, 16))
    detector = FakeDetector(
        *([dog] * 5),
        *([_detections()] * 4),
        _detections((0, 0, 10, 10, 16), (20, 20, 40, 40, 16)),
    )
    track = track_animal(_frames(10), detector)
    np.testing.assert_array_equal(track.boxes[5:8], np.tile([0, 0, 10, 10], (3, 1)))
    assert np.isnan(track.boxes[8]).all()
    np.testing.assert_array_equal(track.boxes[9], [20, 20, 40, 40])
    assert track.present.tolist() == [True] * 5 + [False] * 4 + [True]


def test_leading_gap_and_exactly_half_present_are_allowed() -> None:
    track = track_animal(_frames(2), FakeDetector(_detections(), _detections((0, 0, 10, 10, 15))))
    assert np.isnan(track.boxes[0]).all()
    assert track.present.tolist() == [False, True]
    assert track.boxes.dtype == np.float32
    assert track.present.dtype == np.bool_


def test_reused_boxes_do_not_count_toward_half_present() -> None:
    with pytest.raises(NoAnimalFoundError, match="half the frames"):
        track_animal(
            _frames(3),
            FakeDetector(_detections((0, 0, 10, 10, 16)), _detections(), _detections()),
        )


def test_empty_frames_raise_no_animal() -> None:
    with pytest.raises(NoAnimalFoundError):
        track_animal(_frames(0), FakeDetector())


@pytest.mark.parametrize("box", [(0, 0, 0, 1, 16), (2, 0, 1, 1, 16), (0, 0, np.nan, 1, 16)])
def test_invalid_detector_boxes_fail_loudly(box: tuple[float, float, float, float, int]) -> None:
    with pytest.raises(ValueError, match="positive area"):
        track_animal(_frames(1), FakeDetector(_detections(box)))


def test_mismatched_detector_classes_fail_loudly() -> None:
    detector = FakeDetector((np.zeros((2, 4), dtype=np.float32), np.array([16], dtype=np.int64)))
    with pytest.raises(ValueError, match="classes"):
        track_animal(_frames(1), detector)


@pytest.mark.slow
@pytest.mark.parametrize("model_key", ["yolox_tiny", "vitpose_s_apt36k"])
def test_cached_lightweight_models_run_on_cpu(model_key: str) -> None:
    model = REGISTRY[model_key]
    cache = Path(os.environ.get("COPYPAW_CACHE", "~/.cache/copypaw/models")).expanduser()
    path = cache / model.filename
    if not path.is_file():
        pytest.skip(f"Cached model is absent: {model.filename}")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    if model_key == "yolox_tiny":
        detector = YOLOX(str(path))
        boxes, classes = detector(image)
        assert detector.model_input_size == (416, 416)
        assert boxes.shape == (len(classes), 4)
        assert boxes.dtype == np.float32
        assert classes.dtype == np.int64
        assert np.isfinite(boxes).all()
    else:
        pose = ViTPose(str(path))
        keypoints, scores = pose(image, np.array([[0, 0, 256, 256]], dtype=np.float32))
        assert keypoints.shape == (1, 17, 2)
        assert scores.shape == (1, 17)
        assert np.isfinite(keypoints).all()
        assert np.isfinite(scores).all()
