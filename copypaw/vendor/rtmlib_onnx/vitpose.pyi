import numpy as np
from numpy.typing import NDArray

from .base import BaseTool

class ViTPose(BaseTool):
    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple[int, int] = (192, 256),
        mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: tuple[float, float, float] = (58.395, 57.12, 57.375),
    ) -> None: ...
    def __call__(
        self, image: NDArray[np.uint8], bboxes: NDArray[np.float32] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]: ...
    def preprocess(
        self, img: NDArray[np.uint8], bbox: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float32]]: ...
    def postprocess(
        self,
        outputs: list[NDArray[np.float32]],
        center: NDArray[np.float32],
        scale: NDArray[np.float32],
        score_threshold: float = 0.0,
        dark_kernel: int = 11,
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]: ...
