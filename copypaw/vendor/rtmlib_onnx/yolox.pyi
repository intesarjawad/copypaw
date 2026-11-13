import numpy as np
from numpy.typing import NDArray

from .base import BaseTool

class YOLOX(BaseTool):
    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple[int, int] | None = None,
        nms_thr: float = 0.45,
        score_thr: float = 0.7,
    ) -> None: ...
    def __call__(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...
    def preprocess(self, img: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], float]: ...
    def postprocess(
        self, outputs: NDArray[np.float32], ratio: float = 1.0
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...
