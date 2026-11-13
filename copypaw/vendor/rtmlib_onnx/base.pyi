from typing import Protocol

import numpy as np
from numpy.typing import NDArray

class InputMetadata(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> list[int | str | None]: ...

class Session(Protocol):
    def get_inputs(self) -> list[InputMetadata]: ...
    def run(
        self, output_names: list[str] | None, input_feed: dict[str, NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]: ...

class BaseTool:
    session: Session
    model_input_size: tuple[int, int]
    def inference(self, img: NDArray[np.generic]) -> list[NDArray[np.float32]]: ...
