from collections.abc import Mapping, Sequence
from os import PathLike

import numpy as np
from numpy.typing import NDArray

type TensorValue = (
    NDArray[np.generic]
    | SparseTensor
    | str
    | int
    | float
    | bool
    | list[TensorValue]
    | dict[str | int, TensorValue]
)

class SparseTensor: ...

class NodeArg:
    @property
    def name(self) -> str: ...

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes | PathLike[str],
        *,
        providers: Sequence[str] | None = None,
    ) -> None: ...
    def get_inputs(self) -> Sequence[NodeArg]: ...
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, TensorValue],
    ) -> Sequence[TensorValue]: ...

def get_available_providers() -> list[str]: ...
