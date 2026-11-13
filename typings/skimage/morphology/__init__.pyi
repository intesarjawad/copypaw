from typing import Literal

import numpy as np
from numpy.typing import NDArray

def binary_opening(
    image: NDArray[np.generic],
    footprint: NDArray[np.generic] | None = None,
    out: NDArray[np.bool_] | None = None,
    *,
    mode: Literal["ignore", "min", "max"] = "ignore",
) -> NDArray[np.bool_]: ...
def disk(
    radius: int,
    *,
    strict_radius: bool = True,
    decomposition: None = None,
) -> NDArray[np.uint8]: ...
def skeletonize(image: NDArray[np.generic]) -> NDArray[np.bool_]: ...
