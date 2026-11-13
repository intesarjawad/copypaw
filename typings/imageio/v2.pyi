from collections.abc import Sequence
from os import PathLike

import numpy as np
from numpy.typing import NDArray

def imread(uri: str | PathLike[str]) -> NDArray[np.generic]: ...
def mimsave(
    uri: str | PathLike[str],
    ims: Sequence[NDArray[np.generic]],
    *,
    fps: float,
) -> None: ...
