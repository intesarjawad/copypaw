from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

@overload
def label(
    label_image: NDArray[np.generic],
    background: int | None = None,
    return_num: Literal[False] = False,
    connectivity: int | None = None,
) -> NDArray[np.int32 | np.int64]: ...
@overload
def label(
    label_image: NDArray[np.generic],
    background: int | None = None,
    *,
    return_num: Literal[True],
    connectivity: int | None = None,
) -> tuple[NDArray[np.int32 | np.int64], int]: ...
