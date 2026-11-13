"""Camera-independent horizontal root travel from planted paw displacements."""

import numpy as np
from numpy.typing import NDArray

from copypaw.capture.contacts import PAW_INDICES, ROOT_KP, body_length_px
from copypaw.capture.smoothing import validate_timestamps

__all__ = ["travel_root_x"]


def travel_root_x(
    kp: NDArray[np.float32], contacts: NDArray[np.bool_], timestamps_s: NDArray[np.float64]
) -> NDArray[np.float32]:
    """Integrate signed horizontal travel in body lengths, starting at zero.

    contacts is a boolean (F, 4) array in near-front, far-front, near-back,
    far-back order. Only paws planted on both sides of a frame interval contribute.
    Average root displacement minus paw displacement cancels camera translation.
    Divide each supported displacement by its elapsed seconds. Intervals without
    support retain the previous speed, initially zero, and integrate over their
    own duration. Accumulation uses float64 and returns a fresh float32 array.
    """
    body_len = body_length_px(kp)
    validate_timestamps(timestamps_s, len(kp))
    if contacts.shape != (len(kp), len(PAW_INDICES)) or contacts.dtype != np.bool_:
        raise ValueError("Expected boolean contacts with shape (F, 4)")
    paw_x = kp[:, PAW_INDICES, 0].astype(np.float64)
    root_x = kp[:, ROOT_KP, 0].astype(np.float64)
    displacement: NDArray[np.float64] = (
        np.diff(root_x)[:, None] - np.diff(paw_x, axis=0)
    ) / body_len
    supported = contacts[:-1] & contacts[1:]
    durations_s = np.diff(timestamps_s)
    result = np.zeros(len(kp), dtype=np.float64)
    previous = 0.0
    for frame in range(1, len(kp)):
        planted = supported[frame - 1]
        if planted.any():
            planted_displacement: NDArray[np.float64] = displacement[frame - 1, planted]
            previous = float(np.mean(planted_displacement)) / durations_s[frame - 1]
        result[frame] = result[frame - 1] + previous * durations_s[frame - 1]
    return result.astype(np.float32)
