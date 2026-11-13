"""Animation joints on identity-resolved tracks, with near limbs in left slots."""

from dataclasses import dataclass
from typing import Literal

__all__ = ["BONES", "HELPER_JOINTS", "ROOT_KP", "SIDE_OFFSET_RATIO", "Bone"]


@dataclass(frozen=True)
class Bone:
    """A segment named for its distal joint; a missing parent attaches to Root."""

    name: str
    parent: str | None
    prox_kp: int
    dist_kp: int
    side: Literal["near", "far", "centre"]


BONES = (
    Bone("Neck", None, 4, 3, "centre"),
    Bone("Nose", "Neck", 3, 2, "centre"),
    Bone("Near_Shoulder", "Neck", 3, 5, "near"),
    Bone("Near_Elbow", "Near_Shoulder", 5, 6, "near"),
    Bone("Near_F_Paw", "Near_Elbow", 6, 7, "near"),
    Bone("Far_Shoulder", "Neck", 3, 8, "far"),
    Bone("Far_Elbow", "Far_Shoulder", 8, 9, "far"),
    Bone("Far_F_Paw", "Far_Elbow", 9, 10, "far"),
    Bone("Near_Hip", None, 4, 11, "near"),
    Bone("Near_Knee", "Near_Hip", 11, 12, "near"),
    Bone("Near_B_Paw", "Near_Knee", 12, 13, "near"),
    Bone("Far_Hip", None, 4, 14, "far"),
    Bone("Far_Knee", "Far_Hip", 14, 15, "far"),
    Bone("Far_B_Paw", "Far_Knee", 15, 16, "far"),
)
ROOT_KP = 4
SIDE_OFFSET_RATIO = 0.08  # Chain depth as a fraction of Root to Neck body length.
# Static Root children: (name, z offset), ordered to define forward as +x.
HELPER_JOINTS = (("Fwd_Near", 0.5), ("Fwd_Far", -0.5))
