"""Stand the drawing on its contact paws and pin them with analytic two-bone IK."""

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from copypaw.animate.bake import BakeMode

__all__ = ["ContactFilter", "Filter", "make_contact_filter"]

type Filter = Callable[[int, npt.NDArray[np.float32]], npt.NDArray[np.float32]]

# Contact order follows the motion file: near-front, far-front, near-back, far-back.
_PAW_JOINTS: tuple[tuple[str, str, str], ...] = (
    ("left_hip_f", "left_knee_f", "left_foot_f"),
    ("right_hip_f", "right_knee_f", "right_foot_f"),
    ("left_hip_b", "left_knee_b", "left_foot_b"),
    ("right_hip_b", "right_knee_b", "right_foot_b"),
)
_END_HIPS: tuple[tuple[str, str], ...] = (
    ("left_hip_f", "right_hip_f"),
    ("left_hip_b", "right_hip_b"),
)
_ROOT = "root"
# A hip slides at most this share of its leg's posed length away from the body.
_MAX_HIP_SLIDE = 0.15
_FULL_REACH = 0.999
_DIMENSIONS = 2


def _cross(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return inclusive first and last frames of each True run."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for frame, value in enumerate(mask):
        if bool(value) and start is None:
            start = frame
        elif not bool(value) and start is not None:
            runs.append((start, frame - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _two_bone_ik(
    hip: npt.NDArray[np.float64],
    knee: npt.NDArray[np.float64],
    foot: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return knee and foot positions after solving toward target.

    Segment lengths come from the unfiltered pose. Targets beyond 0.999 of the
    full reach are pulled back along the hip-to-target line. The knee keeps the
    side it had in the unfiltered pose. A paw within that margin of its target
    is left as posed.
    """
    upper = float(np.linalg.norm(knee - hip))
    lower = float(np.linalg.norm(foot - knee))
    if upper <= 0 or lower <= 0:
        return knee.copy(), foot.copy()
    # A paw already on its target keeps its pose; solving would pull a straight leg short.
    if float(np.linalg.norm(target - foot)) <= (1 - _FULL_REACH) * (upper + lower):
        return knee.copy(), foot.copy()

    offset = target - hip
    distance = float(np.linalg.norm(offset))
    if distance <= 0:
        offset = foot - hip
        distance = float(np.linalg.norm(offset))
        if distance <= 0:
            return knee.copy(), foot.copy()

    direction = offset / distance
    reach = _FULL_REACH * (upper + lower)
    distance = min(max(distance, abs(upper - lower)), reach)
    solved_foot = hip + direction * distance

    cosine = (upper * upper + distance * distance - lower * lower) / (2 * upper * distance)
    angle = math.acos(min(max(cosine, -1.0), 1.0))
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    bend = _cross(knee - hip, foot - knee)
    side = -1.0 if bend > 0 else 1.0
    solved_knee = hip + upper * math.cos(angle) * direction + side * upper * math.sin(angle) * perp
    return solved_knee, solved_foot


class ContactFilter:
    """Stateful filter that first records poses, then stands the drawing on its contact paws."""

    def __init__(
        self,
        joint_names: Sequence[str],
        contacts: npt.NDArray[np.bool_],
        mode: BakeMode,
    ) -> None:
        """Index the joints and the contact runs; the ground is learned on the first pass."""
        if mode not in ("treadmill", "travel"):
            raise ValueError(f"Unknown contact mode {mode!r}")
        self._mode: BakeMode = mode
        self._contacts = np.asarray(contacts, dtype=np.bool_)
        if self._contacts.ndim != _DIMENSIONS or self._contacts.shape[1] != len(_PAW_JOINTS):
            raise ValueError(
                f"contacts must have shape (frames, {len(_PAW_JOINTS)}), got {self._contacts.shape}"
            )
        self._frames = int(self._contacts.shape[0])
        self._joints: dict[str, int] = {}
        for index, name in enumerate(joint_names):
            if name in self._joints:
                raise ValueError(f"Duplicate joint name {name!r}")
            self._joints[name] = index
        for name in (_ROOT, *(name for leg in _PAW_JOINTS for name in leg)):
            if name not in self._joints:
                raise ValueError(f"Joint names must include {name!r}")

        self._recorded: list[npt.NDArray[np.float32] | None] = [None] * self._frames
        self._contact_frames = [
            frame for frame in range(self._frames) if bool(self._contacts[frame].any())
        ]
        self._run_onset: list[npt.NDArray[np.intp]] = [
            np.full(self._frames, -1, dtype=np.intp) for _ in _PAW_JOINTS
        ]
        self._blend_onset: list[npt.NDArray[np.intp]] = [
            np.full(self._frames, -1, dtype=np.intp) for _ in _PAW_JOINTS
        ]
        for paw in range(len(_PAW_JOINTS)):
            for start, end in _runs(self._contacts[:, paw]):
                self._run_onset[paw][start : end + 1] = start
                if start > 0:
                    self._blend_onset[paw][start - 1] = start
                if end + 1 < self._frames:
                    self._blend_onset[paw][end + 1] = start
        self._ground: list[float | None] | None = None
        self._carried = np.zeros(self._frames)
        self._ready = False

    def _ground_levels(self) -> list[float | None] | None:
        """Return each paw's ground height once every contact frame is recorded.

        A paw's ground is as far below the root as the paw reaches while
        planted, measured from the median root height. Paws drawn at
        different heights therefore keep their own ground lines.
        """
        rows = [self._recorded[frame] for frame in self._contact_frames]
        if any(row is None for row in rows):
            return None
        root = self._joints[_ROOT]
        root_heights = [float(row[root, 1]) for row in self._recorded if row is not None]
        levels: list[float | None] = []
        for paw, (_, _, foot_name) in enumerate(_PAW_JOINTS):
            foot = self._joints[foot_name]
            drops = [
                float(row[foot, 1] - row[root, 1])
                for frame, row in zip(self._contact_frames, rows, strict=True)
                if row is not None and bool(self._contacts[frame, paw])
            ]
            levels.append(float(np.median(root_heights)) + min(drops) if drops else None)
        return levels

    def _carry(self) -> npt.NDArray[np.float64]:
        """Return how far each frame moves sideways so the drawing travels on its own paws.

        The captured travel is the pet's. A drawing with other leg proportions
        covers a different distance per step, so between frames it advances by
        as much as its planted paws swing back under it, and coasts at the
        last such advance while no paw is planted. Treadmill mode stays put.
        """
        carried = np.zeros(self._frames)
        rows = self._recorded
        if self._mode == "treadmill" or any(row is None for row in rows):
            return carried
        root = self._joints[_ROOT]
        feet = [self._joints[foot] for _, _, foot in _PAW_JOINTS]
        first = rows[0]
        assert first is not None
        own = 0.0
        advance = 0.0
        for frame in range(1, self._frames):
            before, now = rows[frame - 1], rows[frame]
            assert before is not None
            assert now is not None
            swings = [
                float(now[foot, 0] - now[root, 0]) - float(before[foot, 0] - before[root, 0])
                for paw, foot in enumerate(feet)
                if bool(self._contacts[frame - 1, paw]) and bool(self._contacts[frame, paw])
            ]
            if swings:
                advance = -float(np.mean(swings))
            own += advance
            carried[frame] = own - float(now[root, 0] - first[root, 0])
        return carried

    def lowest_ground(self) -> float | None:
        """Return the lowest paw's ground height, once the first pass has found it."""
        levels = [level for level in self._ground or () if level is not None]
        return min(levels) if levels else None

    def _leg_indices(self, paw: int) -> tuple[int, int, int]:
        hip, knee, foot = _PAW_JOINTS[paw]
        return self._joints[hip], self._joints[knee], self._joints[foot]

    def _target_x(self, paw: int, onset: int, positions: npt.NDArray[np.float32]) -> float:
        foot = self._joints[_PAW_JOINTS[paw][2]]
        if self._mode == "treadmill":
            return float(positions[foot, 0])
        row = self._recorded[onset]
        if row is None:
            return float(positions[foot, 0])
        return float(row[foot, 0]) + float(self._carried[onset])

    def _solve_leg(
        self, paw: int, onset: int, positions: npt.NDArray[np.float32], ground: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the leg toward the ground where its paw was planted.

        The leg has already slid that way as far as its hip allows. If the
        spot is still out of reach the paw slips along the ground under the
        leg; reaching back for it would lay the leg flat.
        """
        hip_i, knee_i, foot_i = self._leg_indices(paw)
        hip, knee, foot = (
            np.asarray(positions[joint], dtype=np.float64) for joint in (hip_i, knee_i, foot_i)
        )
        planted = np.array([self._target_x(paw, onset, positions), ground], dtype=np.float64)
        length = float(np.linalg.norm(knee - hip)) + float(np.linalg.norm(foot - knee))
        if float(np.linalg.norm(planted - hip)) > _FULL_REACH * length:
            planted[0] = foot[0]
        return _two_bone_ik(hip, knee, foot, planted)

    def _leg_slide(
        self, paw: int, onset: int, positions: npt.NDArray[np.float32], ground: float
    ) -> tuple[float, float]:
        """Return how far the leg, as posed, must move to stand its paw where it is planted."""
        foot = positions[self._leg_indices(paw)[2]]
        return self._target_x(paw, onset, positions) - float(foot[0]), ground - float(foot[1])

    def _slide_limit(self, paw: int, positions: npt.NDArray[np.float32]) -> float:
        hip_i, _, foot_i = self._leg_indices(paw)
        offset = np.asarray(positions[foot_i], dtype=np.float64) - positions[hip_i]
        return _MAX_HIP_SLIDE * math.hypot(*offset.tolist())

    def _settled(
        self, frame: int, positions: npt.NDArray[np.float32], ground: list[float | None]
    ) -> npt.NDArray[np.float32]:
        """Lower, raise and tip the whole drawing until its contact paws stand on their ground.

        Each end of the body takes the average rise its contact legs ask for,
        and the body tips between its front and back hips when both ends have
        one. Each contact leg then slides at the hip by what is left, as a
        shoulder does, so it stands where it is planted in its posed shape
        instead of floating or bending. The slide is limited to a share of the
        leg's length, so a paw wrongly marked as planted cannot pull its leg
        off the body. A leg entering or leaving contact slides halfway.
        """
        slides: dict[int, tuple[float, float, bool]] = {}
        for paw, level in enumerate(ground):
            onset = int(self._run_onset[paw][frame])
            planted = onset >= 0
            if not planted:
                onset = int(self._blend_onset[paw][frame])
            if level is not None and onset >= 0:
                slides[paw] = (*self._leg_slide(paw, onset, positions, level), planted)
        ends = [
            [rise for paw, (_, rise, planted) in slides.items() if planted and paw // 2 == end]
            for end in range(len(_END_HIPS))
        ]
        front, back = (float(np.mean(rises)) if rises else None for rises in ends)
        settled = positions.astype(np.float32, copy=True)
        if front is None and back is None:
            return settled
        front_x, back_x = (
            float(np.mean([positions[self._joints[name], 0] for name in names]))
            for names in _END_HIPS
        )
        if front is None or back is None or front_x == back_x:
            level_rise = back if front is None else front if back is None else (front + back) / 2
            body = np.full(len(positions), level_rise)
        else:
            body = back + (front - back) * (positions[:, 0] - back_x) / (front_x - back_x)
        aside = np.zeros(len(positions))
        for paw in range(len(_PAW_JOINTS)):
            hip, knee, foot = self._leg_indices(paw)
            if paw in slides:
                across, rise, planted = slides[paw]
                weight = 1.0 if planted else 0.5
                # A hip that slid further would leave the body; the solver takes the rest.
                limit = self._slide_limit(paw, positions)
                body[hip] += min(max(weight * (rise - body[hip]), -limit), limit)
                aside[[hip, knee, foot]] = min(max(weight * across, -limit), limit)
            body[[knee, foot]] = body[hip]
        settled[:, 0] += aside.astype(np.float32)
        settled[:, 1] += body.astype(np.float32)
        return settled

    def _filter(self, frame: int, positions: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert self._ground is not None
        positions = positions.copy()
        positions[:, 0] += np.float32(self._carried[frame])
        positions = self._settled(frame, positions, self._ground)
        filtered = positions.copy()
        for paw, ground in enumerate(self._ground):
            if ground is None:
                continue
            onset = int(self._run_onset[paw][frame])
            if onset >= 0:
                knee, foot = self._solve_leg(paw, onset, positions, ground)
                _, knee_i, foot_i = self._leg_indices(paw)
                filtered[knee_i] = knee.astype(np.float32)
                filtered[foot_i] = foot.astype(np.float32)
                continue
            onset = int(self._blend_onset[paw][frame])
            if onset >= 0:
                knee, foot = self._solve_leg(paw, onset, positions, ground)
                _, knee_i, foot_i = self._leg_indices(paw)
                filtered[knee_i] = (
                    (knee + np.asarray(positions[knee_i], dtype=np.float64)) * 0.5
                ).astype(np.float32)
                filtered[foot_i] = (
                    (foot + np.asarray(positions[foot_i], dtype=np.float64)) * 0.5
                ).astype(np.float32)
        return filtered

    def __call__(
        self, frame_index: int, positions: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Return world positions for one frame, filtering after the first pass."""
        array = np.asarray(positions, dtype=np.float32)
        if array.ndim != _DIMENSIONS or array.shape[1] != _DIMENSIONS:
            raise ValueError(f"positions must have shape (joints, 2), got {array.shape}")
        if array.shape[0] != len(self._joints):
            raise ValueError(
                f"positions has {array.shape[0]} joints but {len(self._joints)} names were given"
            )
        if frame_index < 0 or frame_index >= self._frames:
            return array.copy()

        if self._recorded[frame_index] is None:
            self._recorded[frame_index] = array.copy()
            return array.copy()

        if not self._ready:
            ground = self._ground_levels()
            if ground is None:
                return array.copy()
            self._ground = ground
            self._carried = self._carry()
            self._ready = True
        return self._filter(frame_index, array)


def make_contact_filter(
    joint_names: Sequence[str],
    contacts: npt.NDArray[np.bool_],
    mode: BakeMode,
) -> ContactFilter:
    """Build a filter that stands a drawing on its contact paws in world space.

    The first pass records positions without changing them, so each paw's
    ground can be the lowest it reaches under the root while planted. Later
    passes lower, raise and tip the whole drawing so its contact paws stand on
    their ground, solve each contact leg toward it and blend halfway into and
    out of each contact run. Joint names must follow the engine's depth-first traversal
    order.
    """
    return ContactFilter(joint_names, contacts, mode)
