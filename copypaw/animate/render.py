"""Render a drawing through the animation engine and return its RGBA frames."""

import math
import multiprocessing
import os
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import cast

import cv2
import glfw
import numpy as np
import yaml
from numpy.typing import NDArray

from copypaw.animate.bake import BakeMode, pose_for_drawing
from copypaw.animate.configs import AnimationConfigs, write_configs
from copypaw.animate.contact_ik import ContactFilter, make_contact_filter
from copypaw.animate.handles import tail_handles, with_bone_handles
from copypaw.capture.video import OUTPUT_FPS
from copypaw.errors import CopypawError
from copypaw.motion.bvh_writer import write_bvh
from copypaw.motion.motion_file import MotionFile
from copypaw.vendor.animated_drawings.config import Config, ControllerConfig
from copypaw.vendor.animated_drawings.controller.video_render_controller import (
    VideoRenderController,
)
from copypaw.vendor.animated_drawings.model.animated_drawing import (
    AnimatedDrawing,
    AnimatedDrawingRig,
)
from copypaw.vendor.animated_drawings.model.arap import ARAP
from copypaw.vendor.animated_drawings.model.scene import Scene
from copypaw.vendor.animated_drawings.view.view import View

__all__ = [
    "CharacterError",
    "RenderResult",
    "TravelCanvas",
    "frame_camera",
    "posed_scene",
    "render_character",
    "resample_motion",
    "run_engine",
    "travel_canvas",
]

_BAKED_BVH = "baked.bvh"
_ANIMATION_GIF = "animation.gif"
_RENDER_JOINTS = "render_joints.npz"
_RENDER_CONTACTS = "render_contacts.npz"
_MANIFEST = "copypaw.yaml"
_ROOT_JOINT = "Root"
_ROOT_CHAR_JOINT = "root"
_RENDER_TIMEOUT_S = 600
_DEFAULT_FRAME_MS = 1000 / OUTPUT_FPS
_RGB_CHANNELS = 3
_RGBA_CHANNELS = 4
_FRAME_DIMENSIONS = 3
_JOINT_DIMENSIONS = 3
_JOINT_XY = 2
_IMAGEIO = import_module("imageio.v3")

# The canvas predicts the drawing's projected width before the engine has a
# context to measure it. Travel is stored in body lengths, so the character's
# body length turns it into the same drawing units the box uses.
_WINDOW_HEIGHT = 512
_TRAVEL_MARGIN_RATIO = 0.1
_MAX_CANVAS_ASPECT = 4
_CHARACTER_CFG = "char_cfg.yaml"
_MASK_NAME = "mask.png"
_TEXTURE_NAME = "texture.png"
_BOX_COORDINATES = 2
# The engine's default view, recorded here because the canvas must predict the
# drawing's horizontal size before the engine has a context to measure it.
_CHARACTER_DEPTH = -0.5
# Vertical field of view the engine's projection is built with.
_FIELD_OF_VIEW_DEG = 35.0
# Clear space kept around the drawing's reach, as a share of that reach.
_FRAME_MARGIN_RATIO = 0.05
_CAMERA_STRAIGHT_ON = (0.0, 0.0, 1.0)
# The solver forms its normal equations in single precision, and they hold this
# weight squared. Its default of 1000 loses the drawn pose far from any handle;
# 100 pins the handles as firmly and returns the drawn pose when nothing moves.
_HANDLE_WEIGHT = 100
# A tail turns this share of the hind legs' swing, and takes about this long to follow.
_TAIL_SWAY = 0.3
_TAIL_LAG_S = 0.2
# A tail drawn down to the paws' level is held this far off the ground, or it reads as a fifth leg.
_TAIL_LIFT = 0.6


@dataclass(frozen=True)
class RenderResult:
    """Frames and drawing joints sampled at the output frame rate."""

    frames_rgba: NDArray[np.uint8]
    fps: float
    joints_per_frame: NDArray[np.float32]
    unfiltered_joints_per_frame: NDArray[np.float32] | None = None
    joint_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TravelCanvas:
    """Canvas size and first-frame offset for a drawing that travels."""

    width: int
    height: int
    frames: int
    start_x: float


def _coordinate(value: object, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Character joint {name!r} needs numeric coordinates")
    return float(value)


class CharacterError(CopypawError):
    """The folder passed as a drawing does not hold a rigged drawing."""

    codes = ("character_missing",)


def _character_joints(character: Path) -> tuple[int, dict[str, tuple[float, float]]]:
    """Return img_dim and the joint positions in drawing pixels."""
    config = character / _CHARACTER_CFG
    data = cast(
        "dict[str, object]",
        yaml.safe_load(config.read_text(encoding="utf-8")) or {},
    )
    width = data.get("width")
    height = data.get("height")
    skeleton = data.get("skeleton")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"Character config needs positive width and height: {config}")
    if not isinstance(skeleton, list):
        raise ValueError(f"Character config needs a skeleton list: {config}")

    joints: dict[str, tuple[float, float]] = {}
    for entry in cast("list[object]", skeleton):
        item = cast("dict[str, object]", entry)
        name = item.get("name")
        location = item.get("loc")
        if not isinstance(name, str) or not isinstance(location, list):
            raise ValueError(f"Character skeleton entry needs a name and loc: {config}")
        coordinates = cast("list[object]", location)
        if len(coordinates) != _BOX_COORDINATES:
            raise ValueError(f"Character joint {name!r} needs two coordinates: {config}")
        joints[name] = (
            _coordinate(coordinates[0], name),
            _coordinate(coordinates[1], name),
        )
    return max(width, height), joints


def _character_geometry(character: Path) -> tuple[int, float, float, float, float]:
    """Return img_dim, box left, box width, box height and body length in drawing pixels."""
    config = character / _CHARACTER_CFG
    img_dim, joints = _character_joints(character)
    root = joints.get("root", joints.get("hip"))
    front_hip = joints.get("left_hip_f")
    back_hip = joints.get("left_hip_b")
    if root is None or front_hip is None or back_hip is None:
        raise ValueError(f"Character config needs root, left_hip_f and left_hip_b: {config}")
    body_length = math.hypot(front_hip[0] - back_hip[0], front_hip[1] - back_hip[1])

    mask = cv2.imread(str(character / _MASK_NAME), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read character mask: {character / _MASK_NAME}")
    columns = np.flatnonzero(mask.any(axis=0))
    if columns.size == 0:
        raise ValueError(f"Character mask is empty: {character / _MASK_NAME}")
    return (
        img_dim,
        float(columns[0]) - root[0],
        float(columns[-1] - columns[0] + 1),
        float(np.count_nonzero(mask.any(axis=1))),
        body_length,
    )


def travel_canvas(
    character: Path, motion: MotionFile, *, height: int = _WINDOW_HEIGHT
) -> TravelCanvas:
    """Size a canvas that fits the drawing's box plus its whole path.

    The drawing starts one margin from the left edge and ends one margin from
    the right. A path wider than four times the height keeps only the longest
    prefix that fits and says which mode covers the whole motion.
    """
    if height <= 0:
        raise ValueError(f"Canvas height must be positive, got {height}")
    travel = np.asarray(motion.travel_root_x, dtype=np.float64)
    if len(travel) == 0 or len(travel) != len(motion.kp):
        raise ValueError("Travel must hold one value per frame")
    if not np.isfinite(travel).all():
        raise ValueError("Travel must contain finite values")

    img_dim, box_left_px, box_width_px, box_height_px, body_px = _character_geometry(character)
    box_left = box_left_px / img_dim
    box_width = box_width_px / img_dim
    # The camera fits the drawing's height to the canvas less its margin (frame_camera),
    # so a canvas as wide as the path at that scale shows the whole of it.
    pixels_per_unit = height / ((1 + 2 * _FRAME_MARGIN_RATIO) * box_height_px / img_dim)
    span = (np.maximum.accumulate(travel) - np.minimum.accumulate(travel)) * body_px / img_dim
    widths = (1 + 2 * _TRAVEL_MARGIN_RATIO) * (box_width + span) * pixels_per_unit
    fits = np.flatnonzero(widths <= _MAX_CANVAS_ASPECT * height)
    frames = int(fits[-1]) + 1 if fits.size else 1
    if frames < len(travel):
        print(
            f"Warning: travel does not fit in {_MAX_CANVAS_ASPECT} times the canvas height; "
            f"keeping the first {frames} of {len(travel)} frames. "
            "Use --mode treadmill for the whole motion.",
            file=sys.stderr,
        )

    width = max(1, math.ceil(float(widths[frames - 1])))
    margin = _TRAVEL_MARGIN_RATIO * (box_width + float(span[frames - 1]))
    # The root sits at the middle of the view, so the box's own offset from the
    # root is what decides where the drawing starts.
    start_x = -width / (2 * pixels_per_unit) + margin - box_left
    return TravelCanvas(width=width, height=height, frames=frames, start_x=start_x)


def _prefix(motion: MotionFile, frames: int) -> MotionFile:
    if frames >= len(motion.kp):
        return motion
    return MotionFile(
        kp=motion.kp[:frames].copy(),
        scores=motion.scores[:frames].copy(),
        fps=motion.fps,
        facing=motion.facing,
        contacts=motion.contacts[:frames].copy(),
        ground_line_px=motion.ground_line_px,
        travel_root_x=motion.travel_root_x[:frames].copy(),
        source=motion.source,
        version=motion.version,
    )


def _lerp(
    values: NDArray[np.float64],
    lower: NDArray[np.intp],
    upper: NDArray[np.intp],
    weight: NDArray[np.float64],
) -> NDArray[np.float64]:
    shape = (-1, *([1] * (values.ndim - 1)))
    weights = weight.reshape(shape)
    return np.asarray(
        values[lower] + (values[upper] - values[lower]) * weights,
        dtype=np.float64,
    )


def resample_motion(motion: MotionFile, fps: int = OUTPUT_FPS) -> MotionFile:
    """Resample keypoints linearly and contacts to their nearest analysis frame.

    Analysis frames are motion.fps apart from the first one. The last output
    time stays at or before the final analysis timestamp, so output frame k is
    at k / fps seconds.
    """
    if not math.isfinite(motion.fps) or motion.fps <= 0:
        raise ValueError("Motion fps must be finite and positive")
    if fps <= 0:
        raise ValueError("Output fps must be positive")
    frames = len(motion.kp)
    if frames == 0:
        raise ValueError("Motion must contain at least one frame")

    duration_s = (frames - 1) / motion.fps
    count = math.floor(duration_s * fps) + 1
    times_s = np.arange(count, dtype=np.float64) / fps
    position = np.clip(times_s * motion.fps, 0.0, frames - 1)
    lower = np.minimum(np.floor(position).astype(np.intp), frames - 1)
    upper = np.minimum(lower + 1, frames - 1)
    weight = position - lower
    nearest = np.minimum(np.floor(times_s * motion.fps + 0.5).astype(np.intp), frames - 1)
    return MotionFile(
        kp=np.asarray(
            _lerp(np.asarray(motion.kp, dtype=np.float64), lower, upper, weight),
            dtype=np.float32,
        ),
        scores=np.asarray(
            _lerp(np.asarray(motion.scores, dtype=np.float64), lower, upper, weight),
            dtype=np.float32,
        ),
        fps=float(fps),
        facing=motion.facing,
        contacts=motion.contacts[nearest],
        ground_line_px=motion.ground_line_px,
        travel_root_x=np.asarray(
            _lerp(np.asarray(motion.travel_root_x, dtype=np.float64), lower, upper, weight),
            dtype=np.float32,
        ),
        source=motion.source,
    )


def _bake(motion: MotionFile, mode: str, directory: Path, character: Path) -> Path:
    if mode not in ("treadmill", "travel"):
        raise ValueError(f"Unknown render mode {mode!r}")
    root_x = (
        np.zeros(len(motion.kp), dtype=np.float32) if mode == "treadmill" else motion.travel_root_x
    )
    bvh = directory / _BAKED_BVH
    _, joints = _character_joints(character)
    posed = pose_for_drawing(motion, joints, still_legs=_legs_guessed(character))
    write_bvh(posed, motion.fps, root_x, out=bvh)
    return bvh


def _include_root_depth(retarget_cfg: Path) -> None:
    """Give the root a projection depth; the body group uses it as its driver."""
    # The config writer emits !!python/tuple for joint mappings, which safe_load rejects.
    data = yaml.load(retarget_cfg.read_text(encoding="utf-8"), Loader=yaml.FullLoader)  # noqa: S506
    groups = data["bvh_projection_bodypart_groups"]
    names = groups[0]["bvh_joint_names"]
    if _ROOT_JOINT not in names:
        names.insert(0, _ROOT_JOINT)
    retarget_cfg.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")


def _prepare_motion_config(motion_cfg: Path, frame_count: int) -> None:
    """Make the engine read every frame and face the skeleton along +X.

    Its end index is exclusive. The far-to-near helper pair points forward
    because the up axis crossed with near-minus-far faces away from +X.
    """
    data = cast(
        "dict[str, object]",
        yaml.safe_load(motion_cfg.read_text(encoding="utf-8")) or {},
    )
    data["end_frame_idx"] = frame_count
    data["forward_perp_joint_vectors"] = [["Fwd_Far", "Fwd_Near"]]
    motion_cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _manifest(character: Path) -> dict[str, object]:
    manifest = character / _MANIFEST
    if not manifest.is_file():
        return {}
    return cast(
        "dict[str, object]",
        yaml.safe_load(manifest.read_text(encoding="utf-8")) or {},
    )


def _legs_guessed(character: Path) -> bool:
    return _manifest(character).get("legs_guessed") is True


def _tails(character: Path) -> NDArray[np.float32]:
    """Return each recorded tail's base and tip with y up from the drawing's foot, in pixels."""
    recorded = _manifest(character).get("tails", [])
    tails = np.asarray(recorded, dtype=np.float32).reshape(-1, 2, 2)
    config = cast(
        "dict[str, object]",
        yaml.safe_load((character / _CHARACTER_CFG).read_text(encoding="utf-8")) or {},
    )
    tails[..., 1] = cast("int", config["height"]) - tails[..., 1]
    return tails


def frame_camera(
    low: NDArray[np.floating], high: NDArray[np.floating], window: tuple[int, int]
) -> tuple[float, float, float]:
    """Return where a straight-on camera shows the box from low to high at full size.

    The box is everything the drawing reaches over the whole motion, so one
    fixed view keeps every frame whole and the body's bob stays visible. The
    tighter of width and height fills the window less a margin on each side.
    """
    width, height = window
    reach = (np.asarray(high, dtype=np.float64) - low) * (1 + 2 * _FRAME_MARGIN_RATIO)
    if width <= 0 or height <= 0 or not np.all(np.isfinite(reach)) or np.any(reach <= 0):
        raise ValueError("The drawing's reach and the window must be finite and positive")
    visible_height = max(float(reach[1]), float(reach[0]) * height / width)
    distance = visible_height / (2 * math.tan(math.radians(_FIELD_OF_VIEW_DEG) / 2))
    centre = (np.asarray(low, dtype=np.float64) + high) / 2
    return float(centre[0]), float(centre[1]), _CHARACTER_DEPTH + distance


class _Handles:
    """The engine's control-point hook: stand on the contact paws, then add the extra handles.

    While learning, the contact filter only records each frame; afterwards
    every frame it returns is kept as the drawing's filtered joints.
    """

    def __init__(
        self,
        names: list[str],
        drawn: NDArray[np.float32],
        tails: NDArray[np.float32],
        frame_time: float,
    ) -> None:
        self._names = names
        self._root = names.index(_ROOT_CHAR_JOINT)
        self._drawn = drawn
        self._tails = tails
        self._hind = [
            (names.index(f"{side}_hip_b"), names.index(f"{side}_foot_b"))
            for side in ("left", "right")
            if f"{side}_hip_b" in names and f"{side}_foot_b" in names
        ]
        self._follow = 1 - math.exp(-frame_time / _TAIL_LAG_S)
        paws = max((drawn[foot, 1] for _, foot in self._hind), default=-math.inf)
        self._lift = np.where(tails[:, 1, 1] <= paws, _TAIL_LIFT, 0.0)
        self._tail_turn: dict[int, float] = {}
        self.contact_filter: ContactFilter | None = None
        self.learning = True
        self.unfiltered: list[NDArray[np.float32]] = []
        self.filtered: list[NDArray[np.float32]] = []

    def drawn_pose(self) -> NDArray[np.float32]:
        """Return every handle where the drawing has it, which is the pose the mesh rests in."""
        return np.vstack(
            (
                with_bone_handles(self._names, self._drawn),
                tail_handles(self._tails, np.zeros(2), None),
            )
        )

    def _hind_swing(self, joints: NDArray[np.float32]) -> float:
        """Return how far the hind legs have turned from their drawn direction, in radians."""
        turns: list[float] = []
        for hip, foot in self._hind:
            drawn = self._drawn[foot] - self._drawn[hip]
            posed = joints[foot] - joints[hip]
            turns.append(
                math.atan2(float(drawn[0] * posed[1] - drawn[1] * posed[0]), float(drawn @ posed))
            )
        return float(np.mean(turns)) if turns else 0.0

    def _with_handles(
        self, frame: int, joints: NDArray[np.float32], ground: float | None
    ) -> NDArray[np.float32]:
        """Add the bone handles and each tail, which follows the pelvis.

        The captured keypoints hold only the root of the tail, so a tail has
        no motion of its own to copy. It turns about its base a share of the
        way the hind legs swing, lagging behind them, and its tip keeps clear
        of the rearmost hind leg.
        """
        wanted = _TAIL_SWAY * self._hind_swing(joints)
        previous = self._tail_turn.get(frame - 1, wanted)
        turn = previous + self._follow * (wanted - previous)
        self._tail_turn[frame] = turn
        behind: NDArray[np.float64] | None = None
        if self._hind and len(self._tails):
            hips = np.mean([self._drawn[hip, 0] for hip, _ in self._hind])
            feet = np.array([joints[foot, 0] for _, foot in self._hind])
            away = np.where(self._tails[:, 0, 0] < hips, -1.0, 1.0)
            behind = np.where(away < 0, float(feet.min()), float(feet.max())).astype(np.float64)
        shift = joints[self._root] - self._drawn[self._root]
        return np.vstack(
            (
                with_bone_handles(self._names, joints),
                tail_handles(self._tails, shift, ground, turn=turn, behind=behind, lift=self._lift),
            )
        )

    def __call__(self, frame: int, positions: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.contact_filter is None:
            return self._with_handles(frame, positions, None)
        if self.learning:
            self.unfiltered.append(positions.copy())
        result = self.contact_filter(frame, positions)
        if not self.learning:
            self.filtered.append(result.copy())
        return self._with_handles(frame, result, self.contact_filter.lowest_ground())


def _pose_every_frame(drawing: AnimatedDrawing) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Pose the drawing on each frame and return the box its mesh stays inside."""
    low = np.full(2, np.inf, dtype=np.float32)
    high = np.full(2, -np.inf, dtype=np.float32)
    for frame in range(drawing.retargeter.bvh.frame_max_num):
        drawing.set_time(frame * drawing.retargeter.bvh.frame_time)
        drawing.update()
        low = np.minimum(low, drawing.vertices[:, :2].min(axis=0))
        high = np.maximum(high, drawing.vertices[:, :2].max(axis=0))
    return low, high


def posed_scene(mvc: Path) -> tuple[Config, Scene, _Handles, list[str]]:
    """Build the scene with every handle in place and the camera fitted to the whole motion.

    No OpenGL context is needed until a view is created, so the returned
    drawing can be posed frame by frame with set_time and update.
    """
    cfg = Config(str(mvc))
    scene = Scene(cfg.scene)
    drawing = next(child for child in scene.get_children() if isinstance(child, AnimatedDrawing))
    names = drawing.rig.root_joint.get_chain_joint_names()
    # The engine's own rig is already posed, so a second one supplies the drawn pose.
    drawn_rig = AnimatedDrawingRig(Config(str(mvc)).scene.animated_characters[0][0])
    contacts_path = mvc.with_name(_RENDER_CONTACTS)
    tails = np.zeros((0, 2, 2), dtype=np.float32)
    if contacts_path.is_file():
        with np.load(contacts_path, allow_pickle=False) as archive:
            tails = np.asarray(archive["tails"] / drawing.img_dim, dtype=np.float32)
    handles = _Handles(
        names,
        np.asarray(drawn_rig.get_joints_2D_positions(), dtype=np.float32),
        tails,
        drawing.retargeter.bvh.frame_time,
    )
    drawing.arap = ARAP(
        handles.drawn_pose(), drawing.mesh["triangles"], drawing.mesh["vertices"], w=_HANDLE_WEIGHT
    )
    drawing.control_point_filter = handles

    if contacts_path.is_file():
        with np.load(contacts_path, allow_pickle=False) as archive:
            contacts = np.asarray(archive["contacts"], dtype=np.bool_)
            mode = cast("BakeMode", str(archive["mode"]))
        handles.contact_filter = make_contact_filter(names, contacts, mode)
        # The filter learns the ground from a silent pass; later passes apply it.
        _pose_every_frame(drawing)
        handles.learning = False

    # The view is fitted to the finished motion, so it is created after a second silent pass.
    low, high = _pose_every_frame(drawing)
    handles.filtered.clear()
    drawing.set_time(0.0)
    drawing.update()
    cfg.view.camera_pos = list(frame_camera(low, high, cfg.view.window_dimensions))
    cfg.view.camera_fwd = list(_CAMERA_STRAIGHT_ON)
    return cfg, scene, handles, names


def _engine_child(mvc: Path, joints_path: Path) -> None:
    use_mesa = os.environ.get("PYOPENGL_PLATFORM") == "osmesa"
    if not use_mesa:
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize an OpenGL context")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    class RecordingController(VideoRenderController):
        """Collect the drawing's joints after each scene update."""

        def __init__(self, cfg: ControllerConfig, scene: Scene, render_view: View) -> None:
            super().__init__(cfg, scene, render_view)
            self.joint_frames: list[NDArray[np.float32]] = []

        def _update(self) -> None:
            super()._update()
            for child in self.scene.get_children():
                if isinstance(child, AnimatedDrawing):
                    joints = np.asarray(child.rig.get_joints_2D_positions(), dtype=np.float32)
                    self.joint_frames.append(joints)
                    break

    cfg, scene, handles, names = posed_scene(mvc)
    recorder = RecordingController(cfg.controller, scene, View.create_view(cfg.view))
    recorder.run()

    filtered = (
        np.stack(handles.filtered[: len(recorder.joint_frames)])
        if handles.filtered
        else np.stack(recorder.joint_frames)
    )
    np.savez_compressed(
        joints_path,
        joints=filtered,
        joints_unfiltered=np.stack(handles.unfiltered) if handles.unfiltered else filtered.copy(),
        names=np.asarray(names, dtype=np.str_),
        fps=np.float64(OUTPUT_FPS),
    )


def run_engine(mvc: Path, joints_path: Path, timeout_s: float = _RENDER_TIMEOUT_S) -> None:
    """Run the engine in a spawned process whose main thread owns OpenGL.

    GLFW must initialize and create its window on the main thread, and the
    local app renders from worker threads.
    """
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_engine_child, args=(mvc, joints_path))
    process.start()
    try:
        process.join(timeout_s)
        if process.is_alive():
            raise RuntimeError(f"Render timed out after {timeout_s:g} seconds")
        if process.exitcode != 0:
            raise RuntimeError(f"Render process exited with {process.exitcode}")
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()


def _rgba(frame: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != _FRAME_DIMENSIONS or array.shape[2] not in (
        _RGB_CHANNELS,
        _RGBA_CHANNELS,
    ):
        raise RuntimeError(f"Rendered frame has unsupported shape {array.shape}")
    if array.shape[2] == _RGBA_CHANNELS:
        return array
    alpha = np.full((*array.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate((array, alpha), axis=2)


def _read_frames(path: Path, frame_count: int) -> NDArray[np.uint8]:
    """Read the video back and expand merged GIF frames to the render timeline."""
    images: list[NDArray[np.uint8]] = []
    with _IMAGEIO.imopen(path, "r", plugin="pillow") as video:
        stored = video.properties().n_images or 0
        for index in range(stored):
            frame = _rgba(video.read(index=index))
            duration_ms = float(video.metadata(index=index).get("duration", _DEFAULT_FRAME_MS))
            repeats = max(1, round(duration_ms * OUTPUT_FPS / 1000))
            images.extend([frame] * repeats)
    if not images:
        raise RuntimeError(f"Rendered video contains no frames: {path}")
    # GIF durations are stored in 10 ms units, so rounding can miss one frame.
    if len(images) < frame_count:
        images.extend([images[-1]] * (frame_count - len(images)))
    return np.stack(images[:frame_count])


def _load_joints(
    path: Path, frame_count: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], tuple[str, ...] | None]:
    with np.load(path, allow_pickle=False) as archive:
        joints = _checked_joints(np.asarray(archive["joints"]), frame_count)
        if "joints_unfiltered" in archive.files:
            unfiltered = _checked_joints(np.asarray(archive["joints_unfiltered"]), frame_count)
        else:
            unfiltered = joints
        names = tuple(str(name) for name in archive["names"]) if "names" in archive.files else None
    return joints, unfiltered, names


def _checked_joints(joints: NDArray[np.generic], frame_count: int) -> NDArray[np.float32]:
    if (
        joints.ndim != _JOINT_DIMENSIONS
        or joints.shape[0] != frame_count
        or joints.shape[2] != _JOINT_XY
    ):
        raise RuntimeError(
            f"Rendered joints have shape {joints.shape}, expected ({frame_count}, J, 2)"
        )
    return joints.astype(np.float32, copy=False)


def render_character(
    char_dir: Path,
    motion: MotionFile,
    mode: str,
    out_dir: Path,
) -> RenderResult:
    """Animate a character with captured motion and return the drawn frames.

    The motion is resampled to the output rate before baking, so the engine
    renders exactly the frames this function returns. A drawing stored
    mirrored is flipped back after rendering, including its joint positions.
    """
    if mode not in ("treadmill", "travel"):
        raise ValueError(f"Unknown render mode {mode!r}")
    character = Path(char_dir).resolve()
    for name in (_CHARACTER_CFG, _TEXTURE_NAME, _MASK_NAME):
        if not (character / name).is_file():
            raise CharacterError(path=character, name=name)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    resampled = resample_motion(motion)
    window = (_WINDOW_HEIGHT, _WINDOW_HEIGHT)
    start_x = 0.0
    if mode == "travel":
        canvas = travel_canvas(character, resampled)
        resampled = _prefix(resampled, canvas.frames)
        window = (canvas.width, canvas.height)
        start_x = canvas.start_x
    np.savez_compressed(
        directory / _RENDER_CONTACTS,
        # Guessed legs stay as drawn, so there is no stance to stand the drawing on.
        contacts=resampled.contacts & (not _legs_guessed(character)),
        mode=np.str_(mode),
        tails=_tails(character),
    )
    bvh = _bake(resampled, mode, directory, character)
    configs: AnimationConfigs = write_configs(
        bvh,
        character,
        directory,
        use_mesa=os.environ.get("PYOPENGL_PLATFORM") == "osmesa",
        output=directory / _ANIMATION_GIF,
        window_dimensions=window,
        char_start_x=start_x,
    )
    _include_root_depth(configs.retarget)
    _prepare_motion_config(configs.motion, len(resampled.kp))

    joints_path = directory / _RENDER_JOINTS
    run_engine(configs.mvc, joints_path)
    frames = _read_frames(configs.gif, len(resampled.kp))
    joints, unfiltered, names = _load_joints(joints_path, len(resampled.kp))
    if _manifest(character).get("mirrored") is True:
        frames = np.ascontiguousarray(frames[:, :, ::-1, :])
        joints = joints.copy()
        joints[..., 0] = 1 - joints[..., 0]
        unfiltered = unfiltered.copy()
        unfiltered[..., 0] = 1 - unfiltered[..., 0]
    return RenderResult(
        frames_rgba=frames,
        fps=float(OUTPUT_FPS),
        joints_per_frame=joints,
        unfiltered_joints_per_frame=unfiltered,
        joint_names=names,
    )
