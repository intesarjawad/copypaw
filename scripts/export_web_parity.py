"""Write the browser port's parity fixture from the Python pipeline on the sample drawing.

The fixture holds what each ported step produced here: the drawing mask, the
snapped joints, the mesh and its draw order, the posed keypoints, the camera
and, for a few frames, the joints, mesh handles and mesh vertices the engine
was given and returned. web/src/compute/parity.test.ts
runs the TypeScript port on the same inputs and compares.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from copypaw.animate.bake import pose_for_drawing
from copypaw.animate.render import posed_scene, render_character, resample_motion
from copypaw.motion.motion_file import MotionFile
from copypaw.rig.char_cfg import write_character
from copypaw.rig.edits import load_drawing, rig_drawing
from copypaw.rig.snap import detect_facing
from copypaw.vendor.animated_drawings.model.animated_drawing import AnimatedDrawing

ROOT = Path(__file__).resolve().parents[1]
DRAWING = ROOT / "copypaw/selftest/dragon.png"
MOTION = ROOT / "tests/golden/dog_walk.motion.npz"
OUT = ROOT / "web/src/compute/parity.fixture.json"
# Frames before, during and after the first stride of the sample walk.
SAMPLE_FRAMES = (0, 12, 24, 36)


def mask_runs(mask: NDArray[np.uint8]) -> list[list[int]]:
    """Return each row's filled runs as flat [start, end, start, end, ...] column pairs."""
    rows: list[list[int]] = []
    for row in mask > 0:
        edges = np.flatnonzero(np.diff(np.concatenate(([False], row, [False])).astype(np.int8)))
        rows.append([int(edge) for edge in edges])
    return rows


def _floats(values: NDArray[np.floating]) -> list[Any]:
    return np.asarray(values, dtype=np.float64).round(6).tolist()


def _character_joints(config: Path) -> dict[str, list[float]]:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    return {entry["name"]: [float(v) for v in entry["loc"]] for entry in data["skeleton"]}


def build_fixture() -> dict[str, Any]:
    """Run the pipeline on the sample drawing and motion and record each step's result."""
    image = load_drawing(DRAWING)
    rig = rig_drawing(image)
    snapped = rig.snapped
    motion = resample_motion(MotionFile.load(MOTION))
    with TemporaryDirectory() as tmp:
        character = write_character(image, snapped, Path(tmp) / "drawing")
        out = Path(tmp) / "render"
        render_character(character.directory, MotionFile.load(MOTION), "treadmill", out)
        char_joints = _character_joints(character.char_cfg)
        cfg, scene, handles, names = posed_scene(out / "mvc.yaml")
        drawing = next(c for c in scene.get_children() if isinstance(c, AnimatedDrawing))
        with np.load(out / "render_contacts.npz", allow_pickle=False) as archive:
            contacts = np.asarray(archive["contacts"], dtype=np.bool_)
            tails = np.asarray(archive["tails"], dtype=np.float64) / drawing.img_dim
        unfiltered = [frame.copy() for frame in handles.unfiltered]
        # Replay the frames in order, as the engine does after its two silent passes.
        controls = [handles(frame, joints) for frame, joints in enumerate(unfiltered)]
        vertices: list[NDArray[np.float32]] = []
        for frame in range(len(unfiltered)):
            drawing.set_time(frame * drawing.retargeter.bvh.frame_time)
            drawing.update()
            vertices.append(drawing.vertices[:, :2].copy())
        posed = pose_for_drawing(motion, {name: (x, y) for name, (x, y) in char_joints.items()})
        frame_time_s = drawing.retargeter.bvh.frame_time
        mesh = {
            "vertices": _floats(drawing.mesh["vertices"]),
            "triangles": np.asarray(drawing.mesh["triangles"]).tolist(),
            "indices": drawing.indices.tolist(),
        }
        drawn = handles.drawn_pose()[: len(names)]

    return {
        "drawing": DRAWING.relative_to(ROOT).as_posix(),
        "motion": MOTION.relative_to(ROOT).as_posix(),
        "mask": {
            "width": rig.mask.shape[1],
            "height": rig.mask.shape[0],
            "runs": mask_runs(rig.mask),
        },
        "facing": detect_facing(rig.mask),
        "legs": snapped.legs,
        "legs_guessed": snapped.legs_guessed,
        "tails": [[list(base), list(tip)] for base, tip in snapped.tails],
        "joints": {name: [joint.x, joint.y] for name, joint in snapped.items()},
        "mesh_vertices": len(mesh["vertices"]),
        "mesh": mesh,
        "pose": {
            "kp": _floats(motion.kp),
            "contacts": motion.contacts.tolist(),
            "ground_line_px": _floats(motion.ground_line_px),
            "char_joints": char_joints,
            "frames": list(SAMPLE_FRAMES),
            "posed": [_floats(posed[frame]) for frame in SAMPLE_FRAMES],
        },
        "engine": {
            "names": names,
            "contacts": contacts.tolist(),
            "mode": "treadmill",
            "frame_time_s": frame_time_s,
            "drawn": _floats(drawn),
            "tails": _floats(tails),
            "unfiltered": [_floats(frame) for frame in unfiltered],
            "frames": list(SAMPLE_FRAMES),
            "controls": [_floats(controls[frame]) for frame in SAMPLE_FRAMES],
            "vertices": [_floats(vertices[frame]) for frame in SAMPLE_FRAMES],
            "window": list(cfg.view.window_dimensions),
            "camera": [float(value) for value in cfg.view.camera_pos],
        },
    }


def main() -> None:
    OUT.write_text(json.dumps(build_fixture(), separators=(",", ":")) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
