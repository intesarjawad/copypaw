"""Write the web page's light table from a real copypaw run on the self-test clip and drawing.

The page shows the clip under the paper and the rigged drawing on top, frame for frame,
with footfall.json as an exposure sheet beside them. Every frame and number it shows comes
from the files this script writes: frame images in web/public/light-table/ and the
record in web/src/lightTable/sheet.json. --clip and --drawing swap in another side-on walk.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from copypaw.animate.export import align_facing, drawing_facing, load_clip_frames
from copypaw.capture.video import OUTPUT_FPS, read_frames
from copypaw.evaluate.fidelity import contact_f1, onsets

ROOT = Path(__file__).resolve().parents[1]
SELFTEST = ROOT / "copypaw/selftest"
OUT = ROOT / "web/public/light-table"
SHEET = ROOT / "web/src/lightTable/sheet.json"
COLUMNS = 8
DOG_SCALE = 0.5
DRAWING_SCALE = 0.5
DOG_QUALITY = 84
DRAWING_QUALITY = 92
INK_THRESHOLD = 200
CROP_MARGIN_PX = 12
PAWS = ("near front", "far front", "near back", "far back")
# AP-10K keypoints: the four paws.
PAW_KEYPOINTS = (7, 10, 13, 16)
# A pixel belongs to the dog when it differs from the empty set by more than this many grey levels.
MATTE_THRESHOLD = 22
# The dog's keypoint box, widened by these fractions of its width and height, bounds its matte.
MATTE_REACH = np.array([0.12, 0.3])
# AP-10K bones, as pairs of keypoints: eyes to nose, nose to neck, the legs, spine and tail.
BONES = (
    (0, 2), (1, 2), (2, 3), (3, 5), (3, 8), (5, 6), (6, 7), (8, 9), (9, 10),
    (3, 4), (4, 11), (4, 14), (11, 12), (12, 13), (14, 15), (15, 16), (5, 11), (8, 14),
)  # fmt: skip
TORSO = (5, 8, 14, 11)
# The front elbows and the hind knees: with the paws they bound the gap between the legs.
LEG_TOPS = (6, 9, 12, 15)
# The legs' bones, shoulder or hip to paw: only along them can the dog cross the floor.
LEG_BONES = ((5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13), (14, 15), (15, 16))
NOSE, TAIL_ROOT = 2, 4
SHOULDERS = (5, 8)
BONE_WIDTH = 0.035
# How far either side of a leg's bone the dog may reach over the floor, in bone widths.
LEG_REACH = 0.8
# How far past the nose the dog may reach, as a fraction of its keypoints' width.
NOSE_REACH = 0.015
# How far from the skeleton, in bone widths, a pixel that matches the empty set may still be dog.
STILL_REACH = 2
GRABCUT_ROUNDS = 5
MATTE_SOFTEN_PX = 5
MATTE_QUALITY = 90
# Half the matte's soft edge: brighter than this is dog for the box the page crops to.
MATTE_COVER = 128
# Paws are the drawing's ink within this fraction of its height above its lowest row.
PAW_BAND = 0.12
# Frames either side averaged into a paw track, so a swinging leg does not shake the registration.
TRACK_SMOOTHING = 3
# is_locomotion counts a clip as a walk only when every paw lands at least this often.
LANDINGS_NEEDED = 2
_WHITE = 255


def run_copypaw(work: Path, clip_path: Path, drawing_path: Path) -> None:
    """Capture the clip, rig the drawing and animate one with the other."""
    copypaw = str(Path(sys.executable).with_name("copypaw"))
    clip = str(clip_path)
    motion, dragon = work / "motion", work / "drawing"
    for command in (
        ["capture", clip, "-o", str(motion)],
        ["rig", str(drawing_path), "-o", str(dragon)],
        [
            "animate",
            str(dragon),
            str(motion / "motion.npz"),
            "--clip",
            clip,
            "-o",
            str(work / "result"),
        ],
    ):
        subprocess.run([copypaw, *command], check=True, stdout=subprocess.DEVNULL)  # noqa: S603


def grey(frames: NDArray[np.uint8]) -> NDArray[np.uint8]:
    weights = np.array([0.299, 0.587, 0.114])
    return np.clip(np.rint(frames[..., :3] @ weights), 0, _WHITE).astype(np.uint8)


def ink_box(frames: NDArray[np.uint8], margin: int = CROP_MARGIN_PX) -> tuple[int, int, int, int]:
    """Return the left, top, right and bottom edges of the ink in any frame, widened by a margin."""
    inked = frames < INK_THRESHOLD
    rows = np.flatnonzero(inked.any(axis=(0, 2)))
    cols = np.flatnonzero(inked.any(axis=(0, 1)))
    if rows.size == 0:
        raise ValueError("the drawing frames hold no ink")
    height, width = frames.shape[1:3]
    return (
        max(0, int(cols[0]) - margin),
        max(0, int(rows[0]) - margin),
        min(width, int(cols[-1]) + 1 + margin),
        min(height, int(rows[-1]) + 1 + margin),
    )


def shrink(frames: NDArray[np.uint8], scale: float) -> NDArray[np.uint8]:
    height, width = frames.shape[1:3]
    size = (round(width * scale), round(height * scale))
    return np.stack([_resize(frame, size) for frame in frames])


def _resize(frame: NDArray[np.uint8], size: tuple[int, int]) -> NDArray[np.uint8]:
    return np.asarray(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def atlas(frames: NDArray[np.uint8], columns: int = COLUMNS) -> NDArray[np.uint8]:
    """Lay grey frames out row by row, left to right, padding the last row with white."""
    count, height, width = frames.shape
    rows = math.ceil(count / columns)
    sheet = np.full((rows * height, columns * width), _WHITE, dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return sheet


def at_frames(values: NDArray[Any], clip_fps: float, count: int) -> NDArray[Any]:
    """Pick the clip sample nearest in time to each output frame."""
    times = np.arange(len(values)) / clip_fps
    targets = np.arange(count) / OUTPUT_FPS
    return values[np.abs(times[None, :] - targets[:, None]).argmin(axis=1)]


def facing_keypoints(
    keypoints: NDArray[np.float32], facing: str, width: int
) -> NDArray[np.float32]:
    """motion.npz stores right-facing keypoints, so a dog walking left is mirrored back."""
    turned = keypoints.copy()
    if facing == "left":
        turned[..., 0] = width - turned[..., 0]
    return turned


def smooth(track: NDArray[np.float64], reach: int = TRACK_SMOOTHING) -> NDArray[np.float64]:
    """Average each value with `reach` frames either side, holding the ends."""
    padded = np.pad(track, reach, mode="edge")
    return np.convolve(padded, np.ones(2 * reach + 1) / (2 * reach + 1), mode="valid")


def floor_band_top(empty: NDArray[np.floating], floor_y: float) -> int:
    """Return the top row of the dark floor band that runs up from the floor line to the wall."""
    rows = np.median(empty, axis=1)
    floor_row = min(int(floor_y), len(rows) - 1)
    middle = (rows[:floor_row].max() + rows[floor_row]) / 2
    top = floor_row
    while top > 0 and rows[top - 1] < middle:
        top -= 1
    return top


def still_wall(
    points: NDArray[np.float32],
    skeleton: NDArray[np.uint8],
    moving: NDArray[np.bool_],
    bone: int,
    *,
    facing_left: bool,
) -> NDArray[np.bool_]:
    """Return the wall GrabCut may take for pale fur: unchanged and far from the skeleton.

    Only below the torso or ahead of the shoulders, as under the chin. Above and behind, the
    tail is spared: it is tracked only at its root and can match the wall too.
    """
    away = cv2.distanceTransform(1 - skeleton, cv2.DIST_L2, 5) > bone * STILL_REACH
    shoulders = int(points[list(SHOULDERS), 0].mean())
    behind = np.zeros(skeleton.shape, bool)
    if facing_left:
        behind[:, shoulders:] = True
    else:
        behind[:, :shoulders] = True
    behind[int(points[list(TORSO), 1].max()) :] = False
    return away & ~moving & ~behind


def matte(
    frames: NDArray[np.uint8], keypoints: NDArray[np.float32], floor_y: float
) -> NDArray[np.uint8]:
    """Cut the dog out of each frame as one filled, soft-edged silhouette.

    The dog is pale on a pale wall, so a difference alone leaves holes. GrabCut is seeded from
    the tracked skeleton: its bones and torso are surely dog, what differs from the median
    frame near them is probably dog, and the gap between the legs, everything below the floor
    line, the floor except along the legs, what lies ahead of the nose and everything outside the
    keypoint box is not. The floor is dark like the dog's shadow and muzzle, and a person stands
    ahead of the dog at the end of the clip, so neither is left for GrabCut to judge. Wall that
    GrabCut still takes for pale fur is cut from its result, and of what is left only the largest
    piece is kept.
    """
    empty = np.median(frames, axis=0)
    floor_top = floor_band_top(empty, floor_y)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mattes = np.zeros_like(frames)
    for index, (frame, points) in enumerate(zip(frames, keypoints, strict=True)):
        seeds = np.full(frame.shape, cv2.GC_BGD, np.uint8)
        low, high = points.min(axis=0), points.max(axis=0)
        reach = (high - low) * MATTE_REACH
        x0, y0 = (int(v) for v in np.maximum(low - reach, 0))
        x1, y1 = (int(v) for v in high + reach)
        seeds[y0:y1, x0:x1] = cv2.GC_PR_BGD
        moving = np.abs(frame.astype(np.int16) - empty) > MATTE_THRESHOLD
        seeds[moving & (seeds == cv2.GC_PR_BGD)] = cv2.GC_PR_FGD
        legs = points[[*LEG_TOPS, *PAW_KEYPOINTS]].reshape(4, 2, 2).mean(axis=1)
        cv2.fillPoly(seeds, [legs[[0, 1, 3, 2]].astype(np.int32)], int(cv2.GC_BGD))
        bone = max(4, int((high - low)[0] * BONE_WIDTH))
        along_legs = np.zeros(frame.shape, np.uint8)
        for start, end in LEG_BONES:
            first, second = (tuple(int(v) for v in points[i]) for i in (start, end))
            cv2.line(along_legs, first, second, 1, 2 * int(bone * LEG_REACH) + 1)
        seeds[floor_top:][along_legs[floor_top:] == 0] = cv2.GC_BGD
        nose = int(points[NOSE, 0])
        ahead = int((high - low)[0] * NOSE_REACH)
        facing_left = points[NOSE, 0] < points[TAIL_ROOT, 0]
        if facing_left:
            seeds[:, : max(nose - ahead, 0)] = cv2.GC_BGD
        else:
            seeds[:, nose + ahead :] = cv2.GC_BGD
        skeleton = np.zeros(frame.shape, np.uint8)
        for start, end in BONES:
            first, second = (tuple(int(v) for v in points[i]) for i in (start, end))
            cv2.line(skeleton, first, second, 1, bone)
        torso = cast("NDArray[np.int32]", cv2.convexHull(points[list(TORSO)].astype(np.int32)))
        cv2.fillConvexPoly(skeleton, torso, 1)
        away = still_wall(points, skeleton, moving, bone, facing_left=facing_left)
        seeds[skeleton == 1] = cv2.GC_FGD
        seeds[int(floor_y) + 3 :] = cv2.GC_BGD
        background, foreground = np.zeros((1, 65)), np.zeros((1, 65))
        colour = cast("NDArray[np.uint8]", cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        cv2.grabCut(
            colour,
            seeds,
            (0, 0, 1, 1),
            background,
            foreground,
            GRABCUT_ROUNDS,
            cv2.GC_INIT_WITH_MASK,
        )
        dog = np.isin(seeds, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * _WHITE
        dog[away] = 0
        # The largest piece is picked between opening and closing, so a thread the closing would
        # weld back cannot carry a stray piece along.
        dog = cv2.morphologyEx(dog, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dog)
        if count > 1:
            dog = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(
                np.uint8
            ) * _WHITE
        dog = cv2.morphologyEx(dog, cv2.MORPH_CLOSE, kernel)
        mattes[index] = cv2.GaussianBlur(dog, (0, 0), MATTE_SOFTEN_PX / 2)
    return mattes


def drawing_paws(drawing: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Return the mean x of the ink in each frame's paw band, as a fraction of the frame width."""
    inked = drawing < INK_THRESHOLD
    rows = np.flatnonzero(inked.any(axis=(0, 2)))
    band = slice(int(rows[-1] - PAW_BAND * (rows[-1] - rows[0])), int(rows[-1]) + 1)
    columns = np.arange(drawing.shape[2], dtype=np.float64)
    paws: list[float] = []
    for frame in inked:
        weights: NDArray[np.int64] = (
            frame[band].sum(axis=0) if frame[band].any() else frame.sum(axis=0)
        )
        paws.append(float((columns * weights).sum() / max(weights.sum(), 1)) / drawing.shape[2])
    return smooth(np.array(paws))


def dog_box(mattes: NDArray[np.uint8]) -> list[float]:
    """Return the box the dog covers in any frame, as fractions: left, top, right, bottom."""
    covered = (mattes > MATTE_COVER).any(axis=0)
    rows, columns = np.flatnonzero(covered.any(axis=1)), np.flatnonzero(covered.any(axis=0))
    height, width = covered.shape
    return [
        round(float(columns[0]) / width, 4),
        round(float(rows[0]) / height, 4),
        round(float(columns[-1] + 1) / width, 4),
        round(float(rows[-1] + 1) / height, 4),
    ]


def exposure_sheet(  # noqa: PLR0917
    footfall: dict[str, list[str]],
    fidelity: dict[str, Any],
    motion: dict[str, Any],
    dog_size: tuple[int, int],
    drawing: NDArray[np.uint8],
    box: tuple[int, int, int, int],
    dog_cover: list[float],
) -> dict[str, object]:
    """Shape the run into the record the page reads, in fractions of each picture.

    The drawing keeps the scale it has in side_by_side.mp4, where both pictures are one height.
    """
    count = len(drawing)
    width, height = dog_size
    if any(len(track) < count for track in (*footfall["pet"], *footfall["drawing"])):
        raise ValueError("footfall.json holds fewer frames than the run rendered")
    pet_contacts = np.array(
        [[bit == "1" for bit in track[:count]] for track in footfall["pet"]], dtype=np.bool_
    ).T
    rendered_contacts = np.array(
        [[bit == "1" for bit in track[:count]] for track in footfall["drawing"]], dtype=np.bool_
    ).T
    slope, intercept = (float(value) for value in motion["ground_line_px"])
    left, top, right, bottom = box
    # Per frame, the lowest ink is the planted paw: the page stands it exactly on the floor line.
    inked = (drawing[:, top:bottom, left:right] < INK_THRESHOLD).any(axis=2)
    lows = [
        (int(np.flatnonzero(rows)[-1]) + 1) / (bottom - top) if rows.any() else 1.0
        for rows in inked
    ]
    keypoints = at_frames(motion["kp"], float(motion["fps"]), count)
    paw_x = facing_keypoints(keypoints, str(motion["facing"]), width)[:, PAW_KEYPOINTS, 0]
    dog_paws = smooth(paw_x.mean(axis=1).astype(np.float64))
    # Front to back paw, averaged over the run: the tool scales any drawing to this stride.
    span = float((paw_x.max(axis=1) - paw_x.min(axis=1)).mean()) / width
    return {
        "scene": str(motion.get("scene", "dog_walk")),
        "character": str(motion.get("character", "dragon")),
        "fps": OUTPUT_FPS,
        "frames": count,
        "paws": list(PAWS),
        "footfalls": {
            "dog": [track[:count] for track in footfall["pet"]],
            "drawing": [track[:count] for track in footfall["drawing"]],
        },
        "contactMatch": contact_f1(pet_contacts, rendered_contacts),
        "contactMatchStage": "corrected render footfalls",
        "uncorrectedContactF1": fidelity["contact_f1"],
        "locomotion": fidelity["locomotion"],
        "landings": [len(frames) for frames in onsets(pet_contacts)],
        "landingsNeeded": LANDINGS_NEEDED,
        "columns": COLUMNS,
        "dog": {
            "cell": [round(width * DOG_SCALE), round(height * DOG_SCALE)],
            "paws": [round(float(x) / width, 4) for x in dog_paws],
            "span": round(span, 4),
            "box": dog_cover,
            "floor": round((slope * width / 2 + intercept) / height, 4),
        },
        "drawing": {
            "cell": [round((right - left) * DRAWING_SCALE), round((bottom - top) * DRAWING_SCALE)],
            "height": round((bottom - top) / drawing.shape[1], 4),
            "lows": [round(low, 4) for low in lows],
            "paws": [round(float(x), 4) for x in drawing_paws(drawing[:, top:bottom, left:right])],
        },
    }


def build(work: Path, out: Path, sheet_path: Path, clip: Path, name: str) -> dict[str, object]:
    dog = load_clip_frames(clip)
    drawing = grey(read_frames(work / "result/animation.mp4", target_fps=OUTPUT_FPS).rgb)
    with np.load(work / "motion/motion.npz") as data:
        motion: dict[str, Any] = {
            key: data[key] for key in ("kp", "fps", "ground_line_px", "facing", "contacts")
        }
    motion["scene"] = clip.stem
    motion["character"] = name
    # As side_by_side.mp4 does, the drawing is turned to walk the way the dog walks.
    facing = "left" if str(motion["facing"]) == "left" else "right"
    drawing = align_facing(drawing, facing, drawing_facing(work / "drawing"), False)
    count = min(len(dog), len(drawing))
    dog_grey, drawing = grey(dog[:count]), drawing[:count]
    height, width = dog_grey.shape[1:3]
    box = ink_box(drawing)
    slope, intercept = (float(value) for value in motion["ground_line_px"])
    keypoints = facing_keypoints(
        at_frames(motion["kp"], float(motion["fps"]), count), facing, width
    )
    mattes = matte(dog_grey, keypoints, slope * width / 2 + intercept)
    sheet = exposure_sheet(
        json.loads((work / "result/footfall.json").read_text(encoding="utf-8")),
        json.loads((work / "result/fidelity.json").read_text(encoding="utf-8")),
        motion,
        (width, height),
        drawing,
        box,
        dog_box(mattes),
    )
    left, top, right, bottom = box
    dog_cells = shrink(dog_grey, DOG_SCALE)
    drawing_cells = shrink(drawing[:, top:bottom, left:right], DRAWING_SCALE)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "dog.jpg"), atlas(dog_cells), [cv2.IMWRITE_JPEG_QUALITY, DOG_QUALITY])
    matte_atlas = atlas(255 - shrink(mattes, DOG_SCALE))
    cv2.imwrite(str(out / "matte.jpg"), matte_atlas, [cv2.IMWRITE_JPEG_QUALITY, MATTE_QUALITY])
    cv2.imwrite(
        str(out / "drawing.jpg"), atlas(drawing_cells), [cv2.IMWRITE_JPEG_QUALITY, DRAWING_QUALITY]
    )
    sheet_path.write_text(json.dumps(sheet, indent=1) + "\n", encoding="utf-8")
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=SELFTEST / "dog_walk.mp4", help="Side-on walk")
    parser.add_argument("--drawing", type=Path, default=SELFTEST / "dragon.png", help="The drawing")
    parser.add_argument("--name", help="What the drawing shows (default: its file name)")
    parser.add_argument("--run", type=Path, help="Reuse a run folder holding motion/ and result/")
    parser.add_argument("--out", type=Path, default=OUT, help="Folder for the frame images")
    parser.add_argument("--sheet", type=Path, default=SHEET, help="Where to write the record")
    args = parser.parse_args()
    run = cast("Path | None", args.run)
    with tempfile.TemporaryDirectory() as scratch:
        work = run or Path(scratch)
        clip, drawing = cast("Path", args.clip), cast("Path", args.drawing)
        if run is None:
            run_copypaw(work, clip, drawing)
        name = cast("str | None", args.name) or drawing.stem
        sheet = build(work, cast("Path", args.out), cast("Path", args.sheet), clip, name)
    print(f"Wrote {sheet['frames']} frames of the light table to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
