import type { Animation } from "../compute/animate";
import { CHARACTER_DEPTH, viewProjection } from "../compute/camera";
import type { Rect, Sheet } from "./sheet";
import { wrapFrame } from "./sheet";

// The same band the build script reads the drawing's paws from: the lowest eighth of its height.
const PAW_BAND = 0.12;
// Where the red floor line runs, and the headroom left above the drawing, as parts of the window.
const FLOOR_AT = 0.86;
const HEADROOM = 0.04;
// Paper left around the action when the sheet is cropped to it, as a part of the action's size.
const CROP_MARGIN = 0.08;
// The peg bar covers the sheet's top edge, so the crop leaves it this much more room on top.
const PEG_ROOM = 0.14;

/** Where the animated drawing stands in its window, in pixels. */
export interface Footing {
  /** Highest and lowest point over every frame. */
  readonly top: number;
  readonly floor: number;
  /** Per frame, the middle of the points in the paw band. */
  readonly paws: readonly number[];
  /** Per frame, the lowest point: the planted paw. */
  readonly lows: readonly number[];
  /** Front to back paw, averaged over the frames. */
  readonly span: number;
  /** How far the drawing reaches left and right of its paws in any frame. */
  readonly reach: readonly [left: number, right: number];
}

/** Project every vertex of every frame through the player's camera and find the paws. */
export function footing(animation: Animation): Footing {
  const [width, height] = animation.window;
  const m = viewProjection(animation.camera, animation.window, animation.mirrored);
  const vertices = animation.uv.length / 2;
  const xs = new Float32Array(vertices * animation.frameCount);
  const ys = new Float32Array(vertices * animation.frameCount);
  for (let i = 0; i < xs.length; i++) {
    const x = animation.positions[2 * i] ?? 0;
    const y = animation.positions[2 * i + 1] ?? 0;
    const column = (row: number): number =>
      (m[row] ?? 0) * x +
      (m[4 + row] ?? 0) * y +
      (m[8 + row] ?? 0) * CHARACTER_DEPTH +
      (m[12 + row] ?? 0);
    const w = column(3);
    xs[i] = ((column(0) / w + 1) / 2) * width;
    ys[i] = ((1 - column(1) / w) / 2) * height;
  }
  const top = Math.min(...ys);
  const floor = Math.max(...ys);
  const band = floor - PAW_BAND * (floor - top);
  let spans = 0;
  let left = 0;
  let right = 0;
  const lows: number[] = [];
  const paws = Array.from({ length: animation.frameCount }, (_, frame) => {
    let sum = 0;
    let count = 0;
    let back = Number.POSITIVE_INFINITY;
    let front = Number.NEGATIVE_INFINITY;
    let lowest = Number.POSITIVE_INFINITY;
    let highest = Number.NEGATIVE_INFINITY;
    let low = Number.NEGATIVE_INFINITY;
    for (let i = frame * vertices; i < (frame + 1) * vertices; i++) {
      low = Math.max(low, ys[i] ?? 0);
      const x = xs[i] ?? 0;
      lowest = Math.min(lowest, x);
      highest = Math.max(highest, x);
      if ((ys[i] ?? 0) >= band) {
        sum += x;
        count += 1;
        back = Math.min(back, x);
        front = Math.max(front, x);
      }
    }
    const middle = count === 0 ? width / 2 : sum / count;
    spans += count === 0 ? 0 : front - back;
    left = Math.max(left, middle - lowest);
    right = Math.max(right, highest - middle);
    lows.push(low);
    return middle;
  });
  return { top, floor, paws, lows, span: spans / animation.frameCount, reach: [left, right] };
}

/** A 2D transform from the animation's window into the stage, as CSS `matrix(a, 0, 0, d, e, f)`. */
export interface Placing {
  readonly a: number;
  readonly d: number;
  readonly e: number;
  readonly f: number;
}

/** The dog's clip under an animated drawing, both in the animation window's pixels. */
export interface Tracing {
  readonly clip: Rect;
  readonly floorY: number;
  /** The part of the window the sheet shows: the action, with a margin of paper. */
  readonly view: Rect;
  /** The clip frame shot at the same moment as a frame of the drawing. */
  clipFrame(frame: number): number;
  /** Where the drawing goes so its paws stand on the dog's. */
  drawingAt(frame: number): Placing;
}

/**
 * Lay the drawing over the clip it was traced from, as the light table does: the drawing is
 * scaled so its stride matches the dog's, turned to face the way the dog walks, stood on the
 * dog's floor line with paws over paws, and the pair is sized to fit the window.
 */
export function traceOver(animation: Animation, sheet: Sheet): Tracing {
  const [width, height] = animation.window;
  const drawn = footing(animation);
  const { cellWidth, cellHeight } = sheet.clip.atlas;
  const dogPaws = sheet.clip.paws;
  // The engine turns a drawing to walk right and mirrors it back if it faced left.
  const dogWalksLeft = (dogPaws.at(-1) ?? 0) < (dogPaws[0] ?? 0);
  const turned = dogWalksLeft !== animation.mirrored;
  const [left, right] = turned ? [drawn.reach[1], drawn.reach[0]] : drawn.reach;
  // The drawing's scale for each pixel of clip width.
  const perClip = drawn.span > 0 ? sheet.clip.span / drawn.span : 0;
  const floorY = FLOOR_AT * height;
  const nearest = Math.min(...dogPaws);
  const furthest = Math.max(...dogPaws);
  const clipWidth = Math.min(
    width,
    floorY / ((sheet.clip.floor * cellHeight) / cellWidth),
    perClip > 0 ? (floorY - HEADROOM * height) / (perClip * (drawn.floor - drawn.top)) : width,
    width / Math.max(1 - 2 * nearest + 2 * perClip * left, 2 * furthest - 1 + 2 * perClip * right),
  );
  const clipHeight = (clipWidth * cellHeight) / cellWidth;
  const clip = {
    x: (width - clipWidth) / 2,
    y: floorY - sheet.clip.floor * clipHeight,
    width: clipWidth,
    height: clipHeight,
  };
  const scale = perClip * clipWidth;
  const clipFrame = (frame: number): number =>
    wrapFrame(frame * animation.frameTimeS * sheet.fps, sheet.frames);
  function drawingAt(frame: number): Placing {
    const dogX = clip.x + (dogPaws[clipFrame(frame)] ?? 0.5) * clipWidth;
    const paws = drawn.paws[frame] ?? width / 2;
    return {
      a: turned ? -scale : scale,
      d: scale,
      e: turned ? dogX + scale * paws : dogX - scale * paws,
      // The frame's lowest point is its planted paw, so it stands exactly on the line.
      f: floorY - scale * (drawn.lows[frame] ?? drawn.floor),
    };
  }
  const dog = dogCover(sheet, clip, floorY);
  let [x0, y0, x1] = [dog.x, dog.y, dog.x + dog.width];
  for (let frame = 0; frame < animation.frameCount; frame++) {
    const { a, d, e, f } = drawingAt(frame);
    const middle = drawn.paws[frame] ?? width / 2;
    const ends = [a * (middle - drawn.reach[0]) + e, a * (middle + drawn.reach[1]) + e];
    x0 = Math.min(x0, ...ends);
    x1 = Math.max(x1, ...ends);
    y0 = Math.min(y0, d * drawn.top + f);
  }
  return { clip, floorY, view: margined(x0, y0, x1, floorY), clipFrame, drawingAt };
}

/** Where the dog walks in the window, from its cover box down to the floor line. */
function dogCover(sheet: Sheet, clip: Rect, floorY: number): Rect {
  const { box } = sheet.clip;
  const y = clip.y + box.y * clip.height;
  return { x: clip.x + box.x * clip.width, y, width: box.width * clip.width, height: floorY - y };
}

function margined(x0: number, y0: number, x1: number, floorY: number): Rect {
  const pad = CROP_MARGIN * Math.max(x1 - x0, floorY - y0);
  const top = pad + PEG_ROOM * (floorY - y0);
  return { x: x0 - pad, y: y0 - top, width: x1 - x0 + 2 * pad, height: floorY - y0 + pad + top };
}

/** The dog's clip alone on a sheet, cropped to its walk, before any drawing is animated. */
export function clipAlone(sheet: Sheet): Tracing {
  const { cellWidth, cellHeight } = sheet.clip.atlas;
  const clip = { x: 0, y: 0, width: cellWidth, height: cellHeight };
  const floorY = sheet.clip.floor * cellHeight;
  const dog = dogCover(sheet, clip, floorY);
  return {
    clip,
    floorY,
    view: margined(dog.x, dog.y, dog.x + dog.width, floorY),
    clipFrame: (frame) => wrapFrame(frame, sheet.frames),
    drawingAt: () => ({ a: 1, d: 1, e: 0, f: 0 }),
  };
}
