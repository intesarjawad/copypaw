import type { Rig } from "../backend";
import { jointsFrom, type JointId, type Joints, type Point } from "../joints";
import {
  components,
  morphSquare,
  outline,
  resizeNearest,
  roundHalfEven,
  segment,
  type Mask,
  type Pixels,
} from "./mask";

export type Facing = "left" | "right";

/** Where a tail leaves the body and where it ends, in source pixels. */
export type Tail = readonly [base: Point, tip: Point];

export interface Snapped {
  readonly joints: Joints;
  /** 2 for a side view with one visible leg per pair, so the far chain can copy the near one. */
  readonly legs: 2 | 4;
  /** Tails that reach down among the legs; the rig has no joints for them. */
  readonly tails: readonly Tail[];
  /** The leg joints are the template's, not found on the drawing, so the legs stay as drawn. */
  readonly legsGuessed: boolean;
}

const TEMPLATE_WIDTH = 564;
const TEMPLATE_HEIGHT = 352;
// Pixel location of each joint in the upstream character's 564 x 352 box.
const TEMPLATE: Readonly<Record<JointId, readonly [number, number]>> = {
  root: [238, 264],
  hip: [238, 264],
  torso: [453, 181],
  neck: [455, 121],
  right_hip_b: [139, 266],
  right_knee_b: [138, 290],
  right_foot_b: [132, 324],
  left_hip_b: [82, 249],
  left_knee_b: [65, 279],
  left_foot_b: [47, 308],
  right_hip_f: [442, 245],
  right_knee_f: [455, 284],
  right_foot_f: [458, 320],
  left_hip_f: [364, 266],
  left_knee_f: [364, 305],
  left_foot_f: [360, 336],
};

const LEG_REGION_TOP = 0.6;
const OPEN_KERNEL_PX = 5;
const MIN_LEG_FRACTION = 0.005;
const EDGE_ROWS = 3;
const KNEE_FRACTION = 0.55;
const NOSE_REGION = 0.5;
const NECK_FRACTION = 0.7;
// A head stands above the back, so the top quarter of the drawing leans to its side.
const HEAD_REGION = 0.25;
// Share of the drawing's width the top quarter must lean before it counts as a head.
const FACING_DEADBAND = 0.02;
// An appendage has joined the body where its run of mask pixels grows half again wider.
const BODY_RUN_FACTOR = 1.5;
// Drawn legs stand within a few degrees of vertical; the sample's tail leans 14 degrees back.
const TAIL_LEAN_DEG = 8;
// Leaning back this far, an appendage under the body is not standing on it either.
const STEEP_LEAN_DEG = 20;
// A tail narrows towards the ground; a leg keeps its width or ends in a wider foot.
const TAIL_TAPER = 0.6;

/**
 * An appendage below the body: its left edge, joint positions and outline measures.
 *
 * The hip is where the appendage widens into the body, so the leg swings from
 * the body and not from the row where the search began. taper is its median
 * width over the bottom quarter divided by that over the top quarter.
 */
interface Leg {
  readonly left: number;
  readonly hip: Point;
  readonly knee: Point;
  readonly foot: Point;
  readonly taper: number;
}

/** Scale the template into an xyxy drawing box, mirrored for a left-facing drawing. */
export function fitTemplate(
  left: number,
  top: number,
  right: number,
  bottom: number,
  facing: Facing = "right",
): Joints {
  if (![left, top, right, bottom].every(Number.isFinite) || right <= left || bottom <= top) {
    throw new Error("The box must be finite xyxy coordinates with positive area");
  }
  return jointsFrom((id) => {
    const [x, y] = TEMPLATE[id];
    return {
      x: left + ((facing === "right" ? x : TEMPLATE_WIDTH - x) / TEMPLATE_WIDTH) * (right - left),
      y: top + (y / TEMPLATE_HEIGHT) * (bottom - top),
    };
  });
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const upper = sorted[middle] ?? Number.NaN;
  return sorted.length % 2 === 1 ? upper : ((sorted[middle - 1] ?? Number.NaN) + upper) / 2;
}

function filled(mask: Mask, x: number, y: number): boolean {
  return (mask.data[y * mask.width + x] ?? 0) > 0;
}

/** Follow an appendage up through the mask until its run widens into the body. */
function attachment(mask: Mask, x: number, top: number, width: number): Point {
  let centre = x;
  for (let row = top - 1; row >= 0; row--) {
    const column = roundHalfEven(centre);
    if (!filled(mask, column, row)) {
      return { x: centre, y: row + 1 };
    }
    let left = column;
    while (left > 0 && filled(mask, left - 1, row)) {
      left--;
    }
    let right = column + 1;
    while (right < mask.width && filled(mask, right, row)) {
      right++;
    }
    if (right - left > BODY_RUN_FACTOR * width) {
      return { x: centre, y: row };
    }
    centre = (left + right - 1) / 2;
  }
  return { x: centre, y: 0 };
}

function legGeometry(rows: readonly number[], columns: readonly number[], mask: Mask): Leg {
  const top = Math.min(...rows);
  const bottom = Math.max(...rows);
  const topX = median(columns.filter((_, index) => (rows[index] ?? 0) < top + EDGE_ROWS));
  const bottomX = median(columns.filter((_, index) => (rows[index] ?? 0) > bottom - EDGE_ROWS));
  const widths = new Array<number>(bottom - top + 1).fill(0);
  for (const row of rows) {
    widths[row - top] = (widths[row - top] ?? 0) + 1;
  }
  const quarter = Math.max(1, Math.floor(widths.length / 4));
  const taper = median(widths.slice(-quarter)) / Math.max(1, median(widths.slice(0, quarter)));
  const hip = attachment(mask, topX, top, median(widths));
  const foot = { x: bottomX, y: bottom };
  return {
    left: Math.min(...columns),
    hip,
    knee: {
      x: hip.x + KNEE_FRACTION * (foot.x - hip.x),
      y: hip.y + KNEE_FRACTION * (foot.y - hip.y),
    },
    foot,
    taper,
  };
}

function legs(mask: Mask): Leg[] {
  const { width, height } = mask;
  const legTop = Math.floor(height * LEG_REGION_TOP);
  const region = new Uint8Array(width * height);
  for (let index = legTop * width; index < region.length; index++) {
    region[index] = (mask.data[index] ?? 0) > 0 ? 255 : 0;
  }
  const opened = morphSquare(
    morphSquare({ width, height, data: region }, OPEN_KERNEL_PX, false),
    OPEN_KERNEL_PX,
    true,
  );
  const { labels, areas } = components(opened);
  const rows = areas.map((): number[] => []);
  const columns = areas.map((): number[] => []);
  labels.forEach((label, index) => {
    if (label > 0) {
      rows[label - 1]?.push(Math.floor(index / width));
      columns[label - 1]?.push(index % width);
    }
  });
  const minimumArea = width * height * MIN_LEG_FRACTION;
  return areas
    .flatMap((area, index) =>
      area > minimumArea ? [legGeometry(rows[index] ?? [], columns[index] ?? [], mask)] : [],
    )
    .sort((a, b) => a.left - b.left);
}

/** Degrees from vertical that the appendage leans back from its attachment to its foot. */
function rearLeanDeg(leg: Leg, facing: Facing): number {
  const rear = facing === "right" ? -1 : 1;
  const radians = Math.atan2(rear * (leg.foot.x - leg.hip.x), Math.max(leg.foot.y - leg.hip.y, 1));
  return (radians * 180) / Math.PI;
}

/**
 * Split the tails from more than four appendages, if exactly four legs remain.
 *
 * A tail is the rearmost appendage, attached behind every other one, that
 * leans back or narrows to the ground, or any appendage in the rear half that
 * leans steeply back instead of standing under the body. Otherwise the
 * appendages are returned unchanged and the caller keeps the template.
 */
function withoutTails(found: readonly Leg[], facing: Facing): [Leg[], Tail[]] {
  if (found.length <= 4) {
    return [[...found], []];
  }
  const rear = facing === "right" ? -1 : 1;
  const attachments = found.map((leg) => leg.hip.x);
  const middle = (Math.min(...attachments) + Math.max(...attachments)) / 2;
  const rearmost = found.reduce((best, leg) => (rear * leg.hip.x > rear * best.hip.x ? leg : best));
  const tails = found.filter(
    (leg) =>
      (leg === rearmost &&
        (rearLeanDeg(leg, facing) >= TAIL_LEAN_DEG || leg.taper <= TAIL_TAPER)) ||
      (rear * (leg.hip.x - middle) > 0 && rearLeanDeg(leg, facing) >= STEEP_LEAN_DEG),
  );
  const kept = found.filter((leg) => !tails.includes(leg));
  if (kept.length !== 4) {
    return [[...found], []];
  }
  return [kept, tails.map((tail): Tail => [tail.hip, tail.foot])];
}

type Placed = Record<JointId, Point>;

function placeLeg(joints: Placed, leg: Leg, side: "left" | "right", end: "f" | "b"): void {
  joints[`${side}_hip_${end}`] = leg.hip;
  joints[`${side}_knee_${end}`] = leg.knee;
  joints[`${side}_foot_${end}`] = leg.foot;
}

function placePair(joints: Placed, first: Leg, second: Leg, end: "f" | "b"): void {
  // A leg drawn closer to the viewer reaches lower on the page.
  const [near, far] = first.foot.y >= second.foot.y ? [first, second] : [second, first];
  placeLeg(joints, near, "left", end);
  placeLeg(joints, far, "right", end);
}

/**
 * Pixels of the mask's medial axis in raster order, by Zhang and Suen's parallel thinning.
 *
 * Zhang and Suen, "A fast parallel algorithm for thinning digital patterns",
 * 1984. scikit-image's skeletonize uses its own lookup table, so a few axis
 * pixels can differ from the Python rig; joints snap to the nearest one either way.
 */
export function medialAxis(mask: Mask): Point[] {
  const width = mask.width + 2;
  const image = new Uint8Array(width * (mask.height + 2));
  let foreground: number[] = [];
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (filled(mask, x, y)) {
        const index = (y + 1) * width + x + 1;
        image[index] = 1;
        foreground.push(index);
      }
    }
  }
  // Clockwise from north: P2 to P9 in the paper's numbering.
  const around = [-width, -width + 1, 1, width + 1, width, width - 1, -1, -width - 1];
  for (let removed = true; removed;) {
    removed = false;
    for (const first of [true, false]) {
      const doomed = foreground.filter((index) => {
        const p = around.map((offset) => image[index + offset] ?? 0);
        const [n = 0, , e = 0, , s = 0, , w = 0] = p;
        const set = p.reduce((total, value) => total + value, 0);
        const transitions = p.filter((value, k) => value === 0 && p[(k + 1) % 8] === 1).length;
        const sides = first ? n * e * s + e * s * w : n * e * w + n * s * w;
        return set >= 2 && set <= 6 && transitions === 1 && sides === 0;
      });
      for (const index of doomed) {
        image[index] = 0;
      }
      if (doomed.length > 0) {
        removed = true;
        foreground = foreground.filter((index) => image[index] === 1);
      }
    }
  }
  return foreground
    .sort((a, b) => a - b)
    .map((index) => ({ x: (index % width) - 1, y: Math.floor(index / width) - 1 }));
}

function onAxis(axis: readonly Point[], point: Point): Point {
  let nearest = axis[0] ?? point;
  let best = Infinity;
  for (const pixel of axis) {
    const distance = Math.hypot(pixel.x - point.x, pixel.y - point.y);
    if (distance < best) {
      best = distance;
      nearest = pixel;
    }
  }
  return nearest;
}

/** The middle of the body over a pair of legs, or null if they float free. */
function above(mask: Mask, pair: readonly Leg[]): Point | null {
  const column = roundHalfEven(pair.reduce((total, leg) => total + leg.hip.x, 0) / pair.length);
  let bottom = roundHalfEven(Math.min(...pair.map((leg) => leg.hip.y))) - 1;
  while (bottom >= 0 && !filled(mask, column, bottom)) {
    bottom--;
  }
  if (bottom < 0) {
    return null;
  }
  let top = bottom;
  while (top > 0 && filled(mask, column, top - 1)) {
    top--;
  }
  return { x: column, y: (top + bottom) / 2 };
}

/**
 * Move the hip, torso and neck onto the medial axis of the outline.
 *
 * The hip sits over the back legs and the torso over the front legs; without
 * legs they start from the template. The neck starts 70% of the way from the
 * torso to the nose. A joint outside the outline cannot move the drawing, so
 * each one ends on the nearest medial pixel.
 */
function body(
  joints: Placed,
  mask: Mask,
  facing: Facing,
  front: readonly Leg[],
  back: readonly Leg[],
): void {
  const axis = medialAxis(mask);
  const hip = onAxis(axis, (back.length > 0 ? above(mask, back) : null) ?? joints.hip);
  const torso = onAxis(axis, (front.length > 0 ? above(mask, front) : null) ?? joints.torso);

  let firstRow = -1;
  let lastRow = -1;
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (filled(mask, x, y)) {
        firstRow = firstRow < 0 ? y : firstRow;
        lastRow = y;
        break;
      }
    }
  }
  const noseBottom = Math.trunc(firstRow + Math.max(1, (lastRow + 1 - firstRow) * NOSE_REGION));
  let nose: Point | null = null;
  for (let y = 0; y < noseBottom; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (filled(mask, x, y) && (nose === null || (facing === "right" ? x > nose.x : x < nose.x))) {
        nose = { x, y };
      }
    }
  }
  if (nose === null) {
    throw new Error("The mask must contain the drawing");
  }
  const neck = onAxis(axis, {
    x: torso.x + NECK_FRACTION * (nose.x - torso.x),
    y: torso.y + NECK_FRACTION * (nose.y - torso.y),
  });
  joints.root = hip;
  joints.hip = hip;
  joints.torso = torso;
  joints.neck = neck;
}

/** The side the head is on, or right when no head stands out. */
export function detectFacing(mask: Mask): Facing {
  let [left, top, right, bottom] = [mask.width, mask.height, -1, -1];
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (filled(mask, x, y)) {
        left = Math.min(left, x);
        top = Math.min(top, y);
        right = Math.max(right, x);
        bottom = Math.max(bottom, y);
      }
    }
  }
  if (right < 0) {
    throw new Error("The mask must contain the drawing");
  }
  const width = right + 1 - left;
  // The head is the mass in the top quarter; a raised tail is there too but thinner.
  const headBottom = top + Math.max(1, (bottom + 1 - top) * HEAD_REGION);
  let sum = 0;
  let count = 0;
  for (let y = top; y < headBottom; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (filled(mask, x, y)) {
        sum += x;
        count++;
      }
    }
  }
  const lean = (sum / count - (left + (width - 1) / 2)) / width;
  return lean < -FACING_DEADBAND ? "left" : "right";
}

/**
 * Move leg and body joints onto the drawing's mask.
 *
 * Four leg components become front and back pairs; the leg that reaches lower
 * in each pair is near. With more than four, tails are set aside first and stay
 * part of the body. Two components describe a side view, so both sides share
 * one position per pair. Any other count keeps the fitted template legs. The
 * hip, torso and neck always end inside the outline.
 */
export function snap(mask: Mask, joints: Joints, facing: Facing): Snapped {
  if (mask.width <= 0 || mask.height <= 0) {
    throw new Error("Expected a nonempty mask");
  }
  if (!mask.data.some((value) => value > 0)) {
    throw new Error("The mask must contain the drawing");
  }
  const updated: Placed = { ...joints };
  const [found, tails] = withoutTails(legs(mask), facing);
  let count: 2 | 4 = 4;
  let front: readonly Leg[] = [];
  let back: readonly Leg[] = [];
  const [a, b, c, d] = found;
  if (found.length === 4 && a && b && c && d) {
    const [first, second, third, fourth] = facing === "right" ? [c, d, a, b] : [a, b, c, d];
    placePair(updated, first, second, "f");
    placePair(updated, third, fourth, "b");
    front = [first, second];
    back = [third, fourth];
  } else if (found.length === 2 && a && b) {
    const [frontLeg, backLeg] = facing === "right" ? [b, a] : [a, b];
    for (const side of ["left", "right"] as const) {
      placeLeg(updated, frontLeg, side, "f");
      placeLeg(updated, backLeg, side, "b");
    }
    front = [frontLeg];
    back = [backLeg];
    count = 2;
  }
  body(updated, mask, facing, front, back);
  return {
    joints: jointsFrom((id) => updated[id]),
    legs: count,
    tails,
    legsGuessed: front.length === 0,
  };
}

/** A rigged drawing: what the page shows, and the mask and snap the animation is built from. */
export interface RiggedDrawing {
  readonly rig: Omit<Rig, "id">;
  readonly mask: Mask;
  readonly snapped: Snapped;
}

/** Segment a drawing, fit the template to its box and snap its joints, facing the head's side. */
export function rigDrawing(pixels: Pixels): RiggedDrawing {
  const segmented = segment(pixels);
  const mask = resizeNearest(segmented, pixels.width, pixels.height);
  let [left, top, right, bottom] = [mask.width, mask.height, 0, 0];
  mask.data.forEach((value, index) => {
    if (value > 0) {
      const x = index % mask.width;
      const y = Math.floor(index / mask.width);
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x + 1);
      bottom = Math.max(bottom, y + 1);
    }
  });
  const facing = detectFacing(mask);
  const snapped = snap(mask, fitTemplate(left, top, right, bottom, facing), facing);
  return {
    rig: {
      width: pixels.width,
      height: pixels.height,
      boxHeight: bottom - top,
      outline: outline(mask),
      joints: snapped.joints,
    },
    mask,
    snapped,
  };
}
