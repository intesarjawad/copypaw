import { JOINT_IDS, jointsFrom, type JointId, type Joints, type Point } from "../joints";
import { Arap } from "./arap";
import { poseForDrawing, type CapturedMotion } from "./bake";
import { frameCamera, type CameraPosition } from "./camera";
import { ContactFilter } from "./contactIk";
import { drawOrder } from "./drawOrder";
import { HandleFilter, type TailPair } from "./handles";
import { roundHalfEven, type Mask } from "./mask";
import { characterMesh, type CharacterMesh } from "./mesh";
import { drawnJoints, retarget } from "./retarget";
import type { Tail } from "./snap";

/** A captured motion as `scripts/make_goldens.py --web` writes it. */
export interface MotionRecord {
  readonly fps: number;
  readonly kp: readonly (readonly (readonly number[])[])[];
  readonly contacts: readonly (readonly boolean[])[];
  readonly ground_line_px: readonly number[];
  readonly travel_root_x: readonly number[];
}

/** A rigged drawing: its mask and its joints in source pixels. */
export interface Character {
  readonly mask: Mask;
  readonly joints: Joints;
  readonly legs: 2 | 4;
  readonly legsGuessed: boolean;
  readonly tails: readonly Tail[];
}

/** Everything a renderer needs to play the animated drawing. */
export interface Animation {
  readonly window: readonly [width: number, height: number];
  readonly camera: CameraPosition;
  readonly frameTimeS: number;
  readonly frameCount: number;
  /** Per vertex, where it samples the drawing, as fractions of its width and height. */
  readonly uv: Float32Array;
  /** Triangle corners in the order the engine draws them. */
  readonly indices: Uint32Array;
  /** Per frame, every vertex's x and y in world units, frame after frame. */
  readonly positions: Float32Array;
  /** The drawing faced left, so the engine animated its mirror image and the picture flips back. */
  readonly mirrored: boolean;
}

/** The engine renders at this rate, as the CLI does. */
export const OUTPUT_FPS = 15;
const WINDOW_HEIGHT = 512;
// The solver forms its normal equations in single precision, and they hold this
// weight squared; 100 pins the handles firmly and keeps the drawn pose at rest.
const HANDLE_WEIGHT = 100;
// The engine reads frame times written to six decimals.
const FRAME_TIME_DECIMALS = 1e6;

/** Resample keypoints linearly and contacts to their nearest analysis frame, as the CLI does. */
export function resampleMotion(
  motion: MotionRecord,
  fps = OUTPUT_FPS,
): CapturedMotion & { readonly travelRootX: readonly number[] } {
  const frames = motion.kp.length;
  if (frames === 0 || !(motion.fps > 0) || !Number.isFinite(motion.fps)) {
    throw new Error("A motion needs at least one frame and a finite, positive frame rate");
  }
  const count = Math.floor(((frames - 1) / motion.fps) * fps) + 1;
  const lerp = (from: number, to: number, weight: number): number =>
    Math.fround(from + (to - from) * weight);
  const kp: Point[][] = [];
  const contacts: boolean[][] = [];
  const travelRootX: number[] = [];
  for (let k = 0; k < count; k++) {
    const timeS = k / fps;
    const position = Math.min(Math.max(timeS * motion.fps, 0), frames - 1);
    const lower = Math.min(Math.floor(position), frames - 1);
    const upper = Math.min(lower + 1, frames - 1);
    const weight = position - lower;
    const nearest = Math.min(Math.floor(timeS * motion.fps + 0.5), frames - 1);
    const [low, high] = [motion.kp[lower] ?? [], motion.kp[upper] ?? []];
    kp.push(
      low.map(([x = 0, y = 0], index) => {
        const [hx = x, hy = y] = high[index] ?? [];
        return { x: lerp(x, hx, weight), y: lerp(y, hy, weight) };
      }),
    );
    contacts.push([...(motion.contacts[nearest] ?? [])]);
    const [rootLow = 0, rootHigh = 0] = [motion.travel_root_x[lower], motion.travel_root_x[upper]];
    travelRootX.push(lerp(rootLow, rootHigh, weight));
  }
  const [slope = 0, intercept = 0] = motion.ground_line_px;
  return { kp, contacts, groundLinePx: [slope, intercept], travelRootX };
}

function mirrored(character: Character): Character {
  const { mask, joints, tails } = character;
  const flipX = (point: Point): Point => ({ x: mask.width - 1 - point.x, y: point.y });
  const data = new Uint8Array(mask.data.length);
  for (let row = 0; row < mask.height; row++) {
    for (let column = 0; column < mask.width; column++) {
      data[row * mask.width + column] = mask.data[row * mask.width + mask.width - 1 - column] ?? 0;
    }
  }
  return {
    ...character,
    mask: { width: mask.width, height: mask.height, data },
    joints: jointsFrom((id) => flipX(joints[id])),
    tails: tails.map(([base, tip]): Tail => [flipX(base), flipX(tip)]),
  };
}

/** Joints as the drawing's config stores them: whole pixels inside the image, the root on the hip. */
function pixelJoints(joints: Joints, width: number, height: number): Record<JointId, Point> {
  const clamp = (value: number, size: number): number =>
    Math.min(Math.max(roundHalfEven(value), 0), size - 1);
  const pixels = jointsFrom((id) => ({
    x: clamp(joints[id].x, width),
    y: clamp(joints[id].y, height),
  }));
  return { ...pixels, root: pixels.hip };
}

/**
 * Animate a drawing on a treadmill with a captured motion, as the CLI's render does.
 *
 * The motion is posed for the drawing, retargeted onto its joints, stood on
 * its contact paws and handed to the as-rigid-as-possible solver, which moves
 * the mesh. A first pass teaches the contact filter its ground; the second
 * pass gives the frames, and the camera is fitted to everything they reach.
 * meshFor builds the mesh from the drawing's upright mask; parity tests pass
 * the engine's own to check the rest of the chain apart from triangulation.
 */
export function animateDrawing(
  drawing: Character,
  motion: MotionRecord,
  meshFor: (mask: Mask) => CharacterMesh = characterMesh,
): Animation {
  const flip = drawing.joints.neck.x < drawing.joints.hip.x;
  const character = flip ? mirrored(drawing) : drawing;
  const { mask, legs, legsGuessed } = character;
  const size = Math.max(mask.width, mask.height);
  const joints = pixelJoints(character.joints, mask.width, mask.height);
  const resampled = resampleMotion(motion);
  const posed = poseForDrawing(resampled, joints, legsGuessed);
  const drawn = drawnJoints(joints, mask.height, size);
  const unfiltered = retarget(
    posed,
    posed.map(() => 0),
    drawn,
    legs,
  );
  const frameTimeS = Math.round(FRAME_TIME_DECIMALS / OUTPUT_FPS) / FRAME_TIME_DECIMALS;
  const tails = character.tails.map(([base, tip]): TailPair => [
    [Math.fround(base.x / size), Math.fround((mask.height - base.y) / size)],
    [Math.fround(tip.x / size), Math.fround((mask.height - tip.y) / size)],
  ]);
  // Guessed legs stay as drawn, so there is no stance to stand the drawing on.
  const contacts = resampled.contacts.map((row) => row.map((planted) => planted && !legsGuessed));
  const handles = new HandleFilter(
    JOINT_IDS,
    drawn,
    tails,
    frameTimeS,
    new ContactFilter(JOINT_IDS, contacts, "treadmill"),
  );

  const mesh = meshFor(mask);
  const vertices = mesh.vertices.map(
    ([row, column]) => [Math.fround(row / size), Math.fround(column / size)] as const,
  );
  const arap = new Arap(
    points(handles.drawnPose()),
    mesh.triangles,
    vertices.map(([x, y]) => ({ x, y })),
    HANDLE_WEIGHT,
  );

  unfiltered.forEach((frame, index) => handles.apply(index, frame));
  const positions = new Float32Array(unfiltered.length * vertices.length * 2);
  const low: [number, number] = [Infinity, Infinity];
  const high: [number, number] = [-Infinity, -Infinity];
  unfiltered.forEach((frame, index) => {
    const [rootX = 0, rootY = 0] = frame;
    const controls = points(handles.apply(index, frame)).map(({ x, y }) => ({
      x: x - rootX,
      y: y - rootY,
    }));
    arap.solve(controls).forEach(({ x, y }, vertex) => {
      const offset = 2 * (index * vertices.length + vertex);
      positions[offset] = x + rootX;
      positions[offset + 1] = y + rootY;
      for (const [axis, value] of [x + rootX, y + rootY].entries()) {
        low[axis] = Math.min(low[axis] ?? value, Math.fround(value));
        high[axis] = Math.max(high[axis] ?? value, Math.fround(value));
      }
    });
  });

  const window = [WINDOW_HEIGHT, WINDOW_HEIGHT] as const;
  return {
    window,
    camera: frameCamera(low, high, window),
    frameTimeS,
    frameCount: unfiltered.length,
    uv: Float32Array.from(
      mesh.vertices.flatMap(([row, column]) => [
        row / mask.width,
        (mask.height - column) / mask.height,
      ]),
    ),
    indices: drawOrder(mask, joints, vertices, mesh.triangles, legs),
    positions,
    mirrored: flip,
  };
}

function points(interleaved: Float32Array): Point[] {
  return Array.from({ length: interleaved.length / 2 }, (_, index) => ({
    x: interleaved[2 * index] ?? 0,
    y: interleaved[2 * index + 1] ?? 0,
  }));
}
