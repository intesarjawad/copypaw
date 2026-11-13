import { MalformedResponse, field, list, number, text } from "../parse";

/** A box in CSS pixels, measured from the top left. */
export interface Rect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** Frames laid out row by row in one image, `columns` to a row. */
export interface Atlas {
  readonly cellWidth: number;
  readonly cellHeight: number;
  readonly columns: number;
  readonly rows: number;
}

/** The light table's record, written by scripts/build_light_table.py from one copypaw run. */
export interface Sheet {
  readonly scene: string;
  readonly character: string;
  readonly fps: number;
  readonly frames: number;
  readonly paws: readonly string[];
  /** Per paw, per frame: true where the paw is planted. */
  readonly dog: readonly (readonly boolean[])[];
  readonly drawing: readonly (readonly boolean[])[];
  readonly contactMatch: number;
  readonly locomotion: boolean;
  readonly landings: readonly number[];
  readonly landingsNeeded: number;
  readonly clip: {
    readonly atlas: Atlas;
    /** Middle of the four paws per frame, as a fraction of the clip's width. */
    readonly paws: readonly number[];
    /** Front to back paw over the run, as a fraction of the clip's width. */
    readonly span: number;
    /** What the dog covers in any frame, as fractions of the clip. */
    readonly box: Rect;
    /** Floor line as a fraction of the clip's height. */
    readonly floor: number;
  };
  readonly figure: {
    readonly atlas: Atlas;
    /** Middle of the drawing's paws per frame, as a fraction of the figure's width. */
    readonly paws: readonly number[];
    /** Height as a fraction of the clip's height: both share one scale. */
    readonly height: number;
    /** Per frame, the lowest ink, the planted paw, as a fraction of the figure's height. */
    readonly lows: readonly number[];
  };
}

function cell(value: unknown, key: string): readonly [number, number] {
  const [width, height] = list(value, key);
  if (typeof width !== "number" || typeof height !== "number" || !(width > 0 && height > 0)) {
    throw new MalformedResponse(key);
  }
  return [width, height];
}

function tracks(value: unknown, key: string, frames: number): boolean[][] {
  return list(value, key).map((track) => {
    if (typeof track !== "string" || !new RegExp(`^[01]{${String(frames)}}$`).test(track)) {
      throw new MalformedResponse(key);
    }
    return Array.from({ length: frames }, (_, index) => track[index] === "1");
  });
}

function fraction(value: unknown, key: string): number {
  const found = number(value, key);
  if (found < 0 || found > 1) {
    throw new MalformedResponse(key);
  }
  return found;
}

function perFrame(value: unknown, key: string, frames: number): number[] {
  const found = list(value, key).map((_, index, all) => fraction(all, String(index)));
  if (found.length !== frames) {
    throw new MalformedResponse(key);
  }
  return found;
}

function coverBox(value: unknown): Rect {
  const [left, top, right, bottom] = list(value, "box").map((_, index, all) =>
    fraction(all, String(index)),
  );
  if (left === undefined || top === undefined || right === undefined || bottom === undefined) {
    throw new MalformedResponse("box");
  }
  if (!(right > left && bottom > top)) {
    throw new MalformedResponse("box");
  }
  return { x: left, y: top, width: right - left, height: bottom - top };
}

export function readSheet(json: unknown): Sheet {
  const frames = number(json, "frames");
  const columns = number(json, "columns");
  if (!Number.isInteger(frames) || frames < 1 || !Number.isInteger(columns) || columns < 1) {
    throw new MalformedResponse("frames");
  }
  const atlas = (picture: unknown): Atlas => {
    const [cellWidth, cellHeight] = cell(picture, "cell");
    return { cellWidth, cellHeight, columns, rows: Math.ceil(frames / columns) };
  };
  const paws = list(json, "paws").map((paw) => {
    if (typeof paw !== "string") {
      throw new MalformedResponse("paws");
    }
    return paw;
  });
  const footfalls = field(json, "footfalls");
  const dog = tracks(footfalls, "dog", frames);
  const drawing = tracks(footfalls, "drawing", frames);
  if (dog.length !== paws.length || drawing.length !== paws.length) {
    throw new MalformedResponse("footfalls");
  }
  const clip = field(json, "dog");
  const figure = field(json, "drawing");
  const landings = list(json, "landings").map((_, index, all) => number(all, String(index)));
  const locomotion = field(json, "locomotion");
  if (typeof locomotion !== "boolean") {
    throw new MalformedResponse("locomotion");
  }
  return {
    scene: text(json, "scene"),
    character: text(json, "character"),
    fps: number(json, "fps"),
    frames,
    paws,
    dog,
    drawing,
    contactMatch: fraction(json, "contactMatch"),
    locomotion,
    landings,
    landingsNeeded: number(json, "landingsNeeded"),
    clip: {
      atlas: atlas(clip),
      paws: perFrame(clip, "paws", frames),
      span: fraction(clip, "span"),
      box: coverBox(clip),
      floor: fraction(clip, "floor"),
    },
    figure: {
      atlas: atlas(figure),
      paws: perFrame(figure, "paws", frames),
      height: fraction(figure, "height"),
      lows: perFrame(figure, "lows", frames),
    },
  };
}

export interface Placement {
  readonly clip: Rect;
  readonly figure: Rect;
  readonly floorY: number;
}

/**
 * Put the drawing on the clip's floor line with its paws over the dog's, at the scale the
 * side-by-side video gives it. `clip` is where the clip frame is projected.
 */
export function place(sheet: Sheet, frame: number, clip: Rect): Placement {
  const floorY = clip.y + sheet.clip.floor * clip.height;
  const { cellWidth, cellHeight } = sheet.figure.atlas;
  const height = sheet.figure.height * clip.height;
  const width = (height * cellWidth) / cellHeight;
  const dogPaws = sheet.clip.paws[frame] ?? 0.5;
  const figurePaws = sheet.figure.paws[frame] ?? 0.5;
  return {
    clip,
    floorY,
    figure: {
      x: clip.x + dogPaws * clip.width - figurePaws * width,
      y: floorY - (sheet.figure.lows[frame] ?? 1) * height,
      width,
      height,
    },
  };
}

/** The frame shown when motion is reduced, and in the photograph that stands in without WebGL. */
export const STILL_FRAME = 18;

/** Wrap any whole number onto a frame of the loop. */
export function wrapFrame(frame: number, frames: number): number {
  return ((Math.round(frame) % frames) + frames) % frames;
}

const SWITCH_ON_MS = 420;
const SWITCH_OFF_MS = 700;
// A tube strikes in stutters before it holds, and gives one last flicker as it dies: [until ms, level].
const STRIKE: readonly (readonly [number, number])[] = [
  [70, 0.55],
  [130, 0.06],
  [200, 0.8],
  [250, 0.25],
];
const DIE: readonly (readonly [number, number])[] = [
  [60, 0.5],
  [110, 0.85],
];

function stutter(steps: readonly (readonly [number, number])[], elapsedMs: number): number | null {
  return steps.find(([until]) => elapsedMs < until)?.[1] ?? null;
}

/**
 * How bright the lamp is `elapsedMs` after it was switched. Switching on stutters like
 * a fluorescent tube striking; switching off flickers once and fades as the tube's glow dies.
 */
export function lampLevel(on: boolean, elapsedMs: number): number {
  const steps = on ? STRIKE : DIE;
  const settleMs = on ? SWITCH_ON_MS : SWITCH_OFF_MS;
  if (elapsedMs >= settleMs) {
    return on ? 1 : 0;
  }
  const level = stutter(steps, elapsedMs);
  if (level !== null) {
    return level;
  }
  const last = steps[steps.length - 1] ?? [0, 1];
  const progress = (elapsedMs - last[0]) / (settleMs - last[0]);
  return on ? last[1] + (1 - last[1]) * progress : last[1] * Math.exp(-4.5 * progress);
}

export function lampSettled(on: boolean, elapsedMs: number): boolean {
  return elapsedMs >= (on ? SWITCH_ON_MS : SWITCH_OFF_MS);
}
