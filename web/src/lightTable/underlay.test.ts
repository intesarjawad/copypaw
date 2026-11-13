import { describe, expect, it } from "vitest";
import type { Animation } from "../compute/animate";
import { frameCamera } from "../compute/camera";
import record from "./sheet.json";
import { readSheet } from "./sheet";
import { clipAlone, footing, traceOver } from "./underlay";

const WINDOW = [400, 400] as const;

// Three vertices per frame: a head up high and two paws on the ground, world y up.
function walk(frames: readonly (readonly number[])[], mirrored = false): Animation {
  return {
    window: WINDOW,
    camera: frameCamera([0, 0], [10, 10], WINDOW),
    frameTimeS: 1 / 15,
    frameCount: frames.length,
    uv: new Float32Array(6),
    indices: new Uint32Array([0, 1, 2]),
    positions: Float32Array.from(frames.flat()),
    mirrored,
  };
}

describe("footing", () => {
  it("reads the paws from the lowest points, not the head", () => {
    const { paws, floor, top } = footing(
      walk([
        [9, 10, 4, 0, 6, 0],
        [1, 10, 4, 0, 6, 0],
      ]),
    );
    expect(floor).toBeGreaterThan(top);
    expect(paws[0]).toBeCloseTo(WINDOW[0] / 2, 0);
    expect(paws[1]).toBeCloseTo(paws[0] ?? Number.NaN, 6);
  });

  it("follows the paws when they move", () => {
    const { paws } = footing(
      walk([
        [5, 10, 2, 0, 4, 0],
        [5, 10, 6, 0, 8, 0],
      ]),
    );
    expect(paws[1]).toBeGreaterThan(paws[0] ?? Number.POSITIVE_INFINITY);
  });
});

describe("traceOver", () => {
  const sheet = readSheet(record);
  // Paws three world units apart under a head five units up, standing still.
  const standing = (mirrored: boolean): Animation =>
    walk(
      Array.from({ length: 30 }, () => [5, 5, 3.5, 0, 6.5, 0]),
      mirrored,
    );

  it("scales the drawing's stride to the dog's", () => {
    const animation = standing(false);
    const tracing = traceOver(animation, sheet);
    const { span } = footing(animation);
    const { d } = tracing.drawingAt(0);
    expect(span * d).toBeCloseTo(sheet.clip.span * tracing.clip.width, 6);
  });

  it("stands the drawing's paws on the dog's, on the floor line", () => {
    const animation = standing(false);
    const tracing = traceOver(animation, sheet);
    const { paws, floor } = footing(animation);
    for (const frame of [0, 7, 29]) {
      const { a, d, e, f } = tracing.drawingAt(frame);
      const dog = sheet.clip.paws[tracing.clipFrame(frame)] ?? Number.NaN;
      expect(a * (paws[frame] ?? 0) + e).toBeCloseTo(tracing.clip.x + dog * tracing.clip.width, 6);
      expect(d * floor + f).toBeCloseTo(tracing.floorY, 6);
    }
  });

  it("puts each frame's planted paw on the floor line, even mid-hop", () => {
    // The paws leave the ground every other frame.
    const hopping = walk(
      Array.from({ length: 30 }, (_, frame) => {
        const lift = frame % 2 === 0 ? 0 : 0.5;
        return [5, 5 + lift, 3.5, lift, 6.5, lift];
      }),
    );
    const tracing = traceOver(hopping, sheet);
    const { lows } = footing(hopping);
    expect(lows[1]).toBeLessThan(lows[0] ?? Number.NEGATIVE_INFINITY);
    for (const frame of [0, 1, 2, 29]) {
      const { d, f } = tracing.drawingAt(frame);
      expect(d * (lows[frame] ?? Number.NaN) + f).toBeCloseTo(tracing.floorY, 6);
    }
  });

  it("crops the sheet to the walk, ending at the floor line", () => {
    const tracing = traceOver(standing(false), sheet);
    const { view, floorY } = tracing;
    expect(view.y + view.height).toBeGreaterThan(floorY);
    expect(view.width).toBeLessThan(WINDOW[0] * 1.2);
  });

  it("keeps the dog and the drawing inside the window", () => {
    const animation = standing(false);
    const tracing = traceOver(animation, sheet);
    const { clip } = tracing;
    const { top, reach, paws } = footing(animation);
    expect(clip.x).toBeGreaterThanOrEqual(0);
    expect(clip.x + clip.width).toBeLessThanOrEqual(WINDOW[0] + 1e-6);
    expect(clip.y).toBeGreaterThanOrEqual(0);
    for (let frame = 0; frame < animation.frameCount; frame++) {
      const { a, d, e, f } = tracing.drawingAt(frame);
      const middle = paws[frame] ?? 0;
      const ends = [a * (middle - reach[0]) + e, a * (middle + reach[1]) + e];
      expect(Math.min(...ends)).toBeGreaterThanOrEqual(-1e-6);
      expect(Math.max(...ends)).toBeLessThanOrEqual(WINDOW[0] + 1e-6);
      expect(d * top + f).toBeGreaterThanOrEqual(0);
    }
  });

  it("turns the drawing to face the way the dog walks", () => {
    const dogWalksLeft = (sheet.clip.paws.at(-1) ?? 0) < (sheet.clip.paws[0] ?? 0);
    const facingRight = traceOver(standing(false), sheet).drawingAt(0).a;
    const facingLeft = traceOver(standing(true), sheet).drawingAt(0).a;
    expect(facingRight < 0).toBe(dogWalksLeft);
    expect(facingLeft < 0).toBe(!dogWalksLeft);
  });
});

describe("clipAlone", () => {
  const sheet = readSheet(record);

  it("crops the clip's own cell to the dog, floor line included", () => {
    const { clip, floorY, view } = clipAlone(sheet);
    const { box } = sheet.clip;
    expect(view.x).toBeLessThan(box.x * clip.width);
    expect(view.x + view.width).toBeGreaterThan((box.x + box.width) * clip.width);
    expect(view.y).toBeLessThan(box.y * clip.height);
    expect(view.y + view.height).toBeGreaterThan(floorY);
  });
});
