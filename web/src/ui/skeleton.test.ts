import { describe, expect, it } from "vitest";
import { readSkeleton, skeletonLines } from "./skeleton";

const RECORD = {
  fps: 10,
  facing: "right",
  edges: [[0, 1]],
  frames: [
    [
      [1, 2],
      [3, 4],
    ],
    [
      [5, 6],
      [7, 8],
    ],
  ],
};

describe("skeleton overlay", () => {
  it("picks the frame shown at the playback time", () => {
    const skeleton = readSkeleton(RECORD);
    expect(skeletonLines(skeleton, 0.15, 100)).toEqual([
      { from: { x: 5, y: 6 }, to: { x: 7, y: 8 } },
    ]);
  });

  it("holds the last frame after the motion ends", () => {
    const skeleton = readSkeleton(RECORD);
    expect(skeletonLines(skeleton, 9, 100)[0]?.from).toEqual({ x: 5, y: 6 });
  });

  it("mirrors a left-facing clip back to its pixels", () => {
    const skeleton = readSkeleton({ ...RECORD, facing: "left" });
    expect(skeletonLines(skeleton, 0, 100)).toEqual([
      { from: { x: 98, y: 2 }, to: { x: 96, y: 4 } },
    ]);
  });

  it("rejects a record without frames", () => {
    expect(() => readSkeleton({ ...RECORD, frames: "none" })).toThrow();
  });
});
