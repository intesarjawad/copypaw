import { describe, expect, it } from "vitest";
import {
  JOINT_IDS,
  flipJoints,
  jointsFrom,
  moveJoint,
  nextJoint,
  nudgeJoint,
  toImagePoint,
} from "./joints";

const AT_ORIGIN = jointsFrom(() => ({ x: 50, y: 50 }));
const BOX_HEIGHT = 200;

describe("joint selection", () => {
  it("lists sixteen distinct joints", () => {
    expect(new Set(JOINT_IDS).size).toBe(16);
  });

  it("advances to the next joint and wraps after the last", () => {
    expect(nextJoint("root")).toBe("hip");
    expect(nextJoint("left_foot_f")).toBe("root");
  });
});

describe("moving joints", () => {
  it("moves only the chosen joint", () => {
    const moved = moveJoint(AT_ORIGIN, "neck", { x: 3, y: 4 });
    expect(moved.neck).toEqual({ x: 3, y: 4 });
    expect(moved.hip).toEqual({ x: 50, y: 50 });
  });

  it("nudges by 1% of the box height per arrow press", () => {
    expect(nudgeJoint(AT_ORIGIN, "neck", "ArrowRight", false, BOX_HEIGHT)?.neck).toEqual({
      x: 52,
      y: 50,
    });
    expect(nudgeJoint(AT_ORIGIN, "neck", "ArrowUp", false, BOX_HEIGHT)?.neck).toEqual({
      x: 50,
      y: 48,
    });
  });

  it("nudges by 5% with Shift", () => {
    expect(nudgeJoint(AT_ORIGIN, "hip", "ArrowLeft", true, BOX_HEIGHT)?.hip).toEqual({
      x: 40,
      y: 50,
    });
    expect(nudgeJoint(AT_ORIGIN, "hip", "ArrowDown", true, BOX_HEIGHT)?.hip).toEqual({
      x: 50,
      y: 60,
    });
  });

  it("ignores keys that are not arrows", () => {
    expect(nudgeJoint(AT_ORIGIN, "hip", "Enter", false, BOX_HEIGHT)).toBeNull();
  });

  it("flips across the centre of the outline", () => {
    const outline = [
      { x: 10, y: 0 },
      { x: 110, y: 0 },
      { x: 60, y: 90 },
    ];
    const flipped = flipJoints(moveJoint(AT_ORIGIN, "neck", { x: 30, y: 7 }), outline);
    expect(flipped.neck).toEqual({ x: 90, y: 7 });
    expect(flipped.hip).toEqual({ x: 70, y: 50 });
  });
});

describe("pointer positions", () => {
  it("scales from the shown size to image pixels", () => {
    const shown = { left: 100, top: 20, width: 300, height: 200 };
    expect(toImagePoint({ x: 250, y: 70 }, shown, { width: 600, height: 400 })).toEqual({
      x: 300,
      y: 100,
    });
  });
});
