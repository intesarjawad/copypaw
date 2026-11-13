import { describe, expect, it } from "vitest";
import { tailHandles, withBoneHandles } from "./handles";

const NAMES = ["torso", "neck", "left_hip_f", "left_knee_f", "left_foot_f"];
const POSITIONS = Float32Array.from([0, 0, 7, 7, 0, -2, 2, -6, 2, -10]);

function expectClose(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    expect(actual[index]).toBeCloseTo(value, 5);
  });
}

describe("withBoneHandles", () => {
  it("carries handles along each leg bone, either side of it, and past the neck", () => {
    const handles = withBoneHandles(NAMES, POSITIONS);

    expect(handles.subarray(0, POSITIONS.length)).toEqual(POSITIONS);
    // The lower bone runs from (2, -6) to (2, -10): straight down, four long.
    const lower = POSITIONS.length + 14;
    expectClose(
      handles.subarray(lower, lower + 14),
      [2, -7, 2.4, -7, 1.6, -7, 2, -8, 2, -9, 2.4, -9, 1.6, -9],
    );
    expectClose(handles.subarray(-2), [8.5, 8.5]);
  });

  it("turns the side handles with their bone", () => {
    const upright = withBoneHandles(NAMES.slice(2, 4), Float32Array.from([0, 0, 0, -10]));
    const level = withBoneHandles(NAMES.slice(2, 4), Float32Array.from([0, 0, 10, 0]));

    const turned = Array.from(upright).flatMap((_, index) =>
      index % 2 === 0 ? [-(upright[index + 1] ?? 0), upright[index] ?? 0] : [],
    );
    expectClose(level, turned);
  });

  it("requires one position per name", () => {
    expect(() => withBoneHandles(NAMES, POSITIONS.subarray(0, 4))).toThrow("2 joints but 5 names");
  });
});

describe("tailHandles", () => {
  it("moves a tail above the ground with the body", () => {
    expectClose(
      tailHandles(
        [
          [
            [0, 5],
            [-4, 2],
          ],
        ],
        [1, -1],
        0,
      ),
      [1, 4, -3, 1],
    );
  });

  it("drags a tip that would sink along the ground, keeping the tail's length", () => {
    const [baseX = 0, baseY = 0, tipX = 0, tipY = 0] = tailHandles(
      [
        [
          [0, 5],
          [-4, 2],
        ],
      ],
      [0, -3],
      0,
    );

    expectClose([baseX, baseY, tipY], [0, 2, 0]);
    expect(tipX).toBeLessThan(0);
    expect(Math.hypot(tipX - baseX, tipY - baseY)).toBeCloseTo(5, 5);
  });

  it.each([-1, -0.3, 0, 0.3, 1])(
    "keeps the tip off the ground and behind the rearmost leg at turn %d",
    (turn) => {
      const [baseX = 0, baseY = 0, tipX = 0, tipY = 0] = tailHandles(
        [
          [
            [0, 5],
            [-3, 1],
          ],
        ],
        [0, 0],
        0.5,
        turn,
        [-2],
      );

      expect(tipY).toBeGreaterThanOrEqual(0.5 - 1e-6);
      expect(tipX).toBeLessThanOrEqual(-2 + 1e-6);
      expect(Math.hypot(tipX - baseX, tipY - baseY)).toBeCloseTo(5, 5);
    },
  );

  it.each([
    [-4, -1],
    [4, 1],
  ])("lifts a tail at x %d away from the legs, keeping its length", (tipX, side) => {
    const [baseX = 0, baseY = 0, outX = 0, outY = 0] = tailHandles(
      [
        [
          [0, 5],
          [tipX, 1],
        ],
      ],
      [0, 0],
      null,
      0,
      null,
      [Math.PI / 4],
    );

    expect((outX - baseX) * side).toBeGreaterThan(Math.abs(tipX));
    expect(outY).toBeGreaterThan(1);
    expect(Math.hypot(outX - baseX, outY - baseY)).toBeCloseTo(Math.hypot(tipX, 4), 5);
  });

  it("turns a tail about its base", () => {
    expectClose(
      tailHandles(
        [
          [
            [0, 5],
            [-3, 1],
          ],
        ],
        [0, 0],
        null,
        Math.PI / 2,
      ),
      [0, 5, 4, 2],
    );
  });
});
