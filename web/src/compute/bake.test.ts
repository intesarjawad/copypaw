import { describe, expect, it } from "vitest";
import type { Point } from "../joints";
import { poseForDrawing, type CapturedMotion, type DrawnJoints } from "./bake";

const FRAMES = 3;
const DRAWN: DrawnJoints = {
  left_hip_f: { x: 0, y: 0 },
  left_knee_f: { x: 30, y: 40 },
  left_foot_f: { x: 30, y: 120 },
};
// The pet's planted near-front leg spans 200 px; the drawn one spans its hip-to-foot distance.
const SCALE = 200 / Math.hypot(30, 120);
const STILL_KP: readonly Point[] = [
  [110, -110],
  [110, -90],
  [120, -100],
  [100, -100],
  [100, 0],
  [0, -100],
  [0, 0],
  [0, 100],
  [-50, -100],
  [-50, -20],
  [-50, 40],
  [200, -100],
  [200, 0],
  [200, 120],
  [250, -100],
  [250, -30],
  [250, 40],
].map(([x = 0, y = 0]) => ({ x, y }));

function motion(
  paws: readonly (Point | null)[] = [],
  lifted: readonly number[] = [],
): CapturedMotion {
  return {
    kp: Array.from({ length: FRAMES }, (_, frame) =>
      STILL_KP.map((point, index) => (index === 7 ? (paws[frame] ?? point) : point)),
    ),
    contacts: Array.from({ length: FRAMES }, (_, frame) => [
      !lifted.includes(frame),
      true,
      true,
      true,
    ]),
    groundLinePx: [0, 120],
  };
}

/** The near-front knee and paw of each frame, relative to its hip. */
function nearFront(posed: readonly (readonly Point[])[]): [Point, Point][] {
  return posed.map((frame) => {
    const hip = frame[5] ?? { x: 0, y: 0 };
    const relative = (point: Point | undefined): Point => ({
      x: (point?.x ?? Number.NaN) - hip.x,
      y: (point?.y ?? Number.NaN) - hip.y,
    });
    return [relative(frame[6]), relative(frame[7])];
  });
}

function norm(point: Point): number {
  return Math.hypot(point.x, point.y);
}

describe("poseForDrawing", () => {
  it("leaves each leg as drawn for a still pet", () => {
    for (const [knee, paw] of nearFront(poseForDrawing(motion(), DRAWN))) {
      expect(knee.x).toBeCloseTo(30 * SCALE, 3);
      expect(knee.y).toBeCloseTo(40 * SCALE, 3);
      expect(paw.x).toBeCloseTo(30 * SCALE, 3);
      expect(paw.y).toBeCloseTo(120 * SCALE, 3);
    }
  });

  it("swings a planted leg from the hip without changing its shape", () => {
    // The pet's leg shortens as it swings; the drawn leg must not.
    const legs = nearFront(
      poseForDrawing(motion([{ x: -60, y: 60 }, null, { x: 60, y: 60 }]), DRAWN),
    );

    const pawXs = legs.map(([, paw]) => paw.x);
    expect(pawXs[0]).toBeLessThan(pawXs[1] ?? Number.NaN);
    expect(pawXs[1]).toBeLessThan(pawXs[2] ?? Number.NaN);
    for (const [knee, paw] of legs) {
      expect(norm(knee)).toBeCloseTo(50 * SCALE, 3);
      expect(norm({ x: paw.x - knee.x, y: paw.y - knee.y })).toBeCloseTo(80 * SCALE, 3);
      expect(norm(paw)).toBeCloseTo(200, 3);
    }
  });

  it.each([80, 40, -90])(
    "folds the knee no further than a right angle to lift a paw to %d",
    (height) => {
      const [planted, lifted] = nearFront(
        poseForDrawing(motion([null, { x: 0, y: height }], [1]), DRAWN),
      );
      const [upper, lower] = [50 * SCALE, 80 * SCALE];

      expect(lifted?.[1].y).toBeLessThan(planted?.[1].y ?? Number.NaN);
      expect(norm(lifted?.[0] ?? { x: 0, y: 0 })).toBeCloseTo(upper, 3);
      expect(norm(lifted?.[1] ?? { x: 0, y: 0 })).toBeGreaterThanOrEqual(
        Math.hypot(upper, lower) - 1e-3,
      );
    },
  );

  it("keeps guessed legs in their drawn pose whatever the pet does", () => {
    const posed = poseForDrawing(
      motion(
        [
          { x: -60, y: 60 },
          { x: 0, y: -90 },
        ],
        [1],
      ),
      DRAWN,
      true,
    );

    for (const [knee, paw] of nearFront(posed)) {
      expect(knee.x).toBeCloseTo(30 * SCALE, 3);
      expect(paw.y).toBeCloseTo(120 * SCALE, 3);
    }
  });

  it("rejects a drawn joint that is not a finite point", () => {
    expect(() =>
      poseForDrawing(motion(), { ...DRAWN, left_knee_f: { x: Number.NaN, y: 0 } }),
    ).toThrow("left_knee_f");
  });
});
