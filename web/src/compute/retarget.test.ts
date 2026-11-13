import { describe, expect, it } from "vitest";
import { JOINT_IDS, jointsFrom, type Point } from "../joints";
import { drawnJoints, retarget } from "./retarget";

// A drawing standing square: body along x, every leg straight down, neck up and forward.
const BODY: Readonly<Record<string, Point>> = {
  root: { x: 100, y: 200 },
  hip: { x: 100, y: 200 },
  torso: { x: 300, y: 200 },
  neck: { x: 350, y: 150 },
};
const DRAWN = jointsFrom((id) => {
  const x = id.endsWith("_b") ? 100 : 300;
  const y = id.includes("knee") ? 250 : id.includes("foot") ? 300 : 200;
  return BODY[id] ?? { x, y };
});

/** Seventeen keypoints matching the drawing: every bone points the way the drawing's does. */
function keypoints(rootY: number, footTurn = 0): Point[] {
  const down = { x: Math.sin(footTurn), y: Math.cos(footTurn) };
  const leg = (hip: Point): Point[] => [
    hip,
    { x: hip.x, y: hip.y + 10 },
    { x: hip.x + 10 * down.x, y: hip.y + 10 + 10 * down.y },
  ];
  const root = { x: 0, y: rootY };
  const neck = { x: 40, y: rootY };
  return [
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 50, y: rootY - 10 },
    neck,
    root,
    ...leg(neck),
    ...leg(neck),
    ...leg(root),
    ...leg(root),
  ];
}

const SIZE = 400;
const drawn = drawnJoints(DRAWN, SIZE, SIZE);

function at(joints: Float32Array | undefined, id: (typeof JOINT_IDS)[number]): [number, number] {
  const index = JOINT_IDS.indexOf(id);
  return [joints?.[2 * index] ?? Number.NaN, joints?.[2 * index + 1] ?? Number.NaN];
}

describe("drawnJoints", () => {
  it("measures joints from the drawing's bottom left in fractions of its longer side", () => {
    expect(at(drawn, "neck")).toEqual([350 / SIZE, 250 / SIZE]);
  });
});

describe("retarget", () => {
  it("keeps the drawn pose, rooted at the origin, when the motion matches it", () => {
    const [frame] = retarget([keypoints(100)], [0], drawn, 4);

    for (const id of JOINT_IDS) {
      const [x, y] = at(drawn, id);
      expect(at(frame, id)[0]).toBeCloseTo(x - (drawn[0] ?? 0), 6);
      expect(at(frame, id)[1]).toBeCloseTo(y - (drawn[1] ?? 0), 6);
    }
  });

  it("turns a paw by the motion's turn, about its knee", () => {
    const [, frame] = retarget([keypoints(100), keypoints(100, Math.PI / 2)], [0, 0], drawn, 4);
    const [kneeX, kneeY] = at(frame, "left_knee_b");
    const [footX, footY] = at(frame, "left_foot_b");

    // A paw turned a quarter toward +x in image coordinates points along +x.
    expect(footX - kneeX).toBeCloseTo(50 / SIZE, 6);
    expect(footY - kneeY).toBeCloseTo(0, 6);
  });

  it("raises the root by the motion's rise, scaled from its legs to the drawing's", () => {
    const frames = retarget([keypoints(100), keypoints(90)], [0, 0], drawn, 4);

    // The motion's four legs sum to two body lengths (eight 10 px bones, a 40 px body) and the
    // drawing's legs to one side of it, so rising a quarter body raises the root an eighth side.
    expect(at(frames[1], "root")[1]).toBeCloseTo(0.125, 6);
  });

  it("rejects a motion without one root position per frame", () => {
    expect(() => retarget([keypoints(100)], [], drawn, 4)).toThrow("root position");
  });
});
