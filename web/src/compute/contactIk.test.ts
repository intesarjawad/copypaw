import { describe, expect, it } from "vitest";
import { JOINT_IDS, type JointId } from "../joints";
import { ContactFilter, runs, twoBoneIk, type BakeMode } from "./contactIk";

type Pair = readonly [number, number];

const BASE = 0.5;
const NEAR_FRONT = 0;

function at(id: JointId): number {
  return JOINT_IDS.indexOf(id);
}

const HIP = at("left_hip_f");
const KNEE = at("left_knee_f");
const FOOT = at("left_foot_f");

function frame(hip: Pair, knee: Pair, foot: Pair): Float32Array {
  const positions = new Float32Array(2 * JOINT_IDS.length).fill(BASE);
  positions.set(hip, 2 * HIP);
  positions.set(knee, 2 * KNEE);
  positions.set(foot, 2 * FOOT);
  return positions;
}

function joint(positions: Float32Array | undefined, index: number): Pair {
  return [positions?.[2 * index] ?? Number.NaN, positions?.[2 * index + 1] ?? Number.NaN];
}

function contacts(frames: number, planted: (frame: number, paw: number) => boolean): boolean[][] {
  return Array.from({ length: frames }, (_, f) => [0, 1, 2, 3].map((paw) => planted(f, paw)));
}

/** Record every frame, then return the second, filtered pass. */
function twice(filter: ContactFilter, frames: readonly Float32Array[]): Float32Array[] {
  frames.forEach((positions, index) => filter.apply(index, positions));
  return frames.map((positions, index) => filter.apply(index, positions));
}

function length(positions: Float32Array | undefined, from: number, to: number): number {
  const [x0, y0] = joint(positions, from);
  const [x1, y1] = joint(positions, to);
  return Math.hypot(x1 - x0, y1 - y0);
}

describe("runs", () => {
  it("finds inclusive contact runs, including one at the end", () => {
    expect(runs([false, true, true, false, true])).toEqual([
      [1, 2],
      [4, 4],
    ]);
  });
});

describe("twoBoneIk", () => {
  it("pulls an unreachable target back to 0.999 of the reach", () => {
    const [knee, foot] = twoBoneIk([0, 0], [0, 1], [0, 2], [0, 3]);

    expect(foot[0]).toBeCloseTo(0);
    expect(foot[1]).toBeCloseTo(1.998);
    // cos = 0.999, so the knee sits sqrt(1 - 0.998001) to the side.
    expect(knee[0]).toBeCloseTo(-Math.sqrt(0.001999), 6);
    expect(knee[1]).toBeCloseTo(0.999, 6);
  });

  it("leaves a paw already on its target as posed", () => {
    expect(twoBoneIk([0, 0], [0, 1], [0, 2], [0, 2.001])).toEqual([
      [0, 1],
      [0, 2],
    ]);
  });
});

describe("ContactFilter", () => {
  const groundScene = [
    frame([0.15, 0.8], [0.2, 0.5], [0.2, 0.7]),
    frame([0.15, 0.8], [0.2, 0.5], [0.2, 0.3]),
    frame([0.15, 0.8], [0.2, 0.5], [0.2, 0.4]),
    frame([0.15, 0.8], [0.2, 0.5], [0.2, 0.0]),
  ];
  const nearFrontFirstThree = contacts(4, (f, paw) => paw === NEAR_FRONT && f < 3);

  it("records the first pass unchanged", () => {
    const filter = new ContactFilter(JOINT_IDS, nearFrontFirstThree, "treadmill");

    groundScene.forEach((positions, index) => {
      expect(filter.apply(index, positions)).toEqual(positions);
    });
  });

  it("settles the whole drawing until its contact paw stands on its ground", () => {
    const filter = new ContactFilter(JOINT_IDS, nearFrontFirstThree, "treadmill");

    const filtered = twice(filter, groundScene);

    // The root stays at 0.5 and the planted paw reaches down to 0.3 at its lowest.
    expect(filter.lowestGround()).toBeCloseTo(0.3, 6);
    filtered.slice(0, 3).forEach((row, f) => {
      const shift = joint(row, FOOT)[1] - joint(groundScene[f], FOOT)[1];
      expect(joint(row, FOOT)[1]).toBeCloseTo(0.3, 6);
      // Every joint moves by the same amount, so no knee bends to plant the paw.
      JOINT_IDS.forEach((_, index) => {
        expect(joint(row, index)[1] - joint(groundScene[f], index)[1]).toBeCloseTo(shift, 6);
      });
    });
    // Leaving contact, only that leg moves: its paw is halfway back to the pose.
    expect(joint(filtered[3], FOOT)[1]).toBeCloseTo(0.15, 6);
    expect(filtered[3]?.subarray(0, 2 * KNEE)).toEqual(groundScene[3]?.subarray(0, 2 * KNEE));
  });

  it("keeps a separate ground for paws drawn at different heights", () => {
    const farFoot = at("right_foot_f");
    const frames = [0, 1].map(() => {
      const positions = frame([0.2, 0.8], [0.2, 0.5], [0.2, 0.2]);
      positions.set([0.3, 0.8], 2 * at("right_hip_f"));
      positions.set([0.3, 0.6], 2 * at("right_knee_f"));
      positions.set([0.3, 0.3], 2 * farFoot);
      return positions;
    });
    const filter = new ContactFilter(
      JOINT_IDS,
      contacts(2, (_, paw) => paw < 2),
      "treadmill",
    );

    for (const row of twice(filter, frames)) {
      expect(joint(row, FOOT)[1]).toBeCloseTo(0.2, 3);
      expect(joint(row, farFoot)[1]).toBeCloseTo(0.3, 3);
    }
  });

  it.each<BakeMode>(["treadmill", "travel"])(
    "cannot pull a leg off the body with wrong contact labels in %s mode",
    (mode) => {
      const torso = at("torso");
      const frames = Array.from({ length: 40 }, (_, step) => {
        const swing = Math.sin(step / 3);
        const positions = frame(
          [0.2, 0.8],
          [0.2 + 0.2 * swing, 0.5],
          [0.2 + 0.4 * swing, 0.2 + 0.3 * swing ** 2],
        );
        // The torso stands in for the body right where this leg joins it.
        positions.set([0.2, 0.8], 2 * torso);
        return positions.map((value, index) => (index % 2 === 0 ? value + 0.05 * step : value));
      });
      // Contact flags unrelated to the motion: a fixed pattern of three in five frames.
      const labels = contacts(40, (f, paw) => (f * 7 + paw * 3) % 5 < 3);
      const filter = new ContactFilter(JOINT_IDS, labels, mode);

      twice(filter, frames).forEach((row, f) => {
        const posed = frames[f];
        const leg = length(posed, HIP, FOOT);
        const slid = [0, 1].map(
          (axis) =>
            (joint(row, HIP)[axis] ?? 0) -
            (joint(posed, HIP)[axis] ?? 0) -
            ((joint(row, torso)[axis] ?? 0) - (joint(posed, torso)[axis] ?? 0)),
        );
        expect(Math.max(...slid.map(Math.abs))).toBeLessThanOrEqual(0.15 * leg + 1e-6);
        // The leg is never stretched to reach a paw that was not really planted.
        expect(length(row, HIP, KNEE) + length(row, KNEE, FOOT)).toBeLessThanOrEqual(
          length(posed, HIP, KNEE) + length(posed, KNEE, FOOT) + 1e-4,
        );
      });
    },
  );

  it("advances in travel mode by the swing of the planted paw", () => {
    const frames = Array.from({ length: 5 }, (_, step) =>
      frame([0, 2], [0.5 - step / 4, 1], [1 - step / 2, 0]).map((value, index) =>
        index % 2 === 0 ? value + 7 : value,
      ),
    );
    const filter = new ContactFilter(
      JOINT_IDS,
      contacts(5, (_, paw) => paw === NEAR_FRONT),
      "travel",
    );

    twice(filter, frames).forEach((row, step) => {
      // The paw swings back 0.5 a frame under a root the capture left in place.
      expect(joint(row, 0)[0]).toBeCloseTo(7 + BASE + step / 2, 5);
      expect(joint(row, FOOT)[0]).toBeCloseTo(8, 5);
    });
  });

  it("leaves frames outside the contacts alone", () => {
    const filter = new ContactFilter(JOINT_IDS, nearFrontFirstThree, "treadmill");
    const outside = frame([0, 0], [0, 5], [7, 7]);

    expect(filter.apply(5, outside)).toEqual(outside);
  });

  it("requires every leg joint and the root", () => {
    const noContact = [[false, false, false, false]];
    expect(() => new ContactFilter(JOINT_IDS.slice(0, 15), noContact, "travel")).toThrow(
      "left_foot_f",
    );
    expect(() => new ContactFilter(JOINT_IDS.slice(1), noContact, "travel")).toThrow("root");
  });
});
