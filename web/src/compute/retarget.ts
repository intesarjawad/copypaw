import { JOINT_IDS, type JointId, type Point } from "../joints";

/** Where each rig joint hangs, as in the upstream four-legged template. */
export const JOINT_PARENTS: Readonly<Record<JointId, JointId | null>> = {
  root: null,
  hip: "root",
  torso: "hip",
  neck: "torso",
  right_hip_b: "root",
  right_knee_b: "right_hip_b",
  right_foot_b: "right_knee_b",
  left_hip_b: "root",
  left_knee_b: "left_hip_b",
  left_foot_b: "left_knee_b",
  right_hip_f: "root",
  right_knee_f: "right_hip_f",
  right_foot_f: "right_knee_f",
  left_hip_f: "root",
  left_knee_f: "left_hip_f",
  left_foot_f: "left_knee_f",
};

/** A keypoint bone, from its proximal to its distal keypoint. */
type Bone = readonly [prox: number, dist: number];

const ROOT_KP = 4;
const NECK_KP = 3;
const NOSE: Bone = [3, 2];
const NEAR_ELBOW: Bone = [5, 6];
const NEAR_F_PAW: Bone = [6, 7];
const FAR_ELBOW: Bone = [8, 9];
const FAR_F_PAW: Bone = [9, 10];
const NEAR_KNEE: Bone = [11, 12];
const NEAR_B_PAW: Bone = [12, 13];
const FAR_KNEE: Bone = [14, 15];
const FAR_B_PAW: Bone = [15, 16];
/**
 * The motion's leg bones whose length is set against the drawing's four legs. A side view with
 * two legs drives the far pair from the near limbs, as the engine config does.
 */
function motionLegBones(legs: 2 | 4): readonly Bone[] {
  return legs === 2
    ? [NEAR_ELBOW, NEAR_F_PAW, NEAR_ELBOW, NEAR_F_PAW, NEAR_KNEE, NEAR_B_PAW, NEAR_KNEE, NEAR_B_PAW]
    : [NEAR_ELBOW, NEAR_F_PAW, FAR_ELBOW, FAR_F_PAW, NEAR_KNEE, NEAR_B_PAW, FAR_KNEE, FAR_B_PAW];
}
const DRAWING_LEGS: readonly (readonly [JointId, JointId, JointId])[] = [
  ["left_hip_f", "left_knee_f", "left_foot_f"],
  ["right_hip_f", "right_knee_f", "right_foot_f"],
  ["left_hip_b", "left_knee_b", "left_foot_b"],
  ["right_hip_b", "right_knee_b", "right_foot_b"],
];
const FULL_TURN_DEG = 360;
const RIGHT_ANGLE_RAD = Math.PI / 2;

/**
 * The motion bone that turns each drawing bone, named by the bone's distal joint.
 * A two-legged side view drives both legs of a pair from the near limb.
 */
function boneMapping(legs: 2 | 4): readonly (readonly [JointId, Bone])[] {
  const two = legs === 2;
  return [
    ["neck", NOSE],
    ["left_knee_f", NEAR_ELBOW],
    ["left_foot_f", NEAR_F_PAW],
    ["left_knee_b", NEAR_KNEE],
    ["left_foot_b", NEAR_B_PAW],
    ["right_knee_f", two ? NEAR_ELBOW : FAR_ELBOW],
    ["right_foot_f", two ? NEAR_F_PAW : FAR_F_PAW],
    ["right_knee_b", two ? NEAR_KNEE : FAR_KNEE],
    ["right_foot_b", two ? NEAR_B_PAW : FAR_B_PAW],
  ];
}

function keypoint(frame: readonly Point[], index: number): Point {
  const point = frame[index];
  if (point === undefined) {
    throw new RangeError(`Keypoint ${String(index)} is missing`);
  }
  return point;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const upper = sorted[middle] ?? Number.NaN;
  return sorted.length % 2 === 1 ? upper : ((sorted[middle - 1] ?? Number.NaN) + upper) / 2;
}

/** Degrees counter-clockwise from +y, in [0, 360), as the engine measures bones. */
function orientationDeg(x: number, y: number): number {
  const degrees = (((Math.atan2(y, x) - RIGHT_ANGLE_RAD) * 180) / Math.PI) % FULL_TURN_DEG;
  return degrees < 0 ? degrees + FULL_TURN_DEG : degrees;
}

function at(joints: Float32Array, index: number): readonly [number, number] {
  return [joints[2 * index] ?? Number.NaN, joints[2 * index + 1] ?? Number.NaN];
}

function jointAt(joints: Float32Array, id: JointId): readonly [number, number] {
  return at(joints, JOINT_IDS.indexOf(id));
}

/**
 * The drawing's joints as the engine places them: fractions of the drawing's
 * longer side, y up from its bottom edge, interleaved x, y in JOINT_IDS order.
 */
export function drawnJoints(
  pixels: Readonly<Record<JointId, Point>>,
  height: number,
  size: number,
): Float32Array {
  return Float32Array.from(
    JOINT_IDS.flatMap((id) => {
      const { x, y } = pixels[id];
      return [x / size, 1 - (y / size + (1 - height / size))];
    }),
  );
}

/**
 * The drawing's joints on every frame of a posed motion, ported from the
 * engine's retargeter and rig for the planar motions copypaw writes.
 *
 * Each mapped drawing bone turns from its drawn direction to its motion bone's
 * direction; the root follows the motion's root, scaled by the drawing's leg
 * length over the motion's. rootX is the motion's travel in body lengths, zero
 * on a treadmill, and startX where the root starts. Positions are interleaved
 * x, y per joint in JOINT_IDS order.
 */
export function retarget(
  posed: readonly (readonly Point[])[],
  rootX: readonly number[],
  drawn: Float32Array,
  legs: 2 | 4,
  startX = 0,
): Float32Array[] {
  const first = posed[0];
  if (first === undefined || rootX.length !== posed.length) {
    throw new Error("Retargeting needs a nonempty motion with one root position per frame");
  }
  const bodyLength = median(
    posed.map((frame) => {
      const [neck, root] = [keypoint(frame, NECK_KP), keypoint(frame, ROOT_KP)];
      return Math.hypot(neck.x - root.x, neck.y - root.y);
    }),
  );
  const motionLegs = motionLegBones(legs).reduce(
    (total, [prox, dist]) =>
      total +
      median(
        posed.map((frame) => {
          const [a, b] = [keypoint(frame, prox), keypoint(frame, dist)];
          return Math.hypot(b.x - a.x, b.y - a.y);
        }),
      ) /
        bodyLength,
    0,
  );
  const drawingLegs = DRAWING_LEGS.reduce((total, ids) => {
    const [hip, knee, foot] = ids.map((id) => jointAt(drawn, id));
    return hip && knee && foot
      ? total +
          Math.hypot(knee[0] - foot[0], knee[1] - foot[1]) +
          Math.hypot(hip[0] - knee[0], hip[1] - knee[1])
      : total;
  }, 0);
  const scale = drawingLegs / motionLegs;

  const offsets = JOINT_IDS.map((id) => {
    const parent = JOINT_PARENTS[id];
    const [x, y] = jointAt(drawn, id);
    const [px, py] = parent === null ? [x, y] : jointAt(drawn, parent);
    return [x - px, y - py] as const;
  });
  const mapping = boneMapping(legs);
  const startDeg = new Map(
    mapping.map(([id]) => {
      const [x, y] = offsets[JOINT_IDS.indexOf(id)] ?? [0, 0];
      return [id, orientationDeg(x, y)] as const;
    }),
  );

  const firstRootY = -keypoint(first, ROOT_KP).y / bodyLength;
  return posed.map((frame, index) => {
    // A drawing joint's turn from its drawn direction, and the turn its parent hands down.
    const turn = new Map<JointId, number>();
    const local = new Map<JointId, number>();
    for (const [id, [prox, dist]] of mapping) {
      const [a, b] = [keypoint(frame, prox), keypoint(frame, dist)];
      const angle =
        (orientationDeg(b.x - a.x, a.y - b.y) - (startDeg.get(id) ?? 0)) * (Math.PI / 180);
      turn.set(id, angle);
      const parent = JOINT_PARENTS[id];
      if (parent !== null) {
        local.set(parent, angle - (turn.get(parent) ?? 0));
      }
    }
    const rootY = -keypoint(frame, ROOT_KP).y / bodyLength;
    const positions = new Float32Array(2 * JOINT_IDS.length);
    const world = new Map<JointId, readonly [x: number, y: number, angle: number]>();
    JOINT_IDS.forEach((id, joint) => {
      const parent = JOINT_PARENTS[id];
      let placed: readonly [number, number, number];
      if (parent === null) {
        placed = [
          startX + scale * ((rootX[index] ?? 0) - (rootX[0] ?? 0)),
          scale * (rootY - firstRootY),
          local.get(id) ?? 0,
        ];
      } else {
        const [px, py, parentAngle] = world.get(parent) ?? [0, 0, 0];
        const [ox, oy] = offsets[joint] ?? [0, 0];
        const [cos, sin] = [Math.cos(parentAngle), Math.sin(parentAngle)];
        placed = [
          px + cos * ox - sin * oy,
          py + sin * ox + cos * oy,
          parentAngle + (local.get(id) ?? 0),
        ];
      }
      world.set(id, placed);
      positions[2 * joint] = placed[0];
      positions[2 * joint + 1] = placed[1];
    });
    return positions;
  });
}
