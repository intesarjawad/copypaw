import type { JointId, Point } from "../joints";

/** The captured motion the drawing's pose is made from. */
export interface CapturedMotion {
  /** Per frame, the 17 keypoints in video pixels. */
  readonly kp: readonly (readonly Point[])[];
  /** Per frame, whether each paw is planted: near-front, far-front, near-back, far-back. */
  readonly contacts: readonly (readonly boolean[])[];
  /** Slope and intercept of the ground line in video pixels. */
  readonly groundLinePx: readonly [number, number];
}

export type DrawnJoints = Readonly<Partial<Record<JointId, Point>>>;

type Chain = readonly [
  keypoints: readonly [number, number, number],
  contact: number,
  joints: readonly [JointId, JointId, JointId],
];

// Each chain: its three keypoints, its contact column and its three drawing joints.
const LEG_CHAINS: readonly Chain[] = [
  [[5, 6, 7], 0, ["left_hip_f", "left_knee_f", "left_foot_f"]],
  [[8, 9, 10], 1, ["right_hip_f", "right_knee_f", "right_foot_f"]],
  [[11, 12, 13], 2, ["left_hip_b", "left_knee_b", "left_foot_b"]],
  [[14, 15, 16], 3, ["right_hip_b", "right_knee_b", "right_foot_b"]],
];
// Below this sine of the knee angle (about 6 degrees) a drawn leg counts as straight.
const STRAIGHT_LEG_SINE = 0.1;
const NOSE_KP = 2;
const NECK_KP = 3;
const MEAN_DIRECTION_EPSILON = 1e-12;
// Keeps the division finite for a leg pointing sideways or up; such a leg is not lifted.
const TINY_DROP = 1e-300;

function cross(a: Point, b: Point): number {
  return a.x * b.y - a.y * b.x;
}

function minus(a: Point, b: Point): Point {
  return { x: a.x - b.x, y: a.y - b.y };
}

function plus(a: Point, b: Point, scale = 1): Point {
  return { x: a.x + scale * b.x, y: a.y + scale * b.y };
}

function norm(a: Point): number {
  return Math.hypot(a.x, a.y);
}

function turned(vector: Point, angle: number): Point {
  const [cos, sin] = [Math.cos(angle), Math.sin(angle)];
  return { x: cos * vector.x - sin * vector.y, y: sin * vector.x + cos * vector.y };
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const upper = sorted[middle] ?? Number.NaN;
  return sorted.length % 2 === 1 ? upper : ((sorted[middle - 1] ?? Number.NaN) + upper) / 2;
}

function keypoint(frame: readonly Point[], index: number): Point {
  const found = frame[index];
  if (found === undefined) {
    throw new Error(`Each frame needs 17 keypoints; keypoint ${String(index)} is missing`);
  }
  return found;
}

/** Each frame's angle from the mean direction of the vectors. */
function swing(vectors: readonly Point[]): number[] {
  const units = vectors.map((vector) => ({
    x: vector.x / norm(vector),
    y: vector.y / norm(vector),
  }));
  let mean = units.reduce((total, unit) => plus(total, unit, 1 / units.length), { x: 0, y: 0 });
  // A full turn can cancel the mean; the first frame then fixes the reference.
  if (norm(mean) < MEAN_DIRECTION_EPSILON) {
    mean = units[0] ?? mean;
  }
  return units.map((unit) => Math.atan2(cross(mean, unit), unit.x * mean.x + unit.y * mean.y));
}

/**
 * How high the paw is off the ground per frame, in leg lengths.
 *
 * A planted paw is on the ground whatever its tracked height says, and its
 * median tracked height is where the ground is for this paw.
 */
function lift(
  paw: readonly Point[],
  planted: readonly boolean[],
  [slope, intercept]: readonly [number, number],
  fullLength: number,
): number[] {
  const clearance = paw.map((point) => slope * point.x + intercept - point.y);
  const onGround = clearance.filter((_, frame) => planted[frame]);
  const level = onGround.length > 0 ? median(onGround) : Math.min(...clearance);
  return clearance.map((value, frame) =>
    planted[frame] ? 0 : Math.max(value - level, 0) / fullLength,
  );
}

/**
 * The drawn knee and foot per frame, relative to the hip.
 *
 * The leg turns about its hip by the pet leg's swing, keeping its drawn shape.
 * Off the ground its foot clears the ground by at least the pet's lift: where
 * the swing alone does not raise it enough the knee folds, up to a right
 * angle, and the leg keeps the direction of the swing.
 */
function posedLeg(
  swings: readonly number[],
  lifts: readonly number[],
  petBend: number,
  [hip, knee, foot]: readonly [Point, Point, Point],
): { knees: Point[]; feet: Point[] } | null {
  const upper = norm(minus(knee, hip));
  const lower = norm(minus(foot, knee));
  const rest = minus(foot, hip);
  const restLength = norm(rest);
  if (Math.min(upper, lower, restLength) <= 0) {
    return null;
  }
  const drawnBend = cross(minus(knee, hip), minus(foot, knee)) / (upper * lower);
  // A leg drawn straight has no bend side of its own, so it takes the pet's.
  const side =
    Math.abs(drawnBend) < STRAIGHT_LEG_SINE ? Math.sign(petBend) || 1 : Math.sign(drawnBend);
  const knees: Point[] = [];
  const feet: Point[] = [];
  swings.forEach((angle, frame) => {
    const direction = turned({ x: rest.x / restLength, y: rest.y / restLength }, angle);
    // A leg at full stretch straight down reaches the ground; the swing alone may clear it.
    // The paw is lifted along the swung leg, by folding the knee, so the leg never swings
    // further than the pet's did.
    const drop = Math.max(direction.y, TINY_DROP);
    const wanted = direction.y > 0 ? (restLength * (1 - (lifts[frame] ?? 0))) / drop : restLength;
    // Past a right angle the lower leg folds over the upper one, so the knee stops there.
    const reach = Math.min(
      Math.max(wanted, Math.min(Math.hypot(upper, lower), restLength)),
      restLength,
    );
    const normal = { x: -direction.y, y: direction.x };
    const along = (upper ** 2 + reach ** 2 - lower ** 2) / (2 * reach);
    const across = Math.sqrt(Math.max(upper ** 2 - along ** 2, 0));
    knees.push(plus({ x: along * direction.x, y: along * direction.y }, normal, -side * across));
    feet.push({ x: reach * direction.x, y: reach * direction.y });
  });
  return { knees, feet };
}

function drawnPoint(joints: DrawnJoints, id: JointId): Point | null {
  const found = joints[id];
  if (found === undefined) {
    return null;
  }
  if (!Number.isFinite(found.x) || !Number.isFinite(found.y)) {
    throw new Error(`Character joint ${id} must be a finite 2D point`);
  }
  return found;
}

/**
 * Keypoints whose legs and head move as the drawing's own would.
 *
 * Turning each drawn bone to the absolute direction of its motion bone would
 * give a drawing the pet's anatomy as well as its gait. Here every drawn leg
 * keeps its drawn shape while its paw is planted, swings from the hip by the
 * pet's swing about its average direction, and bends at the knee only to lift
 * its foot as high as the pet lifts its paw. The neck turns from its drawn
 * direction by the turn of the pet's head. Joints the drawing lacks, or drew on
 * one spot, stay as captured. With stillLegs every leg keeps its drawn pose
 * throughout, for a drawing whose leg joints are a guess.
 */
export function poseForDrawing(
  motion: CapturedMotion,
  drawn: DrawnJoints,
  stillLegs = false,
): Point[][] {
  const posed = motion.kp.map((frame) => [...frame]);
  for (const [[prox, mid, paw], column, ids] of LEG_CHAINS) {
    const [hip, knee, foot] = ids.map((id) => drawnPoint(drawn, id));
    if (!hip || !knee || !foot) {
      continue;
    }
    const chords = posed.map((frame) => minus(keypoint(frame, paw), keypoint(frame, prox)));
    const lengths = chords.map(norm);
    if (lengths.some((length) => length <= 0)) {
      continue;
    }
    const planted = motion.contacts.map((row) => row[column] ?? false);
    const plantedLengths = lengths.filter((_, frame) => planted[frame]);
    const full = Math.max(...(plantedLengths.length > 0 ? plantedLengths : lengths));
    const petBend = posed.reduce(
      (total, frame) =>
        total +
        cross(
          minus(keypoint(frame, mid), keypoint(frame, prox)),
          minus(keypoint(frame, paw), keypoint(frame, mid)),
        ),
      0,
    );
    const still = posed.map(() => 0);
    const leg = posedLeg(
      stillLegs ? still : swing(chords),
      stillLegs
        ? still
        : lift(
            posed.map((frame) => keypoint(frame, paw)),
            planted,
            motion.groundLinePx,
            full,
          ),
      petBend,
      [hip, knee, foot],
    );
    if (leg === null) {
      continue;
    }
    const scale = full / norm(minus(foot, hip));
    posed.forEach((frame, index) => {
      const root = keypoint(frame, prox);
      frame[mid] = plus(root, leg.knees[index] ?? root, scale);
      frame[paw] = plus(root, leg.feet[index] ?? root, scale);
    });
  }
  const torso = drawnPoint(drawn, "torso");
  const neck = drawnPoint(drawn, "neck");
  if (torso && neck) {
    const drawnNeck = minus(neck, torso);
    const heads = posed.map((frame) => minus(keypoint(frame, NOSE_KP), keypoint(frame, NECK_KP)));
    const length = norm(drawnNeck);
    if (length > 0 && heads.every((head) => norm(head) > 0)) {
      const reach = median(heads.map(norm));
      const unit = { x: drawnNeck.x / length, y: drawnNeck.y / length };
      swing(heads).forEach((angle, index) => {
        const frame = posed[index];
        if (frame) {
          frame[NOSE_KP] = plus(keypoint(frame, NECK_KP), turned(unit, angle), reach);
        }
      });
    }
  }
  return posed;
}
