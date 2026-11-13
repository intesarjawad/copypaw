import type { ContactFilter } from "./contactIk";

/** A tail's drawn base and tip as [x, y] pairs, in the positions' units. */
export type TailPair = readonly [base: readonly [number, number], tip: readonly [number, number]];

// The mesh is deformed as rigidly as its handles allow. A handle at each joint
// alone lets a limb curve along its whole length like a hose. Each leg bone
// therefore carries handles along it, and a pair either side of it near each
// end, which turn with the bone and hold its width, so the leg stays a
// straight drawn limb and the bend stays at the joint.
const LEG_BONES: readonly (readonly [string, string])[] = ["left", "right"].flatMap((side) =>
  ["f", "b"].flatMap((end) => [
    [`${side}_hip_${end}`, `${side}_knee_${end}`] as const,
    [`${side}_knee_${end}`, `${side}_foot_${end}`] as const,
  ]),
);
// Along the bone, then across it, in bone lengths. A tenth of the length either
// side stays inside all but stick-thin legs, whose side handles the solver ignores.
const BONE_HANDLES: readonly (readonly [number, number])[] = [
  [0.25, 0],
  [0.25, 0.1],
  [0.25, -0.1],
  [0.5, 0],
  [0.75, 0],
  [0.75, 0.1],
  [0.75, -0.1],
];
// The neck joint sits 70% of the way from the torso to the nose. A handle half
// of the rest of the way on turns the head with the neck instead of leaving it behind.
const HEAD_REACH = (0.5 * (1 - 0.7)) / 0.7;
// A tail turns this share of the hind legs' swing, and takes about this long to follow.
const TAIL_SWAY = 0.3;
const TAIL_LAG_S = 0.2;
// A tail drawn down to the paws' level is held this far off the ground, or it reads as a fifth leg.
const TAIL_LIFT = 0.6;

function x(positions: Float32Array, row: number): number {
  return positions[2 * row] ?? Number.NaN;
}

function y(positions: Float32Array, row: number): number {
  return positions[2 * row + 1] ?? Number.NaN;
}

/**
 * The joint positions followed by the handles of each leg bone and the head.
 *
 * Bones whose joints are not both named are left out, so the same names always
 * give the same handles in the same order. A handle that falls outside the
 * drawing is ignored by the mesh solver. Positions are interleaved x, y.
 */
export function withBoneHandles(names: readonly string[], positions: Float32Array): Float32Array {
  const index = new Map(names.map((name, row) => [name, row]));
  if (positions.length !== 2 * index.size) {
    throw new Error(
      `positions has ${String(positions.length / 2)} joints but ${String(index.size)} names were given`,
    );
  }
  const handles: number[] = [];
  for (const [upperName, lowerName] of LEG_BONES) {
    const upper = index.get(upperName);
    const lower = index.get(lowerName);
    if (upper === undefined || lower === undefined) {
      continue;
    }
    const boneX = x(positions, lower) - x(positions, upper);
    const boneY = y(positions, lower) - y(positions, upper);
    for (const [along, aside] of BONE_HANDLES) {
      handles.push(
        x(positions, upper) + along * boneX - aside * boneY,
        y(positions, upper) + along * boneY + aside * boneX,
      );
    }
  }
  const torso = index.get("torso");
  const neck = index.get("neck");
  if (torso !== undefined && neck !== undefined) {
    handles.push(
      x(positions, neck) + HEAD_REACH * (x(positions, neck) - x(positions, torso)),
      y(positions, neck) + HEAD_REACH * (y(positions, neck) - y(positions, torso)),
    );
  }
  return Float32Array.from([...positions, ...handles]);
}

/**
 * A base and a tip handle for each drawn tail, interleaved x, y.
 *
 * The base moves with the body and the tail turns about it by turn radians,
 * keeping its length. behind gives, per tail, the x its tip must not pass on
 * the way to the hind legs; the tail swings away from them instead. A tip that
 * would pass through the ground the paws stand on stays on it and the tail
 * swings out about its base, so the tip rests on the ground without being planted.
 * lift, per tail, raises it by that many radians away from the legs.
 */
export function tailHandles(
  tails: readonly TailPair[],
  bodyShift: readonly [number, number],
  ground: number | null,
  turn = 0,
  behind: readonly number[] | null = null,
  lift: readonly number[] | null = null,
): Float32Array {
  return Float32Array.from(
    tails.flatMap(([[baseX0, baseY0], [tipX, tipY]], tail) => {
      const base = [baseX0 + bodyShift[0], baseY0 + bodyShift[1]] as const;
      const drawn = [tipX - baseX0, tipY - baseY0] as const;
      const length = Math.hypot(...drawn);
      const side = drawn[0] < 0 ? -1 : 1;
      const angle = turn + side * (lift?.[tail] ?? 0);
      const [cos, sin] = [Math.cos(angle), Math.sin(angle)];
      let reachX = cos * drawn[0] - sin * drawn[1];
      let reachY = sin * drawn[0] + cos * drawn[1];
      const limit = behind?.[tail];
      if (limit !== undefined) {
        // Measured away from the legs: the tip stays at least as far out as the rearmost leg.
        const out = Math.min(Math.max(reachX * side, (limit - base[0]) * side, -length), length);
        if (out > reachX * side) {
          reachX = out * side;
          reachY = -Math.sqrt(Math.max(length ** 2 - out ** 2, 0));
        }
      }
      if (ground !== null && base[1] + reachY < ground) {
        const drop = Math.min(ground - base[1], 0);
        reachX = side * Math.sqrt(Math.max(length ** 2 - drop ** 2, 0));
        reachY = drop;
      }
      return [base[0], base[1], base[0] + reachX, base[1] + reachY];
    }),
  );
}

/**
 * Turns a frame's joints into the mesh handles: stand on the contact paws, then add the extras.
 *
 * The captured keypoints hold only the root of the tail, so a tail has no
 * motion of its own to copy. It turns about its base a share of the way the
 * hind legs swing, lagging behind them, and its tip keeps clear of the
 * rearmost hind leg. Frames must arrive in order, as the engine plays them.
 */
export class HandleFilter {
  private readonly root: number;
  private readonly hind: readonly (readonly [number, number])[];
  private readonly follow: number;
  private readonly lift: readonly number[];
  private readonly tailTurn = new Map<number, number>();

  constructor(
    private readonly names: readonly string[],
    private readonly drawn: Float32Array,
    private readonly tails: readonly TailPair[],
    frameTimeS: number,
    private readonly contactFilter: ContactFilter | null,
  ) {
    this.root = names.indexOf("root");
    if (this.root < 0) {
      throw new Error("Joint names must include root");
    }
    this.hind = (["left", "right"] as const).flatMap((side) => {
      const hip = names.indexOf(`${side}_hip_b`);
      const foot = names.indexOf(`${side}_foot_b`);
      return hip >= 0 && foot >= 0 ? [[hip, foot] as const] : [];
    });
    this.follow = 1 - Math.exp(-frameTimeS / TAIL_LAG_S);
    const paws = Math.max(...this.hind.map(([, foot]) => y(drawn, foot)));
    this.lift = tails.map(([, [, tipY]]) => (tipY <= paws ? TAIL_LIFT : 0));
  }

  /** Every handle where the drawing has it, which is the pose the mesh rests in. */
  drawnPose(): Float32Array {
    return Float32Array.from([
      ...withBoneHandles(this.names, this.drawn),
      ...tailHandles(this.tails, [0, 0], null),
    ]);
  }

  /** The mesh handles for one frame of engine joint positions. */
  apply(frame: number, positions: Float32Array): Float32Array {
    if (this.contactFilter === null) {
      return this.withHandles(frame, positions, null);
    }
    const stood = this.contactFilter.apply(frame, positions);
    return this.withHandles(frame, stood, this.contactFilter.lowestGround());
  }

  /** How far the hind legs have turned from their drawn direction, in radians. */
  private hindSwing(joints: Float32Array): number {
    const turns = this.hind.map(([hip, foot]) => {
      const drawnX = x(this.drawn, foot) - x(this.drawn, hip);
      const drawnY = y(this.drawn, foot) - y(this.drawn, hip);
      const posedX = x(joints, foot) - x(joints, hip);
      const posedY = y(joints, foot) - y(joints, hip);
      return Math.atan2(drawnX * posedY - drawnY * posedX, drawnX * posedX + drawnY * posedY);
    });
    return turns.length > 0 ? turns.reduce((total, turn) => total + turn, 0) / turns.length : 0;
  }

  private withHandles(frame: number, joints: Float32Array, ground: number | null): Float32Array {
    const wanted = TAIL_SWAY * this.hindSwing(joints);
    const previous = this.tailTurn.get(frame - 1) ?? wanted;
    const turn = previous + this.follow * (wanted - previous);
    this.tailTurn.set(frame, turn);
    let behind: number[] | null = null;
    if (this.hind.length > 0 && this.tails.length > 0) {
      const hips =
        this.hind.reduce((total, [hip]) => total + x(this.drawn, hip), 0) / this.hind.length;
      const feet = this.hind.map(([, foot]) => x(joints, foot));
      behind = this.tails.map(([[baseX]]) =>
        baseX < hips ? Math.min(...feet) : Math.max(...feet),
      );
    }
    const shift = [
      x(joints, this.root) - x(this.drawn, this.root),
      y(joints, this.root) - y(this.drawn, this.root),
    ] as const;
    return Float32Array.from([
      ...withBoneHandles(this.names, joints),
      ...tailHandles(this.tails, shift, ground, turn, behind, this.lift),
    ]);
  }
}
