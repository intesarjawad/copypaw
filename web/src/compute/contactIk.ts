export type BakeMode = "treadmill" | "travel";

// Contact order follows the motion file: near-front, far-front, near-back, far-back.
const PAW_JOINTS = [
  ["left_hip_f", "left_knee_f", "left_foot_f"],
  ["right_hip_f", "right_knee_f", "right_foot_f"],
  ["left_hip_b", "left_knee_b", "left_foot_b"],
  ["right_hip_b", "right_knee_b", "right_foot_b"],
] as const;
const END_HIPS = [
  ["left_hip_f", "right_hip_f"],
  ["left_hip_b", "right_hip_b"],
] as const;
const ROOT = "root";
// A hip slides at most this share of its leg's posed length away from the body.
const MAX_HIP_SLIDE = 0.15;
const FULL_REACH = 0.999;

type Vec = readonly [number, number];

/** Inclusive first and last frames of each true run. */
export function runs(mask: readonly boolean[]): [number, number][] {
  const found: [number, number][] = [];
  let start: number | null = null;
  for (const [frame, value] of mask.entries()) {
    if (value && start === null) {
      start = frame;
    } else if (!value && start !== null) {
      found.push([start, frame - 1]);
      start = null;
    }
  }
  if (start !== null) {
    found.push([start, mask.length - 1]);
  }
  return found;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const upper = sorted[middle] ?? Number.NaN;
  return sorted.length % 2 === 1 ? upper : ((sorted[middle - 1] ?? Number.NaN) + upper) / 2;
}

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function clamp(value: number, limit: number): number {
  return Math.min(Math.max(value, -limit), limit);
}

/**
 * Return knee and foot positions after solving toward target.
 *
 * Segment lengths come from the unfiltered pose. Targets beyond 0.999 of the
 * full reach are pulled back along the hip-to-target line. The knee keeps the
 * side it had in the unfiltered pose. A paw within that margin of its target
 * is left as posed.
 */
export function twoBoneIk(hip: Vec, knee: Vec, foot: Vec, target: Vec): [Vec, Vec] {
  const upper = Math.hypot(knee[0] - hip[0], knee[1] - hip[1]);
  const lower = Math.hypot(foot[0] - knee[0], foot[1] - knee[1]);
  if (upper <= 0 || lower <= 0) {
    return [knee, foot];
  }
  // A paw already on its target keeps its pose; solving would pull a straight leg short.
  if (Math.hypot(target[0] - foot[0], target[1] - foot[1]) <= (1 - FULL_REACH) * (upper + lower)) {
    return [knee, foot];
  }
  let offset: Vec = [target[0] - hip[0], target[1] - hip[1]];
  let distance = Math.hypot(...offset);
  if (distance <= 0) {
    offset = [foot[0] - hip[0], foot[1] - hip[1]];
    distance = Math.hypot(...offset);
    if (distance <= 0) {
      return [knee, foot];
    }
  }
  const direction: Vec = [offset[0] / distance, offset[1] / distance];
  distance = Math.min(Math.max(distance, Math.abs(upper - lower)), FULL_REACH * (upper + lower));
  const solvedFoot: Vec = [hip[0] + direction[0] * distance, hip[1] + direction[1] * distance];

  const cosine = (upper * upper + distance * distance - lower * lower) / (2 * upper * distance);
  const angle = Math.acos(Math.min(Math.max(cosine, -1), 1));
  const perp: Vec = [-direction[1], direction[0]];
  const bend = (knee[0] - hip[0]) * (foot[1] - knee[1]) - (knee[1] - hip[1]) * (foot[0] - knee[0]);
  const side = bend > 0 ? -1 : 1;
  const along = upper * Math.cos(angle);
  const across = side * upper * Math.sin(angle);
  const solvedKnee: Vec = [
    hip[0] + along * direction[0] + across * perp[0],
    hip[1] + along * direction[1] + across * perp[1],
  ];
  return [solvedKnee, solvedFoot];
}

/**
 * Stands a drawing on its contact paws in world space.
 *
 * The first pass records positions without changing them, so each paw's
 * ground can be the lowest it reaches under the root while planted. Later
 * passes lower, raise and tip the whole drawing so its contact paws stand on
 * their ground, solve each contact leg toward it and blend halfway into and
 * out of each contact run. Positions are interleaved x, y per joint, in the
 * order of the joint names.
 */
export class ContactFilter {
  private readonly frames: number;
  private readonly joints = new Map<string, number>();
  private readonly root: number;
  private readonly legs: readonly (readonly [number, number, number])[];
  private readonly recorded: (Float32Array | null)[];
  private readonly contactFrames: readonly number[];
  private readonly runOnset: readonly Int32Array[];
  private readonly blendOnset: readonly Int32Array[];
  private ground: readonly (number | null)[] | null = null;
  private carried: Float64Array;

  constructor(
    jointNames: readonly string[],
    private readonly contacts: readonly (readonly boolean[])[],
    private readonly mode: BakeMode,
  ) {
    this.frames = contacts.length;
    if (contacts.some((row) => row.length !== PAW_JOINTS.length)) {
      throw new Error(`contacts must have ${String(PAW_JOINTS.length)} paws per frame`);
    }
    jointNames.forEach((name, index) => {
      if (this.joints.has(name)) {
        throw new Error(`Duplicate joint name ${name}`);
      }
      this.joints.set(name, index);
    });
    this.root = this.indexOf(ROOT);
    this.legs = PAW_JOINTS.map(([hip, knee, foot]) => [
      this.indexOf(hip),
      this.indexOf(knee),
      this.indexOf(foot),
    ]);
    this.recorded = new Array<Float32Array | null>(this.frames).fill(null);
    this.contactFrames = contacts.flatMap((row, frame) => (row.some(Boolean) ? [frame] : []));
    this.runOnset = this.legs.map(() => new Int32Array(this.frames).fill(-1));
    this.blendOnset = this.legs.map(() => new Int32Array(this.frames).fill(-1));
    this.legs.forEach((_, paw) => {
      const onsets = this.runOnset[paw] ?? new Int32Array(0);
      const blends = this.blendOnset[paw] ?? new Int32Array(0);
      for (const [start, end] of runs(contacts.map((row) => row[paw] ?? false))) {
        onsets.fill(start, start, end + 1);
        if (start > 0) {
          blends[start - 1] = start;
        }
        if (end + 1 < this.frames) {
          blends[end + 1] = start;
        }
      }
    });
    this.carried = new Float64Array(this.frames);
  }

  /** The lowest paw's ground height, once the first pass has found it. */
  lowestGround(): number | null {
    const levels = (this.ground ?? []).filter((level) => level !== null);
    return levels.length > 0 ? Math.min(...levels) : null;
  }

  /** World positions for one frame, filtered after the first pass. */
  apply(frame: number, positions: Float32Array): Float32Array {
    if (positions.length !== 2 * this.joints.size) {
      throw new Error(
        `positions has ${String(positions.length / 2)} joints but ${String(this.joints.size)} names were given`,
      );
    }
    if (frame < 0 || frame >= this.frames) {
      return Float32Array.from(positions);
    }
    if (!this.recorded[frame]) {
      this.recorded[frame] = Float32Array.from(positions);
      return Float32Array.from(positions);
    }
    if (this.ground === null) {
      const ground = this.groundLevels();
      if (ground === null) {
        return Float32Array.from(positions);
      }
      this.ground = ground;
      this.carried = this.carry();
    }
    return this.filter(frame, positions, this.ground);
  }

  private indexOf(name: string): number {
    const index = this.joints.get(name);
    if (index === undefined) {
      throw new Error(`Joint names must include ${name}`);
    }
    return index;
  }

  private contact(frame: number, paw: number): boolean {
    return this.contacts[frame]?.[paw] ?? false;
  }

  private leg(paw: number): readonly [number, number, number] {
    return this.legs[paw] ?? [0, 0, 0];
  }

  /**
   * Each paw's ground height once every contact frame is recorded.
   *
   * A paw's ground is as far below the root as the paw reaches while planted,
   * measured from the median root height. Paws drawn at different heights
   * therefore keep their own ground lines.
   */
  private groundLevels(): (number | null)[] | null {
    const rows = this.contactFrames.map((frame) => this.recorded[frame] ?? null);
    if (rows.some((row) => row === null)) {
      return null;
    }
    const root = this.root;
    const rootHeights = this.recorded.flatMap((row) => (row ? [y(row, root)] : []));
    return this.legs.map(([, , foot], paw) => {
      const drops = this.contactFrames.flatMap((frame, index) => {
        const row = rows[index];
        return row && this.contact(frame, paw) ? [Math.fround(y(row, foot) - y(row, root))] : [];
      });
      return drops.length > 0 ? median(rootHeights) + Math.min(...drops) : null;
    });
  }

  /**
   * How far each frame moves sideways so the drawing travels on its own paws.
   *
   * The captured travel is the pet's. A drawing with other leg proportions
   * covers a different distance per step, so between frames it advances by as
   * much as its planted paws swing back under it, and coasts at the last such
   * advance while no paw is planted. Treadmill mode stays put.
   */
  private carry(): Float64Array {
    const carried = new Float64Array(this.frames);
    const rows = this.recorded;
    const first = rows[0];
    if (this.mode === "treadmill" || !first || rows.some((row) => row === null)) {
      return carried;
    }
    const root = this.root;
    let own = 0;
    let advance = 0;
    for (let frame = 1; frame < this.frames; frame++) {
      const before = rows[frame - 1];
      const now = rows[frame];
      if (!before || !now) {
        return carried;
      }
      const swings = this.legs.flatMap(([, , foot], paw) =>
        this.contact(frame - 1, paw) && this.contact(frame, paw)
          ? [
              Math.fround(x(now, foot) - x(now, root)) -
                Math.fround(x(before, foot) - x(before, root)),
            ]
          : [],
      );
      if (swings.length > 0) {
        advance = -mean(swings);
      }
      own += advance;
      carried[frame] = own - Math.fround(x(now, root) - x(first, root));
    }
    return carried;
  }

  private targetX(paw: number, onset: number, positions: Float32Array): number {
    const foot = this.leg(paw)[2];
    const row = this.recorded[onset];
    if (this.mode === "treadmill" || !row) {
      return x(positions, foot);
    }
    return x(row, foot) + (this.carried[onset] ?? 0);
  }

  /**
   * Solve the leg toward the ground where its paw was planted.
   *
   * The leg has already slid that way as far as its hip allows. If the spot is
   * still out of reach the paw slips along the ground under the leg; reaching
   * back for it would lay the leg flat.
   */
  private solveLeg(
    paw: number,
    onset: number,
    positions: Float32Array,
    ground: number,
  ): [Vec, Vec] {
    const [hipJoint, kneeJoint, footJoint] = this.leg(paw);
    const hip = point(positions, hipJoint);
    const knee = point(positions, kneeJoint);
    const foot = point(positions, footJoint);
    let planted: Vec = [this.targetX(paw, onset, positions), ground];
    const length =
      Math.hypot(knee[0] - hip[0], knee[1] - hip[1]) +
      Math.hypot(foot[0] - knee[0], foot[1] - knee[1]);
    if (Math.hypot(planted[0] - hip[0], planted[1] - hip[1]) > FULL_REACH * length) {
      planted = [foot[0], ground];
    }
    return twoBoneIk(hip, knee, foot, planted);
  }

  private slideLimit(paw: number, positions: Float32Array): number {
    const [hip, , foot] = this.leg(paw);
    return (
      MAX_HIP_SLIDE *
      Math.hypot(x(positions, foot) - x(positions, hip), y(positions, foot) - y(positions, hip))
    );
  }

  /**
   * Lower, raise and tip the whole drawing until its contact paws stand on their ground.
   *
   * Each end of the body takes the average rise its contact legs ask for, and
   * the body tips between its front and back hips when both ends have one.
   * Each contact leg then slides at the hip by what is left, as a shoulder
   * does, so it stands where it is planted in its posed shape instead of
   * floating or bending. The slide is limited to a share of the leg's length,
   * so a paw wrongly marked as planted cannot pull its leg off the body. A leg
   * entering or leaving contact slides halfway.
   */
  private settled(
    frame: number,
    positions: Float32Array,
    ground: readonly (number | null)[],
  ): Float32Array {
    const slides = new Map<number, readonly [across: number, rise: number, planted: boolean]>();
    ground.forEach((level, paw) => {
      let onset = this.runOnset[paw]?.[frame] ?? -1;
      const planted = onset >= 0;
      if (!planted) {
        onset = this.blendOnset[paw]?.[frame] ?? -1;
      }
      if (level !== null && onset >= 0) {
        const foot = this.leg(paw)[2];
        slides.set(paw, [
          this.targetX(paw, onset, positions) - x(positions, foot),
          level - y(positions, foot),
          planted,
        ]);
      }
    });
    const endRise = (end: number): number | null => {
      const rises = [...slides].flatMap(([paw, [, rise, planted]]) =>
        planted && Math.floor(paw / 2) === end ? [rise] : [],
      );
      return rises.length > 0 ? mean(rises) : null;
    };
    const front = endRise(0);
    const back = endRise(1);
    const settled = Float32Array.from(positions);
    if (front === null && back === null) {
      return settled;
    }
    const endX = (names: readonly string[]): number =>
      mean(names.map((name) => x(positions, this.indexOf(name))));
    const frontX = endX(END_HIPS[0]);
    const backX = endX(END_HIPS[1]);
    const count = positions.length / 2;
    const body = new Float64Array(count);
    if (front === null || back === null || frontX === backX) {
      body.fill(front === null ? (back ?? 0) : back === null ? front : (front + back) / 2);
    } else {
      for (let joint = 0; joint < count; joint++) {
        body[joint] = back + ((front - back) * (x(positions, joint) - backX)) / (frontX - backX);
      }
    }
    const aside = new Float64Array(count);
    this.legs.forEach(([hip, knee, foot], paw) => {
      const slide = slides.get(paw);
      if (slide) {
        const [across, rise, planted] = slide;
        const weight = planted ? 1 : 0.5;
        // A hip that slid further would leave the body; the solver takes the rest.
        const limit = this.slideLimit(paw, positions);
        body[hip] = (body[hip] ?? 0) + clamp(weight * (rise - (body[hip] ?? 0)), limit);
        const sideways = clamp(weight * across, limit);
        aside[hip] = sideways;
        aside[knee] = sideways;
        aside[foot] = sideways;
      }
      body[knee] = body[hip] ?? 0;
      body[foot] = body[hip] ?? 0;
    });
    for (let joint = 0; joint < count; joint++) {
      settled[2 * joint] = x(settled, joint) + Math.fround(aside[joint] ?? 0);
      settled[2 * joint + 1] = y(settled, joint) + Math.fround(body[joint] ?? 0);
    }
    return settled;
  }

  private filter(
    frame: number,
    unfiltered: Float32Array,
    ground: readonly (number | null)[],
  ): Float32Array {
    const carried = Math.fround(this.carried[frame] ?? 0);
    const shifted = Float32Array.from(unfiltered, (value, index) =>
      index % 2 === 0 ? value + carried : value,
    );
    const positions = this.settled(frame, shifted, ground);
    const filtered = Float32Array.from(positions);
    ground.forEach((level, paw) => {
      if (level === null) {
        return;
      }
      const onset = this.runOnset[paw]?.[frame] ?? -1;
      const blend = this.blendOnset[paw]?.[frame] ?? -1;
      if (onset < 0 && blend < 0) {
        return;
      }
      const [, knee, foot] = this.leg(paw);
      const solved = this.solveLeg(paw, onset >= 0 ? onset : blend, positions, level);
      [knee, foot].forEach((joint, k) => {
        const [sx, sy] = solved[k] ?? [0, 0];
        filtered[2 * joint] = onset >= 0 ? sx : (sx + x(positions, joint)) * 0.5;
        filtered[2 * joint + 1] = onset >= 0 ? sy : (sy + y(positions, joint)) * 0.5;
      });
    });
    return filtered;
  }
}

function x(positions: Float32Array, joint: number): number {
  return positions[2 * joint] ?? Number.NaN;
}

function y(positions: Float32Array, joint: number): number {
  return positions[2 * joint + 1] ?? Number.NaN;
}

function point(positions: Float32Array, joint: number): Vec {
  return [x(positions, joint), y(positions, joint)];
}
