import { JOINT_IDS, type JointId, type Point } from "../joints";
import type { Triangle } from "./arap";
import { roundHalfEven, type Mask } from "./mask";
import { upright } from "./mesh";
import { JOINT_PARENTS } from "./retarget";

/** A heap entry as the engine orders it: distance, joint index, then pixel. */
type Entry = readonly [distance: number, joint: number, x: number, y: number];

// Seeds the engine spaces along each bone, from its joint toward its parent.
const SEEDS_PER_BONE = 20;
// The engine's starting distance for a pixel no bone has reached.
const UNREACHED = 1 << 12;
const NEIGHBOURS: readonly (readonly [dx: number, dy: number, step: number])[] = [
  [-1, -1, 1.414],
  [0, -1, 1.0],
  [1, -1, 1.414],
  [-1, 0, 1.0],
  [1, 0, 1.0],
  [-1, 1, 1.414],
  [0, 1, 1.0],
  [1, 1, 1.414],
];
const BODY: readonly JointId[] = [
  "right_hip_b",
  "left_hip_b",
  "right_hip_f",
  "left_hip_f",
  "hip",
  "torso",
  "neck",
];
// Depth of each paw chain off the body's plane, as the motion's skeleton places it.
const SIDE_DEPTH = 0.08;

function less(a: Entry, b: Entry): boolean {
  for (let index = 0; index < a.length; index++) {
    const [left = 0, right = 0] = [a[index], b[index]];
    if (left !== right) {
      return left < right;
    }
  }
  return false;
}

function item(heap: readonly Entry[], index: number): Entry {
  const entry = heap[index];
  if (entry === undefined) {
    throw new RangeError(`Heap index ${String(index)} is out of range`);
  }
  return entry;
}

/** Python's heapq sift toward the root, which pop and push both use. */
function siftDown(heap: Entry[], start: number, from: number): void {
  const moving = item(heap, from);
  let position = from;
  while (position > start) {
    const parent = (position - 1) >> 1;
    if (!less(moving, item(heap, parent))) {
      break;
    }
    heap[position] = item(heap, parent);
    position = parent;
  }
  heap[position] = moving;
}

/**
 * Python's heapq pop. The engine pops from its seed list without heapifying it
 * first, so the order pixels are claimed in depends on this exact procedure.
 */
function pop(heap: Entry[]): Entry {
  const last = heap.pop();
  if (last === undefined) {
    throw new RangeError("Pop from an empty heap");
  }
  if (heap.length === 0) {
    return last;
  }
  const top = item(heap, 0);
  heap[0] = last;
  let position = 0;
  let child = 1;
  while (child < heap.length) {
    if (child + 1 < heap.length && !less(item(heap, child), item(heap, child + 1))) {
      child += 1;
    }
    heap[position] = item(heap, child);
    position = child;
    child = 2 * position + 1;
  }
  heap[position] = last;
  siftDown(heap, 0, position);
  return top;
}

function push(heap: Entry[], entry: Entry): void {
  heap.push(entry);
  siftDown(heap, 0, heap.length - 1);
}

/** numpy's linspace from start toward stop, the stop left out. */
function spaced(start: readonly number[], stop: readonly number[], count: number): number[][] {
  const deltas = start.map((value, axis) => (stop[axis] ?? value) - value);
  const steps = deltas.map((delta) => delta / count);
  const anyFlat = steps.some((step) => step === 0);
  return Array.from({ length: count }, (_, index) =>
    start.map((value, axis) =>
      anyFlat ? (index / count) * (deltas[axis] ?? 0) + value : index * (steps[axis] ?? 0) + value,
    ),
  );
}

/**
 * Body parts in the order the engine draws them: farthest from the camera
 * first. A chain behind the body's plane is drawn paw first, then a chain in
 * front, also paw first, then the body from the neck back.
 */
export function bodyPartOrder(legs: 2 | 4): JointId[] {
  const farDepth = legs === 2 ? SIDE_DEPTH : -SIDE_DEPTH;
  const chains: [knee: JointId, foot: JointId, depth: number][] = [
    ["left_knee_f", "left_foot_f", SIDE_DEPTH],
    ["left_knee_b", "left_foot_b", SIDE_DEPTH],
    ["right_knee_f", "right_foot_f", farDepth],
    ["right_knee_b", "right_foot_b", farDepth],
  ];
  // A leg in front of the body tucks under it, so the hip shows no seam; at the body's depth
  // its paw is drawn first, so the upper bone covers the knee as on a leg behind.
  const groups: [JointId[], number][] = [
    ...chains.map(([knee, foot, depth]): [JointId[], number] => [[knee, foot], Math.min(depth, 0)]),
    [[...BODY], 0],
  ];
  return groups
    .sort(([, a], [, b]) => a - b)
    .flatMap(([joints, depth]) => (depth > 0 ? joints : joints.reverse()));
}

/**
 * The index buffer the engine draws a character's triangles with, ported from
 * its joint-to-triangle search.
 *
 * Every mask pixel goes to the bone it is nearest along the mask, found by a
 * Dijkstra search seeded along each bone; each triangle belongs to the bone at
 * its centroid, farthest triangles first. Bones are then drawn in body-part
 * order. joints are the drawing's pixel joints and vertices the mesh in rows
 * and columns of the upright mask, as fractions of its padded side.
 */
export function drawOrder(
  mask: Mask,
  joints: Readonly<Record<JointId, Point>>,
  vertices: readonly (readonly [number, number])[],
  triangles: readonly Triangle[],
  legs: 2 | 4,
): Uint32Array {
  const { size, at } = upright(mask);
  const shortest = new Int32Array(size * size).fill(UNREACHED);
  const closest = new Int8Array(size * size).fill(-1);
  const located = JOINT_IDS.map((id) => {
    const { x, y } = joints[id];
    return [x / size, 1 - (y / size + (1 - mask.height / size))];
  });

  const heap: Entry[] = [];
  JOINT_IDS.forEach((id, joint) => {
    const parent = JOINT_PARENTS[id];
    if (parent === null) {
      return;
    }
    const seeds = spaced(
      located[joint] ?? [],
      located[JOINT_IDS.indexOf(parent)] ?? [],
      SEEDS_PER_BONE,
    );
    for (const [x = 0, y = 0] of seeds) {
      heap.push([0, joint, roundHalfEven(size * x), roundHalfEven(size * y)]);
    }
  });
  while (heap.length > 0) {
    const [distance, joint, x, y] = pop(heap);
    for (const [dx, dy, step] of NEIGHBOURS) {
      const [nx, ny, reached] = [x + dx, y + dy, distance + step];
      if (nx < 0 || nx >= size || ny < 0 || ny >= size || at(nx, ny) === 0) {
        continue;
      }
      const cell = nx * size + ny;
      if ((shortest[cell] ?? 0) <= reached) {
        continue;
      }
      closest[cell] = joint;
      // The engine stores distances as integers but keeps searching with the exact ones.
      shortest[cell] = Math.trunc(reached);
      push(heap, [reached, joint, nx, ny]);
    }
  }

  const byJoint = new Map<number, [Triangle, number][]>();
  for (const triangle of triangles) {
    const corners = triangle.map((index) => vertices[index] ?? [0, 0]);
    const centroid = [0, 1].map((axis) => {
      const sum = corners.reduce((total, corner) => Math.fround(total + (corner[axis] ?? 0)), 0);
      return roundHalfEven(Math.fround(Math.fround(sum / 3) * size));
    });
    const cell = nearestReached(closest, size, centroid[0] ?? 0, centroid[1] ?? 0);
    const joint = closest[cell] ?? -1;
    const entries = byJoint.get(joint) ?? [];
    entries.push([triangle, shortest[cell] ?? 0]);
    byJoint.set(joint, entries);
  }

  return Uint32Array.from(
    bodyPartOrder(legs).flatMap((id) =>
      (byJoint.get(JOINT_IDS.indexOf(id)) ?? [])
        .sort(([, a], [, b]) => b - a)
        .flatMap(([triangle]) => [...triangle]),
    ),
  );
}

/**
 * The cell itself when a bone reached it, else the nearest cell one did: a
 * centroid on the outline can round onto a pixel just outside the mask.
 */
function nearestReached(closest: Int8Array, size: number, x: number, y: number): number {
  if ((closest[x * size + y] ?? -1) >= 0) {
    return x * size + y;
  }
  let best = -1;
  let bestDistance = Infinity;
  for (let radius = 1; radius < size && radius * radius <= bestDistance; radius++) {
    for (let cx = Math.max(0, x - radius); cx <= Math.min(size - 1, x + radius); cx++) {
      for (let cy = Math.max(0, y - radius); cy <= Math.min(size - 1, y + radius); cy++) {
        const distance = (cx - x) ** 2 + (cy - y) ** 2;
        if (distance < bestDistance && (closest[cx * size + cy] ?? -1) >= 0) {
          best = cx * size + cy;
          bestDistance = distance;
        }
      }
    }
  }
  if (best < 0) {
    throw new Error("No mask pixel lies near the mesh");
  }
  return best;
}
