import type { Triangle } from "./arap";
import type { Mask } from "./mask";
import { delaunay } from "./triangulate";

/** A point as the engine stores mesh vertices: row, then column, of the upright padded mask. */
export type RowColumn = readonly [row: number, column: number];

// Interior vertices a character's mesh aims for.
const MESH_INTERIOR_VERTICES = 1000;
// The mask is 0 or 255; the engine traces its outline at this level.
const OUTLINE_LEVEL = 128;
// The engine keeps outline corners that stand this far off a straight run, in pixels.
const CORNER_TOLERANCE_PX = 1;

/** The engine's view of a mask: turned a quarter clockwise and padded to a square. */
export function upright(mask: Mask): { size: number; at: (row: number, column: number) => number } {
  const size = Math.max(mask.width, mask.height);
  return {
    size,
    at: (row, column) =>
      row < mask.width && column < mask.height
        ? (mask.data[(mask.height - 1 - column) * mask.width + row] ?? 0)
        : 0,
  };
}

function pointAt(points: readonly RowColumn[], index: number): RowColumn {
  const point = points[index];
  if (point === undefined) {
    throw new RangeError(`Index ${String(index)} is out of range`);
  }
  return point;
}

function fraction(from: number, to: number): number {
  return to === from ? 0 : (OUTLINE_LEVEL - from) / (to - from);
}

/** Marching squares at the outline level, each segment directed with the inside on its left. */
function segments(size: number, at: (row: number, column: number) => number): RowColumn[][] {
  const found: RowColumn[][] = [];
  for (let r0 = 0; r0 < size - 1; r0++) {
    for (let c0 = 0; c0 < size - 1; c0++) {
      const [ul, ur, ll, lr] = [at(r0, c0), at(r0, c0 + 1), at(r0 + 1, c0), at(r0 + 1, c0 + 1)];
      const square =
        (ul > OUTLINE_LEVEL ? 1 : 0) +
        (ur > OUTLINE_LEVEL ? 2 : 0) +
        (ll > OUTLINE_LEVEL ? 4 : 0) +
        (lr > OUTLINE_LEVEL ? 8 : 0);
      if (square === 0 || square === 15) {
        continue;
      }
      const top: RowColumn = [r0, c0 + fraction(ul, ur)];
      const bottom: RowColumn = [r0 + 1, c0 + fraction(ll, lr)];
      const left: RowColumn = [r0 + fraction(ul, ll), c0];
      const right: RowColumn = [r0 + fraction(ur, lr), c0 + 1];
      // scikit-image's case table, with diagonal corners left unconnected.
      const cases: Record<number, RowColumn[][]> = {
        1: [[top, left]],
        2: [[right, top]],
        3: [[right, left]],
        4: [[left, bottom]],
        5: [[top, bottom]],
        6: [
          [right, top],
          [left, bottom],
        ],
        7: [[right, bottom]],
        8: [[bottom, right]],
        9: [
          [top, left],
          [bottom, right],
        ],
        10: [[bottom, top]],
        11: [[bottom, left]],
        12: [[left, right]],
        13: [[top, right]],
        14: [[left, top]],
      };
      found.push(...(cases[square] ?? []));
    }
  }
  return found;
}

/** Join directed segments end to start into contours, as scikit-image's find_contours does. */
function assemble(pieces: readonly RowColumn[][]): RowColumn[][] {
  const key = ([row, column]: RowColumn): string => `${String(row)},${String(column)}`;
  const contours = new Map<number, RowColumn[]>();
  const starts = new Map<string, number>();
  const ends = new Map<string, number>();
  let next = 0;
  const take = (map: Map<string, number>, point: RowColumn): number | undefined => {
    const found = map.get(key(point));
    map.delete(key(point));
    return found;
  };
  for (const [from, to] of pieces) {
    if (from === undefined || to === undefined || key(from) === key(to)) {
      continue;
    }
    const tailId = take(starts, to);
    const headId = take(ends, from);
    const tail = tailId === undefined ? undefined : contours.get(tailId);
    const head = headId === undefined ? undefined : contours.get(headId);
    if (tail && head && tailId !== undefined && headId !== undefined) {
      if (tailId === headId) {
        head.push(to);
      } else if (tailId > headId) {
        head.push(...tail);
        contours.delete(tailId);
        starts.set(key(head[0] ?? from), headId);
        ends.set(key(head[head.length - 1] ?? to), headId);
      } else {
        tail.unshift(...head);
        starts.delete(key(head[0] ?? from));
        contours.delete(headId);
        starts.set(key(tail[0] ?? from), tailId);
        ends.set(key(tail[tail.length - 1] ?? to), tailId);
      }
    } else if (!tail && !head) {
      contours.set(next, [from, to]);
      starts.set(key(from), next);
      ends.set(key(to), next);
      next++;
    } else if (tail && tailId !== undefined) {
      tail.unshift(from);
      starts.set(key(from), tailId);
    } else if (head && headId !== undefined) {
      head.push(to);
      ends.set(key(to), headId);
    }
  }
  return [...contours.entries()].sort(([a], [b]) => a - b).map(([, contour]) => contour);
}

function distance([r0, c0]: RowColumn, [r1, c1]: RowColumn): number {
  return Math.hypot(r1 - r0, c1 - c0);
}

/** Indices of the points scikit-image's approximate_polygon keeps, a Douglas-Peucker variant. */
function corners(points: readonly RowColumn[], tolerance: number): number[] {
  const kept = new Set([0, points.length - 1]);
  const stack: [number, number][] = [[0, points.length - 1]];
  for (let span = stack.pop(); span !== undefined; span = stack.pop()) {
    const [start, end] = span;
    const [r0, c0] = pointAt(points, start);
    const [r1, c1] = pointAt(points, end);
    const [dr, dc] = [r1 - r0, c1 - c0];
    const angle = -Math.atan2(dr, dc);
    const offset = c0 * Math.sin(angle) + r0 * Math.cos(angle);
    let farthest = -1;
    let worst = -Infinity;
    for (let index = start + 1; index < end; index++) {
      const [r, c] = pointAt(points, index);
      const perpendicular = (r - r0) * dr + (c - c0) * dc > 0 && -(r - r1) * dr - (c - c1) * dc > 0;
      const gap = perpendicular
        ? Math.abs(r * Math.cos(angle) + c * Math.sin(angle) - offset)
        : Math.min(Math.hypot(c - c0, r - r0), Math.hypot(c - c1, r - r1));
      if (gap > worst) {
        worst = gap;
        farthest = index;
      }
    }
    if (worst > tolerance) {
      stack.push([farthest, end], [start, farthest]);
      kept.add(farthest);
    }
  }
  return [...kept].sort((a, b) => a - b);
}

function shoelaceArea(polygon: readonly RowColumn[]): number {
  let twice = 0;
  polygon.forEach(([r0, c0], index) => {
    const [r1, c1] = polygon[(index + 1) % polygon.length] ?? [r0, c0];
    twice += r0 * c1 - r1 * c0;
  });
  return Math.abs(twice) / 2;
}

/** Whether a point lies inside the polygon and more than clearance from its outline. */
function insideBy(polygon: readonly RowColumn[], point: RowColumn, clearance: number): boolean {
  const [pr, pc] = point;
  let inside = false;
  for (let index = 0; index < polygon.length; index++) {
    const a = polygon[index] ?? point;
    const b = polygon[(index + 1) % polygon.length] ?? point;
    if (a[1] > pc !== b[1] > pc && pr < a[0] + ((pc - a[1]) * (b[0] - a[0])) / (b[1] - a[1])) {
      inside = !inside;
    }
    const [dr, dc] = [b[0] - a[0], b[1] - a[1]];
    const length = dr * dr + dc * dc;
    const along =
      length > 0 ? Math.min(Math.max(((pr - a[0]) * dr + (pc - a[1]) * dc) / length, 0), 1) : 0;
    if (Math.hypot(a[0] + along * dr - pr, a[1] + along * dc - pc) <= clearance) {
      return false;
    }
  }
  return inside;
}

/**
 * The vertices the engine's mesh is built on for a character mask, in its row and column order.
 *
 * Vertices are spaced by the character's area, so a thin limb gets as many as
 * a wide one: the outline's corners and samples along it at that spacing, then
 * a grid at the same spacing inside the outline inset by half of it.
 */
export function meshVertices(mask: Mask): RowColumn[] {
  return outlinedVertices(mask).vertices;
}

export interface CharacterMesh {
  readonly vertices: readonly RowColumn[];
  readonly triangles: readonly Triangle[];
}

/**
 * The engine's character mesh: its vertices, in row and column order, and the
 * Delaunay triangles whose centroid lies inside the outline.
 */
export function characterMesh(mask: Mask): CharacterMesh {
  const { outline, vertices } = outlinedVertices(mask);
  // The engine triangulates single-precision vertices.
  const stored = vertices.map(([row, column]): RowColumn => [
    Math.fround(row),
    Math.fround(column),
  ]);
  const triangles = delaunay(stored).filter((corners) => {
    const [row, column] = [0, 1].map((axis) =>
      // The engine averages the corners in single precision.
      Math.fround(
        corners.reduce((sum, index) => Math.fround(sum + (pointAt(stored, index)[axis] ?? 0)), 0) /
          3,
      ),
    );
    return insideBy(outline, [row ?? 0, column ?? 0], 0);
  });
  return { vertices: stored, triangles };
}

function outlinedVertices(mask: Mask): { outline: RowColumn[]; vertices: RowColumn[] } {
  const { size, at } = upright(mask);
  const outline = assemble(segments(size, at)).sort((a, b) => b.length - a.length)[0];
  if (outline === undefined) {
    throw new Error("The mask must contain the drawing");
  }
  const spacing = Math.max(1, Math.sqrt(shoelaceArea(outline) / MESH_INTERIOR_VERTICES));
  const arcs = [0];
  outline.slice(1).forEach((point, index) => {
    arcs.push((arcs[index] ?? 0) + distance(outline[index] ?? point, point));
  });
  const perimeter = arcs[arcs.length - 1] ?? 0;
  // Vertices closer than half the spacing would make sliver triangles the solver cannot keep rigid.
  const clear = (arc: number, kept: readonly number[]): boolean =>
    kept.every(
      (other) => Math.min(Math.abs(arc - other), perimeter - Math.abs(arc - other)) >= spacing / 2,
    );
  const cornerArcs: number[] = [];
  for (const index of corners(outline, CORNER_TOLERANCE_PX).slice(0, -1)) {
    const corner = pointAt(outline, index);
    const first = outline.findIndex((point) => distance(point, corner) === 0);
    const arc = arcs[first] ?? 0;
    if (clear(arc, cornerArcs)) {
      cornerArcs.push(arc);
    }
  }
  const sampleArcs: number[] = [];
  for (let step = 0; step * spacing < perimeter; step++) {
    if (clear(step * spacing, cornerArcs)) {
      sampleArcs.push(step * spacing);
    }
  }
  const onOutline = [...cornerArcs, ...sampleArcs]
    .sort((a, b) => a - b)
    .map((arc) =>
      pointAt(
        outline,
        arcs.findIndex((value) => value >= arc),
      ),
    );
  const grid = Array.from({ length: Math.ceil(size / spacing) }, (_, step) => step * spacing);
  const interior = grid.flatMap((column) =>
    grid.flatMap((row): RowColumn[] =>
      insideBy(outline, [row, column], spacing / 2) ? [[row, column]] : [],
    ),
  );
  return { outline, vertices: [...onOutline, ...interior] };
}
