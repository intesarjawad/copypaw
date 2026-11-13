import type { Triangle } from "./arap";

type Vertex = readonly [number, number];

interface Circled {
  readonly corners: Triangle;
  readonly centreX: number;
  readonly centreY: number;
  readonly radiusSquared: number;
}

// A point this close to a circumcircle, relative to its radius, counts as on it, so
// grid points four to a circle never make the cavity depend on rounding.
const ON_CIRCLE_TOLERANCE = 1e-9;
// The enclosing triangle reaches this many times the points' extent past them.
const ENCLOSING_REACH = 1e3;

function at(points: readonly Vertex[], index: number): Vertex {
  const point = points[index];
  if (point === undefined) {
    throw new RangeError(`Index ${String(index)} is out of range`);
  }
  return point;
}

function circled(points: readonly Vertex[], corners: Triangle): Circled {
  const [[ax, ay], [bx, by], [cx, cy]] = [
    at(points, corners[0]),
    at(points, corners[1]),
    at(points, corners[2]),
  ];
  const twiceArea = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  const [a2, b2, c2] = [ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy];
  const centreX = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / twiceArea;
  const centreY = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / twiceArea;
  return {
    corners,
    centreX,
    centreY,
    radiusSquared: (ax - centreX) ** 2 + (ay - centreY) ** 2,
  };
}

function counterClockwise(points: readonly Vertex[], [a, b, c]: Triangle): Triangle {
  const [[ax, ay], [bx, by], [cx, cy]] = [at(points, a), at(points, b), at(points, c)];
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0 ? [a, c, b] : [a, b, c];
}

/**
 * Delaunay triangles of a point set, each counter-clockwise, by Bowyer-Watson insertion.
 *
 * Points are added one at a time inside a triangle that encloses them all;
 * each removes the triangles whose circumcircle holds it and fans the hole
 * from it. Triangles still touching the enclosing corners are dropped, so a
 * few slivers along the convex hull can be missing; callers keep only
 * triangles inside an outline, where none of them lie.
 */
export function delaunay(points: readonly Vertex[]): Triangle[] {
  if (points.length < 3) {
    return [];
  }
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const [minX, maxX, minY, maxY] = [
    Math.min(...xs),
    Math.max(...xs),
    Math.min(...ys),
    Math.max(...ys),
  ];
  const reach = ENCLOSING_REACH * Math.max(maxX - minX, maxY - minY, 1);
  const [midX, midY] = [(minX + maxX) / 2, (minY + maxY) / 2];
  const all: Vertex[] = [
    ...points,
    [midX - reach, midY - reach],
    [midX + reach, midY - reach],
    [midX, midY + reach],
  ];
  const first = points.length;
  let triangles: Circled[] = [circled(all, [first, first + 1, first + 2])];

  points.forEach(([px, py], index) => {
    const edges = new Map<string, readonly [number, number]>();
    const kept: Circled[] = [];
    for (const triangle of triangles) {
      const distanceSquared = (px - triangle.centreX) ** 2 + (py - triangle.centreY) ** 2;
      if (distanceSquared >= triangle.radiusSquared * (1 - ON_CIRCLE_TOLERANCE)) {
        kept.push(triangle);
        continue;
      }
      const [a, b, c] = triangle.corners;
      for (const [from, to] of [
        [a, b],
        [b, c],
        [c, a],
      ] as const) {
        const key = from < to ? `${String(from)},${String(to)}` : `${String(to)},${String(from)}`;
        // An edge two removed triangles share is inside the hole.
        if (!edges.delete(key)) {
          edges.set(key, [from, to]);
        }
      }
    }
    triangles = kept;
    for (const [from, to] of edges.values()) {
      triangles.push(circled(all, [from, to, index]));
    }
  });

  return triangles
    .filter(({ corners }) => corners.every((corner) => corner < first))
    .map(({ corners }) => counterClockwise(points, corners));
}
