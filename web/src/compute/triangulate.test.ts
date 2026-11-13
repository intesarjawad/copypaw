import { describe, expect, it } from "vitest";
import { delaunay } from "./triangulate";

type Vertex = readonly [number, number];

function corner(points: readonly Vertex[], index: number | undefined): Vertex {
  return points[index ?? -1] ?? [Number.NaN, Number.NaN];
}

function twiceArea(points: readonly Vertex[], [a, b, c]: readonly number[]): number {
  const [[ax, ay], [bx, by], [cx, cy]] = [corner(points, a), corner(points, b), corner(points, c)];
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

describe("delaunay", () => {
  it("splits a grid into two counter-clockwise triangles per cell", () => {
    const grid: Vertex[] = [];
    for (let x = 0; x < 5; x++) {
      for (let y = 0; y < 4; y++) {
        grid.push([x * 1.7, y * 1.7]);
      }
    }

    const triangles = delaunay(grid);

    expect(triangles).toHaveLength(2 * 4 * 3);
    for (const triangle of triangles) {
      expect(twiceArea(grid, triangle)).toBeCloseTo(1.7 * 1.7, 9);
    }
  });

  it("leaves every circumcircle empty of the other points", () => {
    // A fixed scatter, so the check is deterministic.
    const points: Vertex[] = Array.from({ length: 60 }, (_, index) => [
      (index * 37) % 101,
      (index * 59) % 97,
    ]);

    for (const [a, b, c] of delaunay(points)) {
      const [[ax, ay], [bx, by], [cx, cy]] = [
        corner(points, a),
        corner(points, b),
        corner(points, c),
      ];
      for (const [index, [px, py]] of points.entries()) {
        if (index === a || index === b || index === c) {
          continue;
        }
        // The in-circle determinant is negative outside a counter-clockwise triangle's circle.
        const [dax, day, dbx, dby, dcx, dcy] = [
          ax - px,
          ay - py,
          bx - px,
          by - py,
          cx - px,
          cy - py,
        ];
        const inCircle =
          (dax * dax + day * day) * (dbx * dcy - dcx * dby) -
          (dbx * dbx + dby * dby) * (dax * dcy - dcx * day) +
          (dcx * dcx + dcy * dcy) * (dax * dby - dbx * day);
        expect(inCircle).toBeLessThanOrEqual(1e-6);
      }
    }
  });

  it("returns nothing for fewer than three points", () => {
    expect(
      delaunay([
        [0, 0],
        [1, 1],
      ]),
    ).toEqual([]);
  });
});
