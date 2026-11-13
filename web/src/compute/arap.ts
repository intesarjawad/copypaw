import type { Point } from "../joints";

export type Triangle = readonly [number, number, number];

/** A sparse row: column index to value. */
type Row = Map<number, number>;

const PERTURBATION = 1e-8;
const MAX_PERTURBATIONS = 100;

function add(row: Row, column: number, value: number): void {
  row.set(column, (row.get(column) ?? 0) + value);
}

function get<T>(items: readonly T[], index: number): T {
  const item = items[index];
  if (item === undefined) {
    throw new RangeError(`Index ${String(index)} is out of range`);
  }
  return item;
}

/** Lower-triangular Cholesky factor of a dense symmetric matrix, or null if not positive. */
function cholesky(matrix: Float64Array, size: number): Float64Array | null {
  const lower = new Float64Array(size * size);
  for (let j = 0; j < size; j++) {
    let diagonal = matrix[j * size + j] ?? 0;
    for (let k = 0; k < j; k++) {
      const value = lower[j * size + k] ?? 0;
      diagonal -= value * value;
    }
    if (!(diagonal > 0)) {
      return null;
    }
    const pivot = Math.sqrt(diagonal);
    lower[j * size + j] = pivot;
    for (let i = j + 1; i < size; i++) {
      let sum = matrix[i * size + j] ?? 0;
      for (let k = 0; k < j; k++) {
        sum -= (lower[i * size + k] ?? 0) * (lower[j * size + k] ?? 0);
      }
      lower[i * size + j] = sum / pivot;
    }
  }
  return lower;
}

/** A factored normal matrix AᵀA for repeated least-squares solves. */
class NormalSolver {
  private readonly lower: Float64Array;

  constructor(
    private readonly rows: readonly Row[],
    private readonly size: number,
  ) {
    const normal = new Float64Array(size * size);
    for (const row of rows) {
      for (const [i, a] of row) {
        for (const [j, b] of row) {
          normal[i * size + j] = (normal[i * size + j] ?? 0) + a * b;
        }
      }
    }
    // The reference nudges a singular matrix by a small multiple of the identity until it inverts.
    for (let attempt = 0; ; attempt++) {
      const lower = cholesky(normal, size);
      if (lower) {
        this.lower = lower;
        break;
      }
      if (attempt >= MAX_PERTURBATIONS) {
        throw new Error("The mesh matrix stays singular");
      }
      for (let i = 0; i < size; i++) {
        normal[i * size + i] = (normal[i * size + i] ?? 0) + PERTURBATION;
      }
    }
  }

  /** Solve AᵀA x = Aᵀb. */
  solve(b: readonly number[]): Float64Array {
    const { size, lower } = this;
    const x = new Float64Array(size);
    this.rows.forEach((row, r) => {
      const value = get(b, r);
      if (value !== 0) {
        for (const [column, a] of row) {
          x[column] = (x[column] ?? 0) + a * value;
        }
      }
    });
    for (let i = 0; i < size; i++) {
      let sum = x[i] ?? 0;
      for (let k = 0; k < i; k++) {
        sum -= (lower[i * size + k] ?? 0) * (x[k] ?? 0);
      }
      x[i] = sum / (lower[i * size + i] ?? 1);
    }
    for (let i = size - 1; i >= 0; i--) {
      let sum = x[i] ?? 0;
      for (let k = i + 1; k < size; k++) {
        sum -= (lower[k * size + i] ?? 0) * (x[k] ?? 0);
      }
      x[i] = sum / (lower[i * size + i] ?? 1);
    }
    return x;
  }
}

/** Barycentric coordinates of p in triangle abc, from Ericson's Real-Time Collision Detection. */
function barycentric(p: Point, a: Point, b: Point, c: Point): [number, number, number] {
  const v0 = { x: b.x - a.x, y: b.y - a.y };
  const v1 = { x: c.x - a.x, y: c.y - a.y };
  const v2 = { x: p.x - a.x, y: p.y - a.y };
  const d00 = v0.x * v0.x + v0.y * v0.y;
  const d01 = v0.x * v1.x + v0.y * v1.y;
  const d11 = v1.x * v1.x + v1.y * v1.y;
  const d20 = v2.x * v0.x + v2.y * v0.y;
  const d21 = v2.x * v1.x + v2.y * v1.y;
  const denom = d00 * d11 - d01 * d01;
  const v = (d11 * d20 - d01 * d21) / denom;
  const w = (d00 * d21 - d01 * d20) / denom;
  return [1 - v - w, v, w];
}

/** The first triangle containing p, preferring strict interiors over edges, or null. */
function containing(p: Point, vertices: readonly Point[], triangles: readonly Triangle[]): number {
  const det = (u: Point, v: Point): number => u.x * v.y - u.y * v.x;
  const params = triangles.map(([i, j, k]) => {
    const v0 = get(vertices, i);
    const v1 = { x: get(vertices, j).x - v0.x, y: get(vertices, j).y - v0.y };
    const v2 = { x: get(vertices, k).x - v0.x, y: get(vertices, k).y - v0.y };
    const a = (det(p, v2) - det(v0, v2)) / det(v1, v2);
    const b = -(det(p, v1) - det(v0, v1)) / det(v1, v2);
    return [a, b] as const;
  });
  const inside = params.findIndex(([a, b]) => a > 0 && b > 0 && a + b < 1);
  return inside >= 0 ? inside : params.findIndex(([a, b]) => a >= 0 && b >= 0 && a + b <= 1);
}

/**
 * As-rigid-as-possible shape manipulation, ported from the vendored engine's ARAP.
 *
 * Igarashi and Igarashi, "Implementing As-Rigid-As-Possible Shape Manipulation
 * and Surface Flattening", 2009. The first solve frees rotation and scale; its
 * per-edge rotations then rotate the rest edges for a second solve that only
 * frees rotation. The normal matrices are dense and factored once by Cholesky,
 * which is fine at the engine's vertex counts.
 */
export class Arap {
  /** Whether each initial pin lay inside or on the mesh; pins outside are dropped. */
  readonly pinMask: readonly boolean[];
  readonly vertexCount: number;
  private readonly edges: readonly (readonly [number, number])[];
  private readonly edgeVectors: readonly Point[];
  private readonly rotations: readonly Row[];
  private readonly similarity: NormalSolver;
  private readonly rigid: NormalSolver;

  constructor(
    pins: readonly Point[],
    triangles: readonly Triangle[],
    vertices: readonly Point[],
    private readonly w = 1000,
  ) {
    this.vertexCount = vertices.length;
    const edgeKeys = new Map<string, readonly [number, number]>();
    const neighbours = vertices.map(() => new Set<number>());
    for (const [v0, v1, v2] of triangles) {
      for (const [i, j] of [
        [v0, v1],
        [v1, v2],
        [v2, v0],
      ] as const) {
        const edge = [Math.min(i, j), Math.max(i, j)] as const;
        edgeKeys.set(`${String(edge[0])},${String(edge[1])}`, edge);
        get(neighbours, i).add(j);
        get(neighbours, j).add(i);
      }
    }
    this.edges = [...edgeKeys.values()];
    this.edgeVectors = this.edges.map(([i, j]) => ({
      x: get(vertices, j).x - get(vertices, i).x,
      y: get(vertices, j).y - get(vertices, i).y,
    }));

    const pinWeights: [number, number][][] = [];
    this.pinMask = pins.map((pin) => {
      const index = containing(pin, vertices, triangles);
      if (index < 0) {
        return false;
      }
      const triangle = get(triangles, index);
      const [a, b, c] = triangle;
      const weights = barycentric(pin, get(vertices, a), get(vertices, b), get(vertices, c));
      pinWeights.push(triangle.map((v, k): [number, number] => [v, get(weights, k)]));
      return true;
    });

    const first: Row[] = [];
    const rotations: Row[] = [];
    this.edges.forEach(([vi, vj], k) => {
      const shared = [...get(neighbours, vi)].filter((v) => get(neighbours, vj).has(v));
      const around = [vi, vj, ...shared];
      const origin = get(vertices, vi);
      // G stacks [dx, dy] and [dy, -dx] for every vertex but vi, so GᵀG is
      // Σ(dx² + dy²) times the identity and inv(GᵀG) Gᵀ is a scaled Gᵀ.
      let squares = 0;
      for (const v of around.slice(1)) {
        const dx = get(vertices, v).x - origin.x;
        const dy = get(vertices, v).y - origin.y;
        squares += dx * dx + dy * dy;
      }
      const scale = 1 / squares;
      const gBlocks = around.slice(1).map((v) => {
        const dx = get(vertices, v).x - origin.x;
        const dy = get(vertices, v).y - origin.y;
        return [
          [scale * dx, scale * dy],
          [scale * dy, -scale * dx],
        ] as const;
      });
      const block = (index: number): readonly (readonly [number, number])[] => {
        if (index > 0) {
          return get(gBlocks, index - 1);
        }
        return gBlocks.reduce(
          (total, g) => [
            [total[0][0] - g[0][0], total[0][1] - g[0][1]],
            [total[1][0] - g[1][0], total[1][1] - g[1][1]],
          ],
          [
            [0, 0],
            [0, 0],
          ],
        );
      };
      const e = get(this.edgeVectors, k);
      const rowX: Row = new Map();
      const rowY: Row = new Map();
      add(rowX, 2 * vi, -1);
      add(rowY, 2 * vi + 1, -1);
      add(rowX, 2 * vj, 1);
      add(rowY, 2 * vj + 1, 1);
      const gX: Row = new Map();
      const gY: Row = new Map();
      around.forEach((v, index) => {
        const g = block(index);
        const [g0, g1] = [get(g, 0), get(g, 1)];
        // h = [[ex, ey], [ey, -ex]] @ g, subtracted from the edge rows.
        for (let column = 0; column < 2; column++) {
          const top = e.x * get(g0, column) + e.y * get(g1, column);
          const bottom = e.y * get(g0, column) - e.x * get(g1, column);
          add(rowX, 2 * v + column, -top);
          add(rowY, 2 * v + column, -bottom);
          gX.set(2 * v + column, get(g0, column));
          gY.set(2 * v + column, get(g1, column));
        }
      });
      first.push(rowX, rowY);
      rotations.push(gX, gY);
    });
    for (const weights of pinWeights) {
      const rowX: Row = new Map();
      const rowY: Row = new Map();
      for (const [v, weight] of weights) {
        rowX.set(2 * v, w * weight);
        rowY.set(2 * v + 1, w * weight);
      }
      first.push(rowX, rowY);
    }
    const second: Row[] = this.edges.map(
      ([vi, vj]) =>
        new Map([
          [vi, -1],
          [vj, 1],
        ]),
    );
    for (const weights of pinWeights) {
      second.push(new Map(weights.map(([v, weight]) => [v, w * weight])));
    }
    this.rotations = rotations;
    this.similarity = new NormalSolver(first, 2 * vertices.length);
    this.rigid = new NormalSolver(second, vertices.length);
  }

  /** Move the pins, in their original order, and return every vertex's new position. */
  solve(pins: readonly Point[]): Point[] {
    const kept = pins.filter((_, index) => this.pinMask[index] ?? false);
    const edgeCount = this.edges.length;
    const b1 = [
      ...new Array<number>(2 * edgeCount).fill(0),
      ...kept.flatMap((pin) => [this.w * pin.x, this.w * pin.y]),
    ];
    const v1 = this.similarity.solve(b1);
    const b2x: number[] = [];
    const b2y: number[] = [];
    this.edgeVectors.forEach((e0, k) => {
      let c = 0;
      let s = 0;
      for (const [column, value] of get(this.rotations, 2 * k)) {
        c += value * (v1[column] ?? 0);
      }
      for (const [column, value] of get(this.rotations, 2 * k + 1)) {
        s += value * (v1[column] ?? 0);
      }
      const scale = 1 / Math.sqrt(c * c + s * s);
      c *= scale;
      s *= scale;
      b2x.push(c * e0.x + s * e0.y);
      b2y.push(-s * e0.x + c * e0.y);
    });
    for (const pin of kept) {
      b2x.push(this.w * pin.x);
      b2y.push(this.w * pin.y);
    }
    const xs = this.rigid.solve(b2x);
    const ys = this.rigid.solve(b2y);
    return Array.from({ length: this.vertexCount }, (_, v) => ({
      x: xs[v] ?? 0,
      y: ys[v] ?? 0,
    }));
  }
}
