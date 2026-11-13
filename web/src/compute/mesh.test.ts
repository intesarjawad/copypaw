import { describe, expect, it } from "vitest";
import type { Mask } from "./mask";
import { meshVertices } from "./mesh";

function square(size: number, side: number): Mask {
  const data = new Uint8Array(size * size);
  const start = Math.floor((size - side) / 2);
  for (let y = start; y < start + side; y++) {
    data.fill(255, y * size + start, y * size + start + side);
  }
  return { width: size, height: size, data };
}

describe("meshVertices", () => {
  it("spaces vertices by the drawing's area, about a thousand inside it", () => {
    const vertices = meshVertices(square(400, 300));
    const spacing = Math.sqrt((300 * 300) / 1000);

    // The outline is sampled once per spacing, plus its four corners.
    const onOutline = vertices.filter(([row, column]) =>
      [row, column].some((value) => Math.abs(value - 49.5) < 1 || Math.abs(value - 349.5) < 1),
    );
    expect(onOutline.length).toBeCloseTo((4 * 300) / spacing, -1);
    expect(vertices.length - onOutline.length).toBeGreaterThan(800);
    expect(vertices.length - onOutline.length).toBeLessThan(1000);
  });

  it("keeps every vertex on or inside the outline", () => {
    for (const [row, column] of meshVertices(square(100, 60))) {
      expect(row).toBeGreaterThanOrEqual(19);
      expect(row).toBeLessThanOrEqual(81);
      expect(column).toBeGreaterThanOrEqual(19);
      expect(column).toBeLessThanOrEqual(81);
    }
  });

  it("rejects an empty mask", () => {
    expect(() => meshVertices({ width: 4, height: 4, data: new Uint8Array(16) })).toThrow(
      "drawing",
    );
  });
});
