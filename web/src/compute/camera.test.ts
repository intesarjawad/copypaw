import { describe, expect, it } from "vitest";
import { CHARACTER_DEPTH, frameCamera, viewProjection } from "./camera";

function project(matrix: Float32Array, x: number, y: number): [number, number] {
  const at = (index: number): number => matrix[index] ?? Number.NaN;
  const clipX = at(0) * x + at(4) * y + at(8) * CHARACTER_DEPTH + at(12);
  const clipY = at(1) * x + at(5) * y + at(9) * CHARACTER_DEPTH + at(13);
  const clipW = at(3) * x + at(7) * y + at(11) * CHARACTER_DEPTH + at(15);
  return [clipX / clipW, clipY / clipW];
}

describe("frameCamera", () => {
  it("centres the box and fits its taller side with a margin", () => {
    const camera = frameCamera([-1, 0], [1, 4], [512, 512]);
    const matrix = viewProjection(camera, [512, 512], false);

    expect(camera.slice(0, 2)).toEqual([0, 2]);
    // The 5% margin on each side leaves the box spanning 1 / 1.1 of the view.
    expect(project(matrix, 1, 4)[1]).toBeCloseTo(1 / 1.1, 6);
    expect(project(matrix, -1, 0)[1]).toBeCloseTo(-1 / 1.1, 6);
    expect(Math.abs(project(matrix, 1, 4)[0])).toBeLessThan(1);
  });

  it("fits the width when the box is wider than the window", () => {
    const camera = frameCamera([0, 0], [10, 1], [512, 256]);

    expect(project(viewProjection(camera, [512, 256], false), 10, 0.5)[0]).toBeCloseTo(1 / 1.1, 6);
  });

  it("mirrors the picture left to right", () => {
    const camera = frameCamera([0, 0], [2, 2], [512, 512]);

    expect(project(viewProjection(camera, [512, 512], true), 2, 1)[0]).toBeCloseTo(-1 / 1.1, 6);
  });

  it("rejects a box with no reach", () => {
    expect(() => frameCamera([0, 0], [0, 1], [512, 512])).toThrow("positive");
  });
});
