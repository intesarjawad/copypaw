import { describe, expect, it } from "vitest";
import type { Point } from "../joints";
import { Arap, type Triangle } from "./arap";

const SQUARE: readonly Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];
const TRIANGLES: readonly Triangle[] = [
  [0, 1, 2],
  [0, 2, 3],
];

function expectPoints(actual: readonly Point[], expected: readonly Point[]): void {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((point, index) => {
    expect(point.x).toBeCloseTo(expected[index]?.x ?? Number.NaN, 5);
    expect(point.y).toBeCloseTo(expected[index]?.y ?? Number.NaN, 5);
  });
}

describe("Arap", () => {
  it("keeps the mesh when the pins stay put", () => {
    const arap = new Arap(SQUARE, TRIANGLES, SQUARE);

    expectPoints(arap.solve(SQUARE), SQUARE);
  });

  it("translates the mesh with its pins", () => {
    const arap = new Arap(SQUARE, TRIANGLES, SQUARE);
    const moved = SQUARE.map(({ x, y }) => ({ x: x + 2, y: y - 3 }));

    expectPoints(arap.solve(moved), moved);
  });

  it("rotates unpinned vertices rigidly with two diagonal pins", () => {
    const pins = [SQUARE[0] ?? { x: 0, y: 0 }, SQUARE[2] ?? { x: 1, y: 1 }];
    const arap = new Arap(pins, TRIANGLES, SQUARE);
    const quarterTurn = ({ x, y }: Point): Point => ({ x: -y, y: x });

    expectPoints(arap.solve(pins.map(quarterTurn)), SQUARE.map(quarterTurn));
  });

  it("drops a pin outside the mesh and ignores it when solving", () => {
    const pins = [...SQUARE, { x: 5, y: 5 }];
    const arap = new Arap(pins, TRIANGLES, SQUARE);

    expect(arap.pinMask).toEqual([true, true, true, true, false]);
    expectPoints(arap.solve([...SQUARE, { x: 100, y: -100 }]), SQUARE);
  });

  it("follows a pin inside a triangle", () => {
    const pins = [
      { x: 0.25, y: 0.5 },
      { x: 0.75, y: 0.5 },
    ];
    const arap = new Arap(pins, TRIANGLES, SQUARE);
    const shifted = SQUARE.map(({ x, y }) => ({ x: x + 0.5, y }));

    expectPoints(arap.solve(pins.map(({ x, y }) => ({ x: x + 0.5, y }))), shifted);
  });
});
