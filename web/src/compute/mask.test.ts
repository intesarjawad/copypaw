import { describe, expect, it } from "vitest";
import { BackendError } from "../backend";
import {
  ellipseHalfWidths,
  gaussianKernel,
  outline,
  resizeNearest,
  segment,
  type Mask,
  type Pixels,
} from "./mask";

const PAGE = 245;
const INK = 20;

interface Page extends Pixels {
  readonly data: Uint8ClampedArray;
}

function page(width = 300, height = 200): Page {
  const data = new Uint8ClampedArray(width * height * 4).fill(PAGE);
  return { width, height, data };
}

function ink(target: Page, left: number, top: number, right: number, bottom: number): void {
  for (let y = top; y <= bottom; y++) {
    for (let x = left; x <= right; x++) {
      target.data.fill(INK, 4 * (y * target.width + x), 4 * (y * target.width + x) + 3);
    }
  }
}

/** A three-pixel-wide rectangle outline centred on the given edges. */
function rectangle(target: Page, left: number, top: number, right: number, bottom: number): void {
  ink(target, left - 1, top - 1, right + 1, top + 1);
  ink(target, left - 1, bottom - 1, right + 1, bottom + 1);
  ink(target, left - 1, top - 1, left + 1, bottom + 1);
  ink(target, right - 1, top - 1, right + 1, bottom + 1);
}

function value(mask: Mask, x: number, y: number): number | undefined {
  return mask.data[y * mask.width + x];
}

function coverage(mask: Mask): number {
  return mask.data.filter((pixel) => pixel !== 0).length / mask.data.length;
}

function maskCode(run: () => unknown): string {
  try {
    run();
  } catch (error) {
    if (error instanceof BackendError) {
      return error.detail.code;
    }
    throw error;
  }
  return "none";
}

describe("kernels", () => {
  it("derives sigma 17.6 for a 115 px Gaussian block", () => {
    const kernel = gaussianKernel(115);
    expect(kernel.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 12);
    expect(kernel[0]).toBeCloseTo(kernel[114] ?? 0, 15);
    // Centre over edge weight is exp(57² / (2 · 17.6²)).
    expect((kernel[57] ?? 0) / (kernel[0] ?? 1)).toBeCloseTo(Math.exp(3249 / 619.52), 6);
  });

  it("builds OpenCV's 9 px ellipse", () => {
    expect(ellipseHalfWidths(9)).toEqual([0, 3, 3, 4, 4, 4, 3, 3, 0]);
  });
});

describe("segment", () => {
  it("fills a closed outline as one region", () => {
    const drawing = page();
    rectangle(drawing, 40, 50, 260, 150);

    const mask = segment(drawing);

    expect([mask.width, mask.height]).toEqual([300, 200]);
    expect(new Set(mask.data)).toEqual(new Set([0, 255]));
    // The 220 x 100 outline encloses 37% of the page; dilation moves the edge outward.
    expect(coverage(mask)).toBeGreaterThanOrEqual(0.3);
    expect(coverage(mask)).toBeLessThanOrEqual(0.45);
    expect(value(mask, 150, 100)).toBe(255);
    expect(value(mask, 10, 10)).toBe(0);
  });

  it("keeps single-line legs at a minimum width", () => {
    const drawing = page();
    rectangle(drawing, 40, 50, 260, 110);
    for (const x of [80, 150, 220]) {
      ink(drawing, x, 116, x, 180);
    }

    const mask = segment(drawing);

    for (const x of [80, 150, 220]) {
      expect(value(mask, x, 180)).toBe(255);
      expect(value(mask, x - 9, 170)).toBe(255);
      expect(value(mask, x + 9, 170)).toBe(255);
    }
    expect(value(mask, 115, 170)).toBe(0);
  });

  it("does not take flat ruling below the drawing for a leg", () => {
    const drawing = page();
    rectangle(drawing, 40, 50, 260, 110);
    ink(drawing, 60, 120, 240, 120);

    const mask = segment(drawing);

    expect(value(mask, 150, 120)).toBe(0);
  });

  it("fills an inner hole", () => {
    const drawing = page();
    rectangle(drawing, 40, 50, 260, 150);
    rectangle(drawing, 130, 85, 170, 115);

    const mask = segment(drawing);

    expect(value(mask, 150, 100)).toBe(255);
  });

  it("rejects a blank page as too small", () => {
    expect(maskCode(() => segment(page()))).toBe("mask_too_small");
  });

  it("rejects two drawings", () => {
    const drawing = page();
    rectangle(drawing, 20, 50, 120, 150);
    rectangle(drawing, 180, 50, 280, 150);

    expect(maskCode(() => segment(drawing))).toBe("mask_multiple_creatures");
  });

  it("caps the longest side at 1000 px", () => {
    // A wide, short page keeps the scaled image, and so the test, small.
    const drawing = page(1200, 200);
    rectangle(drawing, 200, 50, 1000, 150);

    const mask = segment(drawing);

    // 200 · 1000 / 1200 is 166.67, which rounds to 167.
    expect([mask.width, mask.height]).toEqual([1000, 167]);
    expect(value(mask, 500, 83)).toBe(255);
  });
});

describe("mask geometry", () => {
  it("scales back up by repeating the nearest pixel", () => {
    const small: Mask = { width: 2, height: 1, data: Uint8Array.from([0, 255]) };

    expect(Array.from(resizeNearest(small, 4, 2).data)).toEqual([0, 0, 255, 255, 0, 0, 255, 255]);
  });

  it("traces a square's corners, going down its left side first", () => {
    const data = new Uint8Array(25);
    for (const y of [1, 2, 3]) {
      data.fill(255, y * 5 + 1, y * 5 + 4);
    }

    expect(outline({ width: 5, height: 5, data })).toEqual([
      { x: 1, y: 1 },
      { x: 1, y: 3 },
      { x: 3, y: 3 },
      { x: 3, y: 1 },
    ]);
  });

  it("keeps every turn of an L shape", () => {
    // Column 0 runs down rows 0-2, then row 2 runs right to x 2.
    const data = Uint8Array.from([255, 0, 0, 255, 0, 0, 255, 255, 255]);

    expect(outline({ width: 3, height: 3, data })).toEqual([
      { x: 0, y: 0 },
      { x: 0, y: 2 },
      { x: 2, y: 2 },
      { x: 1, y: 2 },
      { x: 0, y: 1 },
    ]);
  });
});
