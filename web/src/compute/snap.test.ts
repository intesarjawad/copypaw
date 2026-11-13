import { describe, expect, it } from "vitest";
import type { Point } from "../joints";
import type { Mask } from "./mask";
import { detectFacing, fitTemplate, rigDrawing, snap } from "./snap";

function blank(width = 200, height = 200): Mask {
  return { width, height, data: new Uint8Array(width * height) };
}

function fill(mask: Mask, top: number, bottom: number, left: number, right: number): void {
  for (let y = top; y <= bottom; y++) {
    mask.data.fill(255, y * mask.width + left, y * mask.width + right + 1);
  }
}

/** Fill the pixels whose centres lie inside a convex polygon. */
function fillPolygon(mask: Mask, corners: readonly Point[]): void {
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      const sides = corners.map((a, k) => {
        const b = corners[(k + 1) % corners.length] ?? a;
        return Math.sign((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x));
      });
      if (sides.every((side) => side >= 0) || sides.every((side) => side <= 0)) {
        mask.data[y * mask.width + x] = 255;
      }
    }
  }
}

function inside(mask: Mask, point: Point): boolean {
  return (mask.data[Math.round(point.y) * mask.width + Math.round(point.x)] ?? 0) > 0;
}

/** A right-facing body with a neck rising at its front end and a head jutting forward. */
function headed(): Mask {
  const mask = blank();
  fill(mask, 60, 110, 30, 150);
  fill(mask, 20, 80, 150, 170);
  fill(mask, 20, 30, 170, 190);
  fill(mask, 110, 189, 40, 61);
  fill(mask, 110, 189, 120, 141);
  return mask;
}

const FOUR_LEFTS = [155, 235, 345, 425];

function fourLegs(): Mask {
  const mask = blank();
  fill(mask, 40, 110, 60, 140); // body, above the leg region
  fill(mask, 135, 189, 20, 41); // back near
  fill(mask, 125, 185, 50, 71); // back far
  fill(mask, 130, 190, 120, 141); // front near
  fill(mask, 120, 180, 150, 171); // front far
  return mask;
}

describe("fitTemplate", () => {
  it("returns template pixels for the template's own box", () => {
    const joints = fitTemplate(0, 0, 564, 352);

    expect(joints.neck).toEqual({ x: 455, y: 121 });
    expect(joints.left_foot_b).toEqual({ x: 47, y: 308 });
  });

  it("scales into the box", () => {
    expect(fitTemplate(10, 20, 10 + 282, 20 + 176).torso).toEqual({ x: 236.5, y: 110.5 });
  });

  it("rejects an empty box", () => {
    expect(() => fitTemplate(5, 5, 5, 10)).toThrow("positive area");
  });
});

describe("snap", () => {
  it("places four legs from median edge rows with knees at 55%", () => {
    const { joints, legs } = snap(fourLegs(), fitTemplate(20, 40, 171, 190), "right");

    // Bottom three rows of the left back leg span x 20..41, so their median is 30.5.
    expect(joints.left_hip_b).toEqual({ x: 30.5, y: 135 });
    expect(joints.left_foot_b).toEqual({ x: 30.5, y: 189 });
    expect(joints.left_knee_b.x).toBeCloseTo(30.5);
    expect(joints.left_knee_b.y).toBeCloseTo(135 + 0.55 * 54);
    expect(joints.left_hip_f).toEqual({ x: 130.5, y: 130 });
    expect(joints.right_hip_f).toEqual({ x: 160.5, y: 120 });
    expect(joints.right_hip_b).toEqual({ x: 60.5, y: 125 });
    expect(legs).toBe(4);
  });

  it("takes the leftmost pair as front when facing left", () => {
    const { joints } = snap(fourLegs(), fitTemplate(20, 40, 171, 190), "left");

    expect(joints.left_hip_f.y).toBe(135);
    expect(joints.right_hip_f.y).toBe(125);
    expect(joints.left_hip_b.y).toBe(130);
    expect(joints.right_hip_b.y).toBe(120);
  });

  it("shares one position per pair for two legs", () => {
    const mask = blank();
    fill(mask, 40, 110, 60, 140);
    fill(mask, 130, 189, 30, 51);
    fill(mask, 120, 190, 150, 171);

    const { joints, legs } = snap(mask, fitTemplate(30, 40, 171, 190), "right");

    expect(legs).toBe(2);
    expect(joints.right_knee_f).toEqual(joints.left_knee_f);
    expect(joints.right_foot_b).toEqual(joints.left_foot_b);
    expect(joints.left_hip_f.x).toBe(160.5);
    expect(joints.left_hip_b.x).toBe(40.5);
  });

  it("keeps the template legs for three components", () => {
    const mask = blank();
    fill(mask, 40, 110, 60, 140);
    fill(mask, 140, 189, 20, 41);
    fill(mask, 130, 189, 150, 171);
    fill(mask, 120, 189, 175, 196);
    const fitted = fitTemplate(20, 40, 196, 189);

    const { joints, legs } = snap(mask, fitted, "right");

    expect(legs).toBe(4);
    expect(joints.left_knee_f).toEqual(fitted.left_knee_f);
    expect(joints.right_foot_b).toEqual(fitted.right_foot_b);
  });

  it("stands the hip, torso and neck inside the outline over their legs", () => {
    const mask = headed();

    const { joints } = snap(mask, fitTemplate(30, 20, 190, 189), "right");

    for (const id of ["root", "hip", "torso", "neck"] as const) {
      expect(inside(mask, joints[id]), id).toBe(true);
    }
    expect(joints.root).toEqual(joints.hip);
    // Over the middle of the back and front legs, within a few pixels of the medial axis.
    expect(Math.abs(joints.hip.x - 50.5)).toBeLessThanOrEqual(3);
    expect(Math.abs(joints.torso.x - 130.5)).toBeLessThanOrEqual(3);
  });

  it("follows a bent neck instead of cutting the corner to the nose", () => {
    const mask = headed();

    const { joints } = snap(mask, fitTemplate(30, 20, 190, 189), "right");

    const torso = joints.torso;
    const straight = { x: torso.x + 0.7 * (190 - torso.x), y: torso.y + 0.7 * (20 - torso.y) };
    expect(inside(mask, straight)).toBe(false);
    expect(inside(mask, joints.neck)).toBe(true);
    expect(joints.neck.x).toBeGreaterThanOrEqual(150);
    expect(joints.neck.y).toBeLessThan(60);
  });

  it("puts each hip where its leg joins the body", () => {
    const { joints } = snap(headed(), fitTemplate(30, 20, 190, 189), "right");

    // The legs leave the body at row 110, above row 120 where the leg search starts.
    expect(joints.left_hip_f.y).toBeCloseTo(110, 0);
    expect(joints.left_hip_b.y).toBeCloseTo(110, 0);
  });

  it("sets aside a tail that leans back to the ground behind four legs", () => {
    const mask = blank(600, 400);
    fill(mask, 110, 230, 120, 480);
    fill(mask, 30, 110, 440, 520);
    for (const left of FOUR_LEFTS) {
      fill(mask, 200, 380, left, left + 29);
    }
    fillPolygon(mask, [
      { x: 118, y: 160 },
      { x: 138, y: 175 },
      { x: 82, y: 380 },
      { x: 62, y: 380 },
    ]);

    const { joints, legs, tails, legsGuessed } = snap(mask, fitTemplate(62, 30, 520, 381), "right");

    expect([legs, legsGuessed, tails.length]).toEqual([4, false, 1]);
    const feet = (["left", "right"] as const).flatMap((side) =>
      (["f", "b"] as const).map((end) => joints[`${side}_foot_${end}`].x),
    );
    expect(feet.sort((a, b) => a - b)).toEqual(FOUR_LEFTS.map((left) => left + 14.5));
    expect(tails[0]?.[1].y).toBe(380);
  });

  it("opens away a one-pixel bridge between legs", () => {
    const mask = fourLegs();
    fill(mask, 170, 170, 41, 50);

    expect(snap(mask, fitTemplate(20, 40, 171, 190), "right").joints.right_hip_b).toEqual({
      x: 60.5,
      y: 125,
    });
  });

  it("rejects an empty mask", () => {
    expect(() => snap(blank(), fitTemplate(0, 0, 10, 10), "right")).toThrow("drawing");
  });
});

describe("detectFacing", () => {
  it("follows the side of the head", () => {
    const mask = headed();
    const mirrored = blank();
    mask.data.forEach((value, index) => {
      const x = index % mask.width;
      mirrored.data[index - x + mask.width - 1 - x] = value;
    });

    expect(detectFacing(mask)).toBe("right");
    expect(detectFacing(mirrored)).toBe("left");
  });

  it("faces right when no head stands out and rejects an empty mask", () => {
    expect(detectFacing(fourLegs())).toBe("right");
    expect(() => detectFacing(blank())).toThrow("drawing");
  });
});

describe("rigDrawing", () => {
  it("rigs a drawn box at the photo's size", () => {
    const width = 300;
    const height = 200;
    const data = new Uint8ClampedArray(width * height * 4).fill(245);
    for (let y = 49; y <= 151; y++) {
      for (let x = 39; x <= 261; x++) {
        if (y <= 51 || y >= 149 || x <= 41 || x >= 259) {
          data.fill(20, 4 * (y * width + x), 4 * (y * width + x) + 3);
        }
      }
    }

    const { rig } = rigDrawing({ width, height, data });

    expect([rig.width, rig.height]).toEqual([300, 200]);
    // The filled region spans the stroke's 102 px centre-line height, give or take its width.
    expect(rig.boxHeight).toBeGreaterThan(95);
    expect(rig.boxHeight).toBeLessThan(110);
    expect(rig.outline.length).toBeGreaterThanOrEqual(4);
    expect(Math.min(...rig.outline.map((point) => point.x))).toBeLessThan(45);
    expect(Math.max(...rig.outline.map((point) => point.x))).toBeGreaterThan(255);
  });
});
