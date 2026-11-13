import { describe, expect, it } from "vitest";
import { MalformedResponse } from "../parse";
import record from "./sheet.json";
import { lampLevel, lampSettled, place, readSheet, wrapFrame } from "./sheet";

describe("readSheet", () => {
  it("reads the record the build script wrote", () => {
    const sheet = readSheet(record);
    expect(sheet.frames).toBe(record.frames);
    expect(sheet.dog).toHaveLength(sheet.paws.length);
    expect(sheet.dog[0]?.[0]).toBe(record.footfalls.dog[0]?.startsWith("1"));
    expect(sheet.clip.atlas.rows * sheet.clip.atlas.columns).toBeGreaterThanOrEqual(sheet.frames);
  });

  it("rejects a footfall track that is not one 0 or 1 per frame", () => {
    const short = {
      ...record,
      footfalls: { ...record.footfalls, dog: ["0101", ...record.footfalls.dog.slice(1)] },
    };
    expect(() => readSheet(short)).toThrow(MalformedResponse);
  });

  it("rejects a paw track outside the clip", () => {
    const outside = { ...record, dog: { ...record.dog, paws: record.dog.paws.map(() => 2) } };
    expect(() => readSheet(outside)).toThrow(MalformedResponse);
  });

  it("rejects a paw track that misses frames", () => {
    const short = { ...record, drawing: { ...record.drawing, paws: [0.5] } };
    expect(() => readSheet(short)).toThrow(MalformedResponse);
  });
});

describe("place", () => {
  const sheet = readSheet(record);
  const clip = { x: 40, y: 100, width: 900, height: 510 };

  it("stands the drawing's paws on the clip's floor line", () => {
    const { figure, floorY } = place(sheet, 0, clip);
    expect(floorY).toBeCloseTo(clip.y + sheet.clip.floor * clip.height);
    expect(figure.y + (sheet.figure.lows[0] ?? Number.NaN) * figure.height).toBeCloseTo(floorY);
    expect(figure.height).toBeCloseTo(sheet.figure.height * clip.height);
  });

  it("sets the drawing's paws over the dog's in every frame", () => {
    for (const frame of [0, sheet.frames - 1]) {
      const { figure } = place(sheet, frame, clip);
      const drawn = figure.x + (sheet.figure.paws[frame] ?? Number.NaN) * figure.width;
      expect(drawn).toBeCloseTo(clip.x + (sheet.clip.paws[frame] ?? Number.NaN) * clip.width);
    }
  });
});

describe("wrapFrame", () => {
  it("wraps both ways round the loop", () => {
    expect(wrapFrame(74, 74)).toBe(0);
    expect(wrapFrame(-1, 74)).toBe(73);
  });
});

describe("lampLevel", () => {
  it("strikes in stutters and then holds full", () => {
    expect(lampLevel(true, 100)).toBeLessThan(0.2);
    expect(lampLevel(true, 10_000)).toBe(1);
    expect(lampSettled(true, 10_000)).toBe(true);
  });

  it("flickers once and fades to dark when switched off", () => {
    expect(lampLevel(false, 30)).toBeLessThan(lampLevel(false, 90));
    expect(lampLevel(false, 500)).toBeLessThan(lampLevel(false, 200));
    expect(lampLevel(false, 10_000)).toBe(0);
    expect(lampSettled(false, 10_000)).toBe(true);
  });
});
