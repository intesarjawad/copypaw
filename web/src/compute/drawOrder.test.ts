import { describe, expect, it } from "vitest";
import { bodyPartOrder } from "./drawOrder";

const BODY = ["neck", "torso", "hip", "left_hip_f", "right_hip_f", "left_hip_b", "right_hip_b"];

describe("bodyPartOrder", () => {
  it("draws every leg paw first, the far legs before the near ones, all under the body", () => {
    expect(bodyPartOrder(4)).toEqual([
      "right_foot_f",
      "right_knee_f",
      "right_foot_b",
      "right_knee_b",
      "left_foot_f",
      "left_knee_f",
      "left_foot_b",
      "left_knee_b",
      ...BODY,
    ]);
  });

  it("tucks a two-legged side view's legs under the body, as both copy the near limb", () => {
    expect(bodyPartOrder(2)).toEqual([
      "left_foot_f",
      "left_knee_f",
      "left_foot_b",
      "left_knee_b",
      "right_foot_f",
      "right_knee_f",
      "right_foot_b",
      "right_knee_b",
      ...BODY,
    ]);
  });
});
