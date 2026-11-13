import { describe, expect, it } from "vitest";
import { STATES, canAnimate, pageState, type Phases } from "./states";

const EDITING: Phases = { drawing: "editing", motionChosen: false, animate: "idle" };

describe("page state", () => {
  it("names every state the page can reach", () => {
    const reached = new Set([
      pageState({ ...EDITING, drawing: "empty" }),
      pageState({ ...EDITING, drawing: "loading" }),
      pageState({ ...EDITING, drawing: "failed" }),
      pageState(EDITING),
      pageState({ ...EDITING, motionChosen: true }),
      pageState({ ...EDITING, motionChosen: true, animate: "running" }),
      pageState({ ...EDITING, motionChosen: true, animate: "failed" }),
      pageState({ ...EDITING, motionChosen: true, animate: "done" }),
    ]);
    expect([...reached].sort()).toEqual(Object.keys(STATES).sort());
  });

  it("shows a new photo loading even after a result", () => {
    expect(pageState({ drawing: "loading", motionChosen: true, animate: "done" })).toBe(
      "drawingLoading",
    );
  });

  it("enables Animate only with dots and a motion, and not while running", () => {
    expect(canAnimate(EDITING)).toBe(false);
    expect(canAnimate({ ...EDITING, motionChosen: true })).toBe(true);
    expect(canAnimate({ ...EDITING, motionChosen: true, animate: "running" })).toBe(false);
    expect(canAnimate({ ...EDITING, motionChosen: true, drawing: "loading" })).toBe(false);
  });
});
