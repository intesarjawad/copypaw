/** Every state the page can show; the e2e suite visits and screenshots each one. */
export const STATES = {
  drawingEmpty: "No photo chosen yet; step 1 asks for one.",
  drawingLoading: "The photo is being outlined and rigged.",
  drawingFailed: "The photo could not be rigged; the error says what to change.",
  dotsEditing: "The 16 dots sit over the photo and can be moved; no motion is chosen yet.",
  motionReady: "A drawing and a motion are chosen, so Animate is enabled.",
  animating: "The animation job is running and its progress shows.",
  animateFailed: "The animation job failed; the error says what to do next.",
  result: "The result video, timing sentence and downloads are shown.",
} as const;

export type StateName = keyof typeof STATES;

export type DrawingPhase = "empty" | "loading" | "failed" | "editing";
export type AnimatePhase = "idle" | "running" | "failed" | "done";

export interface Phases {
  readonly drawing: DrawingPhase;
  readonly motionChosen: boolean;
  readonly animate: AnimatePhase;
}

const DRAWING_STATES: Readonly<Record<Exclude<DrawingPhase, "editing">, StateName>> = {
  empty: "drawingEmpty",
  loading: "drawingLoading",
  failed: "drawingFailed",
};

export function pageState(phases: Phases): StateName {
  if (phases.animate === "running") {
    return "animating";
  }
  if (phases.drawing !== "editing") {
    return DRAWING_STATES[phases.drawing];
  }
  if (phases.animate === "done") {
    return "result";
  }
  if (phases.animate === "failed") {
    return "animateFailed";
  }
  return phases.motionChosen ? "motionReady" : "dotsEditing";
}

export function canAnimate(phases: Phases): boolean {
  return phases.drawing === "editing" && phases.motionChosen && phases.animate !== "running";
}
