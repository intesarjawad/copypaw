import "./style.css";
import { LocalBackend, type ComputeBackend, type Health } from "./backend";
import { canAnimate, pageState, type Phases } from "./states";
import { STRINGS, TABLE } from "./strings";
import { TabBackend } from "./tabBackend";
import { lightTable, runSection } from "./lightTable/table";
import { toolGlass } from "./lightTable/toolGlass";
import { animateStep } from "./ui/animate";
import { el, showError } from "./ui/dom";
import { drawingStep } from "./ui/drawing";
import { motionStep } from "./ui/motion";

const HEALTH_POLL_MS = 2000;

const root = document.getElementById("app");
if (root === null) {
  throw new Error("The page has no #app element.");
}

// The local app passes its token in the URL fragment, which browsers never send to the
// server; a page opened without one runs everything in the tab.
const token = new URLSearchParams(location.hash.slice(1)).get("t");
const backend: ComputeBackend = token === null ? new TabBackend() : new LocalBackend(token);

let phases: Phases = { drawing: "empty", motionChosen: false, animate: "idle" };

function update(change: Partial<Phases>): void {
  phases = { ...phases, ...change };
  document.body.dataset.state = pageState(phases);
  animate.setEnabled(
    canAnimate(phases),
    phases.drawing === "editing" ? STRINGS.animateWaitsMotion : STRINGS.animateWaits,
  );
}

function changed(): Partial<Phases> {
  return phases.animate === "running" ? {} : { animate: "idle" };
}

const drawing = drawingStep(backend, (phase) => {
  update({ drawing: phase, ...changed() });
});
const motion = motionStep(backend, () => {
  update({ motionChosen: motion.choice() !== null, ...changed() });
});
const sheets = el("div", { class: "sheets" }, drawing.element, motion.element);
const animate = animateStep(
  backend,
  () => {
    const current = drawing.current();
    const choice = motion.choice();
    return current === null || choice === null
      ? null
      : { rig: current.rig.id, joints: current.joints, motion: choice };
  },
  (phase) => {
    update({ animate: phase });
  },
  toolGlass(sheets),
  token !== null,
);
sheets.append(animate.element, el("div", { class: "rail" }, drawing.controls, animate.controls));

const models = el("span", { role: "status", class: "models" });
const errors = el("div", { role: "alert" });

function showModels(health: Health): void {
  const { state, downloaded, total } = health.models;
  const text: Record<typeof state, string> = {
    ready: STRINGS.modelsReady,
    failed: STRINGS.modelsFailed,
    absent: STRINGS.modelsAbsent,
    downloading: STRINGS.modelsDownloading(downloaded, total),
  };
  models.textContent = text[state];
  motion.allowClips(state !== "absent");
}

async function watchModels(): Promise<void> {
  try {
    let health = await backend.health();
    motion.showGallery(health.gallery);
    showModels(health);
    while (health.models.state === "downloading") {
      await new Promise((resolve) => setTimeout(resolve, HEALTH_POLL_MS));
      health = await backend.health();
      showModels(health);
    }
  } catch (error) {
    showError(errors, error);
  }
}

document.title = STRINGS.title;
root.id = "trace";
root.setAttribute("aria-labelledby", "trace-heading");
root.append(
  el(
    "div",
    { class: "trace-head" },
    el("h2", { id: "trace-heading" }, TABLE.traceHeading),
    el("p", {}, TABLE.traceLede, " ", models),
    errors,
  ),
  sheets,
);
root.before(lightTable(), runSection());
update({});
void watchModels();
