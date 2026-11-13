import type { ComputeBackend, Rig } from "../backend";
import {
  JOINT_IDS,
  flipJoints,
  moveJoint,
  nextJoint,
  nudgeJoint,
  toImagePoint,
  type JointId,
  type Joints,
} from "../joints";
import type { DrawingPhase } from "../states";
import { JOINT_LABELS, STRINGS } from "../strings";
import { el, showError, svg } from "./dom";

const DOT_RADIUS_FRACTION = 0.012;
// A press this many pin radii from a pin picks it up; further away it moves the chosen pin there.
const PIN_REACH = 3;

export interface RiggedDrawing {
  readonly rig: Rig;
  readonly joints: Joints;
}

export interface DrawingStep {
  readonly element: HTMLElement;
  /** Choose a photo and Flip, for the table's front rail once a sheet is on the glass. */
  readonly controls: HTMLElement;
  current(): RiggedDrawing | null;
}

export function drawingStep(
  backend: ComputeBackend,
  onPhase: (phase: DrawingPhase) => void,
): DrawingStep {
  const input = el("input", {
    type: "file",
    id: "drawing-input",
    accept: "image/*",
    capture: "environment",
  });
  const status = el("p", { role: "status" });
  const errors = el("div", { role: "alert" });
  const photo = el("img", { alt: "" });
  const overlay = svg("svg", { class: "overlay" });
  const heads = pinHeads();
  const figure = el(
    "figure",
    { class: "editor", role: "group", "aria-label": STRINGS.editorLabel },
    photo,
    overlay,
  );
  const flip = el("button", { type: "button" }, STRINGS.flip);
  // A phone's file input offers the camera too, and nothing can be dropped on it.
  const touch = matchMedia("(pointer: coarse)").matches;
  const photoLabel = el(
    "label",
    { for: "drawing-input" },
    touch ? STRINGS.takePhoto : STRINGS.choosePhoto,
  );
  const heading = el("h3", { id: "drawing-heading" }, STRINGS.drawingHeading);
  const pinBar = el("div", { class: "pin-bar" }, flip);
  const editing = el("div", { class: "editing" }, figure);
  editing.hidden = true;
  pinBar.hidden = true;
  // The blank sheet the photo will replace; the heading stays above it, where it sits over the photo.
  const blank = el(
    "div",
    { class: "blank" },
    photoLabel,
    ...(touch ? [] : [el("p", { class: "hint" }, STRINGS.dropHint)]),
    status,
    errors,
  );
  const element = el(
    "section",
    { class: "step drop-zone", "aria-labelledby": "drawing-heading" },
    heading,
    input,
    blank,
    editing,
  );

  let rig: Rig | null = null;
  let joints: Joints | null = null;
  let selected: JointId = JOINT_IDS[0];
  let photoUrl: string | null = null;
  let selection = 0;
  const dots = new Map<JointId, SVGCircleElement>();
  const shadows = new Map<JointId, SVGGElement>();
  let dragging: JointId | null = null;

  function draw(): void {
    if (joints === null) {
      return;
    }
    for (const id of JOINT_IDS) {
      const { x, y } = joints[id];
      shadows.get(id)?.setAttribute("transform", `translate(${String(x)} ${String(y)})`);
      const dot = dots.get(id);
      if (dot !== undefined) {
        dot.setAttribute("cx", String(joints[id].x));
        dot.setAttribute("cy", String(joints[id].y));
        dot.classList.toggle("selected", id === selected);
        dot.setAttribute("aria-pressed", String(id === selected));
      }
    }
  }

  function select(id: JointId): void {
    selected = id;
    draw();
  }

  function showRig(next: Rig): void {
    rig = next;
    joints = next.joints;
    overlay.setAttribute("viewBox", `0 0 ${String(next.width)} ${String(next.height)}`);
    figure.style.aspectRatio = `${String(next.width)} / ${String(next.height)}`;
    // Pins map onto the photo only while the sheet keeps its shape, so its width follows the height cap.
    figure.style.width = `min(100%, 70vh * ${String(next.width)} / ${String(next.height)})`;
    const size = Math.max(next.width, next.height) * DOT_RADIUS_FRACTION;
    const radius = String(size);
    const [fallX, fallY] = [String(size * 0.8), String(size * 1.1)];
    dots.clear();
    shadows.clear();
    // A map pin stands on its needle, so its head's shadow falls clear of it, down and to the
    // right, joined to the hole by the needle's own shadow.
    const pinShadows = JOINT_IDS.map((id) => {
      const shadow = svg("g", { class: "pin-shadow", "aria-hidden": "true" });
      shadow.append(
        svg("line", { x1: "0", y1: "0", x2: fallX, y2: fallY }),
        svg("circle", { cx: fallX, cy: fallY, r: radius }),
      );
      shadows.set(id, shadow);
      return shadow;
    });
    const circles = JOINT_IDS.map((id) => {
      // Each pin is a button: Tab reaches it, focusing or clicking chooses it, arrows nudge it.
      const dot = svg("circle", {
        class: "dot",
        r: radius,
        "data-joint": id,
        tabindex: "0",
        role: "button",
        "aria-label": JOINT_LABELS[id],
      });
      dot.addEventListener("focus", () => {
        select(id);
      });
      dots.set(id, dot);
      return dot;
    });
    overlay.replaceChildren(heads, ...pinShadows, ...circles);
    editing.hidden = false;
    pinBar.hidden = false;
    blank.hidden = true;
    // With a sheet on the glass, choosing another photo moves down to the rail.
    pinBar.prepend(photoLabel);
    select(JOINT_IDS[0]);
  }

  async function load(file: File): Promise<void> {
    const current = ++selection;
    rig = null;
    joints = null;
    editing.hidden = true;
    pinBar.hidden = true;
    blank.hidden = false;
    blank.prepend(photoLabel);
    errors.replaceChildren();
    status.textContent = STRINGS.rigging;
    if (photoUrl !== null) {
      URL.revokeObjectURL(photoUrl);
    }
    photoUrl = URL.createObjectURL(file);
    photo.src = photoUrl;
    onPhase("loading");
    try {
      const next = await backend.rig(file);
      if (current !== selection) return;
      await Promise.all([photo.decode(), document.fonts.ready]);
      if (current !== selection) return;
      showRig(next);
      onPhase("editing");
    } catch (error) {
      if (current !== selection) return;
      showError(errors, error);
      onPhase("failed");
    } finally {
      if (current === selection) status.textContent = "";
    }
  }

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    // Cleared so choosing the same photo again, say after an error, still loads it.
    input.value = "";
    if (file !== undefined) {
      void load(file);
    }
  });
  element.addEventListener("dragover", (event) => {
    event.preventDefault();
  });
  element.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files[0];
    if (file?.type.startsWith("image/")) {
      void load(file);
    }
  });
  overlay.addEventListener("pointerdown", (event) => {
    if (rig === null || joints === null) {
      return;
    }
    const placed = joints;
    const at = toImagePoint(
      { x: event.clientX, y: event.clientY },
      overlay.getBoundingClientRect(),
      rig,
    );
    const reach = PIN_REACH * Math.max(rig.width, rig.height) * DOT_RADIUS_FRACTION;
    const distance = (id: JointId): number => Math.hypot(placed[id].x - at.x, placed[id].y - at.y);
    const nearest = JOINT_IDS.reduce((best, id) => (distance(id) < distance(best) ? id : best));
    if (distance(nearest) <= reach) {
      dragging = nearest;
      overlay.setPointerCapture(event.pointerId);
      select(nearest);
    } else {
      joints = moveJoint(joints, selected, at);
      select(nextJoint(selected));
    }
  });
  overlay.addEventListener("pointermove", (event) => {
    if (rig === null || joints === null || dragging === null) {
      return;
    }
    const at = toImagePoint(
      { x: event.clientX, y: event.clientY },
      overlay.getBoundingClientRect(),
      rig,
    );
    joints = moveJoint(joints, dragging, at);
    draw();
  });
  const drop = (): void => {
    dragging = null;
  };
  overlay.addEventListener("pointerup", drop);
  overlay.addEventListener("pointercancel", drop);
  figure.addEventListener("keydown", (event) => {
    if (rig === null || joints === null) {
      return;
    }
    const moved = nudgeJoint(joints, selected, event.key, event.shiftKey, rig.boxHeight);
    if (moved !== null) {
      event.preventDefault();
      joints = moved;
      draw();
    }
  });
  flip.addEventListener("click", () => {
    if (rig !== null && joints !== null) {
      joints = flipJoints(joints, rig.outline);
      draw();
    }
  });

  return {
    element,
    controls: pinBar,
    current: () => (rig === null || joints === null ? null : { rig, joints }),
  };
}

/** Map pins seen from above: a domed head lit from the top left, steel or, when chosen, red. */
function pinHeads(): SVGDefsElement {
  const defs = svg("defs");
  const shades = {
    "pin-head": ["#fff4cf", "#d9a520", "#6e4c06"],
    "pin-chosen": ["#ffb8ad", "#b52a1c", "#560c05"],
  } as const;
  for (const [id, [light, body, edge]] of Object.entries(shades)) {
    const gradient = svg("radialGradient", { id, cx: "0.36", cy: "0.32", r: "0.72" });
    for (const [offset, colour] of [
      ["0", light],
      ["0.45", body],
      ["1", edge],
    ] as const) {
      gradient.append(svg("stop", { offset, "stop-color": colour }));
    }
    defs.append(gradient);
  }
  return defs;
}
