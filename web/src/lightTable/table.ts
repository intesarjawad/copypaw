import { TABLE } from "../strings";
import { el } from "../ui/dom";
import type { Backlight } from "./backlight";
import record from "./sheet.json";
import {
  lampLevel,
  lampSettled,
  place,
  readSheet,
  STILL_FRAME,
  wrapFrame,
  type Rect,
  type Sheet,
} from "./sheet";

const BASE = `${import.meta.env.BASE_URL}light-table/`;
const MS_PER_SECOND = 1000;
const ROW_MARK_EVERY = 6;
const PAGE_FRAMES = 6;
const AIM_EASE = 0.14;
// The front rail runs this much deeper than the others to carry the lamp switch and the reel,
// as the desk's bottom padding in style.css allows for.
const RAIL_APRON = 56;
const AIM_MARGIN = 60;
// The desk lamp stands low in the room just off the table's left edge, level with the top of the
// sheet, so it rakes across it; its base and arm reach 74 px further left. Where the room is too
// narrow to show the whole lamp, it stands out of sight and its light still comes in from the left.
const LAMP_AT = { x: -34, y: 0.15, nearest: 74, unseen: -80 };

function lampX(desk: Rect): number {
  const x = desk.x + LAMP_AT.x;
  return x < LAMP_AT.nearest ? LAMP_AT.unseen : x;
}
// Left alone it points a little left of the sheet's middle; a mouse over the table turns it.
const AIM_REST = { x: 0.4, y: 0.45 };
const MOBILE = "(max-width: 700px), (pointer: coarse)";

function xsheet(
  sheet: Sheet,
  onPick: (frame: number) => void,
): {
  element: HTMLElement;
  mark: (frame: number) => void;
} {
  const short = (paw: string): string => TABLE.pawShort[paw] ?? paw;
  const pawHeads = sheet.paws.map((paw) =>
    el("th", { scope: "col" }, el("abbr", { title: paw }, short(paw))),
  );
  const head = el(
    "thead",
    {},
    el(
      "tr",
      {},
      el("th", { rowspan: "2", scope: "col", class: "fr" }, TABLE.frameColumn),
      el("th", { colspan: String(sheet.paws.length), scope: "colgroup", class: "dog" }, TABLE.dog),
      el(
        "th",
        { colspan: String(sheet.paws.length), scope: "colgroup", class: "drawing" },
        TABLE.drawing,
      ),
    ),
    el("tr", {}, ...pawHeads, ...pawHeads.map((cell) => cell.cloneNode(true))),
  );
  const rows = Array.from({ length: sheet.frames }, (_, frame) => {
    const marks = (tracks: readonly (readonly boolean[])[], kind: string): HTMLElement[] =>
      tracks.map((track, paw) => {
        const planted = track[frame] === true;
        const lands = planted && track[frame - 1] !== true;
        const classes = [kind, planted ? "on" : "", lands ? "lands" : "", paw === 0 ? "first" : ""];
        return el("td", { class: classes.filter(Boolean).join(" ") }, planted ? "1" : "0");
      });
    const row = el(
      "tr",
      { class: (frame + 1) % ROW_MARK_EVERY === 0 || frame === 0 ? "marked" : "" },
      el("th", { scope: "row" }, String(frame + 1)),
      ...marks(sheet.dog, "dog"),
      ...marks(sheet.drawing, "drawing"),
    );
    row.style.setProperty("--row", String(frame));
    row.addEventListener("click", () => {
      onPick(frame);
    });
    return row;
  });
  const element = el(
    "aside",
    { class: "xsheet", "aria-labelledby": "xsheet-caption" },
    el(
      "dl",
      { class: "xsheet-head" },
      el("div", {}, el("dt", {}, TABLE.scene), el("dd", {}, sheet.scene)),
      el("div", {}, el("dt", {}, TABLE.drawing), el("dd", {}, sheet.character)),
      el("div", {}, el("dt", {}, TABLE.frames), el("dd", {}, String(sheet.frames))),
    ),
    el(
      "table",
      {},
      el("caption", { id: "xsheet-caption", class: "visually-hidden" }, TABLE.xsheetCaption),
      head,
      el("tbody", {}, ...rows),
    ),
    el("p", { class: "xsheet-key" }, TABLE.sheetKey),
  );
  let current: HTMLElement | undefined;
  return {
    element,
    mark: (frame) => {
      current?.classList.remove("now");
      current = rows[frame];
      current?.classList.add("now");
    },
  };
}

function relative(inner: DOMRect, outer: DOMRect): Rect {
  return {
    x: inner.left - outer.left,
    y: inner.top - outer.top,
    width: inner.width,
    height: inner.height,
  };
}

/** Below the table: how to run it on your own dog, and what this sample clip can and cannot show. */
export function runSection(): HTMLElement {
  const sheet = readSheet(record);
  return el(
    "section",
    { class: "run", id: "run", "aria-labelledby": "run-heading" },
    el("h2", { id: "run-heading" }, TABLE.runHeading),
    el("p", {}, TABLE.runWhere),
    el(
      "ol",
      { class: "commands" },
      ...TABLE.commands.map((command) => el("li", {}, el("code", {}, command))),
    ),
    el("p", {}, TABLE.runNext),
    el(
      "div",
      { class: "footnote" },
      el("p", {}, TABLE.honest(sheet.frames / sheet.fps, sheet.landings, sheet.landingsNeeded)),
    ),
  );
}

/**
 * The page's first screen: the dog clip glowing under a sheet of paper on a light table, the
 * rigged drawing walking on top of it, and footfall.json as an exposure sheet beside them.
 */
export function lightTable(): HTMLElement {
  const sheet = readSheet(record);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const canvas = el("canvas", {
    class: "glass",
    role: "img",
    "aria-label": TABLE.stageLabel(sheet.scene, sheet.character),
  });
  const still = el("img", { class: "still", src: `${BASE}still.jpg`, alt: "" });
  const projection = el("div", {
    class: "projection",
    role: "slider",
    tabindex: "0",
    "aria-label": TABLE.frameLabel,
    "aria-valuemin": "1",
    "aria-valuemax": String(sheet.frames),
  });
  projection.style.aspectRatio = String(sheet.clip.atlas.cellWidth / sheet.clip.atlas.cellHeight);
  const paper = el("div", { class: "paper" }, projection);
  const lampButton = el(
    "button",
    { type: "button", class: "lamp", "aria-pressed": "true" },
    el("span", { class: "plate", "aria-hidden": "true" }, el("span", { class: "rocker" })),
    TABLE.lamp,
  );
  const reelLabel = el("span", {});
  const reel = el(
    "button",
    { type: "button", class: "reel" },
    el("span", { class: "cap", "aria-hidden": "true" }),
    reelLabel,
  );
  const desk = el(
    "div",
    { class: "desk" },
    still,
    paper,
    el("div", { class: "rail-controls" }, lampButton, reel),
  );
  const counter = el("output", { class: "counter" });
  const hint = el("p", { class: "hint" }, TABLE.hint);
  const sheetView = xsheet(sheet, (picked) => {
    setPlaying(false);
    show(picked);
  });
  const element = el(
    "header",
    { class: "table", "data-lamp": "on" },
    canvas,
    el(
      "div",
      { class: "intro" },
      el("div", {}, el("p", { class: "mark" }, TABLE.mark), el("h1", {}, TABLE.headline)),
      el(
        "div",
        {},
        el("p", { class: "lede" }, TABLE.lede),
        el(
          "p",
          { class: "pointer" },
          TABLE.pointer[0],
          el("a", { href: "#trace" }, TABLE.pointer[1]),
          TABLE.pointer[2],
        ),
      ),
    ),
    el("div", { class: "bench" }, desk, sheetView.element),
    el("div", { class: "under" }, counter, hint),
  );

  let frame = reduced ? STILL_FRAME : 0;
  let playing = !reduced;
  let startedMs = 0;
  let lampOn = true;
  let switchedMs = Number.NEGATIVE_INFINITY;
  let onScreen = true;
  let request = 0;
  let renderer: Backlight | null = null;
  // Without WebGL the still frame stands in, so the frame can no longer change.
  let frozen = false;
  // The desk lamp turns to follow a mouse over the table and otherwise turns back to its rest.
  const aim = { x: Number.NaN, y: Number.NaN, pointer: null as { x: number; y: number } | null };

  function show(next: number): void {
    if (frozen) {
      return;
    }
    frame = wrapFrame(next, sheet.frames);
    const text = TABLE.frameText(frame, sheet.frames);
    projection.setAttribute("aria-valuenow", String(frame + 1));
    projection.setAttribute("aria-valuetext", text);
    counter.textContent = text;
    sheetView.mark(frame);
    schedule();
  }

  /** Turns the desk lamp a step towards where it should point; true while it is still turning. */
  function turnLamp(sheetBox: Rect): boolean {
    const target = aim.pointer ?? {
      x: sheetBox.x + sheetBox.width * AIM_REST.x,
      y: sheetBox.y + sheetBox.height * AIM_REST.y,
    };
    if (reduced || Number.isNaN(aim.x)) {
      aim.x = target.x;
      aim.y = target.y;
    }
    aim.x += (target.x - aim.x) * AIM_EASE;
    aim.y += (target.y - aim.y) * AIM_EASE;
    return Math.hypot(target.x - aim.x, target.y - aim.y) > 0.5;
  }

  function draw(nowMs: number): boolean {
    const box = element.getBoundingClientRect();
    const sheetBox = relative(paper.getBoundingClientRect(), box);
    const deskBox = relative(desk.getBoundingClientRect(), box);
    const placed = place(sheet, frame, relative(projection.getBoundingClientRect(), box));
    const moving = turnLamp(sheetBox);
    renderer?.draw({
      frame,
      desk: deskBox,
      paper: sheetBox,
      clip: placed.clip,
      figure: placed.figure,
      floorY: placed.floorY,
      apron: RAIL_APRON,
      lamp: lampLevel(lampOn, reduced ? Number.POSITIVE_INFINITY : nowMs - switchedMs),
      deskLight: [lampX(deskBox), sheetBox.y + sheetBox.height * LAMP_AT.y],
      deskAim: [aim.x, aim.y],
    });
    return moving;
  }

  function tick(): void {
    // The switch time comes from performance.now(), so the frame time must too.
    const nowMs = performance.now();
    request = 0;
    if (!onScreen || document.hidden) {
      return;
    }
    if (playing) {
      const next = Math.floor(((nowMs - startedMs) * sheet.fps) / MS_PER_SECOND);
      if (wrapFrame(next, sheet.frames) !== frame) {
        show(next);
      }
    }
    const moving = draw(nowMs);
    const settling = !lampSettled(lampOn, nowMs - switchedMs) || (moving && !lampOn);
    if (playing || (settling && !reduced)) {
      schedule();
    }
  }
  function schedule(): void {
    if (request === 0) {
      request = requestAnimationFrame(tick);
    }
  }

  function setPlaying(next: boolean): void {
    playing = next;
    reelLabel.textContent = next ? TABLE.stop : TABLE.run;
    // While the reel runs, each mark is pencilled in as its frame passes.
    sheetView.element.classList.toggle("live", next);
    startedMs = performance.now() - (frame * MS_PER_SECOND) / sheet.fps;
    schedule();
  }

  lampButton.addEventListener("click", () => {
    lampOn = !lampOn;
    switchedMs = performance.now();
    lampButton.setAttribute("aria-pressed", String(lampOn));
    element.dataset.lamp = lampOn ? "on" : "off";
    schedule();
  });
  reel.addEventListener("click", () => {
    setPlaying(!playing);
  });

  // Dragging walks the dog: the frame follows the hand at the dog's own pace across the glass.
  let drag: { x: number; frame: number; wasPlaying: boolean } | null = null;
  projection.addEventListener("pointerdown", (event) => {
    drag = { x: event.clientX, frame, wasPlaying: playing };
    projection.setPointerCapture(event.pointerId);
    setPlaying(false);
  });
  projection.addEventListener("pointermove", (event) => {
    if (drag === null) {
      return;
    }
    const first = sheet.clip.paws[0] ?? 0;
    const last = sheet.clip.paws[sheet.frames - 1] ?? 0;
    const step = ((last - first) * projection.getBoundingClientRect().width) / sheet.frames;
    const moved = Math.round((event.clientX - drag.x) / (step === 0 ? 1 : step));
    show(Math.min(sheet.frames - 1, Math.max(0, drag.frame + moved)));
  });
  const release = (): void => {
    if (drag?.wasPlaying === true) {
      setPlaying(true);
    }
    drag = null;
  };
  projection.addEventListener("pointerup", release);
  projection.addEventListener("pointercancel", release);
  projection.addEventListener("keydown", (event) => {
    const steps: Readonly<Record<string, number>> = {
      ArrowRight: 1,
      ArrowUp: 1,
      ArrowLeft: -1,
      ArrowDown: -1,
      PageUp: PAGE_FRAMES,
      PageDown: -PAGE_FRAMES,
    };
    const step = steps[event.key];
    let next: number | null = step === undefined ? null : frame + step;
    if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = sheet.frames - 1;
    }
    if (next !== null) {
      event.preventDefault();
      setPlaying(false);
      show(next);
    }
  });
  element.addEventListener("pointermove", (event) => {
    if (event.pointerType !== "mouse") {
      return;
    }
    const box = element.getBoundingClientRect();
    const table = desk.getBoundingClientRect();
    // Only over the table: a lamp turned on the words would wash out the room behind them.
    const over =
      Math.abs(event.clientX - (table.left + table.width / 2)) < table.width / 2 + AIM_MARGIN &&
      Math.abs(event.clientY - (table.top + table.height / 2)) < table.height / 2 + AIM_MARGIN;
    aim.pointer = over ? { x: event.clientX - box.left, y: event.clientY - box.top } : null;
    schedule();
  });
  element.addEventListener("pointerleave", () => {
    aim.pointer = null;
    schedule();
  });

  new IntersectionObserver(([entry]) => {
    onScreen = entry?.isIntersecting ?? true;
    if (onScreen) {
      setPlaying(playing);
    }
  }).observe(element);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      setPlaying(playing);
    }
  });

  function fallBack(): void {
    element.classList.add("no-glass");
    lampButton.hidden = true;
    reel.hidden = true;
    hint.textContent = TABLE.stillHint(sheet.frames);
    setPlaying(false);
    show(STILL_FRAME);
    frozen = true;
    projection.inert = true;
  }

  async function lightUp(): Promise<void> {
    const images = ["dog.jpg", "matte.jpg", "drawing.jpg"].map((name) => {
      const image = new Image();
      image.src = `${BASE}${name}`;
      return image;
    });
    const [{ backlight }] = await Promise.all([
      import("./backlight"),
      ...images.map((image) => image.decode()),
    ]);
    const [clipFrames, clipMatte, figureFrames] = images;
    if (clipFrames === undefined || clipMatte === undefined || figureFrames === undefined) {
      return;
    }
    renderer = backlight(
      canvas,
      clipFrames,
      clipMatte,
      sheet.clip.atlas,
      figureFrames,
      sheet.figure.atlas,
    );
    if (renderer === null) {
      fallBack();
      return;
    }
    const size = (): void => {
      const box = element.getBoundingClientRect();
      const ratio = matchMedia(MOBILE).matches ? 1 : Math.min(devicePixelRatio, 2);
      renderer?.resize(box.width, box.height, ratio);
      draw(performance.now());
    };
    new ResizeObserver(size).observe(element);
    size();
    element.classList.add("lit");
    // The tubes strike as the table first lights, unless motion is reduced.
    if (!reduced) {
      switchedMs = performance.now();
    }
    setPlaying(playing);
  }

  setPlaying(playing);
  show(frame);
  // The words and the sheet paint first; the GPU starts once they are on screen.
  requestAnimationFrame(() => {
    setTimeout(() => {
      lightUp().catch(fallBack);
    }, 0);
  });
  return element;
}
