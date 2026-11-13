import { el } from "../ui/dom";
import type { Backlight } from "./backlight";
import record from "./sheet.json";
import { readSheet, type Rect } from "./sheet";
import type { Tracing } from "./underlay";

const BASE = `${import.meta.env.BASE_URL}light-table/`;
const MOBILE = "(max-width: 1000px)";
// The width of the steel rail round the glass, as the shader draws it.
const RAIL = 16;
// The tool's table is always lit, so its desk lamp stays out of sight off the left edge.
const DESK_LIGHT = { x: -260, y: 0.15 };

/** The dog's clip under a stage, and the frame it is at. */
export interface Traced {
  readonly stage: HTMLElement;
  readonly tracing: Tracing;
  readonly frame: number;
}

export interface ToolGlass {
  /** Lay the clip under the stage at a frame, or take it away with null. */
  trace(traced: Traced | null): void;
}

function relative(inner: DOMRect, outer: DOMRect): Rect {
  return {
    x: inner.left - outer.left,
    y: inner.top - outer.top,
    width: inner.width,
    height: inner.height,
  };
}

function shown(element: Element | null): element is HTMLElement {
  return element instanceof HTMLElement && element.getClientRects().length > 0;
}

/**
 * The tool's table: the same rail, glass, paper and blue ink as the light table at the top of
 * the page, drawn behind the tool's sheets. Without WebGL the sheets keep their CSS glass.
 */
export function toolGlass(sheets: HTMLElement): ToolGlass {
  const sheet = readSheet(record);
  const canvas = el("canvas", { class: "tool-glass", "aria-hidden": "true" });
  sheets.prepend(canvas);
  let renderer: Backlight | null = null;
  let traced: Traced | null = null;
  let onScreen = true;
  let request = 0;
  let resized: ResizeObserver | null = null;

  function draw(): void {
    request = 0;
    if (renderer === null || !onScreen) {
      return;
    }
    const box = canvas.getBoundingClientRect();
    const desk = { x: 0, y: 0, width: box.width, height: box.height };
    const editor = sheets.querySelector(".editor");
    const empty = sheets.querySelector(".blank");
    const first = shown(editor) ? editor : empty;
    const rail = sheets.querySelector(".rail");
    const paper = first === null ? desk : relative(first.getBoundingClientRect(), box);
    let clip = { x: -1e4, y: -1e4, width: 1, height: 1 };
    let floorY = -1e4;
    let second: Rect | undefined;
    if (traced !== null && shown(traced.stage)) {
      const stage = relative(traced.stage.getBoundingClientRect(), box);
      const { clip: inWindow, view } = traced.tracing;
      const scale = stage.width / view.width;
      second = stage;
      clip = {
        x: stage.x + (inWindow.x - view.x) * scale,
        y: stage.y + (inWindow.y - view.y) * scale,
        width: inWindow.width * scale,
        height: inWindow.height * scale,
      };
      floorY = stage.y + (traced.tracing.floorY - view.y) * scale;
    }
    renderer.draw({
      frame: traced === null ? 0 : traced.tracing.clipFrame(traced.frame),
      desk,
      paper,
      second,
      clip,
      figure: { x: -1e4, y: -1e4, width: 1, height: 1 },
      floorY,
      lamp: 1,
      deskLight: [DESK_LIGHT.x, box.height * DESK_LIGHT.y],
      deskAim: [paper.x + paper.width / 2, paper.y + paper.height / 2],
      // The buttons sit on the rail, so the steel runs up to the top of their row.
      apron: rail === null ? 0 : Math.max(box.bottom - rail.getBoundingClientRect().top - RAIL, 0),
    });
  }

  function schedule(): void {
    if (request === 0) {
      request = requestAnimationFrame(draw);
    }
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
      canvas.remove();
      return;
    }
    const size = (): void => {
      const box = canvas.getBoundingClientRect();
      const ratio = matchMedia(MOBILE).matches ? 1 : Math.min(devicePixelRatio, 2);
      renderer?.resize(box.width, box.height, ratio);
      schedule();
    };
    const watch = new ResizeObserver(size);
    for (const element of [sheets, ...sheets.querySelectorAll(".editor, .blank")]) {
      watch.observe(element);
    }
    resized = watch;
    size();
    sheets.classList.add("lit");
  }

  new IntersectionObserver(([entry]) => {
    onScreen = entry?.isIntersecting ?? true;
    schedule();
  }).observe(sheets);
  // A new photo, an error or a result changes where the sheets lie.
  new MutationObserver(schedule).observe(document.body, { attributeFilter: ["data-state"] });
  lightUp().catch(() => {
    canvas.remove();
  });

  return {
    trace(next) {
      if (
        next?.stage === traced?.stage &&
        next?.tracing === traced?.tracing &&
        next?.frame === traced?.frame
      ) {
        return;
      }
      if (next !== null && next.stage !== traced?.stage) {
        resized?.observe(next.stage);
      }
      traced = next;
      schedule();
    },
  };
}
