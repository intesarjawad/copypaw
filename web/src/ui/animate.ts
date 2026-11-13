import {
  BackendError,
  type ComputeBackend,
  type Job,
  type JobResults,
  type LiveResults,
  type MotionChoice,
} from "../backend";
import type { Joints } from "../joints";
import type { AnimatePhase } from "../states";
import { STRINGS } from "../strings";
import { el, showError, svg } from "./dom";
import type { ToolGlass } from "../lightTable/toolGlass";
import { clipAlone, traceOver } from "../lightTable/underlay";
import { readSheet, STILL_FRAME } from "../lightTable/sheet";
import record from "../lightTable/sheet.json";
import { animationPlayer, type Player } from "./draw";
import { followClip, readSkeleton } from "./skeleton";

const POLL_MS = 500;
const MS_PER_SECOND = 1000;
const SEEK_STEPS = 1000;

export interface AnimateRequest {
  readonly rig: string;
  readonly joints: Joints;
  readonly motion: MotionChoice;
}

export interface AnimateStep {
  readonly element: HTMLElement;
  /** Animate, Run and the frame slot, for the table's front rail. */
  readonly controls: HTMLElement;
  setEnabled(enabled: boolean, reason: string): void;
}

type FinishedJob = Exclude<Job, { readonly state: "queued" | "running" }>;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function animateStep(
  backend: ComputeBackend,
  request: () => AnimateRequest | null,
  onPhase: (phase: AnimatePhase) => void,
  glass: ToolGlass,
  /** Whether results are saved as files, which can carry the credit line. */
  saves: boolean,
): AnimateStep {
  const button = el("button", { type: "button", class: "animate" }, STRINGS.animate);
  const credit = el("input", { type: "checkbox", id: "credit-input" });
  credit.checked = true;
  const progress = el("progress", { "aria-label": STRINGS.progressLabel });
  progress.hidden = true;
  const status = el("p", { role: "status" });
  const errors = el("div", { role: "alert" });

  const video = el("video", {
    id: "result-video",
    loop: "",
    muted: "",
    playsinline: "",
    "aria-label": STRINGS.resultVideo,
  });
  const videoToggle = el("button", { type: "button" }, STRINGS.play);
  const videoSeek = el("input", {
    type: "range",
    min: "0",
    max: String(SEEK_STEPS),
    value: "0",
    "aria-label": STRINGS.position,
  });
  const videoPlayer = el(
    "div",
    { class: "player" },
    video,
    el("div", { class: "player-controls" }, videoToggle, videoSeek),
  );
  videoToggle.addEventListener("click", () => {
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  });
  video.addEventListener("play", () => {
    videoToggle.textContent = STRINGS.pause;
  });
  video.addEventListener("pause", () => {
    videoToggle.textContent = STRINGS.play;
  });
  video.addEventListener("timeupdate", () => {
    videoSeek.value = String(Math.round((video.currentTime / (video.duration || 1)) * SEEK_STEPS));
  });
  videoSeek.addEventListener("input", () => {
    video.pause();
    video.currentTime = (Number(videoSeek.value) / SEEK_STEPS) * (video.duration || 0);
  });
  const timing = el("p", { class: "timing" });
  const links = {
    gif: el("a", { download: "copypaw.gif" }, STRINGS.downloadGif),
    mp4: el("a", { download: "copypaw.mp4" }, STRINGS.downloadMp4),
    bvh: el("a", { download: "motion.bvh" }, STRINGS.downloadBvh),
    character: el("a", { download: "character.zip" }, STRINGS.downloadCharacter),
  };
  const explained = el("div", { class: "skeleton-view" });
  const live = el("div", { class: "live" });
  // Downloads and the tracked skeleton exist only for results the local app wrote.
  const files = el(
    "div",
    {},
    el("h5", {}, STRINGS.downloadsHeading),
    el("ul", { class: "downloads" }, ...Object.values(links).map((link) => el("li", {}, link))),
    el(
      "details",
      {},
      el("summary", {}, STRINGS.howItWorks),
      el("p", {}, STRINGS.howItWorksText),
      explained,
    ),
  );
  // Animate, and once a result plays its run button and frame slider, sit on the table's rail.
  // While Animate waits for a drawing, a note on the rail says what it needs.
  const waits = el("p", { class: "rail-note", id: "animate-waits" }, STRINGS.animateWaits);
  const rail = el("div", { class: "rail-bar" }, button, waits);
  let running = false;
  const result = el(
    "section",
    { class: "result", "aria-labelledby": "result-heading" },
    el("h4", { id: "result-heading", class: "visually-hidden" }, STRINGS.resultHeading),
    live,
    videoPlayer,
    timing,
    files,
  );
  result.hidden = true;

  // Before any drawing is animated, the result sheet shows the dog it will follow.
  const sheet = readSheet(record);
  const alone = clipAlone(sheet);
  const waiting = el("div", {
    class: "stage waiting",
    role: "img",
    "aria-label": STRINGS.waitingLabel,
  });
  waiting.style.aspectRatio = `${String(alone.view.width)} / ${String(alone.view.height)}`;
  let waitRequest = 0;
  function wait(nowMs: number): void {
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const frame = still ? STILL_FRAME : Math.floor((nowMs * sheet.fps) / MS_PER_SECOND);
    glass.trace({ stage: waiting, tracing: alone, frame });
    waitRequest = still ? 0 : requestAnimationFrame(wait);
  }
  function stopWaiting(): void {
    cancelAnimationFrame(waitRequest);
    waiting.hidden = true;
  }
  waitRequest = requestAnimationFrame(wait);

  const element = el(
    "section",
    { class: "step", "aria-labelledby": "animate-heading" },
    // The Animate button on the rail names the step; the heading is for screen readers.
    el("h3", { id: "animate-heading", class: "visually-hidden" }, STRINGS.animateHeading),
    ...(saves ? [el("label", { class: "check" }, credit, STRINGS.credit)] : []),
    progress,
    status,
    errors,
    waiting,
    result,
  );

  let objectUrls: string[] = [];
  let player: Player | null = null;

  function showTiming(sentence: string | null): void {
    timing.textContent = sentence ?? "";
    timing.hidden = sentence === null;
  }

  function showLive(results: LiveResults, sentence: string | null, motion: MotionChoice): void {
    player?.stop();
    stopWaiting();
    glass.trace(null);
    // The light table's clip is the one the gallery walk was tracked from, so it can lie under it.
    const tracedFrom = motion.kind === "gallery" && motion.id === record.scene;
    const { animation } = results;
    const tracing = tracedFrom ? traceOver(animation, readSheet(record)) : null;
    player = animationPlayer(
      animation,
      results.drawing,
      tracing === null
        ? undefined
        : {
            tracing,
            show: (frame, stage) => {
              glass.trace({ stage, tracing, frame });
            },
          },
    );
    live.replaceChildren(player.element);
    rail.replaceChildren(button, waits, player.controls);
    // Once there is a result, Run is the button to press; Animate stays for another take.
    button.classList.add("used");
    live.hidden = false;
    videoPlayer.hidden = true;
    files.hidden = true;
    showTiming(sentence);
    result.hidden = false;
  }

  async function blobUrl(url: string): Promise<string> {
    const made = URL.createObjectURL(await backend.file(url));
    objectUrls.push(made);
    return made;
  }

  async function poll(id: string): Promise<FinishedJob> {
    for (;;) {
      const job = await backend.job(id);
      if (job.state === "done" || job.state === "failed") {
        return job;
      }
      status.textContent = STRINGS.stages[job.stage];
      await delay(POLL_MS);
    }
  }

  async function showResult(results: JobResults, sentence: string | null): Promise<void> {
    for (const old of objectUrls) {
      URL.revokeObjectURL(old);
    }
    objectUrls = [];
    const [videoUrl, gif, mp4, bvh, character, clip] = await Promise.all([
      blobUrl(results.video),
      blobUrl(results.gif),
      blobUrl(results.mp4),
      blobUrl(results.bvh),
      blobUrl(results.character),
      blobUrl(results.clip),
    ]);
    const skeletonJson: unknown = JSON.parse(await (await backend.file(results.skeleton)).text());
    const skeleton = readSkeleton(skeletonJson);
    player?.stop();
    player = null;
    stopWaiting();
    glass.trace(null);
    rail.replaceChildren(button, waits);
    button.classList.add("used");
    live.hidden = true;
    videoPlayer.hidden = false;
    files.hidden = false;
    video.src = videoUrl;
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      // Muted, so the browser lets it start on its own; a refusal leaves the Play button.
      video.play().catch(() => undefined);
    }
    links.gif.href = gif;
    links.mp4.href = mp4;
    links.bvh.href = bvh;
    links.character.href = character;
    showTiming(sentence);
    const source = el("video", {
      src: clip,
      controls: "",
      muted: "",
      playsinline: "",
      "aria-label": STRINGS.skeletonLabel,
    });
    const overlay = svg("svg", { class: "overlay skeleton", "aria-hidden": "true" });
    followClip(source, overlay, skeleton);
    explained.replaceChildren(source, overlay);
    result.hidden = false;
  }

  async function run(): Promise<void> {
    const current = request();
    if (current === null) {
      return;
    }
    errors.replaceChildren();
    status.textContent = STRINGS.stages.queued;
    progress.hidden = false;
    running = true;
    onPhase("running");
    try {
      await backend.saveJoints(current.rig, current.joints);
      const job = await poll(await backend.animate(current.rig, current.motion, credit.checked));
      if (job.state === "failed") {
        throw new BackendError(job.error);
      }
      if (job.results.kind === "live") {
        showLive(job.results, job.timing, current.motion);
      } else {
        await showResult(job.results, job.timing);
      }
      onPhase("done");
    } catch (error) {
      showError(errors, error);
      onPhase("failed");
    } finally {
      running = false;
      progress.hidden = true;
      status.textContent = "";
    }
  }

  button.addEventListener("click", () => {
    void run();
  });

  return {
    element,
    controls: rail,
    setEnabled: (enabled, reason) => {
      button.disabled = !enabled;
      waits.textContent = reason;
      waits.hidden = enabled || running;
      if (waits.hidden) {
        button.removeAttribute("aria-describedby");
      } else {
        button.setAttribute("aria-describedby", waits.id);
      }
    },
  };
}
