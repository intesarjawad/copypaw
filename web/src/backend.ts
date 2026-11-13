import messages from "../../copypaw/messages.json";
import type { Animation } from "./compute/animate";
import { JOINT_IDS, jointsFrom, type Joints, type Point } from "./joints";
import { field, list, number, oneOf, point, text } from "./parse";

export const TOKEN_HEADER = "X-Copypaw-Token";

export interface ErrorDetail {
  readonly code: string;
  readonly what: string;
  readonly next: string;
}

export class BackendError extends Error {
  constructor(readonly detail: ErrorDetail) {
    super(detail.what);
    this.name = "BackendError";
  }
}

export const INTERNAL_ERROR: ErrorDetail = { code: "internal", ...messages.internal };

export interface Rig {
  readonly id: string;
  readonly width: number;
  readonly height: number;
  readonly boxHeight: number;
  readonly outline: readonly Point[];
  readonly joints: Joints;
}

export interface GalleryMotion {
  readonly id: string;
  readonly thumbnail: string;
}

/** "absent" when the backend tracks no clips at all, so only gallery motions work. */
export type ModelState = "downloading" | "ready" | "failed" | "absent";

export interface Health {
  readonly version: string;
  readonly models: {
    readonly state: ModelState;
    readonly downloaded: number;
    readonly total: number;
  };
  readonly gallery: readonly GalleryMotion[];
}

export type MotionChoice =
  | { readonly kind: "gallery"; readonly id: string }
  | { readonly kind: "capture"; readonly job: string };

/** Files the local app wrote, fetched by URL. */
export interface JobResults {
  readonly kind: "files";
  readonly video: string;
  readonly gif: string;
  readonly mp4: string;
  readonly bvh: string;
  readonly character: string;
  readonly clip: string;
  readonly skeleton: string;
}

/** An animation computed in the tab, played on the page instead of as a video. */
export interface LiveResults {
  readonly kind: "live";
  readonly animation: Animation;
  readonly drawing: ImageBitmap;
}

export type JobStage = "queued" | "models" | "capture" | "render" | "done";

export type Job =
  | { readonly state: "queued" | "running"; readonly stage: JobStage }
  | {
      readonly state: "done";
      readonly results: JobResults | LiveResults;
      readonly timing: string | null;
    }
  | { readonly state: "failed"; readonly error: ErrorDetail };

/** The compute the page needs, whether it runs in the local app or in the tab. */
export interface ComputeBackend {
  health(): Promise<Health>;
  rig(image: Blob): Promise<Rig>;
  saveJoints(rig: string, joints: Joints): Promise<void>;
  capture(clip: Blob): Promise<string>;
  animate(rig: string, motion: MotionChoice, credit: boolean): Promise<string>;
  job(id: string): Promise<Job>;
  file(url: string): Promise<Blob>;
}

export function readError(value: unknown): ErrorDetail {
  try {
    return { code: text(value, "code"), what: text(value, "what"), next: text(value, "next") };
  } catch {
    return INTERNAL_ERROR;
  }
}

export function errorDetail(error: unknown): ErrorDetail {
  return error instanceof BackendError ? error.detail : INTERNAL_ERROR;
}

function readRig(value: unknown): Rig {
  const joints = field(value, "joints");
  return {
    id: text(value, "rig"),
    width: number(value, "width"),
    height: number(value, "height"),
    boxHeight: number(value, "box_height"),
    outline: list(value, "outline").map(point),
    joints: jointsFrom((id) => point(field(joints, id))),
  };
}

function readHealth(value: unknown): Health {
  const models = field(value, "models");
  return {
    version: text(value, "version"),
    models: {
      state: oneOf(text(models, "state"), ["downloading", "ready", "failed"]),
      downloaded: number(models, "downloaded"),
      total: number(models, "total"),
    },
    gallery: list(value, "gallery").map((entry) => ({
      id: text(entry, "id"),
      thumbnail: text(entry, "thumbnail"),
    })),
  };
}

function readJob(value: unknown): Job {
  const state = oneOf(text(value, "state"), ["queued", "running", "done", "failed"]);
  switch (state) {
    case "queued":
    case "running":
      return {
        state,
        stage: oneOf(text(value, "stage"), ["queued", "models", "capture", "render", "done"]),
      };
    case "done": {
      const results = field(value, "results");
      const timing = field(value, "timing");
      return {
        state,
        timing: typeof timing === "string" ? timing : null,
        results: {
          kind: "files",
          video: text(results, "video"),
          gif: text(results, "gif"),
          mp4: text(results, "mp4"),
          bvh: text(results, "bvh"),
          character: text(results, "character"),
          clip: text(results, "clip"),
          skeleton: text(results, "skeleton"),
        },
      };
    }
    case "failed":
      return { state, error: readError(field(value, "error")) };
  }
}

/** Calls the loopback JSON API of `copypaw app`, sending the launch token on every request. */
export class LocalBackend implements ComputeBackend {
  constructor(
    private readonly token: string,
    private readonly fetcher: typeof fetch = (input, init) => fetch(input, init),
  ) {}

  async health(): Promise<Health> {
    return readHealth(await this.call("GET", "/api/health"));
  }

  async rig(image: Blob): Promise<Rig> {
    return readRig(await this.call("POST", "/api/rig", image));
  }

  async saveJoints(rig: string, joints: Joints): Promise<void> {
    const positions = Object.fromEntries(JOINT_IDS.map((id) => [id, [joints[id].x, joints[id].y]]));
    await this.call("POST", "/api/joints", JSON.stringify({ rig, joints: positions }));
  }

  async capture(clip: Blob): Promise<string> {
    return text(await this.call("POST", "/api/capture", clip), "job");
  }

  async animate(rig: string, motion: MotionChoice, credit: boolean): Promise<string> {
    const source = motion.kind === "gallery" ? { gallery: motion.id } : { capture: motion.job };
    const body = JSON.stringify({ rig, credit, ...source });
    return text(await this.call("POST", "/api/animate", body), "job");
  }

  async job(id: string): Promise<Job> {
    return readJob(await this.call("GET", `/api/jobs/${encodeURIComponent(id)}`));
  }

  async file(url: string): Promise<Blob> {
    const response = await this.fetcher(url, { headers: { [TOKEN_HEADER]: this.token } });
    if (!response.ok) {
      throw new BackendError(INTERNAL_ERROR);
    }
    return response.blob();
  }

  private async call(method: string, url: string, body?: Blob | string): Promise<unknown> {
    const init: RequestInit = { method, headers: { [TOKEN_HEADER]: this.token } };
    if (body !== undefined) {
      init.body = body;
    }
    const response = await this.fetcher(url, init);
    let data: unknown;
    try {
      data = await response.json();
    } catch {
      throw new BackendError(INTERNAL_ERROR);
    }
    if (!response.ok) {
      throw new BackendError(readError(data));
    }
    return data;
  }
}
