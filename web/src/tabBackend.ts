import packageJson from "../package.json";
import {
  BackendError,
  errorDetail,
  INTERNAL_ERROR,
  type ComputeBackend,
  type Health,
  type Job,
  type MotionChoice,
  type Rig,
} from "./backend";
import type { Joints } from "./joints";
import { STRINGS } from "./strings";
import { TAB_GALLERY_IDS } from "./tabGallery";
import type { WorkerRequest, WorkerResponse } from "./worker";

type Unsent<T> = T extends unknown ? Omit<T, "id"> : never;

/** Runs the ported pipeline in a Web Worker, so a drawing animates with no server at all. */
export class TabBackend implements ComputeBackend {
  private readonly worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  private readonly pending = new Map<
    number,
    { resolve: (response: WorkerResponse) => void; reject: (error: BackendError) => void }
  >();
  private failed = false;
  private readonly drawings = new Map<string, Blob>();
  private readonly joints = new Map<string, Joints>();
  private readonly jobs = new Map<string, Job>();
  private requests = 0;

  constructor() {
    this.worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
      this.pending.get(event.data.id)?.resolve(event.data);
      this.pending.delete(event.data.id);
    });
    const fail = (): void => {
      this.failed = true;
      const error = new BackendError({
        code: "worker_failed",
        what: "Copypaw could not start its processing worker.",
        next: "Reload this page and choose your photo again.",
      });
      for (const request of this.pending.values()) request.reject(error);
      this.pending.clear();
    };
    this.worker.addEventListener("error", fail);
    this.worker.addEventListener("messageerror", fail);
  }

  health(): Promise<Health> {
    return Promise.resolve({
      version: packageJson.version,
      models: { state: "absent", downloaded: 0, total: 0 },
      gallery: TAB_GALLERY_IDS.map((id) => ({ id, thumbnail: "" })),
    });
  }

  async rig(image: Blob): Promise<Rig> {
    const bitmap = await createImageBitmap(image).catch(() => {
      throw new BackendError({
        code: "drawing_unreadable",
        what: "Copypaw could not read this image.",
        next: "Save the drawing as a PNG or JPEG, then choose it again.",
      });
    });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d");
    if (context === null) {
      throw new BackendError(INTERNAL_ERROR);
    }
    context.drawImage(bitmap, 0, 0);
    const { width, height, data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
    const response = await this.request({ kind: "rig", image: { width, height, data } });
    if (response.kind !== "rig") {
      throw new BackendError(INTERNAL_ERROR);
    }
    this.drawings.set(response.rig.id, image);
    return response.rig;
  }

  saveJoints(rig: string, joints: Joints): Promise<void> {
    this.joints.set(rig, joints);
    return Promise.resolve();
  }

  capture(): Promise<string> {
    // The page offers no clip input while the models are absent.
    return Promise.reject(new BackendError(INTERNAL_ERROR));
  }

  animate(rig: string, motion: MotionChoice): Promise<string> {
    const drawing = this.drawings.get(rig);
    const joints = this.joints.get(rig);
    if (motion.kind !== "gallery" || drawing === undefined || joints === undefined) {
      return Promise.reject(new BackendError(INTERNAL_ERROR));
    }
    const id = `job-${String(this.jobs.size + 1)}`;
    this.jobs.set(id, { state: "running", stage: "render" });
    const startedMs = performance.now();
    void Promise.all([
      this.request({ kind: "animate", rig, joints, motion: motion.id }),
      createImageBitmap(drawing),
    ]).then(
      ([response, bitmap]) => {
        if (response.kind !== "animate") {
          throw new BackendError(INTERNAL_ERROR);
        }
        this.jobs.set(id, {
          state: "done",
          results: { kind: "live", animation: response.animation, drawing: bitmap },
          timing: STRINGS.tabTiming(response.animation.frameCount, performance.now() - startedMs),
        });
      },
      (error: unknown) => {
        this.jobs.set(id, { state: "failed", error: errorDetail(error) });
      },
    );
    return Promise.resolve(id);
  }

  job(id: string): Promise<Job> {
    const job = this.jobs.get(id);
    return job === undefined
      ? Promise.reject(new BackendError(INTERNAL_ERROR))
      : Promise.resolve(job);
  }

  file(): Promise<Blob> {
    // Results stay in the tab; there are no files to fetch.
    return Promise.reject(new BackendError(INTERNAL_ERROR));
  }

  private request(request: Unsent<WorkerRequest>): Promise<Extract<WorkerResponse, { ok: true }>> {
    if (this.failed) {
      return Promise.reject(
        new BackendError({
          code: "worker_failed",
          what: "Copypaw could not start its processing worker.",
          next: "Reload this page and choose your photo again.",
        }),
      );
    }
    const id = (this.requests += 1);
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        reject,
        resolve: (response) => {
          if (response.ok) {
            resolve(response);
          } else {
            reject(new BackendError(response.error));
          }
        },
      });
      this.worker.postMessage({ ...request, id });
    });
  }
}
