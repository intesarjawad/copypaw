import { errorDetail, type ErrorDetail, type Rig } from "./backend";
import { animateDrawing, type Animation, type MotionRecord } from "./compute/animate";
import type { Mask, Pixels } from "./compute/mask";
import { rigDrawing, type Snapped } from "./compute/snap";
import type { Joints } from "./joints";
import { galleryMotion } from "./tabGallery";

export type WorkerRequest =
  | { readonly id: number; readonly kind: "rig"; readonly image: Pixels }
  | {
      readonly id: number;
      readonly kind: "animate";
      readonly rig: string;
      readonly joints: Joints;
      readonly motion: string;
    };

export type WorkerResponse =
  | { readonly id: number; readonly ok: true; readonly kind: "rig"; readonly rig: Rig }
  | {
      readonly id: number;
      readonly ok: true;
      readonly kind: "animate";
      readonly animation: Animation;
    }
  | { readonly id: number; readonly ok: false; readonly error: ErrorDetail };

let rigs = 0;
// Only the page's current drawing can be animated, so only its mask is kept.
let current: { readonly id: string; readonly mask: Mask; readonly snapped: Snapped } | null = null;

async function animate(rig: string, joints: Joints, motionId: string): Promise<Animation> {
  if (current?.id !== rig) {
    throw new Error(`Drawing ${rig} is no longer loaded`);
  }
  const motion: MotionRecord = await galleryMotion(motionId);
  const { mask, snapped } = current;
  return animateDrawing({ ...snapped, mask, joints }, motion);
}

// The tab backend's compute runs here so the page thread only draws.
self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  const reply = async (): Promise<WorkerResponse> => {
    if (request.kind === "rig") {
      const { rig, mask, snapped } = rigDrawing(request.image);
      rigs += 1;
      const id = `tab-${String(rigs)}`;
      current = { id, mask, snapped };
      return { id: request.id, ok: true, kind: "rig", rig: { id, ...rig } };
    }
    const animation = await animate(request.rig, request.joints, request.motion);
    return { id: request.id, ok: true, kind: "animate", animation };
  };
  reply().then(
    (response) => {
      const buffers =
        response.ok && response.kind === "animate"
          ? [
              response.animation.uv.buffer,
              response.animation.indices.buffer,
              response.animation.positions.buffer,
            ]
          : [];
      self.postMessage(response, { transfer: buffers });
    },
    (error: unknown) => {
      self.postMessage({ id: request.id, ok: false, error: errorDetail(error) });
    },
  );
});
