import type { Point } from "../joints";
import { MalformedResponse, list, number, oneOf, point, text } from "../parse";
import { svg } from "./dom";

export interface Skeleton {
  readonly fps: number;
  readonly facing: "left" | "right";
  readonly edges: readonly (readonly [number, number])[];
  readonly frames: readonly (readonly Point[])[];
}

export interface Line {
  readonly from: Point;
  readonly to: Point;
}

export function readSkeleton(value: unknown): Skeleton {
  return {
    fps: number(value, "fps"),
    facing: oneOf(text(value, "facing"), ["left", "right"]),
    edges: list(value, "edges").map((edge) => {
      const pair = point(edge);
      return [pair.x, pair.y] as const;
    }),
    frames: list(value, "frames").map((frame) => {
      if (!Array.isArray(frame)) {
        throw new MalformedResponse("frames");
      }
      const keypoints: readonly unknown[] = frame;
      return keypoints.map(point);
    }),
  };
}

/**
 * Bones of the frame shown at a playback time, in the clip's pixels.
 * Captured motion is stored facing right, so a left-facing clip is mirrored back.
 */
export function skeletonLines(skeleton: Skeleton, timeS: number, frameWidth: number): Line[] {
  const index = Math.min(Math.max(Math.floor(timeS * skeleton.fps), 0), skeleton.frames.length - 1);
  const frame = skeleton.frames[index] ?? [];
  const place = (at: Point): Point =>
    skeleton.facing === "left" ? { x: frameWidth - 1 - at.x, y: at.y } : at;
  return skeleton.edges.flatMap(([start, end]) => {
    const from = frame[start];
    const to = frame[end];
    return from === undefined || to === undefined ? [] : [{ from: place(from), to: place(to) }];
  });
}

/** Keep an SVG skeleton over a playing clip in step with its current frame. */
export function followClip(
  video: HTMLVideoElement,
  overlay: SVGSVGElement,
  skeleton: Skeleton,
): void {
  function draw(): void {
    const lines = skeletonLines(skeleton, video.currentTime, video.videoWidth);
    overlay.replaceChildren(
      ...lines.map((line) =>
        svg("line", {
          x1: String(line.from.x),
          y1: String(line.from.y),
          x2: String(line.to.x),
          y2: String(line.to.y),
        }),
      ),
    );
  }
  function tick(): void {
    draw();
    if (!video.paused && !video.ended) {
      requestAnimationFrame(tick);
    }
  }
  video.addEventListener("loadedmetadata", () => {
    overlay.setAttribute("viewBox", `0 0 ${String(video.videoWidth)} ${String(video.videoHeight)}`);
    draw();
  });
  video.addEventListener("play", tick);
  video.addEventListener("seeked", draw);
}
