import type { MotionRecord } from "./compute/animate";

// Written by `scripts/make_goldens.py --web <motion> --web-dir web/src/gallery`; each file
// is loaded only when its motion is animated.
const MOTIONS = import.meta.glob<MotionRecord>("./gallery/*.motion.json", { import: "default" });

function idOf(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1, -".motion.json".length);
}

/** The gallery motions the tab animates without the local app, by id. */
export const TAB_GALLERY_IDS: readonly string[] = Object.keys(MOTIONS).map(idOf).sort();

export async function galleryMotion(id: string): Promise<MotionRecord> {
  const load = Object.entries(MOTIONS).find(([path]) => idOf(path) === id)?.[1];
  if (load === undefined) {
    throw new Error(`The gallery has no motion ${id}`);
  }
  return load();
}
