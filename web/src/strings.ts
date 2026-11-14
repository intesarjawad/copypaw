import type { JointId } from "./joints";

const PERCENT = 100;
const MS_PER_SECOND = 1000;

export const STRINGS = {
  title: "Copypaw",
  drawingHeading: "Your drawing",
  choosePhoto: "Choose a photo",
  takePhoto: "Take or choose a photo",
  dropHint: "Or drop a photo of the drawing here.",
  rigging: "Finding the outline and joints.",
  editorLabel:
    "Drawing with its joint pins. Drag a pin, or click a pin and then where it belongs. Tab moves between pins; arrow keys nudge the chosen one, and Shift moves it further.",
  flip: "Flip",
  motionHeading: "The motion",
  chooseClip: "Choose a clip",
  clipTracking: "Tracking the clip starts now; Animate waits for it.",
  galleryLegend: "From the gallery",
  motionName: (id: string): string => {
    const words = id.replaceAll(/[_-]+/g, " ");
    return words.charAt(0).toUpperCase() + words.slice(1);
  },
  animateHeading: "Animate",
  animate: "Animate",
  animateWaits: "Choose a photo first",
  animateWaitsMotion: "Choose a clip or gallery motion",
  credit: "Add the Copypaw link under the saved GIF and MP4",
  progressLabel: "Animation progress",
  stages: {
    queued: "Waiting to start.",
    models: "Downloading the pose models.",
    capture: "Tracking the pet in the clip.",
    render: "Drawing the animation.",
    done: "Finishing.",
  },
  resultHeading: "Result",
  resultVideo: "Your drawing beside the pet clip, with the footfall strip below",
  downloadsHeading: "Downloads",
  downloadGif: "Download GIF",
  downloadMp4: "Download MP4",
  downloadBvh: "Download BVH",
  downloadCharacter: "Download character (zip)",
  howItWorks: "How it works",
  howItWorksText:
    "Copypaw tracks the pet's joints in every frame of the clip, then moves the drawing's joints the same way.",
  skeletonLabel: "The tracked skeleton over the clip",
  modelsReady: "Pose models are ready.",
  modelsFailed:
    "The pose models did not download. Check your connection, then run `copypaw models fetch` and restart the app.",
  modelsAbsent: "For a clip of your own dog, set up the app with the commands above.",
  tabTiming: (frames: number, elapsedMs: number): string =>
    `Animated ${String(frames)} frames in ${(elapsedMs / MS_PER_SECOND).toFixed(1)} s on this computer.`,
  animationLabel: "Your drawing, animated with the chosen motion",
  waitingLabel: "The dog walking on the result sheet, waiting for a drawing to follow it",
  play: "Play",
  pause: "Pause",
  frame: "Frame",
  position: "Position",
  noWebgl: {
    what: "This browser has no WebGL 2, which drawing the animation needs.",
    next: "Enable WebGL 2 and hardware acceleration in your browser, then reload the page.",
  },
  modelsDownloading: (downloaded: number, total: number): string =>
    `Downloading pose models: ${String(total > 0 ? Math.floor((downloaded / total) * PERCENT) : 0)}%.`,
  errorNext: "Next:",
} as const;

/** The light table: the clip under the paper, the drawing on top, footfall.json beside them. */
export const TABLE = {
  mark: "Copypaw",
  repoUrl: "https://github.com/intesarjawad/copypaw",
  repoLink: "Star on GitHub",
  headline: "Your dog walks under the paper. The drawing walks on top.",
  lede: "A real clip of a dog shows through the paper in blue. On top, a drawing moves with the dog\u2019s tracked joints.",
  pointer: ["", "Animate your own drawing", " below, or run Copypaw on a clip of your dog."],
  runHeading: "Run it on your dog",
  runWhere: "Clone the repository, with uv and Node.js installed:",
  commands: [
    "git clone https://github.com/intesarjawad/copypaw.git && cd copypaw",
    "uv sync --all-groups --locked",
    "(cd web && npm ci && npm run build)",
    "uv run copypaw app",
  ],
  runNext: "The app opens this page on 127.0.0.1 with a clip input for your own dog.",
  lamp: "Lamp",
  run: "Run",
  stop: "Stop",
  frameLabel: "Light table reel",
  frameText: (frame: number, frames: number): string =>
    `Frame ${String(frame + 1)} of ${String(frames)}`,
  hint: "Drag the paper, or use the arrow keys, to roll the frames.",
  stillHint: (frames: number): string =>
    `This browser has no WebGL 2, so the table holds one frame; the exposure sheet still lists all ${String(frames)}.`,
  stageLabel: (scene: string, character: string): string =>
    `The ${scene} clip glowing blue under a sheet of paper on a light table, with the rigged ${character} drawing walking on top on the same red floor line`,
  xsheetCaption: "Exposure sheet: footfall.json, one row per frame",
  scene: "Scene",
  drawing: "Drawing",
  frames: "Frames",
  frameColumn: "Fr",
  dog: "Dog",
  pawShort: {
    "near front": "NF",
    "far front": "FF",
    "near back": "NB",
    "far back": "FB",
  } as Readonly<Record<string, string>>,
  sheetKey:
    "NF near front, FF far front, NB near back, FB far back. A bar marks a paw landing; the line under it runs while the paw stays planted.",
  honest: (seconds: number, landings: readonly number[], needed: number): string =>
    `This sample clip runs ${seconds.toFixed(1)} s, too short to show a gait: Copypaw counts a walk only when every paw lands at least ${String(needed)} times, and here they land ${landings.join(", ").replace(/, (\d+)$/, " and $1")}. Use a side-on clip up to 8 seconds long.`,
  traceHeading: "Animate a drawing here",
  traceLede:
    "Lay a photo of a drawing on the glass and Copypaw walks it using footfall timing tracked from the dog clip above. Nothing leaves this machine.",
} as const;

export const JOINT_LABELS: Readonly<Record<JointId, string>> = {
  root: "Root",
  hip: "Hip",
  torso: "Shoulders",
  neck: "Neck",
  right_hip_b: "Right back leg, top",
  right_knee_b: "Right back knee",
  right_foot_b: "Right back paw",
  left_hip_b: "Left back leg, top",
  left_knee_b: "Left back knee",
  left_foot_b: "Left back paw",
  right_hip_f: "Right front leg, top",
  right_knee_f: "Right front knee",
  right_foot_f: "Right front paw",
  left_hip_f: "Left front leg, top",
  left_knee_f: "Left front knee",
  left_foot_f: "Left front paw",
};
