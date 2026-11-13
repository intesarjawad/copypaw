/** The rig's 16 joints, in the order the selection advances through them. */
export const JOINT_IDS = [
  "root",
  "hip",
  "torso",
  "neck",
  "right_hip_b",
  "right_knee_b",
  "right_foot_b",
  "left_hip_b",
  "left_knee_b",
  "left_foot_b",
  "right_hip_f",
  "right_knee_f",
  "right_foot_f",
  "left_hip_f",
  "left_knee_f",
  "left_foot_f",
] as const;

export type JointId = (typeof JOINT_IDS)[number];

export interface Point {
  readonly x: number;
  readonly y: number;
}

export type Joints = Readonly<Record<JointId, Point>>;

export interface Rect {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

export const NUDGE_FRACTION = 0.01;
export const LARGE_NUDGE_FRACTION = 0.05;

const ARROWS: Readonly<Record<string, Point>> = {
  ArrowLeft: { x: -1, y: 0 },
  ArrowRight: { x: 1, y: 0 },
  ArrowUp: { x: 0, y: -1 },
  ArrowDown: { x: 0, y: 1 },
};

/** Build a joint set from one point per joint. */
export function jointsFrom(pointFor: (id: JointId) => Point): Joints {
  return {
    root: pointFor("root"),
    hip: pointFor("hip"),
    torso: pointFor("torso"),
    neck: pointFor("neck"),
    right_hip_b: pointFor("right_hip_b"),
    right_knee_b: pointFor("right_knee_b"),
    right_foot_b: pointFor("right_foot_b"),
    left_hip_b: pointFor("left_hip_b"),
    left_knee_b: pointFor("left_knee_b"),
    left_foot_b: pointFor("left_foot_b"),
    right_hip_f: pointFor("right_hip_f"),
    right_knee_f: pointFor("right_knee_f"),
    right_foot_f: pointFor("right_foot_f"),
    left_hip_f: pointFor("left_hip_f"),
    left_knee_f: pointFor("left_knee_f"),
    left_foot_f: pointFor("left_foot_f"),
  };
}

export function nextJoint(id: JointId): JointId {
  return JOINT_IDS[(JOINT_IDS.indexOf(id) + 1) % JOINT_IDS.length] ?? JOINT_IDS[0];
}

export function moveJoint(joints: Joints, id: JointId, to: Point): Joints {
  return jointsFrom((name) => (name === id ? to : joints[name]));
}

/** Move a joint by 1% of the drawing's box height per arrow press, or 5% with Shift. */
export function nudgeJoint(
  joints: Joints,
  id: JointId,
  key: string,
  large: boolean,
  boxHeight: number,
): Joints | null {
  const direction = ARROWS[key];
  if (direction === undefined) {
    return null;
  }
  const step = (large ? LARGE_NUDGE_FRACTION : NUDGE_FRACTION) * boxHeight;
  const from = joints[id];
  return moveJoint(joints, id, { x: from.x + direction.x * step, y: from.y + direction.y * step });
}

/** Mirror every joint across the vertical centre line of the drawing's outline. */
export function flipJoints(joints: Joints, outline: readonly Point[]): Joints {
  const xs = outline.map((point) => point.x);
  const mirror = Math.min(...xs) + Math.max(...xs);
  return jointsFrom((id) => ({ x: mirror - joints[id].x, y: joints[id].y }));
}

/** Convert a pointer position on the displayed photo to image pixels. */
export function toImagePoint(
  client: Point,
  shown: Rect,
  image: { readonly width: number; readonly height: number },
): Point {
  return {
    x: ((client.x - shown.left) * image.width) / shown.width,
    y: ((client.y - shown.top) * image.height) / shown.height,
  };
}
