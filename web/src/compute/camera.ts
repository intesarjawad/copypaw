/** A camera position, x and y over the drawing and z toward the viewer. */
export type CameraPosition = readonly [x: number, y: number, z: number];

// Where the engine places the drawing's plane along z.
export const CHARACTER_DEPTH = -0.5;
// The engine's vertical field of view and clip planes.
const FIELD_OF_VIEW_DEG = 35;
const NEAR_PLANE = 0.1;
const FAR_PLANE = 10000;
// Clear space kept around the drawing's reach, as a share of that reach.
const FRAME_MARGIN_RATIO = 0.05;

/**
 * Where a straight-on camera shows the box from low to high at full size.
 *
 * The box is everything the drawing reaches over the whole motion, so one
 * fixed view keeps every frame whole. The tighter of width and height fills
 * the window less a margin on each side.
 */
export function frameCamera(
  low: readonly [number, number],
  high: readonly [number, number],
  window: readonly [width: number, height: number],
): CameraPosition {
  const [width, height] = window;
  const reach = [0, 1].map(
    (axis) => ((high[axis] ?? 0) - (low[axis] ?? 0)) * (1 + 2 * FRAME_MARGIN_RATIO),
  );
  const [reachX = 0, reachY = 0] = reach;
  if (!(width > 0 && height > 0 && reachX > 0 && reachY > 0) || !reach.every(Number.isFinite)) {
    throw new Error("The drawing's reach and the window must be finite and positive");
  }
  const visibleHeight = Math.max(reachY, (reachX * height) / width);
  const distance = visibleHeight / (2 * Math.tan((FIELD_OF_VIEW_DEG * Math.PI) / 180 / 2));
  return [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, CHARACTER_DEPTH + distance];
}

/**
 * The engine's perspective projection times its view from the camera, column
 * major for WebGL. mirrored flips the picture left to right, as the engine's
 * output is flipped back for a drawing stored mirrored.
 */
export function viewProjection(
  camera: CameraPosition,
  window: readonly [width: number, height: number],
  mirrored: boolean,
): Float32Array {
  const [width, height] = window;
  const top = NEAR_PLANE * Math.tan((FIELD_OF_VIEW_DEG * Math.PI) / 360);
  const right = (top * width) / height;
  const xScale = ((mirrored ? -1 : 1) * NEAR_PLANE) / right;
  const yScale = NEAR_PLANE / top;
  const depthScale = (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE);
  const depthShift = (2 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE);
  const [x, y, z] = camera;
  // proj * translate(-camera): the camera looks down -z with +y up.
  return Float32Array.from([
    xScale,
    0,
    0,
    0,
    0,
    yScale,
    0,
    0,
    0,
    0,
    depthScale,
    -1,
    -xScale * x,
    -yScale * y,
    -depthScale * z + depthShift,
    z,
  ]);
}
