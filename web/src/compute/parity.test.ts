import { describe, expect, it } from "vitest";
import dragonUrl from "../../../copypaw/selftest/dragon.png?inline";
import walk from "../gallery/dog_walk.motion.json";
import { JOINT_IDS, jointsFrom, type Point } from "../joints";
import type { Mask, Pixels } from "./mask";
import { segment } from "./mask";
import { animateDrawing, resampleMotion } from "./animate";
import type { Triangle } from "./arap";
import { poseForDrawing, type DrawnJoints } from "./bake";
import { ContactFilter } from "./contactIk";
import { drawOrder } from "./drawOrder";
import { HandleFilter, type TailPair } from "./handles";
import { characterMesh, meshVertices, type RowColumn } from "./mesh";
import { drawnJoints, retarget } from "./retarget";
import { detectFacing, fitTemplate, snap } from "./snap";
import fixture from "./parity.fixture.json";

// Written by scripts/export_web_parity.py from the Python pipeline on the same drawing and motion.

// Python poses in float32 and the port in float64, so poses agree to well under a pixel.
const POSE_TOLERANCE_PX = 1e-3;
// Engine coordinates are fractions of the drawing's padded size, held in float32.
const HANDLE_TOLERANCE = 1e-5;
// Retargeted joints are fractions of the drawing's side, held in float32 by the engine.
const JOINT_TOLERANCE = 1e-5;
// The engine forms the mesh solver's normal equations in float32 and the port in
// float64, so vertices, and the camera fitted to them, agree to half a drawing pixel.
const MESH_TOLERANCE_PX = 0.5;
// Four mesh points on one circle can be split along either diagonal; this close counts as on it.
const ON_CIRCLE_TOLERANCE = 1e-5;
// Solving the mesh for every frame of the walk takes a few seconds on one core.
const WHOLE_MOTION_TIMEOUT_MS = 180_000;
const MESH_TIMEOUT_MS = 30_000;
const PNG_SIGNATURE_BYTES = 8;
const RGB_CHANNELS = 3;

/** Decode an 8-bit RGB, non-interlaced PNG data URL into RGBA pixels. */
async function decodePng(dataUrl: string): Promise<Pixels> {
  const bytes = Uint8Array.from(atob(dataUrl.slice(dataUrl.indexOf(",") + 1)), (c) =>
    c.charCodeAt(0),
  );
  const view = new DataView(bytes.buffer);
  const idat: Uint8Array<ArrayBuffer>[] = [];
  let width = 0;
  let height = 0;
  for (let offset = PNG_SIGNATURE_BYTES; offset < bytes.length;) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    const body = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = view.getUint32(offset + 8);
      height = view.getUint32(offset + 12);
      expect([body[8], body[9], body[12]]).toEqual([8, 2, 0]);
    } else if (type === "IDAT") {
      idat.push(body);
    }
    offset += 12 + length;
  }
  const stream = new Blob(idat).stream().pipeThrough(new DecompressionStream("deflate"));
  const raw = new Uint8Array(await new Response(stream).arrayBuffer());
  const stride = width * RGB_CHANNELS;
  const rgb = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)] ?? 0;
    for (let x = 0; x < stride; x++) {
      const value = raw[y * (stride + 1) + 1 + x] ?? 0;
      const left = x >= RGB_CHANNELS ? (rgb[y * stride + x - RGB_CHANNELS] ?? 0) : 0;
      const up = y > 0 ? (rgb[(y - 1) * stride + x] ?? 0) : 0;
      const corner =
        x >= RGB_CHANNELS && y > 0 ? (rgb[(y - 1) * stride + x - RGB_CHANNELS] ?? 0) : 0;
      const estimate = left + up - corner;
      const paeth =
        Math.abs(estimate - left) <= Math.abs(estimate - up) &&
        Math.abs(estimate - left) <= Math.abs(estimate - corner)
          ? left
          : Math.abs(estimate - up) <= Math.abs(estimate - corner)
            ? up
            : corner;
      const predicted = [0, left, up, (left + up) >> 1, paeth][filter] ?? 0;
      rgb[y * stride + x] = (value + predicted) & 0xff;
    }
  }
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data.set(rgb.subarray(i * RGB_CHANNELS, (i + 1) * RGB_CHANNELS), 4 * i);
    data[4 * i + 3] = 255;
  }
  return { width, height, data };
}

function fixtureMask(): Mask {
  const { width, height, runs } = fixture.mask;
  const data = new Uint8Array(width * height);
  runs.forEach((row, y) => {
    for (let i = 0; i + 1 < row.length; i += 2) {
      data.fill(255, y * width + (row[i] ?? 0), y * width + (row[i + 1] ?? 0));
    }
  });
  return { width, height, data };
}

const pixels = decodePng(dragonUrl);
const size = Math.max(fixture.mask.width, fixture.mask.height);

const toPoint = ([x = Number.NaN, y = Number.NaN]: readonly number[]): Point => ({ x, y });

function worstDifference(actual: ArrayLike<number>, expected: readonly number[]): number {
  expect(actual.length).toBe(expected.length);
  return Math.max(...expected.map((value, index) => Math.abs((actual[index] ?? 0) - value)));
}

function triangleKey(corners: Triangle): string {
  return [...corners].sort((a, b) => a - b).join(",");
}

const engineMesh = {
  vertices: fixture.mesh.vertices.map(([row = 0, column = 0]): RowColumn => [
    row * size,
    column * size,
  ]),
  triangles: fixture.mesh.triangles.map(([a = 0, b = 0, c = 0]): Triangle => [a, b, c]),
};

/** Whether a mesh vertex other than the triangle's corners lies on its circumcircle. */
function sharesCircle(vertices: readonly RowColumn[], corners: Triangle): boolean {
  const [a, b, c] = corners.map((index) => vertices[index]);
  if (!a || !b || !c) {
    return false;
  }
  const [[ax, ay], [bx, by], [cx, cy]] = [a, b, c];
  const twiceArea = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  const [a2, b2, c2] = [ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy];
  const centreX = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / twiceArea;
  const centreY = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / twiceArea;
  const radius = Math.hypot(ax - centreX, ay - centreY);
  return vertices.some(
    ([x, y], index) =>
      !corners.includes(index) &&
      Math.abs(Math.hypot(x - centreX, y - centreY) - radius) < ON_CIRCLE_TOLERANCE * radius,
  );
}

describe("parity with the Python pipeline on the sample drawing", () => {
  it("segments the same mask", async () => {
    const mask = segment(await pixels);
    const expected = fixtureMask();
    expect([mask.width, mask.height]).toEqual([expected.width, expected.height]);
    const differing = mask.data.filter((value, index) => value !== expected.data[index]).length;

    expect(differing).toBe(0);
  });

  it("finds the same facing, legs, tails and joints", () => {
    const mask = fixtureMask();
    const facing = detectFacing(mask);
    const rows = fixture.mask.runs.flatMap((row, y) => (row.length > 0 ? [y] : []));
    const columns = fixture.mask.runs.flat();
    const template = fitTemplate(
      Math.min(...columns),
      Math.min(...rows),
      Math.max(...columns),
      Math.max(...rows) + 1,
      facing,
    );
    const snapped = snap(mask, template, facing);

    expect(facing).toBe(fixture.facing);
    expect(snapped.legs).toBe(fixture.legs);
    expect(snapped.legsGuessed).toBe(fixture.legs_guessed);
    expect(
      snapped.tails.map(([base, tip]) => [
        [base.x, base.y],
        [tip.x, tip.y],
      ]),
    ).toEqual(fixture.tails);
    for (const id of JOINT_IDS) {
      const [x, y] = fixture.joints[id];
      expect([id, snapped.joints[id].x, snapped.joints[id].y]).toEqual([id, x, y]);
    }
  });

  it("resamples the gallery motion to the output rate", () => {
    const resampled = resampleMotion(walk);

    expect(resampled.kp).toHaveLength(fixture.pose.kp.length);
    expect(resampled.contacts).toEqual(fixture.pose.contacts);
    expect(
      worstDifference(
        resampled.kp.flatMap((frame) => frame.flatMap(({ x, y }) => [x, y])),
        fixture.pose.kp.flat(2),
      ),
    ).toBeLessThan(POSE_TOLERANCE_PX);
  });

  it("poses the drawing's legs and head as the Python bake does", () => {
    const { pose } = fixture;
    const drawn: DrawnJoints = Object.fromEntries(
      Object.entries(pose.char_joints).map(([id, xy]) => [id, toPoint(xy)]),
    );
    const posed = poseForDrawing(
      {
        kp: pose.kp.map((frame) => frame.map(toPoint)),
        contacts: pose.contacts,
        groundLinePx: [pose.ground_line_px[0] ?? 0, pose.ground_line_px[1] ?? 0],
      },
      drawn,
    );

    pose.frames.forEach((frame, k) => {
      const expected = pose.posed[k] ?? [];
      posed[frame]?.forEach((point, index) => {
        const [x = Number.NaN, y = Number.NaN] = expected[index] ?? [];
        expect(Math.abs(point.x - x)).toBeLessThan(POSE_TOLERANCE_PX);
        expect(Math.abs(point.y - y)).toBeLessThan(POSE_TOLERANCE_PX);
      });
    });
  });

  it("gives the mesh the same handles on the engine's joints, frame by frame", () => {
    const { engine } = fixture;
    const flat = (rows: readonly (readonly number[])[]): Float32Array =>
      Float32Array.from(rows.flat());
    const tails = engine.tails.map(([base = [], tip = []]): TailPair => [
      [base[0] ?? 0, base[1] ?? 0],
      [tip[0] ?? 0, tip[1] ?? 0],
    ]);
    const handles = new HandleFilter(
      engine.names,
      flat(engine.drawn),
      tails,
      engine.frame_time_s,
      new ContactFilter(engine.names, engine.contacts, "treadmill"),
    );
    const frames = engine.unfiltered.map(flat);

    // The engine records a silent pass, poses once more to fit its camera, then plays.
    frames.forEach((joints, frame) => handles.apply(frame, joints));
    frames.forEach((joints, frame) => handles.apply(frame, joints));
    const played = frames.map((joints, frame) => handles.apply(frame, joints));

    engine.frames.forEach((frame, k) => {
      const expected = flat(engine.controls[k] ?? []);
      const actual = played[frame] ?? new Float32Array();
      expect(actual.length).toBe(expected.length);
      const worst = Math.max(
        ...actual.map((value, index) => Math.abs(value - (expected[index] ?? 0))),
      );
      expect(worst).toBeLessThan(HANDLE_TOLERANCE);
    });
  });

  it("retargets the posed motion onto the drawing's joints as the engine does", () => {
    const { pose, engine } = fixture;
    const drawn = drawnJoints(
      jointsFrom((id) => toPoint(pose.char_joints[id])),
      fixture.mask.height,
      size,
    );
    const posed = poseForDrawing(
      {
        kp: pose.kp.map((frame) => frame.map(toPoint)),
        contacts: pose.contacts,
        groundLinePx: [pose.ground_line_px[0] ?? 0, pose.ground_line_px[1] ?? 0],
      },
      jointsFrom((id) => toPoint(pose.char_joints[id])),
    );
    const joints = retarget(
      posed,
      posed.map(() => 0),
      drawn,
      fixture.legs === 2 ? 2 : 4,
    );

    expect(worstDifference(drawn, engine.drawn.flat())).toBeLessThan(HANDLE_TOLERANCE);
    expect(joints).toHaveLength(engine.unfiltered.length);
    expect(
      worstDifference(
        joints.flatMap((frame) => [...frame]),
        engine.unfiltered.flat(2),
      ),
    ).toBeLessThan(JOINT_TOLERANCE);
  });

  it(
    "spaces the same number of mesh vertices over the drawing",
    () => {
      expect(meshVertices(fixtureMask())).toHaveLength(fixture.mesh_vertices);
    },
    MESH_TIMEOUT_MS,
  );

  it(
    "triangulates the mesh into the engine's triangles, up to the split of cocircular points",
    () => {
      const { vertices, triangles } = characterMesh(fixtureMask());
      const engine = new Set(engineMesh.triangles.map(triangleKey));
      const differing = triangles.filter((corners) => !engine.has(triangleKey(corners)));

      expect(triangles.length).toBe(engineMesh.triangles.length);
      expect(differing.filter((corners) => !sharesCircle(vertices, corners))).toEqual([]);
    },
    MESH_TIMEOUT_MS,
  );

  it("draws the engine's triangles in the engine's order", () => {
    const indices = drawOrder(
      fixtureMask(),
      jointsFrom((id) => toPoint(fixture.pose.char_joints[id])),
      fixture.mesh.vertices.map(
        ([row = 0, column = 0]) => [Math.fround(row), Math.fround(column)] as const,
      ),
      engineMesh.triangles,
      fixture.legs === 2 ? 2 : 4,
    );

    expect([...indices]).toEqual(fixture.mesh.indices);
  });

  it(
    "moves the engine's mesh and fits its camera as the engine does",
    () => {
      const animation = animateDrawing(
        {
          mask: fixtureMask(),
          joints: jointsFrom((id) => toPoint(fixture.joints[id])),
          legs: fixture.legs === 2 ? 2 : 4,
          legsGuessed: fixture.legs_guessed,
          tails: fixture.tails.map(
            ([base = [], tip = []]) => [toPoint(base), toPoint(tip)] as const,
          ),
        },
        walk,
        () => engineMesh,
      );
      const vertexCount = engineMesh.vertices.length;

      expect(animation.frameCount).toBe(fixture.engine.unfiltered.length);
      expect([...animation.window]).toEqual(fixture.engine.window);
      expect(worstDifference(animation.camera, fixture.engine.camera)).toBeLessThan(
        MESH_TOLERANCE_PX / size,
      );
      expect([...animation.indices]).toEqual(fixture.mesh.indices);
      fixture.engine.frames.forEach((frame, k) => {
        const start = 2 * frame * vertexCount;
        const actual = animation.positions.subarray(start, start + 2 * vertexCount);
        expect(worstDifference(actual, (fixture.engine.vertices[k] ?? []).flat())).toBeLessThan(
          MESH_TOLERANCE_PX / size,
        );
      });
    },
    WHOLE_MOTION_TIMEOUT_MS,
  );
});
