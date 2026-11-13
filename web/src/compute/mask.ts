import messages from "../../../copypaw/messages.json";
import { BackendError } from "../backend";
import type { Point } from "../joints";

/** RGBA pixels in row order, as an ImageData holds them. */
export interface Pixels {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
}

/** One byte per pixel, 0 outside the drawing and 255 inside. */
export interface Mask {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array;
}

export type MaskErrorKind = "too_small" | "too_large" | "multiple_creatures";

const MAX_SIDE_PX = 1000;
const ADAPTIVE_BLOCK_PX = 115;
const ADAPTIVE_C = 8;
const MORPH_ITERATIONS = 2;
const DILATE_SIZE_PX = 3;
const SEED_STEP_PX = 10;
const MIN_COVERAGE = 0.05;
const MAX_COVERAGE = 0.9;
const MIN_COMPONENT_FRACTION = 0.01;
const CLOSE_SIZES_PX = [3, 5, 7, 9] as const;
// Widest stroke restored around the enclosed region, at the working size.
const MAX_STROKE_REACH_PX = 24;
// A ring of pixels around the enclosed region is stroke while this share of it is ink.
const STROKE_RING_SHARE = 0.5;
// A stroke outside the outline is a leg when it hangs this close to the drawing, is at least
// this share of the drawing's height long, and starts in its lower half.
const LOOSE_REACH_PX = 24;
const LOOSE_MIN_LENGTH_SHARE = 0.5;
// Widest a leg of one pen line is thickened to, so the mesh has width to bend.
const LOOSE_WIDTH_PX = 21;

export function maskError(kind: MaskErrorKind): BackendError {
  const code = `mask_${kind}` as const;
  return new BackendError({ code, ...messages[code] });
}

/** Python's round(), which sends halves to the even neighbour. */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest !== 0.5) {
    return Math.round(value);
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

function at(data: ArrayLike<number>, index: number): number {
  return data[index] ?? 0;
}

/** Bilinear resize with pixel-centre alignment and clamped edges, as OpenCV's INTER_LINEAR. */
function resizeLinear(
  plane: Uint8Array,
  width: number,
  height: number,
  outWidth: number,
  outHeight: number,
): Uint8Array {
  const out = new Uint8Array(outWidth * outHeight);
  const taps = (outSize: number, inSize: number): [Int32Array, Float64Array] => {
    const first = new Int32Array(outSize);
    const weight = new Float64Array(outSize);
    const scale = inSize / outSize;
    for (let i = 0; i < outSize; i++) {
      const position = (i + 0.5) * scale - 0.5;
      let index = Math.floor(position);
      let fraction = position - index;
      if (index < 0) {
        index = 0;
        fraction = 0;
      }
      if (index >= inSize - 1) {
        index = inSize - 1;
        fraction = 0;
      }
      first[i] = index;
      weight[i] = fraction;
    }
    return [first, weight];
  };
  const [xs, fxs] = taps(outWidth, width);
  const [ys, fys] = taps(outHeight, height);
  for (let y = 0; y < outHeight; y++) {
    const y0 = ys[y] ?? 0;
    const y1 = Math.min(y0 + 1, height - 1);
    const fy = fys[y] ?? 0;
    for (let x = 0; x < outWidth; x++) {
      const x0 = xs[x] ?? 0;
      const x1 = Math.min(x0 + 1, width - 1);
      const fx = fxs[x] ?? 0;
      const top = (plane[y0 * width + x0] ?? 0) * (1 - fx) + (plane[y0 * width + x1] ?? 0) * fx;
      const bottom = (plane[y1 * width + x0] ?? 0) * (1 - fx) + (plane[y1 * width + x1] ?? 0) * fx;
      out[y * outWidth + x] = Math.round(top * (1 - fy) + bottom * fy);
    }
  }
  return out;
}

/** Darkest channel per pixel, after capping the longest side at 1000 px. */
function scaledGray(pixels: Pixels): Mask {
  const { width, height } = pixels;
  let outWidth = width;
  let outHeight = height;
  const longest = Math.max(width, height);
  if (longest > MAX_SIDE_PX) {
    const scale = MAX_SIDE_PX / longest;
    outWidth = Math.max(1, roundHalfEven(scale * width));
    outHeight = Math.max(1, roundHalfEven(scale * height));
  }
  const channel = (offset: number): Uint8Array => {
    const plane = new Uint8Array(width * height);
    for (let i = 0; i < plane.length; i++) {
      plane[i] = pixels.data[4 * i + offset] ?? 0;
    }
    return outWidth === width && outHeight === height
      ? plane
      : resizeLinear(plane, width, height, outWidth, outHeight);
  };
  const red = channel(0);
  const green = channel(1);
  const blue = channel(2);
  const gray = new Uint8Array(outWidth * outHeight);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = Math.min(red[i] ?? 0, green[i] ?? 0, blue[i] ?? 0);
  }
  return { width: outWidth, height: outHeight, data: gray };
}

/** OpenCV's Gaussian kernel for a size above 7 when sigma is derived from the size. */
export function gaussianKernel(size: number): Float64Array {
  const sigma = ((size - 1) * 0.5 - 1) * 0.3 + 0.8;
  const kernel = new Float64Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const offset = i - (size - 1) / 2;
    kernel[i] = Math.exp(-(offset * offset) / (2 * sigma * sigma));
    sum += kernel[i] ?? 0;
  }
  return kernel.map((value) => value / sum);
}

/** Separable Gaussian blur with replicated borders, rounded to bytes. */
function gaussianBlur(image: Mask, size: number): Uint8Array {
  const { width, height, data } = image;
  const kernel = gaussianKernel(size);
  const radius = (size - 1) / 2;
  // Each row, then each column, is copied once with its ends repeated, so the kernel's inner
  // loop runs without clamping; the sums keep their order, so the result is unchanged.
  const padded = new Float64Array(Math.max(width, height) + size - 1);
  const rows = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let i = 0; i < width + size - 1; i++) {
      padded[i] = at(data, y * width + Math.min(width - 1, Math.max(0, i - radius)));
    }
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = 0; k < size; k++) {
        sum += (kernel[k] ?? 0) * (padded[x + k] ?? 0);
      }
      rows[y * width + x] = sum;
    }
  }
  const out = new Uint8Array(width * height);
  for (let x = 0; x < width; x++) {
    for (let i = 0; i < height + size - 1; i++) {
      padded[i] = at(rows, Math.min(height - 1, Math.max(0, i - radius)) * width + x);
    }
    for (let y = 0; y < height; y++) {
      let sum = 0;
      for (let k = 0; k < size; k++) {
        sum += (kernel[k] ?? 0) * (padded[y + k] ?? 0);
      }
      out[y * width + x] = Math.min(255, Math.round(sum));
    }
  }
  return out;
}

/** Max (dilate) or min (erode) over a centred square; pixels outside the image are ignored. */
export function morphSquare(image: Mask, size: number, dilate: boolean): Mask {
  const { width, height, data } = image;
  const radius = Math.floor(size / 2);
  const pick = dilate
    ? (a: number, b: number) => Math.max(a, b)
    : (a: number, b: number) => Math.min(a, b);
  const rows = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = data[y * width + x] ?? 0;
      for (let sx = Math.max(0, x - radius); sx <= Math.min(width - 1, x + radius); sx++) {
        value = pick(value, data[y * width + sx] ?? 0);
      }
      rows[y * width + x] = value;
    }
  }
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = rows[y * width + x] ?? 0;
      for (let sy = Math.max(0, y - radius); sy <= Math.min(height - 1, y + radius); sy++) {
        value = pick(value, rows[sy * width + x] ?? 0);
      }
      out[y * width + x] = value;
    }
  }
  return { width, height, data: out };
}

/** A square kernel applied n times equals one pass of a (size - 1) * n + 1 square. */
function repeated(size: number, iterations: number): number {
  return (size - 1) * iterations + 1;
}

/** OpenCV's elliptical structuring element: half-widths of each row around the centre. */
export function ellipseHalfWidths(size: number): number[] {
  const radius = Math.floor(size / 2);
  return Array.from({ length: size }, (_, row) => {
    const dy = row - radius;
    return radius === 0 ? 0 : Math.round(Math.sqrt(radius * radius - dy * dy));
  });
}

/**
 * Dilate a 0/255 mask by a kernel given as each row's half-width around its centre row.
 *
 * Each pixel's distance to the nearest set pixel in its own row is found once,
 * so the cost does not grow with the kernel's width.
 */
function dilateRows(image: Mask, halfWidths: readonly number[]): Mask {
  const { width, height, data } = image;
  const across = new Int32Array(width * height);
  for (let y = 0; y < height; y++) {
    let last = -Infinity;
    for (let x = 0; x < width; x++) {
      last = (data[y * width + x] ?? 0) === 0 ? last : x;
      across[y * width + x] = Math.min(x - last, width);
    }
    last = Infinity;
    for (let x = width - 1; x >= 0; x--) {
      last = (data[y * width + x] ?? 0) === 0 ? last : x;
      across[y * width + x] = Math.min(across[y * width + x] ?? 0, last - x);
    }
  }
  const radius = Math.floor(halfWidths.length / 2);
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = halfWidths.some((half, row) => {
        const sy = y + row - radius;
        return sy >= 0 && sy < height && (across[sy * width + x] ?? 0) <= half;
      })
        ? 255
        : 0;
    }
  }
  return { width, height, data: out };
}

function dilateEllipse(image: Mask, size: number): Mask {
  return dilateRows(image, ellipseHalfWidths(size));
}

/** Set every 4-connected pixel equal to the seed's value to zero, starting from seed. */
function floodToZero(data: Uint8Array, width: number, height: number, seed: number): void {
  if (at(data, seed) === 0) {
    return;
  }
  const stack = [seed];
  data[seed] = 0;
  for (let index = stack.pop(); index !== undefined; index = stack.pop()) {
    const x = index % width;
    const y = (index - x) / width;
    const next = [
      x > 0 ? index - 1 : -1,
      x < width - 1 ? index + 1 : -1,
      y > 0 ? index - width : -1,
      y < height - 1 ? index + width : -1,
    ];
    for (const neighbour of next) {
      if (neighbour >= 0 && at(data, neighbour) !== 0) {
        data[neighbour] = 0;
        stack.push(neighbour);
      }
    }
  }
}

/** Adaptive Gaussian threshold, inverted so ink is 255. */
function ink(gray: Mask): Mask {
  const mean = gaussianBlur(gray, ADAPTIVE_BLOCK_PX);
  const data = new Uint8Array(gray.data.length);
  for (let i = 0; i < data.length; i++) {
    data[i] = at(gray.data, i) - at(mean, i) > -ADAPTIVE_C ? 0 : 255;
  }
  return { width: gray.width, height: gray.height, data };
}

function enclosed(strokes: Mask, closeSizePx: number): Mask {
  const { width, height } = strokes;
  const closeSize = repeated(closeSizePx, MORPH_ITERATIONS);
  let outline = morphSquare(morphSquare(strokes, closeSize, true), closeSize, false);
  outline = morphSquare(outline, repeated(DILATE_SIZE_PX, MORPH_ITERATIONS), true);

  // Paper ruling can touch the stroke, so only whitespace inside the stroke counts.
  const inside = outline.data.map((value) => 255 - value);
  for (let x = 0; x < width - 1; x += SEED_STEP_PX) {
    floodToZero(inside, width, height, x);
    floodToZero(inside, width, height, (height - 1) * width + x);
  }
  for (let y = 0; y < height - 1; y += SEED_STEP_PX) {
    floodToZero(inside, width, height, y * width);
    floodToZero(inside, width, height, y * width + width - 1);
  }
  for (let x = 0; x < width; x++) {
    inside[x] = 0;
    inside[(height - 1) * width + x] = 0;
  }
  for (let y = 0; y < height; y++) {
    inside[y * width] = 0;
    inside[y * width + width - 1] = 0;
  }
  return { width, height, data: inside };
}

/** 8-connected foreground components labelled from 1 in raster order of their first pixel. */
export interface Components {
  readonly labels: Int32Array;
  readonly areas: number[];
  readonly starts: number[];
}

export function components(mask: Mask): Components {
  const { width, height, data } = mask;
  const labels = new Int32Array(width * height);
  const areas: number[] = [];
  const starts: number[] = [];
  for (let start = 0; start < labels.length; start++) {
    if (at(data, start) === 0 || at(labels, start) !== 0) {
      continue;
    }
    const label = areas.length + 1;
    let area = 0;
    const stack = [start];
    labels[start] = label;
    for (let index = stack.pop(); index !== undefined; index = stack.pop()) {
      area++;
      const x = index % width;
      const y = (index - x) / width;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          const neighbour = ny * width + nx;
          if (
            nx >= 0 &&
            nx < width &&
            ny >= 0 &&
            ny < height &&
            at(data, neighbour) !== 0 &&
            at(labels, neighbour) === 0
          ) {
            labels[neighbour] = label;
            stack.push(neighbour);
          }
        }
      }
    }
    areas.push(area);
    starts.push(start);
  }
  return { labels, areas, starts };
}

/**
 * Whether each component is outermost, as findContours' RETR_EXTERNAL keeps.
 *
 * The west neighbour of a component's first raster pixel lies in the background
 * region around it; the component is nested when that region does not reach the
 * image border through 4-connected background.
 */
function outermost(mask: Mask, found: Components): boolean[] {
  const { width, height } = mask;
  const background = mask.data.map((value) => (value === 0 ? 255 : 0));
  for (let x = 0; x < width; x++) {
    floodToZero(background, width, height, x);
    floodToZero(background, width, height, (height - 1) * width + x);
  }
  for (let y = 0; y < height; y++) {
    floodToZero(background, width, height, y * width);
    floodToZero(background, width, height, y * width + width - 1);
  }
  return found.starts.map((start) => start % width === 0 || at(background, start - 1) === 0);
}

/** One component as 0/255 with every hole not 4-connected to the border filled. */
function filledComponent(labels: Int32Array, label: number, width: number, height: number): Mask {
  const outside = new Uint8Array(width * height);
  labels.forEach((value, index) => {
    outside[index] = value === label ? 0 : 255;
  });
  for (let x = 0; x < width; x++) {
    floodToZero(outside, width, height, x);
    floodToZero(outside, width, height, (height - 1) * width + x);
  }
  for (let y = 0; y < height; y++) {
    floodToZero(outside, width, height, y * width);
    floodToZero(outside, width, height, y * width + width - 1);
  }
  // Pixels the border flood reached are now 0; the component and its holes keep 255 or label.
  const filled = new Uint8Array(width * height);
  labels.forEach((value, index) => {
    filled[index] = value === label || at(outside, index) !== 0 ? 255 : 0;
  });
  return { width, height, data: filled };
}

// Stands in for infinity in the distance transform; exceeds any squared distance in a 1000 px image.
const FAR_SQUARED_PX = 1e20;

/** Lower envelope of parabolas rooted at f: squared distance along one line, by Felzenszwalb and Huttenlocher. */
function squaredDistance1d(
  f: Float64Array,
  out: Float64Array,
  hull: Int32Array,
  edges: Float64Array,
): void {
  let k = 0;
  hull[0] = 0;
  edges[0] = -Infinity;
  edges[1] = Infinity;
  const intersection = (q: number, v: number): number =>
    (at(f, q) + q * q - (at(f, v) + v * v)) / (2 * q - 2 * v);
  for (let q = 1; q < f.length; q++) {
    // edges[0] is minus infinity, so the search stops at the first parabola.
    while (intersection(q, at(hull, k)) <= at(edges, k)) {
      k--;
    }
    const s = intersection(q, at(hull, k));
    k++;
    hull[k] = q;
    edges[k] = s;
    edges[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < f.length; q++) {
    while (at(edges, k + 1) < q) {
      k++;
    }
    const v = at(hull, k);
    out[q] = (q - v) * (q - v) + at(f, v);
  }
}

/** Exact Euclidean distance from every pixel to the nearest filled pixel, as OpenCV's precise L2. */
export function distanceToFilled(mask: Mask): Float64Array {
  const { width, height, data } = mask;
  const size = Math.max(width, height);
  const f = new Float64Array(size);
  const out = new Float64Array(size);
  const hull = new Int32Array(size);
  const edges = new Float64Array(size + 1);
  const squared = new Float64Array(width * height);
  for (let x = 0; x < width; x++) {
    const column = f.subarray(0, height);
    for (let y = 0; y < height; y++) {
      column[y] = at(data, y * width + x) === 0 ? FAR_SQUARED_PX : 0;
    }
    const result = out.subarray(0, height);
    squaredDistance1d(column, result, hull, edges);
    for (let y = 0; y < height; y++) {
      squared[y * width + x] = at(result, y);
    }
  }
  for (let y = 0; y < height; y++) {
    const row = squared.subarray(y * width, (y + 1) * width);
    const result = out.subarray(0, width);
    squaredDistance1d(Float64Array.from(row), result, hull, edges);
    row.set(result);
  }
  return squared.map(Math.sqrt);
}

/**
 * Distance from the enclosed region to the outer edge of the stroke around it.
 *
 * The enclosed region stops short of the stroke, so the rings of pixels around
 * it are paper, then ink, then paper again. The stroke ends at the last ring
 * that is mostly ink; paper ruling crosses the rings beyond it but never fills one.
 */
function strokeReachPx(filled: Mask, strokes: Mask): number {
  const distance = distanceToFilled(filled);
  const total = new Array<number>(MAX_STROKE_REACH_PX + 1).fill(0);
  const inked = new Array<number>(MAX_STROKE_REACH_PX + 1).fill(0);
  distance.forEach((value, index) => {
    const ring = Math.ceil(value);
    if (ring >= 1 && ring <= MAX_STROKE_REACH_PX) {
      total[ring] = at(total, ring) + 1;
      inked[ring] = at(inked, ring) + (at(strokes.data, index) > 0 ? 1 : 0);
    }
  });
  let reach = 0;
  for (let ring = 1; ring <= MAX_STROKE_REACH_PX; ring++) {
    if (at(inked, ring) >= STROKE_RING_SHARE * Math.max(at(total, ring), 1)) {
      reach = ring;
    }
  }
  return reach;
}

/** Grow the enclosed region over the stroke drawn around it, so the outline is part of the drawing. */
function withStroke(filled: Mask, strokes: Mask): Mask {
  const size = 2 * strokeReachPx(filled, strokes) + 1;
  const grown = dilateEllipse(filled, size);
  // A round kernel cuts the tip off a sharp corner, so ink within the square reach is kept.
  const square = dilateRows(filled, new Array<number>(size).fill(Math.floor(size / 2)));
  const data = grown.data.map((value, index) =>
    value !== 0 || (at(square.data, index) !== 0 && at(strokes.data, index) !== 0) ? 255 : 0,
  );
  return { width: filled.width, height: filled.height, data };
}

/** A one-pixel line from a stroke's tip to the nearest pixel of the drawing, or null when it is out of reach. */
function bridge(mask: Mask, tipX: number, tipY: number): number[] | null {
  const { width, data } = mask;
  let nearest = -1;
  let best = Infinity;
  for (let index = 0; index < data.length; index++) {
    if (at(data, index) === 0) {
      continue;
    }
    const x = index % width;
    const y = (index - x) / width;
    const squared = (x - tipX) ** 2 + (y - tipY) ** 2;
    if (squared < best) {
      best = squared;
      nearest = index;
    }
  }
  const dx = (nearest % width) - tipX;
  const dy = (nearest - (nearest % width)) / width - tipY;
  const steps = Math.max(Math.abs(dx), Math.abs(dy), 1);
  if (steps + 1 > LOOSE_REACH_PX + 1) {
    return null;
  }
  const line: number[] = [];
  for (let along = 0; along <= steps; along++) {
    line.push(
      Math.floor(tipY + (dy * along) / steps + 0.5) * width +
        Math.floor(tipX + (dx * along) / steps + 0.5),
    );
  }
  return line;
}

/**
 * Add single-line legs hanging off the drawing, thickened to a minimum width.
 *
 * The enclosed region leaves out strokes that enclose nothing, so a stick leg is
 * found as ink outside the mask. Ruling is rejected by shape: it runs across the
 * page or lies flat.
 */
function withLegs(mask: Mask, strokes: Mask): Mask {
  const { width, height } = mask;
  const outside: Mask = {
    width,
    height,
    data: mask.data.map((value, index) => (value === 0 && at(strokes.data, index) > 0 ? 255 : 0)),
  };
  const { labels, areas } = components(outside);
  const first = new Array<number>(areas.length).fill(height);
  const last = new Array<number>(areas.length).fill(-1);
  const left = new Array<number>(areas.length).fill(width);
  const right = new Array<number>(areas.length).fill(-1);
  labels.forEach((label, index) => {
    if (label === 0) {
      return;
    }
    const x = index % width;
    const y = (index - x) / width;
    first[label - 1] = Math.min(at(first, label - 1), y);
    last[label - 1] = Math.max(at(last, label - 1), y);
    left[label - 1] = Math.min(at(left, label - 1), x);
    right[label - 1] = Math.max(at(right, label - 1), x);
  });
  let bodyTop = height;
  let bodyBottom = -1;
  mask.data.forEach((value, index) => {
    if (value !== 0) {
      const y = Math.floor(index / width);
      bodyTop = Math.min(bodyTop, y);
      bodyBottom = Math.max(bodyBottom, y);
    }
  });
  const bodyHeight = bodyBottom - bodyTop + 1;
  const legs = new Uint8Array(width * height);
  for (let label = 1; label <= areas.length; label++) {
    const top = at(first, label - 1);
    const strokeWidth = at(right, label - 1) - at(left, label - 1) + 1;
    const strokeHeight = at(last, label - 1) - top + 1;
    const hangs =
      strokeHeight > strokeWidth &&
      strokeHeight >= LOOSE_MIN_LENGTH_SHARE * bodyHeight &&
      top >= bodyTop + Math.floor(bodyHeight / 2) &&
      at(left, label - 1) > 0 &&
      top > 0 &&
      at(right, label - 1) + 1 < width &&
      top + strokeHeight < height;
    if (!hangs) {
      continue;
    }
    let tipX = 0;
    while (at(labels, top * width + tipX) !== label) {
      tipX++;
    }
    const line = bridge(mask, tipX, top);
    if (line === null) {
      continue;
    }
    labels.forEach((value, index) => {
      if (value === label) {
        legs[index] = 255;
      }
    });
    line.forEach((index) => {
      legs[index] = 255;
    });
  }
  const grown = dilateEllipse({ width, height, data: legs }, LOOSE_WIDTH_PX);
  return {
    width,
    height,
    data: mask.data.map((value, index) => (value !== 0 || at(grown.data, index) !== 0 ? 255 : 0)),
  };
}

function count(mask: Mask): number {
  let total = 0;
  for (const value of mask.data) {
    total += value === 0 ? 0 : 1;
  }
  return total;
}

/**
 * Return the filled drawing on a light page as a 0/255 mask at the capped size.
 *
 * The longest side is capped at 1000 px. A broken outline is retried with wider
 * closing kernels. Pixel counts of the outermost components stand in for
 * OpenCV's external contour areas when choosing significant and largest regions.
 */
export function segment(pixels: Pixels): Mask {
  if (pixels.width <= 0 || pixels.height <= 0) {
    throw new Error("Image width and height must be positive");
  }
  const gray = scaledGray(pixels);
  const { width, height } = gray;
  const strokes = ink(gray);
  const minimumArea = width * height * MIN_COMPONENT_FRACTION;
  let failure: MaskErrorKind = "too_small";
  for (const closeSizePx of CLOSE_SIZES_PX) {
    const inside = enclosed(strokes, closeSizePx);
    const found = components(inside);
    const { labels, areas } = found;
    const outer = outermost(inside, found);
    const significant = areas.flatMap((area, index) =>
      outer[index] === true && area >= minimumArea ? [index + 1] : [],
    );
    if (significant.length === 0) {
      continue;
    }
    const largest = significant.reduce((best, label) =>
      at(areas, label - 1) > at(areas, best - 1) ? label : best,
    );
    const mask = withLegs(
      withStroke(filledComponent(labels, largest, width, height), strokes),
      strokes,
    );
    const coverage = count(mask) / mask.data.length;
    if (coverage < MIN_COVERAGE) {
      continue;
    }
    if (coverage > MAX_COVERAGE) {
      throw maskError("too_large");
    }
    if (significant.length > 1) {
      if (closeSizePx === CLOSE_SIZES_PX[0]) {
        throw maskError("multiple_creatures");
      }
      failure = "multiple_creatures";
      continue;
    }
    return mask;
  }
  throw maskError(failure);
}

/** Nearest-neighbour resize that samples floor(x * in / out), as OpenCV's INTER_NEAREST. */
export function resizeNearest(mask: Mask, width: number, height: number): Mask {
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(mask.height - 1, Math.floor((y * mask.height) / height));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(mask.width - 1, Math.floor((x * mask.width) / width));
      data[y * width + x] = at(mask.data, sy * mask.width + sx);
    }
  }
  return { width, height, data };
}

// Chain codes counter-clockwise from east, with y pointing down.
const STEPS: readonly (readonly [number, number])[] = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
  [1, 1],
];

/**
 * Trace the outer border of the first foreground component in raster order.
 *
 * Follows Suzuki and Abe's border following, as OpenCV's findContours does,
 * and keeps only the points where the direction changes (CHAIN_APPROX_SIMPLE).
 */
export function outline(mask: Mask): Point[] {
  const { width, height, data } = mask;
  const on = (x: number, y: number): boolean =>
    x >= 0 && x < width && y >= 0 && y < height && at(data, y * width + x) !== 0;
  const start = data.findIndex((value) => value !== 0);
  if (start < 0) {
    return [];
  }
  const x0 = start % width;
  const y0 = (start - x0) / width;
  const step = (code: number): readonly [number, number] => STEPS[(code + 8) % 8] ?? [0, 0];

  let first = -1;
  for (let k = 0; k < 8; k++) {
    const [dx, dy] = step(4 - k);
    if (on(x0 + dx, y0 + dy)) {
      first = (4 - k + 8) % 8;
      break;
    }
  }
  if (first < 0) {
    return [{ x: x0, y: y0 }];
  }
  const [fx, fy] = step(first);
  const x1 = x0 + fx;
  const y1 = y0 + fy;

  const points: Point[] = [];
  const codes: number[] = [];
  let [px, py] = [x1, y1];
  let [cx, cy] = [x0, y0];
  for (;;) {
    const back = STEPS.findIndex(([dx, dy]) => dx === px - cx && dy === py - cy);
    let code = back;
    for (let k = 1; k <= 8; k++) {
      code = (back + k) % 8;
      const [dx, dy] = step(code);
      if (on(cx + dx, cy + dy)) {
        break;
      }
    }
    const [dx, dy] = step(code);
    points.push({ x: cx, y: cy });
    codes.push(code);
    const [nx, ny] = [cx + dx, cy + dy];
    if (nx === x0 && ny === y0 && cx === x1 && cy === y1) {
      break;
    }
    [px, py] = [cx, cy];
    [cx, cy] = [nx, ny];
  }
  return points.filter(
    (_, index) => index === 0 || codes[index] !== codes[(index - 1 + codes.length) % codes.length],
  );
}
