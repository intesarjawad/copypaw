import type { Point } from "./joints";

/** A server response that does not have the shape the page expects. */
export class MalformedResponse extends Error {}

export function field(value: unknown, key: string): unknown {
  if (typeof value !== "object" || value === null) {
    throw new MalformedResponse(key);
  }
  const found: unknown = Reflect.get(value, key);
  return found;
}

export function text(value: unknown, key: string): string {
  const found = field(value, key);
  if (typeof found !== "string") {
    throw new MalformedResponse(key);
  }
  return found;
}

export function number(value: unknown, key: string): number {
  const found = field(value, key);
  if (typeof found !== "number" || !Number.isFinite(found)) {
    throw new MalformedResponse(key);
  }
  return found;
}

export function list(value: unknown, key: string): readonly unknown[] {
  const found = field(value, key);
  if (!Array.isArray(found)) {
    throw new MalformedResponse(key);
  }
  const items: readonly unknown[] = found;
  return items;
}

export function point(value: unknown): Point {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new MalformedResponse("point");
  }
  const pair: readonly unknown[] = value;
  const [x, y] = pair;
  if (typeof x !== "number" || typeof y !== "number" || !Number.isFinite(x + y)) {
    throw new MalformedResponse("point");
  }
  return { x, y };
}

export function oneOf<T extends string>(value: string, allowed: readonly T[]): T {
  const match = allowed.find((option) => option === value);
  if (match === undefined) {
    throw new MalformedResponse(value);
  }
  return match;
}
