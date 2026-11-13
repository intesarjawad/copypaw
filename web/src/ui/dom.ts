import { errorDetail } from "../backend";
import { STRINGS } from "../strings";

const SVG_NS = "http://www.w3.org/2000/svg";

type Attributes = Readonly<Record<string, string>>;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attributes: Attributes = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, value);
  }
  element.append(...children);
  return element;
}

export function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Attributes = {},
): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, value);
  }
  return element;
}

/** Show what happened and the next step, both from messages.json via the backend. */
export function showError(region: HTMLElement, error: unknown): void {
  const detail = errorDetail(error);
  region.replaceChildren(
    el("p", {}, detail.what),
    el("p", {}, `${STRINGS.errorNext} ${detail.next}`),
  );
}
