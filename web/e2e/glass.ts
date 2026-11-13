import { toolGlass } from "../src/lightTable/toolGlass";
import { clipAlone } from "../src/lightTable/underlay";
import { readSheet } from "../src/lightTable/sheet";
import record from "../src/lightTable/sheet.json";

const sheets = document.getElementById("table");
const stage = document.getElementById("stage");
if (sheets === null || stage === null) throw new Error("Missing glass or stage.");
const glass = toolGlass(sheets);
const canvas = sheets.querySelector("canvas");
if (canvas === null) throw new Error("Missing glass canvas.");
canvas.style.width = "128px";
canvas.style.height = "128px";
canvas.style.position = "absolute";

await new Promise<void>((resolve) => {
  const watch = new MutationObserver(() => {
    if (sheets.classList.contains("lit")) {
      watch.disconnect();
      resolve();
    }
  });
  watch.observe(sheets, { attributeFilter: ["class"] });
});

async function settled(): Promise<void> {
  await new Promise(requestAnimationFrame);
  await new Promise(requestAnimationFrame);
  await new Promise(requestAnimationFrame);
}

await settled();
const context = canvas.getContext("webgl2");
if (context === null) throw new Error("WebGL 2 is unavailable.");
const draw = context.drawArrays.bind(context);
let draws = 0;
context.drawArrays = (mode, first, count) => {
  draws += 1;
  draw(mode, first, count);
};
const tracing = clipAlone(readSheet(record));
glass.trace({ stage, tracing, frame: 0 });
await settled();
const beforeRepeat = draws;
glass.trace({ stage, tracing, frame: 0 });
await settled();
glass.trace({ stage, tracing, frame: 0 });
await settled();
document.body.dataset.repeated = String(draws - beforeRepeat);
const beforeChange = draws;
glass.trace({ stage, tracing, frame: 1 });
await settled();
document.body.dataset.changed = String(draws - beforeChange);
const beforeResize = draws;
stage.style.width = "80px";
await settled();
document.body.dataset.resized = String(draws - beforeResize);
