import { expect, test, type Page, type Route } from "@playwright/test";
import { JOINT_IDS } from "../src/joints";
import { STATES, type StateName } from "../src/states";

const DRAWING = "../samples/worst_case/plain.png";
const SIZES = [
  { width: 1440, height: 900 },
  { width: 390, height: 844 },
] as const;

// Fixture geometry for the 600 x 800 sample drawing; the joints only need to sit inside it.
const RIG = {
  rig: "r1",
  width: 600,
  height: 800,
  box_height: 400,
  outline: [
    [100, 200],
    [500, 200],
    [500, 600],
    [100, 600],
  ],
  joints: Object.fromEntries(
    JOINT_IDS.map((id, index) => [id, [120 + (index % 4) * 110, 220 + Math.floor(index / 4) * 95]]),
  ),
};
const RESULTS = {
  video: "/api/files/jobs/j/side_by_side.mp4",
  gif: "/api/files/jobs/j/side_by_side.gif",
  mp4: "/api/files/jobs/j/side_by_side.mp4",
  bvh: "/api/files/jobs/j/motion.bvh",
  character: "/api/files/jobs/j/character.zip",
  clip: "/api/files/jobs/j/clip.mp4",
  skeleton: "/api/files/jobs/j/skeleton.json",
};
const SKELETON = {
  fps: 15,
  facing: "right",
  edges: [[0, 1]],
  frames: [
    [
      [10, 10],
      [40, 40],
    ],
  ],
};
const RIG_ERROR = {
  code: "mask_too_small",
  what: "The drawing covers too little of the page.",
  next: "Draw a larger pet on a plain, light background.",
};
const JOB_ERROR = {
  code: "internal",
  what: "Copypaw hit a bug it did not expect.",
  next: "Run `copypaw doctor` and open an issue with its output.",
};

interface Latch {
  readonly wait: Promise<void>;
  readonly open: () => void;
}

function latch(): Latch {
  let open = (): void => undefined;
  const wait = new Promise<void>((resolve) => {
    open = resolve;
  });
  return { wait, open };
}

interface Mocks {
  rig: (route: Route) => Promise<void>;
  job: () => object;
}

function json(route: Route, body: object, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function mockApi(page: Page, mocks: Mocks): Promise<void> {
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/health") {
      await json(route, {
        version: "0.1.0",
        models: { state: "ready", downloaded: 0, total: 0 },
        gallery: [{ id: "walk", thumbnail: "/api/files/gallery/walk.jpg" }],
      });
    } else if (path === "/api/rig") {
      await mocks.rig(route);
    } else if (path === "/api/joints") {
      await json(route, { character: RESULTS.character });
    } else if (path === "/api/animate") {
      await json(route, { job: "j" });
    } else if (path === "/api/jobs/j") {
      await json(route, mocks.job());
    } else if (path.endsWith(".json")) {
      await json(route, SKELETON);
    } else if (path.endsWith(".jpg")) {
      await route.fulfill({ path: DRAWING, contentType: "image/png" });
    } else {
      await route.fulfill({ body: "file", contentType: "application/octet-stream" });
    }
  });
}

async function readyLayout(page: Page): Promise<void> {
  await page.evaluate(() => document.fonts.ready);
  await expect(page.locator("header.table")).toHaveClass(/\b(?:lit|no-glass)\b/);
}

async function shoot(page: Page, state: StateName, seen: Set<StateName>): Promise<void> {
  await expect(page.locator("body")).toHaveAttribute("data-state", state, { timeout: 30_000 });
  await readyLayout(page);
  for (const size of SIZES) {
    await page.setViewportSize(size);
    await page.screenshot({
      path: `${process.env.SHOTS_DIR ?? "screenshots"}/${state}-${String(size.width)}x${String(size.height)}.png`,
      fullPage: true,
    });
  }
  seen.add(state);
}

test("every listed state is reachable and captured at both sizes", async ({ page }) => {
  // Capturing 18 full-page screenshots takes longer than the default timeout.
  test.setTimeout(900_000);
  const seen = new Set<StateName>();
  let rigGate = latch();
  let rigReply: "error" | "ok" = "error";
  let job: object = { state: "running", stage: "render", results: {}, timing: null, error: null };
  await mockApi(page, {
    rig: async (route) => {
      await rigGate.wait;
      await (rigReply === "error" ? json(route, RIG_ERROR, 400) : json(route, RIG));
    },
    job: () => job,
  });

  await page.goto("/#t=test-token");
  await shoot(page, "drawingEmpty", seen);

  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await shoot(page, "drawingLoading", seen);
  rigGate.open();
  await shoot(page, "drawingFailed", seen);
  await expect(page.getByText(RIG_ERROR.what)).toBeVisible();

  rigReply = "ok";
  rigGate = latch();
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await expect(page.locator("body")).toHaveAttribute("data-state", "drawingLoading");
  rigGate.open();
  await expect(page.locator("svg .dot")).toHaveCount(16);
  await expect(page.locator("svg .dot.selected")).toHaveAttribute("data-joint", "root");
  await shoot(page, "dotsEditing", seen);

  await page.getByRole("radio", { name: "Walk", exact: true }).check();
  await expect(
    page.getByRole("checkbox", { name: "Add the Copypaw link under the saved GIF and MP4" }),
  ).toBeChecked();
  await shoot(page, "motionReady", seen);

  await page.getByRole("button", { name: "Animate" }).click();
  await shoot(page, "animating", seen);

  job = { state: "failed", stage: "render", results: {}, timing: null, error: JOB_ERROR };
  await shoot(page, "animateFailed", seen);

  job = { state: "running", stage: "render", results: {}, timing: null, error: null };
  await page.getByRole("button", { name: "Animate" }).click();
  await expect(page.locator("body")).toHaveAttribute("data-state", "animating");
  job = { state: "done", stage: "done", results: RESULTS, timing: "Steps match.", error: null };
  await shoot(page, "result", seen);
  for (const name of ["Download GIF", "Download MP4", "Download BVH", "Download character (zip)"]) {
    await expect(page.getByRole("link", { name })).toHaveAttribute("href", /^blob:/);
  }

  expect([...seen].sort()).toEqual(Object.keys(STATES).sort());
});

test("a click moves the selected dot and advances the selection", async ({ page }) => {
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  await page.goto("/#t=test-token");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  const root = page.locator('svg .dot[data-joint="root"]');
  await expect(root).toHaveAttribute("cx", "120");

  const overlay = page.locator("svg.overlay").first();
  await readyLayout(page);
  await overlay.scrollIntoViewIfNeeded();
  const box = await overlay.boundingBox();
  expect(box).not.toBeNull();
  if (box !== null) {
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  }

  await expect.poll(async () => Number(await root.getAttribute("cx"))).toBeCloseTo(300, 0);
  await expect(page.locator("svg .dot.selected")).toHaveAttribute("data-joint", "hip");
});

test("dragging a pin moves that joint and chooses it", async ({ page }) => {
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  await page.goto("/#t=test-token");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  const neck = page.locator('svg .dot[data-joint="neck"]');
  await expect(neck).toHaveAttribute("cx", "450");

  const overlay = page.locator("svg.overlay").first();
  await readyLayout(page);
  await overlay.scrollIntoViewIfNeeded();
  const box = await overlay.boundingBox();
  expect(box).not.toBeNull();
  if (box !== null) {
    const scale = box.width / RIG.width;
    await page.mouse.move(box.x + 450 * scale, box.y + 220 * scale);
    await page.mouse.down();
    const targetX = Math.round(box.x + 300 * scale);
    const targetY = Math.round(box.y + 700 * scale);
    await page.mouse.move(targetX, targetY, { steps: 4 });
    await page.mouse.up();
    await expect
      .poll(async () => Number(await neck.getAttribute("cx")))
      .toBeCloseTo((targetX - box.x) / scale, 0);
    await expect
      .poll(async () => Number(await neck.getAttribute("cy")))
      .toBeCloseTo((targetY - box.y) / scale, 0);
  }

  await expect(page.locator("svg .dot.selected")).toHaveAttribute("data-joint", "neck");
});

test("arrow keys nudge the focused pin by 1% of the box height, Shift by 5%", async ({ page }) => {
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  await page.goto("/#t=test-token");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  const root = page.locator('svg .dot[data-joint="root"]');
  await expect(root).toHaveAttribute("cx", "120");

  await page.locator('svg .dot[data-joint="root"]').focus();
  await page.keyboard.press("ArrowRight");
  await expect(root).toHaveAttribute("cx", "124");
  await page.keyboard.press("Shift+ArrowDown");
  await expect(root).toHaveAttribute("cy", "240");
});

test("Tab moves to the next pin and chooses it", async ({ page }) => {
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  await page.goto("/#t=test-token");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await page.locator('svg .dot[data-joint="root"]').focus();
  await page.keyboard.press("Tab");
  await expect(page.locator("svg .dot.selected")).toHaveAttribute("data-joint", "hip");
  await expect(page.getByRole("button", { name: "Hip", pressed: true })).toBeFocused();
});

test("a late clip response cannot replace a gallery selection", async ({ page }) => {
  const gate = latch();
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  await page.route("**/api/capture", async (route) => {
    await gate.wait;
    await json(route, { job: "clip-job" });
  });
  await page.route("**/api/animate", (route) => json(route, { job: "j" }));
  await page.goto("/#t=test-token");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await page
    .locator("#clip-input")
    .setInputFiles({ name: "clip.mp4", mimeType: "video/mp4", buffer: Buffer.from("clip") });
  await expect(page.getByText("Tracking the clip starts now; Animate waits for it.")).toBeVisible();
  await page.getByRole("radio", { name: "Walk", exact: true }).check();
  const response = page.waitForResponse("**/api/capture");
  gate.open();
  await response;
  const submitted = page.waitForRequest("**/api/animate");
  await page.getByRole("button", { name: "Animate" }).click();
  expect((await submitted).postDataJSON()).toMatchObject({ gallery: "walk" });
});

test("a completed clip clears its tracking status", async ({ page }) => {
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  const gate = latch();
  await page.route("**/api/capture", async (route) => {
    await gate.wait;
    await json(route, { job: "clip-job" });
  });
  await page.goto("/#t=test-token");
  await page
    .locator("#clip-input")
    .setInputFiles({ name: "clip.mp4", mimeType: "video/mp4", buffer: Buffer.from("clip") });
  const status = page.getByText("Tracking the clip starts now; Animate waits for it.");
  await expect(status).toBeVisible();
  gate.open();
  await expect(status).toBeHidden({ timeout: 30_000 });
});

for (const reply of ["success", "failure"] as const) {
  for (const pending of [true, false]) {
    test(`a late drawing ${reply} preserves the newer ${pending ? "loading" : "ready"} photo`, async ({
      page,
    }) => {
      const first = latch();
      const second = latch();
      let requests = 0;
      await mockApi(page, {
        rig: async (route) => {
          const index = ++requests;
          await (index === 1 ? first.wait : second.wait);
          await (index === 1
            ? reply === "success"
              ? json(route, { ...RIG, rig: "old" })
              : json(route, RIG_ERROR, 400)
            : json(route, RIG));
        },
        job: () => ({ state: "done", results: RESULTS, timing: null }),
      });
      await page.goto("/#t=test-token");
      await page.locator("#drawing-input").setInputFiles(DRAWING);
      await expect.poll(() => requests).toBe(1);
      await page.locator("#drawing-input").setInputFiles("../copypaw/selftest/dragon.png");
      await expect.poll(() => requests).toBe(2);
      if (!pending) {
        second.open();
        await expect(page.locator("body")).toHaveAttribute("data-state", "dotsEditing");
      }
      const response = page.waitForResponse("**/api/rig");
      first.open();
      await (await response).finished();
      await page.evaluate(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
      expect(await page.locator("body").getAttribute("data-state")).toBe(
        pending ? "drawingLoading" : "dotsEditing",
      );
      await expect(page.getByText(RIG_ERROR.what)).toBeHidden();
      if (pending) {
        await expect(page.getByText("Finding the outline and joints.")).toBeVisible();
        await expect(page.locator(".editor")).toBeHidden();
        second.open();
      }
      await expect(page.locator("body")).toHaveAttribute("data-state", "dotsEditing");
      await page.getByRole("radio", { name: "Walk", exact: true }).check();
      const submitted = page.waitForRequest("**/api/animate");
      await page.getByRole("button", { name: "Animate" }).click();
      expect((await submitted).postDataJSON()).toMatchObject({ rig: RIG.rig });
    });
  }
}

test("Animate names the missing drawing or motion", async ({ page }, testInfo) => {
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  await page.goto("/#t=test-token");
  const animate = page.getByRole("button", { name: "Animate" });
  await expect(animate).toHaveAccessibleDescription("Choose a photo first");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await expect(animate).toBeDisabled();
  await expect(animate).toHaveAccessibleDescription("Choose a clip or gallery motion");
  if (testInfo.project.name === "chromium" && process.env.SHOTS_DIR !== undefined) {
    for (const size of SIZES) {
      await page.setViewportSize(size);
      await animate.evaluate((element) => {
        element.scrollIntoView({ behavior: "instant", block: "end" });
      });
      await page.screenshot({
        path: `${process.env.SHOTS_DIR}/choose-motion-${String(size.width)}x${String(size.height)}.png`,
      });
    }
  }
  await page.getByRole("radio", { name: "Walk", exact: true }).check();
  await expect(animate).toBeEnabled();
  await expect(animate).not.toHaveAttribute("aria-describedby");
});

test("drawing pins wait for the photo to decode", async ({ page }) => {
  const photo = latch();
  await page.addInitScript(() => {
    const createObjectURL = URL.createObjectURL.bind(URL);
    URL.createObjectURL = (blob) =>
      blob instanceof File ? "/chosen-photo.png" : createObjectURL(blob);
  });
  await page.route("**/chosen-photo.png", async (route) => {
    await photo.wait;
    await route.fulfill({ path: DRAWING, contentType: "image/png" });
  });
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  await page.goto("/#t=test-token");
  const response = page.waitForResponse("**/api/rig");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await (await response).finished();
  await page.evaluate(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
  expect(await page.locator("body").getAttribute("data-state")).toBe("drawingLoading");
  await expect(page.locator(".editor")).toBeHidden();
  photo.open();
  await expect(page.locator("body")).toHaveAttribute("data-state", "dotsEditing");
  await expect(page.locator(".editor")).toBeVisible();
});

test("drawing pins wait for fonts to settle the layout", async ({ page }) => {
  const fonts = latch();
  await page.route("**/*.woff2", async (route) => {
    await fonts.wait;
    await route.continue();
  });
  await mockApi(page, { rig: (route) => json(route, RIG), job: () => ({}) });
  await page.goto("/#t=test-token", { waitUntil: "domcontentloaded" });
  const response = page.waitForResponse("**/api/rig");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await (await response).finished();
  await page.evaluate(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
  expect(await page.locator("body").getAttribute("data-state")).toBe("drawingLoading");
  await expect(page.locator(".editor")).toBeHidden();
  fonts.open();
  await expect(page.locator("body")).toHaveAttribute("data-state", "dotsEditing");
  expect(await page.evaluate(() => document.fonts.status)).toBe("loaded");
});

test("desktop result editing keeps the drawing and pins at their editing size", async ({
  page,
}) => {
  await page.setViewportSize(SIZES[0]);
  await mockApi(page, {
    rig: (route) => json(route, RIG),
    job: () => ({ state: "done", results: RESULTS, timing: null }),
  });
  await page.goto("/#t=test-token");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  const editor = page.locator(".editor");
  await expect(editor).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  const before = await editor.boundingBox();
  expect(before).not.toBeNull();
  await page.getByRole("radio", { name: "Walk", exact: true }).check();
  await page.getByRole("button", { name: "Animate" }).click();
  await expect(page.locator("body")).toHaveAttribute("data-state", "result");
  await expect.poll(async () => (await editor.boundingBox())?.width).toBe(before?.width);
  const pin = await page.locator('svg .dot[data-joint="root"]').boundingBox();
  expect(pin?.width).toBeGreaterThan(10);
});

test("mobile errors wrap and result editing retains its full width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  let fail = true;
  await mockApi(page, {
    rig: (route) =>
      fail
        ? json(
            route,
            {
              ...RIG_ERROR,
              next: "https://github.com/intesarjawad/copypaw/issues/new?template=bug.yml",
            },
            400,
          )
        : json(route, RIG),
    job: () => ({ state: "done", results: RESULTS, timing: null }),
  });
  await page.goto("/#t=test-token");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await expect(page.getByText(RIG_ERROR.what)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  fail = false;
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await page.getByRole("radio", { name: "Walk", exact: true }).check();
  await page.getByRole("button", { name: "Animate" }).click();
  await expect(page.locator("body")).toHaveAttribute("data-state", "result", { timeout: 30_000 });
  expect((await page.locator(".editor").boundingBox())?.width).toBeGreaterThan(168);
});
