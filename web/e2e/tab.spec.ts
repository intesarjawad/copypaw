import { expect, test } from "@playwright/test";
import { STRINGS } from "../src/strings";

const DRAWING = "../copypaw/selftest/dragon.png";
// Frames the CLI writes at 0, 1.6 and 3.2 s of the sample walk, at its 15 frames a second.
const FRAMES = [0, 24, 48];
// Rigging, posing and solving every frame in the worker, with headroom for a busy machine.
const ANIMATE_TIMEOUT_MS = 60_000;

test("animates the sample drawing in the tab with no server", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    errors.push(error.message);
  });
  await page.route("**/api/**", async (route) => {
    errors.push(`The page asked the server for ${route.request().url()}`);
    await route.abort();
  });

  await page.goto("/");
  await expect(page.getByText(STRINGS.modelsAbsent)).toBeVisible();
  await expect(page.locator("#clip-input")).toBeHidden();
  const chooser = page.waitForEvent("filechooser");
  await page.getByText(STRINGS.choosePhoto, { exact: true }).click();
  await (await chooser).setFiles(DRAWING);
  // The tab offers one walk and no clip of its own, so the walk is chosen already.
  await expect(page.locator("body")).toHaveAttribute("data-state", "motionReady", {
    timeout: 30_000,
  });
  await expect(page.locator("section.motion")).toBeHidden();
  await page.getByRole("button", { name: STRINGS.animate }).click();

  const canvas = page.getByRole("img", { name: STRINGS.animationLabel });
  await expect(canvas).toBeVisible({ timeout: ANIMATE_TIMEOUT_MS });
  await expect(page.locator("body")).toHaveAttribute("data-state", "result");
  const slider = page.getByRole("slider", { name: STRINGS.frame });
  const shown: string[] = [];
  for (const frame of FRAMES) {
    await slider.fill(String(frame));
    shown.push(await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL()));
    await canvas.screenshot({ path: testInfo.outputPath(`frame-${String(frame)}.png`) });
  }
  const timing = (await page.locator(".timing").textContent()) ?? "";
  testInfo.annotations.push({ type: "timing", description: timing });

  expect(new Set(shown).size).toBe(FRAMES.length);
  expect(errors).toEqual([]);
  if (testInfo.project.name === "chromium" && process.env.SHOTS_DIR !== undefined) {
    for (const size of [
      { width: 1440, height: 900 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(size);
      await page.locator(".rail").evaluate((element) => {
        element.scrollIntoView({ behavior: "instant", block: "end" });
      });
      await page.screenshot({
        path: `${process.env.SHOTS_DIR}/live-result-${String(size.width)}x${String(size.height)}.png`,
      });
    }
  }
});

test("worker load failure rejects uploads with a retry instruction", async ({ page }, testInfo) => {
  await page.addInitScript(() => {
    class FailedWorker extends EventTarget {
      postMessage(): void {
        this.dispatchEvent(new Event("error"));
      }
    }
    Object.defineProperty(window, "Worker", { value: FailedWorker });
  });
  await page.goto("/");
  await page.locator("#drawing-input").setInputFiles(DRAWING);
  await expect(page.getByText("Copypaw could not start its processing worker.")).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByText("Next: Reload this page and choose your photo again.")).toBeVisible();
  if (testInfo.project.name === "chromium" && process.env.SHOTS_DIR !== undefined) {
    for (const size of [
      { width: 1440, height: 900 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(size);
      await page.screenshot({
        path: `${process.env.SHOTS_DIR}/worker-failed-${String(size.width)}x${String(size.height)}.png`,
        fullPage: true,
      });
    }
  }
});

test("an unreadable image gets the PNG or JPEG recovery message", async ({ page }) => {
  await page.goto("/");
  await page.locator("#drawing-input").setInputFiles({
    name: "broken.png",
    mimeType: "image/png",
    buffer: Buffer.from("not an image"),
  });
  await expect(page.getByText("Copypaw could not read this image.")).toBeVisible();
  await expect(page.getByText(/Save the drawing as a PNG or JPEG/)).toBeVisible();
});
