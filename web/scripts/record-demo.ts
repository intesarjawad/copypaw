// Run after npm run build: node scripts/record-demo.ts
import { chromium } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";

const root = resolve("../copypaw/app/static");
const scratch = await mkdtemp(join(tmpdir(), "copypaw-demo-"));
const types: Record<string, string> = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".jpg": "image/jpeg",
  ".json": "application/json",
  ".woff2": "font/woff2",
};
const browser = await chromium.launch({
  executablePath: "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
});
try {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: scratch, size: { width: 1280, height: 800 } },
  });
  await context.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== "https://copypaw.intesarjawad.com") return route.abort();
    const path = url.pathname;
    const file = resolve(root, `.${path === "/" ? "/index.html" : path}`);
    if (!file.startsWith(`${root}/`)) return route.abort();
    try {
      await route.fulfill({
        body: await readFile(file),
        contentType: types[extname(file)] ?? "application/octet-stream",
      });
    } catch {
      await route.abort();
    }
  });
  const page = await context.newPage();
  const started = Date.now();
  await page.goto("https://copypaw.intesarjawad.com/");
  await page.locator(".table.lit").waitFor();
  await page.waitForTimeout(1000);
  const start = (Date.now() - started) / 1000;
  await page.waitForTimeout(3000);
  await page.locator(".lamp").click();
  await page.waitForTimeout(2500);
  await page.locator(".lamp").click();
  await page.waitForTimeout(4500);
  const video = page.video();
  await context.close();
  if (!video) throw new Error("Recording was not created");
  const input = await video.path();
  execFileSync(
    "ffmpeg",
    [
      "-y",
      "-ss",
      String(start),
      "-i",
      input,
      "-t",
      "10",
      "-vf",
      "fps=15,scale=960:-2:flags=lanczos",
      "-an",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "../docs/demo.mp4",
    ],
    { stdio: "ignore" },
  );
  execFileSync(
    "ffmpeg",
    [
      "-y",
      "-i",
      "../docs/demo.mp4",
      "-filter_complex",
      "[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
      "../docs/demo.gif",
    ],
    { stdio: "ignore" },
  );
  if (process.argv.includes("--contact-sheet")) {
    await mkdir("../build", { recursive: true });
    execFileSync(
      "ffmpeg",
      [
        "-y",
        "-i",
        "../docs/demo.gif",
        "-vf",
        "scale=192:-1,tile=15x10",
        "-frames:v",
        "1",
        "../build/demo_contact_sheet.png",
      ],
      { stdio: "ignore" },
    );
  }
} finally {
  await browser.close();
  await rm(scratch, { recursive: true, force: true });
}
