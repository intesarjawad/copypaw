// Photograph the light table for the page's still frame, the README and the social card.
// Start the dev server first (npm run dev), then run npm run capture.
import { chromium, type Page } from "@playwright/test";

const url = "http://localhost:5173/";
const STILL_FRAME = 18;

async function open(page: Page, frame: number): Promise<void> {
  await page.goto(url);
  await page.locator(".table.lit").waitFor();
  await page.getByRole("button", { name: "Stop" }).click();
  const slider = page.getByRole("slider", { name: "Light table reel" });
  await slider.press("Home");
  for (let step = 0; step < frame; step += 1) {
    await slider.press("ArrowRight");
  }
  await slider.blur();
  await page.mouse.move(0, 0);
  // Two frames: one to draw, one to present.
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            resolve(null);
          });
        }),
      ),
  );
}

const browser = await chromium.launch();
try {
  const wide = await browser.newPage({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  await open(wide, STILL_FRAME);
  await wide.screenshot({ path: "../docs/hero.jpg", type: "jpeg", quality: 82 });
  // The still stands in for the canvas over the table alone, so the room around it stays dark;
  // the words and buttons on and around the table stay out of it.
  await wide.addStyleTag({
    content: ".intro, .under, .rail-controls, .xsheet { visibility: hidden; }",
  });
  const desk = await wide.locator(".desk").boundingBox();
  if (desk === null) {
    throw new Error("The light table did not lay out.");
  }
  await wide.screenshot({
    path: "public/light-table/still.jpg",
    type: "jpeg",
    quality: 80,
    clip: desk,
  });

  const card = await browser.newPage({
    viewport: { width: 1280, height: 640 },
    deviceScaleFactor: 1,
  });
  await open(card, STILL_FRAME);
  // The card is cut at 640 px; the line under the table would be cut mid-line.
  await card.addStyleTag({
    content: ".table { min-height: 640px; } .under { display: none; }",
  });
  // Rolling the reel scrolled the slider into view; the card starts at the top of the page.
  await card.evaluate(() => {
    scrollTo(0, 0);
  });
  // Let the table lay out again and the shader redraw at the new size.
  await card.waitForTimeout(300);
  await card.screenshot({ path: "../docs/social-preview.png" });
} finally {
  await browser.close();
}
