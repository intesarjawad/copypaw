import { expect, test } from "@playwright/test";

test("the tool glass repaints changed frames and layout, without repainting identical traces", async ({
  page,
}) => {
  await page.goto("/e2e/glass.html");
  const body = page.locator("body");
  await expect(body).toHaveAttribute("data-resized", /\d+/);
  expect(await body.getAttribute("data-repeated")).toBe("0");
  expect(await body.getAttribute("data-changed")).toBe("1");
  expect(Number(await body.getAttribute("data-resized"))).toBeGreaterThan(0);
});
