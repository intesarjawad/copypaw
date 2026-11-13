import { defineConfig, devices } from "@playwright/test";

const BASE_URL = "http://127.0.0.1:4173";

export default defineConfig({
  testDir: "e2e",
  // Each tab animates in a Web Worker and WebGL; more than two at once starve each other.
  workers: 2,
  // A hosted runner without a GPU animates in software, well past the 30 s default.
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: { baseURL: BASE_URL },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    // Headless Firefox has no WebGL 2 without a display. CI runs under xvfb and sets FIREFOX_HEADED.
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"], headless: process.env.FIREFOX_HEADED === undefined },
    },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173 --strictPort",
    url: BASE_URL,
    reuseExistingServer: false,
  },
});
