import { defineConfig } from "vitest/config";

export default defineConfig({
  // The wheel serves the built page from the Python package.
  build: { outDir: "../copypaw/app/static", emptyOutDir: true },
  // Error text is read from the Python package's messages.json.
  server: { fs: { allow: [".."] } },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
