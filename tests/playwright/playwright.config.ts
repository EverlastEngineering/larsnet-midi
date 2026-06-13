/**
 * Playwright config for the larsnet webui E2E suite.
 *
 * The webui dev server is a long-running Flask process on port 4915
 * (hardcoded in `webui/app.py::main()` — `app.run(host='0.0.0.0',
 * port=4915, ...)`). For local development we assume it is already
 * running; CI may launch it via `start_webui.sh` (or equivalent) before
 * invoking `npm test`.
 *
 * Snapshot policy:
 *   - On every test: a deterministic baseline screenshot is written
 *     into `__snapshots__/<spec-name>/<test-name>.png` so the verifier
 *     can eyeball the UI even when the test passes.
 *   - On failure: Playwright auto-captures a screenshot + video trace
 *     into the same `__snapshots__/<spec-name>/` tree for triage.
 */
import { defineConfig, devices } from "@playwright/test";

const PORT = process.env.LARSNET_WEBUI_PORT ?? "4915";

export default defineConfig({
  testDir: "./specs",
  fullyParallel: false, // serial — the shared webui has a single job queue
  forbidOnly: !!process.env.CI,
  retries: 0, // smoke spec is the regression baseline; we want a red light on flake
  workers: 1,
  reporter: [
    ["list"],
    ["html", { outputFolder: "__snapshots__/html-report", open: "never" }],
  ],

  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
    video: "retain-on-failure",
    screenshot: "only-on-failure",
    actionTimeout: 10_000,
    navigationTimeout: 30_000,
  },

  timeout: 60_000, // slider debounce ~300ms + endpoint round-trip + Flask cold start
  expect: { timeout: 5_000 },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
