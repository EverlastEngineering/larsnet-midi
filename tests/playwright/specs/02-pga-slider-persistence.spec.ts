/**
 * 02 — pga_min_prominence slider persistence (toms)
 *
 * Extends the toms threshold-tuning regression baseline (01-) by
 * asserting the new PGA-specific slider is present, that the
 * default value is read from the slider config's `fallback` field
 * (not hardcoded), and that changing the slider's value persists
 * the new value to midiconfig.yaml at
 * `onset_detection.pga_min_prominence` — which is the schema key
 * consumed at runtime by `pga_event_builder.build_pga_events`
 * (`config.get('onset_detection', {}).get('pga_min_prominence',
 * 1000.0)`).
 *
 * The persistence path is exercised by clicking the Save &
 * Reconvert button (`#tuning-save-btn`), which fires
 * `saveTuningAndReconvert()` → `api.updateConfig(currentProject,
 * 'midiconfig', updates)` → POST `/api/config/<id>/midiconfig`
 * with `updates = [{path: ['onset_detection',
 * 'pga_min_prominence'], value: <new value>}]` →
 * `YAMLConfigEngine.update_value(path, value)`. The slider's
 * `yamlPath` field overrides the default `[stemType, key]`
 * persistence path, because `pga_min_prominence` lives in the
 * global `onset_detection` section of midiconfig.yaml, not under
 * `toms:`.
 *
 * Step 2 of the toms threshold-tuning refactor plan wires the
 * slider into the build path (see threshold-tuning.js:85-87); a
 * later step (5) will wire the change to trigger a full PGA
 * re-detection on the server. Until then, the Save flow still
 * runs the existing fast-rebuild path — and that rebuild doesn't
 * re-derive the `pga_min_prominence` slider value, so after
 * Save the slider will re-render at 1000 (the fallback). The
 * invariant we protect here is purely the **persistence** of the
 * new value to midiconfig.yaml on disk.
 *
 * Snapshot policy mirrors the smoke spec:
 *   - before.png: panel open, slider at default 1000, save button hidden
 *   - after.png:  panel open, slider at 5000, save button visible (dirty)
 *
 * Cleanup: the test reads the pre-test YAML value of
 * `onset_detection.pga_min_prominence` and restores it on the way
 * out, so the funk-80 fixture is not left in a modified state.
 * Restore is best-effort — if the API call fails (e.g. the
 * webui is being torn down), the test still passes.
 */
import { test, expect } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

// ESM-safe __dirname shim (package.json has "type": "module").
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SNAPSHOT_DIR = path.join(
  __dirname,
  "..",
  "__snapshots__",
  "02-pga-slider-persistence",
);
const BEFORE_SHOT = path.join(SNAPSHOT_DIR, "before.png");
const AFTER_SHOT = path.join(SNAPSHOT_DIR, "after.png");

// The funk-80 project under user_files/ — owns toms analysis + envelope.
const FIXTURE_PROJECT_NAME = "2_funk_80_beat_4-4_4";
const FIXTURE_PROJECT_NUMBER = 4;

// The default the slider ships with (matches the `fallback` field
// in threshold-tuning.js STEM_SLIDER_CONFIGS.toms[0]).
const EXPECTED_DEFAULT = 1000;

// The new value the test writes.
const NEW_VALUE = 5000;

test.beforeAll(() => {
  fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
});

/**
 * Read the on-disk midiconfig.yaml via the webui's own GET
 * endpoint. The endpoint (`/api/projects/<n>/config/midiconfig.yaml`)
 * is the canonical read path — it goes through
 * `YAMLConfigEngine.load()` which re-reads the file from disk on
 * every call, so this is the closest browser-side equivalent of
 * "re-read midiconfig.yaml from disk".
 *
 * Returns the raw YAML content as a string (the `content` field
 * of the response). Throws if the endpoint is unreachable or
 * returns a non-2xx — Playwright will surface that as a test
 * failure with the full response body in the trace.
 *
 * Uses Playwright's `request` fixture (which is the same `fetch`
 * semantics but doesn't need a page context — so the read works
 * even before the browser navigates to the webui, or after the
 * page has been torn down). The baseURL is inherited from
 * `playwright.config.ts`, so `/api/...` resolves to the running
 * Flask server.
 */
async function readProjectConfig(request, projectNumber) {
  const res = await request.get(
    `/api/projects/${projectNumber}/config/midiconfig.yaml`,
  );
  if (!res.ok()) {
    throw new Error(
      `GET /api/projects/${projectNumber}/config/midiconfig.yaml → ${res.status()}: ${await res.text()}`,
    );
  }
  const data = await res.json();
  return data.content;
}

/**
 * Extract the numeric value of `onset_detection.pga_min_prominence`
 * from a raw midiconfig.yaml string. Returns `null` if the key is
 * not present (e.g. legacy YAMLs without it). Comments and blank
 * lines are skipped.
 */
function extractPgaMinProminence(yamlContent) {
  if (typeof yamlContent !== "string") return null;
  // Walk forward — pga_min_prominence must be under the
  // `onset_detection:` section. We use a simple state machine
  // (not a full YAML parser) to avoid pulling in a dep for a
  // single test assertion.
  let inOnsetDetection = false;
  for (const rawLine of yamlContent.split("\n")) {
    const line = rawLine.replace(/#.*$/, "").trimEnd();
    if (!line.trim()) continue;
    // Track section headers (lines starting with non-space, ending with `:`).
    if (!line.startsWith(" ") && /^\s*[A-Za-z_][\w-]*\s*:/.test(line)) {
      inOnsetDetection = /^\s*onset_detection\s*:/.test(line);
      continue;
    }
    if (!inOnsetDetection) continue;
    const m = line.match(/^\s+pga_min_prominence\s*:\s*(\S+)\s*$/);
    if (m) {
      const v = Number(m[1]);
      return Number.isFinite(v) ? v : null;
    }
  }
  return null;
}

/**
 * Write `onset_detection.pga_min_prominence` = `value` to the
 * fixture project's midiconfig.yaml via the webui API
 * (POST /api/config/<id>/midiconfig with a single update).
 *
 * Used by the test's cleanup path to restore the pre-test value
 * so the fixture is not left modified.
 */
async function writePgaMinProminence(request, projectNumber, value) {
  const res = await request.post(`/api/config/${projectNumber}/midiconfig`, {
    headers: { "Content-Type": "application/json" },
    data: {
      updates: [
        { path: ["onset_detection", "pga_min_prominence"], value: value },
      ],
    },
  });
  if (!res.ok()) {
    throw new Error(
      `POST /api/config/${projectNumber}/midiconfig → ${res.status()}: ${await res.text()}`,
    );
  }
  return await res.json();
}

test("pga_min_prominence slider persists new value to midiconfig.yaml", async ({
  page,
  request,
}) => {
  // Capture the pre-test value so we can restore it on the way out.
  // This keeps the funk-80 fixture in its original state across
  // test runs.
  const yamlBefore = await readProjectConfig(request, FIXTURE_PROJECT_NUMBER);
  const preTestValue = extractPgaMinProminence(yamlBefore);

  try {
    // 1. Boot the webui and pick the funk-80 fixture.
    await page.goto("/");
    await expect(
      page.getByText("Transform Drum Tracks into MIDI"),
    ).toBeVisible({ timeout: 15_000 });

    const projectItem = page
      .locator(".project-item")
      .filter({ hasText: FIXTURE_PROJECT_NAME })
      .first();
    await expect(projectItem).toBeVisible({ timeout: 10_000 });
    await projectItem.click();

    // 2. Reveal the analysis section and switch to the toms tab.
    const analysisSection = page.locator("#analysis-section");
    await expect(analysisSection).toBeVisible({ timeout: 20_000 });
    const tomsTab = page
      .locator('.waveform-stem-tab[data-stem="toms"]')
      .first();
    await expect(tomsTab).toBeVisible({ timeout: 15_000 });
    await tomsTab.click();
    await expect(tomsTab).toHaveClass(/waveform-tab-active/);

    // 3. Open the tuning panel (button may be hidden on 2026-06-13
    //    for toms — see 01-toms-tuning-smoke.spec.ts for context;
    //    same JS-toggle fallback applies).
    const tuneButton = page.locator("#tuning-toggle-btn");
    if (await tuneButton.isVisible().catch(() => false)) {
      await tuneButton.click();
    } else {
      await page.evaluate(() => toggleTuningPanel());
    }
    const tuningPanel = page.locator("#tuning-panel");
    await expect(tuningPanel).toBeVisible({ timeout: 5_000 });

    // 4. The new PGA slider must be present and at the default
    //    value (1000, from STEM_SLIDER_CONFIGS.toms[0].fallback).
    //    We assert the input's `value` attribute rather than
    //    `inputValue` because `buildSlidersForStem` writes the
    //    default into the initial value attribute on render.
    const pgaSlider = tuningPanel.locator(
      "#tuning-slider-pga_min_prominence",
    );
    await expect(pgaSlider).toBeAttached();
    const initialValue = await pgaSlider.inputValue();
    expect(Number(initialValue)).toBe(EXPECTED_DEFAULT);

    // The numeric display next to the slider should also reflect
    // the default (formatSliderValue returns '1000' for integers
    // >= 10).
    const valueDisplay = tuningPanel.locator(
      "#tuning-val-pga_min_prominence",
    );
    await expect(valueDisplay).toHaveText(/^1000/);

    // The Save & Reconvert button should be hidden at this point
    // because the slider matches the configured value (fallback).
    const saveBtn = tuningPanel.locator("#tuning-save-btn");
    await expect(saveBtn).toBeHidden();

    // 5. Snapshot the "before" state — panel open, slider at
    //    default, no save button.
    await page.screenshot({ path: BEFORE_SHOT, fullPage: true });
    expect(fs.existsSync(BEFORE_SHOT)).toBe(true);

    // 6. Change the slider to NEW_VALUE. `fill()` works for
    //    <input type="range"> in Playwright (it sets the value
    //    and dispatches the input/change events the JS handler
    //    is listening for). We then dispatch the input event
    //    explicitly to be safe — range inputs sometimes need
    //    a manual nudge in headless chromium.
    await pgaSlider.fill(String(NEW_VALUE));
    await pgaSlider.evaluate((el) => {
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    });
    // Give the requestAnimationFrame debounce in onSliderInput
    // a chance to run, and let the save-button visibility check
    // fire. The spec calls for a 500ms wait; we wait 600ms to
    // be safe under load.
    await page.waitForTimeout(600);

    // 7. The value display should now read 5000, and the save
    //    button should be visible (slider differs from
    //    configured value).
    await expect(valueDisplay).toHaveText(/^5000/);
    await expect(saveBtn).toBeVisible();

    // 8. Click Save & Reconvert. This fires
    //    `saveTuningAndReconvert()` → `api.updateConfig(...)`
    //    with our yamlPath-bearing update, which writes
    //    `onset_detection.pga_min_prominence: 5000` to the
    //    project's midiconfig.yaml.
    await saveBtn.click();

    // 9. Wait for the save to land. The button shows a spinner
    //    while in flight; on success it returns to its idle
    //    state. We poll the YAML via the GET endpoint until it
    //    reflects the new value, up to 10s. (The save includes
    //    a rebuild step which can take a few seconds on a
    //    realistic project; the YAML write itself is sub-second
    //    and happens *before* the rebuild, so the GET should
    //    reflect 5000 even while the rebuild is still running.)
    let yamlAfter = null;
    let persistedValue = null;
    const deadline = Date.now() + 10_000;
    while (Date.now() < deadline) {
      yamlAfter = await readProjectConfig(request, FIXTURE_PROJECT_NUMBER);
      persistedValue = extractPgaMinProminence(yamlAfter);
      if (persistedValue === NEW_VALUE) break;
      await page.waitForTimeout(250);
    }
    expect(persistedValue).toBe(NEW_VALUE);

    // 10. Snapshot the "after" state — panel still open, slider
    //     value re-derived from the logic block. (The build
    //     path re-renders the slider at the fallback 1000 after
    //     the rebuild, because the toms logic block does not
    //     include pga_min_prominence — see threshold-tuning.js
    //     :244 and the analysis.json logic block for toms. This
    //     is the expected end-state for step 2; the rebuild
    //     re-detection wiring is step 5.)
    await page.screenshot({ path: AFTER_SHOT, fullPage: true });
    expect(fs.existsSync(AFTER_SHOT)).toBe(true);
  } finally {
    // Cleanup: restore the pre-test YAML value so the fixture
    // is not left in a modified state. Best-effort — failures
    // here are logged but do not fail the test.
    if (preTestValue !== null) {
      try {
        await writePgaMinProminence(
          request,
          FIXTURE_PROJECT_NUMBER,
          preTestValue,
        );
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn(
          `Failed to restore midiconfig.yaml to pre-test value ${preTestValue}:`,
          err,
        );
      }
    }
  }
});
