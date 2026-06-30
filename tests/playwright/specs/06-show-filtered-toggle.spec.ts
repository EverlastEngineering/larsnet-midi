/**
 * 06 — "Show Filtered" toggle rendering regression
 *
 * User report (2026-06-30): clicking the "Show Filtered" checkbox in
 * the waveform panel doesn't make filtered events appear until a
 * threshold slider is moved. Then they suddenly show up.
 *
 * Test signal: the legend bar (`#waveform-legend-items`) shows
 *   - "Kept (N)" — always
 *   - "PGA (M)" — when the stem has events_pga
 *   - "Filtered (K)" — ONLY when the toggle is ON and there are
 *     filtered events
 * So the legend's text content is the test signal. When the toggle
 * works, ON state includes "Filtered (K)" with K > 0 and OFF state
 * does not. The regression manifests as: ON state has no "Filtered"
 * entry (the legend update isn't running).
 *
 * Fixture: project #6 (Taylor Swift), snare stem (~1200 filtered
 * events — the biggest spread, most likely to expose the bug).
 *
 * CRITICAL: do NOT touch any slider. The whole point of this test
 * is that the toggle should work standalone.
 */
import { test, expect } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SNAPSHOT_DIR = path.join(
  __dirname,
  "..",
  "__snapshots__",
  "06-show-filtered-toggle",
);

const FIXTURE_PROJECT_NAME = "01_Taylor_Swift_The_Fate_of_Ophelia_Drums";

test.beforeAll(() => {
  fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
});

/** Read the legend's full text content (e.g. "Kept (173)PGA (811)"). */
async function readLegendText(page): Promise<string> {
  return await page.locator("#waveform-legend-items").innerText();
}

/** Parse a count out of a label like "Filtered (1234)". Returns 0 if absent. */
function parseCount(label: string, prefix: string): number {
  const m = new RegExp(`${prefix}\\s*\\((\\d+)\\)`).exec(label);
  return m ? Number(m[1]) : 0;
}

test("show-filtered toggle renders Filtered legend entry without slider interaction", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByText("Transform Drum Tracks into MIDI")).toBeVisible({
    timeout: 15_000,
  });

  const projectItem = page
    .locator(".project-item")
    .filter({ hasText: FIXTURE_PROJECT_NAME })
    .first();
  await expect(projectItem).toBeVisible({ timeout: 10_000 });
  await projectItem.click();

  const analysisSection = page.locator("#analysis-section");
  await expect(analysisSection).toBeVisible({ timeout: 20_000 });

  // Snare stem — biggest KEPT/FILTERED split for the project.
  const snareTab = page.locator('.waveform-stem-tab[data-stem="snare"]').first();
  await expect(snareTab).toBeVisible({ timeout: 15_000 });
  await snareTab.click();
  await expect(snareTab).toHaveClass(/waveform-tab-active/);

  const filteredToggle = page.locator("#waveform-filtered-toggle");
  await expect(filteredToggle).toBeVisible({ timeout: 5_000 });

  // Scroll the events canvas into view so the screenshot shows the
  // waveform (not the page header). The bug manifests as missing
  // red bars on the canvas — need to see the canvas.
  const eventsCanvas = page.locator("#events-canvas");
  await eventsCanvas.scrollIntoViewIfNeeded();
  await page.waitForTimeout(200);

  // CRITICAL: the user reports the toggle doesn't show filtered
  // events unless they first click "Tune" (which enters tuning
  // mode). The test exercises BOTH paths: first without clicking
  // Tune (the bug path), then with Tune clicked (the workaround).
  // After the fix, both should work the same.

  // === Path A: WITHOUT clicking Tune ===
  // Make sure we start in the OFF state.
  if (await filteredToggle.isChecked()) {
    await filteredToggle.click();
    await page.waitForTimeout(200);
  }

  // Baseline: toggle OFF. Legend should NOT include "Filtered (N)".
  const offLegend = await readLegendText(page);
  await page.screenshot({
    path: path.join(SNAPSHOT_DIR, "01-toggle-off-baseline.png"),
    fullPage: false,
  });
  const offFiltered = parseCount(offLegend, "Filtered");
  const offKept = parseCount(offLegend, "Kept");
  console.log(`TOGGLE OFF legend: "${offLegend}"`);
  console.log(`  → Kept=${offKept}  Filtered=${offFiltered}`);
  expect(offFiltered, "toggle OFF must not show Filtered legend entry").toBe(0);
  expect(offKept, "Kept legend should be present").toBeGreaterThan(0);

  // Toggle ON. Legend SHOULD include "Filtered (N)" with N > 0.
  // CRITICAL: do NOT touch any slider before this measurement.
  // The regression is "filtered events aren't visible until a
  // slider is moved." If the toggle is wired right, the legend
  // should immediately show the Filtered entry.
  await filteredToggle.click();
  await page.waitForTimeout(200);
  const onLegend = await readLegendText(page);
  await page.screenshot({
    path: path.join(SNAPSHOT_DIR, "02-toggle-on-no-slider.png"),
    fullPage: false,
  });
  const onFiltered = parseCount(onLegend, "Filtered");
  const onKept = parseCount(onLegend, "Kept");
  console.log(`TOGGLE ON legend:  "${onLegend}"`);
  console.log(`  → Kept=${onKept}  Filtered=${onFiltered}`);

  // CRITICAL ASSERTION: the bug manifests as onFiltered == 0
  // (the legend update isn't running). The fix: onFiltered must
  // match the number of FILTERED events in the sidecar.
  expect(
    onFiltered,
    `toggle ON (no slider touch) should reveal Filtered legend entry. ` +
    `Got onLegend="${onLegend}". ` +
    `Bug: legend doesn't update until a slider is touched.`
  ).toBeGreaterThan(0);

  // Stronger assertion: count RED pixels on the events canvas.
  // The events panel draws FILTERED events in red (#ef4444).
  // When the toggle works, ON state should produce a substantial
  // number of red pixels. The user reports that without clicking
  // Tune, the canvas doesn't show the filtered bars — so the
  // red pixel count should be 0 (or close to it). With the fix,
  // it should be > 500.
  const onRed = await page.evaluate(() => {
    const canvas = document.getElementById("events-canvas");
    if (!canvas) return -1;
    const ctx = canvas.getContext("2d");
    if (!ctx) return -1;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let n = 0;
    for (let i = 0; i < img.data.length; i += 4) {
      const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2], a = img.data[i + 3];
      if (a < 200) continue;
      if (Math.abs(r - 239) < 25 && Math.abs(g - 68) < 25 && Math.abs(b - 68) < 25) n++;
    }
    return n;
  });
  console.log(`  → red FILTERED pixels on canvas: ${onRed}`);
  expect(
    onRed,
    `toggle ON (no slider touch) should render red FILTERED bars on the events canvas. ` +
    `Got ${onRed} red pixels. ` +
    `Bug: filtered events visible only after clicking Tune.`
  ).toBeGreaterThan(500);

  // The Kept count should be unchanged by the toggle.
  expect(onKept, "Kept count should not change when toggling show-filtered").toBe(offKept);

  // Toggle OFF again — should match baseline.
  await filteredToggle.click();
  await page.waitForTimeout(200);
  const offAgainLegend = await readLegendText(page);
  expect(parseCount(offAgainLegend, "Filtered"), "toggle OFF should remove Filtered entry").toBe(0);

  // Toggle ON again — idempotent: should produce the same Filtered count.
  await filteredToggle.click();
  await page.waitForTimeout(200);
  const onAgainFiltered = parseCount(await readLegendText(page), "Filtered");
  expect(onAgainFiltered, "toggle should be idempotent").toBe(onFiltered);
});

test("show-filtered toggle works after clicking Tune (the workaround)", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByText("Transform Drum Tracks into MIDI")).toBeVisible({
    timeout: 15_000,
  });

  const projectItem = page
    .locator(".project-item")
    .filter({ hasText: FIXTURE_PROJECT_NAME })
    .first();
  await expect(projectItem).toBeVisible({ timeout: 10_000 });
  await projectItem.click();

  const analysisSection = page.locator("#analysis-section");
  await expect(analysisSection).toBeVisible({ timeout: 20_000 });

  const snareTab = page.locator('.waveform-stem-tab[data-stem="snare"]').first();
  await expect(snareTab).toBeVisible({ timeout: 15_000 });
  await snareTab.click();
  await expect(snareTab).toHaveClass(/waveform-tab-active/);

  const eventsCanvas = page.locator("#events-canvas");
  await eventsCanvas.scrollIntoViewIfNeeded();
  await page.waitForTimeout(200);

  // === Click Tune to enter tuning mode (the user's workaround) ===
  const tuneButton = page.locator("#tuning-toggle-btn");
  if (await tuneButton.isVisible().catch(() => false)) {
    await tuneButton.click();
    await page.waitForTimeout(300);
  } else {
    // If button is hidden, use the same eval-path the spec 01 uses.
    await page.evaluate(() => toggleTuningPanel());
    await page.waitForTimeout(300);
  }
  const tuningPanel = page.locator("#tuning-panel");
  await expect(tuningPanel).toBeVisible({ timeout: 5_000 });

  const filteredToggle = page.locator("#waveform-filtered-toggle");
  await expect(filteredToggle).toBeVisible({ timeout: 5_000 });

  // Make sure we start in the OFF state.
  if (await filteredToggle.isChecked()) {
    await filteredToggle.click();
    await page.waitForTimeout(200);
  }

  const offLegend = await readLegendText(page);
  const offFiltered = parseCount(offLegend, "Filtered");
  expect(offFiltered, "tuning mode: toggle OFF must not show Filtered legend entry").toBe(0);

  // Toggle ON — should reveal Filtered entry AND red bars on canvas.
  await filteredToggle.click();
  await page.waitForTimeout(200);
  const onLegend = await readLegendText(page);
  const onFiltered = parseCount(onLegend, "Filtered");
  await page.screenshot({
    path: path.join(SNAPSHOT_DIR, "03-tuning-mode-toggle-on.png"),
    fullPage: false,
  });
  expect(
    onFiltered,
    `tuning mode: toggle ON should reveal Filtered entry. Legend: "${onLegend}"`
  ).toBeGreaterThan(0);
});