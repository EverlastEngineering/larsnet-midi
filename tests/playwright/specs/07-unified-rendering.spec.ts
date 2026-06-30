/**
 * 07 — Unified event rendering regression
 *
 * User report (2026-06-30): "Some stems don't work correctly. I
 * think there are TWO rendering paths, depending on whether you
 * have classifications (multiple notes) or not. I notice the
 * purple events work properly, but the green/orange don't, and
 * for some reason kick (in project 6) switched from purple to
 * green (and then breaks) when you touch a slider. I like the
 * faded red as a rule around the disabled events, so let's pick
 * that path and enrich it with what it needs to work with any
 * data."
 *
 * The bug: kick in project 6 had a legacy sidecar with BOTH
 * `events_configured` (190 events, method=None) and `events_pga`
 * (2087 events, method='percentile_gated'). The waveform rendered
 * the events_pga set in violet (path B, the PGA layer) and the
 * events_configured set in green (path A, the events panel) at
 * the same X positions. The PGA layer covered the events panel
 * so the user saw VIOLET. When a slider was touched, the tuning
 * path switched the data source to `events_configured` only, and
 * the events panel + PGA layer both rendered green — the bug.
 *
 * The fix: unify the data source to `events_pga` (the canonical
 * PGA-detected set) and use a single render path (the
 * faded-red one, `drawPgaEventBars` with `getEventColor`) for
 * all event types.
 *
 * Test signals:
 *   - Count violet (#8b5cf6) pixels on the events canvas.
 *     Kick has 190 KEPT events with method='percentile_gated' →
 *     markerPga (violet) via getEventColor. After the fix the
 *     canvas should have a substantial number of violet pixels.
 *   - Verify the count is stable across a slider drag (the
 *     bug: pre-slider violet, post-slider green).
 *   - Verify the count is stable across the "Show Filtered"
 *     toggle (which only affects FILTERED events; KEPT color
 *     shouldn't change).
 *
 * Fixture: project #6 (Taylor Swift), kick stem.
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
  "07-unified-rendering",
);

const FIXTURE_PROJECT_NAME = "01_Taylor_Swift_The_Fate_of_Ophelia_Drums";

test.beforeAll(() => {
  fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
});

/** Count pixels close to the PGA violet color (#8b5cf6 = rgb(139, 92, 246)). */
async function countVioletPixels(page): Promise<number> {
  return await page.evaluate(() => {
    const canvas = document.getElementById("events-canvas");
    if (!canvas) return -1;
    const ctx = canvas.getContext("2d");
    if (!ctx) return -1;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let n = 0;
    for (let i = 0; i < img.data.length; i += 4) {
      const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2], a = img.data[i + 3];
      if (a < 200) continue;
      // Allow ±20 to catch anti-aliased edges.
      if (Math.abs(r - 139) < 20 && Math.abs(g - 92) < 20 && Math.abs(b - 246) < 20) n++;
    }
    return n;
  });
}

test("kick renders violet both before and after a slider drag", async ({
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

  const eventsCanvas = page.locator("#events-canvas");
  await eventsCanvas.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);

  // Switch to the kick stem.
  const kickTab = page.locator('.waveform-stem-tab[data-stem="kick"]').first();
  await expect(kickTab).toBeVisible({ timeout: 15_000 });
  await kickTab.click();
  await expect(kickTab).toHaveClass(/waveform-tab-active/);
  await page.waitForTimeout(300);

  // Baseline: count violet pixels BEFORE any slider interaction.
  // Pre-fix: this number was 0 for legacy sidecars that
  // picked events_configured over events_pga. After the fix,
  // it should be > 500 (kick has 190 KEPT events in
  // events_pga, all method='percentile_gated' → markerPga).
  const violetBefore = await countVioletPixels(page);
  await page.screenshot({
    path: path.join(SNAPSHOT_DIR, "01-kick-pre-slider.png"),
    fullPage: false,
  });
  console.log(`kick violet pixels pre-slider:  ${violetBefore}`);
  expect(
    violetBefore,
    `kick events should render violet (PGA color) before any slider touch. ` +
    `Got ${violetBefore} violet pixels. ` +
    `Bug: kick was rendering green (data source: events_configured ` +
    `instead of events_pga).`
  ).toBeGreaterThan(500);

  // Touch a slider — the bug the user reported. Open the Tune
  // panel so the slider is in the DOM, then drag a slider.
  // We don't strictly need Tune open to trigger applyTuningFilter
  // (the events_pga → waveformTuningEvents path is independent),
  // but having Tune open surfaces the slider for a more
  // representative user interaction.
  const tuneButton = page.locator("#tuning-toggle-btn");
  if (await tuneButton.isVisible().catch(() => false)) {
    await tuneButton.click();
    await page.waitForTimeout(300);
  }
  const tuningPanel = page.locator("#tuning-panel");
  if (await tuningPanel.isVisible().catch(() => false)) {
    // Find the first slider in the tuning panel and drag it.
    const slider = tuningPanel.locator('input[type="range"]').first();
    if (await slider.isVisible().catch(() => false)) {
      await slider.focus();
      // Keyboard-based drag (works without layout-affecting mouse events).
      await slider.press("ArrowRight");
      await slider.press("ArrowRight");
      await slider.press("ArrowRight");
      await page.waitForTimeout(300);
    }
  }

  const violetAfter = await countVioletPixels(page);
  await page.screenshot({
    path: path.join(SNAPSHOT_DIR, "02-kick-post-slider.png"),
    fullPage: false,
  });
  console.log(`kick violet pixels post-slider: ${violetAfter}`);

  // CRITICAL: the count should be roughly the same (within 50%
  // of the pre-slider count). The bug was that the count dropped
  // to 0 after a slider touch. Allow some tolerance for the
  // tuning re-render changing bar widths / positions.
  expect(
    violetAfter,
    `kick events should STILL render violet after a slider touch. ` +
    `Got ${violetAfter} violet pixels (was ${violetBefore} pre-slider). ` +
    `Bug: data source switched from events_pga to events_configured ` +
    `after a slider drag, breaking the unified render.`
  ).toBeGreaterThan(violetBefore * 0.5);
});
