/**
 * 08b — Save button UX regression (2026-06-30 followup)
 *
 * User followup after the original event-override fix: "the
 * save button appears but goes away in about 1/2 a second."
 *
 * The bug: the Save button's visibility was driven by
 * `eventOverridesDirty`, which the debounced save (500ms) cleared
 * right after the click. The button appeared → disappeared in
 * ~500ms, even though the user hadn't clicked Save yet.
 *
 * The fix: split the dirty flag into two:
 *   - `eventOverridesDirty`: in-memory ≠ JSON (cleared by
 *     debounced save; used internally for the debounce trigger).
 *   - `sessionOverridesDirty`: user has unsaved changes waiting
 *     for Save & Reconvert (cleared only by the sync from the
 *     rebuild response). The Save button checks this.
 *
 * This test verifies the regression fix:
 *   1. Click event → Save button visible.
 *   2. Wait 1 second (twice the debounce) → button STILL visible
 *      AND sessionOverridesDirty is still true.
 *      eventOverridesDirty is false (cleared by debounce).
 */
import { test, expect } from "@playwright/test";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SNAPSHOT_DIR = path.join(
  __dirname,
  "..",
  "__snapshots__",
  "08b-save-button-stays-visible",
);
const FIXTURE_PROJECT_NAME = "01_Taylor_Swift_The_Fate_of_Ophelia_Drums";

test.beforeAll(() => {
  fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
});

test("Save button stays visible 1s after the cycle click (UX regression)", async ({
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
  await page.waitForTimeout(300);

  // Reset overrides to a clean slate.
  await page.evaluate(async () => {
    const w = (window as any);
    const projectNumber = w.currentProject?.number;
    await fetch(`/api/projects/${projectNumber}/event-overrides`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ overrides: {} }),
    });
    if (typeof w.syncEventOverridesFromServer === "function") {
      w.syncEventOverridesFromServer({});
    }
  });

  // Click an event on snare to set the override.
  await page.evaluate(() => {
    const w = (window as any);
    const pga = w.waveformAnalysisData?.()?.stems?.snare?.events_pga || [];
    const ev = pga
      .filter((e: any) => e.status === "FILTERED")
      .sort((a: any, b: any) => a.frame - b.frame)
      .find((e: any) => e.frame > 10000);
    if (!ev) throw new Error("no filtered event found");
    if (typeof w.cycleEventOverride === "function") {
      w.cycleEventOverride("snare", ev);
    } else {
      throw new Error("cycleEventOverride not exposed on window");
    }
  });

  // Phase 1: Save button visible immediately after the click.
  const saveBtn = page.locator("#session-save-btn");
  await expect(saveBtn).toBeVisible({ timeout: 2_000 });

  // Phase 2: After 1 second (twice the 500ms debounce), the button
  // is STILL visible. This is the regression for the user's
  // "save button appears but goes away in 1/2 a second" bug.
  await page.waitForTimeout(1_000);
  const afterDebounce = await page.evaluate(() => {
    const w = (window as any);
    return {
      eventOverridesDirty:
        typeof w.eventOverridesDirty === "function"
          ? w.eventOverridesDirty()
          : null,
      sessionOverridesDirty:
        typeof w.sessionOverridesDirty === "function"
          ? w.sessionOverridesDirty()
          : null,
      saveBtnVisible: !document
        .getElementById("session-save-btn")
        ?.classList.contains("hidden"),
    };
  });

  // Internal dirty (in-memory ≠ JSON) cleared by the debounce:
  //   should be FALSE.
  expect(
    afterDebounce.eventOverridesDirty,
    "eventOverridesDirty should be FALSE 1s after the click " +
    "(the debounced save fires at 500ms)",
  ).toBe(false);

  // Session dirty (user has unsaved changes for Save & Reconvert):
  //   should STILL be TRUE.
  expect(
    afterDebounce.sessionOverridesDirty,
    "sessionOverridesDirty should STILL be TRUE 1s after the click",
  ).toBe(true);

  // Button visible: the user-facing behavior. This is the test
  // signal that would have caught the user's bug.
  expect(
    afterDebounce.saveBtnVisible,
    "Save button should STILL be visible 1s after the click. " +
    "Bug: the button disappeared when the debounced save fired. " +
    "Fix: drive the Save button from sessionOverridesDirty " +
    "(cleared only by Save & Reconvert), not eventOverridesDirty " +
    "(cleared by the debounce).",
  ).toBe(true);

  await page.screenshot({
    path: path.join(SNAPSHOT_DIR, "01-button-still-visible.png"),
    fullPage: false,
  });
});
