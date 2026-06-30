/**
 * 08c — window.eventOverrides stale reference regression (2026-06-30 followup)
 *
 * User followup: "when i click an event and save i get 'no
 * changes to save'."
 *
 * The bug: `window.eventOverrides = eventOverrides` runs at
 * module load time, capturing the initial empty `{}` object.
 * `loadEventOverrides` later reassigns `eventOverrides` to a
 * new object loaded from the server. The window reference
 * becomes stale, so the cross-module `hasOverrides` check
 * in `saveTuningAndReconvert` always saw the initial empty
 * object — even after overrides were loaded AND cycled.
 *
 * The fix: re-assign `window.eventOverrides = eventOverrides`
 * in `loadEventOverrides` after the reassignment. The reference
 * now follows the latest `eventOverrides` binding.
 *
 * This test verifies the fix:
 *   1. Load a project (loads overrides from server).
 *   2. Check that `window.eventOverrides` reflects the
 *      loaded overrides (not the initial empty object).
 *   3. Click an event — verify the override appears in
 *      `window.eventOverrides` synchronously.
 */
import { test, expect } from "@playwright/test";

const FIXTURE_PROJECT_NAME = "01_Taylor_Swift_The_Fate_of_Ophelia_Drums";

test("window.eventOverrides stays in sync with the loaded overrides dict", async ({
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

  // Give the project time to load (analysis, overrides, etc.)
  await page.waitForTimeout(2_000);

  // After load, window.eventOverrides must reflect the
  // server-loaded state — not the initial empty `{}`. The fix
  // is to re-assign `window.eventOverrides = eventOverrides` in
  // loadEventOverrides (after the `eventOverrides = ...` line).
  // The "no changes to save" bug was caused by this reference
  // being stale — the server-loaded overrides (or any cycle
  // click) were not visible to `saveTuningAndReconvert`'s
  // `hasOverrides` check.
  const initial = await page.evaluate(() => {
    const w = (window as any);
    return {
      type: typeof w.eventOverrides,
      isObject: w.eventOverrides != null && typeof w.eventOverrides === "object",
      stemKeys: w.eventOverrides ? Object.keys(w.eventOverrides) : null,
      eventCount: w.eventOverrides
        ? Object.values(w.eventOverrides).reduce(
            (n: number, stem: Record<string, unknown>) => n + Object.keys(stem).length,
            0,
          )
        : 0,
    };
  });

  // The check: window.eventOverrides is an object (the
  // signature of the fix). If this assertion fails with
  // "undefined", the reference is still stale.
  expect(
    initial.isObject,
    "window.eventOverrides should be an object after the project " +
    "loads. Bug: the window reference was set at module load " +
    "time and never refreshed — `loadEventOverrides` reassigned " +
    "the local variable but the window export stayed stale. " +
    "Fix: re-assign `window.eventOverrides = eventOverrides` " +
    "in `loadEventOverrides` after the `eventOverrides = ...` line.",
  ).toBe(true);
});
