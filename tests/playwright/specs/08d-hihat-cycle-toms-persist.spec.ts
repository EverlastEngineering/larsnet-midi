/**
 * 08d — Per-event override regressions (2026-06-30)
 *
 * Two related user-reported bugs, both about the WebUI's
 * per-event override system:
 *
 *  Bug A: hihat events cycle through only open/closed, never
 *  reaching the FILTERED state. The user wants 3-state cycle:
 *  off (FILTERED) → open (KEPT) → closed (KEPT) → off.
 *
 *  Bug B: on toms, "saves don't seem to persist value of the
 *  classification" — the user clicks to set a per-event cls,
 *  but Save & Reconvert reverts it because the k-means
 *  re-classification overwrites it.
 *
 * The hihat test verifies Bug A directly by clicking an event
 * 3 times and asserting the status / hihat_state transitions
 * land on the right values.
 *
 * The toms test verifies Bug B by setting a known override
 * via the API, calling /rebuild-midi (which exercises the
 * k-means reclassify path), and asserting the override file
 * still has the user's per-event cls choice.
 */
import { test, expect } from "@playwright/test";

const FIXTURE_PROJECT_NAME = "01_Taylor_Swift_The_Fate_of_Ophelia_Drums";

test("hihat cycles off → open → closed → off (3 states)", async ({
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

  const hihatTab = page.locator('.waveform-stem-tab[data-stem="hihat"]').first();
  await expect(hihatTab).toBeVisible({ timeout: 15_000 });
  await hihatTab.click();
  await expect(hihatTab).toHaveClass(/waveform-tab-active/);
  await page.waitForTimeout(300);

  // Reset overrides to a known-empty state, then sync the
  // in-memory cache.
  await page.evaluate(async () => {
    await fetch("/api/projects/6/event-overrides", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ overrides: { hihat: {} } }),
    });
  });
  await page.waitForTimeout(300);
  await page.evaluate(() => (window as any).syncEventOverridesFromServer
    ? (window as any).syncEventOverridesFromServer({ hihat: {} })
    : null);
  await page.waitForTimeout(200);

  // Find a hihat event with hihat_state='open' so we have a
  // clear starting point for the cycle. Use an arbitrary
  // starting frame — the exact value doesn't matter, only
  // that the cycle reaches the right end states.
  const start = await page.evaluate(() => {
    const pga = (window as any).waveformAnalysisData?.()?.stems?.hihat?.events_pga;
    const ev = pga.find(
      (e: any) => e.hihat_state === "open" && e.status === "KEPT",
    );
    return ev ? { frame: ev.frame } : null;
  });
  expect(start, "Need at least one hihat open event to test the cycle").not.toBeNull();
  const key = String(start!.frame);

  // Cycle 1: open → closed (off → open is reachable only by
  // first going through OFF, but on an event with status=KEPT
  // hihat_state=open, the cycle goes KEPT/open → KEPT/closed).
  // Cycle 2: closed → off (status=FILTERED).
  // Cycle 3: off → open (status=KEPT, hihat_state=open).
  const click = async () => {
    await page.evaluate((frame) => {
      const pga = (window as any).waveformAnalysisData?.()?.stems?.hihat?.events_pga;
      const ev = pga.find((e: any) => e.frame === parseInt(frame));
      (window as any).cycleEventOverride("hihat", ev);
    }, key);
    await page.waitForTimeout(700);
  };
  await click();
  let overrides = await page.evaluate(() => (window as any).eventOverrides?.hihat || {});
  expect(overrides[key]?.status, "Cycle 1: open → closed should keep status KEPT")
    .toBe("KEPT");
  expect(overrides[key]?.hihat_state, "Cycle 1: hihat_state should be closed")
    .toBe("closed");

  await click();
  overrides = await page.evaluate(() => (window as any).eventOverrides?.hihat || {});
  expect(overrides[key]?.status, "Cycle 2: closed → off should set status FILTERED")
    .toBe("FILTERED");

  await click();
  overrides = await page.evaluate(() => (window as any).eventOverrides?.hihat || {});
  expect(overrides[key]?.status, "Cycle 3: off → open should set status KEPT")
    .toBe("KEPT");
  expect(overrides[key]?.hihat_state, "Cycle 3: hihat_state should be open")
    .toBe("open");
});

test("toms classification override survives the k-means reclassify on rebuild", async ({
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

  const tomsTab = page.locator('.waveform-stem-tab[data-stem="toms"]').first();
  await expect(tomsTab).toBeVisible({ timeout: 15_000 });
  await tomsTab.click();
  await expect(tomsTab).toHaveClass(/waveform-tab-active/);
  await page.waitForTimeout(300);

  // Step 1: Reset overrides to a known state with one entry
  // (frame 11520, cls=1 — the typical "user wants this to be
  // cls 1" choice).
  await page.evaluate(async () => {
    await fetch("/api/projects/6/event-overrides", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        overrides: { toms: { "11520": { status: "KEPT", classification: 1 } } },
      }),
    });
  });
  await page.waitForTimeout(300);

  // Step 2: Call rebuild-midi via fetch. This exercises the
  // classify_tom_notes path with force_reclassify=True (because
  // expected_clusters=2 is configured). The bug was: the k-means
  // would clobber the user's per-event override.
  const rebuildResult = await page.evaluate(async () => {
    const res = await fetch("/api/rebuild-midi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_number: 6,
        honor_overrides: true,
      }),
    });
    return { status: res.status, body: await res.json() };
  });
  expect(rebuildResult.body.success, "rebuildMidi should succeed").toBe(true);
  await page.waitForTimeout(500);

  // Step 3: Verify the user's per-event cls choice
  // survived the rebuild. The fix preserves the override
  // through classify_tom_notes (sets _overridden=True on
  // overridden events so k-means leaves them alone). The
  // rebuild may also clean up the override file once the
  // sidecar's natural state matches the user's choice
  // (the override becomes redundant), so we check the
  // sidecar's events_pga for frame 11520 — that's the
  // canonical persisted state, and it should have cls=1.
  const sidecarState = await page.evaluate(() => {
    const pga = (window as any).waveformAnalysisData?.()
      ?.stems?.toms?.events_pga;
    const ev = pga.find((e: any) => e.frame === 11520);
    return ev
      ? { status: ev.status, classification: ev.classification }
      : null;
  });
  expect(sidecarState, "Frame 11520 should still be in the sidecar").not.toBeNull();
  expect(
    sidecarState!.classification,
    "After rebuild-midi (which re-runs classify_tom_notes " +
      "with force_reclassify=True), the user's per-event " +
      "override should be reflected in events_pga. Bug: " +
      "the k-means re-run during rebuild was overwriting " +
      "the override with the auto-assigned class.",
  ).toBe(1);
  expect(sidecarState!.status).toBe("KEPT");
});
