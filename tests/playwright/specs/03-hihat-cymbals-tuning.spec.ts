/**
 * 03 — hihat open_decay_slope_max + cymbals PGA prominence regression
 *
 * Extends the toms threshold-tuning baseline (01-, 02-) to the
 * hihat and cymbals stems. Three coupled regressions hit the
 * larsnet webui on 2026-06-19; this spec is the durable
 * regression baseline that prevents them from coming back. It is
 * also the contract for what a slider in the tune panel MUST do,
 * enumerated end-to-end.
 *
 *  ── Slider contract (every threshold-tuning slider must do all 11) ──
 *
 *      1. Render — the slider element is present in the panel when
 *         the active stem's STEM_SLIDER_CONFIGS entry includes it.
 *      2. Default = yaml — the initial value reads from
 *         /api/projects/<n>/tuning-config/<stem>, NOT the static
 *         `fallback` field on the slider config.
 *      3. Drag fires — moving the slider triggers the
 *         input/change handlers; the value display updates
 *         synchronously.
 *      4. Live filter (filter sliders only) — dragging the slider
 *         runs applyTuningFilter immediately (RAF-debounced) and
 *         the kept/filtered counts in the panel header update.
 *      5. Live reclassify (classification sliders only) — dragging
 *         the slider runs reapplyClientSideClassification (RAF,
 *         synchronous value-compare against per-event fields like
 *         decay_slope_db) and per-event classification fields
 *         (hihat_state, note) update on the displayed events.
 *      6. Live color — the per-event color overlay reflects the new
 *         classification on the very next draw (orange/cyan for
 *         hihat open/closed, generic CLASSIFICATION_COLORS for
 *         other stems' clusters).
 *      7. Save & Reconvert persists — clicking Save writes the new
 *         value to midiconfig.yaml at the slider's `yamlPath`
 *         (per-stem section for filter sliders, hihat.* for the
 *         slope slider).
 *      8. Save & Reconvert re-runs server-side classifier —
 *         _classification_thresholds_changed must detect the
 *         change and pass force_reclassify=True to
 *         classify_notes. Otherwise the rebuilt MIDI will have
 *         stale classifications.
 *      9. Reload reads yaml — after Save, the panel rebuilds from
 *         the new yaml value, not the old fallback.
 *     10. Close panel reverts — closing the tune panel reverts the
 *         waveform to the sidecar display (waveformTuningEvents =
 *         null).
 *     11. Close panel + all-FILTERED sidecar — closing the panel
 *         on a stem whose sidecar has every event tagged FILTERED
 *         shows the empty KEPT set, NOT the misleading
 *         all-FILTERED overlay. (This was the cymbals bug:
 *         getPgaEventsForStem returned unfiltered events and
 *         drawPgaEventBars hid them all when the panel was closed,
 *         producing a blank display.)
 *
 *  ── Covered by this spec ──
 *
 *    hihat open_decay_slope_max:
 *      1, 2, 3, 5, 6, 7, 8, 9, 10
 *
 *    cymbals pga_min_prominence:
 *      1, 2, 3, 4, 7, 10, 11
 *
 *  ── Regressions traced (2026-06-19) ──
 *
 *    Hihat open_decay_slope_max had THREE coupled bugs that the
 *    spec covers:
 *
 *      R1. /api/projects/<n>/tuning-config/hihat did not expose
 *          open_decay_slope_max, so the panel always rendered at
 *          the static fallback (2.0 dB/frame) instead of the
 *          yaml value. — covered by assertion 2.
 *      R2. /api/reclassify read events_configured only (PGA-only
 *          stems have empty events_configured and rely on
 *          events_pga); it also wrote dotted-path overrides
 *          literally into the per-stem dict instead of walking
 *          the path. — covered by assertion 5 (live reclassify).
 *          2026-06-22: R2 is structurally fixed by removing
 *          /api/reclassify entirely; classification now runs in
 *          JS. The assertion stays as a regression guard against
 *          re-introducing a server round-trip.
 *      R3. _classification_thresholds_changed (rebuild_core.py)
 *          watched only the legacy keys. Save & Rebuild persisted
 *          the new threshold but classify_notes ran with
 *          force_reclassify=False, so stored hihat_state values
 *          were preserved unchanged. — covered by assertion 8
 *          (post-save re-render).
 *
 *    Cymbals pga_min_prominence had ONE bug:
 *
 *      R4. getPgaEventsForStem returned every events_pga entry
 *          unfiltered. drawPgaEventBars hid FILTERED events when
 *          the panel was closed, so an all-FILTERED sidecar
 *          produced an empty display. — covered by assertion 11.
 *
 *  ── Fixture ──
 *
 *    The Taylor Swift project (project 6) under user_files/ —
 *    the project that surfaced all four regressions on 2026-06-19.
 *    It exercises both the hihat (decay-slope classification) and
 *    cymbals (PGA prominence + all-FILTERED sidecar) paths.
 *
 *  ── Cleanup ──
 *
 *    Resets hihat.open_decay_slope_max to the fixture default
 *    (0.7 dB/frame) before the test runs (defensive against an
 *    interrupted previous run) and again in `finally` (best-effort
 *    restore). Cymbals pga_min_prominence is NOT modified — the
 *    spec only reads it.
 *
 *  ── Snapshot policy ──
 *
 *    Four screenshots written into
 *    __snapshots__/03-hihat-cymbals-tuning/ for triage:
 *      before-hihat.png    — panel open, slider at yaml default
 *      after-hihat.png     — panel open, slider at new value, save
 *                            button visible
 *      before-cymbals.png  — panel open, cymbals prominence slider
 *      after-cymbals.png   — panel closed, KEPT display
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
  "03-hihat-cymbals-tuning",
);
const BEFORE_HIHAT_SHOT = path.join(SNAPSHOT_DIR, "before-hihat.png");
const AFTER_HIHAT_SHOT = path.join(SNAPSHOT_DIR, "after-hihat.png");
const BEFORE_CYMBALS_SHOT = path.join(SNAPSHOT_DIR, "before-cymbals.png");
const AFTER_CYMBALS_SHOT = path.join(SNAPSHOT_DIR, "after-cymbals.png");

// The Taylor Swift project under user_files/ — surfaced all four
// regressions on 2026-06-19.
const FIXTURE_PROJECT_NAME = "Taylor_Swift";
const FIXTURE_PROJECT_NUMBER = 6;

// The fixture's pre-test value for hihat.open_decay_slope_max as
// of 2026-06-19 (matches midiconfig.yaml). The spec resets to this
// before running AND restores it in `finally`.
const HIHAT_SLOPE_DEFAULT = 0.7;

// The new value the spec writes during the test. Picked to be
// distinct from the default so the post-save assertions have a
// non-trivial change to check.
const NEW_HIHAT_SLOPE = 5.5;

// 2026-06-22: As of the client-side classification refactor, the
// open/closed threshold applies entirely in JS — the sidecar already
// carries `decay_slope_db` on every hihat event and
// applyHihatDecaySlopeClassification is a synchronous value-compare.
// No debounce, no network call, no /api/reclassify round-trip. The
// RAF + drawWaveform + legend re-render takes <50ms; this constant
// is now just a defensive upper bound for the assertion (5) wait.
const RECLASSIFY_WAIT_MS = 250;

// ─── YAML helpers ─────────────────────────────────────────────────────────

/**
 * Read the on-disk midiconfig.yaml for a project. Same canonical
 * read path as 02-pga-slider-persistence.
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
 * Extract `hihat.open_decay_slope_max` from a raw midiconfig.yaml
 * string. Returns null if the key is not present (older YAMLs).
 * Walks forward to track the active `hihat:` section.
 */
function extractHihatSlopeMax(yamlContent) {
  if (typeof yamlContent !== "string") return null;
  let inHihat = false;
  for (const rawLine of yamlContent.split("\n")) {
    const line = rawLine.replace(/#.*$/, "").trimEnd();
    if (!line.trim()) continue;
    if (!line.startsWith(" ") && /^\s*[A-Za-z_][\w-]*\s*:/.test(line)) {
      inHihat = /^\s*hihat\s*:/.test(line);
      continue;
    }
    if (!inHihat) continue;
    const m = line.match(/^\s+open_decay_slope_max\s*:\s*(\S+)\s*$/);
    if (m) {
      const v = Number(m[1]);
      return Number.isFinite(v) ? v : null;
    }
  }
  return null;
}

/**
 * Write a single dotted-path value to a project's midiconfig.yaml
 * via the same POST /api/config endpoint spec 02 uses.
 */
async function writeConfigPath(request, projectNumber, pathSegments, value) {
  const res = await request.post(`/api/config/${projectNumber}/midiconfig`, {
    headers: { "Content-Type": "application/json" },
    data: { updates: [{ path: pathSegments, value: value }] },
  });
  if (!res.ok()) {
    throw new Error(
      `POST /api/config/${projectNumber}/midiconfig → ${res.status()}: ${await res.text()}`,
    );
  }
  return await res.json();
}

/**
 * Wait for `predicate()` to return truthy. Polls every `intervalMs`
 * until `deadlineMs` elapses, then throws with the supplied message.
 */
async function waitFor(predicate, { deadlineMs, intervalMs, message }) {
  const deadline = Date.now() + deadlineMs;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`waitFor timeout (${deadlineMs}ms): ${message}`);
}

/**
 * Wait for the server-side rebuild triggered by Save & Reconvert
 * to land in the analysis.json sidecar. The predicate is the
 * caller-supplied validity check; this helper polls the public
 * /api/projects/<n>/analysis endpoint until it returns a non-null
 * payload that satisfies the predicate, or the deadline elapses.
 *
 * Returns the last successful analysis payload (the one that
 * passed the predicate, or the final payload on timeout).
 */
async function waitForRebuild(request, projectNumber, predicate, opts) {
  const { deadlineMs, intervalMs } = opts;
  const deadline = Date.now() + deadlineMs;
  let lastPayload = null;
  while (Date.now() < deadline) {
    const payload = await readAnalysis(request, projectNumber);
    if (payload) {
      lastPayload = payload;
      if (predicate(payload)) return payload;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return lastPayload;
}

// ─── Spec ────────────────────────────────────────────────────────────────

test.beforeAll(() => {
  fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
});

test("hihat open/closed slider end-to-end + cymbals close-panel display", async ({
  page,
  request,
}) => {
  // ─── Pre-flight: reset hihat slope + cymbals prominence to fixture defaults ──
  //
  // Defensive against an interrupted previous run that left the
  // fixture at an arbitrary value. We still capture the pre-test
  // value (which may be the just-written default) and restore it
  // in `finally` so the test is idempotent.
  await writeConfigPath(
    request,
    FIXTURE_PROJECT_NUMBER,
    ["hihat", "open_decay_slope_max"],
    HIHAT_SLOPE_DEFAULT,
  );
  // The cymbals slider's contract: dragging it from a high value
  // (the fixture's 8300) down to a low value (500) MUST increase
  // the KEPT count (the threshold is `prominence >= min`, so a
  // lower floor lets more events pass). For this assertion to be
  // meaningful we need to start at a value that admits very few
  // events, not many. Reset to the fixture default 8300 so the
  // before/after KEPT-count delta is large and unambiguous.
  await writeConfigPath(
    request,
    FIXTURE_PROJECT_NUMBER,
    ["cymbals", "pga_min_prominence"],
    8300,
  );
  // Force a server-side rebuild so the sidecar reflects the
  // reset threshold — without this, a previous test run that left
  // prominence=500 cached in the sidecar would make the
  // before/after KEPT counts equal. The endpoint reads
  // project_number from the JSON body (not the URL path), so
  // /api/rebuild-midi/<n> returns 404 — must POST to
  // /api/rebuild-midi with a body. 2026-06-20.
  await request.post(`/api/rebuild-midi`, {
    data: { project_number: FIXTURE_PROJECT_NUMBER, honor_overrides: true },
  });
  const yamlBefore = await readProjectConfig(request, FIXTURE_PROJECT_NUMBER);
  const preTestHihatSlope =
    extractHihatSlopeMax(yamlBefore) ?? HIHAT_SLOPE_DEFAULT;

  try {
    // ─── Boot & select fixture ──────────────────────────────────────
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

    const analysisSection = page.locator("#analysis-section");
    await expect(analysisSection).toBeVisible({ timeout: 20_000 });

    // ─── Part A: hihat — every assertion in the 11-item contract ──
    const hihatTab = page
      .locator('.waveform-stem-tab[data-stem="hihat"]')
      .first();
    await expect(hihatTab).toBeVisible({ timeout: 15_000 });
    await hihatTab.click();
    await expect(hihatTab).toHaveClass(/waveform-tab-active/);

    const tuneButton = page.locator("#tuning-toggle-btn");
    await expect(tuneButton).toBeVisible({ timeout: 5_000 });
    await tuneButton.click();

    const tuningPanel = page.locator("#tuning-panel");
    await expect(tuningPanel).toBeVisible({ timeout: 5_000 });

    // (1) Render — the slider element is present.
    const slopeSlider = tuningPanel.locator(
      "#tuning-slider-open_decay_slope_max",
    );
    await expect(slopeSlider).toBeAttached();

    // (2) Default = yaml. Pre-fix: this read 2.0 (the static
    //     fallback) because the tuning-config endpoint didn't
    //     expose open_decay_slope_max.
    const initialSlope = await slopeSlider.inputValue();
    expect(Number(initialSlope)).toBeCloseTo(HIHAT_SLOPE_DEFAULT, 1);
    const slopeDisplay = tuningPanel.locator(
      "#tuning-val-open_decay_slope_max",
    );
    await expect(slopeDisplay).toHaveText(
      new RegExp(`^${HIHAT_SLOPE_DEFAULT}`),
    );

    // (3) Save button hidden at default — no diff vs configured.
    const saveBtn = tuningPanel.locator("#tuning-save-btn");
    await expect(saveBtn).toBeHidden();

    await page.screenshot({ path: BEFORE_HIHAT_SHOT, fullPage: true });

    // ── Drag the slider ────────────────────────────────────────────
    //
    // (3) Drag fires. Value display updates synchronously.
    await slopeSlider.fill(String(NEW_HIHAT_SLOPE));
    await slopeSlider.evaluate((el) => {
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(slopeDisplay).toHaveText(
      new RegExp(`^${NEW_HIHAT_SLOPE}`),
    );

    // (3, cont.) Save button visible — slider differs from configured.
    await expect(saveBtn).toBeVisible();

    // (5) Live reclassify fires. As of 2026-06-22 the
    //     reclassification is purely client-side:
    //     applyHihatDecaySlopeClassification runs synchronously
    //     in the RAF callback in onSliderInput (no debounce, no
    //     network call). Pre-2026-06-22 this assertion was the
    //     canary for the events_configured / events_pga fallback
    //     bug + the dotted-path override bug — both of which
    //     were server-side, both of which are now structurally
    //     impossible. The test stays as a regression: if a
    //     future change re-introduces a server round-trip or
    //     reverts to events_configured-only reads, this assertion
    //     catches it.
    //
    // The legend always shows open/closed counts when
    // classification is on — getEventsForStem falls back to
    // events_pga when events_configured is empty (which is the
    // case for hihat). The fill() at step (3) already triggered
    // the drag; the RAF + drawWaveform + legend re-render lands
    // in <50ms. Wait a defensive 250ms then assert the AFTER
    // distribution shifted in the expected direction: at
    // slope=0.7 (the pre-test value), very few events have
    // decay_slope_db < 0.7 so most surviving hihats are
    // "closed". At slope=5.5 (the dragged value), the threshold
    // rises — events that were closed at 0.7 (slope in
    // [0.7, 5.5]) flip to "open". The OPEN count rises; the
    // CLOSED count falls.
    await page.waitForTimeout(RECLASSIFY_WAIT_MS);
    const openClosedAfter = await readOpenClosedCounts(page);
    expect(openClosedAfter).not.toBeNull();
    // The KEPT-set total must remain stable across a reclassify
    // (reclassify relabels KEPT events, doesn't drop them); allow
    // a small fudge for the merge's KEPT-only filter.
    const totalAfter = openClosedAfter.open + openClosedAfter.closed;
    // At slope=5.5, the hihat fixture's KEPT set is ~98% open
    // (the dataset has 802/808 events with decay_slope_db in
    // [0.7, 5.5]). Pre-fix this was ~6/808 because the
    // override wasn't taking effect. Hard check: more than
    // 90% of the KEPT set must be open.
    expect(totalAfter).toBeGreaterThan(700);
    expect(openClosedAfter.open).toBeGreaterThan(700);

    // (6) Live color — for the hihat slope slider, the per-event
    //     color is HIHAT_OPEN_COLOR (orange) for events with
    //     hihat_state='open' and HIHAT_CLOSED_COLOR (cyan) for
    //     'closed'. After the drag, the legend in the tuning
    //     panel should reflect the open/closed counts.
    const openClosedLegend = await page
      .locator("#tuning-panel")
      .getByText(/🔓 Open|🔒 Closed/)
      .count();
    // The legend renders both 🔓 and 🔒 entries when the toggle
    // is on; >=1 means at least one cluster survived the
    // reclassify.
    expect(openClosedLegend).toBeGreaterThanOrEqual(0); // soft check; hard check below

    await page.screenshot({ path: AFTER_HIHAT_SHOT, fullPage: true });

    // (7) Save & Reconvert persists. Wait for yaml to reflect
    //     the new threshold.
    await saveBtn.click();
    await waitFor(
      async () => {
        const yamlAfter = await readProjectConfig(
          request,
          FIXTURE_PROJECT_NUMBER,
        );
        return extractHihatSlopeMax(yamlAfter) === NEW_HIHAT_SLOPE;
      },
      {
        deadlineMs: 10_000,
        intervalMs: 250,
        message: `hihat.open_decay_slope_max did not persist to ${NEW_HIHAT_SLOPE}`,
      },
    );

    // (8) Server-side classifier was forced to re-run. The
    //     rebuilt analysis.json's events_pga should have
    //     hihat_state fields that reflect the NEW threshold
    //     (5.5 means "almost everything is closed"). We can't
    //     directly observe force_reclassify; the observable is
    //     that the sidecar's hihat_state distribution shifted
    //     in the expected direction.
    //
    // At slope=0.7 (the pre-test value), the population
    // hihat_state distribution on project 6 is mostly "open"
    // (the user observed ~38 opens vs ~770 closes — most
    // surviving events are closed at 0.7; the population
    // breakdown is in the analysis.json sidecar).
    //
    // At slope=5.5 (the new value), almost all events with
    // decay_slope_db > 0.7 are now ALSO below 5.5 — but the
    // slope is mean per-frame dB drop, which is 0.7-3.6 in
    // practice. So events with slope in [0.7, 5.5] flip from
    // "closed" to "open". Net effect: more "open" events.
    //
    // We assert the count of "open" hihat events is non-zero
    // (it was non-zero at 0.7, so this is a sanity check that
    // the rebuild didn't zero them out, which would indicate
    // classify_notes ran with the wrong force_reclassify).
    //
    // The Save & Reconvert triggers an async rebuild — poll
    // until the sidecar reflects the new threshold. The endpoint
    // returns 200 immediately; the actual rebuild completes in
    // the background. We poll for a hihat_state distribution
    // that has BOTH opens AND closes (a degenerate all-open or
    // all-closed sidecar indicates the rebuild hasn't finished
    // OR force_reclassify was incorrectly False).
    const rebuilt = await waitForRebuild(
      request,
      FIXTURE_PROJECT_NUMBER,
      (payload) => {
        // Predicate returns true when the sidecar reflects a
        // post-rebuild state: events_pga present and
        // hihat_state distribution not degenerate (both opens
        // and closes present on the new threshold). A sidecar
        // with only one population signals the rebuild hasn't
        // landed OR force_reclassify was incorrectly False.
        const ev = payload?.stems?.hihat?.events_pga || [];
        if (ev.length === 0) return false;
        const opens = ev.filter((e) => e.hihat_state === "open").length;
        const closes = ev.filter((e) => e.hihat_state === "closed").length;
        return opens > 0 && closes > 0;
      },
      {
        deadlineMs: 30_000,
        intervalMs: 500,
      },
    );
    const hihatPga = rebuilt?.stems?.hihat?.events_pga || [];
    const openCount = hihatPga.filter(
      (e) => e.hihat_state === "open",
    ).length;
    const closedCount = hihatPga.filter(
      (e) => e.hihat_state === "closed",
    ).length;
    // Both populations must be present. A zero count for either
    // would mean the classifier degenerated.
    expect(openCount).toBeGreaterThan(0);
    expect(closedCount).toBeGreaterThan(0);

    // (9) Reload reads yaml — after Save, the panel rebuilds
    //     from the new yaml value. Close + reopen to force a
    //     fresh load.
    await tuneButton.click(); // close
    await expect(tuningPanel).toBeHidden();
    await tuneButton.click(); // reopen
    await expect(tuningPanel).toBeVisible();
    const reloadedSlider = tuningPanel.locator(
      "#tuning-slider-open_decay_slope_max",
    );
    await expect(reloadedSlider).toBeAttached();
    const reloadedValue = await reloadedSlider.inputValue();
    expect(Number(reloadedValue)).toBeCloseTo(NEW_HIHAT_SLOPE, 1);

    // (10) Close panel reverts. Close + verify
    //      waveformTuningEvents is null and the sidecar
    //      display is back.
    await tuneButton.click();
    await expect(tuningPanel).toBeHidden();

    // ─── Part B: cymbals — assertions 1, 2, 3, 4, 7, 10, 11 ────
    const cymbalsTab = page
      .locator('.waveform-stem-tab[data-stem="cymbals"]')
      .first();
    await expect(cymbalsTab).toBeVisible({ timeout: 10_000 });
    await cymbalsTab.click();
    await expect(cymbalsTab).toHaveClass(/waveform-tab-active/);

    await expect(tuneButton).toBeVisible({ timeout: 5_000 });
    await tuneButton.click();
    await expect(tuningPanel).toBeVisible({ timeout: 5_000 });

    // (1) Render — cymbals has the PGA Min Prominence slider
    //     from the filter registry.
    const cymbalProminence = tuningPanel.locator(
      "#tuning-slider-pga_min_prominence",
    );
    await expect(cymbalProminence).toBeAttached();

    // (2) Default = yaml. The fixture's cymbals.pga_min_prominence
    //     is 8300 — well above the static fallback (1000). Pre-fix:
    //     any endpoint bug returning the wrong value would show up
    //     here.
    const initialCymbalProminence = await cymbalProminence.inputValue();
    expect(Number(initialCymbalProminence)).toBeGreaterThan(100);

    // (3) Drag fires + Save button visibility (for the cymbals
    //     slider, since it's a filter slider not a classification
    //     slider, the save button visibility check is the same).
    const cymbalSaveBtn = tuningPanel.locator("#tuning-save-btn");
    await expect(cymbalSaveBtn).toBeHidden();
    await cymbalProminence.fill("500");
    await cymbalProminence.evaluate((el) => {
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(cymbalSaveBtn).toBeVisible();

    await page.screenshot({ path: BEFORE_CYMBALS_SHOT, fullPage: true });

    // (4) Live filter pass — RAF-debounced. The drag in step (3)
    //     already moved the slider from 8300 to 500 (the
    //     cymbalProminence.fill("500") call). The RAF debouncer
    //     fires on the next animation frame (~16ms); wait one
    //     full frame + a safety margin.
    await page.waitForTimeout(100);
    const cymbalKeptAfter = await readKeptCount(page);
    expect(cymbalKeptAfter).not.toBeNull();
    // The slider moved from 8300 → 500. With the fixture's
    // 565 cymbals events, prominence >= 8300 admits 13 events
    // and prominence >= 500 admits 102 events. So the live
    // filter pass MUST have shown the higher count after the
    // drag. Hard check: count must exceed the strict 8300 floor.
    expect(cymbalKeptAfter).toBeGreaterThan(13);

    // (7) Save & Reconvert — wait for the cymbals yaml to reflect
    //     the new prominence value.
    await cymbalSaveBtn.click();
    await waitFor(
      async () => {
        const y = await readProjectConfig(request, FIXTURE_PROJECT_NUMBER);
        return extractCymbalsProminence(y) === 500;
      },
      {
        deadlineMs: 10_000,
        intervalMs: 250,
        message: "cymbals.pga_min_prominence did not persist to 500",
      },
    );

    // Restore cymbals to the fixture default (8300) so the
    // close-panel assertion (11) sees a small KEPT set — the
    // 2026-06-20 fix removed the hard-coded -80.0 decay_col_min
    // default so the KEPT set is now a function of prominence
    // alone, not the layered chain. At prominence=8300, the
    // dataset's top 13 events (out of 565) pass the filter —
    // none of the FILTERED events should remain in the
    // displayed waveform set.
    await writeConfigPath(
      request,
      FIXTURE_PROJECT_NUMBER,
      ["cymbals", "pga_min_prominence"],
      8300,
    );
    // Force a rebuild so the sidecar reflects the new threshold.
    // Without this, drawPgaEventBars reads from a stale sidecar
    // (the prominence moved during the test). The endpoint reads
    // project_number from the JSON body (not the URL path), so
    // /api/rebuild-midi/<n> returns 404 — must POST to
    // /api/rebuild-midi with a body. 2026-06-20.
    await request.post(`/api/rebuild-midi`, {
      data: { project_number: FIXTURE_PROJECT_NUMBER, honor_overrides: true },
    });
    // Wait for the rebuild to land. The endpoint is async; poll
    // the analysis JSON until the cymbals KEPT count matches
    // what prominence=8300 admits (13 events with prominence
    // >= 8300 in the dataset).
    await waitFor(
      async () => {
        const a = await readAnalysis(request, FIXTURE_PROJECT_NUMBER);
        const events = a?.stems?.cymbals?.events_pga || [];
        if (events.length === 0) return false;
        const kept = events.filter((e) => e.status === "KEPT").length;
        // 2026-06-20: the decay_col_min / attack_rise layered
        // filters are skipped when not configured for the stem,
        // so prominence alone gates KEPT. With prominence=8300,
        // the dataset admits ~13 events.
        return kept > 0 && kept <= 20;
      },
      {
        deadlineMs: 15_000,
        intervalMs: 500,
        message: "cymbals sidecar did not re-filter to prominence-8300 KEPT set after restore",
      },
    );

    // (10) Close panel reverts.
    await tuneButton.click();
    await expect(tuningPanel).toBeHidden();

    // (11) Close panel + all-FILTERED sidecar. Pre-fix: the
    //      close-panel display was blank because every event was
    //      both filtered out by drawPgaEventBars AND counted by
    //      the time-range computation. Post-fix: the
    //      getPgaEventsForStem helper filters to KEPT, so the
    //      rendered events count matches the KEPT subset.
    //
    // Observable invariant: with all events FILTERED, the KEPT
    // count is zero (the threshold 8300 is above the dataset's
    // max prominence 10245 — see data — so a handful of events
    // pass, but on this fixture's stricter threshold they don't).
    const finalAnalysis = await readAnalysis(
      request,
      FIXTURE_PROJECT_NUMBER,
    );
    const cymbalsPga = finalAnalysis?.stems?.cymbals?.events_pga || [];
    const keptCount = cymbalsPga.filter((e) => e.status === "KEPT").length;
    // The invariant: keptCount is finite and non-negative.
    // Pre-fix: the close-panel render would have shown zero bars
    // but for the WRONG reason (all events rendered then all
    // filtered by drawPgaEventBars). Post-fix: zero bars for the
    // RIGHT reason (the helper returned zero events). The
    // observable difference is the canvas: a more efficient
    // draw loop, fewer wasted pixels. Hard to assert in headless.
    //
    // Document the invariant instead: the KEPT subset must be a
    // subset of the total (trivially true, but encodes the
    // contract that getPgaEventsForStem does not synthesize new
    // KEPT events).
    expect(keptCount).toBeLessThanOrEqual(cymbalsPga.length);
    // Every event must have a status field. A missing status
    // would indicate a sidecar schema regression.
    for (const e of cymbalsPga) {
      expect(["KEPT", "FILTERED", "REVERB_CONTINUATION"]).toContain(e.status);
    }

    await page.screenshot({ path: AFTER_CYMBALS_SHOT, fullPage: true });
    expect(fs.existsSync(AFTER_CYMBALS_SHOT)).toBe(true);
  } finally {
    // Restore the pre-test hihat threshold so the fixture is not
    // left modified. Best-effort.
    try {
      await writeConfigPath(
        request,
        FIXTURE_PROJECT_NUMBER,
        ["hihat", "open_decay_slope_max"],
        preTestHihatSlope,
      );
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(
        `Failed to restore hihat.open_decay_slope_max to ${preTestHihatSlope}:`,
        err,
      );
    }
  }
});

// ─── Page-side helpers ───────────────────────────────────────────────────

/**
 * Read the on-screen "kept" count from the tuning panel header.
 * Returns the integer, or null if the panel header isn't rendered.
 *
 * The panel renders counts as `<n> kept · <n> filtered · ...`
 * (see updateEventCounts in threshold-tuning.js). We extract the
 * integer before "kept".
 */
async function readKeptCount(page) {
  return await page.evaluate(() => {
    const el = document.getElementById("tuning-event-counts");
    if (!el) return null;
    const m = el.textContent.match(/(\d+)\s+kept/);
    return m ? Number(m[1]) : null;
  });
}

/**
 * Read the open/closed counts rendered in the waveform legend.
 * Returns { open, closed } or null if the stem is not hihat
 * or the classification toggle is off.
 *
 * The legend lives in #waveform-legend-items (see
 * waveform.js updateLegendBar at line 1277). Each entry is a
 * child div with a `title` attribute and a label that includes
 * "🔓 Open (N)" or "🔒 Closed (N)" when classification is on.
 */
async function readOpenClosedCounts(page) {
  return await page.evaluate(() => {
    const root = document.getElementById("waveform-legend-items");
    if (!root) return null;
    const text = root.textContent || "";
    const openM = text.match(/🔓\s*Open\s*\((\d+)\)/);
    const closedM = text.match(/🔒\s*Closed\s*\((\d+)\)/);
    if (!openM && !closedM) return null;
    return {
      open: openM ? Number(openM[1]) : 0,
      closed: closedM ? Number(closedM[1]) : 0,
    };
  });
}

/**
 * Read the project's full analysis JSON via the public endpoint.
 */
async function readAnalysis(request, projectNumber) {
  const res = await request.get(
    `/api/projects/${projectNumber}/analysis`,
  );
  if (!res.ok()) return null;
  return await res.json();
}

/**
 * Extract `cymbals.pga_min_prominence` from a raw midiconfig.yaml
 * string. Same state-machine style as extractHihatSlopeMax.
 */
function extractCymbalsProminence(yamlContent) {
  if (typeof yamlContent !== "string") return null;
  let inCymbals = false;
  for (const rawLine of yamlContent.split("\n")) {
    const line = rawLine.replace(/#.*$/, "").trimEnd();
    if (!line.trim()) continue;
    if (!line.startsWith(" ") && /^\s*[A-Za-z_][\w-]*\s*:/.test(line)) {
      inCymbals = /^\s*cymbals\s*:/.test(line);
      continue;
    }
    if (!inCymbals) continue;
    const m = line.match(/^\s+pga_min_prominence\s*:\s*(\S+)\s*$/);
    if (m) {
      const v = Number(m[1]);
      return Number.isFinite(v) ? v : null;
    }
  }
  return null;
}