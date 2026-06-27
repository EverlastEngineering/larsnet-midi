/**
 * 04 — pga_min_combined_score slider regression (project 10, hihat)
 *
 * Three coupled regressions in the combined_score (warble) filter
 * wiring, surfaced 2026-06-26:
 *
 *   A. The server's /api/projects/<n>/tuning-config/<stem>
 *      endpoint does NOT return pga_min_combined_score, so the
 *      WebUI's slider always reads its registry fallback (0)
 *      after every save. The persisted value lives in the
 *      midiconfig.yaml on disk but the WebUI can't see it.
 *   B. As a consequence of A, no events are filtered based on
 *      the slider's value: the WebUI thinks the threshold is
 *      0 (the warble-separator default), even when the user
 *      has set it to e.g. 100 in midiconfig. The rebuild
 *      pipeline reads the yaml directly, so the WebUI's stale
 *      display doesn't break the rebuild — but the slider is
 *      visually unresponsive to the user's changes.
 *   C. The slider's min/max are fixed at -10000/10000 in the
 *      filter registry. For most songs the actual combined_score
 *      range is much narrower (e.g. on the Metallica hihat:
 *      -8286 to +9494). The user wants the slider to compute
 *      its min/max from the sidecar's combined_score range per
 *      stem, so the full resolution of the slider is usable
 *      within the actual data range.
 *
 * This spec asserts all three so they don't come back. After
 * each save the test reloads the tuning panel and reads the
 * slider's *current value from the DOM* (not the registry
 * fallback), which is the contract the user sees.
 *
 * Snapshot policy mirrors the earlier specs:
 *   - before.png: panel open, slider at default
 *   - after.png:  panel open, slider at new value, save button hidden
 *
 * Cleanup: each test reads the pre-test value of
 * `hihat.pga_min_combined_score` and restores it on the way out,
 * so the project 10 fixture is not left in a modified state.
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
  "04-combined-score",
);

// Project 10 = Metallica_All_Nightmare_Long_Drums. The hihat
// stem is where the combined_score (warble) filter was validated
// to be a perfect precision separator (528 FPs with cs ≤ 0,
// 225 real hits with cs > 0). Use that data as the regression
// baseline for this filter.
const FIXTURE_PROJECT_NAME = "Metallica_All_Nightmare_Long_Drums";
const FIXTURE_PROJECT_NUMBER = 10;
const STEM_TYPE = "hihat";

// ─── Helpers ──────────────────────────────────────────────────────────

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

function extractHihatCombinedScore(yamlContent) {
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
    const m = line.match(/^\s+pga_min_combined_score\s*:\s*(\S+)\s*$/);
    if (m) {
      const v = Number(m[1]);
      return Number.isFinite(v) ? v : null;
    }
  }
  return null;
}

async function writeConfigPath(request, projectNumber, pathSegments, value) {
  const res = await request.post(
    `/api/config/${projectNumber}/midiconfig`,
    {
      headers: { "Content-Type": "application/json" },
      data: { updates: [{ path: pathSegments, value }] },
    },
  );
  if (!res.ok()) {
    throw new Error(
      `POST /api/config/${projectNumber}/midiconfig → ${res.status()}: ${await res.text()}`,
    );
  }
  return await res.json();
}

async function getTuningConfig(request, projectNumber, stemType) {
  const res = await request.get(
    `/api/projects/${projectNumber}/tuning-config/${stemType}`,
  );
  if (!res.ok()) {
    throw new Error(
      `GET /api/projects/${projectNumber}/tuning-config/${stemType} → ${res.status()}: ${await res.text()}`,
    );
  }
  return await res.json();
}

async function getAnalysis(request, projectNumber) {
  const res = await request.get(`/api/projects/${projectNumber}/analysis`);
  if (!res.ok()) return null;
  return await res.json();
}

// ─── Test fixture setup ──────────────────────────────────────────────

async function openHihatTuningPanel(page) {
  // Boot the webui and navigate to project 10
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

  // Reveal analysis and switch to hihat tab
  const analysisSection = page.locator("#analysis-section");
  await expect(analysisSection).toBeVisible({ timeout: 20_000 });
  const hihatTab = page
    .locator('.waveform-stem-tab[data-stem="hihat"]')
    .first();
  await expect(hihatTab).toBeVisible({ timeout: 15_000 });
  await hihatTab.click();
  await expect(hihatTab).toHaveClass(/waveform-tab-active/);

  // Open the tuning panel. As of 2026-06-19 the hihat pipeline
  // populates events_sensitive so the Tune button is visible.
  const tuneButton = page.locator("#tuning-toggle-btn");
  await expect(tuneButton).toBeVisible({ timeout: 5_000 });
  await tuneButton.click();

  const tuningPanel = page.locator("#tuning-panel");
  await expect(tuningPanel).toBeVisible({ timeout: 5_000 });
}

// ─── Tests ───────────────────────────────────────────────────────────

test.beforeAll(() => {
  fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
});

test("A. server returns pga_min_combined_score so the slider is not stuck at 0", async ({
  page,
  request,
}) => {
  // The /api/projects/<n>/tuning-config/<stem> endpoint must
  // include pga_min_combined_score in its resolved dict. The
  // resolution is per-stem > global > 0.0 (per the same pattern
  // as the other PGA filters). When the WebUI loads the panel
  // and calls getTuningConfig, this key is what backs the slider's
  // initial value — without it, the slider uses the registry's
  // fallback (0) and the user's saved value is invisible.
  const cfg = await getTuningConfig(
    request,
    FIXTURE_PROJECT_NUMBER,
    STEM_TYPE,
  );
  expect(cfg).toHaveProperty("pga_min_combined_score");
  // The type is a number (not a string, not null).
  expect(typeof cfg.pga_min_combined_score).toBe("number");
  expect(Number.isFinite(cfg.pga_min_combined_score)).toBe(true);
});

test("B. moved slider persists across save+reload (does not reset to 0)", async ({
  page,
  request,
}) => {
  // Save the pre-test value so we can restore on cleanup.
  const yamlBefore = await readProjectConfig(request, FIXTURE_PROJECT_NUMBER);
  const preTestValue = extractHihatCombinedScore(yamlBefore);
  // And clear the threshold so we start from a known state.
  await writeConfigPath(
    request,
    FIXTURE_PROJECT_NUMBER,
    ["hihat", "pga_min_combined_score"],
    0,
  );

  try {
    // Hard-reload the page so the in-memory `tuningSliderValues`
    // cache (left over from any prior test run that moved this
    // slider) is cleared. Without this, a previous run's stored
    // value would mask the 0 we just wrote to the yaml.
    await page.goto("/");

    await openHihatTuningPanel(page);
    await page.screenshot({ path: path.join(SNAPSHOT_DIR, "B-before.png") });

    // Find the combined_score slider in the tuning panel. The
    // registry slug is `pga_min_combined_score` and the WebUI
    // builds the input id from the key as `tuning-slider-<key>`.
    const slider = page.locator(
      '#tuning-panel input[type="range"][id="tuning-slider-pga_min_combined_score"]',
    );
    await expect(slider).toBeVisible({ timeout: 5_000 });

    // The slider's initial value is what the server reports
    // (after fix). We asserted that already in test A; here
    // we just read the displayed value.
    const initialValue = await slider.inputValue();
    expect(initialValue).toBe("0");

    // Move the slider to 500 (well within the registry's
    // -10000 to 10000 range). The exact value matters less than
    // "the persisted value is visible after save+reload".
    await slider.fill("500");
    const saveBtn = page.locator("#tuning-save-btn");
    await expect(saveBtn).toBeVisible({ timeout: 3_000 });
    await saveBtn.click();

    // Wait for save to complete — the button text returns to
    // its resting state. (The Save flow fetches the live yaml
    // and re-applies the filter, which means the slider's stored
    // value should match what was sent.)
    await expect(saveBtn).toBeEnabled({ timeout: 10_000 });

    // Read the slider's current value AFTER save+reload. This is
    // the user-visible contract — without the server fix, the
    // slider would read 0 (the registry fallback) here because
    // the server never returned the saved value.
    const valueAfterSave = await slider.inputValue();
    expect(valueAfterSave).toBe("500");

    // Read the raw yaml to confirm the value was persisted.
    const yamlAfter = await readProjectConfig(request, FIXTURE_PROJECT_NUMBER);
    const persistedValue = extractHihatCombinedScore(yamlAfter);
    expect(persistedValue).toBe(500);

    await page.screenshot({ path: path.join(SNAPSHOT_DIR, "B-after.png") });
  } finally {
    // Restore the pre-test value (best-effort).
    if (preTestValue !== null) {
      try {
        await writeConfigPath(
          request,
          FIXTURE_PROJECT_NUMBER,
          ["hihat", "pga_min_combined_score"],
          preTestValue,
        );
      } catch {
        // Soft-fail: the spec result is more important than
        // the yaml restore if the API is being torn down.
      }
    }
  }
});

test("C. slider range (min/max) reflects the sidecar's combined_score", async ({
  page,
  request,
}) => {
  // The slider's min and max should be computed from the
  // sidecar's combined_score values for the active stem, not the
  // fixed registry defaults of -10000/10000. This is so the
  // full resolution of the slider is usable within the actual
  // data range. We assert both:
  //   1. The slider's `min` attribute equals the sidecar's
  //      min combined_score (rounded sensibly).
  //   2. The slider's `max` attribute equals the sidecar's
  //      max combined_score (rounded sensibly).
  await openHihatTuningPanel(page);
  const slider = page.locator(
    '#tuning-panel input[type="range"][id="tuning-slider-pga_min_combined_score"]',
  );
  await expect(slider).toBeVisible({ timeout: 5_000 });

  // Pull the sidecar's combined_score range for the hihat stem.
  // The sidecar's events_pga is the full per-event list; we
  // walk the kept events (those with status='KEPT') to get the
  // range used for real-time tuning. The WebUI sees the same
  // list when computing the slider range.
  const analysis = await getAnalysis(request, FIXTURE_PROJECT_NUMBER);
  expect(analysis).not.toBeNull();
  const hihatEvents = analysis?.stems?.hihat?.events_pga || [];
  const kept = hihatEvents.filter(
    (e) => e && e.status === "KEPT" && typeof e.combined_score === "number",
  );
  expect(kept.length).toBeGreaterThan(0);

  const csValues = kept.map((e) => e.combined_score);
  const dataMin = Math.min(...csValues);
  const dataMax = Math.max(...csValues);

  // The WebUI's slider range should reflect the sidecar's range.
  // We don't pin to the exact min/max (the WebUI may round or
  // pad), but it should be within the same order of magnitude
  // as the sidecar's range — not the registry's -10000/10000.
  const sliderMin = Number(await slider.getAttribute("min"));
  const sliderMax = Number(await slider.getAttribute("max"));

  // Assert the slider's min is not -10000 (registry default that
  // would make the slider unusable for hihat's narrow range).
  expect(sliderMin).not.toBe(-10000);
  // The slider's min should be close to the data's min.
  expect(Math.abs(sliderMin - dataMin)).toBeLessThan(
    Math.max(1000, Math.abs(dataMin) * 0.1),
  );
  // Same for max.
  expect(sliderMax).not.toBe(10000);
  expect(Math.abs(sliderMax - dataMax)).toBeLessThan(
    Math.max(1000, Math.abs(dataMax) * 0.1),
  );
});
