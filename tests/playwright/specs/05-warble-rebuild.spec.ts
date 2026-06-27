/**
 * 05 — end-to-end test for the warble filter (pga_min_combined_score)
 *
 * The earlier unit-style tests (04-) checked the slider's
 * cosmetic behavior in the tuning panel. THIS test checks
 * the actual end-to-end effect on the sidecar's kept count
 * when the slider is changed. The earlier failure mode was
 * that the rebuild pipeline didn't re-apply the warble filter,
 * so the rebuilt sidecar's Kept count was identical regardless
 * of the slider value.
 *
 * This test:
 *   1. Reads the current sidecar (baseline Kept count).
 *   2. Sets pga_min_combined_score = 100 via the API.
 *   3. Calls POST /api/rebuild-midi to rebuild the sidecar.
 *   4. Reads the new sidecar and checks the Kept count
 *      has changed (slider actually filtered something).
 *   5. Checks that NO kept event has combined_score < 100
 *      (the filter is actually applied, not a no-op).
 *   6. Restores the original threshold and rebuilds.
 *
 * The test uses the HTTP API directly rather than the WebUI to
 * isolate the pipeline behavior from the UI layer (which has its
 * own known issues per the user's earlier feedback). This catches
 * regressions where the warble filter is removed from the
 * rebuild chain but the UI still looks fine in isolation.
 */
import { test, expect } from "@playwright/test";

const BASE = "http://localhost:4915";
const PROJ = 10;

async function getAnalysis(request: any): Promise<any> {
  const r = await request.get(`${BASE}/api/projects/${PROJ}/analysis`);
  if (!r.ok()) throw new Error(`getAnalysis ${r.status()}`);
  return r.json();
}

async function getMidiconfig(request: any): Promise<string> {
  const r = await request.get(`${BASE}/api/projects/${PROJ}/config/midiconfig.yaml`);
  if (!r.ok()) throw new Error(`getMidiconfig ${r.status()}`);
  const data = await r.json();
  return data.content;
}

async function writeConfig(
  request: any,
  pathSegments: string[],
  value: any,
): Promise<any> {
  const r = await request.post(`${BASE}/api/config/${PROJ}/midiconfig`, {
    data: { updates: [{ path: pathSegments, value }] },
  });
  if (!r.ok()) {
    throw new Error(`writeConfig ${r.status()}: ${await r.text()}`);
  }
  return r.json();
}

async function rebuild(request: any): Promise<any> {
  const r = await request.post(`${BASE}/api/rebuild-midi`, {
    data: { project_number: PROJ, honor_overrides: true },
  });
  if (!r.ok()) {
    throw new Error(`rebuild ${r.status()}: ${await r.text()}`);
  }
  return r.json();
}

function countKept(sidecar: any): {
  total: number;
  kept: number;
  keptWithCSBelowThreshold: number;
} {
  const events = sidecar?.stems?.hihat?.events_pga ?? [];
  const kept = events.filter((e: any) => e?.status === "KEPT");
  return {
    total: events.length,
    kept: kept.length,
    keptWithCSBelowThreshold: kept.filter(
      (e: any) => (e?.combined_score ?? 0) < 100,
    ).length,
  };
}

test("warble filter actually filters on rebuild (Kept count changes)", async ({
  request,
}) => {
  // ---- Baseline ----
  const baseline = countKept(await getAnalysis(request));
  const baselineConfig = await getMidiconfig(request);
  let baselineThreshold = 0;
  for (const line of baselineConfig.split("\n")) {
    const m = line.match(/^\s*pga_min_combined_score\s*:\s*(\S+)\s*$/);
    if (m && !line.trim().startsWith("#")) {
      baselineThreshold = Number(m[1]);
    }
  }

  // ---- Apply the new threshold ----
  await writeConfig(request, ["hihat", "pga_min_combined_score"], 100);
  const r = await rebuild(request);
  expect(r.success).toBe(true);

  // ---- Verify the new sidecar ----
  const filtered = countKept(await getAnalysis(request));
  // The point: Kept should differ from baseline (slider had
  // an effect). If the rebuild path didn't re-apply the
  // warble filter, the Kept count would be unchanged.
  expect(filtered.kept).not.toBe(baseline.kept);
  // And: the filter is actually doing something, not a no-op.
  // Every kept event should have combined_score >= 100.
  expect(filtered.keptWithCSBelowThreshold).toBe(0);

  // ---- Revert and verify Kept returns to baseline ----
  await writeConfig(request, ["hihat", "pga_min_combined_score"], baselineThreshold);
  const r2 = await rebuild(request);
  expect(r2.success).toBe(true);
  const restored = countKept(await getAnalysis(request));
  expect(restored.kept).toBe(baseline.kept);
});
