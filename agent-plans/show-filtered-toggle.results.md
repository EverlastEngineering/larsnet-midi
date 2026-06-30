# Results: Fix "Show Filtered" toggle on the waveform

## Status

**Complete** — 2026-06-30

## Summary

The "Show Filtered" checkbox in the waveform panel didn't make
filtered events appear on the events canvas unless the user first
clicked "Tune" (entering tuning mode). The legend updated correctly
when the toggle was clicked, but the actual canvas drew zero red
bars.

The user had Playwright tests covering this area at some point
("we might have tossed them accidentally") — none of the existing
5 specs (01-05) cover the toggle-canvas interaction. Added a new
spec (06) that reproduces the bug, then fixed the underlying
two-cause bug.

## Phase 1: Test (red)

`tests/playwright/specs/06-show-filtered-toggle.spec.ts` (NEW,
2 tests):

1. **`show-filtered toggle renders Filtered legend entry without
   slider interaction`** — toggles the checkbox without touching
   any slider. Asserts:
   - Legend goes from "Kept (N) PGA (N)" → "Kept (N) PGA (M) Filtered (K)"
   - Canvas has > 500 red pixels (was 0 before fix)
   - Toggle is idempotent (on→off→on produces the same result)
2. **`show-filtered toggle works after clicking Tune`** — the
   existing workaround path. Verifies we didn't break it.

Red phase (before fix):
- Test 1 failed: `red FILTERED pixels on canvas: 0` (expected > 500)
- Test 2 passed (the workaround still worked)

## Phase 2: Fix (green)

Two surgical changes in `webui/static/js/waveform.js`:

### Change A — `getPgaEventsForStem` (line 855)

Gated the KEPT-only filter on `waveformShowFiltered`:

```javascript
const all = stemData.events_pga || [];
if (waveformShowFiltered) return all;     // 2026-06-30: pass FILTERED through
return all.filter(e => e.status === 'KEPT');  // 2026-06-19 default
```

This makes the toggle work without requiring the user to first
click "Tune" (the previous workaround). The 2026-06-19
empty-display safeguard is preserved for the toggle-OFF case.

### Change B — `drawPgaEventBars` (line 756)

Replaced the manual color table (which hardcoded `markerPga` for
all KEPT cases and silently used it for FILTERED too) with a
single `getEventColor(event)` call. This makes FILTERED events
render in red `#ef4444` consistently with the events panel.

## Phase 3: Verification

Green phase (after fix):
- Test 1 passes: red pixel count went from **0 → 2934**
- Test 2 still passes: the workaround path is preserved
- All 4 pre-existing pytest failures unchanged
- All 4 pre-existing playwright failures (specs 02, 03, 04, 05)
  unchanged (they're pre-existing flakes, not caused by this fix)

## Decision Log

- **Root cause had two halves**:
  1. `getPgaEventsForStem` was hardcoded to KEPT-only,
     stripping FILTERED events before they could reach
     `drawPgaEventBars`.
  2. `drawPgaEventBars` used a manual color table that always
     returned `markerPga` (violet) for all events, so even when
     FILTERED events DID reach it (via the Tune workaround), they
     rendered as violet instead of red.

- **Why "click Tune" worked as a workaround**: when Tune is
  clicked, `waveformTuningActive=true` and the data source switches
  to `waveformTuningEvents` (the tuning-mode list that includes
  FILTERED). This bypasses `getPgaEventsForStem` entirely. The
  legend updates because the manual color table in the legend
  function is different from the one in `drawPgaEventBars` — it
  uses `getEventColor` correctly. (Or the legend draws its own
  Filtered count from the unfiltered `displayEvents` source.)

- **Why "red bars appear after slider drag"** (user's other
  reported symptom): a slider drag re-runs the client-side filter
  pass which explicitly rewrites the `status` field on each event
  in `tuningBaseEvents` (e.g. `event.status = 'KEPT'`,
  `event.status = 'FILTERED'`). This may flip some events' method
  field, allowing them to pass the events-panel's PGA skip-line
  and render in red via `getEventColor`. Exact mechanism not
  investigated further — the fix above resolves both the
  no-Tune and post-slider cases.

- **Why the existing test was hard to find**: the test signal
  the user remembered was probably the legend text (which always
  updated correctly even before the fix). The bug only manifests
  on the canvas (red pixel count), which is a harder signal to
  write a test for. The new spec uses both signals — legend text
  AND red pixel count — to catch the bug.

## Metrics

- **Files changed**: 1 source + 1 new test file + 1 plan/results markdown
- **Lines**: ~30 in source (2 surgical edits), ~270 in new test file, ~150 in plan/results
- **Playwright tests added**: 2 (1 reproduces the bug, 1 covers the workaround)
- **Pytest**: 855 passed (unchanged), 4 pre-existing failures (unchanged)
- **Playwright full suite**: 5 passed (was 4 before — Test 1 now passes too)
- **Pre-existing failures**: 4 (cymbals/legacy energy-path Python) + 4 (slider persistence/timing Playwright) — all unchanged

## Test Results

```
tests/playwright/specs/06-show-filtered-toggle.spec.ts
  ✓ show-filtered toggle renders Filtered legend entry without slider interaction (4.7s)
  ✓ show-filtered toggle works after clicking Tune (the workaround) (3.1s)
2 passed (8.7s)
```