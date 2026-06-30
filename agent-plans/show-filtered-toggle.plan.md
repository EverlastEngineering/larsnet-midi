# Plan: Fix "Show Filtered" toggle on the waveform

## Problem (user-reported 2026-06-30)

Clicking the "Show Filtered" checkbox in the waveform panel does NOT
make filtered events appear on the events canvas — UNLESS the user
first clicks "Tune" (which enters tuning mode). The legend updates
correctly when the toggle is clicked (shows `Filtered (1203)`), but
the actual canvas shows zero red bars.

Verified by Playwright:
- Toggle OFF (no Tune): canvas shows green + violet bars (KEPT events)
- Toggle ON (no Tune): canvas UNCHANGED — still only green + violet, no red
- Toggle ON (with Tune clicked): canvas shows green + violet + **red**

Test signal in `tests/playwright/specs/06-show-filtered-toggle.spec.ts`
(new spec added during this investigation, was missing from the suite
despite the user remembering prior coverage).

## Root cause

`webui/static/js/waveform.js` has two layered draw passes for events:

1. **`drawEventBars` (events panel)** at line 660 — draws
   `displayEvents`. Skips events where
   `event.method === 'percentile_gated'` (line 678) so the PGA
   layer below isn't doubled up. Colors via `getEventColor()`:
   FILTERED → red `#ef4444`, KEPT → classification color (green /
   cyan / etc.).

2. **`drawPgaEventBars` (PGA layer)** at line 756 — draws
   `pgaEvents` (a separate list, drawn LAST so it sits on top).
   Always uses `WAVEFORM_COLORS.markerPga` (violet `#8b5cf6`).
   Has its own `isFiltered && !waveformShowFiltered` skip at line 787.

The two data sources are computed in `drawWaveform()` (line 437):

```javascript
const pgaEvents = (waveformTuningActive && waveformTuningEvents)
    ? waveformTuningEvents
    : getPgaEventsForStem(stemData);
const displayEvents = (waveformTuningActive && waveformTuningEvents)
    ? waveformTuningEvents
    : configuredEvents;
```

For snare (PGA-only stem):
- `waveformTuningActive=false` (no Tune clicked) → `pgaEvents = getPgaEventsForStem(stemData)`
- `getPgaEventsForStem` at line 855 HARDCODES a filter to KEPT-only:
  ```javascript
  return all.filter(e => e.status === 'KEPT');
  ```
- So `pgaEvents` has only 247 KEPT events; the 1203 FILTERED events
  are never passed to `drawPgaEventBars`.
- `displayEvents = configuredEvents = events_pga` (1450 events).
- Events panel (`drawEventBars`) SKIPS all 1450 because they're all
  PGA method. Nothing is drawn in the events panel.
- PGA panel draws only the 247 KEPT events as violet. No red.

That's the bug: without Tune, `getPgaEventsForStem` strips the
FILTERED events before they can be rendered, and `drawEventBars`
skips PGA events. The legend counts from `getEventsForStem` (the
unfiltered `displayEvents` source) and updates correctly, but no
filtered bars reach the canvas.

With Tune clicked:
- `waveformTuningActive=true` → `pgaEvents = waveformTuningEvents`
  (the deep-copy list with both KEPT and FILTERED).
- `displayEvents = waveformTuningEvents` (same list).
- Events panel still skips all events (PGA method).
- PGA panel now sees FILTERED events, and since the toggle is on,
  its own isFiltered check passes → draws them.
- But `drawPgaEventBars` always uses markerPga (violet) regardless
  of status. So FILTERED events render as violet, not red.

So WITH Tune, FILTERED events DO show — but they're violet, not red.
The user perceives the red bars because something else is happening
that's not yet understood. (Possibly the events panel re-runs through
`drawEventBars` on a subsequent slider drag, where the slider's
applyTuningFilter rebuilds `tuningBaseEvents` with explicit status
rewrites that flip the per-event method, so PGA events become
non-PGA and pass the events-panel skip-line.)

The expected behavior (after the fix): the toggle should work
standalone. FILTERED events should render in red. The Tune click
should be optional.

## Approach

Two small, surgical changes in `webui/static/js/waveform.js`:

### Change A — `getPgaEventsForStem` (line 855)

Make the function respect `waveformShowFiltered`:

```javascript
function getPgaEventsForStem(stemData) {
    const all = stemData.events_pga || [];
    if (waveformShowFiltered) {
        // User has toggled "Show Filtered" ON — pass FILTERED events
        // through so drawPgaEventBars can render them in red.
        // drawPgaEventBars has its own isFiltered check that
        // re-hides them if the user toggles back off. This makes
        // the toggle work without requiring the Tune button to be
        // clicked first.
        return all;
    }
    // 2026-06-19 default: KEPT-only. Avoids the empty-display
    // regression when the sidecar is mostly-FILTERED — same
    // rationale as before. The toggle gates this.
    return all.filter(e => e.status === 'KEPT');
}
```

This is the smaller, safer change. It doesn't change the
default behavior (KEPT-only when toggle is off).

### Change B — `drawPgaEventBars` (line 756, color picker at ~line 805)

Replace the hardcoded `WAVEFORM_COLORS.markerPga` with
`getEventColor(event)` so FILTERED events render red:

```javascript
// Before:
const color = WAVEFORM_COLORS.markerPga;
// After:
const color = getEventColor(event);
```

Now PGA-layer events use the same color rules as the events panel:
- KEPT → violet (PGA, via getEventColor method='percentile_gated')
- FILTERED → red
- REVERB_CONTINUATION → orange

This makes the visual identity "red = filtered" consistent
across the waveform, regardless of which layer drew the bar.
Also makes the red bars visible WITHOUT clicking Tune (which is
the user's reported bug).

## Files Changed

1. `webui/static/js/waveform.js` — two surgical changes:
   - `getPgaEventsForStem` (line 855) — gate the KEPT-only filter
     on `waveformShowFiltered`
   - `drawPgaEventBars` (line 805) — use `getEventColor` instead
     of hardcoded `markerPga`

2. `tests/playwright/specs/06-show-filtered-toggle.spec.ts` (NEW)
   - Test 1: toggle works without Tune (the regression)
   - Test 2: toggle still works after Tune (the workaround,
     confirms we haven't broken the existing path)

3. `agent-plans/show-filtered-toggle.results.md` (NEW)

## Risks

- **Visual identity change**: PGA-layer events used to be
  universally violet. With Change B, FILTERED PGA events become
  red. The legend's "PGA (N)" entry still says "PGA" but the
  bars may now be a mix of violet (KEPT) and red (FILTERED). This
  is a deliberate improvement — "red = filtered" is a stronger
  visual invariant than "violet = PGA method".
- **Toggle+slider interaction**: The current behavior (Tune +
  slider + toggle) was already showing red bars. The fix doesn't
  touch that path; it just extends the same red-bar behavior to
  the no-Tune case.
- **Time-range computation**: `computeTimeRange` uses `pgaEvents`
  for the auto-zoom. With Change A, `pgaEvents` includes FILTERED
  when toggle is on. The time range widens to cover FILTERED events
  outside the KEPT range. This is the correct behavior — the user
  can see the FILTERED events, so the time axis should include them.

## Success Criteria

1. `tests/playwright/specs/06-show-filtered-toggle.spec.ts::show-filtered
   toggle renders Filtered legend entry without slider interaction`
   passes (currently fails with `red FILTERED pixels on canvas: 0`).
2. The "with Tune" test (Test 2) still passes — the workaround
   path is preserved.
3. Full pytest suite: no new regressions.
4. Visual: clicking the toggle on snare shows red bars within
   ~200ms (one animation frame after the onchange), without
   any other UI interaction.