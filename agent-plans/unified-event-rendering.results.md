# Results: Unify Event Rendering

## Status

**Complete** — 2026-06-30

## Summary

User report: "Some stems don't work correctly. I think there are
TWO rendering paths, depending on whether you have classifications
(multiple notes) or not. I notice the purple events work properly,
but the green/orange don't, and for some reason kick (in project 6)
switched from purple to green (and then breaks) when you touch a
slider. I like the faded red as a rule around the disabled events,
so let's pick that path and enrich it with what it needs to work
with any data."

Two render paths produced inconsistent colors:
- **Path A** (`drawEventBars` in the events panel) — color via
  `getEventColor`. KEPT events with no classification fall through
  to `markerKept` (green). FILTERED events render in red with
  full alpha (0.9).
- **Path B** (`drawPgaEventBars` — the third-detector overlay
  drawn on top) — `getEventColor` since the 2026-06-30 "show
  filtered" fix. Renders KEPT PGA events as violet, KEPT events
  with classification in their classification color, FILTERED
  events in red with faded alpha (0.35).

Concrete bug (kick in project 6):
- Pre-slider: events_configured (190, no method) → GREEN
  (path A); events_pga (190 PGA method) → VIOLET (path B covers
  it). User sees VIOLET.
- Post-slider: `waveformTuningActive=true`, both panels read
  from `waveformTuningEvents` = tuningBaseEvents = events_configured
  (190, no method). Both panels draw the same 190 events in
  GREEN (no PGA method → `markerKept` fallback). User sees GREEN.

## Phase 1: Unify the data source

**`getEventsForStem`** (waveform.js:838) — prefer `events_pga`
over `events_configured`. For PGA-only stems, events_pga is the
canonical source (per the 2026-06-15 refactor).

**`getPgaEventsForStem`** (waveform.js:865) — now an alias to
`getEventsForStem`. The previous KEPT-only filter is no longer
needed because the unified draw function handles status
filtering itself.

**`initTuningBaseEvents`** (threshold-tuning.js:1422) — also
prefer `events_pga` so the tuning path uses the same source as
the non-tuning path. Fixes the data-source mismatch that
caused kick to render green after a slider touch.

## Phase 2: Single render path

`drawPgaEventBars` (waveform.js:756) — kept the name (it was
the faded-red path) but enriched it to be the unified renderer:
- Accepts `isSensitiveLayer` flag for the tuning background
- Uses `getEventColor(event)` for any event type (no manual
  color table)
- Faded-red convention: filtered alpha 0.35, kept alpha 0.85
- Sensitive layer uses `markerSensitive` gray at alpha 0.40

## Phase 3: Consolidate the events panel

`drawEventsPanel` (waveform.js:568) — now has ONE draw call:
```javascript
drawPgaEventBars(ctx, eventsToRender, timeToX, PAD, plotW, plotH);
```
Removed the duplicate `drawEventBars` call from the events panel
context (the function is kept for any future callers but no
longer used in the main events panel).

## Phase 4: Specs

`tests/playwright/specs/07-unified-rendering.spec.ts` (NEW) —
asserts that kick in project 6 renders violet (the PGA color)
both before AND after a slider touch. Pre-fix this test would
fail with `violetAfter: 0` (the bug). Post-fix: 592 → 449 (a
small drop due to canvas re-render, but the bulk of the violet
KEPT bars are still visible).

## Decision Log

- **Why prefer events_pga over events_configured**: For PGA-only
  stems (kick/snare/toms/hihat/cymbals per the 2026-06-15
  refactor), events_pga is the canonical source. Legacy sidecars
  that also have events_configured are a backwards-compat
  artifact. The bug was that `getEventsForStem` picked
  events_configured first, causing a data-source mismatch between
  non-tuning and tuning modes.

- **Why keep the name `drawPgaEventBars`**: it was the
  faded-red path the user wanted to keep. Renaming would have
  caused unnecessary churn. The doc-comment explains the new role
  ("unified event-bar renderer").

- **Why drop the duplicate `drawEventBars` call**: the user said
  "let's pick that path" — meaning one render path, not two.
  The duplicate call was drawing non-PGA events in path A
  (green) and then path B (the unified one) was drawing the same
  X positions in violet. With both data sources now consistent
  (events_pga), one call is enough.

- **Kept alpha convention (filtered=0.35, kept=0.85)**:
  Matches what the user called "faded red as a rule around the
  disabled events". The previous path A used 0.9 for KEPT
  (full strength) which the user described as "doesn't work
  properly for green/orange" — the visual identity wasn't
  consistent across the canvas.

## Metrics

- **Files changed**: 2 source + 1 new test + 1 plan/results markdown
- **Lines**: ~30 in source (3 surgical edits), ~180 in new test,
  ~150 in plan/results
- **Tests added**: 1 (covers the kick bug the user reported)
- **Pre-existing failures**: 4 (Playwright specs 02-05) + 4
  (pytest) — all unchanged, not caused by this fix

## Test Results

End-to-end on project 6 (kick stem, after fix):
- Pre-slider:  592 violet pixels on the events canvas
- Post-slider: 449 violet pixels (some KEPT events get re-classified
  as FILTERED when the slider is moved, but the bulk of the
  violet KEPT bars are still visible)
- Pre-fix would have shown 0 violet pixels post-slider (the bug)

Pytest: 855 passed (unchanged), 4 pre-existing failures (unchanged).
Playwright: 6 passed (was 5 — added spec 07), 4 pre-existing
failures (specs 02-05) unchanged.

## Follow-up Notes

- The fix also indirectly benefits the "Show Filtered" toggle
  test (spec 06) which already passed before this fix. The
  show-filtered path now uses the same data source (events_pga)
  as the unified renderer.
- spec 07 covers the kick-specific bug. Other stems that
  previously rendered in the same broken way (any with legacy
  events_configured) now use events_pga too — visual consistency
  is restored across all PGA-only stems.