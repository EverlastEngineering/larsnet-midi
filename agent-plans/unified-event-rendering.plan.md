# Plan: Unify Event Rendering — Pick the Faded-Red Path

## Problem (user-reported 2026-06-30)

Two render paths produce inconsistent colors:
- **Path A** (`drawEventBars` in the events panel) — renders non-PGA
  events using `getEventColor`. KEPT events with no classification
  fall through to `markerKept` (green). FILTERED events render
  in red with full alpha (0.9). This is the path that doesn't
  work right for the user's data.
- **Path B** (`drawPgaEventBars` — the third-detector overlay
  drawn on top) — uses `getEventColor` since the 2026-06-30
  "show filtered" fix. Renders KEPT PGA events as violet,
  KEPT events with classification in their classification color,
  FILTERED events in red with faded alpha (0.35). This is the
  "faded red" path the user wants to keep and enrich.

Concrete bugs the user observed:
- **Kick in project 6, pre-slider**: events_configured is
  rendered as GREEN (path A, no classification → markerKept),
  events_pga is rendered as VIOLET (path B, method='percentile_gated'
  → markerPga). The violet path B covers the green path A at the
  same X positions. User sees VIOLET.
- **Kick in project 6, post-slider**: `waveformTuningActive=true`,
  `waveformTuningEvents = tuningBaseEvents = events_configured`
  (190 events, no method, no classification). Both panels draw
  the SAME 190 events. Path A renders them GREEN. Path B calls
  getEventColor which returns `markerKept` (no PGA method, no
  classification → green). User sees GREEN — and it "breaks"
  because the data source suddenly switched from the canonical
  events_pga to a stale events_configured subset.

The kick sidecar has BOTH events_pga (2087 events, method='percentile_gated')
and events_configured (190 events, method=None) — legacy data from
before the 2026-06-15 PGA-only refactor. `getEventsForStem` picks
events_configured first (current behavior), causing the data source
to differ between non-tuning and tuning modes.

## Approach

Pick the faded-red path as the unified render. Enrich it to handle
any data (PGA, energy, spectral, with/without classification) by
relying on `getEventColor` (which already encodes all cases). Drop
the duplicate `drawEventBars` call from the events panel; the
existing `drawPgaEventBars` (renamed) becomes the only path.

### Phase 1: Unify the data source

**`getEventsForStem` (line 839)** — prefer `events_pga` so the
display layer always uses the canonical PGA-detected set:

```javascript
function getEventsForStem(stemData) {
    if (stemData.events_pga) return stemData.events_pga;        // 2026-06-30: prefer PGA
    if (stemData.events_configured) return stemData.events_configured;
    if (stemData.events) return stemData.events;
    return [];
}
```

The data source is now consistent across non-tuning and tuning
modes — both pull from the same list. The kick bug (green-after-
slider) is fixed because `waveformTuningEvents` is initialized from
the same `events_pga` source.

**`initTuningBaseEvents` (threshold-tuning.js)** — same change:

```javascript
const configuredEvents = stemData?.events_pga || stemData?.events_configured;
```

### Phase 2: Single render path

Replace the two draw calls in `drawEventsPanel` with one call to
the unified renderer. Rename `drawPgaEventBars` → `drawUnifiedEventBars`
to reflect its new role. The unified function:

- Calls `getEventColor(event)` for color (handles all event types)
- Uses faded-red alpha convention: filtered = 0.35, kept = 0.85
- Respects `waveformShowFiltered` toggle (skip filtered when off)
- Handles the sensitive-layer case via an `isSensitiveLayer` flag
  (for the tuning background; uses markerSensitive color + 0.4 alpha)

```javascript
function drawUnifiedEventBars(ctx, events, timeToX, PAD, plotW, plotH, isSensitiveLayer) {
    const barWidth = isSensitiveLayer ? 1.5 : 2.5;
    for (const event of events) {
        if (event.time == null) continue;
        const isFiltered = event.status === 'FILTERED';
        // Sensitive layer shows filtered events for context; the
        // main layer respects the "Show Filtered" toggle.
        if (!isSensitiveLayer && isFiltered && !waveformShowFiltered) continue;
        const x = timeToX(event.time);
        if (x < PAD.left - barWidth || x > PAD.left + plotW + barWidth) continue;

        const color = isSensitiveLayer
            ? WAVEFORM_COLORS.markerSensitive
            : getEventColor(event);

        let velocity = event.midi_velocity;
        if (velocity == null) velocity = isFiltered ? 60 : 100;
        const barH = Math.max(2, (velocity / 127) * plotH);
        const barTop = PAD.top + plotH - barH;

        // Faded red convention: filtered = 0.35, kept = 0.85
        const alpha = isSensitiveLayer ? 0.4 : (isFiltered ? 0.35 : 0.85);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.fillRect(x - barWidth / 2, barTop, barWidth, barH);

        // Outline
        ctx.globalAlpha = isFiltered ? 0.4 : (isSensitiveLayer ? 0.5 : 1.0);
        ctx.strokeStyle = color;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x - barWidth / 2, barTop, barWidth, barH);

        if (event._overridden) {
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = '#ffffff';
            const dSize = 3;
            ctx.beginPath();
            ctx.moveTo(x, barTop - dSize);
            ctx.lineTo(x + dSize, barTop);
            ctx.lineTo(x, barTop + dSize);
            ctx.lineTo(x - dSize, barTop);
            ctx.closePath();
            ctx.fill();
        }
    }
    ctx.globalAlpha = 1.0;
}
```

### Phase 3: drop the duplicate `drawEventBars` call

`drawEventsPanel` now has ONE draw call:

```javascript
// Before: two draw calls
//   drawEventBars(ctx, eventsToRender, ..., false);  // main events
//   drawPgaEventBars(ctx, pgaEvents, ...);             // PGA overlay

// After: one call, using the same data source
const eventsToRender = displayEvents;   // events_pga for PGA-only stems
drawUnifiedEventBars(ctx, eventsToRender, timeToX, PAD, plotW, plotH, false);
```

The `displayEvents` is now `events_pga` (via getEventsForStem).
`getPgaEventsForStem` becomes redundant — can be removed or kept
as an alias.

### Phase 4: simplify `getPgaEventsForStem`

The function is no longer needed (drawPgaEventBars is renamed and
takes a single list of events). But callers may still reference
it. Either:

- **A**: Remove the function and update callers.
- **B**: Keep it as `function getPgaEventsForStem(stemData) {
  return getEventsForStem(stemData); }` (alias).

Choose **B** for minimal blast radius — no other callers need
to change.

## Files Changed

1. `webui/static/js/waveform.js`:
   - `getEventsForStem` (line 839) — prefer events_pga
   - Rename `drawPgaEventBars` → `drawUnifiedEventBars`
   - Enrich the function: any event type, faded red convention,
     sensitive-layer flag
   - `drawEventsPanel` (line 568) — single draw call
   - `getPgaEventsForStem` (line 855) — alias to `getEventsForStem`
   - `drawWaveform` (line 437) — `displayEvents` and `pgaEvents`
     become the same source (events_pga); consolidate the pgaEvents
     variable since both panels use the same data

2. `webui/static/js/threshold-tuning.js`:
   - `initTuningBaseEvents` (line 1422) — prefer events_pga

3. `tests/playwright/specs/`:
   - New spec `07-unified-rendering.spec.ts` covering:
     - Kick in project 6 renders violet (PGA color) both before
       and after a slider drag (the regression)
     - Snare in project 6 renders KEPT events in their
       classification color (snare body = green, rimshot = purple)
       via the unified path
     - Filtered events render in faded red when toggle is ON
   - Optionally extend `06-show-filtered-toggle.spec.ts` if the
     current "with Tune" test needs to be re-anchored to the
     unified path (the spec is already correct; should pass
     unchanged)

## Risks

- **Visual identity change**: KEPT events that were green (path A)
  become their classification color (or markerKept via getEventColor's
  fallback for no classification). For project 6 kick (no classification,
  no method), they're now GREEN (still — getEventColor returns
  markerKept). The visual change is mainly for kicks that
  previously showed two different colors stacked.
- **Stems without events_pga**: legacy non-PGA stems that
  exclusively have events_configured still render correctly
  via the same getEventColor pipeline.
- **drawEventBars / drawPgaEventBars removal**: any other callers
  of these functions (in the tuning background) need to be
  updated. The tuning path passes `isSensitiveLayer=true` to
  the unified function.

## Success Criteria

1. `tests/playwright/specs/07-unified-rendering.spec.ts` passes
   for kick (violet, both before and after slider).
2. The pre-existing `06-show-filtered-toggle.spec.ts` tests
   continue to pass (no regression in the show-filtered path).
3. Full pytest suite: no new regressions (855 + new tests).
4. Full playwright suite: no new regressions (5 + new tests).