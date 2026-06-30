# Plan: Event Overrides — Real Cycling, Real Save, Real MIDI

## User request (verbatim, 2026-06-30)

> "Let's turn out focus to event_overrides.json. I can't be sure
> how and when they're used. The user may click an event in the UI
> and it should add and remove it from the ui, it should produce
> a midi with the event included or excluded. I don't think it's
> doing anything other than the initial UI hide / show.
>
> When the user clicks and overrides an event, it would have to
> be 'reprocessed' anyway, so we'll have to consider the UX
> around that. Right now the Save and Reconvert is in the Tune
> slideout, but you don't have to have that open to override an
> event, so it's likely we'll want to move the Save button out
> to the top by the text on the right. Make a plan and dig
> deep before doing anything.
>
> Also, there's no way to set the note (or the classification)
> so let's do that now too. The user can click and if there's
> only 1 class, it alternates enable / disable. If there ARE
> classes:
>   - if off, it turns on and is class 1
>   - if class 1, it cycles through the classes unless there
>     aren't any more
>   - then it disables
>
> we'll have to track the class in the event_overrides.json
> and if it matches the sidecar data, i suggest we remove it
> from the overrides to clear things up."

## User follow-up (2026-06-30)

> "I don't think the current flow is right. When I click, it DOES
> show in the override json. But when I save it shows in the UI
> again, and the MIDI has the FILTERED note anyway. A refresh
> doesn't reflect the overridden events either.
>
> I agree with the phases of work, but don't bother to keep the
> old format, we can nuke the old files and start fresh."

→ The override actually IS NOT reaching the MIDI correctly. See
"Three confirmed bugs" below. The "nuke old format" guidance
removes the schema-migration substep from Phase 1.

## What the current code does (and the 3 confirmed bugs)

### Data flow today

1. **Click event on canvas** — `toggleEventOverride(stemType, event)`
   in [`webui/static/js/waveform.js`](webui/static/js/waveform.js)
   (line 2186).
2. **In-memory** — `event.status` flips KEPT↔FILTERED. The override
   is stored in `eventOverrides[stemType][time_key]` (a string).
3. **Debounced save (500ms)** — `saveEventOverrides()` POSTs to
   `/api/projects/:n/event-overrides` (PUT handler at
   [`webui/api/projects.py:948`](webui/api/projects.py)).
4. **The file** lives at
   `user_files/<project>/midi/event_overrides.json`. Format
   today: `{stem: {time_str: "KEPT"|"FILTERED"}}`. Example for
   project 6 (current file at the time of writing):
   ```json
   {"kick": {"0.3273": "FILTERED", "1.3192": "KEPT", ...}}
   ```
5. **Rebuild path** — `/api/rebuild-midi`
   ([`webui/api/operations.py:398`](webui/api/operations.py))
   calls `rebuild_midi_for_project` which:
   - Loads `event_overrides.json` via `_load_overrides`
   - Calls `rebuild_events_from_analysis(overrides=overrides, ...)`
   - `_apply_overrides` (in
     [`stems_to_midi/rebuild_core.py:90`](stems_to_midi/rebuild_core.py))
     mutates `event['status']` to match the override.
   - The mutated events flow through
     `classify_notes → _map_note` to produce the MIDI events.
6. **The Save button** lives at `tuning-save-btn` inside
   `#tuning-panel` ([`index.html:687`](webui/templates/index.html)).
   It's hidden by default and only shown when there are unsaved
   tuning-slider changes.

### Three confirmed bugs the user just reported

Verified by reading the code end-to-end. All three need to be
fixed in this commit.

**Bug 1 — `applyOverridesToEvents` skips events_pga (JS).**
`waveform.js:2159` iterates only `events_configured ||
events_sensitive`. For PGA-only stems (the entire post-2026-06-15
refactor world, including project 6 — kick, snare, toms, hihat,
cymbals), the in-memory sidecar carries `events_pga`, not
`events_configured`. So the override is in JSON but **never
applied to the visible events** on initial load. The
canvas shows the sidecar's natural state, not the override.
→ "A refresh doesn't reflect the overridden events either"

**Bug 2 — `_apply_overrides` only mutates `pga_kept` (Python).**
`rebuild_core.py:79` iterates `pga_kept` only — the events that
PASSED the PGA prominence filter. The override is meant to be
applied AFTER the filter, but for events that the FILTER
removed (FILTERED in pga_filtered), the override is silently
ignored. So a user who toggles a FILTERED event to KEPT finds
the event is STILL FILTERED in the rebuilt MIDI — the filter
"wins" over the override.
→ "the MIDI has the FILTERED note anyway"

**Bug 3 — Filter status is re-stamped from `pga_kept` after
override (Python).** `rebuild_core.py:347` re-stamps every
`raw_pga_events[i].status` based on whether the time is in
`pga_kept` or `pga_filtered`. This loop runs AFTER
`_apply_overrides`. If the override was applied to `pga_kept`
(only KEPT cases), the FILTERED events in `pga_filtered`
retain their filter-derived FILTERED status — and the sidecar
sees them as FILTERED. This compounds Bug 2: the sidecar doesn't
update the way the user expects, so the user clicks Save again
and the sidecar is "wrong" again.
→ "shows in the UI again"

The fix for all three is the same direction: the override
should be treated as a **pre-filter veto** — the user's
"keep this event" / "drop this event" decision runs before
the filter chain, then the filter runs on the vetoed status.
For classification, it's a post-filter annotation (the
classification is applied after the filter decides KEPT/FILTERED).

## Goals

The user wants the override system to:
1. **Cycle classifications on click** — off → cls 0 → cls 1 →
   … → off. For single-class stems, behave as today (toggle).
2. **Track classification in `event_overrides.json`** so the
   override persists across reloads and is reflected in the
   rebuilt MIDI without any further user action beyond Save.
3. **Auto-cleanup** when an override matches the sidecar's
   current state for that event. Avoid the "stale shadow"
   failure mode where a user moves a slider and the override
   silently wins over the new threshold.
4. **Move the Save button out of the Tune panel** so users
   don't have to open Tune to commit their toggles. The button
   should appear when there are unsaved overrides OR unsaved
   tuning changes.
5. **Fix the 3 confirmed bugs** above. The override should
   actually reach the MIDI.

## Approach

### Phase 1: Schema + click cycle + front-end bug fix (JS)

The schema is **brand new** (no back-compat per user
direction). Old `event_overrides.json` files in
`user_files/*/midi/` will be **deleted** as part of the
commit. The new shape:

```json
{
  "snare": {
    "0.5267": {
      "status": "KEPT",
      "classification": 1
    },
    "1.2345": {
      "status": "FILTERED"
    }
  }
}
```

That is: `{stem: {time_str: {status, [classification]?}}}`. All
fields except `status` are optional. `classification` is set
when the user has cycled past the first "on" state.

**Click cycle logic** (replaces the binary `toggleEventOverride`):

```
is currently KEPT?
  no  → next: KEPT, classification = smallest available class
         (or null if no classes, e.g. kick)
  yes → has classification?
    no  → next: FILTERED
    yes → at the highest class?
      yes → next: FILTERED
      no  → next: KEPT, classification = current + 1
```

This matches the user's description: "if off, it turns on and
is class 1" (= smallest class in 0-indexed terms, displayed as
"class 1" in 1-indexed cluster card labels). "If class 1, it
cycles through the classes" — sequential, not wrap-around.
"Then it disables" — at the last class, the next click
disables.

Single-class / no-class stems: 2-state toggle (off ↔ on).
Same UX as today, but goes through the same code path.

Hihat: the open/closed check runs first (when `hihat_state` is
present in the override); then status. The hihat classifier
is server-side, so per-event classification override is moot
for hihat (use the cluster card for that). The override
applies `status` only for hihat.

**Files**:
- `webui/static/js/waveform.js`:
  - New `collectClassesForStem(stemType)` — returns the sorted
    unique `classification` values from `events_pga` for that
    stem.
  - New `getOverride(stemType, time_key)` / `setOverride(...)`
    — read/write the override record object.
  - Replace `toggleEventOverride` with `cycleEventOverride`
    that implements the cycle above.
  - Update `applyOverridesToEvents` to iterate
    `events_pga` (fixes Bug 1). This is the in-memory
    "the user has overrides in the JSON — apply them to the
    sidecar data the canvas is about to draw" path that runs
    on initial load and after a save round-trip.
  - Update `saveEventOverrides` to send the new shape.
- `webui/api/projects.py`:
  - `get_event_overrides` returns the new shape.
  - `save_event_overrides` accepts the new shape (just JSON
    pass-through, no validation).
- `user_files/*/midi/event_overrides.json` (existing files):
  - **DELETE** the old files. The user's "nuke the old files and
    start fresh" direction. (One file: project 6.)

### Phase 2: Fix the rebuild path (Python)

The override should be a **pre-filter veto** for `status` and
a **post-filter annotation** for `classification`.

- `stems_to_midi/rebuild_core.py`:
  - New `_apply_overrides(raw_events, overrides)` (replacing
    the old one) — iterates ALL `raw_events` (not just
    `pga_kept`). For each event, if the time is in overrides:
    - `event['status'] = override['status']` (veto — this
      sets the input the filter will see)
    - `event['_overridden'] = True`
  - The filter chain then runs on the raw events with the
    overridden statuses, but since the filter only operates
    on prominence / envelope_value / etc. (NOT status), the
    override's status is preserved through the filter.
  - The "Stamp filter_reason" loop (line 347) currently
    overwrites status based on `kept_times`/`filtered_times`.
    Fix: skip the re-stamp for overridden events (preserve
    the override's status).
  - The classification override (if present in the override
    record) is applied AFTER the filter, as a post-filter
    annotation — `classify_notes` runs with the stored
    `classification` if the override set one, otherwise it
    uses the default (k-means) result.

**Files**:
- `stems_to_midi/rebuild_core.py`:
  - Rewrite `_apply_overrides` to iterate ALL events (not just
    `pga_kept`). Mark each overridden event with
    `_overridden = True` so the downstream re-stamp can skip it.
  - In the "Stamp filter_reason" loop, skip overridden events
    (preserve their override status). For non-overridden
    events, keep the existing behavior.
  - The classification override (override['classification'])
    is applied at the end: after `classify_notes` runs the
    default k-means, walk the kept events and set
    `event.classification = override.classification` for any
    overridden event that has a classification in its
    override record. The note comes from the classification
    via the existing `_map_note` path.
- `stems_to_midi/rebuild_shell.py`:
  - `_load_overrides` reads the new shape (just `json.load`,
    no schema check needed since we own the file).
  - After `rebuild_events_from_analysis` returns, the
    overrides are still valid (they were applied during
    rebuild). No further processing needed here.

### Phase 3: Auto-cleanup at rebuild time

When the rebuild path completes, walk the overrides and the
sidecar. For each (stem, time) override:
1. Look up the sidecar event at the same time.
2. Compare the override's `status` to the sidecar event's
   natural status. If they match AND there's no classification
   override, drop the entry.
3. If the override has a classification that matches the
   sidecar's natural classification, drop the entry.
4. Write the cleaned dict back to disk.

The user's `event_overrides.json` stays intentionally minimal.

**Files**:
- `stems_to_midi/event_overrides.py`:
  - Add `clean_overrides(overrides, analysis_data, config)` —
    pure function, returns a new dict with stale entries
    dropped. Compare each (stem, time) override to the
    sidecar's natural state for that event.
  - Drop the load + schema check (the function is
    `clean_overrides` only; the file is read by
    `rebuild_shell._load_overrides`).
- `stems_to_midi/rebuild_shell.py`:
  - After `rebuild_events_from_analysis`, call
    `clean_overrides(overrides, analysis_data, config)`.
  - If the cleaned dict differs from the input, write it back
    to disk (so the file stays clean).
  - Return the cleaned dict in the API response so the WebUI
    can sync its in-memory state and toast a "N overrides
    cleaned up" message.

### Phase 4: Move the Save button out of the Tune panel

The button is at the bottom of `#tuning-panel`. Move a copy of
it to the right of the Tune button (next to the "Show Filtered"
toggle) at the top of the analysis section. Drive its
visibility from a session-dirty flag (overrides OR tuning
changes).

**Files**:
- `webui/templates/index.html`:
  - Add a new Save button at the top (sibling of the
    "Show Filtered" toggle and the Tune button). Make it
    always-rendered but hidden when the session is not dirty.
  - Remove the in-panel Save button (one way to do the
    same thing).
- `webui/static/js/threshold-tuning.js`:
  - Consolidate `updateTuningSaveButton` into a
    `updateSaveButton` that consults both the tuning-dirty
    flag AND `eventOverridesDirty` (exposed from
    waveform.js). Expose it on `window` so the front-end
    modules can call it.
- `webui/static/js/waveform.js`:
  - After every `cycleEventOverride`, call
    `window.updateSaveButton?.()`.

### Phase 5: Tests

**New pytest** in `stems_to_midi/tests/test_event_overrides.py`:
- `test_clean_removes_matching_overrides` — override matches
  sidecar → entry removed.
- `test_clean_keeps_differing_status` — override says KEPT,
  sidecar says FILTERED → entry kept.
- `test_clean_keeps_classification_override` — override sets
  classification that doesn't match the sidecar's natural
  → entry kept.
- `test_clean_keeps_when_event_missing_from_sidecar` — override
  at a time the sidecar no longer has → entry kept (or
  dropped; pick a side and document it).
- `test_clean_handles_old_string_format` — old
  `{time: "KEPT"}` entries are dropped after cleanup (we
  want them gone).

**New Playwright** `08-event-override-cycle.spec.ts`:
- Open project 6, switch to snare.
- Use `page.evaluate` to call `cycleEventOverride` directly
  with a known event time. Assert the override record has
  the right `status` and `classification`.
- Click an event on the canvas. Assert the cycle progresses:
  off → cls 0 → cls 1 → cls 2 → off.
- Save & Reconvert. Assert the sidecar's events_pga at the
  overridden times match the override's status and
  classification.
- Refresh the page. Assert the canvas reflects the override
  (Bug 1 fix).
- Verify the Save button at the top of the analysis section
  appears when there are unsaved changes (Phase 4 fix).

## Files Changed

1. `webui/static/js/waveform.js` — `cycleEventOverride`,
   `collectClassesForStem`, `getOverride`/`setOverride`,
   `applyOverridesToEvents` reads new shape + iterates
   `events_pga` (Bug 1 fix), `saveEventOverrides` sends new
   shape.
2. `webui/static/js/threshold-tuning.js` — `updateSaveButton`
   consolidates dirty flag, exported on `window`.
3. `webui/templates/index.html` — Save button at top of
   analysis section; remove in-panel Save button.
4. `webui/api/projects.py` — `get_event_overrides` returns new
   shape; `save_event_overrides` accepts new shape.
5. `webui/api/operations.py` — the rebuild endpoint
   surfaces the cleaned overrides in its response.
6. `stems_to_midi/rebuild_core.py` — `_apply_overrides` iterates
   ALL events (not just `pga_kept`); fix the "Stamp
   filter_reason" loop to skip overridden events.
7. `stems_to_midi/rebuild_shell.py` — call `clean_overrides`
   after rebuild; write the cleaned dict back to disk.
8. `stems_to_midi/event_overrides.py` — `clean_overrides` helper.
9. `user_files/*/midi/event_overrides.json` — DELETE existing
   files (per user direction).
10. `stems_to_midi/tests/test_event_overrides.py` (new).
11. `tests/playwright/specs/08-event-override-cycle.spec.ts`
    (new).

## Risks

- **The 3 bugs are real and serious.** The override currently
  has a "feels broken" UX (per the user's report) because
  FILTERED → KEPT overrides don't reach the MIDI, and on
  refresh the canvas doesn't reflect the override. The
  pre-filter veto fix is the right direction — it matches the
  user's mental model ("the user clicked this, keep it") and
  it generalizes to classification.
- **Auto-cleanup surprise**: a user who toggles a single
  event, then changes the threshold so that event's natural
  state becomes the override's state, will see the override
  entry disappear on next Save. The user might be surprised.
  Mitigation: surface the cleanup in the API response so the
  WebUI can toast a "N overrides cleaned up" message.
- **Schema migration in production**: per user direction, no
  migration. We delete the existing `event_overrides.json`
  files in `user_files/*/midi/`. The new code writes the new
  shape. Anything relying on the old shape is broken — but
  nothing should be, because the override system was barely
  functional to begin with.
- **Click cycle UX edge case**: if a user has the panel open
  and is iterating cluster notes via the cluster cards, the
  per-event classification override might conflict with the
  per-class classification override. Resolution: per-event
  wins (it was set later, more specific). Document in the
  override record's `reason` field.

## Success Criteria

1. Click an event on snare → cycles through off → cls 0 →
   cls 1 → cls 2 → off.
2. The override is persisted to `event_overrides.json` as
   `{status, classification?}` (not a string).
3. The Save button at the top of the analysis section shows
   when there are unsaved changes (overrides OR tuning) and
   is hidden otherwise.
4. Save & Reconvert applies the override to the sidecar and
   the MIDI. The sidecar's events_pga at the overridden times
   now has the override's status and classification. The MIDI
   reflects the override.
5. After Save & Reconvert, the override file is auto-cleaned:
   any entry whose status+classification now matches the
   sidecar's natural value is removed.
6. **Bug 1 fixed**: refresh the page, the canvas reflects the
   override.
7. **Bug 2 fixed**: toggling a FILTERED event to KEPT and
   saving produces a MIDI with the event as KEPT.
8. **Bug 3 fixed**: the sidecar's events_pga at the overridden
   times shows the override's status after Save.
9. The existing `TestEventOverridesRoute` in
   `webui/test_api.py` is updated to cover the new shape.
   New pytest covers `clean_overrides`. New Playwright covers
   the click cycle and refresh.

## Out of scope (follow-up ideas, not this commit)

- **Per-event note override** (the user mentioned "set the
  note"). The current cluster-card system overrides the WHOLE
  class. A per-event note would let the user remap individual
  events to a different MIDI note. This is straightforward to
  add (the override record already has space for a `note`
  field) but it's a separate feature. Filed for follow-up.
- **Bulk override** (toggle all events in a time range) — not
  requested, not in scope.
- **Hihat per-event open/closed toggle** — hihat classification
  is server-side, so per-event `hihat_state` override is out
  of scope until that's redesigned.

## Goals

The user wants the override system to:
1. **Cycle classifications on click** — off → cls 0 → cls 1 →
   … → off. For single-class stems, behave as today (toggle).
2. **Track classification in `event_overrides.json`** so the
   override persists across reloads and is reflected in the
   rebuilt MIDI without any further user action beyond Save.
3. **Auto-cleanup** when an override matches the sidecar's
   current state for that event. Avoid the "stale shadow"
   failure mode where a user moves a slider and the override
   silently wins over the new threshold.
4. **Move the Save button out of the Tune panel** so users
   don't have to open Tune to commit their toggles. The button
   should appear when there are unsaved overrides OR unsaved
   tuning changes (it's already wired for the tuning case).
5. **Decide the reprocess UX** — when does toggling an event
   regenerate the MIDI? Options:
   - (A) Every click → debounced MIDI rebuild (cheap rebuild,
     but still a round-trip per click).
   - (B) Save button click only (current behavior, but the
     button is hidden in the Tune panel).
   - (C) Click → debounced JS-side state update + Save
     button lights up → user clicks Save → MIDI regenerates.
   - **Recommendation: C** — the click is instant (no latency),
   the in-memory waveform and sidecar data are updated
   client-side, the override is debounced-saved to JSON, the
   Save button shows a dirty badge. The MIDI regenerates on
   Save. This matches the existing "click is local, Save is
   server" model and keeps the round-trip explicit.

## Approach

### Phase 1: Schema migration (no behavior change yet)

Update `event_overrides.json` to support per-event metadata. The
new shape is a superset of the old one — old files are read
transparently, old writes still work.

**File shape (new)**:
```json
{
  "snare": {
    "0.5267": {
      "status": "KEPT",
      "classification": 1
    },
    "1.2345": {
      "status": "FILTERED"
    }
  }
}
```

That is: `{stem: {time_str: {status, [classification], [note]}}}`. All
fields except `status` are optional. Old-format files
(`{stem: {time_str: "KEPT"}}`) are still valid — the new loader
coerces the string value to `{status: <string>}`.

**Files**:
- `stems_to_midi/event_overrides.py` — add `migrate_overrides()`
  that walks the dict and coerces string values to dicts. The
  schema check in `load_event_overrides` updates to accept
  either form. Add `clean_overrides(overrides, sidecar)` that
  drops any override entry whose effective state matches the
  sidecar (this is the "auto-cleanup" hook).
- `webui/api/projects.py` — `save_event_overrides` accepts the
  new shape. The existing test in
  [`webui/test_api.py:401`](webui/test_api.py) (the
  `TestEventOverridesRoute` class) gets extended to cover both
  shapes.

### Phase 2: Click-to-cycle in the front-end

Replace the binary toggle in `toggleEventOverride` with a
classification-aware cycle. Pseudocode:

```javascript
function cycleEventOverride(stemType, event) {
    const sidecarEvent = findSidecarEvent(stemType, event);  // by time
    const classes = collectClassesForStem(stemType);  // 0..N-1 from sidecar
    const override = getOverride(stemType, event);
    const currentClass = override?.classification
                        ?? event.classification
                        ?? null;
    const isCurrentlyKept = (override?.status ?? event.status) === 'KEPT';

    // Build the next state:
    //   off → cls (smallest non-null class, usually 0 or 1)
    //   cls i → cls i+1 (if there's a next class) OR off
    //   cls N (last) → off
    let nextStatus, nextClass;
    if (!isCurrentlyKept) {
        // Turning on. Default to cls 0 if no classes, else cls 0
        // (snare's "Type 1" is cls 0 — 0-indexed — but the user
        // calls it "class 1" in 1-indexed UI terms; we'll use
        // cls 0 as the first on-state, matching the cluster card
        // labels).
        nextStatus = 'KEPT';
        nextClass = classes.length > 0 ? Math.min(...classes) : null;
    } else if (currentClass == null) {
        // No class on this event. Cycle off.
        nextStatus = 'FILTERED';
        nextClass = null;
    } else {
        // Find next class index. If we're at the highest
        // classification, cycle off.
        const idx = classes.indexOf(currentClass);
        if (idx === -1) {
            // current class isn't in the sidecar's class set
            // anymore (slider changed). Fall back to the lowest.
            nextStatus = 'KEPT';
            nextClass = classes.length > 0 ? Math.min(...classes) : null;
        } else if (idx === classes.length - 1) {
            // At the highest class. Cycle off.
            nextStatus = 'FILTERED';
            nextClass = null;
        } else {
            // Advance to the next class.
            nextStatus = 'KEPT';
            nextClass = classes[idx + 1];
        }
    }

    // Update in-memory state
    event.status = nextStatus;
    event.classification = nextClass;
    event._overridden = true;

    // Persist override
    setOverride(stemType, event, {status: nextStatus, classification: nextClass});
    scheduleOverrideSave();
    drawWaveform();
}
```

Edge cases:
- **Stem with no classes** (kick, single-class toms): cycle
  KEPT↔FILTERED exactly as today. The cycle is "off ↔ on",
  2 states.
- **Stem with 1 class** (hihat with single cluster result):
  cycle has 2 states — off, cls 0. After 1 click, off. After
  2 clicks, back to cls 0. Same UX as no-classes.
- **Stem with N classes** (snare/cymbals/toms with k-means):
  cycle has N+1 states — off, cls 0, cls 1, …, cls (N-1).
- **Hihat with hihat_state** (open/closed): cycle handles
  `hihat_state` first (open ↔ closed), then KEPT↔FILTERED.
  Currently the hihat classifier is server-side, so the
  override's classification field doesn't apply; only status.
  (We can revisit if hihat gets a per-event open/closed toggle
  later.)

Files:
- `webui/static/js/waveform.js`:
  - New `collectClassesForStem(stemType)` helper — walks
    `waveformAnalysisData.stems[stem].events_pga` and returns
    the sorted unique `classification` values.
  - New `getOverride(stemType, event)` / `setOverride(stemType,
    event, partial)` — read/write the override record (now an
    object, not a string).
  - Replace `toggleEventOverride` with `cycleEventOverride`.
  - Update `applyOverridesToEvents` to handle the new shape
    (already works if we just store the override record
    object and apply `.status` and `.classification` to the
    event).
  - Update `saveEventOverrides` to send the new shape (object
    per time_key).

### Phase 3: Move the Save button out of the Tune panel

Currently the button is at the bottom of `#tuning-panel`. The
user's request: put it "out to the top by the text on the
right." Looking at the layout, the right side of the stem
tab row already has the "Show Filtered" toggle and the "Tune"
button. The Save button can sit between them (or replace the
filter toggle's neighbor).

**File** — `webui/templates/index.html`:
- Move the Save button (or a new copy of it) to the right of
  the Tune button.
- Make it visible whenever there are unsaved changes (overrides
  OR tuning-slider moves). Rename the underlying condition
  from "tuning is dirty" to "session is dirty" — show this
  button when `eventOverridesDirty || tuningDirty` (or
  equivalent).
- The Save button inside the Tune panel can either stay
  (for users who want to use the panel as a context) or be
  removed. **Recommendation: remove the in-panel one** to avoid
  two ways to do the same thing.

**File** — `webui/static/js/threshold-tuning.js`:
- The "session is dirty" flag already exists implicitly
  (tuningDirty via `updateTuningSaveButton`, and
  `eventOverridesDirty` in waveform.js). Move the
  enable/disable logic to a single function `updateSaveButton`
  that consults both flags. Drive the new top-of-waveform Save
  button's visibility from it.

**File** — `webui/static/js/waveform.js`:
- After every `cycleEventOverride`, call `updateSaveButton()`
  (export it from threshold-tuning.js or move the function
  to a shared module).

### Phase 4: Auto-cleanup at rebuild time

When the rebuild path completes, walk the merged overrides and
the sidecar. For each (stem, time) override:
1. Look up the sidecar event at the same time.
2. If the override's `status` and `classification` both match
   the sidecar event's natural values, drop the override entry.
3. Write the cleaned overrides back to disk.

This means the user's `event_overrides.json` stays
intentionally minimal — only entries that override a non-default
state. When the user lowers a threshold and the sidecar's
natural state for an event flips to FILTERED, the override is
no longer needed and gets removed on the next Save.

**File** — `stems_to_midi/rebuild_shell.py`:
- After `rebuild_events_from_analysis` returns the updated
  analysis, call a new `clean_overrides(overrides, analysis_data,
  config)` helper that returns the cleaned dict.
- Write the cleaned dict to `event_overrides.json` (only if
  something changed — avoid touching the file on no-op saves).
- The cleaned dict is also returned in the API response so the
  WebUI can sync its in-memory state.

**File** — `stems_to_midi/event_overrides.py`:
- Add `clean_overrides(overrides, analysis_data)` that walks
  each stem, each time, and returns a new dict with stale
  entries dropped. Pure function.
- Add unit tests for the new shape (string vs dict) and the
  cleanup logic.

**File** — `webui/api/projects.py`:
- The PUT endpoint accepts the new shape (already does, since
  the body is just JSON-serialized).
- Optionally add a GET-cleaned endpoint that returns the
  post-cleanup overrides so the WebUI can verify its view.

### Phase 5: Tests

**New pytest** in `stems_to_midi/tests/test_event_overrides.py`:
- `test_load_old_format` — old string values are coerced to
  `{status: "KEPT"}` shape.
- `test_load_new_format` — new dict values are passed through.
- `test_clean_removes_matching_overrides` — override matches
  sidecar → entry removed.
- `test_clean_keeps_differing_overrides` — override differs
  from sidecar → entry kept.
- `test_clean_keeps_classification_override` — override has a
  classification that matches a slider-derived value, entry
  kept.
- `test_clean_handles_missing_event` — override at a time that
  no longer exists in the sidecar → entry removed (stale).

**Extend** `webui/test_api.py::TestEventOverridesRoute`:
- Round-trip the new shape via GET/PUT.
- Verify that PUT with the new shape returns 200.
- Verify that PUT with the old shape still works (back-compat).

**New Playwright** `08-event-override-cycle.spec.ts`:
- Open project 6, switch to snare.
- Use `evaluate()` to call `cycleEventOverride` directly with a
  known event time. Assert the override record has the right
  `status` and `classification`.
- Click an event on the canvas. Assert the cycle progresses:
  off → cls 0 → cls 1 → cls 2 → off.
- Save & Reconvert. Assert the sidecar's events_pga at the
  overridden times match the override's status and
  classification.

## Files Changed

1. `stems_to_midi/event_overrides.py` — `migrate_overrides`,
   `clean_overrides`, schema-check update, unit tests.
2. `stems_to_midi/rebuild_shell.py` — call `clean_overrides` after
   rebuild, write cleaned dict, surface to API.
3. `stems_to_midi/rebuild_core.py` — `apply_overrides` reads the
   new shape (override record) and applies both `status` and
   `classification`.
4. `webui/api/projects.py` — `get_event_overrides` returns the
   new shape; `save_event_overrides` accepts it. `rebuild-midi`
   endpoint passes overrides through unchanged.
5. `webui/api/operations.py` — the rebuild endpoint includes
   the cleaned overrides in its response so the WebUI can sync.
6. `webui/templates/index.html` — Save button at the top right
   of the analysis section, driven by session-dirty flag.
7. `webui/static/js/waveform.js` — `cycleEventOverride`,
   `collectClassesForStem`, `getOverride`/`setOverride` helpers,
   `applyOverridesToEvents` reads the new shape.
8. `webui/static/js/threshold-tuning.js` — `updateSaveButton`
   consolidates the dirty flag for both tuning and overrides.
9. `stems_to_midi/tests/test_event_overrides.py` (new) — unit
   tests for the new shape and clean function.
10. `webui/test_api.py` — extend `TestEventOverridesRoute`.
11. `tests/playwright/specs/08-event-override-cycle.spec.ts`
    (new) — end-to-end click cycle + Save & Reconvert test.

## Risks

- **Auto-cleanup surprise**: a user who toggles a single
  event, then changes the threshold so that event's natural
  state becomes the override's state, will see the override
  entry disappear on next Save. The user might be surprised.
  Mitigation: surface the cleanup in the API response so the
  WebUI can toast a "N overrides cleaned up" message.

- **Schema migration in production**: any project with an
  old-format `event_overrides.json` will be migrated on next
  read. The migration is lossless (string values are coerced to
  `{status: <string>}`). No risk of data loss.

- **Apply-path bug carryover**: the current
  `applyOverridesToEvents` in waveform.js only iterates
  `events_configured || events_sensitive`, NOT `events_pga`.
  For PGA-only stems (the entire post-2026-06-15 refactor
  world), the in-UI display of an override is broken until a
  Save & Reconvert completes. This pre-existing bug should be
  fixed in Phase 1: change the function to also iterate
  `events_pga` (or just iterate whichever list exists).

- **Click cycle on a stem with N classes and a "off" entry the
  user just typed** — the cycle always goes through the full
  class list. If the user wants a specific class quickly, they
  have to click multiple times. The cluster cards in the Tune
  panel already let the user remap WHOLE classes at once
  (cls 0 → note 37). The per-event cycling is for fine-grained
  control. The two systems are complementary.

- **Save & Reconvert is still expensive** (sub-second for project
  6, but can grow with project length). The user is opting in
  to a server round-trip every time they commit changes. This
  is acceptable — the override is per-event, the rebuild is
  fast for the sidecar-only path.

## Success Criteria

1. Click an event on snare → cycles through off → cls 0 → cls 1
   → cls 2 → off.
2. The override is persisted to `event_overrides.json` as an
   object `{status, classification}` (not a string).
3. The Save button at the top of the analysis section shows
   when there are unsaved changes (overrides OR tuning) and
   is hidden otherwise.
4. Save & Reconvert applies the override to the sidecar and
   MIDI. The sidecar's events_pga at the overridden times now
   has the override's status and classification.
5. After Save & Reconvert, the override file is auto-cleaned:
   any entry whose status/classification matches the sidecar's
   natural value is removed.
6. Existing tests (`TestEventOverridesRoute` in
   `webui/test_api.py`) continue to pass. New pytest covers
   the migration, the clean function, and the new shape.

## Out of scope (follow-up ideas, not this commit)

- **Per-event note override** (the user mentioned "set the
  note"). The current cluster-card system overrides the WHOLE
  class. A per-event note would let the user remap individual
  events to a different MIDI note. This is straightforward to
  add (the override record already has space for a `note`
  field; the rebuild path would just need to use it instead of
  the cluster-default note) but it's a separate feature. Filed
  for follow-up.
- **Visual indicator on the canvas for override status** —
  the current `_overridden` flag already exists, and
  `drawEventBars` already draws a white diamond at the top of
  overridden bars. This works today; no change needed.
- **Bulk override** (toggle all events in a time range) —
  not requested, not in scope.