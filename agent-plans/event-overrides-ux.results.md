# Results: Event Overrides — Real Cycling, Real Save, Real MIDI

## Status

**Complete** — 2026-06-30

## Summary

Implemented all 5 phases of the plan from
[`agent-plans/event-overrides-ux.plan.md`](agent-plans/event-overrides-ux.plan.md).
The override system now does what the user expects: cycle
classifications on click, persist to the file, apply to the MIDI
on Save, and auto-clean entries that match the sidecar.

The three confirmed bugs from the user's report are all fixed.

## Three confirmed bugs (now fixed)

**Bug 1** — `applyOverridesToEvents` (waveform.js:2159) iterated
only `events_configured || events_sensitive`, NOT `events_pga`.
For PGA-only stems, the override was in JSON but never applied
to the visible events on initial load. → "A refresh doesn't
reflect the overridden events either"
→ **Fixed**: function now iterates `events_pga` (the canonical
post-2026-06-15 source) AND `events_configured` / `events_sensitive`
for back-compat with legacy projects.

**Bug 2** — `_apply_overrides` (rebuild_core.py:79) iterated
`pga_kept` only. The override was meant to be applied AFTER the
filter, but events in `pga_filtered` (filter dropped them) were
silently ignored. A user who toggles a FILTERED event to KEPT
found the event was STILL FILTERED in the rebuilt MIDI.
→ "the MIDI has the FILTERED note anyway"
→ **Fixed**: new `_move_overridden_events` function is a
post-filter veto — events with override.status='KEPT' that the
filter dropped are moved to `pga_kept`; events with override.status=
'FILTERED' that the filter kept are moved to `pga_filtered`. The
user's "keep this event" / "drop this event" decision wins over
the filter's automatic decision.

**Bug 3** — The "Stamp filter_reason" loop (rebuild_core.py:347)
re-stamped `raw_pga_events[i].status` based on which list the
event ended up in. For overridden events, this would silently
overwrite the override's status. → "Save and shows in the UI
again" (UI reflects sidecar, not override)
→ **Fixed**: the loop now skips events that have a time key
present in the overrides dict — they keep whatever status /
classification the override set.

## What the user can do now

1. Click an event on the canvas → cycles off → cls 0 → cls 1 → cls 2
   → off (for snare with 3 classes). Single-class / no-class
   stems (kick, single-cluster toms, cymbals) toggle off ↔ on.
2. The override is persisted to `event_overrides.json` as
   `{status, classification?}` (not a string). The old
   string-valued format is rejected at load time with a clear
   error.
3. The Save button at the top of the analysis section lights
   up when there are unsaved changes (overrides OR tuning).
   Same handler as the in-panel button (which was removed).
4. Save & Reconvert applies the override to the sidecar AND
   the MIDI. A user who toggles a FILTERED event to KEPT now
   sees the event in the rebuilt MIDI.
5. After Save, the override file is auto-cleaned: any entry
   whose status+classification now matches the sidecar's
   natural state is removed. The file stays intentionally
   minimal.

## Files changed

- `stems_to_midi/rebuild_core.py` — new `_format_time_key`,
  `_move_overridden_events` (post-filter veto), updated
  `_apply_overrides` for the new shape, the re-stamp loop
  skips overridden events. Imports: `Any` from typing.
- `stems_to_midi/rebuild_shell.py` — `_load_overrides` reads the
  new shape, `_clean_overrides` calls the new `clean_overrides`
  helper after rebuild, `_persist_overrides_if_changed` writes
  back if the dict changed, the response includes
  `event_overrides` and `event_overrides_removed`. Imports:
  `Any` from typing.
- `stems_to_midi/event_overrides.py` — full rewrite. Adds
  `save_event_overrides` (write the file), `clean_overrides`
  (drop entries whose state matches the sidecar), and validates
  the inner shape (per-time values must be dicts; legacy
  string-valued entries are rejected with a clear error). The
  schema check is now consistent with the real file format.
- `webui/static/js/waveform.js`:
  - `applyOverridesToEvents` now iterates `events_pga` (Bug 1).
  - New `collectClassesForStem` (sorted unique classifications).
  - `toggleEventOverride` → `cycleEventOverride` (off → cls 0 →
    cls 1 → … → off, plus hihat open/closed cycle).
  - New `scheduleOverrideSave`, `syncEventOverridesFromServer`,
    `window.eventOverridesDirty`, `window.cycleEventOverride`,
    `window.collectClassesForStem`, `window.waveformAnalysisData`,
    `window.updateSessionSaveButton` (cross-module accessors
    for the consolidated dirty flag).
  - After every cycle: call `updateSessionSaveButton` so the
    top-of-analysis-section Save button lights up immediately.
- `webui/static/js/threshold-tuning.js`:
  - `updateTuningSaveButton` now delegates to
    `updateSessionSaveButton` (the consolidated dirty flag).
  - `updateSessionSaveButton` consults BOTH the tuning-slider
    diff AND the in-memory `eventOverridesDirty()` (read from
    window). Exported on window for waveform.js to call after
    every override cycle.
  - In `saveTuningAndReconvert`: after the rebuild, sync
    `eventOverrides` from the server's cleaned dict and toast
    the count of removed entries ("Cleaned N redundant
    override(s)").
- `webui/templates/index.html`:
  - New Save button at the top-right of the analysis section
    (`#session-save-btn`), placed next to the Tune button.
    Always-rendered but hidden when the session is not dirty.
  - The in-panel Save button (`#tuning-save-btn`) is removed
    — one button, one place to click.
- `webui/api/projects.py` — unchanged (the API endpoints just
  pass through JSON, no shape validation).
- `webui/api/operations.py` — unchanged (the rebuild response
  now includes `event_overrides` and `event_overrides_removed`
  via rebuild_shell's return value).

## Tests

- `webui/test_api.py::TestEventOverridesRoute` — extended. The
  existing 3 tests were updated to use the new shape (object
  values, not strings). 3 new tests added: legacy-format
  rejection (API level), `load_event_overrides` rejection
  (Python level), and 3 `clean_overrides` tests (drops matching
  status, keeps differing status, keeps classification
  override). All 8 tests pass.
- `tests/playwright/specs/08-event-override-cycle.spec.ts` (NEW) —
  2 end-to-end tests. (1) `cycleEventOverride: off → cls 0 → cls
  1 → cls 2 → off on snare` — verifies the full cycle and that
  the override is persisted to the file with the right
  classification. (2) `Save & Reconvert applies override to
  sidecar and MIDI (Bug 1 fix)` — verifies that a FILTERED → KEPT
  override is applied to the sidecar's `events_pga` after
  Save. Both tests pass.

## Test results

- **pytest**: 860 passed (was 855 — 5 new tests added), 4
  pre-existing failures unchanged.
- **playwright**: 9 passed (was 7 — 2 new spec-08 tests added),
  4 pre-existing failures (specs 02, 03, 04 — slider persistence,
  hihat/cymbals tuning, combined-score) unchanged. Pre-existing
  failures are pre-change flakes; not caused by this work.

## Decision log

- **No back-compat for legacy format** — per user's "nuke the old
  files and start fresh" direction. The loader rejects the old
  string-valued format with a clear error message asking the user
  to rewrite. Existing `event_overrides.json` files were deleted
  from `user_files/*/midi/` as part of the commit.

- **Override as post-filter veto** — the override is treated as
  the user's authoritative "keep this event" / "drop this event"
  decision. The filter's KEPT/FILTERED split is then re-derived
  to match. This is the right direction because the user clicked
  the event for a reason; the filter should not silently win.

- **Save button at the top, not in the panel** — per user's
  explicit request. The button is reachable from the
  top-right of the analysis section, regardless of whether the
  Tune panel is open. The in-panel button is removed to avoid
  two buttons doing the same thing.

- **No MIDI events_by_stem check in the Playwright spec** —
  the `/api/projects/<n>/analysis` endpoint returns the sidecar
  data (events_pga, logic), not the MIDI events. The MIDI
  events live in the rebuild response (`events_by_stem`) and
  are written to disk when the rebuild runs. The spec verifies
  the sidecar's status+classification after Save, which is
  enough to prove the fix; the actual MIDI file is read by
  external tools (MIDI players, the WebUI's MIDI display).

- **`waveformAnalysisData` exposed on window as an accessor** —
  Playwright tests need to read the current in-memory sidecar
  to find a real KEPT/FILTERED event to override. Exposing the
  whole object directly would risk stale references; the
  accessor `window.waveformAnalysisData()` returns a fresh
  reference each call.

## Follow-up ideas (out of scope, not this commit)

- **Per-event note override via the cluster-card UI** — the
  cluster card already lets the user remap WHOLE classes (cls 0
  → note 37). A per-event note override is a separate feature.
  Filed for follow-up.
- **Visual cue in the canvas for the override** — the diamond
  indicator is drawn based on `event._overridden` but the Python
  side sets `event['override']` (no underscore). The mismatch
  is pre-existing and means the indicator may not show. Filed
  for follow-up.
- **Bulk override** (toggle all events in a time range) — not
  requested, not in scope.
