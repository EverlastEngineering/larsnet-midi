# Rebuild-from-Analysis Pipeline — Results

**Plan**: `rebuild-from-analysis.plan.md`  
**Started**: 2026-02-13  
**Branch**: `daw-waveform-detection`

## Phase 1: Rebuild Backend ✅

- [x] `stems_to_midi/rebuild_core.py` — Pure functional core (408 lines)
  - `_merge_event_pools()` — Merges configured + sensitive pools, deduplicates by time window
  - `_apply_overrides()` — Applies manual include/exclude from event_overrides.json
  - `_refilter_events()` — Re-runs `should_keep_onset()` with current thresholds, skips overridden events
  - `_events_to_midi()` — Converts kept events to MIDI dicts with velocity normalization, note resolution
  - `rebuild_events_from_analysis()` — Main entry point, validates v3 format, deep-copies, returns updated analysis + MIDI events
- [x] `stems_to_midi/rebuild_shell.py` — I/O shell (170 lines)
  - `rebuild_midi_for_project()` — Loads analysis/config/overrides, calls core, writes MIDI + analysis.json
- [x] `stems_to_midi/__init__.py` — Exports added
- [x] `stems_to_midi/test_rebuild_core.py` — 30 tests passing
  - Merge pool tests (6): configured-only, sensitive-only, dedup, non-overlapping, empty, sorted
  - Override tests (4): KEPT, FILTERED, no-match, empty
  - Refilter tests (6): below-threshold, promote, override-survives-strict, override-survives-permissive, no-threshold, require_both
  - Integration tests (8): basic rebuild, threshold up/down, overrides through rebuild, per-stem, sensitive promotion, updated analysis statuses, does-not-mutate, multi-stem
  - Error tests (4): version mismatch, empty analysis, no stems, missing stem

## Phase 2: API Endpoint + UI Integration ✅

- [x] `POST /api/rebuild-midi` in `webui/api/operations.py` — Synchronous endpoint (no job queue), returns 200 with analysis_data on success, 409 if full pipeline needed
- [x] `api.rebuildMidi()` in `webui/static/js/api.js`
- [x] `saveTuningAndReconvert()` updated in `webui/static/js/threshold-tuning.js`:
  - Tries rebuild first (sub-second)
  - On success: updates `waveformAnalysisData` in place, re-renders waveform, no page refresh
  - On failure/409: falls back to full pipeline via `stemsToMidi()` + job queue
- [x] Cache-busting version bumps: api.js v22, threshold-tuning.js v22

## Test Results

- 802 passed, 8 skipped, 0 failures
- 2 pre-existing flaky tests excluded (synthetic audio onset detection)
- 30 new rebuild-specific tests all passing

## Phase 3: Event Override Integration

- [ ] Not started

## Phase 4: Dynamic Parameter Exposure

- [ ] Not started
