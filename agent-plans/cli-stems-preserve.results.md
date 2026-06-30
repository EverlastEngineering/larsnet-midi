# Results: Preserve Other Stems When `--stems <subset>` Is Used

## Status

**Complete** — 2026-06-30

## Phase 1: Implementation

- [x] Add `_deserialize_sidecar_stems_for_merge` helper to `stems_to_midi_cli.py`
- [x] Modify `_process_stems_to_midi` to load existing sidecar and merge preserved stems

## Phase 2: Tests

- [x] Unit test: empty sidecar → empty output
- [x] Unit test: preserves KEPT events only
- [x] Unit test: extracts MIDI event fields correctly
- [x] Unit test: hihat_state preserved
- [x] Unit test: filters to requested stems only
- [x] Unit test: clamps duration to max_note_duration (global)
- [x] Unit test: clamps duration to max_note_duration (per-stem wins)
- [x] Unit test: preserves extras in pga_onset_data
- [x] Unit test: no config uses safe defaults (0.5s)
- [x] Unit test: missing duration_ms falls back to default_note_duration

## Phase 3: E2E verification

- [x] Full conversion → --stems snare → other stems' events_pga counts unchanged
- [x] CLI prints "Preserving 4 non-reprocessed stem(s)..." message
- [x] MIDI contains all 5 stems after --stems snare (kick 190, hihat 795, toms 10, cymbals 13, snare 247)

## Decision Log

- **CLI is the right layer.** `save_analysis_sidecar` is correctly pure-write
  — no merge logic. The CLI is the only place that knows `--stems` was used.
  Both surfaces (CLI `__main__` and WebUI's `run_stems_to_midi`) call
  `_process_stems_to_midi`, so the fix benefits both.
- **No new schema/config.** Existing fields (events_pga, note, midi_velocity,
  duration_ms, hihat_state) already carry everything needed for round-trip
  reconstruction.
- **Pass midi_path to load_analysis_sidecar, not .analysis.json path.** First
  attempt passed `midi_path.with_suffix('.analysis.json')` — but the loader
  applies with_suffix internally, so it ended up looking for
  `.analysis.analysis.json` and returned None silently. Caught with
  debug-print tracing; fixed to pass the .mid path directly. Added a comment
  in the code so this doesn't regress.
- **Helper must be defined BEFORE `__main__` block.** First attempt placed the
  helper after the `if __name__ == '__main__':` block (alongside
  `_load_project_config_for_project`, which is only used by WebUI's
  importlib-loaded module). The `__main__` block runs at module-load time
  when the file is invoked directly, BEFORE later def statements execute.
  Result: `NameError: _deserialize_sidecar_stems_for_merge is not defined`.
  Moved the helper before `__main__` (alongside `_build_argparser`). The
  helper's docstring documents this trap for future contributors.
- **Learning mode excluded from merge.** `--learn` changes per-event semantics
  (velocity=1 for FPs, etc.) so merging the pre-learn sidecar would corrupt
  the learning-mode output. The merge is gated on `not learning_mode`.

## Metrics

- **Files changed**: 1 source + 1 new test file + 1 plan/results markdown + 1 bug-tracking append
- **Lines**: ~155 in source (mostly the helper + comments), ~270 in new test file, ~115 in plan/results
- **Tests added**: 10 (all pass)
- **Pytest**: 855 passed (up from 845 — the 10 new tests), 4 pre-existing
  failures (unchanged — cymbals kept=6 vs 48 ground-truth, etc.)

## End-to-end on project 6

| Step | Sidecar stems | events_pga counts (kick/toms/hihat/cymbals/snare) |
|------|---------------|---------------------------------------------------|
| Full conversion baseline | 5 | 2087 / 2246 / 1307 / 776 / 1450 |
| After `--stems snare` | 5 (preserved) | 2087 / 2246 / 1307 / 776 / 1450 (snare re-processed, identical count) |
| MIDI total notes after | — | 1256 events across 10 pitches (kick 36=190, snare 37/38/39=247, hihat 42/46=795, toms 45/47=10, cymbals 49=13) |

The user's exact reproduction case (full conversion → `--stems snare`) now
preserves the other 4 stems' sidecar data and MIDI events. The bug is fixed.