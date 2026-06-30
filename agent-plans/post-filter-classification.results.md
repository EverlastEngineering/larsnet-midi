# Results: Wire Post-Filter Note Classification + Spread Guardrail

## Status

**Complete** — 2026-06-30

## Phase 1: Wiring

### 1a — Live path (`processing_shell_percentile_gated.py`)
- [x] Replace hihat-only special case with generic `classify_notes`
- [x] Change MIDI loop to use `ev.get('note') ?? stem_note`

### 1b — Rebuild path (`rebuild_core.py`)
- [x] Change `ev_note = stem_note` to use `ev.get('note')`

### 1c — End-to-end on project 6
- [x] Re-run live pipeline (CLI: `python stems_to_midi_cli.py 6 --stems toms`)
- [x] Verify sidecar `classification` and `note` non-None for KEPT events
- [x] Verify MIDI has ≥ 2 distinct notes for toms

## Phase 2: Spread guardrail

### 2a — Functional core (`note_classification_core.py`)
- [x] Add `_has_sufficient_spread(values, relative_threshold)` helper
      (relative IQR = IQR / median — unitless, works across features)
- [x] Gate `classify_tom_notes` with `min_cluster_spread_ratio` (default 0.10)
- [x] Gate `classify_snare_notes` with `min_cluster_spread_ratio` (default 0.10)
- [x] Gate `classify_cymbal_notes` with `min_cluster_spread_ratio` (default 0.10)

### 2b — Schema (`webui/settings_schema.py`)
- [x] Add `toms.min_cluster_spread_ratio` SettingDefinition (3 settings total)
- [x] Add `snare.min_cluster_spread_ratio` SettingDefinition
- [x] Add `cymbals.min_cluster_spread_ratio` SettingDefinition

## Phase 4: Tests

- [x] `TestSpreadGuardrail` — flat / close-but-distinct / well-spread / unitless fixtures
- [x] `classify_tom_notes` close-pitch case → single classification (66.1–66.9 Hz)
- [x] `classify_tom_notes` well-separated case → 2+ classifications (70 vs 130 Hz)
- [x] `classify_tom_notes` custom threshold (0.0001) disables guardrail
- [x] `classify_snare_notes` close-width case → single classification
- [x] `classify_snare_notes` bimodal-width case → 2 classifications
- [x] `classify_cymbal_notes` close-centroid case → single classification
- [x] `classify_cymbal_notes` well-separated centroid case → 2 classifications
- [x] Full pytest run (842 passed, 4 pre-existing failures)

## Decision Log

- **Spread metric (revised)**: Started with absolute IQR (5.0 Hz for toms),
  but absolute IQR breaks when the user picks a non-default `cluster_feature`
  (e.g. `toms.cluster_feature='stereo_width'` has values 0..1, not Hz).
  Switched to relative IQR (`IQR / median`) — unitless, works across any
  feature. One threshold per stem (default 0.10 = 10% of median).
- **Default threshold**: 0.10 (10% of median). On project 6, IQR/median
  for the 10 KEPT toms = 0.21, well above 0.10 — real split not blocked.
  On the user's hypothetical "10 notes at 66.1-66.9 Hz", IQR/median ≈ 0.006,
  far below 0.10 — guardrail triggers, all 10 collapse to mid tom.
- **Single-value handling**: Helper returns True for n_unique < 2
  (one population). Lets the existing `_cluster_values` single-value
  path produce classification=0 (matches pre-guardrail behavior).
- **Outlier merge when guardrail triggers**: classification=1 (mid) for toms,
  0 for snare/cymbals — matches each function's empty-data default.
- **Scope**: All non-hihat stems (toms, snare, cymbals). Hihat already
  classified correctly via the existing special case.
- **Phase 3 (`expected_clusters+1` merge)**: Deferred. Spread guardrail
  addresses the user's concrete concern; the merge strategy is a
  research direction that should not be conflated with the bug fix.

## Metrics

- **Bug fix**: 6 files changed (4 source, 1 test, 1 schema), 1 plan file,
  1 results file.
- **Tests added**: 14 new tests in `test_note_classification_core.py`
  (TestSpreadGuardrail: 7, TestClassifyTomNotesSpread: 3,
  TestClassifySnareNotesSpread: 2, TestClassifyCymbalNotesSpread: 2).
- **Pre-existing test failures (unchanged)**: 4 total
  (`test_process_stem_returns_events`, `test_midi_file_created`,
  `test_multiple_stems_combined`, `test_ground_truth_midi_vs_pga_loose`).
  All are cymbals-related or legacy energy-path; none caused by this work.

## Test Results

End-to-end on project 6 (`python stems_to_midi_cli.py 6 --stems toms`):

| Metric | Before | After |
|--------|--------|-------|
| KEPT events with `classification` non-None | 0 / 10 | 10 / 10 |
| KEPT events with `note` non-None | 0 / 10 | 10 / 10 |
| Distinct MIDI notes in toms output | 1 (47 only) | 2 (45, 47) |
| IQR / median on pitch_hz (10 events) | n/a | 0.2142 |
| Guardrail triggered | n/a | No (correctly — bimodal data) |

Pytest: `842 passed, 4 pre-existing failures (unchanged), 8 skipped, 29 deselected in 19.75s`

Up from 828 passed before this work (14 new tests, all green).