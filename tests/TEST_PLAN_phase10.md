# Test Plan — Phase 10 (CLI + WebUI end-to-end verification)

**Date:** 2026-06-20
**Goal:** Verify the PGA-universal cleanup didn't break end-to-end usage.
**Methodology:** TDD — write failing tests first, fix code until they pass.

## Context

The previous phase (1-9) cleanup verified the test suite passed (1207/0) but
did NOT run any end-to-end CLI test. The user discovered this gap when
`python stems_to_midi_cli.py 8` crashed on the first read of the now-removed
`onset_detection.threshold` key. Several other places still reference the
deleted config keys (see audit below).

Project 8 (`user_files/8 - 2_funk_80_beat_4-4_4/`) is the user's test fixture:
- All 5 stems present (kick, snare, hihat, toms, cymbals)
- midiconfig.yaml is clean (no dead keys)
- midi/ is empty (CLI crashed before producing output)
- .drumtomidi_project.json shows `midi_generated: false`

## Known broken references (from grep audit)

- `stems_to_midi_cli.py` — reads `config['onset_detection']['threshold']`
  etc. (FIXED in this commit)
- `stems_to_midi/processing_shell_percentile_gated.py:124` — reads
  `config.get('onset_detection', {}).get('hop_length', 512)` (missing key)
- `stems_to_midi/midi.py:583-647` — writes dead keys to the analysis
  sidecar (geomean_threshold, min_sustain_ms, reverb_continuation_attack_threshold,
  open_geomean_min, open_sustain_ms, expected_clusters, cluster_feature)
- `stems_to_midi/energy_detection_core.py:80,94,118,243-244,468,487,506,509`
  — still has `peak_hold_ms` parameter for the spectral branch
- `export_energy_detection_data.py` — still on disk (audit said dead)
- `stems_to_midi/analysis_core/spectral_utils.py`, `onset_filtering.py`,
  `threshold_learning.py` — dead helper modules whose functions are only
  called by the now-deleted detection paths

## Tests to write

### A. CLI end-to-end (tests/test_cli_e2e.py)

For each stem type, run `stems_to_midi_cli.py 8 --stems <stem>` and verify
the output. Plus one all-stems test. Plus a sidecar-shape test that verifies
the dead keys are GONE from the analysis sidecar.

#### A.1 Single-stem tests

| Test | Command | Assertion |
|---|---|---|
| `test_cli_e2e_kick` | `cli 8 --stems kick` | exit 0; `midi/*.kick.mid` exists, non-empty; sidecar has events_pga.kick |
| `test_cli_e2e_snare` | `cli 8 --stems snare` | same for snare |
| `test_cli_e2e_toms` | `cli 8 --stems toms` | same for toms |
| `test_cli_e2e_hihat` | `cli 8 --stems hihat` | same for hihat |
| `test_cli_e2e_cymbals` | `cli 8 --stems cymbals` | same for cymbals |

#### A.2 All-stems

| Test | Command | Assertion |
|---|---|---|
| `test_cli_e2e_all_stems` | `cli 8` | exit 0; all 5 `.mid` files exist; all 5 sidecar stems have events_pga |

#### A.3 Sidecar shape

| Test | Assertion |
|---|---|
| `test_cli_sidecar_no_dead_keys` | after CLI run, `midi/*.analysis.json` does NOT contain `geomean_threshold`, `min_sustain_ms`, `reverb_continuation_attack_threshold`, `open_geomean_min`, `open_sustain_ms`, `expected_clusters`, `cluster_feature`, `events_spectral` (any of these is a failure) |
| `test_cli_sidecar_has_pga_events` | after CLI run, sidecar has `events_pga` array (non-empty) for each of the 5 stems |

#### A.4 Project metadata

| Test | Assertion |
|---|---|
| `test_cli_updates_project_status` | after CLI run, project 8's `.drumtomidi_project.json` shows `midi_generated: true` |

### B. WebUI end-to-end (tests/playwright/specs/)

#### B.1 Project creation

| Test | Steps | Assertion |
|---|---|---|
| `04-create-project.spec.ts` | open WebUI → click Create Project → upload AIFF → submit | project appears in project list with next available number; new project has expected folder structure |

#### B.2 MIDI conversion trigger

| Test | Steps | Assertion |
|---|---|---|
| `05-trigger-midi-conversion.spec.ts` | open project 8 → click "Convert to MIDI" → wait for completion | all 5 `.mid` files appear in the project tree |

#### B.3 Analysis view

| Test | Steps | Assertion |
|---|---|---|
| `06-analysis-view.spec.ts` | open project 8 → click a stem tab | waveform + events render without errors |

## Strategy

1. Write `tests/test_cli_e2e.py` with all the A.* tests.
2. Run them — most will FAIL (CLI is broken, sidecar still has dead keys, etc.).
3. Fix `stems_to_midi_cli.py` (already done in this commit) → some tests pass.
4. Fix `stems_to_midi/processing_shell_percentile_gated.py` (hop_length).
5. Fix `stems_to_midi/midi.py` (stop writing dead keys to sidecar).
6. Fix `stems_to_midi/energy_detection_core.py` (remove peak_hold_ms from spectral branch param docstring).
7. Delete `export_energy_detection_data.py`.
8. Re-run tests until all pass.
9. Add Playwright specs (B.1-B.3).
10. Run Playwright manually to verify (or via CI).

## End state

- All A.* tests pass.
- B.* specs exist and run successfully (or skip with clear reason if WebUI unavailable).
- Sidecar shape verified: only live keys (events_pga, logic, events_configured, events_sensitive, midi_keys, events_pga_filter metadata).
- No dead YAML keys read by any code path.
