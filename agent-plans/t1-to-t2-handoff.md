# T1 → T2 handoff (2026-06-06)

Notes from the T1 drift-fix coder's wrap-up, so the T2 coder doesn't
have to rediscover them.

## What T1 already did (T2 does NOT need to redo)

- **detect_open CLI param is GONE.** Replaced by schema-driven flags
  `--hihat-open-geomean` and `--hihat-open-sustain-ms` (driven by
  `hihat_open_geomean_min` and `hihat_open_sustain_ms`
  `SettingDefinition`s). This already resolves the "Part 1: Parameter
  Name Mismatch" portion of the hihat bug. T2 A1 (unify parameter names)
  is effectively done — verify with a grep, then move to A2-A4.
- **`hihat.midi_note_handclap` is now plumbed through.** Was hardcoded
  39 in `stems_to_midi/config.py::DrumMapping`. The field is now
  `DrumMapping.hihat_handclap` and the `handclap` property reads from
  config. To use from hihat detection: `dm.hihat_handclap`.
- **Hihat detection code untouched per scope rule.**
  `stems_to_midi/processing_shell.py` hihat code paths and
  `stems_to_midi/note_classification_core.py::classify_hihat_notes` are
  unchanged. T2 owns these.

## What T2 still needs to do

- **A2** — grep the webui for any `detect_hihat_open` stragglers in
  `operations.py` or JS. Should be already gone since the field is
  removed, but worth a grep.
- **A3** — add hihat tuning UI in
  `webui/static/js/threshold-tuning.js`: sliders for `open_geomean_min`
  and `open_sustain_ms`, plus cluster visualization similar to
  snare/toms/cymbals.
- **A4** — `classify_hihat_notes()` should preserve stored `hihat_state`
  on rebuild (don't re-classify if the event already has a state), like
  the other stems do.
- **B** — save `pan_confidence`, `pitch_hz`, `stereo_width` to
  `analysis.json` in `save_analysis_sidecar()`. Features are computed
  in `energy_detection_core.py` and `stereo_core.py` but not serialized.
  Use `dm.hihat_handclap` if B involves hihat handclap classification.
- **C** — `events_configured ⊆ events_sensitive` validation in
  `load_analysis_sidecar()`, surface a warning the WebUI can toast.
- **D** — Reverb filter UI vs. actual parity (UI shows the same events
  the analysis stores). Hardest to test without driving the UI.
- **E** — MIDI timing parity (initial conversion == reconvert, within
  1ms). New regression test in
  `stems_to_midi/test_rebuild_core.py` is required.

## Useful field references

- New CLI flag → `SettingDefinition` entry in
  `webui/settings_schema.py` with `cli_flag` set. The schema's
  `cli_builder.py` handles argparse wiring.
- `DrumMapping` is now a real dataclass; `dm.hihat_handclap` is the
  configurable handclap note (replaces hardcoded 39).
- The CLI builder is `webui/cli_builder.py` (250 LOC). If T2 needs a
  new CLI flag, it auto-flows from the schema entry.
- Schema is the source of truth: every setting must be a
  `SettingDefinition`. YAML is persistence only.

## T1 verification result

- 941 pass, 11 fail (all pre-existing audio-fixture failures, NOT
  regressions). The 11 are fixtures the user removed; they were
  documented in the original suite. T2 should not try to fix them.
- Commit: `aad9836` on branch `revert-to-analysis-method`.

## Hard rule (still applies)

If T2 finds any new bug while working, append to
`agent-plans/bug-tracking.md` in the existing format with today's date
(2026-06-06). Non-negotiable.
