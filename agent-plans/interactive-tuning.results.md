# Interactive Tuning System — Results

Tracks progress against interactive-tuning.plan.md.

## Phase 1: Standardize Naming

- [x] `get_spectral_config_for_stem()` returns domain-specific keys
- [x] `analyze_onset_spectral()` uses domain-specific keys
- [x] `filter_onsets_by_spectral()` uses domain-specific keys, `body_wire_geomean` removed
- [x] `save_analysis_sidecar()` writes domain-specific fields + `freq_bands` metadata
- [x] `processing_shell.py` debug output updated
- [x] `detection_shell.py` hihat spectral references updated
- [x] `learning.py` updated
- [x] `optimization/extract_features.py` updated
- [x] `calculate_badness_score()` updated
- [x] `midi_types.py` type contracts updated (SpectralOnsetData, OnsetFeatures, field sets)
- [x] `clustering_core.py` feature names updated (15 features with all 6 domain bands)
- [x] Export/analysis scripts updated (4 files)
- [x] Tests pass (695 passed, 5 pre-existing failures unrelated to naming)
- [ ] Integration test: run conversion and verify analysis.json output

## Phase 2: Persist Waveform Data

- [x] `detect_stereo_transient_peaks()` returns L/R envelope arrays + time axis
- [x] `detect_onsets_energy_based()` passes envelope through `extra_data`
- [x] `process_stem_to_midi()` returns `envelope_data` dict (times, left, right, sr, hop_length, method)
- [x] `save_envelope_data()` / `load_envelope_data()` in midi.py — compressed .npz per stem
- [x] CLI saves `{base_name}.{stem_type}.envelope.npz` alongside analysis.json
- [x] 6 tests: smoke, round-trip, multi-stem, missing stem, None handling, file size
- [x] Tests pass (701 passed, 5 pre-existing failures)
- [ ] WebUI API endpoint to serve envelope data (deferred to Step 4)

## Phase 3: Dual-Sensitivity Detection

- [ ] Max-sensitivity detection run added
- [ ] Both event sets stored in analysis.json
- [ ] Proofing MIDI export function

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-07 | Domain-specific names per stem, not universal generic names | analysis.json must be self-documenting; generic names lose physical meaning |
| 2026-02-07 | Per-stem .npz files, not one combined file | WebUI loads stems on demand; separate files avoid loading all stem data at once |
| 2026-02-07 | float32 compression in .npz | 3-min song envelope is ~62KB uncompressed, <100KB compressed — negligible overhead |
