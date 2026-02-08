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

- [ ] Energy envelope saved as .npz during conversion
- [ ] WebUI API endpoint to serve envelope data

## Phase 3: Dual-Sensitivity Detection

- [ ] Max-sensitivity detection run added
- [ ] Both event sets stored in analysis.json
- [ ] Proofing MIDI export function

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-07 | Domain-specific names per stem, not universal generic names | analysis.json must be self-documenting; generic names lose physical meaning |
