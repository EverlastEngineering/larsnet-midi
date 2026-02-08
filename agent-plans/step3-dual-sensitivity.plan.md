# Plan: Step 3 — Dual-Sensitivity Detection Run

## Goal

Run energy detection twice per stem in `process_stem_to_midi()`: once at max sensitivity to capture *all possible* events (for WebUI interactive re-filtering), and once at configured settings (for MIDI output). Store both in analysis.json.

## Approach

### Phase A: Sensitive Detection Run in processing_shell.py

After the existing configured energy detection + spectral filtering, add a second `detect_onsets_energy_based()` call with:
- `threshold_db=1.0` (near-zero prominence threshold)
- `min_absolute_energy=0.0001` (extremely low noise floor)
- All other params same as configured run (hop_length, method, peak_hold_ms, etc.)

Then run `filter_onsets_by_spectral()` with `learning_mode=True` on the sensitive results. This computes spectral features (geomean, band energies, sustain_ms, Phase 2 metadata) for every detected onset without filtering any out. The WebUI can re-apply filtering client-side.

**Key insight**: The sensitive run reuses the same audio already loaded — no extra I/O. The spectral analysis is the expensive part, but it's needed to pre-compute features for client-side filtering (Step 5 requirement).

### Phase B: Plumbing — Return Dict & CLI

Add `sensitive_onset_data` key to `process_stem_to_midi()` return dict. Update `stems_to_midi_cli.py` to pass it through `analysis_by_stem`.

### Phase C: Sidecar Format v3

Update `save_analysis_sidecar()` to write:
- `events_configured`: Current events array (KEPT + FILTERED from configured detection)
- `events_sensitive`: All events from sensitive detection (all have spectral features)
- Bump version to `3.0`

### Phase D: Tests

- Sensitive run produces >= configured event count
- Sidecar v3 has both `events_sensitive` and `events_configured` keys
- All sensitive events have spectral features pre-computed
- Existing tests still pass (configured behavior unchanged)

## Risks

1. **Performance**: Sensitive detection may find 10x more onsets → spectral analysis on each is O(n). Mitigation: acceptable for offline CLI processing; WebUI loads pre-computed results.
2. **Librosa fallback path**: The sensitive run only applies to energy-based detection. If `use_librosa_detection: true`, skip the sensitive run (legacy mode, no dual-sensitivity).
3. **Sidecar size**: More events = larger JSON. Mitigation: ~500-2000 events × ~15 fields ≈ <100KB per stem (acceptable per plan notes).

## Success Criteria

- `process_stem_to_midi()` returns both configured and sensitive onset data
- analysis.json v3 contains `events_sensitive` per stem with full spectral features
- All existing tests pass unchanged
- New tests cover the dual-sensitivity contract
