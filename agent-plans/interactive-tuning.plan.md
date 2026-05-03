# Interactive Tuning System — Plan

**Created:** 2026-02-07
**Scope:** Steps 1–3 of plan-interactiveTuning.prompt.md (backend/data foundation)
**Steps 4–6:** Deferred to a separate plan (WebUI work)

## Phases

### Phase 1: Standardize Naming End-to-End
Rename generic `primary`/`secondary`/`tertiary` energy keys to domain-specific names throughout the pipeline. Remove `body_wire_geomean`. Make analysis.json self-documenting.

**Approach:** Backward-compatible rename. The spectral config function returns domain-specific keys. All downstream code uses those keys. analysis.json includes a `freq_bands` metadata block per stem so consumers can interpret the fields.

**Key constraint:** The geomean calculation and filtering logic must remain identical — only names change, not behavior.

**Files to modify:**
- `stems_to_midi/analysis_core.py` — `get_spectral_config_for_stem()`, `analyze_onset_spectral()`, `filter_onsets_by_spectral()`, `calculate_badness_score()`, `calculate_geomean()`
- `stems_to_midi/midi.py` — `save_analysis_sidecar()`
- `stems_to_midi/processing_shell.py` — debug output, velocity mapping references
- `stems_to_midi/detection_shell.py` — hihat `filtered_spectral` usage
- `stems_to_midi/learning.py` — spectral analysis references
- `stems_to_midi/optimization/extract_features.py` — feature extraction
- Tests: `test_midi_core.py`, `test_sidechain_core.py`, `test_render_video_core.py`, `test_integration.py`, etc.

**Risks:**
- Many scattered references to `primary_energy`, `secondary_energy` across tests and analysis scripts
- `body_wire_geomean` used as dict key in processing_shell.py debug output
- WebUI settings_schema.py may reference old field names

### Phase 2: Persist Waveform + Envelope Data
Save energy envelope arrays alongside analysis.json during conversion.

### Phase 3: Dual-Sensitivity Detection
Run detection twice (max sensitivity + configured) and store both result sets.

## Success Criteria
- All tests pass after rename
- analysis.json contains domain-specific field names per stem
- analysis.json logic block includes `freq_bands` metadata mapping field names to Hz ranges
- No behavioral change in detection or filtering
- `body_wire_geomean` variable eliminated from codebase
