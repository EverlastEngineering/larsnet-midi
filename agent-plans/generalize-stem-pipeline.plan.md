# Plan: Generalize Stem Pipeline — Config-Driven Capabilities

## Problem

`processing_shell.py` and `analysis_core.py` use hardcoded `stem_type` name checks for flow control, creating:
1. **Bug**: Cymbals have geomeans but fall through to peak_amplitude for velocity (not in `['snare', 'kick', 'toms']` list)
2. **Fragility**: Adding a new stem or capability requires changes in 3+ scattered locations
3. **Dead code**: `stem_type in ['snare', 'kick', 'toms', 'hihat', 'cymbals']` gate is always true
4. **Inconsistent logging**: Onset params only printed for stems with config overrides

## Approach

Extend `get_spectral_config_for_stem()` to return **capability flags** alongside spectral config. The pipeline reads these flags instead of checking stem names.

### New fields in spectral_config:
- `velocity_source`: `'geomean'` | `'onset_strength'` | `'peak_amplitude'` — what drives velocity
- `has_sustain_analysis`: `bool` — whether sustain data is collected during spectral filtering
- `use_sustain_duration`: `bool` — whether MIDI note duration comes from sustain envelope
- `has_spectral_data`: `bool` — whether spectral data is collected for classification (hihat body/sizzle)
- `filter_mode`: `'require_both'` | `'geomean_only'` — how should_keep_onset combines criteria

### Files changed:
1. `analysis_core.py` — `get_spectral_config_for_stem()`, `filter_onsets_by_spectral()`, `should_keep_onset()`
2. `processing_shell.py` — `process_stem_to_midi()`, onset logging
3. `test_analysis_core.py` — Update spectral config assertions

### What stays stem-specific (legitimate):
- `_detect_tom_pitches()`, `_detect_cymbal_pitches()`, `_detect_snare_pitches()` — separate helper functions
- `_create_midi_events()` note routing via classifications (already generic: checks `classifications is not None`)
- `detect_hihat_state()` call — hihat-specific detection shell
- Foot-close generation — config-driven (`generate_foot_close` already in config)

## Risks
- Behavioral change if cymbals velocity source changes from peak_amplitude to geomean (the bug fix)
- Must ensure all 594 tests pass after each phase

## Success Criteria
- Zero `stem_type ==` or `stem_type in [` checks in `process_stem_to_midi()` flow control
- All capability decisions driven by spectral_config fields
- Cymbals use geomean for velocity (bug fix)
- Onset params always logged for every stem
- 594 tests pass
