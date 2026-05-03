# Results: Generalize Stem Pipeline — Config-Driven Capabilities

## Phase 1: Extend spectral config with capabilities
- [x] Add velocity_source, has_sustain_analysis, use_sustain_duration, has_spectral_data, filter_mode
- [x] Update test assertions
- [x] Tests pass (6/6 spectral config tests, 114/114 total)

## Phase 2: Generalize filter_onsets_by_spectral
- [x] Use has_sustain_analysis instead of stem_type checks for sustain collection
- [x] Use has_spectral_data instead of stem_type == 'hihat' for spectral data
- [x] Generalize decay filter: config-driven enable_decay_filter (default False) replaces stem_type == 'cymbals'
- [x] Generalize statistical filter: config-driven enable_statistical_filter replaces stem_type == 'kick'
- [x] Generalize decay_analysis return: presence check replaces stem_type == 'cymbals'
- [x] Generalize filtered_spectral dict: uses geomean_bands dynamically instead of hardcoded body/sizzle
- [x] Tests pass (114/114)

## Phase 3: Generalize should_keep_onset
- [x] Replace stem_type branching with filter_mode parameter ('require_both' | 'geomean_only')
- [x] Backward compatibility via stem_type parameter inference
- [x] Updated all callers: analysis_core.py, processing_shell.py, extract_features.py
- [x] Updated 25+ test calls to use filter_mode
- [x] Tests pass (114/114)

## Phase 4: Generalize processing_shell.py pipeline
- [x] Replace sustain extraction stem_type checks with has_sustain_analysis/has_spectral_data
- [x] Replace velocity source stem_type checks with velocity_source capability flag
- [x] **Fixed cymbals velocity bug**: cymbals now use geomean (was falling through to peak_amplitude)
- [x] Remove dead stem_type gate (always-true `stem_type in ['snare', 'kick', 'toms', 'hihat', 'cymbals']`)
- [x] Fix onset params logging: always prints detection params, not just when overridden
- [x] Unified sustain_durations/spectral_data variables (was hihat_sustain_durations/cymbal_sustain_durations)
- [x] Removed `in locals()` fragile checks by proper initialization
- [x] Tests pass (695 passed, 5 pre-existing failures)

## Phase 4b: Generalize _create_midi_events duration
- [x] Replace stem_type == 'cymbals' duration check with use_sustain_duration parameter
- [x] Add use_sustain_duration bool param to _create_midi_events
- [x] Caller passes flag from spectral_config
- [x] Updated test_foot_close_not_for_cymbals to pass use_sustain_duration=True
- [x] Tests pass (695 passed, 5 pre-existing failures)

## Phase 5: Commit
- [x] All 695 tests pass (5 pre-existing failures unchanged)
- [ ] Committed

## Decision Log
| Decision | Rationale |
|----------|-----------|
| enable_decay_filter defaults to False | Only cymbals currently use it; other stems opt in via config |
| Backward compat for should_keep_onset stem_type param | Allows incremental migration; can remove later |
| Unified sustain_durations variable | Eliminates need for separate hihat/cymbal variables |
