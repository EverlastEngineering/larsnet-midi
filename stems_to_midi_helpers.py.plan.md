# Refactoring Plan: stems_to_midi.py → Functional Core, Imperative Shell

**Date:** 2025-10-16  
**Goal:** Eliminate code duplication and improve testability using FCIS pattern

## Current State Analysis

### Issues Identified

1. **REPEATED CODE / DUPLICATION**
   - Sustain duration calculation (repeated 4+ times)
   - Spectral energy calculation (repeated 3 times)
   - Peak amplitude calculation (repeated 3+ times)
   - Onset sample/segment extraction (repeated many times)

2. **MAGIC NUMBERS (Should be in YAML config)**
   - `0.001` - Silent audio threshold (-60dB)
   - `0.01 * sr` - 10ms peak window
   - `0.05 * sr` - 50ms FFT analysis window
   - `0.2 * sr` - 200ms sustain analysis window
   - `0.1` - 10% envelope threshold for sustain
   - `kernel_size=51` - Median filter kernel
   - `100` - Minimum segment length for FFT
   - Cymbal frequency ranges `1000-4000Hz`, `4000-10000Hz` (hardcoded)
   - `0.05` - 50ms match tolerance (learning mode)
   - MIDI anchor note `27`, duration `0.01`

3. **UNUSED VARIABLES**
   - `onset_threshold` parameter in `process_stem_to_midi` (config overrides it)
   - `min_velocity`, `max_velocity` parameters (config used instead)
   - `ratios_kept`, `amplitudes_kept` (populated but never used)
   - `body_to_wire`, `body_times_wire` (calculated but only geomean used)
   - `onset_times_orig`, `onset_strengths_orig`, `peak_amplitudes_orig` (only for counting)
   - `kick_alt`, `snare_alt`, `crash`, `ride` in DrumMapping (never referenced)
   - `beats_per_second` in `create_midi_file` (only used once, can inline)

4. **INCONSISTENT PATTERNS**
   - Config access: sometimes `.get()` with defaults, sometimes direct access
   - Stereo-to-mono conversion repeated in 3+ places
   - Energy label assignment via complex if/elif chains

## Functional Core, Imperative Shell Strategy

### Phase 1: Extract Pure Functions (Functional Core) ✅ COMPLETED

**Status:** All 26 tests passing, 86% coverage

Pure functions created in `stems_to_midi_helpers.py`:
- ✅ `ensure_mono()` - Audio channel handling
- ✅ `calculate_peak_amplitude()` - Peak detection in window
- ✅ `calculate_sustain_duration()` - Envelope analysis and sustain measurement
- ✅ `calculate_spectral_energies()` - FFT analysis across frequency ranges
- ✅ `get_spectral_config_for_stem()` - Extract config for stem type
- ✅ `calculate_geomean()` - Geometric mean calculation
- ✅ `should_keep_onset()` - Filtering logic (stem-type aware)
- ✅ `normalize_values()` - Value normalization to 0-1

**Test Coverage:**
- ✅ 26 unit tests for pure functions
- ✅ 86% code coverage
- ✅ All edge cases tested
- ✅ Integration tests still passing (21 tests)

### Phase 2: Refactor process_stem_to_midi() - Use Functional Core

**Target:** `process_stem_to_midi()` function (lines 451-1058)

**Changes:**
1. Import helper functions from `stems_to_midi_helpers`
2. Replace inline mono conversion with `ensure_mono()`
3. Replace peak amplitude calculation loop with `calculate_peak_amplitude()`
4. Replace spectral filtering loop with calls to:
   - `get_spectral_config_for_stem()`
   - `calculate_spectral_energies()`
   - `calculate_sustain_duration()` (for hihat/cymbals)
   - `calculate_geomean()`
   - `should_keep_onset()`
5. Replace normalization with `normalize_values()`
6. Remove unused variables: `ratios_kept`, `amplitudes_kept`, etc.
7. Simplify function signature (remove unused `onset_threshold`, `min_velocity`, `max_velocity`)

**Expected Outcome:**
- ~300-400 lines removed through deduplication
- Logic simplified and more readable
- All tests still pass
- No breaking changes to external API

### Phase 3: Add Magic Numbers to YAML Config

**Target:** `midiconfig.yaml`

**Add new section:**
```yaml
audio:
  force_mono: true
  silent_threshold: 0.001
  peak_window_ms: 10
  fft_window_ms: 50
  sustain_window_ms: 200
  sustain_envelope_threshold: 0.1
  envelope_smooth_kernel: 51
  min_segment_samples: 100
```

**Add to cymbals config:**
```yaml
cymbals:
  body_freq_min: 1000
  body_freq_max: 4000
  brilliance_freq_min: 4000
  brilliance_freq_max: 10000
  # ... existing config
```

**Add to midi config:**
```yaml
midi:
  anchor_note: 27
  anchor_duration: 0.01
  # ... existing config
```

**Add to learning_mode config:**
```yaml
learning_mode:
  match_tolerance_ms: 50
  # ... existing config
```

**Changes in code:**
- Update `stems_to_midi_helpers.py` to read from config
- Update `process_stem_to_midi()` to pass config values
- Update `get_spectral_config_for_stem()` to use cymbal freq ranges from config

**Expected Outcome:**
- All magic numbers in config
- Easy to tune without code changes
- Config validation in tests

### Phase 4: Refactor detect_hihat_state() - Use Functional Core

**Target:** `detect_hihat_state()` function (lines 340-448)

**Changes:**
1. Remove duplicate sustain calculation code
2. Use `calculate_sustain_duration()` when needed
3. Simplify to two clear modes:
   - Mode A: Use pre-calculated sustain_durations (fast)
   - Mode B: Calculate on-demand (fallback)

**Expected Outcome:**
- ~50-80 lines removed
- Clearer separation of concerns
- Tests still pass

### Phase 5: Refactor learn_threshold_from_midi() - Use Functional Core

**Target:** `learn_threshold_from_midi()` function (lines 1160-1450)

**Changes:**
1. Replace inline spectral analysis with helper functions
2. Use `calculate_spectral_energies()` consistently
3. Use `calculate_sustain_duration()` for cymbals
4. Use `get_spectral_config_for_stem()` for config access

**Expected Outcome:**
- ~100-150 lines removed
- Consistent with main processing pipeline
- Learning mode tests still pass

### Phase 6: Clean Up Unused Code

**Remove:**
1. Unused parameters from function signatures
2. Unused variables throughout
3. Unused DrumMapping fields
4. Inline single-use variables

**Standardize:**
1. Config access patterns (always use `.get()` with defaults)
2. Error handling patterns
3. Code style and formatting

**Expected Outcome:**
- Cleaner, more maintainable code
- Consistent patterns throughout
- No functional changes
- All tests pass

### Phase 7: Update Documentation

**Add:**
1. Docstring for FCIS pattern explanation
2. Comments explaining pure vs impure boundaries
3. Update function docstrings with new signatures
4. Add examples in docstrings

**Update:**
1. README.md with architecture overview
2. Code comments for complex logic
3. Config file comments

## Success Criteria

### Functional Requirements
- ✅ All existing tests pass (47 tests currently)
- ✅ No breaking changes to public API
- ✅ Same MIDI output for same inputs
- ✅ All features still work (learning mode, tom detection, etc.)

### Quality Metrics
- ✅ Code coverage maintained or improved (currently 32-35%)
- ✅ Pure functions have >80% coverage (currently 86%)
- ✅ ~300-400 lines removed through deduplication
- ✅ Zero magic numbers in Python code
- ✅ All configuration in YAML
- ✅ Clear FCIS boundary

### Maintainability
- ✅ Pure functions easy to test
- ✅ Imperative shell thin and simple
- ✅ No duplicate code
- ✅ Consistent patterns throughout
- ✅ Clear separation of concerns

## Testing Strategy

### For Each Phase:
1. Run existing tests before changes
2. Make changes incrementally
3. Run tests after each change
4. Check test coverage
5. Verify integration tests pass
6. Manual smoke test if needed

### Test Commands:
```bash
# Unit tests
docker exec larsnet-larsnet_env-1 bash -c "cd /app && python -m pytest test_stems_to_midi_helpers.py -v"

# Integration tests  
docker exec larsnet-larsnet_env-1 bash -c "cd /app && python -m pytest test_stems_to_midi.py -v"

# All tests with coverage
docker exec larsnet-larsnet_env-1 bash -c "cd /app && python -m pytest test_stems_to_midi.py test_stems_to_midi_helpers.py --cov=stems_to_midi --cov=stems_to_midi_helpers --cov-report=term-missing -v"
```

## Rollback Plan

If any phase fails:
1. Git revert to last known good state
2. Review test failures
3. Fix issues in isolation
4. Re-run tests
5. Continue from that phase

## Risk Assessment

### Low Risk ✅
- Phase 1: Pure function extraction (DONE - working)
- Phase 3: Config additions (backward compatible)
- Phase 6: Cleanup unused code

### Medium Risk ⚠️
- Phase 2: Refactor main processing (high test coverage)
- Phase 4: Refactor hihat detection (tested)

### Higher Risk ⚡
- Phase 5: Refactor learning mode (low test coverage)
  - Mitigation: Add tests first, or test manually

## Timeline Estimate

- Phase 1: ✅ COMPLETED (2-3 hours)
- Phase 2: 1-2 hours
- Phase 3: 30 minutes
- Phase 4: 30 minutes
- Phase 5: 1 hour
- Phase 6: 30 minutes
- Phase 7: 30 minutes

**Total:** ~4-6 hours remaining

## Notes

- Follow the instruction: "Before refactoring, write tests to cover existing behavior"
- Commit after each successful phase
- Keep running all tests frequently
- Document any unexpected findings
- Update this plan if scope changes
