# Stems-to-MIDI FCIS Refactoring Plan

## Executive Summary
Refactor the stems-to-MIDI module architecture to strictly adhere to Functional Core, Imperative Shell (FCIS) pattern. Current architecture has 4 files with good FCIS adherence (helpers, main, config, midi) but 3 files with significant violations (processor, detection, learning).

## Current State Analysis

### FCIS Adherence Scores
| Module | Score | Category | Primary Issue |
|--------|-------|----------|---------------|
| stems_to_midi_helpers.py | 95% | ✅ Functional Core | Excellent |
| stems_to_midi.py | 95% | ✅ Imperative Shell | Excellent |
| stems_to_midi_config.py | 85% | ✅ Imperative Shell | Good |
| stems_to_midi_midi.py | 90% | ✅ Imperative Shell | Very Good |
| stems_to_midi_learning.py | 70% | ⚠️ Shell (too much logic) | Needs extraction |
| stems_to_midi_processor.py | 65% | ⚠️ Shell (too much logic) | Needs refactoring |
| stems_to_midi_detection.py | 60% | ⚠️ Mixed (anti-pattern) | Needs redesign |

### Key Problems Identified

1. **stems_to_midi_processor.py** (443 lines)
   - Single function `process_stem_to_midi()` is monolithic
   - Mixes I/O, coordination, and logic
   - Contains inline filtering loops with embedded decisions
   - Too much printing/logging mixed with logic

2. **stems_to_midi_detection.py**
   - Documented as "Mix of Functional Core and Imperative Shell" (anti-pattern)
   - Pure functions (`estimate_velocity`, `classify_tom_pitch`) mixed with coordinators
   - Unclear module purpose and boundaries

3. **stems_to_midi_learning.py**
   - Function `learn_threshold_from_midi()` is 200+ lines
   - Mixes I/O, analysis logic, and display formatting
   - Pure threshold calculation logic not extracted

## Refactoring Approach

### Architectural Principles
1. **Functional Core**: Pure functions with no side effects (I/O, logging, state mutation)
2. **Imperative Shell**: Thin coordinators that handle I/O and call functional core
3. **Clear Boundaries**: Each module has one clear role
4. **Testability**: Pure functions are easily testable without mocks

### Target Architecture
```
stems_to_midi_helpers.py    [Functional Core - Pure Logic]
├── Audio utilities (mono conversion, segment extraction)
├── Spectral analysis (energy calculation, geomean)
├── Filtering logic (should_keep_onset, threshold decisions)
├── MIDI transformations (time to beats, event preparation)
├── Classification logic (tom pitch classification, threshold calculation)
└── Analysis/statistics (accuracy calculation, distribution analysis)

stems_to_midi_detection.py  [Imperative Shell - Algorithm Coordinators]
├── detect_onsets() - coordinates librosa calls
├── detect_tom_pitch() - coordinates pitch detection
└── detect_hihat_state() - coordinates classification

stems_to_midi_processor.py  [Imperative Shell - Processing Coordinator]
├── process_stem_to_midi() - THIN orchestrator (< 100 lines)
├── Delegates to: load_audio, detect_events, filter_events, create_midi_events
└── Handles: I/O, logging, coordination only

stems_to_midi_learning.py   [Imperative Shell - Learning Coordinator]
├── learn_threshold_from_midi() - THIN orchestrator (< 100 lines)
├── Delegates to: threshold calculation, accuracy analysis, formatting
└── Handles: I/O, logging, coordination only

stems_to_midi_config.py      [Imperative Shell - Configuration]
stems_to_midi_midi.py         [Imperative Shell - MIDI I/O]
stems_to_midi.py              [Imperative Shell - CLI]
```

## Phases

### Phase 1: Extract Pure Functions from Detection Module
**Goal**: Move pure functions from detection.py to helpers.py

**Actions**:
1. Move `estimate_velocity()` → helpers.py
2. Move `classify_tom_pitch()` → helpers.py (already pure)
3. Document detection.py as "Algorithm Coordinators" (not mixed)
4. Update tests to import from helpers

**Success Criteria**:
- All pure functions in helpers.py
- detection.py contains only coordinators
- All tests pass
- No functional changes to behavior

**Risk**: Medium - tests may need import updates

### Phase 2: Extract Pure Logic from Processor Module
**Goal**: Extract filtering, classification, and transformation logic to functional core

**Actions**:
1. Create `filter_onsets_by_spectral()` in helpers.py
   - Input: onset data, thresholds, config
   - Output: filtered onset data
   - Pure function, no I/O or logging
2. Create `classify_drum_events()` in helpers.py
   - Input: onset times, audio data, stem type, config
   - Output: classified events with notes
   - Pure function for tom/hihat classification
3. Create `calculate_velocities_from_features()` in helpers.py
   - Input: feature values (geomeans or amplitudes), min/max velocity
   - Output: velocity array
   - Pure function
4. Update `process_stem_to_midi()` to use new helpers
   - Keep: I/O (audio loading), logging, coordination
   - Remove: inline filtering loops, classification logic

**Success Criteria**:
- New helper functions with 100% test coverage
- process_stem_to_midi() reduced from 443 to < 150 lines
- All existing tests pass
- No functional changes to output

**Risk**: High - complex refactoring with many dependencies

### Phase 3: Extract Pure Logic from Learning Module
**Goal**: Extract threshold calculation and analysis logic to functional core

**Actions**:
1. Create `calculate_threshold_from_distributions()` in helpers.py
   - Input: kept_values, removed_values
   - Output: suggested threshold (midpoint between max_removed and min_kept)
   - Pure function
2. Create `calculate_classification_accuracy()` in helpers.py
   - Input: user_actions, predictions
   - Output: accuracy percentage
   - Pure function
3. Create `analyze_threshold_performance()` in helpers.py
   - Input: onset_data, threshold, sustain_threshold (optional)
   - Output: classification results for each onset
   - Pure function
4. Create `format_learning_analysis_table()` in helpers.py
   - Input: analysis results, config
   - Output: formatted strings for display
   - Pure function (string formatting)
5. Update `learn_threshold_from_midi()` to use new helpers
   - Keep: I/O (audio/MIDI loading), logging, coordination
   - Remove: calculation logic, analysis logic

**Success Criteria**:
- New helper functions with 100% test coverage
- learn_threshold_from_midi() reduced from 200+ to < 100 lines
- All existing tests pass
- No functional changes to output

**Risk**: Medium - moderate complexity, less critical path than processor

### Phase 4: Reduce process_stem_to_midi() Complexity
**Goal**: Break down monolithic processor function into smaller coordinators

**Actions**:
1. Extract `_load_and_validate_audio()` helper
   - Loads audio, converts to mono, checks silence
   - Returns audio, sr, or None if invalid
2. Extract `_configure_onset_detection()` helper
   - Reads config, handles learning mode vs normal mode
   - Returns onset detection parameters
3. Extract `_detect_and_analyze_onsets()` helper
   - Calls detect_onsets(), calculates amplitudes
   - Returns onset data for filtering
4. Extract `_filter_and_classify_onsets()` helper
   - Uses functional core for filtering/classification
   - Returns final events
5. Update `process_stem_to_midi()` to orchestrate these helpers
   - Each helper is < 50 lines
   - Main function becomes < 100 lines of pure coordination

**Success Criteria**:
- process_stem_to_midi() is < 100 lines
- Each helper function has single clear purpose
- All tests pass
- No functional changes

**Risk**: Low - pure structural refactoring

### Phase 5: Update Tests and Documentation
**Goal**: Ensure all changes are well-tested and documented

**Actions**:
1. Add tests for new helper functions in test_stems_to_midi_helpers.py
2. Update existing tests if imports changed
3. Add docstring documentation for all new functions
4. Update module docstrings to reflect new architecture
5. Run full test suite with coverage report

**Success Criteria**:
- Test coverage ≥ 85% for helpers.py
- All tests pass
- Documentation complete
- No regressions

**Risk**: Low

## Success Criteria (Overall)

### Quantitative Metrics
- [ ] All FCIS scores ≥ 85%
- [ ] stems_to_midi_helpers.py contains 90%+ pure functions
- [ ] No function > 100 lines (except generated/boilerplate)
- [ ] Test coverage ≥ 85% for helpers.py
- [ ] All existing tests pass
- [ ] No behavioral changes (output identical for same inputs)

### Qualitative Metrics
- [ ] Clear separation: functional core vs imperative shell
- [ ] Each module has single, well-defined purpose
- [ ] Easy to test: pure functions don't need mocks
- [ ] Easy to understand: small functions with clear names
- [ ] Easy to maintain: logic changes in one place (helpers)

## Risks and Mitigations

### Risk 1: Breaking Existing Behavior
**Mitigation**: 
- Write tests before refactoring
- Run tests after each phase
- Use git commits at phase boundaries
- Compare outputs before/after with real audio files

### Risk 2: Test Suite Gaps
**Mitigation**:
- Add comprehensive tests for new helper functions
- Use property-based testing for pure functions
- Test edge cases explicitly

### Risk 3: Performance Regression
**Mitigation**:
- Profile before and after
- Ensure no unnecessary data copying
- Keep numpy operations vectorized

### Risk 4: Incomplete Extraction
**Mitigation**:
- Use grep to find remaining logic in shell modules
- Code review each phase
- Check FCIS scores after each phase

## Timeline Estimate
- Phase 1: 2 hours (straightforward extraction)
- Phase 2: 6 hours (complex refactoring with testing)
- Phase 3: 4 hours (moderate complexity)
- Phase 4: 3 hours (structural refactoring)
- Phase 5: 2 hours (testing and documentation)
- **Total**: ~17 hours

## Dependencies
- All phases depend on existing test suite
- Phase 2-4 are independent (can be done in parallel)
- Phase 5 requires all previous phases complete

## Rollback Plan
- Each phase committed separately
- Can revert individual phases if issues found
- Original code preserved in git history
- Plan file immutable (tracks original intent)
