# Functional Core, Imperative Shell Architecture Review
## stems_to_midi Module Analysis

**Review Date:** October 16, 2025  
**Reviewer:** AI Assistant  
**Architecture Pattern:** Functional Core, Imperative Shell (FCIS)

---

## Executive Summary

The stems_to_midi module has been **successfully refactored** to follow the Functional Core, Imperative Shell architecture pattern with good separation of concerns. The module is well-organized into distinct layers with clear responsibilities.

**Overall Grade: B+**

### Strengths ✅
- Clear module separation with well-defined responsibilities
- Pure helper functions properly isolated in `stems_to_midi_helpers.py`
- Good use of functional core for business logic (spectral analysis, filtering decisions)
- Thin imperative shell in `stems_to_midi.py` (CLI orchestration)
- Testable pure functions with no side effects
- Config loading properly isolated as I/O operation

### Areas for Improvement ⚠️
- Some algorithmic functions still perform I/O (especially in detection module)
- Mixed responsibilities in `stems_to_midi_detection.py` and `stems_to_midi_processor.py`
- MIDI file operations could be more functionally separated
- Learning module has some code duplication with processor

---

## Module-by-Module Analysis

### 1. `stems_to_midi.py` (Imperative Shell) ✅ **Grade: A**

**Role:** CLI orchestration and main entry point

**Architecture Compliance:** Excellent
- Thin imperative shell as intended
- Delegates all logic to specialized modules
- Handles I/O (file paths, config loading)
- Coordinates workflow without implementing business logic
- Proper separation of concerns

**Code Example:**
```python
def stems_to_midi(...):
    # I/O operations
    stems_dir = Path(stems_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Delegate to functional modules
    config = load_config()
    drum_mapping = DrumMapping()
    
    # Orchestrate workflow
    for audio_dir in audio_files_to_process:
        events = process_stem_to_midi(...)  # Delegate
        create_midi_file(...)               # Delegate
```

**Issues:** None significant

---

### 2. `stems_to_midi_config.py` (Imperative Shell) ✅ **Grade: A-**

**Role:** Configuration loading and data structures

**Architecture Compliance:** Good
- Properly identified as "Part of the Imperative Shell"
- `load_config()` performs I/O (YAML loading) - correct placement
- `DrumMapping` is a pure data structure (dataclass)
- No business logic mixed in

**Code Example:**
```python
def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load MIDI conversion configuration from YAML file."""
    # I/O operation - appropriate for imperative shell
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
```

**Minor Issue:**
- Could benefit from config validation logic being extracted to a pure function

---

### 3. `stems_to_midi_helpers.py` (Functional Core) ✅ **Grade: A**

**Role:** Pure helper functions for audio processing

**Architecture Compliance:** Excellent
- **All functions are pure** - no I/O, no side effects
- Well-documented as "functional core functions"
- Highly testable
- Clear input → output transformations

**Pure Functions Include:**
- `ensure_mono()` - audio transformation
- `calculate_peak_amplitude()` - audio analysis
- `calculate_sustain_duration()` - envelope analysis
- `calculate_spectral_energies()` - frequency domain analysis
- `get_spectral_config_for_stem()` - config extraction (pure!)
- `calculate_geomean()` - mathematical operation
- `should_keep_onset()` - decision logic (pure!)
- `normalize_values()` - array transformation

**Excellent Example of Pure Function:**
```python
def should_keep_onset(
    geomean: float,
    sustain_ms: Optional[float],
    geomean_threshold: Optional[float],
    min_sustain_ms: Optional[float],
    stem_type: str
) -> bool:
    """
    Determine if an onset should be kept based on spectral/sustain criteria.
    
    Pure function - decision logic without side effects.
    """
    # Complex decision logic, but NO side effects or I/O
    if stem_type == 'cymbals':
        if geomean_threshold is not None and min_sustain_ms is not None:
            return (geomean > geomean_threshold) and (sustain_ms >= min_sustain_ms)
    # ... more logic
```

**Issues:** None

---

### 4. `stems_to_midi_detection.py` (Mixed) ⚠️ **Grade: B**

**Role:** Onset detection and drum classification

**Architecture Compliance:** Mixed - some violations

**Self-Assessment:** "Mix of Functional Core and Imperative Shell"

**Analysis:**

#### Pure Functions ✅
- `estimate_velocity()` - pure transformation
- `classify_tom_pitch()` - pure classification logic

#### Impure Functions (Algorithmic Coordination) ⚠️
- `detect_onsets()` - coordinates librosa calls, no I/O but does algorithmic work
- `detect_tom_pitch()` - coordinates librosa, has try/except logic
- `detect_hihat_state()` - partially impure, does calculations inline

**Problem Areas:**

1. **`detect_onsets()` performs stereo conversion inline:**
```python
def detect_onsets(audio: np.ndarray, sr: int, ...) -> Tuple[...]:
    # Should use helper function instead
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)  # ⚠️ Should call ensure_mono()
```
**Fix:** Use `ensure_mono()` from helpers

2. **`detect_hihat_state()` has inline calculations:**
```python
def detect_hihat_state(...):
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)  # ⚠️ Duplicate logic
    
    # Inline sustain calculation - should delegate to helper
    sustain_duration_ms = calculate_sustain_duration(...)  # ✅ Good!
```

**Recommendations:**
- Reuse `ensure_mono()` from helpers everywhere
- Extract more logic to pure functions in helpers
- These functions are more "algorithmic coordinators" than pure functions, which is acceptable, but should be clearly documented

---

### 5. `stems_to_midi_processor.py` (Mixed) ⚠️ **Grade: B-**

**Role:** Main audio processing pipeline

**Architecture Compliance:** Moderate - violates FCIS in several areas

**Self-Assessment:** "Handles the main processing pipeline"

**Analysis:**

#### Good Practices ✅
- Imports and uses helpers extensively
- Delegates to detection functions
- Separates I/O (audio loading) from processing

#### Problem Areas ⚠️

1. **Mixed I/O and Logic:**
```python
def process_stem_to_midi(...) -> List[Dict]:
    # I/O operation
    audio, sr = sf.read(str(audio_path))  # ⚠️ I/O in processing function
    
    # Then does processing...
    audio = ensure_mono(audio)  # ✅ Uses helper
    
    # More processing...
```
**Issue:** Function name says "process" but it also loads files. Should be split.

2. **Complex Inline Logic:**
```python
# Massive inline filtering loop (lines 150-250)
for onset_time, strength, peak_amplitude in zip(...):
    # Calculate spectral energies
    energies = calculate_spectral_energies(...)  # ✅ Uses helper
    
    # Inline decision making mixed with data collection
    if learning_mode or is_real_hit:
        filtered_times.append(...)
    
    # Inline data structure building
    all_onset_data.append({...})  # ⚠️ Side effect accumulation
```

**Issue:** This loop is doing too much - filtering, data collection, conditional logic, all mixed together.

3. **Excessive Print Statements:**
```python
print(f"  Processing {stem_type} from: {audio_path.name}")
print(f"    Max amplitude: {max_amplitude:.6f}")
# ... 20+ more print statements
```
**Issue:** Mixing presentation logic with business logic. Should use logging or return messages.

**Recommendations:**
1. Split into two functions:
   - `load_stem_audio()` - pure I/O
   - `process_loaded_audio()` - pure processing (takes audio array)
2. Extract filtering loop logic to pure functions
3. Replace print statements with structured logging or return status objects

---

### 6. `stems_to_midi_midi.py` (Imperative Shell) ⚠️ **Grade: B**

**Role:** MIDI file creation and reading

**Architecture Compliance:** Acceptable but could be improved

**Analysis:**

#### Current Structure:
```python
def create_midi_file(
    events_by_stem: Dict[str, List[Dict]],
    output_path: Union[str, Path],
    ...
):
    """Create a MIDI file from detected drum events."""
    # Import here to avoid circular dependency
    from stems_to_midi_config import load_config  # ⚠️ Circular dependency hint
    
    # I/O operation
    midi = MIDIFile(1)  # ✅ Correct placement
    
    # Logic mixed with I/O
    for stem_type, events in events_by_stem.items():
        for event in events:
            time_in_beats = event['time'] * beats_per_second  # ⚠️ Logic
            midi.addNote(...)  # ✅ I/O
    
    # I/O operation
    with open(output_path, 'wb') as f:  # ✅ Correct
        midi.writeFile(f)
```

**Issues:**
1. Time conversion logic (`time_in_beats = event['time'] * beats_per_second`) should be in a pure function
2. Circular dependency hint suggests architectural issue
3. `read_midi_notes()` mixes parsing logic with I/O

**Recommendations:**
1. Create pure functions:
   - `convert_events_to_beats()` - pure time conversion
   - `parse_midi_messages()` - pure parsing logic
2. Keep file operations in imperative shell
3. Resolve circular dependencies through better module organization

---

### 7. `stems_to_midi_learning.py` (Mixed) ⚠️ **Grade: C+**

**Role:** Threshold learning from user-edited MIDI

**Architecture Compliance:** Poor - significant violations

**Analysis:**

#### Major Problems:

1. **I/O Mixed with Business Logic:**
```python
def learn_threshold_from_midi(...) -> Dict[str, float]:
    # I/O operations
    original_times = read_midi_notes(original_midi_path, target_note)
    edited_times = read_midi_notes(edited_midi_path, target_note)
    audio, sr = sf.read(str(audio_path))  # ⚠️ I/O in logic function
    
    # Business logic
    for orig_time in original_times:
        # Analysis logic...
        geomean = calculate_geomean(...)  # ✅ Uses helper
        
        if is_kept:
            kept_geomeans.append(geomean)
    
    # More business logic
    suggested_threshold = (max_removed + min_kept) / 2.0
    
    # Side effects
    print(f"\n  Learning thresholds for {stem_type}...")  # ⚠️ Print statements
```

**Issues:**
- Function does I/O (read files), business logic (threshold calculation), AND presentation (print)
- Violates Single Responsibility Principle
- Not testable without file system access

2. **Code Duplication with Processor:**
```python
# Similar spectral analysis loop as in processor.py
for orig_time in original_times:
    onset_sample = int(orig_time * sr)
    segment = audio[onset_sample:end_sample]
    
    # Same spectral analysis as processor does
    energies = calculate_spectral_energies(segment, sr, spectral_config['freq_ranges'])
    geomean = calculate_geomean(primary_energy, secondary_energy)
```
**Issue:** Duplicates logic from `stems_to_midi_processor.py`

3. **Massive Print Output Logic:**
- 100+ lines of print formatting
- Should be extracted to presentation layer

**Recommendations:**
1. Split into three functions:
   - `load_learning_data()` - I/O only
   - `calculate_optimal_threshold()` - pure logic
   - `format_learning_results()` - presentation
2. Extract shared analysis logic to helpers
3. Return structured data instead of printing

---

## Summary of Violations and Recommendations

### Critical Issues (Must Fix)

1. **`stems_to_midi_processor.py`** - Split I/O from processing
   ```python
   # Current (BAD):
   def process_stem_to_midi(audio_path, ...):
       audio, sr = sf.read(str(audio_path))  # I/O + logic
   
   # Proposed (GOOD):
   def load_stem_audio(audio_path) -> Tuple[np.ndarray, int]:
       return sf.read(str(audio_path))  # I/O only
   
   def process_audio_to_events(audio, sr, ...) -> List[Dict]:
       # Pure processing logic
   ```

2. **`stems_to_midi_learning.py`** - Major refactoring needed
   - Extract I/O operations
   - Create pure threshold calculation functions
   - Move presentation logic out

3. **`stems_to_midi_midi.py`** - Extract logic from I/O
   ```python
   # Pure function
   def convert_seconds_to_beats(time_sec: float, tempo: float) -> float:
       return time_sec * (tempo / 60.0)
   
   # Pure function
   def prepare_midi_events(events: List[Dict], tempo: float) -> List[Dict]:
       # Convert all times to beats
   
   # Imperative shell
   def write_midi_file(prepared_events: List[Dict], output_path: Path):
       # Just writes to file
   ```

### Medium Priority Issues

4. **`stems_to_midi_detection.py`** - Reuse helpers consistently
   - Use `ensure_mono()` everywhere instead of inline conversion
   - Document algorithmic coordination functions clearly

5. **Print Statement Overuse** - Throughout all modules
   - Replace with structured logging
   - Or return status/message objects
   - Separate presentation from logic

### Low Priority Issues

6. **Config Validation** - Add pure validation functions
7. **Error Handling** - Extract error recovery logic to pure functions
8. **Documentation** - Some functions need clearer FCIS classification

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    IMPERATIVE SHELL                         │
│  (I/O, Side Effects, Orchestration)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  stems_to_midi.py (CLI Orchestration)          ✅ Grade: A  │
│  ├─ Argument parsing                                       │
│  ├─ File path handling                                     │
│  └─ Workflow coordination                                  │
│                                                             │
│  stems_to_midi_config.py (Config I/O)          ✅ Grade: A- │
│  ├─ YAML file reading                                      │
│  └─ DrumMapping data structure                            │
│                                                             │
│  stems_to_midi_midi.py (MIDI I/O)              ⚠️ Grade: B  │
│  ├─ MIDI file writing                                      │
│  ├─ MIDI file reading                                      │
│  └─ ⚠️ Mixed: Some logic should be in core                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    MIXED LAYER                              │
│  (Algorithmic Coordination - Some Violations)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  stems_to_midi_processor.py                    ⚠️ Grade: B- │
│  ├─ ⚠️ I/O mixed with processing                          │
│  ├─ ✅ Uses helpers extensively                            │
│  └─ ⚠️ Too many responsibilities                           │
│                                                             │
│  stems_to_midi_detection.py                    ⚠️ Grade: B  │
│  ├─ Algorithmic coordination (acceptable)                 │
│  ├─ ⚠️ Some duplicate logic                               │
│  └─ ✅ Some pure functions                                │
│                                                             │
│  stems_to_midi_learning.py                     ⚠️ Grade: C+ │
│  ├─ ⚠️ Major violations: I/O + logic + presentation       │
│  ├─ ⚠️ Code duplication                                    │
│  └─ ⚠️ Not testable                                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    FUNCTIONAL CORE                          │
│  (Pure Functions, No I/O, No Side Effects)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  stems_to_midi_helpers.py                      ✅ Grade: A  │
│  ├─ ✅ All functions are pure                              │
│  ├─ ✅ Highly testable                                     │
│  ├─ ✅ No side effects                                     │
│  ├─ ✅ Clear transformations                               │
│  └─ ✅ Well-documented                                     │
│                                                             │
│  Functions:                                                 │
│  ├─ ensure_mono()                                          │
│  ├─ calculate_peak_amplitude()                             │
│  ├─ calculate_sustain_duration()                           │
│  ├─ calculate_spectral_energies()                          │
│  ├─ get_spectral_config_for_stem()                         │
│  ├─ calculate_geomean()                                    │
│  ├─ should_keep_onset()                                    │
│  └─ normalize_values()                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Implications

### Easily Testable (Pure Functions) ✅
- All functions in `stems_to_midi_helpers.py`
- `estimate_velocity()` in detection
- `classify_tom_pitch()` in detection

**Example Test:**
```python
def test_should_keep_onset():
    # Pure function - easy to test
    assert should_keep_onset(
        geomean=100.0,
        sustain_ms=200.0,
        geomean_threshold=50.0,
        min_sustain_ms=150.0,
        stem_type='cymbals'
    ) == True
```

### Hard to Test (Mixed I/O and Logic) ⚠️
- `process_stem_to_midi()` - requires file system
- `learn_threshold_from_midi()` - requires multiple files
- `create_midi_file()` - requires file system

**These need refactoring to be testable.**

---

## Recommendations Priority

### High Priority (Do First)
1. **Refactor `stems_to_midi_learning.py`**
   - Biggest FCIS violation
   - Create: `load_learning_data()`, `calculate_optimal_threshold()`, `format_learning_results()`

2. **Split `process_stem_to_midi()` in processor**
   - Separate `load_stem_audio()` from `process_audio_to_events()`
   - Makes testing possible

3. **Extract logic from `stems_to_midi_midi.py`**
   - Create pure time conversion functions
   - Create pure MIDI message preparation functions

### Medium Priority
4. **Eliminate code duplication**
   - Reuse `ensure_mono()` everywhere
   - Extract shared spectral analysis patterns

5. **Replace print statements**
   - Use structured logging
   - Return status objects

### Low Priority
6. **Add config validation**
7. **Improve documentation**
8. **Add more helper functions**

---

## Conclusion

The stems_to_midi module has made **significant progress** toward Functional Core, Imperative Shell architecture:

**Successes:**
- ✅ `stems_to_midi_helpers.py` is exemplary - all pure functions
- ✅ `stems_to_midi.py` is a proper thin shell
- ✅ Good module separation and clear responsibilities
- ✅ Many functions properly use helpers

**Remaining Work:**
- ⚠️ `stems_to_midi_learning.py` needs major refactoring
- ⚠️ `stems_to_midi_processor.py` needs I/O separation
- ⚠️ `stems_to_midi_midi.py` needs logic extraction
- ⚠️ Excessive print statements throughout

**Overall Assessment:**
The architecture is **75% compliant** with FCIS principles. The core helpers are excellent, but some imperative shell components still contain too much business logic. With focused refactoring on the identified issues, this could easily reach 90%+ compliance.

**Next Steps:**
1. Create refactoring plan for learning module (highest priority)
2. Add tests for existing pure functions to establish baseline
3. Incrementally extract logic from mixed modules
4. Document FCIS classification clearly in each module
