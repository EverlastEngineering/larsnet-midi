# Duplicate Code Analysis - stems_to_midi*.py Files

**Date:** October 16, 2025  
**Analyzed Files:** 7 Python modules  

## Executive Summary

Analysis of all `stems_to_midi*.py` files reveals **minimal duplication** overall. The codebase demonstrates good separation of concerns following the "Functional Core, Imperative Shell" architecture. However, there are a few areas with duplicate or redundant code patterns that could be consolidated.

---

## 1. Critical Duplication Issues

### 1.1 Duplicate Imports at Bottom of `stems_to_midi_midi.py` ⚠️

**Location:** `stems_to_midi_midi.py:130-145`

**Issue:** The file has duplicate import blocks - one at the top (lines 1-12) and one at the bottom (lines 130-145).

```python
# Lines 1-12 (TOP - CORRECT)
from midiutil import MIDIFile
import mido
from pathlib import Path
from typing import Dict, List, Union, Optional

# Lines 130-145 (BOTTOM - DUPLICATE)
from pathlib import Path
from typing import Union, List, Dict, Optional
from midiutil import MIDIFile
import mido

# Import config
from stems_to_midi_config import load_config

# Import helpers
from stems_to_midi_helpers import prepare_midi_events_for_writing
```

**Impact:** Medium - confusing structure, but functionally works  
**Recommendation:** **Remove the duplicate imports at the bottom (lines 130-145)** and keep only the top imports.

---

## 2. Minor Code Patterns (Not True Duplication)

### 2.1 Audio Loading Pattern

**Occurrences:**
- `stems_to_midi_processor.py:72-73`
- `stems_to_midi_learning.py:67-68`

```python
# Pattern appears in multiple files:
audio, sr = sf.read(str(audio_path))
```

**Analysis:** This is a **standard library call pattern**, not duplication. Each usage is in a different context (processing vs learning) and doesn't warrant extraction.

**Recommendation:** No action needed - this is idiomatic usage.

---

### 2.2 Mono Conversion Pattern

**Occurrences:**
- `stems_to_midi_processor.py:76-78`
- `stems_to_midi_learning.py:70-71`

```python
# Pattern:
if config['audio']['force_mono'] and audio.ndim == 2:
    audio = ensure_mono(audio)  # or np.mean(audio, axis=1)
```

**Analysis:** This pattern appears in different contexts:
- `processor.py` uses the helper function `ensure_mono()` (correct)
- `learning.py` uses `np.mean(audio, axis=1)` directly

**Recommendation:** **Minor inconsistency** - `learning.py` should use `ensure_mono()` for consistency:

```python
# In stems_to_midi_learning.py:70-71, change:
if config['audio']['force_mono'] and audio.ndim == 2:
    audio = np.mean(audio, axis=1)

# To:
if config['audio']['force_mono'] and audio.ndim == 2:
    audio = ensure_mono(audio)
```

---

### 2.3 Spectral Analysis Pattern

**Occurrences:**
- `stems_to_midi_processor.py:144-184` (uses unified helper)
- `stems_to_midi_learning.py:76-97` (partially uses helper)

**Analysis:** Both files analyze spectral properties of onsets:
- `processor.py` properly uses the unified `analyze_onset_spectral()` helper
- `learning.py` uses the helper but then recalculates some values manually

**Recommendation:** **Learning module is correctly using the helper** - no duplication issue.

---

## 3. Well-Designed Shared Code (Not Duplication)

These are examples of **proper code reuse** via the `stems_to_midi_helpers.py` module:

### 3.1 Pure Helper Functions (Functional Core)

All these are centralized in `stems_to_midi_helpers.py` and properly imported:

- `ensure_mono()` - Used by: detection, processor, learning
- `calculate_peak_amplitude()` - Used by: processor, learning  
- `calculate_sustain_duration()` - Used by: processor, learning
- `calculate_spectral_energies()` - Used by: processor, learning
- `get_spectral_config_for_stem()` - Used by: processor, learning
- `calculate_geomean()` - Used by: processor, learning
- `should_keep_onset()` - Used by: processor
- `analyze_onset_spectral()` - Used by: processor, learning
- `prepare_midi_events_for_writing()` - Used by: midi module

**Status:** ✅ Excellent - proper functional core separation

---

### 3.2 Detection Functions

Centralized in `stems_to_midi_detection.py`:

- `detect_onsets()` - Used by: processor
- `detect_tom_pitch()` - Used by: processor
- `classify_tom_pitch()` - Used by: processor
- `detect_hihat_state()` - Used by: processor
- `estimate_velocity()` - Used by: processor

**Status:** ✅ Excellent - proper separation

---

### 3.3 Configuration Functions

Centralized in `stems_to_midi_config.py`:

- `load_config()` - Used by: all modules
- `DrumMapping` - Used by: all modules

**Status:** ✅ Excellent - proper separation

---

## 4. Detailed Findings Summary

### Files Analyzed

| File | Lines | Functions | Status |
|------|-------|-----------|--------|
| `stems_to_midi_config.py` | 79 | 2 | ✅ Clean |
| `stems_to_midi_detection.py` | 415 | 5 | ✅ Clean |
| `stems_to_midi_helpers.py` | 524 | 13 | ✅ Clean |
| `stems_to_midi_learning.py` | 328 | 2 | ⚠️ Minor issue |
| `stems_to_midi_midi.py` | 145 | 3 | ⚠️ Duplicate imports |
| `stems_to_midi_processor.py` | 443 | 1 | ✅ Clean |
| `stems_to_midi.py` | 328 | 1 + CLI | ✅ Clean |

---

## 5. Action Items

### Priority 1: Critical (Do Immediately)

1. **Remove duplicate imports in `stems_to_midi_midi.py`**
   - Delete lines 130-145 (duplicate import block)
   - Keep lines 1-12 (original import block)

### Priority 2: Minor Improvements (Optional)

2. **Consistency improvement in `stems_to_midi_learning.py`**
   - Line 70-71: Use `ensure_mono()` instead of `np.mean(audio, axis=1)`
   - This improves consistency with other modules

---

## 6. Architecture Assessment

### Strengths ✅

1. **Excellent functional core separation** - Pure functions in `helpers.py`
2. **Clear module boundaries** - Each file has a specific purpose
3. **Minimal coupling** - Modules import only what they need
4. **Good use of `__all__`** - Clear public interfaces
5. **Consistent patterns** - Similar problems solved similarly across modules

### Weaknesses ⚠️

1. **One file has duplicate imports** - `stems_to_midi_midi.py`
2. **Minor inconsistency** - `learning.py` doesn't use `ensure_mono()` helper
3. **Some complexity** - Long functions in `processor.py` and `learning.py` (but appropriate for their orchestration role)

---

## 7. Recommendations

### Immediate Actions

1. Fix the duplicate imports in `stems_to_midi_midi.py`
2. Consider standardizing on `ensure_mono()` in `learning.py`

### Long-term Considerations

1. **Keep the current architecture** - It's working well
2. **Monitor for duplication** as new features are added
3. **Consider extracting** some of the long filtering/analysis logic in `processor.py` if it grows further
4. **Document** the module boundaries in architecture docs (partially done)

---

## 8. Conclusion

The `stems_to_midi` codebase shows **excellent separation of concerns** with minimal duplication. The main issue is duplicate imports in one file, which is easy to fix. The architecture follows functional core/imperative shell principles well, with clear boundaries between:

- **Config** (`_config.py`) - Data structures and loading
- **Detection** (`_detection.py`) - Algorithm orchestration  
- **Helpers** (`_helpers.py`) - Pure functions (functional core)
- **MIDI** (`_midi.py`) - MIDI file I/O
- **Learning** (`_learning.py`) - Threshold calibration
- **Processor** (`_processor.py`) - Main processing pipeline
- **Main** (`stems_to_midi.py`) - CLI orchestration

**Overall Grade: A-** (would be A+ after fixing the duplicate imports)
