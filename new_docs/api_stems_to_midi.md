# stems_to_midi Package

## analysis_core

```python
# analysis_core.py
```

Pure helper functions for stem to MIDI conversion.

These are functional core functions - pure, deterministic, no I/O or side effects.
All audio processing logic extracted here for testability.

Detection Output Contract (Producer):
    filter_onsets_by_spectral() produces SpectralOnsetData dicts.
    Contract defined in midi_types.py - see SpectralOnsetData TypedDict.
    Consumers: detection_shell.detect_hihat_state(), learning.py

---

## clustering_core

```python
# clustering_core.py
```

Clustering Core - Pure Functional Core

Pure functions for clustering onset features to identify distinct instruments.
All functions are deterministic with no side effects.

Architecture: Functional Core
- No I/O operations
- Deterministic (same input → same output)
- No external state or side effects
- Testable in isolation

---

## config

```python
# config.py
```

Configuration and data structures for stems-to-MIDI conversion.

This module provides configuration loading and data classes used throughout
the stems-to-MIDI processing pipeline.

Architecture: Part of the Imperative Shell
- Handles I/O (YAML file loading)
- Provides data structures for coordination

---

## detection_shell

```python
# detection_shell.py
```

Audio analysis and detection algorithms for stems-to-MIDI conversion.

This module provides algorithm coordinators for detecting drum hits and analyzing audio.
These functions orchestrate complex multi-step algorithms using librosa and other libraries.

Architecture: Imperative Shell (Algorithm Coordinators)
- Coordinates external library calls (librosa, sklearn)
- Uses functional core helpers for pure logic
- Delegates pure transformations to stems_to_midi_helpers

Detection Output Contract:
- This module CONSUMES SpectralOnsetData from analysis_core.py
- Contract defined in midi_types.py (SpectralOnsetData TypedDict)
- Uses: body_energy, sizzle_energy for hihat open/closed classification

Note: This module contains coordinators, not pure functions. Pure functions are in helpers.

---

## energy_detection_core

```python
# energy_detection_core.py
```

Energy-based onset detection - visual/DAW-like approach.

REPLACES LIBROSA ONSET DETECTION which had fundamental flaws for isolated drums:
1. librosa wait periods (wait=3 frames) caused real events to be skipped
2. Decay comparison to zero triggered false detections during long reverb
3. Generic music tuning doesn't work on clean separated drum stems
4. Over-detected: Thunderstruck 238 events vs 54 actual (4.4x too many)

NEW APPROACH uses energy analysis similar to how DAWs render waveforms:
- Calculate RMS energy envelope (what you SEE in DAW waveform)
- Find peaks using scipy.signal.find_peaks (robust, battle-tested)
- Filter by prominence (15dB above local minimum, calibrated)
- Backtrack from peak to attack start using left_bases + threshold (50-120ms earlier)
- No blind wait periods - every peak evaluated independently
- Result: 72 events detected at attack start, matching drummer timing

Parameters calibrated through iterative testing:
- threshold_db = 15.0 (prominence above local minimum)
- min_absolute_energy = 0.01 (noise floor for real cymbal hits)
- min_peak_spacing_ms = 100.0 (prevent double-detection, not blind wait)

Pure functional core - no side effects.

---

## energy_detection_shell

```python
# energy_detection_shell.py
```

Energy-based detection shell - drop-in replacement for detect_onsets.

This module provides a bridge between the new energy_detection_core and
the existing processing pipeline. It wraps detect_stereo_transient_peaks
to match the interface expected by processing_shell.py.

---

## learning

```python
# learning.py
```

Learning Mode Module

Handles threshold learning from user-edited MIDI files.

Detection Output Contract (Consumer):
    This module CONSUMES SpectralOnsetData from analysis_core.py.
    Uses: domain-specific band energies (body_energy, wire_energy, etc.), strength, amplitude
    Contract defined in midi_types.py - see SpectralOnsetData TypedDict.

---

## midi

```python
# midi.py
```

MIDI File Operations Module

Handles creation and reading of MIDI files for drum transcription.
Includes JSON sidecar export for spectral analysis data (Detection Output Contract).

---

## note_classification_core

```python
# note_classification_core.py
```

Note Classification — Functional Core

Classifies MIDI note assignments from stored spectral features on the
final KEPT event set. Runs identically in both the full pipeline and
the rebuild pipeline.

Pass 2 of the two-pass architecture:
  Pass 1: Detect onsets, compute spectral features, apply threshold filters.
  Pass 2: Classify notes from stored features on KEPT events only (this module).

Pure functions — no I/O, no audio, no side effects.

---

## optimization_core

```python
# optimization_core.py
```

Threshold Optimization Core - Pure Functional Core

Pure functions for optimizing onset detection thresholds using clustering.
All functions are deterministic with no side effects.

Architecture: Functional Core
- No I/O operations
- Deterministic (same input → same output, given fixed random seeds)
- No external state or side effects
- Testable in isolation

The optimization strategy:
1. Detect onsets with current threshold
2. Extract features from onsets
3. Cluster features
4. Compare cluster count to expected
5. Adjust threshold based on cluster count (binary search)
6. Repeat until convergence or max iterations

---

## processing_shell

```python
# processing_shell.py
```

Stem Processing Module

Handles the main processing pipeline for converting audio stems to MIDI events.

---

## rebuild_core

```python
# rebuild_core.py
```

Rebuild MIDI from Analysis — Functional Core

Re-filters cached detection results from analysis.json and produces
MIDI-ready events without re-running audio detection. This enables
sub-second parameter tuning iteration.

The rebuild operates in two modes:
- **Same thresholds**: Trust stored statuses from analysis.json exactly.
  The full pipeline applied multi-pass filtering (geomean, decay, statistical,
  reverb continuation) that cannot be replicated without audio.
- **Changed thresholds**: Re-apply geomean/sustain filtering (Pass 1) to
  events_configured. Merge sensitive events only when thresholds are lowered
  to discover events the original pipeline would not have found.

After filtering, note classification (Pass 2) runs on the final KEPT set
using stored spectral features (spectral_centroid_hz, sustain_ms, energy
bands). This ensures note assignments (open/closed hihat, crash/ride/chinese,
low/mid/high tom, snare types) reflect the actual event population.

Pure functions — no I/O, no side effects.

---

## rebuild_shell

```python
# rebuild_shell.py
```

Rebuild MIDI from Analysis — Imperative Shell

Handles I/O for the rebuild-from-analysis pipeline: loading analysis.json,
reading overrides, applying config updates, writing MIDI, and updating
sidecar files.

This module is the thin I/O wrapper around rebuild_core.py.

---

## stereo_core

```python
# stereo_core.py
```

Stereo Audio Analysis - Pure Functional Core

Pure functions for analyzing stereo audio and extracting spatial information.
All functions are deterministic with no side effects.

Architecture: Functional Core
- No I/O operations
- Deterministic (same input → same output)
- No external state or side effects
- Testable in isolation

---

## test_analysis_core

```python
# test_analysis_core.py
```

Tests for pure analysis functions (functional core).

These functions have no side effects and are easy to test.

---

## test_analysis_core_features

```python
# test_analysis_core_features.py
```

Tests for extract_onset_features() in analysis_core.py

Tests feature extraction for clustering-based threshold optimization.

---

## test_clustering_core

```python
# test_clustering_core.py
```

Tests for clustering_core.py - Pure Functional Core

Tests for onset clustering algorithms (DBSCAN and k-means).

---

## test_detection_shell

```python
# test_detection_shell.py
```

Comprehensive tests for stems_to_midi.detection module.

These tests provide complete coverage of the detection algorithms.

---

## test_learning

```python
# test_learning.py
```

Tests for learning.py module - threshold learning from edited MIDI files.

---

## test_optimization_core

```python
# test_optimization_core.py
```

Tests for optimization_core.py - Threshold Optimization

Tests for threshold optimization loop using clustering.

---

## test_rebuild_core

```python
# test_rebuild_core.py
```

Tests for rebuild_core.py — Rebuild MIDI from Analysis.

Tests the pure functional core that re-filters cached detection results
and produces MIDI-ready events without audio I/O.

---

## test_stems_to_midi

```python
# test_stems_to_midi.py
```

Test suite for stems_to_midi.py

Run with: pytest test_stems_to_midi.py -v

---

## test_stereo_core

```python
# test_stereo_core.py
```

Tests for stereo_core.py - Stereo Audio Analysis

Tests pure functions for analyzing stereo audio and extracting spatial information.

---
