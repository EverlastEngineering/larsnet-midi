# Root Modules

## analyze_clustering_results

```python
# analyze_clustering_results.py
```

Analyze and visualize clustering results for human review.

Creates a detailed markdown report showing:
- All detected onsets with features
- Cluster assignments
- Quality metrics
- Recommendations for parameter tuning

---

## analyze_reverb_continuations

```python
# analyze_reverb_continuations.py
```

Analyze reverb continuation patterns in analysis.json files.

Detects events where:
- Next event starts within 5ms of previous end
- Amplitude continuity (start amplitude matches previous end within 0.001)
- Velocity decreases (decay pattern)

---

## compare_detection_methods

```python
# compare_detection_methods.py
```

Compare librosa onset detection vs energy-based detection.

---

## device_shell

```python
# device_shell.py
```

Device detection and selection utilities for PyTorch.

Provides automatic device detection with priority: MPS → CUDA → CPU

---

## export_clustering_table

```python
# export_clustering_table.py
```

Export clustering data to CSV for analysis in spreadsheet.

---

## export_energy_detection_data

```python
# export_energy_detection_data.py
```

Export onset data using NEW energy-based detection (DAW-like method).

REPLACES LIBROSA DETECTION - Uses calibrated transient peak detection:
- Method: scipy.signal.find_peaks on RMS energy envelope
- Calibrated parameters (from iterative testing on Thunderstruck cymbals):
  * threshold_db = 15.0 (prominence above local minimum)
  * min_absolute_energy = 0.01 (noise floor for real hits)
  * min_peak_spacing_ms = 100.0 (prevent double-detection)

Result: Detects 75 events vs librosa's 238 (3.2x cleaner)
Includes obvious DAW events at 112s and 119s that manual tuning validated.

Shows L/R channels separately with geomean features for threshold tuning.

---

## export_raw_lr_data

```python
# export_raw_lr_data.py
```

Export RAW L/R channel data separately with ultra-high sensitivity.
Shows ALL detected onsets with separate features for left and right channels.

---

## generate_cymbal_midi_new_detection

```python
# generate_cymbal_midi_new_detection.py
```

Generate MIDI file for cymbals using NEW energy-based detection.

One-time script to validate new transient peak detection by creating a MIDI
file that can be overlaid in DAW to visually verify timing accuracy.

---

## mdx23c_optimized

```python
# mdx23c_optimized.py
```

Optimized MDX23C processing with batch support and performance improvements.

Key optimizations:
1. Batch processing of multiple chunks simultaneously
2. Reduced memory allocations with buffer reuse
3. Configurable overlap with quality/speed tradeoffs
4. Optional mixed precision support
5. Optimized STFT operations

---

## mdx23c_utils

```python
# mdx23c_utils.py
```

Utility helpers to load MDX23C-style checkpoints and run inference.

This supports two formats:
1. Legacy ConvTDFNet checkpoints with 'hyper_parameters' dict
2. Modern TFC_TDF_v3 checkpoints with separate YAML config files

The file also provides simple inference helpers that support:
- PyTorch Modules (torch.nn.Module)
- ONNX runtimes (onnxruntime.InferenceSession)

---

## midi_core

```python
# midi_core.py
```

MIDI Core - Functional Core

Pure functions for MIDI data transformations.
No side effects, no I/O - only calculations and data processing.

All functions take data as input and return transformed data.
File I/O is handled by midi_shell.py.

---

## midi_parser

```python
# midi_parser.py
```

MIDI Parser - Backwards Compatibility Wrapper

This module re-exports functions from midi_core.py and midi_shell.py
for backwards compatibility. New code should import from those modules directly.

DEPRECATED: Use midi_core.py (pure functions) and midi_shell.py (I/O) instead.

---

## midi_render_core

```python
# midi_render_core.py
```

MIDI Rendering Core - Pure Functions

Pure functions for MIDI note rendering calculations. These functions have NO side effects
and depend only on their input parameters. They can be used by any renderer (PIL, OpenCV,
ModernGL, etc.) by passing in the appropriate rendering configuration.

Functional Core Design:
- All functions are pure (deterministic, no I/O, no mutation)
- Take explicit parameters instead of reading from objects/globals
- Return new values instead of modifying state
- Can be tested in isolation without mocking

Used by: render_midi_to_video.py (PIL/OpenCV), moderngl_renderer (future)

---

## midi_shell

```python
# midi_shell.py
```

MIDI Shell - Imperative Shell

Handles file I/O and side effects for MIDI parsing.
Loads MIDI files and delegates to pure functions in midi_core.py.

This is the "shell" that wraps the functional "core".

---

## midi_types

```python
# midi_types.py
```

MIDI Data Types - Shared Contract

Defines the data contract between MIDI parsing and rendering systems.
This allows MIDI extraction to be decoupled from rendering implementation.

Type Hierarchy:
    MidiNote (base) → can be used by any renderer
    DrumNote (specialized) → includes rendering metadata (lane, color)
    
Detection Output Contract:
    SpectralOnsetData → standardized spectral analysis fields for onset data
    
See docs/DETECTION_OUTPUT_CONTRACT.md for full specification.

---

## normalize_yaml

```python
# normalize_yaml.py
```

Normalize YAML files for comparison by alphabetizing keys and removing comments.

Usage:
    python normalize_yaml.py input.yaml [output.yaml]
    
If output.yaml is not specified, prints to stdout.

---

## project_manager

```python
# project_manager.py
```

Project Manager - Functional Core for DrumToMIDI Project Management

Manages user projects in the user_files/ directory with auto-numbering,
metadata tracking, and per-project configuration files.

Architecture: Functional Core
- Pure functions for project discovery, validation, and data transformation
- No side effects (file I/O) except in clearly marked functions
- All logic testable without touching filesystem

Project Structure:
    user_files/
    └── 1 - song name/
        ├── .drumtomidi_project.json    # Metadata
        ├── midiconfig.yaml          # Project-specific MIDI config (optional)
        ├── song name.wav            # Original audio
        ├── stems/                   # Separated stems
        ├── cleaned/                 # Cleaned stems
        ├── midi/                    # Generated MIDI
        └── video/                   # Rendered videos

---

## render_midi_video_shell

```python
# render_midi_video_shell.py
```

MIDI to Rock Band-Style Video Renderer

Creates falling notes visualization videos from MIDI drum files, 
perfect for learning to play drums Rock Band style.

Uses project-based workflow: automatically detects projects with MIDI files
and renders videos to the project/video/ directory.

Usage:
    python render_midi_to_video.py              # Auto-detect project
    python render_midi_to_video.py 1            # Render specific project
    python render_midi_to_video.py --fps 60     # Custom settings

---

## render_video_core

```python
# render_video_core.py
```

Video Rendering Core - Functional Core

Pure functions for image conversion, canvas operations, and drawing primitives.
No side effects: no file I/O, no OpenGL/GPU context, no logging.

Architecture: Functional core (this file) called by imperative shell (render_midi_video_shell.py)

---

## separate

```python
# separate.py
```

Separate drums into individual stems using MDX23C.

Uses project-based workflow: automatically detects projects in user_files/
or processes new audio files dropped there.

Usage:
    python separate.py              # Auto-detect project (uses MDX23C)
    python separate.py 1            # Process specific project by number
    python separate.py --device cuda  # Use GPU acceleration
    python separate.py --overlap 8  # High quality separation (slower)

---

## separation_shell

```python
# separation_shell.py
```

Shared utilities for drum separation.

---

## sidechain_core

```python
# sidechain_core.py
```

Sidechain Compression - Functional Core

Pure audio processing functions for envelope following and sidechain compression.
No side effects: no printing, no file I/O, no logging.

Architecture: Functional core (this file) called by imperative shell (sidechain_shell.py)

---

## sidechain_shell

```python
# sidechain_shell.py
```

Sidechain compression to reduce bleed between stems - Imperative Shell

Uses the separated snare track as a sidechain trigger to duck the kick track
when the snare is playing, effectively removing snare bleed from the kick.

Uses project-based workflow: automatically detects projects with stems
and creates cleaned versions in the project/cleaned/ directory.

Architecture: Imperative shell (this file) using functional core (sidechain_core.py)

Usage:
    python sidechain_cleanup.py              # Auto-detect project
    python sidechain_cleanup.py 1            # Process specific project

---

## stems_to_midi_cli

```python
# stems_to_midi_cli.py
```

Convert separated drum stems to MIDI tracks.

Uses project-based workflow: automatically detects projects with stems
and generates MIDI files in the project/midi/ directory.

Architecture: Modular Design (Functional Core, Imperative Shell)
- stems_to_midi/ submodules: Core conversion logic
- project_manager: Project discovery and management
- stems_to_midi_cli.py (this file): CLI orchestration

Usage:
    python stems_to_midi_cli.py              # Auto-detect project
    python stems_to_midi_cli.py 1            # Process specific project
    python stems_to_midi_cli.py --learn      # Learning mode

---

## test_compare_renderers

```python
# test_compare_renderers.py
```

Compare PIL vs GPU rendering for the same note

---

## test_coordinate_system

```python
# test_coordinate_system.py
```

Test coordinate system conversions to sanity check OpenGL vs pixel space.

This script tests all coordinate conversion functions with real values
to verify they produce correct results.

---

## test_cpu_threading

```python
# test_cpu_threading.py
```

Test script to verify CPU threading configuration.

Usage:
    conda run -n drumtomidi python test_cpu_threading.py

---

## test_cv2_rendering

```python
# test_cv2_rendering.py
```

Tests for OpenCV rendering helpers (Phase 1)

Validates that OpenCV drawing functions produce visually similar output to PIL.

---

## test_dual_sensitivity

```python
# test_dual_sensitivity.py
```

Tests for Step 3: Dual-Sensitivity Detection

Verifies:
- _run_sensitive_detection() finds events with max-sensitivity params
- _serialize_onset_events() correctly rounds and serializes onset data
- save_analysis_sidecar() writes v3 format with events_configured + events_sensitive
- Sensitive detection produces >= configured event count
- All sensitive events have spectral features pre-computed

---

## test_energy_detection_integration

```python
# test_energy_detection_integration.py
```

Test energy-based detection integration.

Verifies that:
1. Energy detection is used by default
2. Librosa fallback works when enabled
3. Config parameters are loaded correctly

---

## test_gpu_coordinate_debug

```python
# test_gpu_coordinate_debug.py
```

Debug script to trace through GPU coordinate calculations

---

## test_integration

```python
# test_integration.py
```

Integration tests for the DrumToMIDI pipeline.

Tests the complete workflow: separate → cleanup → midi → video
to ensure refactoring doesn't break functionality.

These tests use synthetic audio/stems to run quickly without ML models.

---

## test_mdx23c_utils

```python
# test_mdx23c_utils.py
```

Test script for MDX23C model loading utilities.

---

## test_mdx_performance

```python
# test_mdx_performance.py
```

Test script to compare MDX23C performance: original vs optimized.

This benchmarks both implementations to quantify speed improvements.

---

## test_midi_core

```python
# test_midi_core.py
```

Tests for MIDI Core - Functional Core

Tests pure functions that process MIDI data.
No file I/O in these tests - only data transformations.

---

## test_midi_parser

```python
# test_midi_parser.py
```

Tests for MIDI Parser

Tests the pure parsing functions without any rendering logic.

---

## test_midi_render_core

```python
# test_midi_render_core.py
```

Tests for midi_render_core.py - Pure rendering calculation functions

These tests verify the functional core of the MIDI rendering system.
All functions are pure (no side effects), so tests are straightforward
and don't require mocking or fixtures.

---

## test_midi_shell

```python
# test_midi_shell.py
```

Tests for MIDI Shell - Imperative Shell

Tests file I/O and integration with the functional core.

---

## test_midi_types

```python
# test_midi_types.py
```

Tests for MIDI Types - Data Contract Validation

Tests the shared type definitions used by MIDI parsers and renderers.

---

## test_normalization

```python
# test_normalization.py
```

Test amplitude normalization on hihat.

---

## test_note_classification_core

```python
# test_note_classification_core.py
```

Tests for Note Classification Core — Pure Functional Tests

Tests the two-pass note classification system that assigns MIDI notes
based on stored spectral features. No audio, no I/O.

---

## test_optimization_real_audio

```python
# test_optimization_real_audio.py
```

Test optimization on real audio - Thunderstruck cymbals

This script validates the optimization loop on actual cymbal audio
to verify it reduces false positives compared to hard-coded thresholds.

---

## test_project_manager

```python
# test_project_manager.py
```

Tests for project_manager.py

Tests the functional core and imperative shell of project management.
Uses temporary directories for file system tests.

---

## test_pure_opencv_speed

```python
# test_pure_opencv_speed.py
```

Test pure OpenCV rendering speed without PIL conversions.

This skips format conversions to isolate OpenCV performance.
These are benchmark tests that require specific project data to exist.

---

## test_render_video_core

```python
# test_render_video_core.py
```

Unit tests for render_video_core.py - Functional Core

Tests pure image conversion and drawing functions with no side effects.
Aims for 95%+ coverage with fast, deterministic tests.

---

## test_separate

```python
# test_separate.py
```

Tests for separate.py with project-based workflow.

Tests the integration between separate.py and project_manager.

---

## test_sidechain_core

```python
# test_sidechain_core.py
```

Unit tests for sidechain_core.py - Functional Core

Tests pure audio processing functions with no side effects.
Aims for 95%+ coverage with fast, deterministic tests.

---

## test_stem_comparison

```python
# test_stem_comparison.py
```

Compare energy-based vs librosa detection for each stem individually.
Processes project 14 (Thunderstruck) one stem at a time in both modes.

---

## test_threshold_sweep

```python
# test_threshold_sweep.py
```

Threshold sweep test - find optimal threshold for Thunderstruck cymbals

This script tests different threshold values to understand the relationship
between threshold and onset count.

---
