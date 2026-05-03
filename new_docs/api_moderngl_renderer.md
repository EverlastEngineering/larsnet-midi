# moderngl_renderer Package

## animation

```python
# animation.py
```

Animation Core - Functional Core

Pure functions for time-based animation calculations.
No side effects, no GPU operations - only animation math.

Used by animation_shell.py to generate frame sequences.

---

## core

```python
# core.py
```

ModernGL Renderer - Functional Core

Pure functions for data transformations.
No side effects, no GPU operations - only calculations.

Follows functional core, imperative shell pattern:
- This module: Pure transformations (testable, predictable)
- moderngl_shell.py: GPU operations (side effects)

---

## midi_animation

```python
# midi_animation.py
```

MIDI Animation Bridge - Functional Core

Converts MIDI DrumNote data into animation-compatible format.
Pure functions only - no side effects, no GPU operations.

This bridges between MIDI parsing (midi_shell.py) and GPU rendering (animation.py).

---

## midi_video_core

```python
# midi_video_core.py
```

MIDI Video Rendering Core - Functional Core

Pure functions for MIDI video rendering calculations.
No side effects, no GPU operations - only transformations.

Used by midi_video_moderngl.py (imperative shell) for video rendering.

---

## midi_video_shell

```python
# midi_video_shell.py
```

ModernGL MIDI Video Renderer - Imperative Shell

High-performance GPU-accelerated MIDI to video rendering using ModernGL.
Provides the same interface as the PIL renderer but with ~2x real-time speedup.

Architecture:
- Functional core: midi_animation.py (pure functions, coordinate calculations)
- Imperative shell: This file (GPU rendering, FFmpeg encoding, I/O)

Usage:
    from moderngl_renderer.midi_video_shell import render_midi_to_video_moderngl
    
    render_midi_to_video_moderngl(
        midi_path="path/to/file.mid",
        output_path="output.mp4",
        audio_path="audio.wav",  # optional
        width=1920,
        height=1080,
        fps=60
    )

---

## shell

```python
# shell.py
```

ModernGL Renderer - Imperative Shell

Handles all GPU operations and side effects.
Uses pure functions from moderngl_core for calculations.

Follows functional core, imperative shell pattern:
- moderngl_core.py: Pure transformations (testable, predictable)
- This module: GPU operations (side effects, resources, I/O)

---

## test_animation

```python
# test_animation.py
```

Tests for animation system functional core

Tests pure functions for time-based animation calculations.
No GPU operations - only animation math.

---

## test_core

```python
# test_core.py
```

Tests for ModernGL renderer functional core

Tests pure functions that transform data for GPU rendering.
No GPU operations in these tests - only data transformations.

---

## test_fade_logic

```python
# test_fade_logic.py
```

Test fade logic to debug

---

## test_midi_animation

```python
# test_midi_animation.py
```

Tests for MIDI Animation Bridge

Tests the conversion from DrumNote to animation format.

---

## test_midi_render_simple

```python
# test_midi_render_simple.py
```

Simple test: Render one frame from project 13 MIDI

Loads project 13 MIDI, converts to animation format,
then renders a single frame using shell.py to verify integration.

---

## test_midi_video_core

```python
# test_midi_video_core.py
```

Tests for MIDI Video Core - Functional Core

Tests pure functions in midi_video_core.py.
All functions are pure (no side effects), so tests are deterministic.

---

## test_midi_video_moderngl

```python
# test_midi_video_moderngl.py
```

Tests for MIDI Video Renderer - Imperative Shell

Level 1 & 2 integration tests for midi_video_shell.py.
Tests the full rendering pipeline without mocking.

Note: These are integration tests that exercise GPU, FFmpeg, and file I/O.
They test observable behavior, not implementation details.

---

## test_shell

```python
# test_shell.py
```

Integration tests for ModernGL imperative shell

Tests GPU operations and rendering pipeline without mocking.
Uses 3-tier approach:
  1. Smoke tests - fast sanity checks
  2. Property tests - verify behavior invariants
  3. Regression tests - pixel-perfect comparisons (manual/pre-release)

These tests exercise the entire GPU pipeline (shaders, framebuffers, blending)
but use smart assertions to remain robust to implementation changes.

---

## test_visual_quality

```python
# test_visual_quality.py
```

Visual Quality Test

Creates test images to verify rendering quality:
- Anti-aliasing on rounded corners
- Color accuracy
- Alpha blending
- Lane alignment

---

## text_overlay_shell

```python
# text_overlay_shell.py
```

Text Overlay Generation - Functional Core

Pure functions for generating text overlays (lane labels).
Renders text using PIL, returns PIL Image with alpha channel.

The imperative shell (midi_video_shell.py) handles uploading to GPU texture.

---
