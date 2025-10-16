"""
Main processing pipeline for stems-to-MIDI conversion.

This module contains the core processing function that orchestrates
onset detection, spectral filtering, and MIDI event creation for drum stems.

Architecture: Imperative Shell
- Coordinates functional core helpers
- Handles I/O (audio file reading, printing)
- Orchestrates detection and analysis pipeline
"""

from pathlib import Path
from typing import Union, List, Dict, Optional
import numpy as np
import soundfile as sf

# Import functional core helpers
from stems_to_midi_helpers import (
    ensure_mono,
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean,
    should_keep_onset,
    normalize_values
)

# Import config and detection modules
from stems_to_midi_config import DrumMapping
from stems_to_midi_detection import (
    detect_onsets,
    detect_tom_pitch,
    classify_tom_pitch,
    detect_hihat_state,
    estimate_velocity
)


__all__ = ['process_stem_to_midi']


# Placeholder function - will be moved from stems_to_midi.py in Phase 6
def process_stem_to_midi(*args, **kwargs):
    """Placeholder - will be moved in Phase 6."""
    pass
