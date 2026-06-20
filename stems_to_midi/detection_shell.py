"""
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
"""

from typing import Tuple, List, Dict
import numpy as np
import librosa

# Import contract types from parent module
try:
    from midi_types import SpectralOnsetData
except ImportError:
    # Running from stems_to_midi/ directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from midi_types import SpectralOnsetData

# Import functional core helpers
from .analysis_core import (
    ensure_mono,
    calculate_sustain_duration,
    estimate_velocity,
    classify_tom_pitch,
    calculate_peak_amplitude,
)

# Import config


__all__ = [
    'detect_onsets',
    'detect_tom_pitch',
    'detect_hihat_state',
    # Re-export pure functions from helpers for backwards compatibility
    'estimate_velocity',
    'classify_tom_pitch'
]


