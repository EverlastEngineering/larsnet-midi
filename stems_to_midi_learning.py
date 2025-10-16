"""
Learning mode and threshold calibration for stems-to-MIDI conversion.

This module provides functions for learning optimal detection thresholds
by comparing user-edited MIDI files with original detections.

Architecture: Mix of Functional Core and Imperative Shell
- Uses functional core helpers for analysis
- Handles I/O (file reading, printing, config saving)
- Coordinates threshold learning workflow
"""

from pathlib import Path
from typing import Union, Dict
import numpy as np
import soundfile as sf
import librosa
import yaml

# Import functional core helpers
from stems_to_midi_helpers import (
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean
)

# Import config and detection modules
from stems_to_midi_config import DrumMapping
from stems_to_midi_midi import read_midi_notes


__all__ = ['learn_threshold_from_midi', 'save_calibrated_config']


# Placeholder functions - will be moved from stems_to_midi.py in Phase 5
def learn_threshold_from_midi(*args, **kwargs):
    """Placeholder - will be moved in Phase 5."""
    pass

def save_calibrated_config(*args, **kwargs):
    """Placeholder - will be moved in Phase 5."""
    pass
