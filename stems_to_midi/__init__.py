"""
Stems to MIDI conversion package.

This package provides functionality to convert separated drum stems to MIDI tracks.
"""

from .config import load_config, DrumMapping
from .midi import create_midi_file, read_midi_notes
from .learning import learn_threshold_from_midi, save_calibrated_config
from .processor import process_stem_to_midi

__all__ = [
    'load_config',
    'DrumMapping',
    'create_midi_file',
    'read_midi_notes',
    'learn_threshold_from_midi',
    'save_calibrated_config',
    'process_stem_to_midi',
]
