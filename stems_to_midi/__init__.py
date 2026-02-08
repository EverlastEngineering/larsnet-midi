"""
Stems to MIDI conversion package.

This package provides functionality to convert separated drum stems to MIDI tracks.
"""

from .config import load_config, DrumMapping
from .midi import create_midi_file, read_midi_notes, save_analysis_sidecar, load_analysis_sidecar, save_envelope_data, load_envelope_data
from .learning import learn_threshold_from_midi, save_calibrated_config
from .processing_shell import process_stem_to_midi

__all__ = [
    'load_config',
    'DrumMapping',
    'create_midi_file',
    'read_midi_notes',
    'save_analysis_sidecar',
    'load_analysis_sidecar',
    'save_envelope_data',
    'load_envelope_data',
    'learn_threshold_from_midi',
    'save_calibrated_config',
    'process_stem_to_midi',
]
