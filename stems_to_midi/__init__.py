"""
Stems to MIDI conversion package.

This package provides functionality to convert separated drum stems to MIDI tracks.

2026-06-22: trimmed to the PGA pipeline surface. The legacy
``learn_threshold_from_midi`` / ``save_calibrated_config`` learning-mode
helpers lived in the now-deleted ``stems_to_midi.learning`` module.
"""

from .config import load_config, DrumMapping
from .midi import create_midi_file, read_midi_notes, save_analysis_sidecar, load_analysis_sidecar, save_envelope_data, load_envelope_data
from .processing_shell import process_stem_to_midi
# from .pga_event_builder import build_pga_events
from .rebuild_core import rebuild_events_from_analysis
from .rebuild_shell import rebuild_midi_for_project

__all__ = [
    'load_config',
    'DrumMapping',
    'create_midi_file',
    'read_midi_notes',
    'save_analysis_sidecar',
    'load_analysis_sidecar',
    'save_envelope_data',
    'load_envelope_data',
    'process_stem_to_midi',
    'build_pga_events',
    'rebuild_events_from_analysis',
    'rebuild_midi_for_project',
]
