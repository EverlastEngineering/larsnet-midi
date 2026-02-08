"""
MIDI File Operations Module

Handles creation and reading of MIDI files for drum transcription.
Includes JSON sidecar export for spectral analysis data (Detection Output Contract).
"""

from midiutil import MIDIFile
import mido
import json
from pathlib import Path
from typing import Dict, List, Union, Optional

# Import helper function for event preparation
from .analysis_core import prepare_midi_events_for_writing

# Import contract for validation
try:
    pass
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'create_midi_file',
    'read_midi_notes',
    'save_analysis_sidecar',
    'load_analysis_sidecar'
]


def create_midi_file(
    events_by_stem: Dict[str, List[Dict]],
    output_path: Union[str, Path],
    tempo: float = 120.0,
    track_name: str = "Drums",
    config: Optional[Dict] = None
):
    """
    Create a MIDI file from detected drum events.
    
    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        output_path: Path to save MIDI file
        tempo: Tempo in BPM
        track_name: Name of the MIDI track
        config: Configuration dictionary (optional, loads default if not provided)
    """
    # Import here to avoid circular dependency
    from .config import load_config
    
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Create MIDI file with 1 track
    midi = MIDIFile(1)
    track = 0
    channel = 9  # Channel 10 (0-indexed as 9) is typically drums in MIDI
    time = 0
    
    midi.addTrackName(track, time, track_name)
    midi.addTempo(track, time, tempo)
    
    # Add a marker/text event at time 0 to anchor the MIDI file
    # This ensures proper alignment when importing into DAWs
    midi.addText(track, 0.0, "START")
    
    # Also add a very quiet anchor note at time 0 (velocity 1, not 0)
    # Some DAWs filter out velocity 0 notes
    very_short_duration = config.get('audio', {}).get('very_short_duration', 0.01)
    midi.addNote(
        track=track,
        channel=9,
        pitch=27,  # Very low note (outside typical drum range)
        time=0.0,  # At the very start
        duration=very_short_duration,  # Very short (beats)
        volume=1  # Very quiet but not silent (velocity 1)
    )
    
    # Prepare all events (convert times to beats using pure function)
    prepared_events = prepare_midi_events_for_writing(events_by_stem, tempo)
    
    # Add all prepared events to MIDI file
    for event in prepared_events:
        midi.addNote(
            track=track,
            channel=channel,
            pitch=event['note'],
            time=event['time_beats'],
            duration=event['duration_beats'],
            volume=event['velocity']
        )
    
    total_events = len(prepared_events)
    
    # Write to file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    print(f"  Created MIDI file with {total_events} notes")


def read_midi_notes(midi_path: Union[str, Path], target_note: int) -> List[float]:
    """
    Read note times from a MIDI file for a specific MIDI note number.
    
    Args:
        midi_path: Path to MIDI file
        target_note: MIDI note number to extract (e.g., 38 for snare)
    
    Returns:
        List of note times in seconds
    """
    midi_file = mido.MidiFile(str(midi_path))
    note_times = []
    current_time = 0.0
    
    # Get ticks per beat for time conversion
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500000  # Default tempo (120 BPM in microseconds per beat)
    
    for track in midi_file.tracks:
        current_time = 0.0
        for msg in track:
            current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
            
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.note == target_note and msg.velocity > 0:
                note_times.append(current_time)
    
    return sorted(note_times)


def _round_value(value, decimals: int):
    """Round numeric value to specified decimals, handle None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(value, decimals)
    return value


def save_analysis_sidecar(
    events_by_stem: Dict[str, List[Dict]],
    midi_path: Union[str, Path],
    tempo: float = 120.0,
    analysis_by_stem: Optional[Dict[str, Dict]] = None
) -> Path:
    """
    Save spectral analysis data as JSON sidecar file (v2 format).
    
    V2 Format:
        - Logic block per stem (thresholds, passes)
        - All onsets included (KEPT + FILTERED)
        - Numeric precision: times=4 decimals, features=2 decimals
    
    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        midi_path: Path to corresponding MIDI file (sidecar uses same name + .analysis.json)
        tempo: Tempo in BPM (for reference)
        analysis_by_stem: Dict with all_onset_data and spectral_config per stem
    
    Returns:
        Path to created sidecar file
    """
    midi_path = Path(midi_path)
    sidecar_path = midi_path.with_suffix('.analysis.json')
    
    # Build sidecar structure v2
    sidecar_data = {
        'version': '2.0',
        'tempo_bpm': round(tempo, 1),
        'stems': {}
    }
    
    total_events = 0
    total_filtered = 0
    
    for stem_type, events in events_by_stem.items():
        # Get analysis data for this stem
        analysis = analysis_by_stem.get(stem_type, {}) if analysis_by_stem else {}
        all_onset_data = analysis.get('all_onset_data', [])
        spectral_config = analysis.get('spectral_config')
        
        # Build logic block from spectral_config
        logic = {}
        if spectral_config:
            logic['geomean_threshold'] = _round_value(spectral_config.get('geomean_threshold'), 2)
            logic['min_sustain_ms'] = _round_value(spectral_config.get('min_sustain_ms'), 2)
            
            # Record frequency band metadata so sidecar is self-documenting
            geomean_bands = spectral_config.get('geomean_bands', [])
            if geomean_bands:
                logic['freq_bands'] = geomean_bands
            
            # Cymbal-specific logic
            if stem_type == 'cymbals':
                logic['decay_filter_enabled'] = spectral_config.get('decay_filter_enabled', True)
                logic['decay_window_sec'] = _round_value(spectral_config.get('decay_window_sec'), 2)
            
            # Kick-specific logic
            if stem_type == 'kick':
                logic['statistical_enabled'] = spectral_config.get('statistical_enabled', False)
            
            # Determine passes (simplified - could be made more sophisticated)
            passes = ['geomean']
            if spectral_config.get('min_sustain_ms'):
                passes.append('sustain')
            if stem_type == 'cymbals' and logic.get('decay_filter_enabled'):
                passes.append('decay')
            if stem_type == 'kick' and logic.get('statistical_enabled'):
                passes.append('statistical')
            logic['passes'] = passes
        
        # Build events list from all_onset_data (includes KEPT + FILTERED)
        stem_events = []
        if all_onset_data:
            # Match MIDI events to KEPT onsets by index (they're in the same order)
            # Filter to only regular MIDI events (exclude foot-close events)
            midi_events = [e for e in events if e.get('note') != 44]  # 44 is foot-close note
            midi_idx = 0
            
            for onset_data in all_onset_data:
                event = {
                    'time': _round_value(onset_data.get('time'), 4),
                    'status': onset_data.get('status', 'UNKNOWN')
                }
                
                # Add spectral features with rounding
                # Band energy fields are dynamic per stem (e.g., body_energy, wire_energy)
                band_fields = [f'{b}_energy' for b in onset_data.get('geomean_bands', [])]
                for field in ['strength', 'amplitude'] + band_fields + ['geomean', 'total_energy', 'sustain_ms']:
                    value = onset_data.get(field)
                    if value is not None:
                        event[field] = _round_value(value, 2)
                
                # Add Phase 2 metadata fields (enriched metadata)
                for field in ['duration_sec', 'amplitude_at_start', 'amplitude_at_end',
                             'attack_sharpness', 'envelope_continuity', 'peak_prominence',
                             'spectral_centroid_hz', 'spectral_flux', 'pitch_hz',
                             'gap_from_previous_sec']:
                    value = onset_data.get(field)
                    if value is not None:
                        event[field] = _round_value(value, 4)
                
                # Add MIDI fields for KEPT events (from events_by_stem by index)
                if event['status'] == 'KEPT':
                    if midi_idx < len(midi_events):
                        event['note'] = midi_events[midi_idx].get('note')
                        event['velocity'] = midi_events[midi_idx].get('velocity')
                        midi_idx += 1
                
                stem_events.append(event)
                total_events += 1
                if event['status'] == 'FILTERED':
                    total_filtered += 1
        else:
            # Fallback: use events_by_stem if no all_onset_data
            for midi_event in events:
                event = {
                    'time': _round_value(midi_event.get('time'), 4),
                    'note': midi_event.get('note'),
                    'velocity': midi_event.get('velocity'),
                    'status': 'KEPT'
                }
                
                # Add spectral fields if present
                # Band energy fields are dynamic per stem (e.g., body_energy, wire_energy)
                band_fields = [f'{b}_energy' for b in midi_event.get('geomean_bands', [])]
                for field in ['onset_strength', 'peak_amplitude'] + band_fields + ['geomean',
                             'total_energy', 'sustain_ms']:
                    value = midi_event.get(field)
                    if value is not None:
                        event[field] = _round_value(value, 2)
                
                stem_events.append(event)
                total_events += 1
        
        # Assemble stem data
        sidecar_data['stems'][stem_type] = {
            'logic': logic,
            'events': stem_events
        }
    
    # Write JSON
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar_data, f, indent=2)
    
    print(f"  Saved analysis sidecar v2: {sidecar_path.name} ({total_events} total events, {total_filtered} filtered)")
    
    return sidecar_path


def load_analysis_sidecar(midi_path: Union[str, Path]) -> Optional[Dict]:
    """
    Load spectral analysis data from JSON sidecar file.
    
    Args:
        midi_path: Path to MIDI file (will look for .analysis.json sidecar)
    
    Returns:
        Sidecar data dict, or None if not found
    """
    midi_path = Path(midi_path)
    sidecar_path = midi_path.with_suffix('.analysis.json')
    
    if not sidecar_path.exists():
        return None
    
    with open(sidecar_path, 'r') as f:
        return json.load(f)

