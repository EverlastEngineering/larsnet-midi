"""
MIDI File Operations Module

Handles creation and reading of MIDI files for drum transcription.
Includes JSON sidecar export for spectral analysis data (Detection Output Contract).
"""

from midiutil import MIDIFile
import mido
import json
import numpy as np
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
    'load_analysis_sidecar',
    'save_envelope_data',
    'load_envelope_data'
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


def _serialize_onset_events(
    onset_data_list: list,
    midi_events: Optional[List[Dict]] = None,
) -> list:
    """
    Serialize onset data dicts into rounded JSON-ready event dicts.

    Args:
        onset_data_list: List of onset dicts from filter_onsets_by_spectral
        midi_events: Optional list of MIDI events to attach note/velocity to KEPT onsets.
                     Should exclude foot-close events (note 44).

    Returns:
        List of serialized event dicts with rounded numeric values.
    """
    events = []
    midi_idx = 0

    for onset_data in onset_data_list:
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

        # Add MIDI fields for KEPT events (from midi_events by index)
        if midi_events is not None and event['status'] == 'KEPT':
            if midi_idx < len(midi_events):
                event['note'] = midi_events[midi_idx].get('note')
                event['velocity'] = midi_events[midi_idx].get('velocity')
                midi_idx += 1

        events.append(event)

    return events


def save_analysis_sidecar(
    events_by_stem: Dict[str, List[Dict]],
    midi_path: Union[str, Path],
    tempo: float = 120.0,
    analysis_by_stem: Optional[Dict[str, Dict]] = None,
    config: Optional[Dict] = None,
) -> Path:
    """
    Save spectral analysis data as JSON sidecar file (v3 format).

    V3 Format:
        - Logic block per stem (thresholds, passes)
        - events_configured: All onsets from configured detection (KEPT + FILTERED)
        - events_sensitive: All onsets from max-sensitivity detection (for interactive tuning)
        - Numeric precision: times=4 decimals, features=2 decimals

    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        midi_path: Path to corresponding MIDI file (sidecar uses same name + .analysis.json)
        tempo: Tempo in BPM (for reference)
        analysis_by_stem: Dict with all_onset_data, sensitive_onset_data, and spectral_config per stem

    Returns:
        Path to created sidecar file
    """
    midi_path = Path(midi_path)
    sidecar_path = midi_path.with_suffix('.analysis.json')

    sidecar_data = {
        'version': '3.0',
        'tempo_bpm': round(tempo, 1),
        'stems': {}
    }

    total_configured = 0
    total_filtered = 0
    total_sensitive = 0

    for stem_type, events in events_by_stem.items():
        # Get analysis data for this stem
        analysis = analysis_by_stem.get(stem_type, {}) if analysis_by_stem else {}
        all_onset_data = analysis.get('all_onset_data', [])
        sensitive_onset_data = analysis.get('sensitive_onset_data', [])
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

        # Include classification thresholds for frontend slider defaults
        if config:
            stem_config = config.get(stem_type, {})
            if stem_type == 'hihat':
                logic['open_geomean_min'] = stem_config.get('open_geomean_min', 262.0)
                logic['open_sustain_ms'] = stem_config.get('open_sustain_ms', 150.0)
            if stem_type == 'snare':
                logic['expected_clusters'] = int(stem_config.get('expected_clusters', 1))

        # Serialize configured events (KEPT + FILTERED from configured detection)
        if all_onset_data:
            midi_events = [e for e in events if e.get('note') != 44]  # Exclude foot-close
            configured_events = _serialize_onset_events(all_onset_data, midi_events=midi_events)
        else:
            # Fallback: use events_by_stem directly if no all_onset_data
            configured_events = []
            for midi_event in events:
                event = {
                    'time': _round_value(midi_event.get('time'), 4),
                    'note': midi_event.get('note'),
                    'velocity': midi_event.get('velocity'),
                    'status': 'KEPT'
                }
                band_fields = [f'{b}_energy' for b in midi_event.get('geomean_bands', [])]
                for field in ['onset_strength', 'peak_amplitude'] + band_fields + ['geomean',
                             'total_energy', 'sustain_ms']:
                    value = midi_event.get(field)
                    if value is not None:
                        event[field] = _round_value(value, 2)
                configured_events.append(event)

        # Serialize sensitive events (all from max-sensitivity detection)
        sensitive_events = _serialize_onset_events(sensitive_onset_data) if sensitive_onset_data else []

        # Count totals
        total_configured += len(configured_events)
        total_filtered += sum(1 for e in configured_events if e.get('status') == 'FILTERED')
        total_sensitive += len(sensitive_events)

        # Assemble stem data
        sidecar_data['stems'][stem_type] = {
            'logic': logic,
            'events_configured': configured_events,
            'events_sensitive': sensitive_events,
        }

    # Write JSON
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar_data, f, indent=2)

    print(f"  Saved analysis sidecar v3: {sidecar_path.name} "
          f"({total_configured} configured events, {total_filtered} filtered, "
          f"{total_sensitive} sensitive events)")

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


def save_envelope_data(
    envelope_by_stem: Dict[str, Dict],
    midi_path: Union[str, Path]
) -> List[Path]:
    """
    Save per-stem energy envelope arrays as .npz files for waveform visualization.
    
    Each stem gets its own file: {base}.{stem_type}.envelope.npz containing
    the L/R energy envelope arrays, time axis, and detection parameters.
    
    Args:
        envelope_by_stem: Dict mapping stem_type to envelope data dict with keys:
            - times: np.ndarray of frame times in seconds
            - left: np.ndarray of left channel energy values
            - right: np.ndarray of right channel energy values
            - sr: int sample rate
            - hop_length: int hop length used
            - method: str energy calculation method ('rms', 'peak_hold', etc.)
        midi_path: Path to corresponding MIDI file (used to derive output paths)
    
    Returns:
        List of paths to created .npz files
    """
    midi_path = Path(midi_path)
    base = midi_path.with_suffix('')  # Remove .mid extension
    saved_paths = []
    
    for stem_type, envelope in envelope_by_stem.items():
        if envelope is None:
            continue
        
        times = envelope.get('times')
        left = envelope.get('left')
        right = envelope.get('right')
        
        # Skip if no envelope data (e.g. librosa detection path)
        if times is None or left is None or right is None:
            continue
        
        npz_path = Path(f"{base}.{stem_type}.envelope.npz")
        np.savez_compressed(
            npz_path,
            times=np.asarray(times, dtype=np.float32),
            left=np.asarray(left, dtype=np.float32),
            right=np.asarray(right, dtype=np.float32),
            sr=np.array(envelope.get('sr', 44100)),
            hop_length=np.array(envelope.get('hop_length', 512)),
            method=np.array(envelope.get('method', 'rms'))
        )
        saved_paths.append(npz_path)
    
    if saved_paths:
        stem_names = [p.suffixes[-2].lstrip('.') for p in saved_paths]
        print(f"  Saved envelope data: {', '.join(stem_names)} ({len(saved_paths)} files)")
    
    return saved_paths


def load_envelope_data(
    midi_path: Union[str, Path],
    stem_type: str
) -> Optional[Dict]:
    """
    Load energy envelope data for a specific stem.
    
    Args:
        midi_path: Path to MIDI file (used to derive .npz path)
        stem_type: Stem type to load ('kick', 'snare', etc.)
    
    Returns:
        Dict with keys: times, left, right, sr, hop_length, method.
        Returns None if file not found.
    """
    midi_path = Path(midi_path)
    base = midi_path.with_suffix('')
    npz_path = Path(f"{base}.{stem_type}.envelope.npz")
    
    if not npz_path.exists():
        return None
    
    data = np.load(npz_path, allow_pickle=False)
    return {
        'times': data['times'],
        'left': data['left'],
        'right': data['right'],
        'sr': int(data['sr']),
        'hop_length': int(data['hop_length']),
        'method': str(data['method'])
    }

