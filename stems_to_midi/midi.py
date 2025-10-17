"""
MIDI File Operations Module

Handles creation and reading of MIDI files for drum transcription.
"""

from midiutil import MIDIFile
import mido
from pathlib import Path
from typing import Dict, List, Union, Optional

# Import helper function for event preparation
from .helpers import prepare_midi_events_for_writing

__all__ = [
    'create_midi_file',
    'read_midi_notes'
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

