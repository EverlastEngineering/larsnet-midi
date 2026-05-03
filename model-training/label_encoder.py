"""
Label Encoder - MIDI to 10-Channel Heatmap

Maps MIDI notes to a 10-channel binary heatmap with causal smearing.
Cymbal hits are split into separate classes (Crash1, Crash2, Ride).

Forward-only smearing ensures model can't "predict the future".
"""

import torch
from typing import List, Union

# Mapping from pitch -> channel index (10 classes, grouped by instrument type)
# Cymbal hits split: Crash1/Crash2 are separate, Ride stays combined
MAPPING = {
    # Kick
    36: 0, 35: 0,
    # Snare
    38: 1, 40: 1, 37: 1, 39: 1,
    # HH Closed
    42: 2, 44: 2, 22: 2,
    # HH Open
    46: 3, 26: 3,
    # Tom High
    48: 4, 50: 4,
    # Tom Mid
    45: 5, 47: 5,
    # Tom Low
    41: 6, 43: 6, 58: 6,
    # Crash 1
    49: 7, 55: 7,
    # Crash 2
    57: 8, 52: 8,
    # Ride
    51: 9, 53: 9, 59: 9,
}

# 10 classes: 0-9
LABEL_NAMES = ['Kick', 'Snare', 'HHC', 'HHO', 'TomHigh', 'TomMid', 'TomLow', 'Crash1', 'Crash2', 'Ride']


class NoteAdapter:
    """Adapter to handle different MIDI note formats"""
    def __init__(self, pitch: int, start_time: float, velocity: int = 100):
        self.pitch = pitch
        self.start_time = start_time
        self.velocity = velocity


def midi_to_frame_array(
    midi_notes: List[NoteAdapter],
    total_frames: int,
    hop_length: int = 512,
    sr: int = 44100
) -> torch.Tensor:
    """
    Maps MIDI notes to an [10, Frames] binary heatmap with causal smearing.
    
    Args:
        midi_notes: List of note objects with .pitch and .start_time attributes
        total_frames: Total number of spectrogram frames
        hop_length: FFT hop length (default 512)
        sr: Sample rate (default 44100)
    
    Returns:
        Tensor of shape [10, total_frames] with causal smeared labels
    """
    labels = torch.zeros((len(LABEL_NAMES), total_frames))
    seconds_per_frame = hop_length / sr
    
    for note in midi_notes:
        if note.pitch in MAPPING:
            # Convert MIDI seconds to the nearest spectrogram frame
            hit_frame = int(note.start_time / seconds_per_frame)
            idx = MAPPING[note.pitch]
            
            # Causal Smear: Probability is 1.0 at impact, then decays
            # This allows the model to be 'close' and still receive partial credit
            if hit_frame < total_frames:
                labels[idx, hit_frame] = 1.0       # Precision Hit
                if hit_frame + 1 < total_frames:
                    labels[idx, hit_frame + 1] = 0.8
                if hit_frame + 2 < total_frames:
                    labels[idx, hit_frame + 2] = 0.5
                if hit_frame + 3 < total_frames:
                    labels[idx, hit_frame + 3] = 0.2
                    
    return labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/jasoncopp/Source/GitHub/larsnet')
    from midi_shell import parse_midi_file
    
    if len(sys.argv) < 2:
        print("Usage: python label_encoder.py <midi.mid>")
        sys.exit(1)
    
    midi_path = sys.argv[1]
    print(f"Loading: {midi_path}")
    
    drum_notes, _ = parse_midi_file(midi_path)
    print(f"Parsed {len(drum_notes)} drum notes")
    
    # Convert DrumNote objects to adapter format
    notes = [
        NoteAdapter(pitch=note.midi_note, start_time=note.time, velocity=note.velocity)
        for note in drum_notes
    ]
    
    # Test with dummy frame count (for now)
    labels = midi_to_frame_array(notes, total_frames=1000)
    print(f"Label shape: {labels.shape}")
    print(f"Non-zero elements: {labels.sum():.0f}")