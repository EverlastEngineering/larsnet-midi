"""
Label Encoder - MIDI to 10-Channel Heatmap

Maps MIDI notes to a 10-channel binary heatmap with causal smearing.
Forward-only smearing ensures model can't "predict the future".

Label mapping:
  0: Kick (35, 36)
  1: Snare (37, 38, 39, 40)
  2: HHC (22, 42, 44)
  3: HHO (26, 46)
  4: TomHigh (48, 50)
  5: TomMid (45, 47)
  6: TomLow (41, 43, 58)
  7: Crash1 (49, 55)
  8: Crash2 (52, 57)
  9: Ride (51, 53, 59)
"""

import torch
from typing import List, Union

# Mapping from Roland TD-17 (pitch -> channel index)
MAPPING = {
    36: 0, 35: 0,   # Kick
    38: 1, 40: 1, 37: 1, 39: 1,  # Snare
    42: 2, 44: 2, 22: 2,  # HH Closed
    46: 3, 26: 3,          # HH Open
    48: 4, 50: 4,   # Tom High
    45: 5, 47: 5,   # Tom Mid
    43: 6, 58: 6,   # Tom Low
    49: 7, 55: 7,   # Crash 1
    52: 8, 57: 8,   # Crash 2
    51: 9, 53: 9, 59: 9    # Ride
}

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
    Maps MIDI notes to a [10, Frames] binary heatmap with causal smearing.
    
    Args:
        midi_notes: List of note objects with .pitch and .start_time attributes
        total_frames: Total number of spectrogram frames
        hop_length: FFT hop length (default 512)
        sr: Sample rate (default 44100)
    
    Returns:
        Tensor of shape [10, total_frames] with causal smeared labels
    """
    labels = torch.zeros((10, total_frames))
    seconds_per_frame = hop_length / sr
    
    for note in midi_notes:
        if note.pitch in MAPPING:
            # Convert MIDI seconds to the nearest spectrogram frame
            hit_frame = int(note.start_time / seconds_per_frame)
            idx = MAPPING[note.pitch]
            
            # Causal Smear: Probability is 1.0 at impact, then decays
            # This allows the model to be 'close' and still receive partial credit
            if 0 <= hit_frame < total_frames:
                labels[idx, hit_frame] = 1.0       # Precision Hit
                if hit_frame + 1 < total_frames:
                    labels[idx, hit_frame + 1] = 0.8
                if hit_frame + 2 < total_frames:
                    labels[idx, hit_frame + 2] = 0.5
                if hit_frame + 3 < total_frames:
                    labels[idx, hit_frame + 3] = 0.2
                    
    return labels


# 24 unique Roland TD-17 pitches
ROLAND_PITCHES = [22, 26, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58, 59]
ROLAND_TO_IDX = {p: i for i, p in enumerate(ROLAND_PITCHES)}

# Category mapping for gatekeeper (3 families)
CATEGORY_MAP = {
    0: {36, 35},   # Kick
    1: {38, 40, 37, 39, 45, 47, 43, 58, 48, 50},  # Snare/Toms
    2: {42, 44, 22, 46, 26, 49, 55, 52, 57, 51, 53, 59}  # Cymbals
}


def get_category_for_pitch(pitch: int) -> int:
    """Map Roland pitch to gatekeeper category (0, 1, or 2)."""
    for cat_idx, pitches in CATEGORY_MAP.items():
        if pitch in pitches:
            return cat_idx
    return -1  # Unknown


def midi_to_multitarget_arrays(
    midi_notes: List[NoteAdapter],
    total_frames: int,
    hop_length: int = 512,
    sr: int = 44100
) -> dict:
    """
    Maps MIDI notes to 4 target arrays for multi-task learning.
    
    Returns:
        dict with keys:
          - gatekeeper: [3, Frames] one-hot category labels
          - groupings: [10, Frames] binary labels with causal smear
          - precision: [24, Frames] per-Roland-pitch binary labels with smear
          - velocity: [24, Frames] velocity-scaled labels (0.3-1.0 range)
    """
    gatekeeper_labels = torch.zeros((3, total_frames))
    groupings_labels = torch.zeros((10, total_frames))
    precision_labels = torch.zeros((24, total_frames))
    velocity_labels = torch.zeros((24, total_frames))
    
    seconds_per_frame = hop_length / sr
    
    for note in midi_notes:
        if note.pitch not in MAPPING:
            continue
            
        hit_frame = int(note.start_time / seconds_per_frame)
        idx = MAPPING[note.pitch]  # 0-9 for groupings
        roland_idx = ROLAND_TO_IDX.get(note.pitch, -1)
        
        if roland_idx < 0:
            continue
        
        # Gatekeeper: one-hot category (Head 1)
        cat_idx = get_category_for_pitch(note.pitch)
        if cat_idx >= 0 and 0 <= hit_frame < total_frames:
            gatekeeper_labels[cat_idx, hit_frame] = 1.0
        
        # Groupings: 10-class smear (Head 2)
        if 0 <= hit_frame < total_frames:
            for offset, val in enumerate([1.0, 0.8, 0.5, 0.2]):
                f = hit_frame + offset
                if f < total_frames:
                    groupings_labels[idx, f] = val
        
        # Precision: 24-channel per-Roland-pitch smear (Head 3)
        if 0 <= hit_frame < total_frames:
            for offset, val in enumerate([1.0, 0.8, 0.5, 0.2]):
                f = hit_frame + offset
                if f < total_frames:
                    precision_labels[roland_idx, f] = val
        
        # Velocity: scale by MIDI velocity 1-127 → 0.3-1.0 (Head 4)
        vel_scale = 0.3 + (note.velocity / 127.0) * 0.7
        if 0 <= hit_frame < total_frames:
            velocity_labels[roland_idx, hit_frame] = vel_scale
    
    return {
        'gatekeeper': gatekeeper_labels,
        'groupings': groupings_labels,
        'precision': precision_labels,
        'velocity': velocity_labels
    }


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