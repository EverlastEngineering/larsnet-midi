"""
Visualizer Runner - Generate Alignment Check PNG

Usage:
    conda run -n drumtomidi python run_visualizer.py
"""

import sys
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent / "visualizer"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "alignment_check.png"

sys.path.insert(0, '/Users/jasoncopp/Source/GitHub/larsnet')

from feature_extractor import get_input_tensor
from label_encoder import midi_to_frame_array, NoteAdapter
from midi_shell import load_midi_file
from midi_core import extract_midi_notes_from_tracks, build_tempo_map_from_tracks
from visualizer import plot_alignment_check

# File paths
audio_path = Path(__file__).parent / "dl-1.wav"
midi_path = Path(__file__).parent / "dl-1.mid"

# Load audio -> spectrogram
print(f"Loading audio: {audio_path}")
spec = get_input_tensor(str(audio_path))
print(f"  Spectrogram shape: {spec.shape}")

# Load MIDI and create labels
print(f"Loading MIDI: {midi_path}")
midi_file = load_midi_file(str(midi_path))
tempo_map = build_tempo_map_from_tracks(midi_file.tracks, midi_file.ticks_per_beat)
midi_notes, _ = extract_midi_notes_from_tracks(midi_file.tracks, midi_file.ticks_per_beat, tempo_map)
notes = [NoteAdapter(pitch=n.midi_note, start_time=n.time) for n in midi_notes]
labels = midi_to_frame_array(notes, spec.shape[2])
print(f"  Labels shape: {labels.shape}")

# Generate visualization
print(f"\nGenerating visualization...")
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from visualizer import plot_alignment_check
plot_alignment_check(spec, labels)

# Move to output directory
import shutil
shutil.move('/tmp/alignment_check.png', output_path)
print(f"\nSaved to: {output_path}")