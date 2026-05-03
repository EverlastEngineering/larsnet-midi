"""
Inference Post-Processor - Step 7 of Deep Learning Roadmap

Converts neural heatmap output back into a standard MIDI file.

Usage:
    # Basic usage with default threshold 0.8
    python inference.py dl-1.wav --output output.mid
    
    # With different threshold
    python inference.py dl-1.wav --output output.mid --threshold 0.5
    
    # Load from saved checkpoint
    python inference.py dl-1.wav --output output.mid --checkpoint models/smoke_test.ckpt
    
    # Compare output to ground truth
    python inference.py dl-1.wav --output output.mid --compare dl-1.mid
"""

import sys
sys.path.insert(0, '/Users/jasoncopp/Source/GitHub/larsnet')

import argparse
import torch
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path

from feature_extractor import get_input_tensor
from model import DrumTranscriber


# ============================================================================
# Drum Class Mapping (from roadmap)
# ============================================================================
INDEX_TO_MIDI = {
    0: 36,   # Kick
    1: 38,   # Snare/Clap
    2: 42,   # HH Closed
    3: 46,   # HH Open
    4: 48,   # Tom High
    5: 45,   # Tom Mid
    6: 41,   # Tom Low
    7: 49,   # Crash
    8: 51,   # Ride
    9: 52,   # China
    10: 55,  # Splash
}

INDEX_TO_NAME = {
    0: 'Kick', 1: 'Snare', 2: 'HHC', 3: 'HHO',
    4: 'TomHigh', 5: 'TomMid', 6: 'TomLow',
    7: 'Crash', 8: 'Ride', 9: 'China', 10: 'Splash'
}

# Global config for inference
HOP_LENGTH = 512
SAMPLE_RATE = 44100
SECONDS_PER_FRAME = HOP_LENGTH / SAMPLE_RATE


# ============================================================================
# Peak Detection with Onset Snapping
# ============================================================================

def find_peaks_with_onset_snap(probabilities: np.ndarray, threshold: float, min_distance: int = 5):
    """
    Find peaks in a probability curve and snap to steepest onset point.
    
    Args:
        probabilities: 1D array of probabilities [Time]
        threshold: Minimum probability to consider
        min_distance: Minimum frames between peaks
    
    Returns:
        List of (peak_frame, peak_value) tuples snapped to steepest onset
    """
    # Find raw peaks above threshold
    peaks, properties = find_peaks(probabilities, height=threshold, distance=min_distance)
    
    if len(peaks) == 0:
        return []
    
    results = []
    for peak_idx in peaks:
        # Find the steepest onset (max positive gradient) before the peak
        # Look back up to 5 frames for the steepest climb
        onset_start = max(0, peak_idx - 5)
        
        if onset_start == peak_idx:
            # No lookback available, use peak itself
            results.append((peak_idx, probabilities[peak_idx]))
            continue
        
        # Compute gradient in the window before peak
        gradient = np.gradient(probabilities[onset_start:peak_idx + 1])
        # Steepest point is where gradient is maximum
        steepest_local = onset_start + np.argmax(gradient)
        
        results.append((steepest_local, probabilities[steepest_local]))
    
    return results


def heatmap_to_notes(prediction: torch.Tensor, threshold: float = 0.8) -> list:
    """
    Convert neural heatmap to MIDI note events.
    
    Args:
        prediction: Tensor of shape [Batch, Time, 10] — probabilities per class
        threshold: Minimum probability to trigger a note
    
    Returns:
        List of (time_seconds, midi_note, velocity) tuples
    """
    # Move to CPU and convert to numpy
    if prediction.is_cuda:
        prediction = prediction.cpu()
    pred_np = prediction.detach().cpu().numpy()
    
    # Use first batch item if batched
    if pred_np.ndim == 3:
        pred_np = pred_np[0]  # [Time, 10]
    
    time_steps = pred_np.shape[0]
    notes = []
    
    for class_idx in range(10):
        probs = pred_np[:, class_idx]
        midi_note = INDEX_TO_MIDI[class_idx]
        
        # Find peaks with onset snapping
        peaks = find_peaks_with_onset_snap(probs, threshold, min_distance=5)
        
        for frame, prob in peaks:
            time_seconds = frame * SECONDS_PER_FRAME
            # Velocity derived from probability strength (0-127)
            velocity = int(min(127, prob * 127))
            notes.append((time_seconds, midi_note, velocity))
    
    # Sort by time
    notes.sort(key=lambda x: x[0])
    return notes


# ============================================================================
# MIDI Writing
# ============================================================================

def seconds_to_beats(time_sec: float, tempo: float) -> float:
    """Convert seconds to beats based on tempo."""
    return time_sec * (tempo / 60.0)


def write_midi(notes: list, output_path: str, bpm: float = 120.0):
    """
    Write notes to a MIDI file using midiutil (same as stems_to_midi/midi.py).
    
    Args:
        notes: List of (time_seconds, midi_note, velocity)
        output_path: Path to write .mid file
        bpm: Tempo in beats per minute (default 120)
    """
    from midiutil import MIDIFile
    
    midi = MIDIFile(1)
    track = 0
    channel = 9  # Drums channel
    
    midi.addTrackName(track, 0, "Drums")
    midi.addTempo(track, 0, bpm)
    
    # Add anchor note at time 0 (D#0 = note 27) to ensure proper timing offset
    midi.addText(track, 0.0, "START")
    midi.addNote(track, 9, 27, 0.0, 0.01, 1)  # Very short, very quiet
    
    # Sort notes by time
    notes.sort(key=lambda x: x[0])
    
    # Add each note
    for time_sec, midi_note, velocity in notes:
        time_beats = seconds_to_beats(time_sec, bpm)
        midi.addNote(track, channel, midi_note, time_beats, 0.08, velocity)
    
    with open(output_path, 'wb') as f:
        midi.writeFile(f)


# ============================================================================
# Comparison Metrics
# ============================================================================

def group_notes(raw_notes: list, time_tolerance: float = 0.05) -> list:
    """
    Convert raw MIDI note events into grouped note on/off events.
    Multiple strikes of the same pitch within time_tolerance are merged.
    
    Args:
        raw_notes: List of (time_seconds, midi_note, velocity) tuples
        time_tolerance: Seconds to consider as same region
    
    Returns:
        List of (start_time, end_time, midi_note) tuples
    """
    if not raw_notes:
        return []
    
    # Sort by time then pitch
    sorted_notes = sorted(raw_notes, key=lambda x: (x[0], x[1]))
    
    groups = []
    current_group = None
    
    for time_sec, midi_note, velocity in sorted_notes:
        if current_group is None:
            current_group = [time_sec, time_sec, midi_note]
        elif (midi_note == current_group[2] and 
              time_sec - current_group[1] <= time_tolerance):
            # Extend the group
            current_group[1] = time_sec
        else:
            # Save current and start new group
            groups.append(tuple(current_group))
            current_group = [time_sec, time_sec, midi_note]
    
    if current_group:
        groups.append(tuple(current_group))
    
    return groups


def compare_midi(generated_notes: list, ground_truth_path: str, time_tolerance: float = 0.05) -> dict:
    """
    Compare generated notes against ground truth MIDI.
    Both sides are grouped before comparison for fair comparison.
    
    Args:
        generated_notes: List of (time_seconds, midi_note, velocity)
        ground_truth_path: Path to ground truth .mid file
        time_tolerance: Seconds to consider as matching (default 50ms)
    
    Returns:
        Dict with precision, recall, F1, and match details
    """
    from midi_shell import load_midi_file
    from midi_core import extract_midi_notes_from_tracks, build_tempo_map_from_tracks
    
    # Load ground truth and convert to (time, pitch) format
    midi_file = load_midi_file(ground_truth_path)
    tempo_map = build_tempo_map_from_tracks(midi_file.tracks, midi_file.ticks_per_beat)
    gt_notes, _ = extract_midi_notes_from_tracks(
        midi_file.tracks, midi_file.ticks_per_beat, tempo_map
    )
    
    # Debug: show raw pitch counts in GT
    from collections import Counter
    raw_pitch_counts = Counter(n.midi_note for n in gt_notes)
    print(f"\n  [DEBUG] Raw GT pitch counts: {dict(sorted(raw_pitch_counts.items()))}")
    
    # Convert raw MIDI to grouped format (start_time, pitch)
    gt_raw = [(n.time, n.midi_note) for n in gt_notes]
    gt_grouped = group_notes([(t, p, 100) for t, p in gt_raw], time_tolerance)
    
    gen_grouped = group_notes(generated_notes, time_tolerance)
    
    # Build sets for comparison using grouped data
    # Map pitches to canonical values for fair comparison
    pitch_aliases = {22: 42, 44: 42, 26: 46, 55: 49, 35: 36}  # Roland -> canonical
    
    gt_set = set()
    for start, end, pitch in gt_grouped:
        t = round(start / time_tolerance) * time_tolerance
        canonical = pitch_aliases.get(pitch, pitch)
        gt_set.add((t, canonical))
    
    gen_set = set()
    for start, end, pitch in gen_grouped:
        t = round(start / time_tolerance) * time_tolerance
        gen_set.add((t, pitch))  # gen is already canonical from INDEX_TO_MIDI
    
    # Calculate metrics
    true_positives = len(gt_set & gen_set)
    false_positives = len(gen_set - gt_set)
    false_negatives = len(gt_set - gen_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Track true positives with offsets for diagnosis
    note_names = {
        36: 'Kick', 38: 'Snare', 42: 'HHC', 46: 'HHO', 48: 'TomHigh', 45: 'TomMid', 41: 'TomLow', 49: 'Crash1', 57: 'Crash2', 51: 'Ride',
        # Roland aliases
        22: 'HHC', 44: 'HHC', 26: 'HHO', 35: 'Kick', 40: 'Snare', 37: 'Snare', 39: 'Snare',
        50: 'TomHigh', 47: 'TomMid', 43: 'TomLow', 58: 'TomLow', 52: 'Crash2', 53: 'Ride', 59: 'Ride', 55: 'Crash1',
    }
    tp_details = []
    for key in sorted(gt_set & gen_set):
        t, pitch = key
        name = note_names.get(pitch, f'Note({pitch})')
        gt_time = next((g[0] for g in gt_grouped if g[0] == t and g[2] == pitch), t)
        gen_time = next((g[0] for g in gen_grouped if g[0] == t and g[2] == pitch), t)
        offset = gt_time - gen_time
        if len(tp_details) < 10:
            tp_details.append({'name': name, 'pitch': pitch, 'gt': gt_time, 'gen': gen_time, 'offset': offset})
    
    # Per-note breakdown by class name
    gt_by_class = Counter()
    for start, end, pitch in gt_grouped:
        name = note_names.get(pitch, f'Note({pitch})')
        gt_by_class[name] += 1
    
    gen_by_class = Counter()
    for start, end, pitch in gen_grouped:
        name = note_names.get(pitch, f'Note({pitch})')
        gen_by_class[name] += 1
    
    all_classes = sorted(set(gt_by_class.keys()) | set(gen_by_class.keys()))
    breakdown = {}
    for name in all_classes:
        breakdown[name] = {'gt': gt_by_class.get(name, 0), 'gen': gen_by_class.get(name, 0)}
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'generated_count': len(gen_set),
        'ground_truth_count': len(gt_set),
        'note_breakdown': breakdown,
        'tp_details': tp_details
    }


# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    audio_path: str,
    output_path: str = None,
    checkpoint_path: str = None,
    threshold: float = 0.8,
    device: str = None,
    compare_path: str = None
):
    """
    Run inference on audio and optionally compare to ground truth.
    
    Args:
        audio_path: Path to input .wav file
        output_path: Path to write output .mid (auto-generated if None)
        checkpoint_path: Path to model checkpoint (trains new if None)
        threshold: Probability threshold for peak detection
        device: 'cpu', 'cuda', 'mps', or None for auto-detect
        compare_path: Path to ground truth MIDI for comparison
    
    Returns:
        List of (time_seconds, midi_note, velocity) tuples
    """
    # Device selection
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")
    
    # Auto-generate output path if not provided
    if output_path is None:
        audio_name = Path(audio_path).stem
        output_path = f"{Path(audio_path).parent}/{audio_name}_predicted.mid"
    
    output_path = Path(output_path)
    
    # Get base name, strip any existing threshold suffix
    import re
    base = output_path.stem
    base_clean = re.sub(r'_t[0-9.]+$', '', base)  # Remove _t0.8 if present
    
    # Find next version number (versions are shared across thresholds)
    ext = output_path.suffix
    parent = output_path.parent
    counter = 1
    
    # Keep incrementing until no files exist with this version number (any threshold)
    while True:
        import glob
        # Check if this version exists with ANY threshold
        pattern = f"{base_clean}_v{counter}_t*"
        existing = glob.glob(str(parent / pattern))
        # Also check without threshold suffix (legacy files like dl-1_predicted_v1.mid)
        if existing or (parent / f"{base_clean}_v{counter}{ext}").exists():
            counter += 1
        else:
            break
    
    output_path = parent / f"{base_clean}_v{counter}_t{threshold}{ext}"
    
    print(f"Loading audio: {audio_path}")
    spec = get_input_tensor(audio_path)
    spec = spec.unsqueeze(0).to(device)  # [1, 1, 128, Time]
    print(f"  Input shape: {spec.shape}")
    
    # Load or train model
    model = DrumTranscriber().to(device)
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint provided — this will produce random output!")
        print("Run smoke_test.py first to train and save a model.")
    
    # Run inference
    print(f"\nRunning inference with threshold={threshold}...")
    model.eval()
    with torch.no_grad():
        prediction = model(spec)  # [1, Time, 10]
    
    print(f"  Prediction shape: {prediction.shape}")
    
    # Debug: show per-class prediction stats
    pred_np = prediction[0].cpu().numpy()
    print(f"  Per-class prediction stats (max/mean):")
    for i in range(10):
        vals = pred_np[:, i]
        print(f"    {INDEX_TO_NAME[i]:<10}: max={vals.max():.4f}, mean={vals.mean():.4f}")
    
    # Convert to MIDI notes
    notes = heatmap_to_notes(prediction, threshold=threshold)
    print(f"  Detected {len(notes)} note events")
    
    # Show per-class breakdown (debug info)
    print(f"\n  Per-class note counts:")
    print(f"  {'Class':<10} {'MIDI':>5} {'Count':>6}")
    print(f"  {'-'*22}")
    for class_idx in range(10):
        class_notes = [n for n in notes if n[1] == INDEX_TO_MIDI[class_idx]]
        if class_notes:
            print(f"  {INDEX_TO_NAME[class_idx]:<10} {INDEX_TO_MIDI[class_idx]:>5} {len(class_notes):>6}")
        else:
            print(f"  {INDEX_TO_NAME[class_idx]:<10} {INDEX_TO_MIDI[class_idx]:>5} {'0':>6} (none detected)")
    
    # Extract tempo from ground truth MIDI if available
    output_bpm = 120.0  # default
    if compare_path:
        try:
            from midi_shell import load_midi_file
            gt_midi = load_midi_file(compare_path)
            for track in gt_midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        output_bpm = 60_000_000 / msg.tempo  # microseconds -> BPM
                        break
                break
        except Exception:
            pass  # use default
    
    # Write MIDI file
    print(f"\nWriting MIDI: {output_path} (tempo: {output_bpm:.1f} BPM)")
    write_midi(notes, output_path, bpm=output_bpm)
    
    # Compare if ground truth provided
    if compare_path:
        print(f"\nComparing to ground truth: {compare_path}")
        metrics = compare_midi(notes, compare_path)
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        print(f"  Generated: {metrics['generated_count']}, Ground truth: {metrics['ground_truth_count']}")
        print(f"\n  Per-note breakdown:")
        print(f"  {'Note':<10} {'GT':>6} {'Gen':>6} {'Diff':>6}")
        print(f"  {'-'*30}")
        for name, data in sorted(metrics['note_breakdown'].items()):
            diff = data['gen'] - data['gt']
            sign = '+' if diff > 0 else ''
            print(f"  {name:<10} {data['gt']:>6} {data['gen']:>6} {sign}{diff:>5}")
        if metrics.get('tp_details'):
            print(f"\n  First 10 True Positives (GT time, Gen time, offset):")
            for tp in metrics['tp_details'][:10]:
                print(f"    {tp['name']:<10} GT {tp['gt']:.4f}s, Gen {tp['gen']:.4f}s, Offset: {tp['offset']:+.4f}s")
    
    return notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DrumToMIDI Inference Post-Processor')
    parser.add_argument('audio', help='Input audio file (.wav)')
    parser.add_argument('--output', '-o', help='Output MIDI file path')
    parser.add_argument('--checkpoint', '-c', help='Model checkpoint path')
    parser.add_argument('--threshold', '-t', type=float, default=0.8,
                        help='Detection threshold (0.0-1.0), default 0.8')
    parser.add_argument('--device', '-d', choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--compare', help='Ground truth MIDI to compare against')
    
    args = parser.parse_args()
    
    run_inference(
        audio_path=args.audio,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
        compare_path=args.compare
    )