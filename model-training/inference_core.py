"""
inference_core.py — Core inference logic for drum transcription.

Pure functions: peak detection, heatmap-to-notes conversion,
note grouping, MIDI writing utilities. No I/O or side effects.
"""

from typing import List, Tuple
from config import SECONDS_PER_FRAME, INDEX_TO_MIDI, INDEX_TO_NAME

import numpy as np
import torch


def find_peaks_with_onset_snap(
    probabilities: np.ndarray,
    threshold: float,
    min_distance: int = 1,
) -> List[Tuple[int, float]]:
    """
    Find peaks in a probability curve and snap to steepest onset point.
    
    Args:
        probabilities: 1D array of probabilities [Time]
        threshold: Minimum probability to consider
        min_distance: Minimum frames between peaks
    
    Returns:
        List of (peak_frame, peak_value) tuples snapped to steepest onset
    """
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(probabilities, height=threshold, distance=min_distance)
    
    if len(peaks) == 0:
        return []
    
    results = []
    for peak_idx in peaks:
        onset_start = max(0, peak_idx - 5)
        
        if onset_start == peak_idx:
            results.append((peak_idx, probabilities[peak_idx]))
            continue
        
        gradient = np.gradient(probabilities[onset_start:peak_idx + 1])
        steepest_local = onset_start + np.argmax(gradient)
        
        results.append((steepest_local, probabilities[steepest_local]))
    
    return results


def heatmap_to_notes(
    prediction: torch.Tensor,
    threshold: float = 0.8,
) -> List[Tuple[float, int, int]]:
    """
    Convert neural heatmap to MIDI note events.
    
    Args:
        prediction: Tensor of shape [Batch, Time, 20] or [Time, 20]
        threshold: Minimum probability to trigger a note
    
    Returns:
        List of (time_seconds, midi_note, velocity) tuples
    """
    if prediction.is_cuda:
        prediction = prediction.cpu()
    pred_np = prediction.detach().cpu().numpy()
    
    if pred_np.ndim == 3:
        pred_np = pred_np[0]  # [Time, 20]
    
    notes = []
    for class_idx in range(10):
        # Channels 0-9: onset probabilities (pass through sigmoid)
        onset_probs = 1.0 / (1.0 + np.exp(-pred_np[:, class_idx]))
        midi_note = INDEX_TO_MIDI[class_idx]
        
        peaks = find_peaks_with_onset_snap(onset_probs, threshold, min_distance=1)
        
        for frame, prob in peaks:
            time_seconds = frame * SECONDS_PER_FRAME
            # Channels 10-19: velocity regression values (sigmoid输出, 0-1范围)
            # Apply sigmoid to match training loss, then power-law scale to MIDI velocity
            # Clamp to valid MIDI velocity range [35, 127]
            raw_vel = pred_np[frame, class_idx + 10]
            velocity_value = 1.0 / (1.0 + np.exp(-raw_vel))  # sigmoid
            velocity = int(min(127, max(35, (velocity_value ** (1.0 / 0.7)) * 127)))
            notes.append((time_seconds, midi_note, velocity))
    
    notes.sort(key=lambda x: x[0])
    return notes


def group_notes(
    raw_notes: List[Tuple[float, int, int]],
    time_tolerance: float = 0.05,
) -> List[Tuple[float, float, int]]:
    """
    Convert raw MIDI note events into grouped note on/off events.
    Multiple strikes of same pitch within time_tolerance are merged.
    
    Args:
        raw_notes: List of (time_seconds, midi_note, velocity)
        time_tolerance: Seconds to consider as same region
    
    Returns:
        List of (start_time, end_time, midi_note) tuples
    """
    if not raw_notes:
        return []
    
    sorted_notes = sorted(raw_notes, key=lambda x: (x[0], x[1]))
    
    groups = []
    current_group = None
    
    for time_sec, midi_note, velocity in sorted_notes:
        if current_group is None:
            current_group = [time_sec, time_sec, midi_note]
        elif midi_note == current_group[2] and time_sec - current_group[1] <= time_tolerance:
            current_group[1] = time_sec
        else:
            groups.append(tuple(current_group))
            current_group = [time_sec, time_sec, midi_note]
    
    if current_group:
        groups.append(tuple(current_group))
    
    return groups


def seconds_to_beats(time_sec: float, bpm: float) -> float:
    """Convert seconds to beats based on tempo."""
    return time_sec * (bpm / 60.0)


def write_midi(
    notes: List[Tuple[float, int, int]],
    output_path: str,
    bpm: float = 120.0,
) -> None:
    """
    Write notes to a MIDI file using midiutil.
    
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
    
    # Anchor note at time 0 for proper timing offset
    midi.addText(track, 0.0, "START")
    midi.addNote(track, 9, 27, 0.0, 0.01, 100)
    
    notes.sort(key=lambda x: x[0])
    
    for time_sec, midi_note, velocity in notes:
        time_beats = seconds_to_beats(time_sec, bpm)
        midi.addNote(track, channel, midi_note, time_beats, 0.08, max(1, min(127, int(velocity))))
    
    with open(output_path, 'wb') as f:
        midi.writeFile(f)


def prediction_stats(
    prediction: torch.Tensor,
) -> dict:
    """
    Compute per-class statistics from a prediction tensor.
    
    Args:
        prediction: Tensor of shape [Batch, Time, 10] or [Time, 10]
        
    Returns:
        Dict mapping class name to {'max': float, 'mean': float}
    """
    if prediction.is_cuda:
        prediction = prediction.cpu()
    pred_np = prediction.detach().cpu().numpy()
    
    if pred_np.ndim == 3:
        pred_np = pred_np[0]  # [Time, 10]
    
    stats = {}
    for i in range(10):
        vals = pred_np[:, i]
        stats[INDEX_TO_NAME[i]] = {
            'max': float(vals.max()),
            'mean': float(vals.mean()),
        }
    return stats


def notes_by_class(
    notes: List[Tuple[float, int, int]],
) -> dict:
    """
    Count notes per drum class.
    
    Args:
        notes: List of (time_seconds, midi_note, velocity)
        
    Returns:
        Dict mapping INDEX_TO_NAME to count
    """
    midi_to_idx = {v: k for k, v in INDEX_TO_MIDI.items()}
    counts = {name: 0 for name in INDEX_TO_NAME.values()}
    
    for _, midi_note, _ in notes:
        if midi_note in midi_to_idx:
            idx = midi_to_idx[midi_note]
            counts[INDEX_TO_NAME[idx]] += 1
    
    return counts
