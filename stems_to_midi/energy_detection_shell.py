"""
Energy-based detection shell - drop-in replacement for detect_onsets.

This module provides a bridge between the new energy_detection_core and
the existing processing pipeline. It wraps detect_stereo_transient_peaks
to match the interface expected by processing_shell.py.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .energy_detection_core import detect_stereo_transient_peaks


def detect_onsets_energy_based(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = 15.0,
    min_peak_spacing_ms: float = 100.0,
    merge_window_ms: float = 150.0,
    min_absolute_energy: float = 0.01,
    frame_length: int = 2048,
    hop_length: int = 512,
    method: str = 'rms',
    enable_amplitude_refinement: bool = True,
    amplitude_smoothing_ms: float = 3.0,
    amplitude_prominence: float = 0.3,
    max_peaks_per_region: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Detect onsets using energy-based transient peak detection (NEW METHOD).
    
    This is a drop-in replacement for detect_onsets() from detection_shell.py,
    but uses the new scipy peak detection with backtracking instead of librosa.
    
    Key differences from librosa:
    - No blind wait periods that skip real events
    - Detects on energy envelope (visual/DAW approach)
    - Backtracks to attack start (50-120ms before peak)
    - Stereo-aware detection with pan information
    
    Args:
        audio: Audio signal (mono or stereo)
        sr: Sample rate
        threshold_db: Prominence threshold in dB (15.0 calibrated for cymbals)
        min_peak_spacing_ms: Minimum spacing between peaks
        merge_window_ms: Merge L/R peaks within this window
        min_absolute_energy: Noise floor threshold
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        method: 'rms' or 'spectral'
    
    Returns:
        Tuple of:
        - onset_times: np.ndarray of onset times in seconds
        - onset_strengths: np.ndarray of peak energies (for velocity calculation)
        - extra_data: Dict with pan_positions, pan_classifications, left_energies, right_energies
    """
    # Ensure stereo (duplicate mono if needed)
    if audio.ndim == 1:
        stereo_audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 2:
        stereo_audio = audio
    elif audio.shape[1] == 2:
        stereo_audio = audio.T
    else:
        # Take first channel and duplicate
        stereo_audio = np.stack([audio[0], audio[0]], axis=0)
    
    # Detect using new energy-based method
    result = detect_stereo_transient_peaks(
        stereo_audio, sr,
        threshold_db=threshold_db,
        min_peak_spacing_ms=min_peak_spacing_ms,
        merge_window_ms=merge_window_ms,
        frame_length=frame_length,
        hop_length=hop_length,
        method=method,
        min_absolute_energy=min_absolute_energy,
        enable_amplitude_refinement=enable_amplitude_refinement,
        amplitude_smoothing_ms=amplitude_smoothing_ms,
        amplitude_prominence=amplitude_prominence,
        max_peaks_per_region=max_peaks_per_region,
    )
    
    # Convert to format expected by processing_shell
    onset_times = np.array(result['onset_times'])
    left_energies = np.array(result['left_energies'])
    right_energies = np.array(result['right_energies'])
    pan_confidence = np.array(result['pan_confidence'])
    
    # BUGFIX: Remove duplicate onset times (within 1ms threshold)
    # This can happen when L/R peaks merge incorrectly, causing midiutil errors
    if len(onset_times) > 0:
        # Round to nearest millisecond for duplicate detection
        onset_times_ms = np.round(onset_times * 1000).astype(int)
        unique_indices = []
        seen_times = set()
        
        for i, time_ms in enumerate(onset_times_ms):
            if time_ms not in seen_times:
                unique_indices.append(i)
                seen_times.add(time_ms)
        
        # Keep only unique onsets
        if len(unique_indices) < len(onset_times):
            num_duplicates = len(onset_times) - len(unique_indices)
            print(f"    ⚠️  Removed {num_duplicates} duplicate onset times (within 1ms)")
            onset_times = onset_times[unique_indices]
            left_energies = left_energies[unique_indices]
            right_energies = right_energies[unique_indices]
            pan_confidence = pan_confidence[unique_indices]
    
    # Calculate onset strengths (total energy) for velocity estimation
    onset_strengths = left_energies + right_energies
    
    # Calculate pan positions and classifications
    pan_positions = []
    pan_classifications = []
    center_threshold = 0.15
    
    for pan in pan_confidence:
        pan_positions.append(pan)
        
        # Classify pan position
        if abs(pan) < center_threshold:
            pan_class = 'center'
        elif pan < 0:
            pan_class = 'left'
        else:
            pan_class = 'right'
        pan_classifications.append(pan_class)
    
    # Package extra data
    extra_data = {
        'pan_positions': pan_positions,
        'pan_classifications': pan_classifications,
        'left_energies': left_energies,
        'right_energies': right_energies,
        'pan_confidence': pan_confidence,
        'detection_method': 'energy_based_scipy',
    }
    
    return onset_times, onset_strengths, extra_data


__all__ = [
    'detect_onsets_energy_based',
]
