"""
Stereo Audio Analysis - Pure Functional Core

Pure functions for analyzing stereo audio and extracting spatial information.
All functions are deterministic with no side effects.

Architecture: Functional Core
- No I/O operations
- Deterministic (same input → same output)
- No external state or side effects
- Testable in isolation
"""

import numpy as np
from typing import Tuple, Optional, List
import librosa  # type: ignore

# Import TypedDict from parent module
try:
    from midi_types import StereoOnsetData, DualChannelOnsetData
except ImportError:
    # Running from stems_to_midi/ directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from midi_types import StereoOnsetData, DualChannelOnsetData


__all__ = [
    'separate_channels',
    'calculate_pan_position',
    'calculate_stereo_width',
    'calculate_stereo_features',
    'classify_onset_by_pan',
    'detect_stereo_onsets',
    'detect_dual_channel_onsets',
]


def separate_channels(stereo_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract left and right channels from stereo audio.
    
    Pure function - no side effects.
    
    Args:
        stereo_audio: Stereo audio array with shape (samples, 2) or (2, samples)
    
    Returns:
        Tuple of (left_channel, right_channel), each with shape (samples,)
    
    Raises:
        ValueError: If audio is not stereo
    
    Examples:
        >>> stereo = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> left, right = separate_channels(stereo)
        >>> left.shape
        (3,)
        >>> right.shape
        (3,)
    """
    if stereo_audio.ndim != 2:
        raise ValueError(f"Expected 2D stereo array, got {stereo_audio.ndim}D")
    
    # Handle both (samples, channels) and (channels, samples) formats
    if stereo_audio.shape[0] == 2:
        # Format: (channels, samples) - librosa style
        left = stereo_audio[0, :]
        right = stereo_audio[1, :]
    elif stereo_audio.shape[1] == 2:
        # Format: (samples, channels) - soundfile style
        left = stereo_audio[:, 0]
        right = stereo_audio[:, 1]
    else:
        raise ValueError(
            f"Expected stereo audio with 2 channels, got shape {stereo_audio.shape}"
        )
    
    return left, right


def calculate_pan_position(
    stereo_audio: np.ndarray,
    onset_sample: int,
    sr: int,
    window_ms: float = 10.0
) -> float:
    """
    Calculate pan position at a specific onset time.
    
    Analyzes the amplitude difference between left and right channels
    in a short window around the onset to determine spatial position.
    
    Pure function - deterministic, no side effects.
    
    Args:
        stereo_audio: Stereo audio array with shape (samples, 2) or (2, samples)
        onset_sample: Sample index of the onset
        sr: Sample rate in Hz
        window_ms: Analysis window duration in milliseconds
    
    Returns:
        Pan position from -1.0 (full left) to +1.0 (full right)
        0.0 indicates centered
    
    Examples:
        >>> # Audio with left channel louder
        >>> stereo = np.array([[0.8, 0.2]] * 1000)
        >>> pan = calculate_pan_position(stereo, 500, 22050, window_ms=10.0)
        >>> pan < 0  # Negative = left
        True
        
        >>> # Centered audio
        >>> stereo = np.array([[0.5, 0.5]] * 1000)
        >>> pan = calculate_pan_position(stereo, 500, 22050)
        >>> abs(pan) < 0.1  # Near zero = centered
        True
    """
    left, right = separate_channels(stereo_audio)
    
    # Calculate window size in samples
    window_samples = int((window_ms / 1000.0) * sr)
    
    # Define analysis window around onset
    start_sample = max(0, onset_sample)
    end_sample = min(len(left), onset_sample + window_samples)
    
    if start_sample >= end_sample:
        return 0.0  # Not enough samples, assume centered
    
    # Extract window from both channels
    left_window = left[start_sample:end_sample]
    right_window = right[start_sample:end_sample]
    
    # Calculate RMS amplitude for each channel
    left_rms = np.sqrt(np.mean(left_window ** 2))
    right_rms = np.sqrt(np.mean(right_window ** 2))
    
    # Avoid division by zero
    total_rms = left_rms + right_rms
    if total_rms < 1e-10:
        return 0.0  # Silent, assume centered
    
    # Calculate pan position
    # Formula: (right - left) / (right + left)
    # Result: -1.0 (full left) to +1.0 (full right)
    pan = (right_rms - left_rms) / total_rms
    
    return float(pan)


def calculate_stereo_width(
    stereo_audio: np.ndarray,
    onset_sample: int,
    sr: int,
    window_ms: float = 30.0
) -> float:
    """
    Calculate stereo width at a specific onset time.

    Measures the ratio of side (L-R) energy to total (mid + side) energy
    in a short window after the onset.  A mono-panned sound (identical L
    and R) produces width ≈ 0.  Uncorrelated L/R (e.g. wide reverb)
    produces width ≈ 0.5.  Fully anti-phase L/R produces width ≈ 1.0.

    Formula: RMS(L - R) / (RMS(L + R) + RMS(L - R))

    Pure function — deterministic, no side effects.

    Args:
        stereo_audio: Stereo audio array with shape (samples, 2) or (2, samples).
        onset_sample: Sample index of the onset.
        sr: Sample rate in Hz.
        window_ms: Analysis window duration in milliseconds (default 30 ms
            to capture attack + early reflections).

    Returns:
        Stereo width from 0.0 (mono) to 1.0 (full side).
        ~0.5 for uncorrelated channels.
        Returns 0.0 when the window is silent or too short.
    """
    left, right = separate_channels(stereo_audio)

    window_samples = int((window_ms / 1000.0) * sr)
    start_sample = max(0, onset_sample)
    end_sample = min(len(left), onset_sample + window_samples)

    if start_sample >= end_sample:
        return 0.0

    left_window = left[start_sample:end_sample]
    right_window = right[start_sample:end_sample]

    mid = left_window + right_window
    side = left_window - right_window

    mid_rms = np.sqrt(np.mean(mid ** 2))
    side_rms = np.sqrt(np.mean(side ** 2))

    total = mid_rms + side_rms
    if total < 1e-10:
        return 0.0

    # side / (side + mid): 0 = mono, 0.5 = uncorrelated, 1.0 = full side
    return float(side_rms / total)


def calculate_stereo_features(
    stereo_audio: np.ndarray,
    onset_times: np.ndarray,
    sr: int,
    pan_window_ms: float = 10.0,
    width_window_ms: float = 30.0,
) -> List[dict]:
    """
    Compute pan_confidence and stereo_width for an array of onset times.

    Batch convenience wrapper around :func:`calculate_pan_position` and
    :func:`calculate_stereo_width`.  Returns one dict per onset with keys
    ``pan_confidence`` and ``stereo_width``.

    Pure function — deterministic, no side effects.

    Args:
        stereo_audio: Stereo audio (2-channel).
        onset_times: 1-D array of onset times in seconds.
        sr: Sample rate in Hz.
        pan_window_ms: Window for pan calculation (default 10 ms).
        width_window_ms: Window for width calculation (default 30 ms).

    Returns:
        List of dicts ``[{'pan_confidence': float, 'stereo_width': float}, ...]``
        aligned 1-to-1 with *onset_times*.  If *stereo_audio* is not stereo
        (e.g. mono fallback), every entry is ``{'pan_confidence': 0.0, 'stereo_width': 0.0}``.
    """
    if stereo_audio.ndim != 2:
        return [{'pan_confidence': 0.0, 'stereo_width': 0.0}
                for _ in onset_times]

    features: List[dict] = []
    for t in onset_times:
        sample = int(t * sr)
        pan = calculate_pan_position(stereo_audio, sample, sr, window_ms=pan_window_ms)
        width = calculate_stereo_width(stereo_audio, sample, sr, window_ms=width_window_ms)
        features.append({'pan_confidence': pan, 'stereo_width': width})
    return features


def classify_onset_by_pan(
    pan_position: float,
    center_threshold: float = 0.15
) -> str:
    """
    Classify onset as 'left', 'right', or 'center' based on pan position.
    
    Pure function - deterministic, no side effects.
    
    Args:
        pan_position: Pan value from -1.0 (left) to +1.0 (right)
        center_threshold: Threshold for center classification
            Values within [-threshold, +threshold] are considered centered
    
    Returns:
        Classification string: 'left', 'right', or 'center'
    
    Examples:
        >>> classify_onset_by_pan(-0.8)
        'left'
        >>> classify_onset_by_pan(0.7)
        'right'
        >>> classify_onset_by_pan(0.05)
        'center'
        >>> classify_onset_by_pan(0.2, center_threshold=0.15)
        'right'
    """
    if pan_position < -center_threshold:
        return 'left'
    elif pan_position > center_threshold:
        return 'right'
    else:
        return 'center'


def detect_stereo_onsets(
    stereo_audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    threshold: float = 0.3,
    delta: float = 0.01,
    wait: int = 3
) -> StereoOnsetData:
    """
    Detect onsets separately in left, right, and mono (averaged) channels.
    
    Algorithm Coordinator - orchestrates librosa onset detection on each channel.
    Returns structured data for comparison and spatial analysis.
    
    Note: This is an imperative shell function (uses librosa), not pure.
    Coordinates external library calls but has deterministic output.
    
    Args:
        stereo_audio: Stereo audio array with shape (samples, 2) or (2, samples)
        sr: Sample rate in Hz
        hop_length: Samples between frames (affects time resolution)
        threshold: Onset strength threshold (0-1)
        delta: Peak picking sensitivity
        wait: Minimum frames between peaks
    
    Returns:
        StereoOnsetData dict with onset times and strengths for each channel
    
    Examples:
        >>> stereo = np.random.randn(44100, 2) * 0.1  # 1 second stereo
        >>> result = detect_stereo_onsets(stereo, sr=44100)
        >>> 'left_onsets' in result
        True
        >>> 'mono_onsets' in result
        True
    """
    # Separate channels
    left, right = separate_channels(stereo_audio)
    
    # Calculate mono (average of channels)
    mono = (left + right) / 2.0
    
    # Detect onsets in each channel
    left_onsets = librosa.onset.onset_detect(
        y=left,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units='time'
    )
    
    right_onsets = librosa.onset.onset_detect(
        y=right,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units='time'
    )
    
    mono_onsets = librosa.onset.onset_detect(
        y=mono,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units='time'
    )
    
    # Get onset strengths for each channel
    left_strength_func = librosa.onset.onset_strength(
        y=left,
        sr=sr,
        hop_length=hop_length
    )
    
    right_strength_func = librosa.onset.onset_strength(
        y=right,
        sr=sr,
        hop_length=hop_length
    )
    
    # Convert onset frames to times and get corresponding strengths
    def get_strengths(onsets: np.ndarray, strength_func: np.ndarray) -> List[float]:
        """Extract strength values at onset times."""
        onset_frames = librosa.time_to_frames(onsets, sr=sr, hop_length=hop_length)
        strengths = []
        for frame in onset_frames:
            if 0 <= frame < len(strength_func):
                strengths.append(float(strength_func[frame]))
            else:
                strengths.append(0.0)
        return strengths
    
    left_strengths = get_strengths(left_onsets, left_strength_func)
    right_strengths = get_strengths(right_onsets, right_strength_func)
    
    # Mono strengths (approximate, for completeness)
    mono_strengths = [1.0] * len(mono_onsets)  # Simplified
    
    return StereoOnsetData(
        left_onsets=left_onsets.tolist() if isinstance(left_onsets, np.ndarray) else left_onsets,
        right_onsets=right_onsets.tolist() if isinstance(right_onsets, np.ndarray) else right_onsets,
        mono_onsets=mono_onsets.tolist() if isinstance(mono_onsets, np.ndarray) else mono_onsets,
        left_strengths=left_strengths,
        right_strengths=right_strengths
    )


def detect_dual_channel_onsets(
    stereo_audio: np.ndarray,
    sr: int,
    merge_window_ms: float = 100.0,
    threshold: float = 0.3,
    delta: float = 0.01,
    wait: int = 3,
    hop_length: int = 512,
) -> DualChannelOnsetData:
    """
    Detect onsets separately in L/R channels, then merge nearby detections.
    
    This function implements the dual-channel detection strategy for clustering:
    1. Separate left and right channels
    2. Run independent onset detection on each channel
    3. Merge onsets that occur within merge_window_ms of each other
    4. Calculate pan confidence from L/R strength differential
    
    Pure function - no side effects.
    
    Args:
        stereo_audio: Stereo audio array with shape (samples, 2) or (2, samples)
        sr: Sample rate in Hz
        merge_window_ms: Merge L/R onsets within this window (milliseconds)
        threshold: Onset strength threshold (0-1, lower=more sensitive)
        delta: Peak picking sensitivity (lower=more sensitive)
        wait: Minimum frames between peaks
        hop_length: Samples between frames for onset detection
    
    Returns:
        DualChannelOnsetData with merged onset times and L/R strengths
    
    Examples:
        >>> # Left-panned signal
        >>> left_signal = np.random.randn(44100)
        >>> right_signal = np.zeros(44100)
        >>> stereo = np.stack([left_signal, right_signal], axis=0)
        >>> result = detect_dual_channel_onsets(stereo, 44100)
        >>> # Expect pan_confidence < 0 (left-biased)
    """
    # Separate channels
    left, right = separate_channels(stereo_audio)
    
    # Detect onsets in each channel independently
    left_strength_func = librosa.onset.onset_strength(
        y=left, sr=sr, hop_length=hop_length
    )
    right_strength_func = librosa.onset.onset_strength(
        y=right, sr=sr, hop_length=hop_length
    )
    
    # librosa.onset.onset_detect uses internal peak picking with different parameter names
    # We use pre_max, post_max, pre_avg, post_avg, wait, delta as **kwargs
    left_frames = librosa.onset.onset_detect(
        onset_envelope=left_strength_func,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        wait=wait,
        delta=delta,
    )
    
    right_frames = librosa.onset.onset_detect(
        onset_envelope=right_strength_func,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        wait=wait,
        delta=delta,
    )
    
    # Convert frames to times
    left_times = librosa.frames_to_time(left_frames, sr=sr, hop_length=hop_length)
    right_times = librosa.frames_to_time(right_frames, sr=sr, hop_length=hop_length)
    
    # Helper function to get peak onset strength near frame
    # Backtracking moves onsets to zero-crossings, so we need to find the nearby peak
    def get_peak_strength(strength_func: np.ndarray, frame: int, search_window: int = 5) -> float:
        if len(strength_func) == 0:
            return 0.0
        # Search for peak in small window around frame
        start = max(0, frame - search_window)
        end = min(len(strength_func), frame + search_window + 1)
        if start >= end:
            return 0.0
        window = strength_func[start:end]
        return float(window.max())
    
    # Get strengths for each onset
    left_onset_strengths = [get_peak_strength(left_strength_func, f) for f in left_frames]
    right_onset_strengths = [get_peak_strength(right_strength_func, f) for f in right_frames]
    
    # Merge nearby onsets within merge_window
    merge_window_sec = merge_window_ms / 1000.0
    
    # Create list of all onsets with their channel and strength
    all_onsets = []
    for t, s in zip(left_times, left_onset_strengths):
        all_onsets.append({'time': t, 'left': s, 'right': 0.0})
    for t, s in zip(right_times, right_onset_strengths):
        all_onsets.append({'time': t, 'left': 0.0, 'right': s})
    
    # Sort by time
    all_onsets.sort(key=lambda x: x['time'])
    
    # Merge onsets within merge_window
    merged_onsets = []
    i = 0
    while i < len(all_onsets):
        current = all_onsets[i]
        merged = {
            'time': current['time'],
            'left': current['left'],
            'right': current['right']
        }
        
        # Look ahead for nearby onsets to merge
        j = i + 1
        while j < len(all_onsets):
            next_onset = all_onsets[j]
            if next_onset['time'] - merged['time'] <= merge_window_sec:
                # Merge: take max strength from each channel
                merged['left'] = max(merged['left'], next_onset['left'])
                merged['right'] = max(merged['right'], next_onset['right'])
                j += 1
            else:
                break
        
        merged_onsets.append(merged)
        i = j if j > i + 1 else i + 1
    
    # Extract final arrays
    onset_times = [o['time'] for o in merged_onsets]
    left_strengths = [o['left'] for o in merged_onsets]
    right_strengths = [o['right'] for o in merged_onsets]
    
    # Calculate pan confidence: (R-L)/(R+L)
    pan_confidence = []
    for l, r in zip(left_strengths, right_strengths):
        total = l + r
        if total > 0:
            pan = (r - l) / total
        else:
            pan = 0.0  # No signal, assume center
        pan_confidence.append(pan)
    
    # Apply threshold filter to remove weak onsets
    # Filter based on max(left, right) strength
    filtered_onsets = []
    filtered_left_strengths = []
    filtered_right_strengths = []
    filtered_pan_confidence = []
    
    for i, (t, l, r, pan) in enumerate(zip(onset_times, left_strengths, right_strengths, pan_confidence)):
        max_strength = max(l, r)
        if max_strength >= threshold:
            filtered_onsets.append(t)
            filtered_left_strengths.append(l)
            filtered_right_strengths.append(r)
            filtered_pan_confidence.append(pan)
    
    return DualChannelOnsetData(
        onset_times=filtered_onsets,
        left_strengths=filtered_left_strengths,
        right_strengths=filtered_right_strengths,
        pan_confidence=filtered_pan_confidence
    )
