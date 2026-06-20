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

__all__ = [
    'separate_channels',
    'calculate_pan_position',
    'calculate_stereo_width',
    'calculate_stereo_features',
    'classify_onset_by_pan',
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


