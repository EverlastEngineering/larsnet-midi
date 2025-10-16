"""
Pure helper functions for stem to MIDI conversion.

These are functional core functions - pure, deterministic, no I/O or side effects.
All audio processing logic extracted here for testability.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.signal import medfilt


# ============================================================================
# AUDIO UTILITIES (Pure Functions)
# ============================================================================

def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.
    
    Args:
        audio: Audio signal (mono or stereo)
    
    Returns:
        Mono audio signal
    """
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    return audio


def calculate_peak_amplitude(
    audio: np.ndarray,
    onset_sample: int,
    sr: int,
    window_ms: float = 10.0
) -> float:
    """
    Calculate peak amplitude in a window after onset.
    
    Pure function - no side effects.
    
    Args:
        audio: Audio signal (mono)
        onset_sample: Sample index of onset
        sr: Sample rate
        window_ms: Window duration in milliseconds
    
    Returns:
        Peak amplitude (0.0 to 1.0+)
    """
    window_samples = int(window_ms * sr / 1000.0)
    peak_end = min(onset_sample + window_samples, len(audio))
    
    peak_segment = audio[onset_sample:peak_end]
    if len(peak_segment) == 0:
        return 0.0
    
    return float(np.max(np.abs(peak_segment)))


def calculate_sustain_duration(
    audio: np.ndarray,
    onset_sample: int,
    sr: int,
    window_ms: float = 200.0,
    envelope_threshold: float = 0.1,
    smooth_kernel: int = 51
) -> float:
    """
    Calculate sustain duration by analyzing envelope decay.
    
    Pure function - no side effects.
    
    Args:
        audio: Audio signal (mono)
        onset_sample: Sample index of onset
        sr: Sample rate
        window_ms: Analysis window in milliseconds
        envelope_threshold: Threshold as fraction of peak (0.0-1.0)
        smooth_kernel: Median filter kernel size for envelope smoothing
    
    Returns:
        Sustain duration in milliseconds
    """
    window_samples = int(window_ms * sr / 1000.0)
    end_sample = min(onset_sample + window_samples, len(audio))
    segment = audio[onset_sample:end_sample]
    
    if len(segment) < 100:
        return 0.0
    
    # Calculate envelope (absolute value)
    envelope = np.abs(segment)
    
    # Smooth envelope
    envelope_smooth = medfilt(envelope, kernel_size=smooth_kernel)
    
    # Find where envelope drops below threshold
    peak_env = np.max(envelope_smooth)
    threshold_level = peak_env * envelope_threshold
    
    # Count samples above threshold
    above_threshold = envelope_smooth > threshold_level
    if not np.any(above_threshold):
        return 0.0
    
    sustain_samples = np.sum(above_threshold)
    sustain_ms = (sustain_samples / sr) * 1000.0
    
    return float(sustain_ms)


def calculate_spectral_energies(
    segment: np.ndarray,
    sr: int,
    freq_ranges: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """
    Calculate spectral energy in specified frequency ranges.
    
    Pure function - no side effects.
    
    Args:
        segment: Audio segment to analyze
        sr: Sample rate
        freq_ranges: Dict mapping names to (min_hz, max_hz) tuples
                     e.g., {'fundamental': (40, 80), 'body': (80, 150)}
    
    Returns:
        Dict mapping names to energy values
    """
    if len(segment) < 100:
        return {name: 0.0 for name in freq_ranges}
    
    # Compute FFT
    fft = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(len(segment), 1/sr)
    magnitude = np.abs(fft)
    
    # Calculate energy in each range
    energies = {}
    for name, (min_hz, max_hz) in freq_ranges.items():
        mask = (freqs >= min_hz) & (freqs < max_hz)
        energy = float(np.sum(magnitude[mask]))
        energies[name] = energy
    
    return energies


def get_spectral_config_for_stem(stem_type: str, config: Dict) -> Dict:
    """
    Get spectral configuration for a specific stem type.
    
    Pure function - extracts config without side effects.
    
    Args:
        stem_type: Type of stem ('kick', 'snare', 'toms', 'hihat', 'cymbals')
        config: Full configuration dictionary
    
    Returns:
        Dict with:
        - freq_ranges: Dict of frequency ranges
        - energy_labels: Dict mapping range names to display labels
        - geomean_threshold: Threshold for filtering (or None)
        - min_sustain_ms: Minimum sustain duration (or None)
    """
    stem_config = config.get(stem_type, {})
    
    if stem_type == 'snare':
        return {
            'freq_ranges': {
                'low': (stem_config['low_freq_min'], stem_config['low_freq_max']),
                'primary': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'secondary': (stem_config['wire_freq_min'], stem_config['wire_freq_max'])
            },
            'energy_labels': {
                'primary': 'BodyE',
                'secondary': 'WireE'
            },
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None
        }
    
    elif stem_type == 'kick':
        return {
            'freq_ranges': {
                'primary': (stem_config['fundamental_freq_min'], stem_config['fundamental_freq_max']),
                'secondary': (stem_config['body_freq_min'], stem_config['body_freq_max'])
            },
            'energy_labels': {
                'primary': 'FundE',
                'secondary': 'BodyE'
            },
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None
        }
    
    elif stem_type == 'toms':
        return {
            'freq_ranges': {
                'primary': (stem_config['fundamental_freq_min'], stem_config['fundamental_freq_max']),
                'secondary': (stem_config['body_freq_min'], stem_config['body_freq_max'])
            },
            'energy_labels': {
                'primary': 'FundE',
                'secondary': 'BodyE'
            },
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None
        }
    
    elif stem_type == 'hihat':
        return {
            'freq_ranges': {
                'primary': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'secondary': (stem_config['sizzle_freq_min'], stem_config['sizzle_freq_max'])
            },
            'energy_labels': {
                'primary': 'BodyE',
                'secondary': 'SizzleE'
            },
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': stem_config.get('min_sustain_ms', 25)
        }
    
    elif stem_type == 'cymbals':
        # Cymbals use hardcoded ranges (to be moved to config later)
        return {
            'freq_ranges': {
                'primary': (1000, 4000),
                'secondary': (4000, 10000)
            },
            'energy_labels': {
                'primary': 'BodyE',
                'secondary': 'BrillE'
            },
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': stem_config.get('min_sustain_ms', 150)
        }
    
    else:
        raise ValueError(f"Unknown stem type: {stem_type}")


def calculate_geomean(primary_energy: float, secondary_energy: float) -> float:
    """
    Calculate geometric mean of two energy values.
    
    Pure function - no side effects.
    
    Args:
        primary_energy: First energy value
        secondary_energy: Second energy value
    
    Returns:
        Geometric mean (sqrt of product)
    """
    return float(np.sqrt(primary_energy * secondary_energy))


def should_keep_onset(
    geomean: float,
    sustain_ms: Optional[float],
    geomean_threshold: Optional[float],
    min_sustain_ms: Optional[float],
    stem_type: str
) -> bool:
    """
    Determine if an onset should be kept based on spectral/sustain criteria.
    
    Pure function - decision logic without side effects.
    
    Args:
        geomean: Geometric mean of primary and secondary energy
        sustain_ms: Sustain duration in milliseconds (None if not calculated)
        geomean_threshold: Threshold for geomean filtering (None to disable)
        min_sustain_ms: Minimum sustain threshold (None to disable)
        stem_type: Type of stem (affects logic for hihat vs others)
    
    Returns:
        True if onset should be kept, False if it should be rejected
    """
    # If no filtering enabled, keep everything
    if geomean_threshold is None and min_sustain_ms is None:
        return True
    
    # For cymbals: require BOTH geomean AND sustain (if both thresholds set)
    if stem_type == 'cymbals':
        if geomean_threshold is not None and min_sustain_ms is not None:
            geomean_ok = geomean > geomean_threshold
            sustain_ok = (sustain_ms is not None) and (sustain_ms >= min_sustain_ms)
            return geomean_ok and sustain_ok
        elif min_sustain_ms is not None:
            return (sustain_ms is not None) and (sustain_ms >= min_sustain_ms)
        elif geomean_threshold is not None:
            return geomean > geomean_threshold
    
    # For hihat: use sustain OR geomean (more permissive)
    elif stem_type == 'hihat':
        if min_sustain_ms is not None and sustain_ms is not None:
            if sustain_ms >= min_sustain_ms:
                return True
        if geomean_threshold is not None:
            return geomean > geomean_threshold
        return False
    
    # For other stems (kick, snare, toms): use geomean only
    else:
        if geomean_threshold is not None:
            return geomean > geomean_threshold
        return True


def normalize_values(values: np.ndarray) -> np.ndarray:
    """
    Normalize array of values to 0-1 range.
    
    Pure function - no side effects.
    
    Args:
        values: Array of values to normalize
    
    Returns:
        Normalized array (0-1 range)
    """
    if len(values) == 0:
        return values
    
    max_val = np.max(values)
    if max_val > 0:
        return values / max_val
    else:
        return np.ones_like(values)
