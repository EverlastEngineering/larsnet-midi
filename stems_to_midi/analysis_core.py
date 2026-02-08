"""
Pure helper functions for stem to MIDI conversion.

These are functional core functions - pure, deterministic, no I/O or side effects.
All audio processing logic extracted here for testability.

Detection Output Contract (Producer):
    filter_onsets_by_spectral() produces SpectralOnsetData dicts.
    Contract defined in midi_types.py - see SpectralOnsetData TypedDict.
    Consumers: detection_shell.detect_hihat_state(), learning.py
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.signal import medfilt

# Note: SPECTRAL_REQUIRED_FIELDS is kept for compatibility with midi_types module contract,
# but may not be directly imported here


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


def calculate_event_durations(
    onset_times: np.ndarray,
    audio: np.ndarray,
    sr: int,
    silence_threshold_db: float = -40.0,
    skip_attack_ms: float = 20.0
) -> np.ndarray:
    """
    Calculate duration for each onset event.
    
    Duration = min(time_to_next_onset, time_to_silence)
    
    Pure function - no side effects.
    
    Args:
        onset_times: Array of onset times in seconds
        audio: Full audio signal (mono)
        sr: Sample rate
        silence_threshold_db: dB threshold for silence detection
        skip_attack_ms: Skip this many ms after onset before checking silence
                       (allows attack to reach peak before checking decay)
        
    Returns:
        durations: Array of durations in seconds for each onset
    """
    if len(onset_times) == 0:
        return np.array([])
    
    durations = np.zeros(len(onset_times))
    
    for i, onset_time in enumerate(onset_times):
        onset_sample = int(onset_time * sr)
        
        # Get next onset time (or end of file)
        if i < len(onset_times) - 1:
            next_onset_time = onset_times[i + 1]
        else:
            next_onset_time = len(audio) / sr
        
        # Find silence threshold crossing
        segment_end = int(next_onset_time * sr)
        segment = audio[onset_sample:segment_end]
        
        # Skip attack period before checking for silence
        # (onset detection finds START of transient, but peak happens after)
        skip_samples = int(skip_attack_ms * sr / 1000)
        
        # Calculate RMS in small windows (10ms)
        window_ms = 10
        window_samples = int(window_ms * sr / 1000)
        
        silence_sample = None
        for j in range(skip_samples, len(segment), window_samples):
            window = segment[j:j+window_samples]
            if len(window) == 0:
                break
            
            # Calculate RMS and convert to dB
            rms = np.sqrt(np.mean(window**2))
            if rms > 0:
                rms_db = 20 * np.log10(rms)
            else:
                rms_db = -100.0  # Effective silence
            
            if rms_db < silence_threshold_db:
                silence_sample = onset_sample + j
                break
        
        # Duration = whichever comes first: next onset or silence
        if silence_sample is not None:
            durations[i] = (silence_sample - onset_sample) / sr
        else:
            durations[i] = next_onset_time - onset_time
    
    return durations


def calculate_amplitude_at_time(
    audio: np.ndarray,
    time_sec: float,
    sr: int,
    window_ms: float = 5.0
) -> float:
    """
    Calculate amplitude at a specific time using windowed RMS.
    
    Pure function - no side effects.
    
    Args:
        audio: Full audio signal (mono)
        time_sec: Time in seconds to measure amplitude
        sr: Sample rate
        window_ms: Window size in milliseconds (centered on time)
        
    Returns:
        RMS amplitude at specified time
    """
    sample = int(time_sec * sr)
    half_window = int(window_ms * sr / 2000)
    
    start = max(0, sample - half_window)
    end = min(len(audio), sample + half_window)
    
    if start >= end:
        return 0.0
    
    segment = audio[start:end]
    return float(np.sqrt(np.mean(segment**2)))


def calculate_attack_sharpness(
    audio: np.ndarray,
    onset_time: float,
    duration: float,
    sr: int,
    attack_portion: float = 0.3
) -> float:
    """
    Calculate attack sharpness using envelope derivative.
    
    Measures how quickly amplitude rises at onset. Sharp transients
    (kick, snare, clap) have high values. Reverb tails have low values.
    
    Pure function - no side effects.
    
    Args:
        audio: Full audio signal (mono)
        onset_time: Onset time in seconds
        duration: Event duration in seconds
        sr: Sample rate
        attack_portion: Fraction of duration to analyze (default 30%)
        
    Returns:
        Attack sharpness in amplitude units per millisecond.
        Typical values: Sharp drums ~0.3-0.5, reverb tails ~0.05-0.1
    """
    onset_sample = int(onset_time * sr)
    attack_samples = int(duration * attack_portion * sr)
    
    if attack_samples < 10:
        return 0.0
    
    end_sample = min(len(audio), onset_sample + attack_samples)
    segment = audio[onset_sample:end_sample]
    
    if len(segment) < 10:
        return 0.0
    
    # Calculate envelope (absolute value with smoothing)
    envelope = np.abs(segment)
    
    # Smooth with small kernel (1ms)
    kernel_size = max(3, int(sr / 1000))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    envelope = np.convolve(envelope, kernel, mode='same')
    
    # Calculate derivative (per sample)
    derivative = np.diff(envelope)
    
    # Scale to per-millisecond for interpretability
    max_derivative_per_sample = float(np.max(derivative)) if len(derivative) > 0 else 0.0
    samples_per_ms = sr / 1000.0
    
    return max_derivative_per_sample * samples_per_ms


def calculate_envelope_continuity(
    audio: np.ndarray,
    onset_time: float,
    duration: float,
    sr: int,
    gap_threshold: float = 0.1
) -> float:
    """
    Calculate envelope continuity (detect gaps/dropouts).
    
    Real hits have continuous energy. Reverb tails often have gaps.
    
    Pure function - no side effects.
    
    Args:
        audio: Full audio signal (mono)
        onset_time: Onset time in seconds
        duration: Event duration in seconds
        sr: Sample rate
        gap_threshold: Relative amplitude threshold for gap detection
        
    Returns:
        Continuity score (0-1, higher = more continuous)
    """
    onset_sample = int(onset_time * sr)
    duration_samples = int(duration * sr)
    
    if duration_samples < 10:
        return 1.0  # Too short to have gaps
    
    end_sample = min(len(audio), onset_sample + duration_samples)
    segment = audio[onset_sample:end_sample]
    
    if len(segment) < 10:
        return 1.0
    
    # Calculate envelope
    envelope = np.abs(segment)
    
    # Smooth with small window
    window_size = max(3, int(sr / 200))  # 5ms
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size) / window_size
    envelope = np.convolve(envelope, kernel, mode='same')
    
    # Find peak amplitude
    peak = np.max(envelope)
    if peak == 0:
        return 1.0
    
    # Count samples above threshold
    threshold = peak * gap_threshold
    above_threshold = envelope > threshold
    continuity = float(np.mean(above_threshold))
    
    return continuity


def calculate_peak_prominence(
    audio: np.ndarray,
    onset_time: float,
    sr: int,
    window_before_ms: float = 20.0,
    window_after_ms: float = 20.0
) -> float:
    """
    Calculate how prominent the peak is relative to surroundings.
    
    Real drum hits stand out from background. Artifacts blend in.
    
    Pure function - no side effects.
    
    Args:
        audio: Full audio signal (mono)
        onset_time: Onset time in seconds
        sr: Sample rate
        window_before_ms: Look-back window in milliseconds
        window_after_ms: Look-ahead window in milliseconds
        
    Returns:
        Prominence ratio (peak / mean_surroundings)
    """
    onset_sample = int(onset_time * sr)
    before_samples = int(window_before_ms * sr / 1000)
    after_samples = int(window_after_ms * sr / 1000)
    
    # Get peak amplitude at onset
    peak_window = int(5 * sr / 1000)  # 5ms around peak
    peak_start = max(0, onset_sample - peak_window // 2)
    peak_end = min(len(audio), onset_sample + peak_window // 2)
    peak_amp = np.max(np.abs(audio[peak_start:peak_end]))
    
    # Get surrounding amplitude
    before_start = max(0, onset_sample - before_samples)
    before_end = onset_sample
    after_start = onset_sample + peak_window
    after_end = min(len(audio), onset_sample + after_samples)
    
    surroundings = np.concatenate([
        audio[before_start:before_end],
        audio[after_start:after_end]
    ])
    
    if len(surroundings) == 0:
        return 1.0
    
    mean_surround = np.mean(np.abs(surroundings))
    
    if mean_surround == 0:
        return 100.0 if peak_amp > 0 else 1.0
    
    return float(peak_amp / mean_surround)


def calculate_spectral_centroid(
    audio: np.ndarray,
    onset_time: float,
    sr: int,
    window_ms: float = 50.0
) -> float:
    """
    Calculate spectral centroid (brightness) of event.
    
    Higher values = brighter sound (cymbals, hi-hat).
    Lower values = darker sound (kick, toms).
    
    Pure function - no side effects.
    
    Args:
        audio: Full audio signal (mono)
        onset_time: Onset time in seconds
        sr: Sample rate
        window_ms: Analysis window in milliseconds
        
    Returns:
        Spectral centroid in Hz
    """
    onset_sample = int(onset_time * sr)
    window_samples = int(window_ms * sr / 1000)
    
    start = onset_sample
    end = min(len(audio), onset_sample + window_samples)
    segment = audio[start:end]
    
    if len(segment) < 100:
        return 0.0
    
    # Compute FFT
    fft = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(len(segment), 1/sr)
    magnitude = np.abs(fft)
    
    # Calculate centroid
    if np.sum(magnitude) == 0:
        return 0.0
    
    centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
    return centroid


def calculate_spectral_flux(
    audio: np.ndarray,
    onset_time: float,
    sr: int,
    window_ms: float = 50.0
) -> float:
    """
    Calculate spectral flux (rate of timbre change).
    
    Higher values = rapidly changing timbre (transients).
    Lower values = stable timbre (sustained notes, reverb).
    
    Pure function - no side effects.
    
    Args:
        audio: Full audio signal (mono)
        onset_time: Onset time in seconds
        sr: Sample rate
        window_ms: Analysis window in milliseconds
        
    Returns:
        Spectral flux (sum of spectral differences)
    """
    onset_sample = int(onset_time * sr)
    window_samples = int(window_ms * sr / 1000)
    
    start = onset_sample
    end = min(len(audio), onset_sample + window_samples)
    segment = audio[start:end]
    
    if len(segment) < 200:
        return 0.0
    
    # Split into two halves
    mid = len(segment) // 2
    first_half = segment[:mid]
    second_half = segment[mid:]
    
    # Compute FFT for each half
    fft1 = np.abs(np.fft.rfft(first_half))
    fft2 = np.abs(np.fft.rfft(second_half))
    
    # Normalize
    if np.sum(fft1) > 0:
        fft1 = fft1 / np.sum(fft1)
    if np.sum(fft2) > 0:
        fft2 = fft2 / np.sum(fft2)
    
    # Calculate flux (difference between spectra)
    min_len = min(len(fft1), len(fft2))
    flux = float(np.sum(np.abs(fft2[:min_len] - fft1[:min_len])))
    
    return flux


def detect_pitch(
    audio: np.ndarray,
    onset_time: float,
    sr: int,
    window_ms: float = 50.0,
    fmin: float = 50.0,
    fmax: float = 2000.0
) -> Optional[float]:
    """
    Detect fundamental pitch using librosa's pyin algorithm.
    
    Pure function - no side effects.
    
    Args:
        audio: Full audio signal (mono)
        onset_time: Onset time in seconds
        sr: Sample rate
        window_ms: Analysis window in milliseconds
        fmin: Minimum frequency to detect (Hz)
        fmax: Maximum frequency to detect (Hz)
        
    Returns:
        Detected pitch in Hz, or None if no pitch detected
    """
    import librosa
    
    onset_sample = int(onset_time * sr)
    window_samples = int(window_ms * sr / 1000)
    
    start = onset_sample
    end = min(len(audio), onset_sample + window_samples)
    segment = audio[start:end]
    
    if len(segment) < 512:
        return None
    
    try:
        # Use pyin for pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            segment,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=512
        )
        
        # Filter out unvoiced frames and NaN values
        valid_pitches = f0[~np.isnan(f0)]
        
        if len(valid_pitches) == 0:
            return None
        
        # Return median pitch
        return float(np.median(valid_pitches))
    except Exception:
        return None


def calculate_gap_from_previous(
    onset_time: float,
    previous_onset_time: Optional[float]
) -> Optional[float]:
    """
    Calculate time gap since previous onset.
    
    Short gaps suggest real rhythm. Long gaps may indicate artifacts.
    
    Pure function - no side effects.
    
    Args:
        onset_time: Current onset time in seconds
        previous_onset_time: Previous onset time in seconds (or None)
        
    Returns:
        Gap in seconds, or None if no previous onset
    """
    if previous_onset_time is None:
        return None
    
    return float(onset_time - previous_onset_time)


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


def analyze_cymbal_decay_pattern(
    audio: np.ndarray,
    onset_sample: int,
    sr: int,
    window_sec: float = 2.0,
    num_windows: int = 8
) -> Dict[str, any]:
    """
    Analyze the spectral energy decay pattern after a cymbal hit.
    
    Divides the analysis window into smaller chunks and measures spectral
    energy in each to detect exponential decay characteristic of a single
    cymbal fading out vs multiple independent hits.
    
    Pure function - no side effects.
    
    Args:
        audio: Audio signal (mono)
        onset_sample: Sample index of onset
        sr: Sample rate
        window_sec: Total analysis window duration in seconds
        num_windows: Number of sub-windows to analyze
    
    Returns:
        Dict with:
        - decay_energies: List of spectral energies over time
        - is_decaying: Boolean indicating if pattern looks like decay
        - decay_rate: Estimated decay rate (negative = decaying)
    """
    window_samples = int(window_sec * sr)
    end_sample = min(onset_sample + window_samples, len(audio))
    total_segment = audio[onset_sample:end_sample]
    
    if len(total_segment) < num_windows * 100:
        return {
            'decay_energies': [],
            'is_decaying': False,
            'decay_rate': 0.0
        }
    
    # Define cymbal frequency range (brilliance/high frequencies)
    freq_ranges = {'cymbal': (4000.0, 20000.0)}
    
    # Measure energy in each sub-window
    chunk_size = len(total_segment) // num_windows
    decay_energies = []
    
    for i in range(num_windows):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = total_segment[start_idx:end_idx]
        
        if len(chunk) < 100:
            break
        
        energies = calculate_spectral_energies(chunk, sr, freq_ranges)
        decay_energies.append(energies['cymbal'])
    
    if len(decay_energies) < 3:
        return {
            'decay_energies': decay_energies,
            'is_decaying': False,
            'decay_rate': 0.0
        }
    
    # Analyze decay pattern
    # A true decay should show monotonic decrease or occasional plateaus
    # NOT increases (which would indicate new hits)
    
    # Calculate relative changes between consecutive windows
    changes = []
    for i in range(1, len(decay_energies)):
        if decay_energies[i-1] > 0:
            change = (decay_energies[i] - decay_energies[i-1]) / decay_energies[i-1]
            changes.append(change)
    
    # Count increases vs decreases
    increases = sum(1 for c in changes if c > 0.1)  # 10% threshold for significant increase
    decreases = sum(1 for c in changes if c < -0.1)
    
    # Pattern is decaying if we see mostly decreases and few increases
    is_decaying = decreases >= increases
    
    # Calculate average decay rate (negative = decaying)
    decay_rate = float(np.mean(changes)) if changes else 0.0
    
    return {
        'decay_energies': decay_energies,
        'is_decaying': is_decaying,
        'decay_rate': decay_rate
    }


def get_spectral_config_for_stem(stem_type: str, config: Dict) -> Dict:
    """
    Get spectral configuration for a specific stem type.
    
    Uses domain-specific frequency band names so that downstream data
    (onset_data dicts, analysis.json) is self-documenting.
    
    Pure function - extracts config without side effects.
    
    Args:
        stem_type: Type of stem ('kick', 'snare', 'toms', 'hihat', 'cymbals')
        config: Full configuration dictionary
    
    Returns:
        Dict with:
        - freq_ranges: Dict of frequency ranges keyed by domain-specific names
        - energy_labels: Dict mapping band names to display labels
        - geomean_bands: Ordered list of band names used for geomean calculation
        - geomean_threshold: Threshold for filtering (or None)
        - min_sustain_ms: Minimum sustain duration (or None)
        - min_strength_threshold: Minimum onset strength (or None)
    """
    stem_config = config.get(stem_type, {})
    
    if stem_type == 'snare':
        return {
            'freq_ranges': {
                'low': (stem_config['low_freq_min'], stem_config['low_freq_max']),
                'body': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'wire': (stem_config['wire_freq_min'], stem_config['wire_freq_max'])
            },
            'energy_labels': {
                'body': 'Body',
                'wire': 'Wire'
            },
            'geomean_bands': ['body', 'wire'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold')
        }
    
    elif stem_type == 'kick':
        return {
            'freq_ranges': {
                'fundamental': (stem_config['fundamental_freq_min'], stem_config['fundamental_freq_max']),
                'body': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'attack': (stem_config['attack_freq_min'], stem_config['attack_freq_max'])
            },
            'energy_labels': {
                'fundamental': 'Fundamental',
                'body': 'Body',
                'attack': 'Attack'
            },
            'geomean_bands': ['fundamental', 'body', 'attack'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold')
        }
    
    elif stem_type == 'toms':
        return {
            'freq_ranges': {
                'fundamental': (stem_config['fundamental_freq_min'], stem_config['fundamental_freq_max']),
                'body': (stem_config['body_freq_min'], stem_config['body_freq_max'])
            },
            'energy_labels': {
                'fundamental': 'Fundamental',
                'body': 'Body'
            },
            'geomean_bands': ['fundamental', 'body'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold')
        }
    
    elif stem_type == 'hihat':
        return {
            'freq_ranges': {
                'body': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'sizzle': (stem_config['sizzle_freq_min'], stem_config['sizzle_freq_max'])
            },
            'energy_labels': {
                'body': 'Body',
                'sizzle': 'Sizzle'
            },
            'geomean_bands': ['body', 'sizzle'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': stem_config.get('min_sustain_ms', 25),
            'min_strength_threshold': stem_config.get('min_strength_threshold')
        }
    
    elif stem_type == 'cymbals':
        return {
            'freq_ranges': {
                'body': (stem_config.get('body_freq_min', 1000), stem_config.get('body_freq_max', 4000)),
                'brilliance': (stem_config.get('brilliance_freq_min', 4000), stem_config.get('brilliance_freq_max', 10000))
            },
            'energy_labels': {
                'body': 'Body',
                'brilliance': 'Brilliance'
            },
            'geomean_bands': ['body', 'brilliance'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': stem_config.get('min_sustain_ms', 150),
            'min_strength_threshold': stem_config.get('min_strength_threshold')
        }
    
    else:
        raise ValueError(f"Unknown stem type: {stem_type}")


def calculate_geomean(primary_energy: float, secondary_energy: float, tertiary_energy: Optional[float] = None) -> float:
    """
    Calculate geometric mean of energy values.
    
    Pure function - no side effects.
    
    Args:
        primary_energy: First energy value
        secondary_energy: Second energy value
        tertiary_energy: Optional third energy value (for 3-way geomean)
    
    Returns:
        Geometric mean (sqrt of product for 2 values, cube root for 3 values)
    """
    if tertiary_energy is not None and tertiary_energy > 0:
        # 3-way geometric mean: cube root of product
        return float(np.cbrt(primary_energy * secondary_energy * tertiary_energy))
    else:
        # 2-way geometric mean: square root of product
        return float(np.sqrt(primary_energy * secondary_energy))


def calculate_statistical_params(onset_data_list: List[Dict]) -> Dict[str, float]:
    """
    Analyze full dataset of onsets to compute normalization parameters.
    
    Used for statistical outlier detection to identify snare bleed in kicks
    by comparing the first two geomean band energies' ratio and total energy
    against dataset medians.
    
    Pure function - no side effects.
    
    Args:
        onset_data_list: List of onset data dicts with domain-specific energy keys
                         (e.g. 'fundamental_energy', 'body_energy') and 'total_energy'.
                         Each dict must have 'geomean_bands' listing band names.
    
    Returns:
        Dict with median and spread values:
        - median_ratio: Median band1/band2 ratio across all events
        - median_total: Median total energy across all events
        - ratio_spread: Standard deviation of ratios
        - total_spread: Standard deviation of total energies
    """
    if not onset_data_list:
        return {
            'median_ratio': 1.0,
            'median_total': 100.0,
            'ratio_spread': 1.0,
            'total_spread': 1.0
        }
    
    # Extract first two geomean band energies using domain-specific names
    bands = onset_data_list[0].get('geomean_bands', [])
    if len(bands) >= 2:
        band1_key = f'{bands[0]}_energy'
        band2_key = f'{bands[1]}_energy'
    else:
        # Fallback: shouldn't happen with proper data
        band1_key = 'fundamental_energy'
        band2_key = 'body_energy'
    
    band1_energies = np.array([d.get(band1_key, 0.0) for d in onset_data_list])
    band2_energies = np.array([d.get(band2_key, 0.0) for d in onset_data_list])
    total_energies = np.array([d['total_energy'] for d in onset_data_list])
    
    # Calculate band1/band2 ratios (with safety for division by zero)
    ratios = band1_energies / np.maximum(band2_energies, 1e-9)
    
    params = {
        'median_ratio': float(np.median(ratios)),
        'median_total': float(np.median(total_energies)),
        'ratio_spread': float(np.std(ratios)) if len(ratios) > 1 else 1.0,
        'total_spread': float(np.std(total_energies)) if len(total_energies) > 1 else 1.0
    }
    
    # Ensure non-zero spreads to avoid division by zero
    if params['ratio_spread'] < 1e-9:
        params['ratio_spread'] = 1e-9
    if params['total_spread'] < 1e-9:
        params['total_spread'] = 1e-9
    
    return params


def calculate_badness_score(
    onset_data: Dict,
    statistical_params: Dict[str, float],
    ratio_weight: float = 0.7,
    total_weight: float = 0.3
) -> float:
    """
    Compute normalized badness score for a single onset.
    
    Measures how much an onset deviates from typical kicks in the dataset.
    Snare bleed typically has lower Primary/Secondary ratio and different total energy.
    
    Pure function - no side effects.
    
    Args:
        onset_data: Dict with domain-specific energy keys and 'geomean_bands'
        statistical_params: Dict from calculate_statistical_params()
        ratio_weight: Weight for ratio deviation (0-1)
        total_weight: Weight for total energy deviation (0-1)
    
    Returns:
        Badness score in range [0, 1]:
        - 0.0 = perfectly typical kick
        - 1.0 = maximum deviation (likely artifact/bleed)
    """
    # Calculate this onset's ratio using first two geomean bands
    bands = onset_data.get('geomean_bands', [])
    if len(bands) >= 2:
        band1 = onset_data.get(f'{bands[0]}_energy', 0.0)
        band2 = onset_data.get(f'{bands[1]}_energy', 0.0)
    else:
        band1 = 0.0
        band2 = 0.0
    ratio = band1 / max(band2, 1e-9)
    total = onset_data['total_energy']
    
    # Calculate normalized deviations
    # Ratio deviation: how much lower is this ratio compared to median?
    ratio_dev = (statistical_params['median_ratio'] - ratio) / (statistical_params['median_ratio'] + 1e-9)
    ratio_dev = float(np.clip(ratio_dev, 0, 1))  # Only penalize lower ratios, not higher
    
    # Total energy deviation: how different is total energy from median?
    total_dev = abs(statistical_params['median_total'] - total) / (statistical_params['median_total'] + 1e-9)
    total_dev = float(np.clip(total_dev, 0, 1))
    
    # Weighted combination
    score = ratio_weight * ratio_dev + total_weight * total_dev
    
    return float(np.clip(score, 0, 1))


def should_keep_onset(
    geomean: float,
    sustain_ms: Optional[float],
    geomean_threshold: Optional[float],
    min_sustain_ms: Optional[float],
    stem_type: str,
    strength: Optional[float] = None,
    min_strength_threshold: Optional[float] = None
) -> bool:
    """
    Determine if an onset should be kept based on spectral/sustain/strength criteria.
    
    Pure function - decision logic without side effects.
    
    Args:
        geomean: Geometric mean of primary and secondary energy
        sustain_ms: Sustain duration in milliseconds (None if not calculated)
        geomean_threshold: Threshold for geomean filtering (None to disable)
        min_sustain_ms: Minimum sustain threshold (None to disable)
        stem_type: Type of stem (affects logic for hihat vs others)
        strength: Onset strength value (0-1, normalized)
        min_strength_threshold: Minimum onset strength required (None to disable)
    
    Returns:
        True if onset should be kept, False if it should be rejected
    """
    # Check strength first (applies to all stem types)
    if min_strength_threshold is not None and strength is not None:
        if strength < min_strength_threshold:
            return False
    
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
        # if min_sustain_ms is not None and sustain_ms is not None:
        #     if sustain_ms >= min_sustain_ms:
        #         return True
        # if geomean_threshold is not None:
        #     return geomean > geomean_threshold
        if geomean_threshold is not None:
            if geomean <= geomean_threshold:
                return False
        return True
    
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


# ============================================================================
# CLASSIFICATION AND MIDI CONVERSION (Pure Functions)
# ============================================================================

def estimate_velocity(strength: float, min_vel: int = 40, max_vel: int = 127) -> int:
    """
    Convert onset strength to MIDI velocity.
    
    Pure function - no side effects.
    
    Args:
        strength: Onset strength (0-1)
        min_vel: Minimum MIDI velocity
        max_vel: Maximum MIDI velocity
    
    Returns:
        MIDI velocity (1-127)
    """
    velocity = int(min_vel + strength * (max_vel - min_vel))
    return np.clip(velocity, 1, 127)


def classify_tom_pitch(pitches: np.ndarray) -> np.ndarray:
    """
    Classify tom pitches into low/mid/high groups using clustering.
    
    Pure function - no side effects.
    
    Args:
        pitches: Array of detected pitches in Hz
    
    Returns:
        Array of classifications: 0=low, 1=mid, 2=high
    """
    if len(pitches) == 0:
        return np.array([])
    
    # Filter out failed detections (0 Hz)
    valid_pitches = pitches[pitches > 0]
    
    if len(valid_pitches) == 0:
        # If no valid pitches, default to mid tom
        return np.ones(len(pitches), dtype=int)
    
    # If only 1-2 unique pitches, simple grouping
    unique_pitches = np.unique(valid_pitches)
    
    if len(unique_pitches) == 1:
        # All same pitch - classify as mid
        return np.ones(len(pitches), dtype=int)
    elif len(unique_pitches) == 2:
        # Two toms - split into low and high
        threshold = np.mean(unique_pitches)
        classifications = np.where(pitches < threshold, 0, 2)
        classifications[pitches == 0] = 1  # Failed detections go to mid
        return classifications
    else:
        # 3+ unique pitches - use k-means clustering with k=3
        try:
            from sklearn.cluster import KMeans
            
            # Reshape for sklearn
            X = valid_pitches.reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Sort clusters by center frequency (0=low, 1=mid, 2=high)
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_cluster_indices = np.argsort(cluster_centers)
            
            # Map cluster labels to sorted positions
            label_mapping = {old: new for new, old in enumerate(sorted_cluster_indices)}
            
            # Classify all pitches (including failed detections)
            classifications = np.ones(len(pitches), dtype=int)  # Default to mid
            valid_idx = 0
            for i, pitch in enumerate(pitches):
                if pitch > 0:
                    cluster_label = kmeans.labels_[valid_idx]
                    classifications[i] = label_mapping[cluster_label]
                    valid_idx += 1
            
            return classifications
            
        except ImportError:
            # Fallback: use percentiles to split into 3 groups
            p33 = np.percentile(valid_pitches, 33)
            p66 = np.percentile(valid_pitches, 66)
            
            classifications = np.ones(len(pitches), dtype=int)  # Default to mid
            for i, pitch in enumerate(pitches):
                if pitch > 0:
                    if pitch < p33:
                        classifications[i] = 0  # Low
                    elif pitch > p66:
                        classifications[i] = 2  # High
                    else:
                        classifications[i] = 1  # Mid
            return classifications


def classify_cymbal_pitch(pitches: np.ndarray) -> np.ndarray:
    """
    Classify cymbal pitches into crash/ride/chinese groups using clustering.
    
    Pure function - no side effects.
    
    Args:
        pitches: Array of detected pitches in Hz
    
    Returns:
        Array of classifications: 0=crash, 1=ride, 2=chinese
    """
    if len(pitches) == 0:
        return np.array([])
    
    # Filter out failed detections (0 Hz)
    valid_pitches = pitches[pitches > 0]
    
    if len(valid_pitches) == 0:
        # If no valid pitches, default to crash
        return np.zeros(len(pitches), dtype=int)
    
    # If only 1-2 unique pitches, simple grouping
    unique_pitches = np.unique(valid_pitches)
    
    if len(unique_pitches) == 1:
        # All same pitch - classify as crash
        return np.zeros(len(pitches), dtype=int)
    elif len(unique_pitches) == 2:
        # Two cymbals - split into crash and chinese
        threshold = np.mean(unique_pitches)
        classifications = np.where(pitches < threshold, 0, 2)
        classifications[pitches == 0] = 0  # Failed detections go to crash
        return classifications
    else:
        # 3+ unique pitches - use k-means clustering with k=3
        try:
            from sklearn.cluster import KMeans
            
            # Reshape for sklearn
            X = valid_pitches.reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Sort clusters by center frequency (0=crash, 1=ride, 2=chinese)
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_cluster_indices = np.argsort(cluster_centers)
            
            # Map cluster labels to sorted positions
            label_mapping = {old: new for new, old in enumerate(sorted_cluster_indices)}
            
            # Classify all pitches (including failed detections)
            classifications = np.zeros(len(pitches), dtype=int)  # Default to crash
            valid_idx = 0
            for i, pitch in enumerate(pitches):
                if pitch > 0:
                    cluster_label = kmeans.labels_[valid_idx]
                    classifications[i] = label_mapping[cluster_label]
                    valid_idx += 1
            
            return classifications
            
        except ImportError:
            # Fallback: use percentiles to split into 3 groups
            p33 = np.percentile(valid_pitches, 33)
            p66 = np.percentile(valid_pitches, 66)
            
            classifications = np.zeros(len(pitches), dtype=int)  # Default to crash
            for i, pitch in enumerate(pitches):
                if pitch > 0:
                    if pitch < p33:
                        classifications[i] = 0  # Crash (lower)
                    elif pitch > p66:
                        classifications[i] = 2  # Chinese (higher)
                    else:
                        classifications[i] = 1  # Ride (mid)
            return classifications


def classify_snare_pitch(pitches: np.ndarray) -> np.ndarray:
    """
    Classify snare hits into 4 types using clustering.
    
    Pure function - no side effects.
    
    Types:
    0 = snare (most common, mid-range pitch)
    1 = rimshot (higher pitch, sharper)
    2 = clap only (highest pitch, thin sound)
    3 = clap+snare (layered, mid-high pitch)
    
    Note: Later can be enhanced with stereo info and envelope profile.
    
    Args:
        pitches: Array of detected pitches in Hz
    
    Returns:
        Array of classifications: 0=snare, 1=rimshot, 2=clap, 3=clap+snare
    """
    if len(pitches) == 0:
        return np.array([])
    
    # Filter out failed detections (0 Hz)
    valid_pitches = pitches[pitches > 0]
    
    if len(valid_pitches) == 0:
        # If no valid pitches, default to snare
        return np.zeros(len(pitches), dtype=int)
    
    # If only 1-3 unique pitches, simple grouping
    unique_pitches = np.unique(valid_pitches)
    
    if len(unique_pitches) == 1:
        # All same pitch - classify as snare
        return np.zeros(len(pitches), dtype=int)
    elif len(unique_pitches) == 2:
        # Two types - split into snare and rimshot
        threshold = np.mean(unique_pitches)
        classifications = np.where(pitches < threshold, 0, 1)
        classifications[pitches == 0] = 0  # Failed detections go to snare
        return classifications
    elif len(unique_pitches) == 3:
        # Three types - use percentiles
        p33 = np.percentile(valid_pitches, 33)
        p66 = np.percentile(valid_pitches, 66)
        
        classifications = np.zeros(len(pitches), dtype=int)
        for i, pitch in enumerate(pitches):
            if pitch > 0:
                if pitch < p33:
                    classifications[i] = 0  # Snare (lower)
                elif pitch > p66:
                    classifications[i] = 2  # Clap (higher)
                else:
                    classifications[i] = 1  # Rimshot (mid)
        return classifications
    else:
        # 4+ unique pitches - use k-means clustering with k=4
        try:
            from sklearn.cluster import KMeans
            
            # Reshape for sklearn
            X = valid_pitches.reshape(-1, 1)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Sort clusters by center frequency (0=low, 1=mid-low, 2=mid-high, 3=high)
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_cluster_indices = np.argsort(cluster_centers)
            
            # Map to snare types based on pitch order
            # Lowest pitch = snare (0)
            # Second = rimshot (1) 
            # Third = clap+snare (3)
            # Highest = clap only (2)
            pitch_to_type = {sorted_cluster_indices[0]: 0,  # Lowest -> snare
                           sorted_cluster_indices[1]: 1,  # Mid-low -> rimshot
                           sorted_cluster_indices[2]: 3,  # Mid-high -> clap+snare
                           sorted_cluster_indices[3]: 2}  # Highest -> clap
            
            # Classify all pitches (including failed detections)
            classifications = np.zeros(len(pitches), dtype=int)  # Default to snare
            valid_idx = 0
            for i, pitch in enumerate(pitches):
                if pitch > 0:
                    cluster_label = kmeans.labels_[valid_idx]
                    classifications[i] = pitch_to_type[cluster_label]
                    valid_idx += 1
            
            return classifications
            
        except ImportError:
            # Fallback: use percentiles to split into 4 groups
            p25 = np.percentile(valid_pitches, 25)
            p50 = np.percentile(valid_pitches, 50)
            p75 = np.percentile(valid_pitches, 75)
            
            classifications = np.zeros(len(pitches), dtype=int)  # Default to snare
            for i, pitch in enumerate(pitches):
                if pitch > 0:
                    if pitch < p25:
                        classifications[i] = 0  # Snare (lowest)
                    elif pitch < p50:
                        classifications[i] = 1  # Rimshot
                    elif pitch < p75:
                        classifications[i] = 3  # Clap+Snare
                    else:
                        classifications[i] = 2  # Clap only (highest)
            return classifications


# ============================================================================
# ONSET FILTERING AND ANALYSIS (Pure Functions)
# ============================================================================

def mark_reverb_continuations(
    onset_data_list: List[Dict],
    time_margin_ms: float = 5.0,
    amplitude_margin: float = 0.001,
    attack_sharpness_threshold: float = 0.2
) -> List[Dict]:
    """
    Mark reverb continuation events in onset data.
    
    Reverb continuations are artifacts where peak-hold detection splits
    a single reverb/echo envelope into multiple events. Characteristics:
    - Next event starts exactly when previous ends (within time_margin_ms)
    - Amplitude continuity: start matches previous end (within amplitude_margin)
    - Low attack sharpness (< threshold): reverb/echo have smooth envelopes, not sharp attacks
    
    Note: Amplitude can increase or decrease with complex reverb (reflections building up).
    
    These events are marked as 'REVERB_CONTINUATION' status to preserve
    data while allowing MIDI export to filter them.
    
    Pure function - modifies input list in place but returns it for chaining.
    
    Args:
        onset_data_list: List of onset dicts (must have time, duration_sec, 
                        amplitude, amplitude_at_start, amplitude_at_end, 
                        attack_sharpness, status)
        time_margin_ms: Maximum gap between events (default 5ms)
        amplitude_margin: Maximum amplitude difference (default 0.001)
        attack_sharpness_threshold: Minimum attack sharpness for real hits (default 0.2)
    
    Returns:
        Modified onset_data_list with 'REVERB_CONTINUATION' status set
    """
    if len(onset_data_list) < 2:
        return onset_data_list
    
    # Sort by time to ensure sequential processing
    onset_data_list.sort(key=lambda e: e['time'])
    
    time_margin_sec = time_margin_ms / 1000.0
    
    for i in range(1, len(onset_data_list)):
        prev = onset_data_list[i - 1]
        curr = onset_data_list[i]
        
        # Skip if current event already rejected, or if previous is rejected
        # (but allow previous to be REVERB_CONTINUATION to catch full decay chains)
        if curr.get('status') != 'KEPT':
            continue
        if prev.get('status') not in ('KEPT', 'REVERB_CONTINUATION'):
            continue
        
        # Check required fields exist
        if ('duration_sec' not in prev or 'amplitude_at_end' not in prev or 
            'amplitude_at_start' not in curr or 'amplitude' not in prev or 
            'amplitude' not in curr):
            continue
        
        # Calculate timing
        prev_end_time = prev['time'] + prev['duration_sec']
        gap = curr['time'] - prev_end_time
        
        # Check continuation conditions
        is_adjacent = abs(gap) <= time_margin_sec
        
        # Amplitude continuity - envelope connects smoothly
        prev_end_amp = prev['amplitude_at_end']
        curr_start_amp = curr['amplitude_at_start']
        amp_diff = abs(curr_start_amp - prev_end_amp)
        is_amplitude_continuous = amp_diff <= amplitude_margin
        
        # Attack sharpness check - real hits have sharp attacks (>= threshold)
        # Reverb/echo tails have smooth envelopes (< threshold)
        # Note: Complex reverb can have increasing amplitude (reflections building up)
        curr_attack_sharpness = curr.get('attack_sharpness')
        is_smooth_envelope = (curr_attack_sharpness is not None and 
                             curr_attack_sharpness < attack_sharpness_threshold)
        
        # Mark as reverb continuation if all conditions met
        if is_adjacent and is_amplitude_continuous and is_smooth_envelope:
            curr['status'] = 'REVERB_CONTINUATION'
    
    return onset_data_list


def filter_onsets_by_spectral(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    peak_amplitudes: np.ndarray,
    audio: np.ndarray,
    sr: int,
    stem_type: str,
    config: Dict,
    learning_mode: bool = False,
    durations: Optional[np.ndarray] = None
) -> Dict:
    """
    Filter onsets by spectral content and analyze each onset.
    
    Pure function - no side effects, no I/O.
    
    Detection Output Contract (Producer):
        This function PRODUCES SpectralOnsetData for each kept onset.
        Contract defined in midi_types.py - see SpectralOnsetData TypedDict.
        filtered_onset_data contains full analysis for all KEPT onsets.
        Consumers: detect_hihat_state(), learning.py, processing_shell.py
    
    Args:
        onset_times: Array of onset times in seconds
        onset_strengths: Array of onset strengths (0-1)
        peak_amplitudes: Array of peak amplitudes
        audio: Audio signal (mono)
        sr: Sample rate
        stem_type: Type of stem ('kick', 'snare', etc.)
        config: Configuration dictionary
        learning_mode: If True, keep all onsets (don't filter)
        durations: Optional array of event durations in seconds. If provided,
                  passed to analyze_onset_spectral for Phase 2 metadata.
    
    Returns:
        Dictionary with:
        - filtered_times: np.ndarray
        - filtered_strengths: np.ndarray
        - filtered_amplitudes: np.ndarray
        - filtered_geomeans: np.ndarray
        - filtered_sustains: List (for hihat and cymbals)
        - filtered_spectral: List (for hihat only)
        - filtered_onset_data: List[SpectralOnsetData] - contract-compliant spectral data
        - all_onset_data: List[Dict] (analysis for all onsets, for debugging)
        - spectral_config: Dict (configuration used)
    """
    if len(onset_times) == 0:
        return {
            'filtered_times': np.array([]),
            'filtered_strengths': np.array([]),
            'filtered_amplitudes': np.array([]),
            'filtered_geomeans': np.array([]),
            'filtered_sustains': [],
            'filtered_spectral': [],
            'filtered_onset_data': [],  # Full spectral data for KEPT onsets
            'all_onset_data': [],
            'spectral_config': None
        }
    
    # Get spectral configuration for this stem type
    spectral_config = get_spectral_config_for_stem(stem_type, config)
    geomean_threshold = spectral_config['geomean_threshold']
    min_sustain_ms = spectral_config['min_sustain_ms']
    energy_labels = spectral_config['energy_labels']
    geomean_bands = spectral_config['geomean_bands']
    
    # Storage for filtered results
    filtered_times = []
    filtered_strengths = []
    filtered_amplitudes = []
    filtered_geomeans = []
    filtered_sustains = []  # For hihat open/closed detection
    filtered_spectral = []  # For hihat handclap detection
    filtered_onset_data = []  # Full spectral data for KEPT onsets (for Detection Output Contract)
    
    # Store raw spectral data for ALL onsets (for debug output)
    all_onset_data = []
    
    # Handle optional durations parameter (backward compatibility)
    if durations is None:
        durations = [None] * len(onset_times)
    
    for onset_time, strength, peak_amplitude, duration in zip(onset_times, onset_strengths, peak_amplitudes, durations):
        # Use unified spectral analysis helper (now with duration)
        analysis = analyze_onset_spectral(audio, onset_time, sr, stem_type, config, duration=duration)
        
        if analysis is None:
            # Segment too short, skip
            continue
        
        # Extract results from analysis (domain-specific band names)
        low_energy = analysis['low_energy']
        total_energy = analysis['total_energy']
        geomean = analysis['geomean']
        sustain_duration = analysis['sustain_ms']
        spectral_ratio = analysis['spectral_ratio']
        
        # Extract band energies using geomean_bands order
        band_energies = {band: analysis.get(f'{band}_energy', 0.0) for band in geomean_bands}
        
        # Phase 2: Calculate extended metadata (if duration available)
        amplitude_at_start = None
        amplitude_at_end = None
        attack_sharpness = None
        envelope_continuity = None
        peak_prominence = None
        spectral_centroid_hz = None
        spectral_flux_value = None
        detected_pitch = None
        gap_from_previous = None
        
        if duration is not None:
            # Amplitude at start and end
            amplitude_at_start = calculate_amplitude_at_time(audio, onset_time, sr, window_ms=5.0)
            amplitude_at_end = calculate_amplitude_at_time(audio, onset_time + duration, sr, window_ms=5.0)
            
            # Attack characteristics
            attack_sharpness = calculate_attack_sharpness(audio, onset_time, duration, sr)
            envelope_continuity = calculate_envelope_continuity(audio, onset_time, duration, sr)
            
            # Peak prominence
            peak_prominence = calculate_peak_prominence(audio, onset_time, sr)
            
            # Spectral features
            spectral_centroid_hz = calculate_spectral_centroid(audio, onset_time, sr)
            spectral_flux_value = calculate_spectral_flux(audio, onset_time, sr)
            
            # Pitch detection (optional, can be slow)
            # detected_pitch = detect_pitch(audio, onset_time, sr)
            
            # Gap from previous onset
            if len(filtered_times) > 0:
                gap_from_previous = calculate_gap_from_previous(onset_time, filtered_times[-1])
        
        # Determine if this onset should be kept
        is_real_hit = should_keep_onset(
            geomean=geomean,
            sustain_ms=sustain_duration,
            geomean_threshold=geomean_threshold,
            min_sustain_ms=min_sustain_ms,
            stem_type=stem_type,
            strength=strength,
            min_strength_threshold=spectral_config.get('min_strength_threshold')
        )
        
        # Store all data for this onset (for debug output AND sidecar v2)
        # Uses domain-specific band names (body_energy, wire_energy, etc.)
        onset_data = {
            'time': onset_time,
            'strength': strength,
            'amplitude': peak_amplitude,
            'low_energy': low_energy,
            'ratio': spectral_ratio,
            'total_energy': total_energy,
            'geomean': geomean,
            'geomean_bands': geomean_bands,
            'status': 'KEPT' if (learning_mode or is_real_hit) else 'FILTERED'
        }
        
        # Add domain-specific band energies and labels
        for band_name in geomean_bands:
            onset_data[f'{band_name}_energy'] = band_energies[band_name]
            onset_data[f'{band_name}_label'] = energy_labels.get(band_name, band_name.title())
        
        if sustain_duration is not None:
            onset_data['sustain_ms'] = sustain_duration
        
        # Add Phase 2 metadata (if calculated)
        if duration is not None:
            onset_data['duration_sec'] = duration
        if amplitude_at_start is not None:
            onset_data['amplitude_at_start'] = amplitude_at_start
        if amplitude_at_end is not None:
            onset_data['amplitude_at_end'] = amplitude_at_end
        if attack_sharpness is not None:
            onset_data['attack_sharpness'] = attack_sharpness
        if envelope_continuity is not None:
            onset_data['envelope_continuity'] = envelope_continuity
        if peak_prominence is not None:
            onset_data['peak_prominence'] = peak_prominence
        if spectral_centroid_hz is not None:
            onset_data['spectral_centroid_hz'] = spectral_centroid_hz
        if spectral_flux_value is not None:
            onset_data['spectral_flux'] = spectral_flux_value
        if detected_pitch is not None:
            onset_data['pitch_hz'] = detected_pitch
        if gap_from_previous is not None:
            onset_data['gap_from_previous_sec'] = gap_from_previous
        
        all_onset_data.append(onset_data)
        
        # In learning mode, keep ALL detections
        if learning_mode or is_real_hit:
            filtered_times.append(onset_time)
            filtered_strengths.append(strength)
            filtered_amplitudes.append(peak_amplitude)
            filtered_geomeans.append(geomean)
            # Store sustain duration and spectral data for hihat/cymbal classification
            if stem_type in ['hihat', 'cymbals'] and sustain_duration is not None:
                filtered_sustains.append(sustain_duration)
                if stem_type == 'hihat':
                    # Hihat open/closed detection uses body and sizzle energies
                    filtered_spectral.append({
                        'body_energy': band_energies.get('body', 0.0),
                        'sizzle_energy': band_energies.get('sizzle', 0.0)
                    })
            # Store full spectral data for this KEPT onset (Detection Output Contract)
            filtered_onset_data.append(onset_data.copy())
    
    # SECOND PASS: Remove cymbal retriggering using decay pattern analysis
    # Cymbals can have energy modulation during sustain that looks like new onsets
    # Analyze spectral decay pattern to distinguish true hits from decay artifacts
    if stem_type == 'cymbals' and not learning_mode and len(filtered_times) > 1:
        # Get decay filter window for cymbals (separate from sustain analysis window)
        cymbal_config = config.get('cymbals', {})
        enable_decay_filter = cymbal_config.get('enable_decay_filter', True)
        
        if not enable_decay_filter:
            # Skip decay filtering if disabled
            pass
        else:
            decay_filter_window_sec = cymbal_config.get('decay_filter_window_sec', 0.5)
            
            # Build list of times to keep
            final_times = []
            final_strengths = []
            final_amplitudes = []
            final_geomeans = []
            final_sustains = []
            final_onset_data = []  # Track spectral data through decay filter
            
            # Track all decay analysis for debug output
            decay_analysis_data = []
            
            # Track active decay zones (onset_time -> decay pattern)
            active_decays = {}
            
            for i in range(len(filtered_times)):
                current_time = filtered_times[i]
                current_sample = int(current_time * sr)
                
                # Check if this onset falls within any active decay zone
                is_retrigger = False
                prev_hit_time = None
                prev_decay_rate = None
                prev_is_decaying = None
                time_since_prev = None
                
                for prev_time, decay_info in active_decays.items():
                    time_diff = current_time - prev_time
                    
                    # If within decay window
                    if 0 < time_diff < decay_filter_window_sec:
                        # Check if we're in a decaying region
                        if decay_info['is_decaying']:
                            is_retrigger = True
                            prev_hit_time = prev_time
                            prev_decay_rate = decay_info['decay_rate']
                            prev_is_decaying = decay_info['is_decaying']
                            time_since_prev = time_diff
                            break
                
                # Store analysis data
                analysis_entry = {
                    'time': current_time,
                    'is_retrigger': is_retrigger,
                    'prev_hit_time': prev_hit_time,
                    'time_since_prev': time_since_prev,
                    'prev_decay_rate': prev_decay_rate,
                    'prev_is_decaying': prev_is_decaying,
                    'geomean': filtered_geomeans[i],
                    'sustain_ms': filtered_sustains[i] if i < len(filtered_sustains) else None
                }
                
                if not is_retrigger:
                    # This is a legitimate hit - keep it
                    final_times.append(filtered_times[i])
                    final_strengths.append(filtered_strengths[i])
                    final_amplitudes.append(filtered_amplitudes[i])
                    final_geomeans.append(filtered_geomeans[i])
                    if i < len(filtered_sustains):
                        final_sustains.append(filtered_sustains[i])
                    if i < len(filtered_onset_data):
                        final_onset_data.append(filtered_onset_data[i])
                    
                    # Analyze decay pattern starting from this onset
                    decay_pattern = analyze_cymbal_decay_pattern(
                        audio, current_sample, sr, 
                        window_sec=decay_filter_window_sec,
                        num_windows=8
                    )
                    
                    # Store decay pattern info in analysis entry
                    analysis_entry['own_decay_rate'] = decay_pattern['decay_rate']
                    analysis_entry['own_is_decaying'] = decay_pattern['is_decaying']
                    
                    # Store for checking subsequent onsets
                    active_decays[current_time] = decay_pattern
                    
                    # Clean up old decays outside the window
                    active_decays = {
                        t: info for t, info in active_decays.items()
                        if current_time - t < decay_filter_window_sec
                    }
                
                decay_analysis_data.append(analysis_entry)
            
            # Update filtered arrays
            filtered_times = final_times
            filtered_strengths = final_strengths
            filtered_amplitudes = final_amplitudes
            filtered_geomeans = final_geomeans
            filtered_sustains = final_sustains
            filtered_onset_data = final_onset_data
            
            # Update status in all_onset_data for retriggered events
            final_times_set = set(final_times)
            for onset_data in all_onset_data:
                if onset_data['status'] == 'KEPT' and onset_data['time'] not in final_times_set:
                    onset_data['status'] = 'FILTERED'  # Filtered by decay pass
    
    # THIRD PASS: Statistical outlier detection (kick only, if enabled)
    # This catches snare bleed that passes geomean threshold but has abnormal Primary/Secondary ratio
    
    if stem_type == 'kick' and not learning_mode:
        stem_config = config.get('kick', {})
        enable_statistical = stem_config.get('enable_statistical_filter', False)
        
        if enable_statistical and len(all_onset_data) > 0:
            # Calculate statistical parameters from ALL detected onsets (including rejected ones)
            # This gives us the population statistics to compare against
            statistical_params = calculate_statistical_params(all_onset_data)
            
            # Get thresholds from config
            badness_threshold = stem_config.get('statistical_badness_threshold', 0.6)
            ratio_weight = stem_config.get('statistical_ratio_weight', 0.7)
            total_weight = stem_config.get('statistical_total_weight', 0.3)
            
            # Calculate badness scores for ALL onsets (for debug output)
            for onset_data in all_onset_data:
                badness = calculate_badness_score(
                    onset_data,
                    statistical_params,
                    ratio_weight,
                    total_weight
                )
                onset_data['badness_score'] = badness
            
            # Re-filter the already-filtered onsets (Pass 1 survivors) using statistical scores
            # Build a map of time -> onset_data for quick lookup
            onset_data_by_time = {d['time']: d for d in all_onset_data}
            
            final_times = []
            final_strengths = []
            final_amplitudes = []
            final_geomeans = []
            final_onset_data = []
            
            for i, (time, strength, amplitude, geomean) in enumerate(zip(
                filtered_times, filtered_strengths, filtered_amplitudes, filtered_geomeans
            )):
                onset_data = onset_data_by_time.get(time)
                if onset_data and onset_data.get('badness_score', 0) <= badness_threshold:
                    final_times.append(time)
                    final_strengths.append(strength)
                    final_amplitudes.append(amplitude)
                    final_geomeans.append(geomean)
                    if i < len(filtered_onset_data):
                        final_onset_data.append(filtered_onset_data[i])
            
            # Update filtered arrays with statistical filter results
            filtered_times = final_times
            filtered_strengths = final_strengths
            filtered_amplitudes = final_amplitudes
            filtered_geomeans = final_geomeans
            filtered_onset_data = final_onset_data
            
            # Update status in all_onset_data for statistically rejected events
            final_times_set = set(final_times)
            for onset_data in all_onset_data:
                if onset_data['status'] == 'KEPT' and onset_data['time'] not in final_times_set:
                    onset_data['status'] = 'FILTERED'  # Filtered by statistical pass
            
            # Store statistical info in config for debug output
            spectral_config['statistical_params'] = statistical_params
            spectral_config['statistical_enabled'] = True
            spectral_config['badness_threshold'] = badness_threshold
    
    # Prepare decay analysis data for return (only for cymbals with decay filter enabled)
    decay_analysis = None
    if stem_type == 'cymbals' and 'decay_analysis_data' in locals():
        decay_analysis = {
            'data': decay_analysis_data,
            'window_sec': decay_filter_window_sec
        }
    
    # Mark reverb continuation events (peak-hold detection artifact filtering)
    # These are kept in the data but marked so MIDI export can filter them
    # Uses attack_sharpness to distinguish real hits from reverb tails
    attack_threshold = config.get('filtering', {}).get('reverb_continuation_attack_threshold', 0.2)
    all_onset_data = mark_reverb_continuations(
        all_onset_data,
        time_margin_ms=5.0,
        amplitude_margin=0.001,
        attack_sharpness_threshold=attack_threshold  # Real hits have sharper attacks (>= 0.2)
    )
    
    # Remove reverb continuations from filtered lists so they don't become MIDI notes
    reverb_times = {e['time'] for e in all_onset_data if e.get('status') == 'REVERB_CONTINUATION'}
    if reverb_times:
        # Filter out reverb continuations from all filtered arrays
        keep_indices = [i for i, t in enumerate(filtered_times) if t not in reverb_times]
        filtered_times = [filtered_times[i] for i in keep_indices]
        filtered_strengths = [filtered_strengths[i] for i in keep_indices]
        filtered_amplitudes = [filtered_amplitudes[i] for i in keep_indices]
        filtered_geomeans = [filtered_geomeans[i] for i in keep_indices]
        if filtered_sustains:
            filtered_sustains = [filtered_sustains[i] for i in keep_indices if i < len(filtered_sustains)]
        if filtered_spectral:
            filtered_spectral = [filtered_spectral[i] for i in keep_indices if i < len(filtered_spectral)]
        filtered_onset_data = [d for d in filtered_onset_data if d['time'] not in reverb_times]
    
    return {
        'filtered_times': np.array(filtered_times),
        'filtered_strengths': np.array(filtered_strengths),
        'filtered_amplitudes': np.array(filtered_amplitudes),
        'filtered_geomeans': np.array(filtered_geomeans),
        'filtered_sustains': filtered_sustains,
        'filtered_spectral': filtered_spectral,
        'filtered_onset_data': filtered_onset_data,  # Full spectral data for KEPT onsets
        'all_onset_data': all_onset_data,
        'spectral_config': spectral_config,
        'decay_analysis': decay_analysis
    }


def calculate_velocities_from_features(
    feature_values: np.ndarray,
    min_velocity: int,
    max_velocity: int
) -> np.ndarray:
    """
    Calculate MIDI velocities from normalized feature values.
    
    Pure function - no side effects.
    
    Args:
        feature_values: Normalized feature values (0-1 range, can be any feature like geomean or amplitude)
        min_velocity: Minimum MIDI velocity
        max_velocity: Maximum MIDI velocity
    
    Returns:
        Array of MIDI velocities (1-127)
    """
    if len(feature_values) == 0:
        return np.array([], dtype=int)
    
    # Calculate velocities using estimate_velocity for each value
    velocities = np.array([
        estimate_velocity(value, min_velocity, max_velocity)
        for value in feature_values
    ])
    
    return velocities


# ============================================================================
# THRESHOLD LEARNING AND ANALYSIS (Pure Functions)
# ============================================================================

def calculate_threshold_from_distributions(
    kept_values: List[float],
    removed_values: List[float]
) -> Optional[float]:
    """
    Calculate optimal threshold as midpoint between max removed and min kept.
    
    Pure function - no side effects.
    
    Args:
        kept_values: List of values that should be kept (true positives)
        removed_values: List of values that should be removed (false positives)
    
    Returns:
        Suggested threshold (midpoint), or None if insufficient data
    """
    if not kept_values or not removed_values:
        return None
    
    min_kept = min(kept_values)
    max_removed = max(removed_values)
    
    # Threshold is midpoint between max removed and min kept
    suggested_threshold = (max_removed + min_kept) / 2.0
    
    return suggested_threshold


def calculate_classification_accuracy(
    user_actions: List[str],
    predictions: List[str]
) -> Dict[str, float]:
    """
    Calculate classification accuracy between user actions and predictions.
    
    Pure function - no side effects.
    
    Args:
        user_actions: List of 'KEPT' or 'REMOVED' (ground truth)
        predictions: List of 'KEPT' or 'REMOVED' (predicted by threshold)
    
    Returns:
        Dict with:
        - correct_count: Number of correct predictions
        - total_count: Total number of predictions
        - accuracy: Accuracy percentage (0-100)
    """
    if len(user_actions) != len(predictions) or len(user_actions) == 0:
        return {
            'correct_count': 0,
            'total_count': 0,
            'accuracy': 0.0
        }
    
    correct_count = sum(1 for user, pred in zip(user_actions, predictions) if user == pred)
    total_count = len(user_actions)
    accuracy = (correct_count / total_count) * 100.0
    
    return {
        'correct_count': correct_count,
        'total_count': total_count,
        'accuracy': accuracy
    }


def predict_classification(
    geomean: float,
    geomean_threshold: float,
    sustain_ms: Optional[float] = None,
    sustain_threshold: Optional[float] = None,
    stem_type: str = 'snare'
) -> str:
    """
    Predict classification (KEPT/REMOVED) based on thresholds.
    
    Pure function - no side effects.
    
    Args:
        geomean: Geometric mean value
        geomean_threshold: Threshold for geomean
        sustain_ms: Sustain duration in milliseconds (optional)
        sustain_threshold: Threshold for sustain (optional)
        stem_type: Type of stem (affects logic for cymbals)
    
    Returns:
        'KEPT' or 'REMOVED'
    """
    # For cymbals, require both geomean AND sustain if both thresholds provided
    if stem_type == 'cymbals' and sustain_threshold is not None and sustain_ms is not None:
        geomean_ok = geomean > geomean_threshold
        sustain_ok = sustain_ms > sustain_threshold
        return 'KEPT' if (geomean_ok and sustain_ok) else 'REMOVED'
    else:
        # For other stems, just check geomean
        return 'KEPT' if geomean > geomean_threshold else 'REMOVED'


def analyze_threshold_performance(
    analysis_data: List[Dict],
    geomean_threshold: float,
    sustain_threshold: Optional[float] = None,
    stem_type: str = 'snare'
) -> Dict:
    """
    Analyze threshold performance on a dataset.
    
    Pure function - no side effects.
    
    Args:
        analysis_data: List of dicts with 'is_kept', 'geomean', 'sustain_ms' (optional)
        geomean_threshold: Threshold to test
        sustain_threshold: Sustain threshold (optional)
        stem_type: Type of stem
    
    Returns:
        Dict with:
        - user_actions: List[str] ('KEPT' or 'REMOVED')
        - predictions: List[str] ('KEPT' or 'REMOVED')
        - results: List[str] (comparison results like '✓ Both OK')
        - accuracy: Dict from calculate_classification_accuracy
    """
    user_actions = []
    predictions = []
    results = []
    
    for data in analysis_data:
        user_action = 'KEPT' if data['is_kept'] else 'REMOVED'
        
        prediction = predict_classification(
            data['geomean'],
            geomean_threshold,
            data.get('sustain_ms'),
            sustain_threshold,
            stem_type
        )
        
        user_actions.append(user_action)
        predictions.append(prediction)
        
        # Determine result string
        if user_action == prediction:
            results.append('✓ Correct')
        else:
            results.append('✗ Wrong')
    
    accuracy = calculate_classification_accuracy(user_actions, predictions)
    
    return {
        'user_actions': user_actions,
        'predictions': predictions,
        'results': results,
        'accuracy': accuracy
    }


# ============================================================================
# TIME AND SAMPLE CONVERSION (Pure Functions)
# ============================================================================

def time_to_sample(time_sec: float, sr: int) -> int:
    """
    Convert time in seconds to sample index.
    
    Pure function - no side effects.
    
    Args:
        time_sec: Time in seconds
        sr: Sample rate
    
    Returns:
        Sample index (integer)
    """
    return int(time_sec * sr)


def seconds_to_beats(time_sec: float, tempo: float) -> float:
    """
    Convert time in seconds to beats based on tempo.
    
    Pure function - no side effects.
    
    Args:
        time_sec: Time in seconds
        tempo: Tempo in BPM (beats per minute)
    
    Returns:
        Time in beats
    """
    beats_per_second = tempo / 60.0
    return time_sec * beats_per_second


def prepare_midi_events_for_writing(
    events_by_stem: Dict[str, List[Dict]],
    tempo: float
) -> List[Dict]:
    """
    Prepare MIDI events for writing by converting times to beats.
    
    Pure function - no side effects.
    
    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        tempo: Tempo in BPM
    
    Returns:
        List of events with times converted to beats, flattened from all stems
    """
    # Minimum duration to prevent MIDI library errors
    # At 120 BPM, 0.01 beats = 5ms
    MIN_DURATION_BEATS = 0.01
    
    prepared_events = []
    
    for stem_type, events in events_by_stem.items():
        for event in events:
            duration_beats = seconds_to_beats(event['duration'], tempo)
            
            # BUGFIX: Enforce minimum duration to prevent midiutil errors
            # Zero or negative durations cause "pop from empty list" in deInterleaveNotes
            if duration_beats < MIN_DURATION_BEATS:
                duration_beats = MIN_DURATION_BEATS
            
            prepared_event = {
                'note': event['note'],
                'velocity': event['velocity'],
                'time_beats': seconds_to_beats(event['time'], tempo),
                'duration_beats': duration_beats,
                'stem_type': stem_type
            }
            prepared_events.append(prepared_event)
    
    return prepared_events


def extract_audio_segment(
    audio: np.ndarray,
    onset_sample: int,
    window_sec: float,
    sr: int
) -> np.ndarray:
    """
    Extract audio segment starting at onset for specified duration.
    
    Pure function - no side effects.
    
    Args:
        audio: Audio signal
        onset_sample: Starting sample index
        window_sec: Window duration in seconds
        sr: Sample rate
    
    Returns:
        Audio segment (may be shorter than requested if at end of audio)
    """
    window_samples = int(window_sec * sr)
    end_sample = min(onset_sample + window_samples, len(audio))
    return audio[onset_sample:end_sample]


def analyze_onset_spectral(
    audio: np.ndarray,
    onset_time: float,
    sr: int,
    stem_type: str,
    config: Dict,
    duration: Optional[float] = None
) -> Optional[Dict]:
    """
    Perform complete spectral analysis for a single onset.
    
    This function encapsulates the common pattern of:
    1. Extract audio segment
    2. Calculate spectral energies
    3. Calculate geomean
    4. Calculate sustain duration (if needed)
    
    Pure function (aside from config reading) - no side effects.
    
    Args:
        audio: Audio signal (mono)
        onset_time: Onset time in seconds
        sr: Sample rate
        stem_type: Type of stem ('kick', 'snare', etc.)
        config: Configuration dictionary
        duration: Optional event duration in seconds. If provided, stored
                 in result dict for downstream use in Phase 2 metadata.
    
    Returns:
        Dictionary with analysis results, or None if segment too short:
        {
            'onset_sample': int,
            'segment': np.ndarray,
            '<band>_energy': float,  # Domain-specific: body_energy, wire_energy, etc.
            'low_energy': float,
            'total_energy': float,
            'geomean': float,
            'geomean_bands': list[str],  # Band names used for geomean
            'sustain_ms': float (if calculated),
            'spectral_ratio': float,
            'duration_sec': float (if duration provided)
        }
        
        Band energy keys are domain-specific per stem type:
            snare: body_energy, wire_energy
            kick: fundamental_energy, body_energy, attack_energy
            toms: fundamental_energy, body_energy
            hihat: body_energy, sizzle_energy
            cymbals: body_energy, brilliance_energy
    """
    # Convert time to sample
    onset_sample = time_to_sample(onset_time, sr)
    
    # Extract segment
    peak_window_sec = config.get('audio', {}).get('peak_window_sec', 0.05)
    segment = extract_audio_segment(audio, onset_sample, peak_window_sec, sr)
    
    # Check minimum length
    min_segment_length = config.get('audio', {}).get('min_segment_length', 512)
    if len(segment) < min_segment_length:
        return None
    
    # Get spectral configuration
    try:
        spectral_config = get_spectral_config_for_stem(stem_type, config)
    except ValueError:
        return None
    
    # Calculate spectral energies (keys are domain-specific: body, wire, fundamental, etc.)
    energies = calculate_spectral_energies(segment, sr, spectral_config['freq_ranges'])
    
    # Extract geomean band energies in order
    geomean_bands = spectral_config['geomean_bands']
    band_energies = [energies.get(band, 0.0) for band in geomean_bands]
    low_energy = energies.get('low', 0.0)
    
    # Calculate geomean (2-way or 3-way depending on stem)
    if len(band_energies) >= 3:
        geomean = calculate_geomean(band_energies[0], band_energies[1], band_energies[2])
    else:
        geomean = calculate_geomean(band_energies[0], band_energies[1])
    
    # Calculate total energy across geomean bands
    total_energy = sum(band_energies)
    
    # Calculate spectral ratio if low energy available
    spectral_ratio = (total_energy / low_energy) if low_energy > 0 else 100.0
    
    # Calculate sustain duration if needed
    sustain_ms = None
    if stem_type in ['hihat', 'cymbals']:
        # Get stem-specific sustain analysis window, fallback to global
        stem_config = config.get(stem_type, {})
        sustain_analysis_window_sec = stem_config.get('sustain_analysis_window_sec')
        if sustain_analysis_window_sec is None:
            sustain_analysis_window_sec = config.get('audio', {}).get('sustain_window_sec', 0.2)
        
        envelope_threshold = config.get('audio', {}).get('envelope_threshold', 0.1)
        smooth_kernel = config.get('audio', {}).get('envelope_smooth_kernel', 51)
        
        sustain_ms = calculate_sustain_duration(
            audio, onset_sample, sr,
            window_ms=sustain_analysis_window_sec * 1000,
            envelope_threshold=envelope_threshold,
            smooth_kernel=smooth_kernel
        )
    
    # Build result with domain-specific energy keys
    result = {
        'onset_sample': onset_sample,
        'segment': segment,
        'low_energy': low_energy,
        'total_energy': total_energy,
        'geomean': geomean,
        'geomean_bands': geomean_bands,
        'sustain_ms': sustain_ms,
        'spectral_ratio': spectral_ratio
    }
    
    # Add each band energy with its domain-specific name
    for band_name in geomean_bands:
        result[f'{band_name}_energy'] = energies.get(band_name, 0.0)
    
    # Add duration if provided (for Phase 2 metadata enrichment)
    if duration is not None:
        result['duration_sec'] = duration
    
    return result


def classify_cymbal_by_pan(
    pan_position: float,
    detected_pitch: float = 0.0,
    spectral_features: Optional[Dict] = None
) -> int:
    """
    DEPRECATED: This function uses hard-coded pan thresholds which don't adapt
    to different recording characteristics. Will be replaced by clustering-based
    classification in Phase 6 of threshold optimization.
    See: agent-plans/clustering-threshold-optimization.plan.md
    
    Classify cymbal type using pan position and optionally spectral features.
    
    Uses spatial information (pan) as primary classifier with spectral
    features as secondary cues. This is more reliable than pitch alone.
    
    Typical cymbal panning in recorded drums:
    - Crash: Often left (-0.5 to -1.0) or center (-0.2 to 0.2)
    - Ride: Often right (0.3 to 1.0) or center
    - Chinese: Variable, may use spectral features
    
    Pure function - deterministic, no side effects.
    
    Args:
        pan_position: Pan value from -1.0 (left) to +1.0 (right)
        detected_pitch: Optional detected pitch in Hz (may be unreliable)
        spectral_features: Optional dict with spectral analysis results
    
    Returns:
        Classification index:
            0 = Crash cymbal
            1 = Ride cymbal  
            2 = Chinese cymbal
    
    Examples:
        >>> classify_cymbal_by_pan(-0.8)  # Left-panned
        0  # Crash
        >>> classify_cymbal_by_pan(0.7)   # Right-panned
        1  # Ride
        >>> classify_cymbal_by_pan(0.05)  # Centered
        1  # Default to ride for center
    """
    # Thresholds for pan-based classification
    LEFT_THRESHOLD = -0.25   # More negative = left
    RIGHT_THRESHOLD = 0.25   # More positive = right
    
    # Classify primarily by pan position
    if pan_position < LEFT_THRESHOLD:
        # Strongly left-panned → Crash
        return 0  # Crash
    elif pan_position > RIGHT_THRESHOLD:
        # Strongly right-panned → Ride
        return 1  # Ride
    else:
        # Center or weak pan → Use secondary cues if available
        
        # Try spectral features if provided
        if spectral_features:
            # Chinese cymbals often have different spectral characteristics
            # (more trashy, less harmonic content)
            # This is placeholder logic - would need training data to refine
            brilliance = spectral_features.get('brilliance_energy', 0)
            body = spectral_features.get('body_energy', 0)
            
            if brilliance > 0 and body > 0:
                ratio = brilliance / body
                # Chinese cymbals tend to have more high-frequency content
                if ratio > 2.0:
                    return 2  # Chinese
        
        # Try pitch if provided and valid
        if detected_pitch > 0:
            # Ride cymbals typically have a more defined pitch (200-400 Hz fundamental)
            # Crashes are more noisy/less pitched
            # This is rough heuristic - pitch detection on cymbals is unreliable
            if 200 <= detected_pitch <= 500:
                return 1  # Ride (more defined pitch)
            else:
                return 0  # Crash (less pitched/noisier)
        
        # Default: center-panned with no other info → Ride
        # (Rides are often centered in mixes)
        return 1  # Ride


# ============================================================================
# FEATURE EXTRACTION FOR CLUSTERING
# ============================================================================

def extract_onset_features(
    audio: np.ndarray,
    sr: int,
    onset_times: List[float],
    pan_confidence: List[float],
    window_ms: float = 50.0,
    pitch_method: str = 'yin',
    min_pitch_hz: float = 60.0,
    max_pitch_hz: float = 1000.0,
    # Spectral band configuration for geomean calculation
    body_freq_range: tuple = (1000, 4000),  # Body range for cymbals
    brilliance_freq_range: tuple = (4000, 10000),  # Brilliance range for cymbals
    calculate_sustain: bool = True,
    sustain_window_ms: float = 200.0,
) -> List['OnsetFeatures']:  # Forward reference for type hint
    """
    Extract feature vectors for each onset for clustering.
    
    Computes spectral, pitch, and temporal features for each onset
    to enable clustering-based instrument identification.
    
    Pure function - no side effects.
    
    Args:
        audio: Mono audio signal
        sr: Sample rate in Hz
        onset_times: List of onset times in seconds
        pan_confidence: List of pan positions for each onset (-1 to +1)
        window_ms: Analysis window size in milliseconds
        pitch_method: Pitch detection method ('yin' or 'pyin')
        min_pitch_hz: Minimum pitch for detection
        max_pitch_hz: Maximum pitch for detection
        body_freq_range: Frequency range for body energy band (Hz tuple)
        brilliance_freq_range: Frequency range for brilliance energy band (Hz tuple)
        calculate_sustain: Whether to calculate sustain duration
        sustain_window_ms: Window size for sustain analysis (milliseconds)
    
    Returns:
        List of OnsetFeatures dicts, one per onset
    
    Examples:
        >>> audio = np.random.randn(44100)
        >>> features = extract_onset_features(
        ...     audio, 44100, [0.5, 1.0], [-0.8, 0.7]
        ... )
        >>> len(features)
        2
        >>> features[0]['pan_confidence']
        -0.8
    """
    import librosa  # type: ignore
    
    # Import the TypedDict - handle both package and standalone execution
    try:
        from midi_types import OnsetFeatures
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from midi_types import OnsetFeatures
    
    features = []
    window_samples = int(window_ms * sr / 1000.0)
    
    for i, (onset_time, pan) in enumerate(zip(onset_times, pan_confidence)):
        onset_sample = int(onset_time * sr)
        
        # Extract window around onset for analysis
        start = max(0, onset_sample)
        end = min(len(audio), onset_sample + window_samples)
        
        if end <= start:
            # Invalid window, skip or use defaults
            features.append(OnsetFeatures(
                time=onset_time,
                pan_confidence=pan,
                spectral_centroid=0.0,
                spectral_rolloff=0.0,
                spectral_flatness=0.0,
                pitch=None,
                timing_delta=None if i == 0 else onset_time - onset_times[i-1],
                body_energy=0.0,
                brilliance_energy=0.0,
                geomean=0.0,
                total_energy=0.0,
                sustain_ms=None
            ))
            continue
        
        window = audio[start:end]
        
        # Spectral features
        if len(window) > 0:
            # Spectral centroid: brightness
            centroid = librosa.feature.spectral_centroid(
                y=window, sr=sr, n_fft=min(2048, len(window))
            )[0]
            spectral_centroid = float(np.mean(centroid)) if len(centroid) > 0 else 0.0
            
            # Spectral rolloff: frequency below which 85% of energy lies
            rolloff = librosa.feature.spectral_rolloff(
                y=window, sr=sr, n_fft=min(2048, len(window)), roll_percent=0.85
            )[0]
            spectral_rolloff = float(np.mean(rolloff)) if len(rolloff) > 0 else 0.0
            
            # Spectral flatness: measure of noise-likeness
            flatness = librosa.feature.spectral_flatness(
                y=window, n_fft=min(2048, len(window))
            )[0]
            spectral_flatness = float(np.mean(flatness)) if len(flatness) > 0 else 0.0
        else:
            spectral_centroid = 0.0
            spectral_rolloff = 0.0
            spectral_flatness = 0.0
        
        # Spectral band energies (for geomean calculation)
        if len(window) > 0:
            # Calculate energy in specific frequency bands
            freq_ranges = {
                'body': body_freq_range,
                'brilliance': brilliance_freq_range
            }
            energies = calculate_spectral_energies(window, sr, freq_ranges)
            body_energy = energies.get('body', 0.0)
            brilliance_energy = energies.get('brilliance', 0.0)
            
            # Calculate geomean and total energy
            geomean = calculate_geomean(body_energy, brilliance_energy)
            total_energy = body_energy + brilliance_energy
        else:
            body_energy = 0.0
            brilliance_energy = 0.0
            geomean = 0.0
            total_energy = 0.0
        
        # Sustain duration (optional)
        sustain_ms = None
        if calculate_sustain and len(audio) > onset_sample:
            sustain_window_samples = int(sustain_window_ms * sr / 1000.0)
            try:
                sustain_ms = calculate_sustain_duration(
                    audio,
                    onset_sample,
                    sr,
                    window_ms=sustain_window_ms,
                    envelope_threshold=0.1,
                    smooth_kernel=51
                )
            except Exception:
                # Sustain calculation can fail at edge cases
                sustain_ms = None
        
        # Pitch detection (optional, may be None)
        pitch_hz = None
        if len(window) >= 2048:  # Need minimum samples for pitch detection
            try:
                if pitch_method == 'pyin':
                    f0_data, voiced_flag, voiced_prob = librosa.pyin(
                        window,
                        fmin=min_pitch_hz,
                        fmax=max_pitch_hz,
                        sr=sr,
                        frame_length=2048
                    )
                    # Take median of voiced frames
                    voiced_f0 = f0_data[voiced_flag]
                    if len(voiced_f0) > 0:
                        pitch_hz = float(np.median(voiced_f0))
                else:  # 'yin'
                    f0 = librosa.yin(
                        window,
                        fmin=min_pitch_hz,
                        fmax=max_pitch_hz,
                        sr=sr,
                        frame_length=2048
                    )
                    # Take median, filter out zeros
                    valid_f0 = f0[f0 > 0]
                    if len(valid_f0) > 0:
                        pitch_hz = float(np.median(valid_f0))
            except Exception:
                # Pitch detection can fail on noisy signals
                pitch_hz = None
        
        # Timing delta: time since previous onset
        timing_delta = None if i == 0 else onset_time - onset_times[i-1]
        
        features.append(OnsetFeatures(
            time=onset_time,
            pan_confidence=pan,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_flatness=spectral_flatness,
            pitch=pitch_hz,
            timing_delta=timing_delta,
            body_energy=body_energy,
            brilliance_energy=brilliance_energy,
            geomean=geomean,
            total_energy=total_energy,
            sustain_ms=sustain_ms
        ))
    
    return features
