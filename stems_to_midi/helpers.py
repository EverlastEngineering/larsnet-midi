"""
Pure helper functions for stem to MIDI conversion.

These are functional core functions - pure, deterministic, no I/O or side effects.
All audio processing logic extracted here for testability.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
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
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold')
        }
    
    elif stem_type == 'kick':
        return {
            'freq_ranges': {
                'primary': (stem_config['fundamental_freq_min'], stem_config['fundamental_freq_max']),
                'secondary': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'tertiary': (stem_config['attack_freq_min'], stem_config['attack_freq_max'])
            },
            'energy_labels': {
                'primary': 'FundE',
                'secondary': 'BodyE',
                'tertiary': 'AttackE'
            },
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold')
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
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold')
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
            'min_sustain_ms': stem_config.get('min_sustain_ms', 25),
            'min_strength_threshold': stem_config.get('min_strength_threshold')
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
    by comparing FundE/BodyE ratio and total energy against dataset medians.
    
    Pure function - no side effects.
    
    Args:
        onset_data_list: List of onset data dicts with 'primary_energy', 
                         'secondary_energy', 'total_energy' keys
    
    Returns:
        Dict with median and spread values:
        - median_ratio: Median FundE/BodyE ratio across all events
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
    
    # Extract fundamental and body energies
    primary_energies = np.array([d['primary_energy'] for d in onset_data_list])
    secondary_energies = np.array([d['secondary_energy'] for d in onset_data_list])
    total_energies = np.array([d['total_energy'] for d in onset_data_list])
    
    # Calculate FundE/BodyE ratios (with safety for division by zero)
    ratios = primary_energies / np.maximum(secondary_energies, 1e-9)
    
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
    Snare bleed typically has lower FundE/BodyE ratio and different total energy.
    
    Pure function - no side effects.
    
    Args:
        onset_data: Dict with 'primary_energy', 'secondary_energy', 'total_energy'
        statistical_params: Dict from calculate_statistical_params()
        ratio_weight: Weight for ratio deviation (0-1)
        total_weight: Weight for total energy deviation (0-1)
    
    Returns:
        Badness score in range [0, 1]:
        - 0.0 = perfectly typical kick
        - 1.0 = maximum deviation (likely artifact/bleed)
    """
    # Calculate this onset's ratio
    ratio = onset_data['primary_energy'] / max(onset_data['secondary_energy'], 1e-9)
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
        
        # Reshape for sklearn
        valid_pitches_2d = valid_pitches.reshape(-1, 1)
        
        # Cluster into 3 groups
        n_clusters = min(3, len(unique_pitches))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(valid_pitches_2d)
        
        # Sort cluster centers to get low/mid/high order
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(cluster_centers)
        
        # Create mapping from cluster label to tom classification
        cluster_to_tom = {sorted_indices[i]: i for i in range(n_clusters)}
        
        # If only 2 clusters, use 0 (low) and 2 (high), skip mid
        if n_clusters == 2:
            cluster_to_tom = {sorted_indices[0]: 0, sorted_indices[1]: 2}
        
        # Classify all pitches
        classifications = np.ones(len(pitches), dtype=int)  # Default to mid
        
        for i, pitch in enumerate(pitches):
            if pitch > 0:
                # Find which cluster this pitch belongs to
                cluster = kmeans.predict([[pitch]])[0]
                classifications[i] = cluster_to_tom[cluster]
        
        return classifications


# ============================================================================
# ONSET FILTERING AND ANALYSIS (Pure Functions)
# ============================================================================

def filter_onsets_by_spectral(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    peak_amplitudes: np.ndarray,
    audio: np.ndarray,
    sr: int,
    stem_type: str,
    config: Dict,
    learning_mode: bool = False
) -> Dict:
    """
    Filter onsets by spectral content and analyze each onset.
    
    Pure function - no side effects, no I/O.
    
    Args:
        onset_times: Array of onset times in seconds
        onset_strengths: Array of onset strengths (0-1)
        peak_amplitudes: Array of peak amplitudes
        audio: Audio signal (mono)
        sr: Sample rate
        stem_type: Type of stem ('kick', 'snare', etc.)
        config: Configuration dictionary
        learning_mode: If True, keep all onsets (don't filter)
    
    Returns:
        Dictionary with:
        - filtered_times: np.ndarray
        - filtered_strengths: np.ndarray
        - filtered_amplitudes: np.ndarray
        - filtered_geomeans: np.ndarray
        - filtered_sustains: List (for hihat and cymbals)
        - filtered_spectral: List (for hihat only)
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
            'all_onset_data': [],
            'spectral_config': None
        }
    
    # Get spectral configuration for this stem type
    spectral_config = get_spectral_config_for_stem(stem_type, config)
    geomean_threshold = spectral_config['geomean_threshold']
    min_sustain_ms = spectral_config['min_sustain_ms']
    energy_labels = spectral_config['energy_labels']
    
    # Storage for filtered results
    filtered_times = []
    filtered_strengths = []
    filtered_amplitudes = []
    filtered_geomeans = []
    filtered_sustains = []  # For hihat open/closed detection
    filtered_spectral = []  # For hihat handclap detection
    
    # Store raw spectral data for ALL onsets (for debug output)
    all_onset_data = []
    
    for onset_time, strength, peak_amplitude in zip(onset_times, onset_strengths, peak_amplitudes):
        # Use unified spectral analysis helper
        analysis = analyze_onset_spectral(audio, onset_time, sr, stem_type, config)
        
        if analysis is None:
            # Segment too short, skip
            continue
        
        # Extract results from analysis
        primary_energy = analysis['primary_energy']
        secondary_energy = analysis['secondary_energy']
        tertiary_energy = analysis.get('tertiary_energy')  # Only for kick
        low_energy = analysis['low_energy']
        total_energy = analysis['total_energy']
        body_wire_geomean = analysis['geomean']
        sustain_duration = analysis['sustain_ms']
        spectral_ratio = analysis['spectral_ratio']
        
        # Store all data for this onset (for debug output)
        onset_data = {
            'time': onset_time,
            'strength': strength,
            'amplitude': peak_amplitude,
            'low_energy': low_energy,
            'primary_energy': primary_energy,
            'secondary_energy': secondary_energy,
            'ratio': spectral_ratio,
            'total_energy': total_energy,
            'body_wire_geomean': body_wire_geomean,
            'energy_label_1': energy_labels['primary'],
            'energy_label_2': energy_labels['secondary']
        }
        
        # Add tertiary energy if present (kick attack range)
        if tertiary_energy is not None:
            onset_data['tertiary_energy'] = tertiary_energy
            onset_data['energy_label_3'] = energy_labels.get('tertiary', 'Tertiary')
        
        if sustain_duration is not None:
            onset_data['sustain_ms'] = sustain_duration
        
        all_onset_data.append(onset_data)
        
        # Determine if this onset should be kept
        is_real_hit = should_keep_onset(
            geomean=body_wire_geomean,
            sustain_ms=sustain_duration,
            geomean_threshold=geomean_threshold,
            min_sustain_ms=min_sustain_ms,
            stem_type=stem_type,
            strength=strength,
            min_strength_threshold=spectral_config.get('min_strength_threshold')
        )
        
        # In learning mode, keep ALL detections
        if learning_mode or is_real_hit:
            filtered_times.append(onset_time)
            filtered_strengths.append(strength)
            filtered_amplitudes.append(peak_amplitude)
            filtered_geomeans.append(body_wire_geomean)
            # Store sustain duration and spectral data for hihat/cymbal classification
            if stem_type in ['hihat', 'cymbals'] and sustain_duration is not None:
                filtered_sustains.append(sustain_duration)
                if stem_type == 'hihat':
                    filtered_spectral.append({
                        'primary_energy': primary_energy,
                        'secondary_energy': secondary_energy
                    })
    
    # SECOND PASS: Remove cymbal retriggering using decay pattern analysis
    # Cymbals can have energy modulation during sustain that looks like new onsets
    # Analyze spectral decay pattern to distinguish true hits from decay artifacts
    if stem_type == 'cymbals' and not learning_mode and len(filtered_times) > 1:
        # Get sustain window for cymbals
        cymbal_config = config.get('cymbals', {})
        sustain_window_sec = cymbal_config.get('sustain_window_sec', 2.0)
        
        # Build list of times to keep
        final_times = []
        final_strengths = []
        final_amplitudes = []
        final_geomeans = []
        final_sustains = []
        
        # Track active decay zones (onset_time -> decay pattern)
        active_decays = {}
        
        for i in range(len(filtered_times)):
            current_time = filtered_times[i]
            current_sample = int(current_time * sr)
            
            # Check if this onset falls within any active decay zone
            is_retrigger = False
            for prev_time, decay_info in active_decays.items():
                time_diff = current_time - prev_time
                
                # If within decay window
                if 0 < time_diff < sustain_window_sec:
                    # Check if we're in a decaying region
                    if decay_info['is_decaying']:
                        is_retrigger = True
                        break
            
            if not is_retrigger:
                # This is a legitimate hit - keep it
                final_times.append(filtered_times[i])
                final_strengths.append(filtered_strengths[i])
                final_amplitudes.append(filtered_amplitudes[i])
                final_geomeans.append(filtered_geomeans[i])
                if i < len(filtered_sustains):
                    final_sustains.append(filtered_sustains[i])
                
                # Analyze decay pattern starting from this onset
                decay_pattern = analyze_cymbal_decay_pattern(
                    audio, current_sample, sr, 
                    window_sec=sustain_window_sec,
                    num_windows=8
                )
                
                # Store for checking subsequent onsets
                active_decays[current_time] = decay_pattern
                
                # Clean up old decays outside the window
                active_decays = {
                    t: info for t, info in active_decays.items()
                    if current_time - t < sustain_window_sec
                }
        
        # Update filtered arrays
        filtered_times = final_times
        filtered_strengths = final_strengths
        filtered_amplitudes = final_amplitudes
        filtered_geomeans = final_geomeans
        filtered_sustains = final_sustains
    
    # THIRD PASS: Statistical outlier detection (kick only, if enabled)
    # This catches snare bleed that passes geomean threshold but has abnormal FundE/BodyE ratio
    
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
            
            for time, strength, amplitude, geomean in zip(
                filtered_times, filtered_strengths, filtered_amplitudes, filtered_geomeans
            ):
                onset_data = onset_data_by_time.get(time)
                if onset_data and onset_data.get('badness_score', 0) <= badness_threshold:
                    final_times.append(time)
                    final_strengths.append(strength)
                    final_amplitudes.append(amplitude)
                    final_geomeans.append(geomean)
            
            # Update filtered arrays with statistical filter results
            filtered_times = final_times
            filtered_strengths = final_strengths
            filtered_amplitudes = final_amplitudes
            filtered_geomeans = final_geomeans
            
            # Store statistical info in config for debug output
            spectral_config['statistical_params'] = statistical_params
            spectral_config['statistical_enabled'] = True
            spectral_config['badness_threshold'] = badness_threshold
    
    return {
        'filtered_times': np.array(filtered_times),
        'filtered_strengths': np.array(filtered_strengths),
        'filtered_amplitudes': np.array(filtered_amplitudes),
        'filtered_geomeans': np.array(filtered_geomeans),
        'filtered_sustains': filtered_sustains,
        'filtered_spectral': filtered_spectral,
        'all_onset_data': all_onset_data,
        'spectral_config': spectral_config
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
    prepared_events = []
    
    for stem_type, events in events_by_stem.items():
        for event in events:
            prepared_event = {
                'note': event['note'],
                'velocity': event['velocity'],
                'time_beats': seconds_to_beats(event['time'], tempo),
                'duration_beats': seconds_to_beats(event['duration'], tempo),
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
    config: Dict
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
    
    Returns:
        Dictionary with analysis results, or None if segment too short:
        {
            'onset_sample': int,
            'segment': np.ndarray,
            'primary_energy': float,
            'secondary_energy': float,
            'low_energy': float (if available),
            'total_energy': float,
            'geomean': float,
            'sustain_ms': float (if calculated),
            'spectral_ratio': float (if low_energy available)
        }
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
    
    # Calculate spectral energies
    energies = calculate_spectral_energies(segment, sr, spectral_config['freq_ranges'])
    primary_energy = energies.get('primary', 0.0)
    secondary_energy = energies.get('secondary', 0.0)
    tertiary_energy = energies.get('tertiary', None)  # Only for kick (attack range)
    low_energy = energies.get('low', 0.0)
    
    # Calculate geomean (2-way for most drums, 3-way for kick with attack)
    geomean = calculate_geomean(primary_energy, secondary_energy, tertiary_energy)
    
    # Calculate total energy (include tertiary if present)
    if tertiary_energy is not None:
        total_energy = primary_energy + secondary_energy + tertiary_energy
    else:
        total_energy = primary_energy + secondary_energy
    
    # Calculate spectral ratio if low energy available
    spectral_ratio = (total_energy / low_energy) if low_energy > 0 else 100.0
    
    # Calculate sustain duration if needed
    sustain_ms = None
    if stem_type in ['hihat', 'cymbals']:
        # Get stem-specific sustain window, fallback to global
        stem_config = config.get(stem_type, {})
        sustain_window_sec = stem_config.get('sustain_window_sec')
        if sustain_window_sec is None:
            sustain_window_sec = config.get('audio', {}).get('sustain_window_sec', 0.2)
        
        envelope_threshold = config.get('audio', {}).get('envelope_threshold', 0.1)
        smooth_kernel = config.get('audio', {}).get('envelope_smooth_kernel', 51)
        
        sustain_ms = calculate_sustain_duration(
            audio, onset_sample, sr,
            window_ms=sustain_window_sec * 1000,
            envelope_threshold=envelope_threshold,
            smooth_kernel=smooth_kernel
        )
    
    result = {
        'onset_sample': onset_sample,
        'segment': segment,
        'primary_energy': primary_energy,
        'secondary_energy': secondary_energy,
        'low_energy': low_energy,
        'total_energy': total_energy,
        'geomean': geomean,
        'sustain_ms': sustain_ms,
        'spectral_ratio': spectral_ratio
    }
    
    # Add tertiary energy if present (kick attack range)
    if tertiary_energy is not None:
        result['tertiary_energy'] = tertiary_energy
    
    return result
