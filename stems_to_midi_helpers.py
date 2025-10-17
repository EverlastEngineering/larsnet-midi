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
        - filtered_sustains: List (for hihat only)
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
        if sustain_duration is not None:
            onset_data['sustain_ms'] = sustain_duration
        
        all_onset_data.append(onset_data)
        
        # Determine if this onset should be kept
        is_real_hit = should_keep_onset(
            geomean=body_wire_geomean,
            sustain_ms=sustain_duration,
            geomean_threshold=geomean_threshold,
            min_sustain_ms=min_sustain_ms,
            stem_type=stem_type
        )
        
        # In learning mode, keep ALL detections
        if learning_mode or is_real_hit:
            filtered_times.append(onset_time)
            filtered_strengths.append(strength)
            filtered_amplitudes.append(peak_amplitude)
            filtered_geomeans.append(body_wire_geomean)
            # Store sustain duration and spectral data for hihat classification
            if stem_type == 'hihat' and sustain_duration is not None:
                filtered_sustains.append(sustain_duration)
                filtered_spectral.append({
                    'primary_energy': primary_energy,
                    'secondary_energy': secondary_energy
                })
    
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
    low_energy = energies.get('low', 0.0)
    
    # Calculate geomean
    geomean = calculate_geomean(primary_energy, secondary_energy)
    
    # Calculate total energy
    total_energy = primary_energy + secondary_energy
    
    # Calculate spectral ratio if low energy available
    spectral_ratio = (total_energy / low_energy) if low_energy > 0 else 100.0
    
    # Calculate sustain duration if needed
    sustain_ms = None
    if stem_type in ['hihat', 'cymbals']:
        sustain_window_sec = config.get('audio', {}).get('sustain_window_sec', 0.2)
        envelope_threshold = config.get('audio', {}).get('envelope_threshold', 0.1)
        smooth_kernel = config.get('audio', {}).get('envelope_smooth_kernel', 51)
        
        sustain_ms = calculate_sustain_duration(
            audio, onset_sample, sr,
            window_ms=sustain_window_sec * 1000,
            envelope_threshold=envelope_threshold,
            smooth_kernel=smooth_kernel
        )
    
    return {
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
