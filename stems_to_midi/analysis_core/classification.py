"""
Classification Functions

Pure helper functions for instrument classification based on pitch and features.

Functions:
- classify_tom_pitch: Classify tom pitches into low/mid/high groups
- classify_cymbal_pitch: Classify cymbal pitches into crash/ride/chinese groups
- classify_snare_pitch: Classify snare hits into snare/rimshot/clap groups
- classify_cymbal_by_pan: Classify cymbal type using pan position
- extract_onset_features: Extract feature vectors for clustering
"""

import numpy as np
from typing import Any, Dict, List, Optional

from .audio_utils import (
    calculate_spectral_energies,
    calculate_spectral_centroid,
    calculate_spectral_flux,
    calculate_sustain_duration,
    detect_pitch,
)


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
    Classify snare hits into 3 types using clustering.
    
    Pure function - no side effects.
    
    Types:
    0 = snare (most common, mid-range pitch)
    1 = rimshot (higher pitch, sharper)
    2 = clap (highest pitch, thin sound)
    
    Note: Later can be enhanced with stereo info and envelope profile.
    
    Args:
        pitches: Array of detected pitches in Hz
    
    Returns:
        Array of classifications: 0=snare, 1=rimshot, 2=clap
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
        # 4+ unique pitches - use k-means clustering with k=3
        try:
            from sklearn.cluster import KMeans
            
            # Reshape for sklearn
            X = valid_pitches.reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Sort clusters by center frequency (0=low, 1=mid, 2=high)
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_cluster_indices = np.argsort(cluster_centers)
            
            # Map to snare types based on pitch order
            # Lowest pitch = snare (0)
            # Middle = rimshot (1) 
            # Highest = clap (2)
            pitch_to_type = {sorted_cluster_indices[0]: 0,  # Lowest -> snare
                           sorted_cluster_indices[1]: 1,  # Mid -> rimshot
                           sorted_cluster_indices[2]: 2}   # Highest -> clap
            
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
            # Fallback: use percentiles to split into 3 groups
            p33 = np.percentile(valid_pitches, 33)
            p66 = np.percentile(valid_pitches, 66)
            
            classifications = np.zeros(len(pitches), dtype=int)  # Default to snare
            for i, pitch in enumerate(pitches):
                if pitch > 0:
                    if pitch < p33:
                        classifications[i] = 0  # Snare (lowest)
                    elif pitch < p66:
                        classifications[i] = 1  # Rimshot (mid)
                    else:
                        classifications[i] = 2  # Clap (highest)
            return classifications


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


# Type alias for onset features
# Type alias for onset features
OnsetFeatures = Dict[str, Any]


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
) -> List[OnsetFeatures]:
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
    """
    import librosa
    
    features = []
    window_samples = int(window_ms * sr / 1000.0)
    
    for i, (onset_time, pan) in enumerate(zip(onset_times, pan_confidence)):
        onset_sample = int(onset_time * sr)
        
        # Extract window around onset for analysis
        start = max(0, onset_sample)
        end = min(len(audio), onset_sample + window_samples)
        
        if end <= start:
            # Invalid window, skip or use defaults
            features.append(dict(
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
            sustain_window_samples = int(sustain_window_ms * sr / 1000)
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
        
        features.append(dict(
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


def calculate_geomean(
    primary_energy: float,
    secondary_energy: float,
    tertiary_energy: Optional[float] = None
) -> float:
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
        return float(np.cbrt(primary_energy * secondary_energy * tertiary_energy))
    else:
        return float(np.sqrt(primary_energy * secondary_energy))
