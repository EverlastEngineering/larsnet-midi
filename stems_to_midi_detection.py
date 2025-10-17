"""
Audio analysis and detection algorithms for stems-to-MIDI conversion.

This module provides functions for detecting drum hits, analyzing pitch,
classifying drum types, and estimating velocity from audio signals.

Architecture: Mix of Functional Core and Imperative Shell
- Onset detection, pitch detection: coordinated algorithms
- Velocity estimation: pure function
- Uses helpers from stems_to_midi_helpers for pure logic
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import librosa
from sklearn.cluster import KMeans
from scipy.signal import medfilt

# Import functional core helpers
from stems_to_midi_helpers import (
    ensure_mono,
    calculate_sustain_duration
)

# Import config
from stems_to_midi_config import DrumMapping


__all__ = [
    'detect_onsets',
    'detect_tom_pitch',
    'classify_tom_pitch',
    'detect_hihat_state',
    'estimate_velocity'
]


def detect_tom_pitch(
    audio: np.ndarray,
    sr: int,
    onset_time: float,
    method: str = 'yin',
    min_hz: float = 60.0,
    max_hz: float = 250.0,
    window_ms: float = 100.0
) -> float:
    """
    Detect the pitch of a tom hit using YIN or pYIN algorithm.
    
    Args:
        audio: Audio signal (mono)
        sr: Sample rate
        onset_time: Time of onset in seconds
        method: 'yin' or 'pyin' (pYIN is more robust but slower)
        min_hz: Minimum expected pitch
        max_hz: Maximum expected pitch
        window_ms: Analysis window in milliseconds
    
    Returns:
        Detected pitch in Hz, or 0 if detection failed
    """
    onset_sample = int(onset_time * sr)
    window_samples = int(window_ms * sr / 1000.0)
    end_sample = min(onset_sample + window_samples, len(audio))
    
    segment = audio[onset_sample:end_sample]
    
    if len(segment) < 512:
        return 0.0
    
    try:
        if method == 'pyin':
            # More robust probabilistic YIN
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=min_hz,
                fmax=max_hz,
                sr=sr,
                frame_length=2048
            )
            # Take median of confident detections
            confident_pitches = f0[(voiced_flag) & (voiced_probs > 0.5)]
            if len(confident_pitches) > 0:
                pitch = np.median(confident_pitches[~np.isnan(confident_pitches)])
                return float(pitch) if not np.isnan(pitch) else 0.0
            else:
                return 0.0
        else:
            # Standard YIN (faster)
            f0 = librosa.yin(
                segment,
                fmin=min_hz,
                fmax=max_hz,
                sr=sr,
                frame_length=2048
            )
            # Take median to smooth out jitter
            pitch = np.median(f0[~np.isnan(f0)])
            return float(pitch) if not np.isnan(pitch) else 0.0
    except Exception as e:
        # Fallback: use spectral peak as estimate
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        magnitude = np.abs(fft)
        
        # Find peak in expected range
        mask = (freqs >= min_hz) & (freqs <= max_hz)
        if np.any(mask):
            peak_idx = np.argmax(magnitude[mask])
            peak_freq = freqs[mask][peak_idx]
            return float(peak_freq)
        else:
            return 0.0


def classify_tom_pitch(pitches: np.ndarray) -> np.ndarray:
    """
    Classify tom pitches into low/mid/high groups using clustering.
    
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
            print("Warning: scikit-learn not installed. Using simple pitch-based classification.")
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


def detect_onsets(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    threshold: float = 0.3,
    pre_max: int = 3,
    post_max: int = 3,
    pre_avg: int = 3,
    post_avg: int = 3,
    delta: float = 0.01,
    wait: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect onsets (drum hits) in audio.
    
    Args:
        audio: Audio signal (mono or stereo)
        sr: Sample rate
        hop_length: Number of samples between successive frames
        threshold: Onset strength threshold (0-1)
        pre_max: Number of frames before peak for comparison
        post_max: Number of frames after peak for comparison
        pre_avg: Number of frames before for moving average
        post_avg: Number of frames after for moving average
        delta: Threshold for peak picking
        wait: Minimum number of frames between peaks
    
    Returns:
        Tuple of (onset_times in seconds, onset_strengths)
    """
    # Convert to mono if stereo (use helper function)
    audio = ensure_mono(audio)
    
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median
    )
    
    # Detect onset frames WITHOUT backtracking first to get strengths
    onset_frames_peak = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,  # Don't backtrack yet - we need peak positions for strengths
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait
    )
    
    # Get onset strengths at the PEAK positions (before backtracking)
    onset_strengths = onset_env[onset_frames_peak]
    
    # Now detect WITH backtracking for accurate timing
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,  # Now backtrack for accurate onset times
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait
    )
    
    # Convert frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    # Filter out zero-strength detections first (these are false positives from librosa)
    if len(onset_strengths) > 0:
        nonzero_mask = onset_strengths > 0
        onset_times = onset_times[nonzero_mask]
        onset_strengths = onset_strengths[nonzero_mask]
    
    # Normalize strengths using percentile-based approach (more robust)
    if len(onset_strengths) > 0:
        # Use 95th percentile instead of max to avoid one loud hit skewing everything
        percentile_95 = np.percentile(onset_strengths, 95)
        
        if percentile_95 > 0:
            onset_strengths = onset_strengths / percentile_95
            # Clip values above 1.0 (those louder than 95th percentile)
            onset_strengths = np.clip(onset_strengths, 0, 1)
        else:
            # Fallback to max if percentile is zero
            max_strength = np.max(onset_strengths)
            if max_strength > 0:
                onset_strengths = onset_strengths / max_strength
            else:
                onset_strengths = np.ones_like(onset_strengths)
        
        # Apply threshold
        mask = onset_strengths >= threshold
        onset_times = onset_times[mask]
        onset_strengths = onset_strengths[mask]
    
    return onset_times, onset_strengths


def estimate_velocity(strength: float, min_vel: int = 40, max_vel: int = 127) -> int:
    """
    Convert onset strength to MIDI velocity.
    
    Args:
        strength: Onset strength (0-1)
        min_vel: Minimum MIDI velocity
        max_vel: Maximum MIDI velocity
    
    Returns:
        MIDI velocity (1-127)
    """
    velocity = int(min_vel + strength * (max_vel - min_vel))
    return np.clip(velocity, 1, 127)


def detect_hihat_state(
    audio: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    sustain_durations: List[float] = None,
    open_sustain_threshold_ms: float = 150.0,
    spectral_data: List[Dict] = None,
    config: Dict = None
) -> List[str]:
    """
    Classify hi-hat hits as open, closed, or handclap based on sustain duration and spectral content.
    
    - Handclaps: High energy (body>20, sizzle>100) + very short transient (<50ms)
    - Open hi-hats: Sustain >150ms
    - Closed hi-hats: Everything else
    
    Args:
        audio: Audio signal
        sr: Sample rate
        onset_times: Times of detected onsets
        sustain_durations: Pre-calculated sustain durations in ms (if available)
        open_sustain_threshold_ms: Threshold in ms (longer = open, shorter = closed)
        spectral_data: List of dicts with 'primary_energy' and 'secondary_energy'
        config: Configuration dict with handclap detection settings
    
    Returns:
        List of 'open', 'closed', or 'handclap' for each onset
    """
    # Convert to mono if stereo (use helper function)
    audio = ensure_mono(audio)
    
    states = []
    
    # Get handclap detection settings from config
    detect_handclap = False
    handclap_body_min = 20
    handclap_sizzle_min = 100
    handclap_sustain_max = 50
    
    if config is not None:
        hihat_config = config.get('hihat', {})
        detect_handclap = hihat_config.get('detect_handclap', False)
        handclap_body_min = hihat_config.get('handclap_body_min', 20)
        handclap_sizzle_min = hihat_config.get('handclap_sizzle_min', 100)
        handclap_sustain_max = hihat_config.get('handclap_sustain_max', 50)
    
    # If sustain durations and spectral data were already calculated, use them
    if (sustain_durations is not None and len(sustain_durations) == len(onset_times) and
        spectral_data is not None and len(spectral_data) == len(onset_times)):
        for i, sustain_ms in enumerate(sustain_durations):
            # Check for handclap first (if enabled)
            if detect_handclap:
                body_energy = spectral_data[i].get('primary_energy', 0)
                sizzle_energy = spectral_data[i].get('secondary_energy', 0)
                
                # Handclap: high energy + very short transient
                if (body_energy > handclap_body_min and 
                    sizzle_energy > handclap_sizzle_min and 
                    sustain_ms < handclap_sustain_max):
                    states.append('handclap')
                    continue
            
            # Otherwise classify as open or closed
            if sustain_ms > open_sustain_threshold_ms:
                states.append('open')
            else:
                states.append('closed')
        return states
    
    # Get audio processing parameters from config (for functional core)
    sustain_window_sec = config.get('audio', {}).get('sustain_window_sec', 0.2)
    envelope_threshold = config.get('audio', {}).get('envelope_threshold', 0.1)
    smooth_kernel = config.get('audio', {}).get('envelope_smooth_kernel', 51)
    
    # Otherwise, calculate sustain duration for each hit using functional core
    for onset_time in onset_times:
        onset_sample = int(onset_time * sr)
        
        # Use functional core helper for sustain calculation
        sustain_duration_ms = calculate_sustain_duration(
            audio, onset_sample, sr,
            window_ms=sustain_window_sec * 1000,
            envelope_threshold=envelope_threshold,
            smooth_kernel=smooth_kernel
        )
        
        # Classify based on sustain duration
        if sustain_duration_ms > open_sustain_threshold_ms:
            states.append('open')
        else:
            states.append('closed')
    
    return states
