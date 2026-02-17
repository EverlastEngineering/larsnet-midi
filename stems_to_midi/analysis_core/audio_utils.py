"""
Audio Analysis Utilities

Pure helper functions for audio signal processing. These are functional core
functions - pure, deterministic, no I/O or side effects.

Functions:
- ensure_mono: Convert stereo to mono
- calculate_peak_amplitude: Calculate peak amplitude in a window
- calculate_sustain_duration: Calculate sustain duration by envelope analysis
- calculate_event_durations: Calculate duration for each onset
- calculate_amplitude_at_time: Calculate RMS amplitude at specific time
- calculate_attack_sharpness: Calculate attack sharpness via envelope derivative
- calculate_envelope_continuity: Detect gaps/dropouts in envelope
- calculate_peak_prominence: Calculate peak prominence relative to surroundings
- calculate_spectral_centroid: Calculate spectral centroid (brightness)
- calculate_spectral_flux: Calculate spectral flux (timbre change rate)
- detect_pitch_autocorrelation: Detect pitch via autocorrelation
- detect_pitch: Detect pitch via YIN algorithm
- calculate_gap_from_previous: Calculate time gap since previous onset
- calculate_spectral_energies: Calculate energy in frequency ranges
- analyze_cymbal_decay_pattern: Analyze cymbal decay pattern
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from scipy.signal import medfilt


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


def detect_pitch_autocorrelation(
    audio: np.ndarray,
    onset_time: float,
    sr: int,
    window_ms: float = 50.0,
    fmin: float = 40.0,
    fmax: float = 500.0,
    peak_search_ms: float = 10.0
) -> Optional[float]:
    """
    Detect fundamental pitch using autocorrelation - optimal for short, decaying percussive sounds.
    
    Autocorrelation is well-suited for drum hits because:
    1. It works directly on the time-domain signal (no FFT needed)
    2. It detects periodicity even in short bursts
    3. It's robust against noise and transients
    
    This implementation also searches for the peak amplitude within a window after
    the onset time, since onset detection finds the START of the transient but
    the actual drum hit peak is slightly later.
    
    Pure function - no side effects.
    
    Args:
        audio: Full audio signal (mono)
        onset_time: Onset time in seconds
        sr: Sample rate
        window_ms: Analysis window in milliseconds  
        fmin: Minimum frequency to detect (Hz)
        fmax: Maximum frequency to detect (Hz)
        peak_search_ms: Window after onset to search for peak (ms)
        
    Returns:
        Detected pitch in Hz, or None if no pitch detected
    """
    import librosa
    
    onset_sample = int(onset_time * sr)
    
    # First, search for the peak within a window after onset
    peak_search_samples = int(peak_search_ms * sr / 1000)
    search_start = onset_sample
    search_end = min(len(audio), onset_sample + peak_search_samples)
    
    peak_amplitude = 0.0
    if search_end > search_start:
        search_segment = audio[search_start:search_end]
        if len(search_segment) > 0:
            peak_idx = np.argmax(np.abs(search_segment))
            peak_amplitude = np.abs(search_segment[peak_idx])
            # Use the peak position for pitch detection
            onset_sample = search_start + peak_idx
    
    window_samples = int(window_ms * sr / 1000)
    start = onset_sample
    end = min(len(audio), onset_sample + window_samples)
    segment = audio[start:end]
    
    # Need at least 2 periods of fmax to detect pitch
    min_samples = int(2 * sr / fmax)
    if len(segment) < min_samples:
        return None
    
    # Compute autocorrelation using librosa
    try:
        max_lag = int(sr / fmin)  # Maximum lag to check (lowest freq)
        min_lag = int(sr / fmax)   # Minimum lag to check (highest freq)
        
        if max_lag >= len(segment):
            max_lag = len(segment) - 1
        
        if min_lag < 1:
            min_lag = 1
            
        # Compute normalized autocorrelation
        ac = librosa.autocorrelate(segment, maxLag=max_lag)
        
        if len(ac) == 0:
            return None
            
        # Normalize by the zero-lag value
        ac = ac / ac[0] if ac[0] > 0 else ac
        
        # Find peaks in the autocorrelation in the valid frequency range
        # Use adaptive threshold based on signal amplitude
        if peak_amplitude > 0.3:
            min_corr = 0.25
        elif peak_amplitude > 0.1:
            min_corr = 0.15
        else:
            min_corr = 0.08  # Very quiet signals need lower threshold
        
        best_lag = None
        best_corr = min_corr
        
        for lag in range(min_lag, min(max_lag + 1, len(ac))):
            if lag > 0 and lag < len(ac) - 1:
                # Check if this is a local maximum
                if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1]:
                    if ac[lag] > best_corr:
                        best_corr = ac[lag]
                        best_lag = lag
        
        # If no clear peak found, try weighted approach
        if best_lag is None:
            total_weight = 0.0
            weighted_lag = 0.0
            for lag in range(min_lag, min(max_lag + 1, len(ac))):
                if ac[lag] > min_corr:
                    weighted_lag += lag * ac[lag]
                    total_weight += ac[lag]
            if total_weight > 0:
                best_lag = int(weighted_lag / total_weight)
                best_corr = min_corr
        
        if best_lag is not None and best_lag > 0:
            pitch = sr / best_lag
            # Verify pitch is in valid range
            if fmin <= pitch <= fmax:
                return float(pitch)
        
        return None
        
    except Exception:
        return None


def detect_pitch(
    audio: np.ndarray,
    onset_time: float,
    sr: int,
    window_ms: float = 50.0,
    fmin: float = 50.0,
    fmax: float = 2000.0
) -> Optional[float]:
    """
    Detect fundamental pitch using YIN algorithm - optimized for percussive sounds.
    
    YIN is specifically designed for short, decaying sounds like drum hits.
    Falls back to autocorrelation if YIN fails.
    
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
    
    # Search for peak within a window after onset
    peak_search_samples = int(10 * sr / 1000)  # 10ms search
    search_start = onset_sample
    search_end = min(len(audio), onset_sample + peak_search_samples)
    
    if search_end > search_start:
        search_segment = audio[search_start:search_end]
        if len(search_segment) > 0:
            peak_idx = np.argmax(np.abs(search_segment))
            onset_sample = search_start + peak_idx
    
    window_samples = int(window_ms * sr / 1000)
    start = onset_sample
    end = min(len(audio), onset_sample + window_samples)
    segment = audio[start:end]
    
    if len(segment) < 512:
        return None
    
    try:
        # Use YIN algorithm - better for short, decaying percussive sounds
        f0 = librosa.yin(
            segment,
            fmin=max(fmin, 40),
            fmax=fmax,
            sr=sr,
            frame_length=2048
        )
        
        # Filter out invalid values
        valid_pitches = f0[~np.isnan(f0) & (f0 > 0)]
        
        if len(valid_pitches) == 0:
            # Fallback to autocorrelation
            return detect_pitch_autocorrelation(audio, onset_time, sr, window_ms, fmin, fmax)
        
        # Return median pitch
        return float(np.median(valid_pitches))
        
    except Exception:
        # Fallback to autocorrelation
        return detect_pitch_autocorrelation(audio, onset_time, sr, window_ms, fmin, fmax)


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
    changes = []
    for i in range(1, len(decay_energies)):
        if decay_energies[i-1] > 0:
            change = (decay_energies[i] - decay_energies[i-1]) / decay_energies[i-1]
            changes.append(change)
    
    # Count increases vs decreases
    increases = sum(1 for c in changes if c > 0.1)
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
