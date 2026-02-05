"""
Energy-based onset detection - visual/DAW-like approach.

REPLACES LIBROSA ONSET DETECTION which had fundamental flaws for isolated drums:
1. librosa wait periods (wait=3 frames) caused real events to be skipped
2. Decay comparison to zero triggered false detections during long reverb
3. Generic music tuning doesn't work on clean separated drum stems
4. Over-detected: Thunderstruck 238 events vs 54 actual (4.4x too many)

NEW APPROACH uses energy analysis similar to how DAWs render waveforms:
- Calculate RMS energy envelope (what you SEE in DAW waveform)
- Find peaks using scipy.signal.find_peaks (robust, battle-tested)
- Filter by prominence (15dB above local minimum, calibrated)
- Backtrack from peak to attack start using left_bases + threshold (50-120ms earlier)
- No blind wait periods - every peak evaluated independently
- Result: 72 events detected at attack start, matching drummer timing

Parameters calibrated through iterative testing:
- threshold_db = 15.0 (prominence above local minimum)
- min_absolute_energy = 0.01 (noise floor for real cymbal hits)
- min_peak_spacing_ms = 100.0 (prevent double-detection, not blind wait)

Pure functional core - no side effects.
"""

import numpy as np
from typing import List, Tuple, Optional
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d


def find_amplitude_peaks_in_region(
    audio: np.ndarray,
    region_start_sample: int,
    region_end_sample: int,
    sr: int,
    min_spacing_ms: float = 90.0,
    prominence_fraction: float = 0.3,
    smoothing_ms: float = 3.0
) -> List[int]:
    """
    Find amplitude peaks within a region - like visual inspection in DAW.
    
    This is Pass 2 of two-pass detection: given a region where energy
    detection found "something happened", find the exact timing of stick
    impacts by looking at raw amplitude peaks.
    
    Uses peak-hold envelope (like DAW waveform display) with minimal
    smoothing (2-5ms) to preserve transients while rejecting noise spikes.
    
    Args:
        audio: Raw audio signal
        region_start_sample: Start of region from energy detection
        region_end_sample: End of region from energy detection
        sr: Sample rate
        min_spacing_ms: Minimum spacing between peaks (prevents doubles)
        prominence_fraction: Min prominence as fraction of max in region (0.3 = 30%)
        smoothing_ms: Peak-hold smoothing window in milliseconds (2-5ms)
    
    Returns:
        List of sample indices for amplitude peaks
    """
    if region_start_sample >= region_end_sample:
        return []
    
    # Extract region
    region = audio[region_start_sample:region_end_sample]
    if len(region) == 0:
        return []
    
    # Apply peak-hold envelope (like DAW waveform display)
    # This preserves sharp transients while smoothing out sample noise
    smooth_samples = max(1, int(smoothing_ms * sr / 1000.0))
    envelope = maximum_filter1d(np.abs(region), size=smooth_samples)
    
    # Find peaks in amplitude envelope
    min_spacing_samples = int(min_spacing_ms * sr / 1000.0)
    max_amplitude = np.max(envelope)
    prominence_threshold = max_amplitude * prominence_fraction
    
    peaks, properties = find_peaks(
        envelope,
        distance=min_spacing_samples,
        prominence=prominence_threshold
    )
    
    # Convert to absolute sample indices
    absolute_peaks = [region_start_sample + p for p in peaks]
    
    return absolute_peaks


def snap_to_amplitude_peak(
    audio: np.ndarray,
    onset_sample: int,
    peak_sample: int,
    search_window_ms: float = 30.0,
    sr: int = 44100
) -> int:
    """
    Snap detection to actual amplitude peak for percussive transients.
    
    Energy-based detection finds RMS energy peaks, but for drums we want
    the exact sample of maximum raw amplitude (stick impact). This searches
    forward from the backtracked onset to find the amplitude peak.
    
    For percussive hits, the amplitude peak is usually within 30ms of the
    energy peak. We search from onset forward to peak+30ms to handle cases
    where backtracking fails (sustained reverb tails keep energy elevated).
    
    Args:
        audio: Raw audio signal
        onset_sample: Backtracked onset position (attack start from energy)
        peak_sample: Energy envelope peak position
        search_window_ms: Forward search buffer in milliseconds (30ms default)
        sr: Sample rate
    
    Returns:
        Sample index of maximum amplitude peak within transient window
    """
    search_samples = int(search_window_ms * sr / 1000.0)
    
    # Search window: from backtracked onset to peak + 30ms buffer
    # This handles both normal cases and cases where backtracking failed
    # due to sustained energy (reverb tails from previous hits)
    search_start = max(0, onset_sample)
    search_end = min(len(audio), peak_sample + search_samples)
    
    if search_start >= search_end:
        return onset_sample
    
    # Find maximum absolute amplitude in window
    window = audio[search_start:search_end]
    max_idx = np.argmax(np.abs(window))
    
    return search_start + max_idx


def calculate_energy_envelope(
    audio: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    method: str = 'rms'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate energy envelope - similar to DAW waveform visualization.
    
    This is what you SEE in a DAW - the energy over time. Much more reliable
    than onset detection for isolated drum stems.
    
    Args:
        audio: Mono audio signal
        sr: Sample rate
        frame_length: Window size for energy calculation
        hop_length: Hop size between frames
        method: 'rms' (root mean square) or 'spectral' (spectral flux)
    
    Returns:
        Tuple of (times, energy_values)
        - times: Time in seconds for each frame
        - energy_values: Energy at each frame
    """
    if method == 'rms':
        # RMS energy - what you see in DAW waveform
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
    elif method == 'spectral':
        # Spectral flux - better for detecting timbral changes
        energy = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=hop_length
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate time for each frame
    times = librosa.frames_to_time(
        np.arange(len(energy)),
        sr=sr,
        hop_length=hop_length
    )
    
    return times, energy


def detect_transient_peaks(
    times: np.ndarray,
    energy: np.ndarray,
    threshold_db: float = 12.0,
    min_peak_spacing_ms: float = 100.0,
    min_absolute_energy: float = 0.001,
    pre_max: int = 3,
    post_max: int = 3,
    audio: Optional[np.ndarray] = None,
    sr: int = 44100,
) -> List[dict]:
    """
    Detect transient peaks - sharp attack moments visible in DAW.
    
    CRITICAL DIFFERENCE FROM LIBROSA ONSET DETECTION:
    - librosa uses spectral flux + wait periods that can miss real events
    - This uses scipy.signal.find_peaks on energy envelope (what you SEE in DAW)
    - No blind wait periods - every peak evaluated independently
    - Prominence-based: peaks must stand out above LOCAL baseline
    - BACKTRACKING: Finds attack start, not peak (crash cymbal takes ~120ms to peak)
    - PEAK SNAPPING: Snaps to actual amplitude peak for percussive precision
    - Result: Detects obvious events like 112s/119s cymbals that librosa missed
    
    Uses scipy.signal.find_peaks for robust peak detection, then backtracks
    from peak to find actual onset (attack start) using left_bases and threshold.
    For percussive instruments, snaps to the actual amplitude peak within the
    transient window for precise timing alignment with visual waveform peaks.
    
    Args:
        times: Time in seconds for each energy frame
        energy: Energy values at each frame
        threshold_db: Minimum prominence above neighbors in dB (calibrated to 15.0)
        min_peak_spacing_ms: Minimum time between peaks (prevents double-detection)
        min_absolute_energy: Absolute minimum energy to consider (noise floor = 0.01)
        pre_max: Unused (kept for API compatibility)
        post_max: Unused (kept for API compatibility)
        audio: Optional raw audio signal for amplitude peak snapping
        sr: Sample rate (for peak snapping)
    
    Returns:
        List of peak dicts with onset_time (snapped to amplitude peak), peak_energy, peak_time
    """
    if len(times) == 0 or len(energy) == 0:
        return []
    
    # Calculate frame spacing
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.01
    min_spacing_frames = int(min_peak_spacing_ms / 1000.0 / frame_duration)
    
    # Convert prominence from dB to linear scale
    # prominence_db means how many dB above local minima
    # For RMS energy, we need to convert this appropriately
    # Using a simpler approach: find peaks with minimum height
    
    # Use scipy's find_peaks - much more robust
    # Convert threshold from dB to linear prominence
    # threshold_db is relative to local minimum
    # Prominence in linear scale = height * (1 - 10^(-threshold_db/20))
    min_prominence_linear = min_absolute_energy * (10 ** (threshold_db / 20) - 1)
    
    peak_indices, properties = find_peaks(
        energy,
        height=min_absolute_energy,  # Minimum absolute height
        distance=min_spacing_frames,  # Minimum spacing between peaks
        prominence=max(min_prominence_linear, min_absolute_energy * 0.1),  # Prominence threshold
        wlen=None,  # Use full signal for prominence calculation
    )
    
    # Get left bases - where each peak starts rising from
    # This is scipy's built-in way to find onset points
    left_bases = properties.get('left_bases', peak_indices)
    
    # Backtrack from peak to find actual onset (attack start)
    # Even fast transients like crashes take 50-120ms to reach peak
    peaks = []
    for i, peak_idx in enumerate(peak_indices):
        peak_energy = energy[peak_idx]
        
        # Start from left base (where rise begins)
        base_idx = left_bases[i] if left_bases is not None else max(0, peak_idx - 50)
        
        # Backtrack further to find where energy crosses threshold
        # Look for point where energy is 10-20% of peak (catches attack start)
        # Use MAX of relative (15% of peak) and absolute threshold to handle quiet hits
        relative_threshold = peak_energy * 0.15  # 15% of peak energy
        absolute_threshold = min_absolute_energy * 1.5  # 1.5x noise floor
        onset_threshold = max(relative_threshold, absolute_threshold)
        onset_idx = peak_idx
        
        # Search backwards from peak to base
        for j in range(peak_idx - 1, max(0, base_idx - 10), -1):
            if energy[j] < onset_threshold:
                onset_idx = j + 1  # Onset is just after it crosses threshold
                break
        
        # PEAK SNAPPING: For percussive instruments, snap to actual amplitude peak
        # Energy envelope peak != raw amplitude peak (RMS smoothing causes offset)
        # This aligns MIDI events with visual waveform peaks for accurate timing
        final_onset_time = times[onset_idx]
        if audio is not None:
            # Calculate sample indices (energy frames -> audio samples)
            hop_length = int((times[1] - times[0]) * sr) if len(times) > 1 else 512
            onset_sample = int(onset_idx * hop_length)
            peak_sample = int(peak_idx * hop_length)
            
            # Snap to maximum amplitude within transient window
            snapped_sample = snap_to_amplitude_peak(
                audio, onset_sample, peak_sample, 
                search_window_ms=50.0, sr=sr
            )
            final_onset_time = float(snapped_sample / sr)
        
        peaks.append({
            'onset_time': final_onset_time,  # Snapped to amplitude peak if audio provided
            'peak_energy': float(peak_energy),
            'peak_time': float(times[peak_idx]),  # Keep energy peak time for reference
        })
    
    return peaks


def detect_transient_peaks_two_pass(
    times: np.ndarray,
    energy: np.ndarray,
    audio: np.ndarray,
    sr: int,
    threshold_db: float = 12.0,
    min_peak_spacing_ms: float = 100.0,
    min_absolute_energy: float = 0.001,
    amplitude_smoothing_ms: float = 3.0,
    amplitude_prominence: float = 0.3,
    max_peaks_per_region: int = 3,
) -> List[dict]:
    """
    Two-pass detection: energy regions + amplitude peak refinement.
    
    Pass 1 (Coarse): Find regions where drum events occur using RMS energy
    Pass 2 (Fine): Find exact amplitude peaks within each region (DAW-like)
    
    This combines the robustness of energy detection (filters noise/silence)
    with the timing precision of amplitude peaks (1-2ms resolution).
    
    Solves:
    - Missed peaks when RMS smoothing blends nearby hits (e.g., 149.467s)
    - False positives in reverb tails (e.g., 149.641s)
    - 90ms timing delays from RMS center-of-mass shift
    
    Args:
        times: Time in seconds for each energy frame
        energy: Energy values at each frame
        audio: Raw audio signal for amplitude peak detection
        sr: Sample rate
        threshold_db: Energy prominence threshold (Pass 1)
        min_peak_spacing_ms: Minimum spacing between peaks (Pass 2)
        min_absolute_energy: Noise floor for energy detection (Pass 1)
        amplitude_smoothing_ms: Peak-hold smoothing window (Pass 2, 2-5ms)
        amplitude_prominence: Min prominence fraction for amplitude peaks (Pass 2)
        max_peaks_per_region: Maximum amplitude peaks per energy region (safety limit)
    
    Returns:
        List of peak dicts with onset_time, peak_energy
    """
    if len(times) == 0 or len(energy) == 0 or audio is None:
        # Fall back to single-pass if audio not provided
        return detect_transient_peaks(
            times, energy, threshold_db, min_peak_spacing_ms,
            min_absolute_energy, audio=audio, sr=sr
        )
    
    # PASS 1: Find energy regions (coarse detection)
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.01
    hop_length = int(frame_duration * sr)
    min_spacing_frames = int(min_peak_spacing_ms / 1000.0 / frame_duration)
    
    min_prominence_linear = min_absolute_energy * (10 ** (threshold_db / 20) - 1)
    
    peak_indices, properties = find_peaks(
        energy,
        height=min_absolute_energy,
        distance=min_spacing_frames,
        prominence=max(min_prominence_linear, min_absolute_energy * 0.1),
        wlen=None,
    )
    
    left_bases = properties.get('left_bases', peak_indices)
    
    # PASS 2: Find amplitude peaks within each energy region
    results = []
    
    for i, peak_idx in enumerate(peak_indices):
        peak_energy = energy[peak_idx]
        
        # Define region boundaries
        base_idx = left_bases[i] if left_bases is not None else max(0, peak_idx - 50)
        
        # Backtrack to find region start (attack onset)
        relative_threshold = peak_energy * 0.15
        absolute_threshold = min_absolute_energy * 1.5
        onset_threshold = max(relative_threshold, absolute_threshold)
        onset_idx = peak_idx
        
        for j in range(peak_idx - 1, max(0, base_idx - 10), -1):
            if energy[j] < onset_threshold:
                onset_idx = j + 1
                break
        
        # Convert to sample indices
        region_start_sample = int(onset_idx * hop_length)
        region_end_sample = int(peak_idx * hop_length + hop_length * 10)  # +10 frames buffer
        region_end_sample = min(len(audio), region_end_sample)
        
        # Find amplitude peaks within this region
        amplitude_peaks = find_amplitude_peaks_in_region(
            audio,
            region_start_sample,
            region_end_sample,
            sr,
            min_spacing_ms=min_peak_spacing_ms,
            prominence_fraction=amplitude_prominence,
            smoothing_ms=amplitude_smoothing_ms
        )
        
        # Limit peaks per region (safety)
        if len(amplitude_peaks) > max_peaks_per_region:
            # Keep only the strongest peaks
            peak_amplitudes = [abs(audio[p]) for p in amplitude_peaks]
            sorted_indices = np.argsort(peak_amplitudes)[-max_peaks_per_region:]
            amplitude_peaks = [amplitude_peaks[idx] for idx in sorted(sorted_indices)]
        
        # If no amplitude peaks found, fall back to energy peak
        if len(amplitude_peaks) == 0:
            amplitude_peaks = [int(peak_idx * hop_length)]
        
        # Create result for each amplitude peak
        for amp_peak_sample in amplitude_peaks:
            results.append({
                'onset_time': float(amp_peak_sample / sr),
                'peak_energy': float(peak_energy),
                'peak_time': float(times[peak_idx]),
            })
    
    return results


def detect_energy_onsets(
    times: np.ndarray,
    energy: np.ndarray,
    threshold_db: float = 6.0,
    baseline_window_ms: float = 100.0,
    min_event_spacing_ms: float = 50.0,
    min_event_duration_ms: float = 20.0,
    max_event_duration_ms: float = 3000.0,
    decay_threshold_db: float = -12.0,
    decay_window_frames: int = 5,
    min_absolute_energy: float = 0.001,
) -> List[dict]:
    """
    Detect onsets from energy envelope using adaptive thresholding.
    
    This mimics visual analysis: look for energy increases above recent baseline,
    not arbitrary onset strength peaks. Works like looking at a DAW waveform.
    
    Algorithm:
    1. Calculate rolling baseline (recent energy average)
    2. Detect when energy exceeds baseline + threshold_db
    3. Track full event envelope (attack + decay)
    4. Next event detected relative to previous event's ending, not zero
    
    Args:
        times: Time in seconds for each energy frame
        energy: Energy values at each frame
        threshold_db: dB increase above baseline to trigger detection
        baseline_window_ms: Window size for calculating baseline (milliseconds)
        min_event_spacing_ms: Minimum time between event starts (milliseconds)
        min_event_duration_ms: Minimum event duration to be considered valid
        max_event_duration_ms: Maximum event duration (clips long decays/reverb)
        decay_threshold_db: dB below peak to consider event ended
        decay_window_frames: Number of consecutive frames below threshold to end event
        min_absolute_energy: Minimum absolute energy to consider (filters noise floor)
    
    Returns:
        List of event dicts with:
        - onset_time: Event start time (seconds)
        - peak_time: Time of peak energy (seconds)
        - end_time: Event end time when decay completes (seconds)
        - peak_energy: Maximum energy during event
        - onset_energy: Energy at onset
        - end_energy: Energy at event end
        - duration: Event duration (seconds)
    """
    if len(times) == 0 or len(energy) == 0:
        return []
    
    # Convert to dB for threshold comparisons
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    
    # Calculate frame intervals
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.01
    baseline_frames = int(baseline_window_ms / 1000.0 / frame_duration)
    min_spacing_frames = int(min_event_spacing_ms / 1000.0 / frame_duration)
    min_duration_frames = int(min_event_duration_ms / 1000.0 / frame_duration)
    max_duration_frames = int(max_event_duration_ms / 1000.0 / frame_duration)
    
    events = []
    i = 0
    
    while i < len(energy_db):
        # Skip if below absolute minimum energy (noise floor)
        if energy[i] < min_absolute_energy:
            i += 1
            continue
        
        # Calculate adaptive baseline from recent history
        baseline_start = max(0, i - baseline_frames)
        baseline_db = np.median(energy_db[baseline_start:i+1])
        
        # Check if current frame exceeds threshold above baseline
        if energy_db[i] > baseline_db + threshold_db:
            # Found onset - now track the full event envelope
            onset_idx = i
            onset_time = times[onset_idx]
            onset_energy = energy[onset_idx]
            
            # Find peak energy in forward window
            peak_idx = onset_idx
            peak_energy = energy[onset_idx]
            
            # Track forward to find peak and decay
            j = onset_idx + 1
            below_threshold_count = 0
            
            while j < len(energy_db):
                # Update peak if we find higher energy
                if energy[j] > peak_energy:
                    peak_energy = energy[j]
                    peak_idx = j
                    below_threshold_count = 0  # Reset decay counter
                
                # Check if we've decayed enough below peak
                if energy_db[j] < energy_db[peak_idx] + decay_threshold_db:
                    below_threshold_count += 1
                    if below_threshold_count >= decay_window_frames:
                        break
                else:
                    below_threshold_count = 0
                
                # Enforce maximum duration
                if j - onset_idx >= max_duration_frames:
                    break
                
                j += 1
            
            end_idx = min(j, len(energy_db) - 1)
            end_time = times[end_idx]
            end_energy = energy[end_idx]
            duration = end_time - onset_time
            
            # Validate minimum duration
            if end_idx - onset_idx >= min_duration_frames:
                events.append({
                    'onset_time': float(onset_time),
                    'peak_time': float(times[peak_idx]),
                    'end_time': float(end_time),
                    'peak_energy': float(peak_energy),
                    'onset_energy': float(onset_energy),
                    'end_energy': float(end_energy),
                    'duration': float(duration),
                })
                
                # Skip forward past this event + minimum spacing
                # Key difference: we skip past the FULL event, not arbitrary wait period
                i = end_idx + min_spacing_frames
            else:
                # Too short, skip just this frame
                i += 1
        else:
            i += 1
    
    return events


def detect_stereo_transient_peaks(
    stereo_audio: np.ndarray,
    sr: int,
    threshold_db: float = 12.0,
    min_peak_spacing_ms: float = 100.0,
    merge_window_ms: float = 150.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    method: str = 'rms',
    min_absolute_energy: float = 0.001,
    enable_amplitude_refinement: bool = True,
    amplitude_smoothing_ms: float = 3.0,
    amplitude_prominence: float = 0.3,
    max_peaks_per_region: int = 3,
) -> dict:
    """
    Detect transient peaks in stereo audio with optional two-pass refinement.
    
    Finds sharp attack transients (what you see in DAW). Can use two-pass
    detection for better timing accuracy in reverb-heavy material.
    
    Args:
        stereo_audio: Stereo audio (2, samples) or (samples, 2)
        sr: Sample rate
        threshold_db: Prominence threshold (12-18dB recommended for clean detection)
        min_peak_spacing_ms: Minimum spacing between peaks in same channel
        merge_window_ms: Merge L/R peaks within this window into one event
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        method: 'rms' or 'spectral'
        min_absolute_energy: Noise floor threshold
        enable_amplitude_refinement: Enable two-pass detection (energy + amplitude)
        amplitude_smoothing_ms: Peak-hold smoothing for amplitude peaks (2-5ms)
        amplitude_prominence: Min prominence fraction for amplitude peaks (0.3 = 30%)
        max_peaks_per_region: Max amplitude peaks per energy region (safety limit)
    
    Returns:
        Dict with onset_times, left_energies, right_energies, pan_confidence
    """
    # Ensure shape is (2, samples)
    if stereo_audio.ndim == 1:
        stereo_audio = np.stack([stereo_audio, stereo_audio], axis=0)
    elif stereo_audio.shape[0] != 2:
        if stereo_audio.shape[1] == 2:
            stereo_audio = stereo_audio.T
        else:
            stereo_audio = stereo_audio[:2]
    
    left_channel = stereo_audio[0]
    right_channel = stereo_audio[1]
    
    # Calculate energy envelopes
    left_times, left_energy = calculate_energy_envelope(
        left_channel, sr, frame_length, hop_length, method
    )
    right_times, right_energy = calculate_energy_envelope(
        right_channel, sr, frame_length, hop_length, method
    )
    
    # Detect transient peaks in each channel
    # Use two-pass detection if enabled (better timing in reverb-heavy material)
    if enable_amplitude_refinement:
        left_peaks = detect_transient_peaks_two_pass(
            left_times, left_energy, left_channel, sr,
            threshold_db, min_peak_spacing_ms, min_absolute_energy,
            amplitude_smoothing_ms, amplitude_prominence, max_peaks_per_region
        )
        right_peaks = detect_transient_peaks_two_pass(
            right_times, right_energy, right_channel, sr,
            threshold_db, min_peak_spacing_ms, min_absolute_energy,
            amplitude_smoothing_ms, amplitude_prominence, max_peaks_per_region
        )
    else:
        # Single-pass detection (legacy behavior)
        left_peaks = detect_transient_peaks(
            left_times, left_energy, threshold_db, min_peak_spacing_ms,
            min_absolute_energy, audio=left_channel, sr=sr
        )
        right_peaks = detect_transient_peaks(
            right_times, right_energy, threshold_db, min_peak_spacing_ms,
            min_absolute_energy, audio=right_channel, sr=sr
        )
    
    # Merge L/R peaks within merge window
    merge_window_sec = merge_window_ms / 1000.0
    merged_onsets = []
    used_right = set()
    
    # For each left peak, find matching right peak
    for left_pk in left_peaks:
        left_time = left_pk['onset_time']
        best_right_idx = None
        best_time_diff = float('inf')
        
        for right_idx, right_pk in enumerate(right_peaks):
            if right_idx in used_right:
                continue
            
            right_time = right_pk['onset_time']
            time_diff = abs(left_time - right_time)
            
            if time_diff <= merge_window_sec and time_diff < best_time_diff:
                best_time_diff = time_diff
                best_right_idx = right_idx
        
        if best_right_idx is not None:
            # Merge left + right
            right_pk = right_peaks[best_right_idx]
            used_right.add(best_right_idx)
            
            # Use earlier onset time
            onset_time = min(left_time, right_pk['onset_time'])
            
            merged_onsets.append({
                'onset_time': onset_time,
                'left_energy': left_pk['peak_energy'],
                'right_energy': right_pk['peak_energy'],
            })
        else:
            # Left-only peak
            merged_onsets.append({
                'onset_time': left_time,
                'left_energy': left_pk['peak_energy'],
                'right_energy': 0.0,
            })
    
    # Add right-only peaks
    for right_idx, right_pk in enumerate(right_peaks):
        if right_idx not in used_right:
            merged_onsets.append({
                'onset_time': right_pk['onset_time'],
                'left_energy': 0.0,
                'right_energy': right_pk['peak_energy'],
            })
    
    # Sort by time
    merged_onsets.sort(key=lambda x: x['onset_time'])
    
    # Extract arrays
    onset_times = [evt['onset_time'] for evt in merged_onsets]
    left_energies = [evt['left_energy'] for evt in merged_onsets]
    right_energies = [evt['right_energy'] for evt in merged_onsets]
    
    # Calculate pan confidence
    pan_confidence = []
    for l_e, r_e in zip(left_energies, right_energies):
        total = l_e + r_e
        if total > 0:
            pan = (r_e - l_e) / total
        else:
            pan = 0.0
        pan_confidence.append(pan)
    
    return {
        'onset_times': onset_times,
        'left_energies': left_energies,
        'right_energies': right_energies,
        'pan_confidence': pan_confidence,
    }


def detect_stereo_energy_onsets(
    stereo_audio: np.ndarray,
    sr: int,
    threshold_db: float = 6.0,
    baseline_window_ms: float = 100.0,
    min_event_spacing_ms: float = 50.0,
    merge_window_ms: float = 100.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    method: str = 'rms',
) -> dict:
    """
    Detect onsets in stereo audio using energy-based method.
    
    Processes L/R channels separately, then merges nearby detections similar
    to detect_dual_channel_onsets but using energy analysis instead of
    librosa onset detection.
    
    Args:
        stereo_audio: Stereo audio (2, samples) or (samples, 2)
        sr: Sample rate
        threshold_db: dB increase above baseline to trigger
        baseline_window_ms: Baseline calculation window
        min_event_spacing_ms: Minimum spacing between events
        merge_window_ms: Merge L/R events within this window
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        method: 'rms' or 'spectral'
    
    Returns:
        Dict with:
        - onset_times: Merged onset times (seconds)
        - left_energies: Left channel peak energy for each onset
        - right_energies: Right channel peak energy for each onset
        - pan_confidence: (R-L)/(R+L) pan position
        - durations: Event durations (seconds)
    """
    # Ensure shape is (2, samples)
    if stereo_audio.ndim == 1:
        stereo_audio = np.stack([stereo_audio, stereo_audio], axis=0)
    elif stereo_audio.shape[0] != 2:
        if stereo_audio.shape[1] == 2:
            stereo_audio = stereo_audio.T
        else:
            # Take first two channels
            stereo_audio = stereo_audio[:2]
    
    left_channel = stereo_audio[0]
    right_channel = stereo_audio[1]
    
    # Calculate energy envelopes
    left_times, left_energy = calculate_energy_envelope(
        left_channel, sr, frame_length, hop_length, method
    )
    right_times, right_energy = calculate_energy_envelope(
        right_channel, sr, frame_length, hop_length, method
    )
    
    # Detect events in each channel
    left_events = detect_energy_onsets(
        left_times, left_energy, threshold_db, baseline_window_ms,
        min_event_spacing_ms, min_event_duration_ms=20.0,
        max_event_duration_ms=3000.0, decay_threshold_db=-12.0,
        decay_window_frames=5, min_absolute_energy=0.001
    )
    right_events = detect_energy_onsets(
        right_times, right_energy, threshold_db, baseline_window_ms,
        min_event_spacing_ms, min_event_duration_ms=20.0,
        max_event_duration_ms=3000.0, decay_threshold_db=-12.0,
        decay_window_frames=5, min_absolute_energy=0.001
    )
    
    # Merge L/R events within merge window
    merge_window_sec = merge_window_ms / 1000.0
    merged_onsets = []
    used_right = set()
    
    # For each left event, find matching right event
    for left_evt in left_events:
        left_time = left_evt['onset_time']
        best_right_idx = None
        best_time_diff = float('inf')
        
        for right_idx, right_evt in enumerate(right_events):
            if right_idx in used_right:
                continue
            
            right_time = right_evt['onset_time']
            time_diff = abs(left_time - right_time)
            
            if time_diff <= merge_window_sec and time_diff < best_time_diff:
                best_time_diff = time_diff
                best_right_idx = right_idx
        
        if best_right_idx is not None:
            # Merge left + right
            right_evt = right_events[best_right_idx]
            used_right.add(best_right_idx)
            
            # Use earlier onset time
            onset_time = min(left_time, right_evt['onset_time'])
            
            merged_onsets.append({
                'onset_time': onset_time,
                'left_energy': left_evt['peak_energy'],
                'right_energy': right_evt['peak_energy'],
                'duration': max(left_evt['duration'], right_evt['duration']),
            })
        else:
            # Left-only event
            merged_onsets.append({
                'onset_time': left_time,
                'left_energy': left_evt['peak_energy'],
                'right_energy': 0.0,
                'duration': left_evt['duration'],
            })
    
    # Add right-only events
    for right_idx, right_evt in enumerate(right_events):
        if right_idx not in used_right:
            merged_onsets.append({
                'onset_time': right_evt['onset_time'],
                'left_energy': 0.0,
                'right_energy': right_evt['peak_energy'],
                'duration': right_evt['duration'],
            })
    
    # Sort by time
    merged_onsets.sort(key=lambda x: x['onset_time'])
    
    # Extract arrays
    onset_times = [evt['onset_time'] for evt in merged_onsets]
    left_energies = [evt['left_energy'] for evt in merged_onsets]
    right_energies = [evt['right_energy'] for evt in merged_onsets]
    durations = [evt['duration'] for evt in merged_onsets]
    
    # Calculate pan confidence
    pan_confidence = []
    for l_e, r_e in zip(left_energies, right_energies):
        total = l_e + r_e
        if total > 0:
            pan = (r_e - l_e) / total
        else:
            pan = 0.0
        pan_confidence.append(pan)
    
    return {
        'onset_times': onset_times,
        'left_energies': left_energies,
        'right_energies': right_energies,
        'pan_confidence': pan_confidence,
        'durations': durations,
    }
