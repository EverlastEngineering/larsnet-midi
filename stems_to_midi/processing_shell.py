"""
Stem Processing Module

Handles the main processing pipeline for converting audio stems to MIDI events.
"""

import numpy as np # type: ignore
import soundfile as sf # type: ignore
from pathlib import Path
from typing import Union, List, Dict, Optional

# Import functional core helpers
from .analysis_core import (
    ensure_mono,
    calculate_peak_amplitude,
    should_keep_onset,
    filter_onsets_by_spectral,
    normalize_values,
    estimate_velocity,
    classify_tom_pitch,
    classify_cymbal_pitch,
    classify_snare_pitch
)

# Import detection functions
from .detection_shell import (
    detect_onsets,
    detect_tom_pitch,
    detect_cymbal_pitch,
    detect_snare_pitch,
    detect_hihat_state
)
from .energy_detection_shell import detect_onsets_energy_based

# Import config structures
from .config import DrumMapping

__all__ = [
    'process_stem_to_midi'
]


def _load_and_validate_audio(
    audio_path: Union[str, Path],
    config: Dict,
    stem_type: str,
    max_duration: Optional[float] = None
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load audio file and validate it's usable.
    
    Helper function for process_stem_to_midi (imperative shell).
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        stem_type: Type of stem (for logging)
        max_duration: Maximum duration in seconds to load (None = load all)
    
    Returns:
        Tuple of (audio, sample_rate) or (None, None) if invalid
    """
    
    print(f"Status Update: Generating MIDI from {stem_type.capitalize()}")
    print(f"    from: {audio_path.name}")
    
    # Load audio (I/O)
    audio, sr = sf.read(str(audio_path))
    
    # Truncate to max_duration if specified
    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"    Truncated to {max_duration:.1f} seconds for faster processing")

    # Debug: Print audio shape and sample rate
    print(f"    Audio shape: {audio.shape}, Sample rate: {sr}")
    print(f"    Audio min: {audio.min():.6f}, max: {audio.max():.6f}, mean: {audio.mean():.6f}")
    if audio.shape[0] > sr:
        print(f"    First second min: {audio[:sr].min():.6f}, max: {audio[:sr].max():.6f}, mean: {audio[:sr].mean():.6f}")

    # Handle stereo/mono conversion based on per-stem or global settings
    # Priority: per-stem use_stereo > global force_mono
    stem_config = config.get(stem_type, {})
    use_stereo = stem_config.get('use_stereo', None)
    
    # If per-stem setting not specified, fall back to global force_mono
    if use_stereo is None:
        # Legacy behavior: respect global force_mono setting
        use_stereo = not config['audio'].get('force_mono', True)
    
    if not use_stereo and audio.ndim == 2:
        # Convert to mono
        audio = ensure_mono(audio)
        print("    Converted stereo to mono")
    elif use_stereo and audio.ndim == 2:
        # Keep stereo for spatial analysis
        print("    Keeping stereo for spatial analysis")
    elif use_stereo and audio.ndim == 1:
        # Mono file but stereo requested - just keep mono
        print("    Audio is mono (no stereo info available)")

    # Check if audio is essentially silent
    max_amplitude = np.max(np.abs(audio))
    print(f"    Max amplitude: {max_amplitude:.6f}")

    silence_threshold = config.get('audio', {}).get('silence_threshold', 0.001)
    if max_amplitude < silence_threshold:
        print("    Audio is silent, skipping...")
        return None, None

    # Amplitude normalization (per-stem or global setting)
    # This ensures spectral energy thresholds work consistently across quiet/loud recordings
    normalize_amplitude = stem_config.get('normalize_amplitude', 
                                          config.get('audio', {}).get('normalize_amplitude', False))
    
    if normalize_amplitude and max_amplitude > 0:
        target_amplitude = config.get('audio', {}).get('target_amplitude', 0.8)
        if max_amplitude < target_amplitude:  # Only normalize if too quiet
            scale_factor = target_amplitude / max_amplitude
            audio = audio * scale_factor
            print(f"    Normalized amplitude: {max_amplitude:.6f} -> {target_amplitude:.2f} (scale: {scale_factor:.2f}x)")

    # Stereo channel balance normalization (per-stem or global setting)
    # This equalizes L/R channel levels for fair detection when one channel is louder
    normalize_stereo_balance = stem_config.get('normalize_stereo_balance',
                                               config.get('audio', {}).get('normalize_stereo_balance', False))
    
    if normalize_stereo_balance and audio.ndim == 2:
        left_rms = np.sqrt(np.mean(audio[:, 0] ** 2))
        right_rms = np.sqrt(np.mean(audio[:, 1] ** 2))
        
        if left_rms > 0 and right_rms > 0:
            # Normalize each channel to equal RMS
            target_rms = (left_rms + right_rms) / 2
            left_scale = target_rms / left_rms
            right_scale = target_rms / right_rms
            
            audio[:, 0] *= left_scale
            audio[:, 1] *= right_scale
            
            print(f"    Balanced stereo channels: L×{left_scale:.2f}, R×{right_scale:.2f} (R/L was {right_rms/left_rms:.2f}x)")

    return audio, sr


def _configure_onset_detection(
    config: Dict,
    stem_type: str
) -> Dict:
    """
    Get onset detection parameters from config.
    
    Helper function for process_stem_to_midi (imperative shell).
    
    Args:
        config: Configuration dictionary
        stem_type: Type of stem
    
    Returns:
        Dictionary with onset detection parameters
    """
    learning_mode = config.get('learning_mode', {}).get('enabled', False)
    onset_config = config['onset_detection']
    stem_config = config.get(stem_type, {})
    
    if learning_mode:
        # Ultra-sensitive detection for learning mode
        learning_config = config['learning_mode']
        return {
            'hop_length': onset_config['hop_length'],
            'threshold': learning_config['learning_onset_threshold'],
            'delta': learning_config['learning_delta'],
            'wait': learning_config['learning_wait'],
            'learning_mode': True
        }
    else:
        # Normal detection - use stem-specific settings if provided, fallback to global if None
        threshold = stem_config.get('onset_threshold')
        if threshold is None:
            threshold = onset_config['threshold']
        delta = stem_config.get('onset_delta')
        if delta is None:
            delta = onset_config['delta']
        wait = stem_config.get('onset_wait')
        if wait is None:
            wait = onset_config['wait']
        timing_offset = stem_config.get('timing_offset', 0.0)

        return {
            'hop_length': onset_config['hop_length'],
            'threshold': threshold,
            'delta': delta,
            'wait': wait,
            'timing_offset': timing_offset,
            'learning_mode': False
        }


def _detect_tom_pitches(
    audio: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    config: Dict
) -> Optional[np.ndarray]:
    """
    Detect and classify tom pitches.
    
    Helper function for process_stem_to_midi (imperative shell).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        onset_times: Times of detected onsets
        config: Configuration dictionary
    
    Returns:
        Array of tom classifications (0=low, 1=mid, 2=high) or None
    """
    if len(onset_times) == 0:
        return None
    
    tom_config = config.get('toms', {})
    enable_pitch = tom_config.get('enable_pitch_detection', True)
    
    if not enable_pitch:
        return None
    
    print("\n    Detecting tom pitches...")
    pitch_method = tom_config.get('pitch_method', 'yin')
    min_pitch = tom_config.get('min_pitch_hz', 60.0)
    max_pitch = tom_config.get('max_pitch_hz', 250.0)
    
    # Detect pitch for each tom hit
    detected_pitches = []
    for onset_time in onset_times:
        pitch = detect_tom_pitch(
            audio, sr, onset_time, 
            method=pitch_method,
            min_hz=min_pitch,
            max_hz=max_pitch
        )
        detected_pitches.append(pitch)
    
    detected_pitches = np.array(detected_pitches)
    
    # Show detected pitches
    valid_pitches = detected_pitches[detected_pitches > 0]
    if len(valid_pitches) > 0:
        print(f"    Detected pitches: min={np.min(valid_pitches):.1f}Hz, max={np.max(valid_pitches):.1f}Hz, mean={np.mean(valid_pitches):.1f}Hz")
        print(f"    Unique pitches: {len(np.unique(valid_pitches))}")
    else:
        print("    Warning: No valid pitches detected, all toms will use default (mid) note")
    
    # Classify into low/mid/high
    tom_classifications = classify_tom_pitch(detected_pitches)
    
    # Show classification summary
    low_count = np.sum(tom_classifications == 0)
    mid_count = np.sum(tom_classifications == 1)
    high_count = np.sum(tom_classifications == 2)
    print(f"    Tom classification: {low_count} low, {mid_count} mid, {high_count} high")
    
    # Show detailed pitch table (if not too many)
    if len(onset_times) <= 20:
        print(f"\n      {'Time':>8s} {'Pitch(Hz)':>10s} {'Tom':>8s}")
        for i, (time, pitch, classification) in enumerate(zip(onset_times, detected_pitches, tom_classifications)):
            tom_name = ['Low', 'Mid', 'High'][classification]
            pitch_str = f"{pitch:.1f}" if pitch > 0 else "N/A"
            print(f"      {time:8.3f} {pitch_str:>10s} {tom_name:>8s}")
    
    return tom_classifications


def _detect_cymbal_pitches(
    audio: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    config: Dict,
    pan_positions: Optional[List[float]] = None,
    spectral_data: Optional[List[Dict]] = None
) -> Optional[np.ndarray]:
    """
    Detect and classify cymbal pitches, optionally using pan position.
    
    Helper function for process_stem_to_midi (imperative shell).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        onset_times: Times of detected onsets
        config: Configuration dictionary
        pan_positions: Optional pan positions for each onset (-1 to +1)
        spectral_data: Optional spectral analysis data for each onset
    
    Returns:
        Array of cymbal classifications (0=crash, 1=ride, 2=chinese) or None
    """
    if len(onset_times) == 0:
        return None
    
    cymbal_config = config.get('cymbals', {})
    enable_pitch = cymbal_config.get('enable_pitch_detection', True)
    
    if not enable_pitch:
        return None
    
    print("\n    Detecting cymbal pitches...")
    pitch_method = cymbal_config.get('pitch_method', 'yin')
    min_pitch = cymbal_config.get('min_pitch_hz', 200.0)
    max_pitch = cymbal_config.get('max_pitch_hz', 1000.0)
    
    # Detect pitch for each cymbal hit
    detected_pitches = []
    for onset_time in onset_times:
        pitch = detect_cymbal_pitch(
            audio, sr, onset_time, 
            method=pitch_method,
            min_hz=min_pitch,
            max_hz=max_pitch
        )
        detected_pitches.append(pitch)
    
    detected_pitches = np.array(detected_pitches)
    
    # Show detected pitches
    valid_pitches = detected_pitches[detected_pitches > 0]
    if len(valid_pitches) > 0:
        print(f"    Detected pitches: min={np.min(valid_pitches):.1f}Hz, max={np.max(valid_pitches):.1f}Hz, mean={np.mean(valid_pitches):.1f}Hz")
        print(f"    Unique pitches: {len(np.unique(valid_pitches))}")
    else:
        print("    Warning: No valid pitches detected, all cymbals will use default (crash) note")
    
    # Classify into crash/ride/chinese
    # TODO: Replace with clustering-based classification (Phase 6)
    # Current implementation uses hard-coded pan thresholds (-0.25/+0.25)
    # which don't adapt to recording characteristics.
    # Use pan-aware classification if pan positions available
    if pan_positions is not None and len(pan_positions) == len(onset_times):
        from .analysis_core import classify_cymbal_by_pan
        
        print(f"    Using pan-aware classification...")
        cymbal_classifications = []
        for i, (pitch, pan) in enumerate(zip(detected_pitches, pan_positions)):
            # Get spectral features for this onset if available
            spectral_features = spectral_data[i] if spectral_data and i < len(spectral_data) else None
            
            # Classify using pan + pitch + spectral
            classification = classify_cymbal_by_pan(pan, pitch, spectral_features)
            cymbal_classifications.append(classification)
        
        cymbal_classifications = np.array(cymbal_classifications)
    else:
        # Fall back to pitch-only classification
        print(f"    Using pitch-only classification (no pan data)...")
        cymbal_classifications = classify_cymbal_pitch(detected_pitches)
    
    # Show classification summary
    crash_count = np.sum(cymbal_classifications == 0)
    ride_count = np.sum(cymbal_classifications == 1)
    chinese_count = np.sum(cymbal_classifications == 2)
    print(f"    Cymbal classification: {crash_count} crash, {ride_count} ride, {chinese_count} chinese")
    
    # Show detailed pitch table (if not too many)
    if len(onset_times) <= 20:
        if pan_positions:
            print(f"\n      {'Time':>8s} {'Pitch(Hz)':>10s} {'Pan':>6s} {'Cymbal':>8s}")
            for i, (time, pitch, pan, classification) in enumerate(zip(onset_times, detected_pitches, pan_positions, cymbal_classifications)):
                cymbal_name = ['Crash', 'Ride', 'Chinese'][classification]
                pitch_str = f"{pitch:.1f}" if pitch > 0 else "N/A"
                pan_str = f"{pan:+.2f}"
                print(f"      {time:8.3f} {pitch_str:>10s} {pan_str:>6s} {cymbal_name:>8s}")
        else:
            print(f"\n      {'Time':>8s} {'Pitch(Hz)':>10s} {'Cymbal':>8s}")
            for i, (time, pitch, classification) in enumerate(zip(onset_times, detected_pitches, cymbal_classifications)):
                cymbal_name = ['Crash', 'Ride', 'Chinese'][classification]
                pitch_str = f"{pitch:.1f}" if pitch > 0 else "N/A"
                print(f"      {time:8.3f} {pitch_str:>10s} {cymbal_name:>8s}")
    
    return cymbal_classifications


def _detect_snare_pitches(
    audio: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    config: Dict
) -> Optional[np.ndarray]:
    """
    Detect and classify snare pitches.
    
    Helper function for process_stem_to_midi (imperative shell).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        onset_times: Times of detected onsets
        config: Configuration dictionary
    
    Returns:
        Array of snare classifications (0=snare, 1=rimshot, 2=clap, 3=clap+snare) or None
    """
    if len(onset_times) == 0:
        return None
    
    snare_config = config.get('snare', {})
    enable_pitch = snare_config.get('enable_pitch_detection', True)
    
    if not enable_pitch:
        return None
    
    print("\n    Detecting snare pitch characteristics...")
    pitch_method = snare_config.get('pitch_method', 'yin')
    min_pitch = snare_config.get('min_pitch_hz', 100.0)
    max_pitch = snare_config.get('max_pitch_hz', 500.0)
    
    # Detect pitch for each snare hit
    detected_pitches = []
    for onset_time in onset_times:
        pitch = detect_snare_pitch(
            audio, sr, onset_time, 
            method=pitch_method,
            min_hz=min_pitch,
            max_hz=max_pitch
        )
        detected_pitches.append(pitch)
    
    detected_pitches = np.array(detected_pitches)
    
    # Show detected pitches
    valid_pitches = detected_pitches[detected_pitches > 0]
    if len(valid_pitches) > 0:
        print(f"    Detected pitches: min={np.min(valid_pitches):.1f}Hz, max={np.max(valid_pitches):.1f}Hz, mean={np.mean(valid_pitches):.1f}Hz")
        print(f"    Unique pitches: {len(np.unique(valid_pitches))}")
    else:
        print("    Warning: No valid pitches detected, all will use default (snare) note")
    
    # Classify into snare/rimshot/clap/clap+snare
    snare_classifications = classify_snare_pitch(detected_pitches)
    
    # Show classification summary
    snare_count = np.sum(snare_classifications == 0)
    rimshot_count = np.sum(snare_classifications == 1)
    clap_count = np.sum(snare_classifications == 2)
    clap_snare_count = np.sum(snare_classifications == 3)
    print(f"    Snare classification: {snare_count} snare, {rimshot_count} rimshot, {clap_count} clap, {clap_snare_count} clap+snare")
    
    # Show detailed pitch table (if not too many)
    if len(onset_times) <= 20:
        print(f"\n      {'Time':>8s} {'Pitch(Hz)':>10s} {'Type':>12s}")
        for i, (time, pitch, classification) in enumerate(zip(onset_times, detected_pitches, snare_classifications)):
            type_name = ['Snare', 'Rimshot', 'Clap', 'Clap+Snare'][classification]
            pitch_str = f"{pitch:.1f}" if pitch > 0 else "N/A"
            print(f"      {time:8.3f} {pitch_str:>10s} {type_name:>12s}")
    
    return snare_classifications


def _create_midi_events(
    onset_times: np.ndarray,
    normalized_values: np.ndarray,
    stem_type: str,
    note: int,
    min_velocity: int,
    max_velocity: int,
    hihat_states: List[str],
    tom_classifications: Optional[np.ndarray],
    cymbal_classifications: Optional[np.ndarray],
    snare_classifications: Optional[np.ndarray],
    drum_mapping: DrumMapping,
    config: Dict,
    sustain_durations: Optional[List[float]] = None,
    spectral_data: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Create MIDI events from onset data.
    
    Helper function for process_stem_to_midi (imperative shell).
    
    Args:
        onset_times: Array of onset times
        normalized_values: Normalized feature values for velocity
        stem_type: Type of stem
        note: Default MIDI note number
        min_velocity: Minimum MIDI velocity
        max_velocity: Maximum MIDI velocity
        hihat_states: List of hihat states (closed/open/handclap)
        tom_classifications: Tom classifications (low/mid/high)
        cymbal_classifications: Cymbal classifications (crash/ride/chinese)
        snare_classifications: Snare classifications (snare/rimshot/clap/clap+snare)
        drum_mapping: MIDI note mapping
        config: Configuration dictionary
        sustain_durations: Optional list of sustain durations in milliseconds (for cymbals and hihat foot-close events)
        spectral_data: Optional list of spectral analysis dicts (Detection Output Contract)
    
    Returns:
        List of MIDI event dictionaries with optional spectral fields
    """
    # Get timing offset for this stem type (applied to MIDI timing only, not audio analysis)
    stem_config = config.get(stem_type, {})
    timing_offset = stem_config.get('timing_offset', 0.0)
    
    # Get hihat foot-close settings
    generate_foot_close = stem_config.get('generate_foot_close', False)
    foot_close_note = stem_config.get('midi_note_foot_close', 44)
    
    events = []
    
    for i, (time, value) in enumerate(zip(onset_times, normalized_values)):
        # Calculate velocity from normalized value
        velocity = estimate_velocity(value, min_velocity, max_velocity)
        
        # Adjust note for handclap, open hi-hat, tom/cymbal/snare classification
        if stem_type == 'hihat' and hihat_states[i] == 'handclap':
            midi_note = drum_mapping.handclap
        elif stem_type == 'hihat' and hihat_states[i] == 'open':
            midi_note = drum_mapping.hihat_open
        elif stem_type == 'toms' and tom_classifications is not None and i < len(tom_classifications):
            # Use low/mid/high tom note based on pitch classification
            if tom_classifications[i] == 0:
                midi_note = drum_mapping.tom_low
            elif tom_classifications[i] == 2:
                midi_note = drum_mapping.tom_high
            else:  # mid or default
                midi_note = drum_mapping.tom_mid
        elif stem_type == 'cymbals' and cymbal_classifications is not None and i < len(cymbal_classifications):
            # Use crash/ride/chinese note based on pitch classification
            if cymbal_classifications[i] == 0:
                midi_note = drum_mapping.crash
            elif cymbal_classifications[i] == 2:
                midi_note = drum_mapping.chinese
            else:  # ride or default
                midi_note = drum_mapping.ride
        elif stem_type == 'snare' and snare_classifications is not None and i < len(snare_classifications):
            # Use snare/rimshot/clap/clap+snare note based on pitch classification
            if snare_classifications[i] == 0:
                midi_note = drum_mapping.snare
            elif snare_classifications[i] == 1:
                midi_note = drum_mapping.snare_rimshot
            elif snare_classifications[i] == 2:
                midi_note = drum_mapping.snare_clap
            else:  # clap+snare
                midi_note = drum_mapping.snare_clap_snare
        else:
            midi_note = note
        
        # Duration: use sustain duration for cymbals, otherwise time until next hit
        if stem_type == 'cymbals' and sustain_durations is not None and i < len(sustain_durations):
            # Use actual sustain duration from envelope analysis (in milliseconds)
            duration = sustain_durations[i] / 1000.0  # Convert ms to seconds
            # Apply a more generous max for cymbals
            cymbal_max = config.get(stem_type, {}).get('max_note_duration', 2.0)
            duration = min(duration, cymbal_max)
        elif i < len(onset_times) - 1:
            # Standard duration: until next hit
            duration = onset_times[i + 1] - time
            max_duration = config.get('midi', {}).get('max_note_duration', 0.5)
            duration = min(duration, max_duration)
        else:
            # Last note: use default duration
            default_duration = config.get('audio', {}).get('default_note_duration', 0.1)
            duration = default_duration
        
        # Apply timing offset to MIDI event (compensates for onset detection timing)
        midi_time = float(time) + timing_offset
        
        # Create base event with MIDI essentials
        event = {
            'time': midi_time,
            'note': int(midi_note),
            'velocity': int(velocity),
            'duration': float(duration)
        }
        
        # Add spectral data if available (Detection Output Contract)
        # MIDI export ignores these extra fields; analysis/learning tools use them
        if spectral_data is not None and i < len(spectral_data):
            onset_info = spectral_data[i]
            # Common fields
            event['onset_strength'] = onset_info.get('strength')
            event['peak_amplitude'] = onset_info.get('amplitude')
            event['geomean'] = onset_info.get('geomean')
            event['total_energy'] = onset_info.get('total_energy')
            event['status'] = onset_info.get('status')
            # Domain-specific energy bands (body_energy, wire_energy, etc.)
            geomean_bands = onset_info.get('geomean_bands', [])
            event['geomean_bands'] = geomean_bands
            for band_name in geomean_bands:
                energy_key = f'{band_name}_energy'
                if energy_key in onset_info:
                    event[energy_key] = onset_info[energy_key]
            if 'sustain_ms' in onset_info:
                event['sustain_ms'] = onset_info.get('sustain_ms')
        
        events.append(event)
        
        # Generate foot-close event for open hihats
        if (stem_type == 'hihat' and hihat_states[i] == 'open' and 
            generate_foot_close and sustain_durations is not None and 
            i < len(sustain_durations)):
            # Foot close occurs at the end of the open hihat sustain
            sustain_seconds = sustain_durations[i] / 1000.0  # Convert ms to seconds
            foot_close_time = midi_time + sustain_seconds
            
            # Use a moderate velocity for foot close (softer than the open hit)
            foot_close_velocity = int(velocity * 0.7)  # 70% of open hit velocity
            foot_close_velocity = max(foot_close_velocity, 40)  # Minimum 40
            foot_close_velocity = min(foot_close_velocity, 100)  # Maximum 100
            
            events.append({
                'time': float(foot_close_time),
                'note': int(foot_close_note),
                'velocity': int(foot_close_velocity),
                'duration': 0.05  # Short duration for foot close
            })
    
    return events


def process_stem_to_midi(
    audio_path: Union[str, Path],
    stem_type: str,
    drum_mapping: DrumMapping,
    config: Dict,
    onset_threshold: float,
    onset_delta: float,
    onset_wait: int,
    hop_length: int,
    min_velocity: int = 80,
    max_velocity: int = 110,
    detect_hihat_open: bool = True,
    max_duration: Optional[float] = None
) -> List[Dict]:
    """
    Process a drum stem and extract MIDI events.
    
    This is a thin coordinator that orchestrates the processing pipeline:
    1. Load and validate audio
    2. Configure and detect onsets
    3. Filter by spectral content (if applicable)
    4. Classify drum types (hihat/tom)
    5. Create MIDI events
    
    Args:
        audio_path: Path to audio file
        stem_type: Type of stem ('kick', 'snare', 'toms', 'hihat', 'cymbals')
        drum_mapping: MIDI note mapping
        onset_threshold: Threshold for onset detection (0-1)
        min_velocity: Minimum MIDI velocity
        max_velocity: Maximum MIDI velocity
        detect_hihat_open: Try to detect open hi-hat
        max_duration: Maximum duration in seconds to analyze (None = all)
    
    Returns:
        Dict with:
            'events': List of MIDI events
            'all_onset_data': List of all detected onsets (kept + filtered)
            'spectral_config': Spectral config used for this stem
    """
    # Step 1: Load and validate audio
    audio, sr = _load_and_validate_audio(audio_path, config, stem_type, max_duration)
    if audio is None:
        return {'events': [], 'all_onset_data': [], 'spectral_config': None}
    
    # Track if we're processing stereo (for pan metadata later)
    is_stereo = audio.ndim == 2
    stereo_audio = audio if is_stereo else None  # Keep reference to stereo data
    
    # If stereo, create mono version for onset detection (average channels)
    # Onset detection works better on mono, but we'll add pan info after
    if is_stereo:
        from .analysis_core import ensure_mono
        audio_mono = ensure_mono(audio)
    else:
        audio_mono = audio
    
    # Step 2: Configure and detect onsets
    onset_params = _configure_onset_detection(config, stem_type)
    learning_mode = onset_params['learning_mode']

    # CLI params are used as fallbacks - per-stem config takes precedence
    # Only override if the stem config didn't specify a value (was None)
    stem_config = config.get(stem_type, {})
    if stem_config.get('onset_threshold') is None:
        onset_params['threshold'] = onset_threshold
    if stem_config.get('onset_delta') is None:
        onset_params['delta'] = onset_delta
    if stem_config.get('onset_wait') is None:
        onset_params['wait'] = onset_wait
    # hop_length always comes from global config (not per-stem)
    onset_params['hop_length'] = hop_length

    # Determine which detection method to use
    # DEFAULT: New energy-based detection (stereo-aware, scipy peaks, backtracked)
    # FALLBACK: Old librosa detection (use_librosa_detection: true per-stem)
    use_librosa = config.get(stem_type, {}).get('use_librosa_detection', False)
    
    if use_librosa:
        # OLD METHOD: Librosa onset detection (fallback option)
        print(f"    Using librosa onset detection (legacy mode)")
        onset_times, onset_strengths = detect_onsets(
            audio_mono,  # Use mono for onset detection
            sr,
            hop_length=onset_params['hop_length'],
            threshold=onset_params['threshold'],
            delta=onset_params['delta'],
            wait=onset_params['wait']
        )
        
        # Calculate pan position for each onset if stereo (post-hoc)
        pan_positions = None
        pan_classifications = None
        if is_stereo and len(onset_times) > 0:
            from .stereo_core import calculate_pan_position, classify_onset_by_pan
            
            print(f"    Calculating pan positions for {len(onset_times)} onsets...")
            pan_positions = []
            pan_classifications = []
            
            for onset_time in onset_times:
                onset_sample = int(onset_time * sr)
                pan = calculate_pan_position(stereo_audio, onset_sample, sr, window_ms=10.0)
                pan_class = classify_onset_by_pan(pan, center_threshold=0.15)
                pan_positions.append(pan)
                pan_classifications.append(pan_class)
            
            # Summary of pan distribution
            left_count = pan_classifications.count('left')
            center_count = pan_classifications.count('center')
            right_count = pan_classifications.count('right')
            print(f"    Pan distribution: {left_count} left, {center_count} center, {right_count} right")
    else:
        # NEW METHOD (DEFAULT): Energy-based detection with scipy peaks + backtracking
        print(f"    Using energy-based detection (scipy peaks, stereo-aware, backtracked)")
        
        # Get per-stem calibration parameters (with sensible defaults)
        threshold_db = config.get(stem_type, {}).get('threshold_db', 15.0)
        min_peak_spacing_ms = config.get(stem_type, {}).get('min_peak_spacing_ms', 100.0)
        min_absolute_energy = config.get(stem_type, {}).get('min_absolute_energy', 0.01)
        merge_window_ms = config.get(stem_type, {}).get('merge_window_ms', 150.0)
        energy_method = config.get(stem_type, {}).get('energy_method', 'rms')
        peak_hold_ms = config.get(stem_type, {}).get('peak_hold_ms', 3.0)
        
        onset_times, onset_strengths, extra_data = detect_onsets_energy_based(
            audio if is_stereo else audio_mono,  # Pass stereo if available
            sr,
            threshold_db=threshold_db,
            min_peak_spacing_ms=min_peak_spacing_ms,
            min_absolute_energy=min_absolute_energy,
            merge_window_ms=merge_window_ms,
            hop_length=onset_params['hop_length'],
            method=energy_method,
            peak_hold_ms=peak_hold_ms,
        )
        
        # Pan information already calculated in detection
        pan_positions = extra_data.get('pan_positions')
        pan_classifications = extra_data.get('pan_classifications')
        
        # Summary of pan distribution
        if pan_classifications:
            left_count = pan_classifications.count('left')
            center_count = pan_classifications.count('center')
            right_count = pan_classifications.count('right')
            print(f"    Pan distribution: {left_count} left, {center_count} center, {right_count} right")

    # Debug: Print all raw detected onset times before filtering
    print(f"    Raw detected onset times (s): {onset_times}")

    # Log detection mode
    if learning_mode:
        print(f"    Learning mode: Ultra-sensitive detection (threshold={onset_params['threshold']})")
    else:
        stem_config = config.get(stem_type, {})
        if (stem_config.get('onset_threshold') is not None or 
            stem_config.get('onset_delta') is not None or 
            stem_config.get('onset_wait') is not None):
            print(f"    {stem_type.capitalize()}-specific onset detection: threshold={onset_params['threshold']}, delta={onset_params['delta']}, wait={onset_params['wait']} (~{onset_params['wait']*11:.0f}ms min spacing)")

    print(f"    Found {len(onset_times)} hits (before filtering) -> MIDI note {getattr(drum_mapping, stem_type)}")
    
    if len(onset_times) == 0:
        return {'events': [], 'all_onset_data': [], 'spectral_config': None}
    
    # Step 3: Calculate event durations (NEW)
    from .analysis_core import calculate_event_durations
    durations = calculate_event_durations(onset_times, audio_mono, sr)
    print(f"    Calculated event durations: min={np.min(durations):.3f}s, max={np.max(durations):.3f}s, mean={np.mean(durations):.3f}s")
    
    # Step 4: Calculate peak amplitudes for all onsets
    peak_amplitudes = np.array([
        calculate_peak_amplitude(audio_mono, int(onset_time * sr), sr, window_ms=10.0)
        for onset_time in onset_times
    ])
    
    # For snare, kick, toms, hihat, and cymbals: filter out artifacts/bleed by checking spectral content
    # This uses the functional core for all calculations
    show_all_onsets = config.get('debug', {}).get('show_all_onsets', False)
    show_spectral_data = config.get('debug', {}).get('show_spectral_data', False)
    
    # Check if spectral filtering is enabled for this stem
    stem_config = config.get(stem_type, {})
    enable_spectral_filter = stem_config.get('enable_spectral_filter', True)
    
    # Initialize variables that may be used later (in case filtering is disabled)
    stem_geomeans = None
    hihat_sustain_durations = None
    hihat_spectral_data = None
    cymbal_sustain_durations = None

    if stem_type in ['snare', 'kick', 'toms', 'hihat', 'cymbals'] and len(onset_times) > 0 and enable_spectral_filter:
        # Use functional core helper for filtering
        filter_result = filter_onsets_by_spectral(
            onset_times,
            onset_strengths,
            peak_amplitudes,
            audio_mono,  # Use mono for spectral analysis
            sr,
            stem_type,
            config,
            learning_mode=learning_mode,
            durations=durations  # NEW: Pass event durations as keyword arg
        )

        # Extract filtered results
        onset_times = filter_result['filtered_times']
        onset_strengths = filter_result['filtered_strengths']
        peak_amplitudes = filter_result['filtered_amplitudes']
        stem_geomeans = filter_result['filtered_geomeans']
        hihat_sustain_durations = filter_result['filtered_sustains'] if stem_type == 'hihat' else None
        hihat_spectral_data = filter_result['filtered_spectral'] if stem_type == 'hihat' else None
        cymbal_sustain_durations = filter_result['filtered_sustains'] if stem_type == 'cymbals' else None
        all_onset_data = filter_result['all_onset_data']
        spectral_config = filter_result['spectral_config']
        filtered_onset_data = filter_result.get('filtered_onset_data', [])

        # Show ALL onset data and spectral chart if debug flags are enabled
        if show_all_onsets or show_spectral_data:
            geomean_threshold = spectral_config['geomean_threshold'] if spectral_config else None
            energy_labels = spectral_config['energy_labels'] if spectral_config else {}
            geomean_bands = spectral_config['geomean_bands'] if spectral_config else []
            freq_ranges = spectral_config.get('freq_ranges', {}) if spectral_config else {}
            has_sustain = spectral_config.get('min_sustain_ms') is not None if spectral_config else False
            show_badness = spectral_config.get('statistical_enabled', False) if spectral_config else False
            display_hints = spectral_config.get('display_hints', []) if spectral_config else []
            band_labels = [energy_labels.get(b, b.title()) for b in geomean_bands]

            # Header block
            print("\n      ALL DETECTED ONSETS - SPECTRAL ANALYSIS:")
            if geomean_threshold is not None:
                print(f"      Using GeoMean threshold: {geomean_threshold}")
            else:
                print("      No threshold filtering (showing all detections)")

            # Column legend: band names with Hz ranges
            col_parts = []
            for band_name in geomean_bands:
                label = energy_labels.get(band_name, band_name.title())
                freq_range = freq_ranges.get(band_name)
                if freq_range:
                    lo, hi = freq_range
                    lo_str = f"{lo/1000:.0f}k" if lo >= 1000 else f"{lo:.0f}"
                    hi_str = f"{hi/1000:.0f}k" if hi >= 1000 else f"{hi:.0f}"
                    col_parts.append(f"{label} ({lo_str}-{hi_str}Hz)")
                else:
                    col_parts.append(label)
            print(f"      Columns: Str=Onset Strength, Amp=Peak Amplitude, {', '.join(col_parts)}")

            # Stem-specific context from spectral config (sustain thresholds, etc.)
            for hint in display_hints:
                print(f"      {hint}")

            # GeoMean formula
            func = 'cbrt' if len(geomean_bands) >= 3 else 'sqrt'
            print(f"      GeoMean={func}({'*'.join(band_labels)}) - measures combined spectral energy")

            # Table header
            band_headers = ' '.join(f'{lbl:>8s}' for lbl in band_labels)
            header = f"\n      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {band_headers} {'Total':>8s} {'GeoMean':>8s}"
            if has_sustain:
                header += f" {'SustainMs':>10s}"
            if show_badness:
                header += f" {'Badness':>8s}"
            header += f" {'Status':>10s}"
            print(header)

            # Table rows
            for idx, data in enumerate(all_onset_data):
                is_real_hit = should_keep_onset(
                    geomean=data['geomean'],
                    sustain_ms=data.get('sustain_ms'),
                    geomean_threshold=geomean_threshold,
                    min_sustain_ms=spectral_config.get('min_sustain_ms') if spectral_config else None,
                    stem_type=stem_type
                )
                status = 'KEPT' if is_real_hit else 'REJECTED'
                
                band_values = ' '.join(f"{data.get(f'{b}_energy', 0.0):8.1f}" for b in geomean_bands)
                row = f"      {data['time']:8.3f} {data['strength']:6.3f} {data['amplitude']:6.3f} {band_values} "
                row += f"{data['total_energy']:8.1f} {data['geomean']:8.1f}"
                if has_sustain:
                    row += f" {data.get('sustain_ms', 0):10.1f}"
                if show_badness and 'badness_score' in data:
                    row += f" {data['badness_score']:8.3f}"
                row += f" {status:>10s}"
                print(row)

            # Show summary statistics
            print("\n      FILTERING SUMMARY:")
            if geomean_threshold is not None:
                kept_geomeans = [d['geomean'] for d in all_onset_data if d['geomean'] > geomean_threshold]
                rejected_geomeans = [d['geomean'] for d in all_onset_data if d['geomean'] <= geomean_threshold]
                print(f"        Pass 1 - GeoMean threshold: {geomean_threshold} (adjustable in midiconfig.yaml)")
                print(f"        Total onsets detected: {len(all_onset_data)}")
                print(f"        Pass 1 Kept (GeoMean > {geomean_threshold}): {len(kept_geomeans)}")
                print(f"        Pass 1 Rejected (GeoMean <= {geomean_threshold}): {len(rejected_geomeans)}")
                if kept_geomeans:
                    print(f"        Pass 1 Kept GeoMean range: {min(kept_geomeans):.1f} - {max(kept_geomeans):.1f}")
                if rejected_geomeans:
                    print(f"        Pass 1 Rejected GeoMean range: {min(rejected_geomeans):.1f} - {max(rejected_geomeans):.1f}")
                
                # Show statistical filter summary if enabled
                if show_badness and spectral_config.get('statistical_params'):
                    badness_threshold = spectral_config.get('badness_threshold', 0.6)
                    stats_params = spectral_config['statistical_params']
                    
                    print("\n        Pass 2 - Statistical Outlier Detection:")
                    print(f"        Badness threshold: {badness_threshold} (adjustable in midiconfig.yaml)")
                    print(f"        Median band ratio: {stats_params['median_ratio']:.3f}")
                    print(f"        Median Total energy: {stats_params['median_total']:.1f}")
                    
                    # Count pass 2 results
                    badness_scores = [d.get('badness_score', 0) for d in all_onset_data if 'badness_score' in d]
                    if badness_scores:
                        pass2_kept = [s for s in badness_scores if s <= badness_threshold]
                        pass2_rejected = [s for s in badness_scores if s > badness_threshold]
                        print(f"        Pass 2 Kept (Badness <= {badness_threshold}): {len(pass2_kept)}")
                        print(f"        Pass 2 Rejected (Badness > {badness_threshold}): {len(pass2_rejected)}")
                        if pass2_kept:
                            print(f"        Pass 2 Kept Badness range: {min(pass2_kept):.3f} - {max(pass2_kept):.3f}")
                        if pass2_rejected:
                            print(f"        Pass 2 Rejected Badness range: {min(pass2_rejected):.3f} - {max(pass2_rejected):.3f}")
            else:
                print("        No threshold filtering enabled")
                print(f"        Total onsets detected: {len(all_onset_data)} (all kept)")
                all_geomeans = [d['geomean'] for d in all_onset_data]
                if all_geomeans:
                    print(f"        GeoMean range: {min(all_geomeans):.1f} - {max(all_geomeans):.1f}")

            num_rejected = len(all_onset_data) - len(onset_times)
            print(f"\n    After spectral filtering: {len(onset_times)} hits (rejected {num_rejected} artifacts)")
        
        # Show decay analysis for cymbals if enabled and debug is on
        decay_analysis = filter_result.get('decay_analysis')
        if decay_analysis is not None and (show_all_onsets or show_spectral_data):
            decay_data = decay_analysis['data']
            window_sec = decay_analysis['window_sec']
            
            print("\n      DECAY PATTERN ANALYSIS (Pass 2 - Retriggering Filter):")
            print("      Checks if onset occurs during decay of previous hit")
            print(f"      Window: {window_sec}s, DecayRate=avg energy change (negative=decaying)")
            print(f"\n      {'Time':>8s} {'PrevHit':>8s} {'TimeDiff':>9s} {'DecayRate':>10s} {'Decaying':>9s} {'OwnDecay':>10s} {'Status':>10s}")
            
            for entry in decay_data:
                time = entry['time']
                prev_time = entry['prev_hit_time']
                time_diff = entry['time_since_prev']
                prev_decay_rate = entry['prev_decay_rate']
                prev_is_decaying = entry['prev_is_decaying']
                own_decay_rate = entry.get('own_decay_rate')
                is_retrigger = entry['is_retrigger']
                
                # Format output
                prev_str = f"{prev_time:.3f}" if prev_time is not None else "N/A"
                diff_str = f"{time_diff*1000:.1f}ms" if time_diff is not None else "N/A"
                decay_str = f"{prev_decay_rate:.3f}" if prev_decay_rate is not None else "N/A"
                decaying_str = "YES" if prev_is_decaying else ("NO" if prev_is_decaying is not None else "N/A")
                own_decay_str = f"{own_decay_rate:.3f}" if own_decay_rate is not None else "N/A"
                status = "REJECTED" if is_retrigger else "KEPT"
                
                print(f"      {time:8.3f} {prev_str:>8s} {diff_str:>9s} {decay_str:>10s} {decaying_str:>9s} {own_decay_str:>10s} {status:>10s}")
            
            # Show summary
            kept_count = sum(1 for e in decay_data if not e['is_retrigger'])
            rejected_count = sum(1 for e in decay_data if e['is_retrigger'])
            print("\n      DECAY FILTER SUMMARY:")
            print(f"        Decay window: {window_sec}s (adjustable via decay_filter_window_sec)")
            print(f"        Onsets after Pass 1: {len(decay_data)}")
            print(f"        Pass 2 Kept (independent hits): {kept_count}")
            print(f"        Pass 2 Rejected (retriggering): {rejected_count}")
    
    if len(onset_times) == 0:
        return {'events': [], 'all_onset_data': all_onset_data, 'spectral_config': spectral_config}
    
    # Step 5: Get MIDI note number
    note = getattr(drum_mapping, stem_type)
    
    # Step 6: Classify drum types (hihat open/closed/handclap)
    if stem_type == 'hihat' and detect_hihat_open:
        hihat_config = config.get('hihat', {})
        open_sustain_threshold = hihat_config.get('open_sustain_ms', 150)
        hihat_states = detect_hihat_state(
            audio_mono, sr, onset_times,
            sustain_durations=hihat_sustain_durations,
            open_sustain_threshold_ms=open_sustain_threshold,
            spectral_data=hihat_spectral_data,
            config=config
        )
    else:
        hihat_states = ['closed'] * len(onset_times)
    
    # Step 7: Calculate normalized values for velocity
    if stem_type in ['snare', 'kick', 'toms'] and stem_geomeans is not None and len(stem_geomeans) > 0:
        # For spectrally-filtered stems, use geometric mean
        normalized_values = normalize_values(stem_geomeans)
    elif stem_type == 'hihat' and len(onset_strengths) > 0:
        # For hihat, use onset strength (from detection) not peak amplitude
        # peak_amplitude is measured 10ms after detection time, which can be:
        # - Too early for loud hits (still in attack → reads as quiet)
        # - Too late for quiet hits (past peak → reads as loud)
        normalized_values = normalize_values(onset_strengths)
    elif len(peak_amplitudes) > 0:
        # For other stems, use peak amplitude
        normalized_values = normalize_values(peak_amplitudes)
    else:
        normalized_values = np.array([])
    
    # Step 8: Detect and classify tom pitches (if applicable)
    tom_classifications = None
    if stem_type == 'toms':
        tom_classifications = _detect_tom_pitches(audio_mono, sr, onset_times, config)
    
    # Step 8b: Detect and classify cymbal pitches (if applicable)
    cymbal_classifications = None
    if stem_type == 'cymbals':
        cymbal_classifications = _detect_cymbal_pitches(
            audio_mono, sr, onset_times, config,
            pan_positions=pan_positions,
            spectral_data=filtered_onset_data if 'filtered_onset_data' in locals() else None
        )
    
    # Step 8c: Detect and classify snare pitches (if applicable)
    snare_classifications = None
    if stem_type == 'snare':
        snare_classifications = _detect_snare_pitches(audio_mono, sr, onset_times, config)
    
    # Step 9: Create MIDI events
    # Pass sustain durations for cymbals (note duration) and hihats (foot-close timing)
    if stem_type == 'cymbals':
        sustain_durations_param = cymbal_sustain_durations
    elif stem_type == 'hihat':
        sustain_durations_param = hihat_sustain_durations
    else:
        sustain_durations_param = None
    
    events = _create_midi_events(
        onset_times,
        normalized_values,
        stem_type,
        note,
        min_velocity,
        max_velocity,
        hihat_states,
        tom_classifications,
        cymbal_classifications,
        snare_classifications,
        drum_mapping,
        config,
        sustain_durations=sustain_durations_param,
        spectral_data=filtered_onset_data
    )
    
    print(f"    Created {len(events)} MIDI events from {len(onset_times)} onsets")
    
    # Return dict with events and analysis data for sidecar v2
    return {
        'events': events,
        'all_onset_data': all_onset_data if 'all_onset_data' in locals() else [],
        'spectral_config': spectral_config if 'spectral_config' in locals() else None
    }
