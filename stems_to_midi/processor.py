"""
Stem Processing Module

Handles the main processing pipeline for converting audio stems to MIDI events.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, List, Dict, Optional

# Import functional core helpers
from .helpers import (
    ensure_mono,
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean,
    should_keep_onset,
    analyze_onset_spectral,
    filter_onsets_by_spectral,
    calculate_velocities_from_features,
    normalize_values,
    estimate_velocity,
    classify_tom_pitch
)

# Import detection functions
from .detection import (
    detect_onsets,
    detect_tom_pitch,
    classify_tom_pitch,
    detect_hihat_state,
    estimate_velocity
)

# Import config structures
from .config import DrumMapping

__all__ = [
    'process_stem_to_midi'
]


def _load_and_validate_audio(
    audio_path: Union[str, Path],
    config: Dict,
    stem_type: str
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load audio file and validate it's usable.
    
    Helper function for process_stem_to_midi (imperative shell).
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        stem_type: Type of stem (for logging)
    
    Returns:
        Tuple of (audio, sample_rate) or (None, None) if invalid
    """
    print(f"  Processing {stem_type} from: {audio_path.name}")
    
    # Load audio (I/O)
    audio, sr = sf.read(str(audio_path))
    
    # Convert to mono if configured
    if config['audio']['force_mono'] and audio.ndim == 2:
        audio = ensure_mono(audio)
        print(f"    Converted stereo to mono")
    
    # Check if audio is essentially silent
    max_amplitude = np.max(np.abs(audio))
    print(f"    Max amplitude: {max_amplitude:.6f}")
    
    silence_threshold = config.get('audio', {}).get('silence_threshold', 0.001)
    if max_amplitude < silence_threshold:
        print(f"    Audio is silent, skipping...")
        return None, None
    
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
        # Normal detection - use stem-specific settings if provided
        threshold = stem_config.get('onset_threshold', onset_config['threshold'])
        delta = stem_config.get('onset_delta', onset_config['delta'])
        wait = stem_config.get('onset_wait', onset_config['wait'])
        
        return {
            'hop_length': onset_config['hop_length'],
            'threshold': threshold,
            'delta': delta,
            'wait': wait,
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
    
    print(f"\n    Detecting tom pitches...")
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
        print(f"    Warning: No valid pitches detected, all toms will use default (mid) note")
    
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


def _create_midi_events(
    onset_times: np.ndarray,
    normalized_values: np.ndarray,
    stem_type: str,
    note: int,
    min_velocity: int,
    max_velocity: int,
    hihat_states: List[str],
    tom_classifications: Optional[np.ndarray],
    drum_mapping: DrumMapping,
    config: Dict
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
        drum_mapping: MIDI note mapping
        config: Configuration dictionary
    
    Returns:
        List of MIDI event dictionaries
    """
    events = []
    
    for i, (time, value) in enumerate(zip(onset_times, normalized_values)):
        # Calculate velocity from normalized value
        velocity = estimate_velocity(value, min_velocity, max_velocity)
        
        # Adjust note for handclap, open hi-hat, or tom classification
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
        else:
            midi_note = note
        
        # Duration: until next hit or default
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - time
        else:
            default_duration = config.get('audio', {}).get('default_note_duration', 0.1)
            duration = default_duration
        
        max_duration = config.get('midi', {}).get('max_note_duration', 0.5)
        duration = min(duration, max_duration)
        
        events.append({
            'time': float(time),
            'note': int(midi_note),
            'velocity': int(velocity),
            'duration': float(duration)
        })
    
    return events


def process_stem_to_midi(
    audio_path: Union[str, Path],
    stem_type: str,
    drum_mapping: DrumMapping,
    config: Dict,
    onset_threshold: float = 0.3,
    min_velocity: int = 80,
    max_velocity: int = 110,
    detect_hihat_open: bool = True
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
    
    Returns:
        List of MIDI events: [{'time': float, 'note': int, 'velocity': int, 'duration': float}, ...]
    """
    # Step 1: Load and validate audio
    audio, sr = _load_and_validate_audio(audio_path, config, stem_type)
    if audio is None:
        return []
    
    # Step 2: Configure and detect onsets
    onset_params = _configure_onset_detection(config, stem_type)
    learning_mode = onset_params['learning_mode']
    
    onset_times, onset_strengths = detect_onsets(
        audio,
        sr,
        hop_length=onset_params['hop_length'],
        threshold=onset_params['threshold'],
        delta=onset_params['delta'],
        wait=onset_params['wait']
    )
    
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
        return []
    
    # Step 3: Calculate peak amplitudes for all onsets
    peak_amplitudes = np.array([
        calculate_peak_amplitude(audio, int(onset_time * sr), sr, window_ms=10.0)
        for onset_time in onset_times
    ])
    
    # For snare, kick, toms, hihat, and cymbals: filter out artifacts/bleed by checking spectral content
    # This uses the functional core for all calculations
    if stem_type in ['snare', 'kick', 'toms', 'hihat', 'cymbals'] and len(onset_times) > 0:
        # Use functional core helper for filtering
        filter_result = filter_onsets_by_spectral(
            onset_times,
            onset_strengths,
            peak_amplitudes,
            audio,
            sr,
            stem_type,
            config,
            learning_mode=learning_mode
        )
        
        # Extract filtered results
        onset_times = filter_result['filtered_times']
        onset_strengths = filter_result['filtered_strengths']
        peak_amplitudes = filter_result['filtered_amplitudes']
        stem_geomeans = filter_result['filtered_geomeans']
        hihat_sustain_durations = filter_result['filtered_sustains'] if stem_type == 'hihat' else None
        hihat_spectral_data = filter_result['filtered_spectral'] if stem_type == 'hihat' else None
        all_onset_data = filter_result['all_onset_data']
        spectral_config = filter_result['spectral_config']
    else:
        # No filtering for this stem type
        stem_geomeans = None
        hihat_sustain_durations = None
        hihat_spectral_data = None
        all_onset_data = []
        spectral_config = None
        
        # Show ALL onset data in chronological order with multiple ratios
        if spectral_config is not None and all_onset_data:
            geomean_threshold = spectral_config['geomean_threshold']
            energy_labels = spectral_config['energy_labels']
            stem_config = config.get(stem_type, {})
            
            print(f"\n      ALL DETECTED ONSETS - SPECTRAL ANALYSIS:")
            if geomean_threshold is not None:
                print(f"      Using GeoMean threshold: {geomean_threshold}")
            else:
                print(f"      No threshold filtering (showing all detections)")
            
            # Configure labels based on stem type
            if stem_type == 'snare':
                print(f"      Str=Onset Strength, Amp=Peak Amplitude, BodyE=Body Energy (150-400Hz), WireE=Wire Energy (2-8kHz)")
            elif stem_type == 'kick':
                print(f"      Str=Onset Strength, Amp=Peak Amplitude, FundE=Fundamental Energy (40-80Hz), BodyE=Body Energy (80-150Hz)")
            elif stem_type == 'toms':
                print(f"      Str=Onset Strength, Amp=Peak Amplitude, FundE=Fundamental Energy (60-150Hz), BodyE=Body Energy (150-400Hz)")
            elif stem_type == 'hihat':
                print(f"      Str=Onset Strength, Amp=Peak Amplitude, BodyE=Body Energy (500-2kHz), SizzleE=Sizzle Energy (6-12kHz), SustainMs=Sustain Duration")
                min_sustain_ms = stem_config.get('min_sustain_ms', 25)
                print(f"      Minimum sustain duration: {min_sustain_ms}ms (filters out handclap bleed)")
                open_sustain_ms = stem_config.get('open_sustain_ms', 150)
                print(f"      Open/Closed threshold: {open_sustain_ms}ms (>={open_sustain_ms}ms = open hihat)")
            elif stem_type == 'cymbals':
                print(f"      Str=Onset Strength, Amp=Peak Amplitude, BodyE=Body/Wash Energy (1-4kHz), BrillE=Brilliance/Attack Energy (4-10kHz), SustainMs=Sustain Duration")
                min_sustain_ms = stem_config.get('min_sustain_ms', 50)
                print(f"      Minimum sustain duration: {min_sustain_ms}ms")
            
            energy_label_1 = energy_labels['primary']
            energy_label_2 = energy_labels['secondary']
            print(f"      GeoMean=sqrt({energy_label_1}*{energy_label_2}) - measures combined spectral energy")
            
            # Header row - add SustainMs for cymbals and hihat
            if stem_type in ['cymbals', 'hihat']:
                print(f"\n      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {energy_label_1:>8s} {energy_label_2:>8s} {'Total':>8s} {'GeoMean':>8s} {'SustainMs':>10s} {'Status':>10s}")
            else:
                print(f"\n      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {energy_label_1:>8s} {energy_label_2:>8s} {'Total':>8s} {'GeoMean':>8s} {'Status':>10s}")
            
            for idx, data in enumerate(all_onset_data):
                # Re-calculate filtering decision for display using functional core
                is_real_hit = should_keep_onset(
                    geomean=data['body_wire_geomean'],
                    sustain_ms=data.get('sustain_ms'),
                    geomean_threshold=geomean_threshold,
                    min_sustain_ms=spectral_config.get('min_sustain_ms'),
                    stem_type=stem_type
                )
                
                status = 'KEPT' if is_real_hit else 'REJECTED'
                
                if stem_type in ['cymbals', 'hihat']:
                    sustain_str = f"{data.get('sustain_ms', 0):10.1f}"
                    print(f"      {data['time']:8.3f} {data['strength']:6.3f} {data['amplitude']:6.3f} {data['primary_energy']:8.1f} {data['secondary_energy']:8.1f} "
                          f"{data['total_energy']:8.1f} {data['body_wire_geomean']:8.1f} {sustain_str} {status:>10s}")
                else:
                    print(f"      {data['time']:8.3f} {data['strength']:6.3f} {data['amplitude']:6.3f} {data['primary_energy']:8.1f} {data['secondary_energy']:8.1f} "
                          f"{data['total_energy']:8.1f} {data['body_wire_geomean']:8.1f} {status:>10s}")
            
            # Show summary statistics
            print(f"\n      FILTERING SUMMARY:")
            if geomean_threshold is not None:
                kept_geomeans = [d['body_wire_geomean'] for d in all_onset_data if d['body_wire_geomean'] > geomean_threshold]
                rejected_geomeans = [d['body_wire_geomean'] for d in all_onset_data if d['body_wire_geomean'] <= geomean_threshold]
                print(f"        GeoMean threshold: {geomean_threshold} (adjustable in midiconfig.yaml)")
                print(f"        Total onsets detected: {len(all_onset_data)}")
                print(f"        Kept (GeoMean > {geomean_threshold}): {len(kept_geomeans)}")
                print(f"        Rejected (GeoMean <= {geomean_threshold}): {len(rejected_geomeans)}")
                if kept_geomeans:
                    print(f"        Kept GeoMean range: {min(kept_geomeans):.1f} - {max(kept_geomeans):.1f}")
                if rejected_geomeans:
                    print(f"        Rejected GeoMean range: {min(rejected_geomeans):.1f} - {max(rejected_geomeans):.1f}")
            else:
                print(f"        No threshold filtering enabled")
                print(f"        Total onsets detected: {len(all_onset_data)} (all kept)")
                all_geomeans = [d['body_wire_geomean'] for d in all_onset_data]
                if all_geomeans:
                    print(f"        GeoMean range: {min(all_geomeans):.1f} - {max(all_geomeans):.1f}")
            
            num_rejected = len(all_onset_data) - len(onset_times)
            print(f"\n    After spectral filtering: {len(onset_times)} hits (rejected {num_rejected} artifacts)")
    
    if len(onset_times) == 0:
        return []
    
    # Step 5: Get MIDI note number
    note = getattr(drum_mapping, stem_type)
    
    # Step 6: Classify drum types (hihat open/closed/handclap)
    if stem_type == 'hihat' and detect_hihat_open:
        hihat_config = config.get('hihat', {})
        open_sustain_threshold = hihat_config.get('open_sustain_ms', 150)
        hihat_states = detect_hihat_state(
            audio, sr, onset_times,
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
    elif len(peak_amplitudes) > 0:
        # For other stems, use peak amplitude
        normalized_values = normalize_values(peak_amplitudes)
    else:
        normalized_values = np.array([])
    
    # Step 8: Detect and classify tom pitches (if applicable)
    tom_classifications = None
    if stem_type == 'toms':
        tom_classifications = _detect_tom_pitches(audio, sr, onset_times, config)
    
    # Step 9: Create MIDI events
    events = _create_midi_events(
        onset_times,
        normalized_values,
        stem_type,
        note,
        min_velocity,
        max_velocity,
        hihat_states,
        tom_classifications,
        drum_mapping,
        config
    )
    
    return events
