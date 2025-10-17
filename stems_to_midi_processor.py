"""
Stem Processing Module

Handles the main processing pipeline for converting audio stems to MIDI events.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, List, Dict

# Import functional core helpers
from stems_to_midi_helpers import (
    ensure_mono,
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean,
    should_keep_onset
)

# Import detection functions
from stems_to_midi_detection import (
    detect_onsets,
    detect_tom_pitch,
    classify_tom_pitch,
    detect_hihat_state,
    estimate_velocity
)

# Import config structures
from stems_to_midi_config import DrumMapping

__all__ = [
    'process_stem_to_midi'
]


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
    print(f"  Processing {stem_type} from: {audio_path.name}")
    
    # Load audio (I/O - imperative shell)
    audio, sr = sf.read(str(audio_path))
    
    # Convert to mono if configured (functional core)
    if config['audio']['force_mono'] and audio.ndim == 2:
        audio = ensure_mono(audio)
        print(f"    Converted stereo to mono")
    
    # Check if audio is essentially silent (max amplitude below threshold)
    max_amplitude = np.max(np.abs(audio))
    print(f"    Max amplitude: {max_amplitude:.6f}")
    
    silence_threshold = config.get('audio', {}).get('silence_threshold', 0.001)
    if max_amplitude < silence_threshold:
        print(f"    Audio is silent, skipping...")
        return []
    
    # Detect onsets using config settings (or learning mode settings if enabled)
    learning_mode = config.get('learning_mode', {}).get('enabled', False)
    
    if learning_mode:
        # Ultra-sensitive detection for learning mode (catch everything)
        learning_config = config['learning_mode']
        onset_times, onset_strengths = detect_onsets(
            audio,
            sr,
            hop_length=config['onset_detection']['hop_length'],
            threshold=learning_config['learning_onset_threshold'],
            delta=learning_config['learning_delta'],
            wait=learning_config['learning_wait']
        )
        print(f"    Learning mode: Ultra-sensitive detection (threshold={learning_config['learning_onset_threshold']})")
    else:
        # Normal detection from config
        onset_config = config['onset_detection']
        
        # Check if this stem has custom onset detection settings
        stem_config = config.get(stem_type, {})
        
        # Use stem-specific settings if provided, otherwise fall back to global
        threshold = stem_config.get('onset_threshold')
        if threshold is None:
            threshold = onset_config['threshold']
        
        delta = stem_config.get('onset_delta')
        if delta is None:
            delta = onset_config['delta']
        
        wait = stem_config.get('onset_wait')
        if wait is None:
            wait = onset_config['wait']
        
        onset_times, onset_strengths = detect_onsets(
            audio,
            sr,
            hop_length=onset_config['hop_length'],
            threshold=threshold,
            delta=delta,
            wait=wait
        )
        
        # Log if using custom settings
        if (stem_config.get('onset_threshold') is not None or 
            stem_config.get('onset_delta') is not None or 
            stem_config.get('onset_wait') is not None):
            print(f"    {stem_type.capitalize()}-specific onset detection: threshold={threshold}, delta={delta}, wait={wait} (~{wait*11:.0f}ms min spacing)")
    
    print(f"    Found {len(onset_times)} hits (before filtering) -> MIDI note {getattr(drum_mapping, stem_type)}")
    
    # Calculate peak amplitudes for all onsets (functional core)
    peak_amplitudes = np.array([
        calculate_peak_amplitude(audio, int(onset_time * sr), sr, window_ms=10.0)
        for onset_time in onset_times
    ])
    
    # Initialize hihat data (will be populated during filtering if hihat)
    hihat_sustain_durations = None
    hihat_spectral_data = None
    
    # For snare, kick, toms, hihat, and cymbals: filter out artifacts/bleed by checking spectral content
    # This uses the functional core for all calculations
    if stem_type in ['snare', 'kick', 'toms', 'hihat', 'cymbals'] and len(onset_times) > 0:
        # Get spectral configuration for this stem type (functional core)
        spectral_config = get_spectral_config_for_stem(stem_type, config)
        freq_ranges = spectral_config['freq_ranges']
        energy_labels = spectral_config['energy_labels']
        geomean_threshold = spectral_config['geomean_threshold']
        min_sustain_ms = spectral_config['min_sustain_ms']
        
        # Storage for filtered results
        filtered_times = []
        filtered_strengths = []
        filtered_amplitudes = []
        filtered_geomeans = []
        filtered_sustains = []  # For hihat open/closed detection
        filtered_spectral = []  # For hihat handclap detection
        ratios_rejected = []
        
        # Store raw spectral data for ALL onsets (for debug output)
        all_onset_data = []
        
        for onset_time, strength, peak_amplitude in zip(onset_times, onset_strengths, peak_amplitudes):
            onset_sample = int(onset_time * sr)
            
            # Extract segment for spectral analysis
            peak_window_sec = config.get('audio', {}).get('peak_window_sec', 0.05)
            window_samples = int(peak_window_sec * sr)
            end_sample = min(onset_sample + window_samples, len(audio))
            segment = audio[onset_sample:end_sample]
            
            min_segment_length = config.get('audio', {}).get('min_segment_length', 512)
            if len(segment) < min_segment_length:
                continue
            
            # Calculate spectral energies (functional core)
            energies = calculate_spectral_energies(segment, sr, freq_ranges)
            primary_energy = energies.get('primary', 0.0)
            secondary_energy = energies.get('secondary', 0.0)
            low_energy = energies.get('low', 0.0)
            
            # Calculate sustain duration if needed for this stem type (functional core)
            sustain_duration = None
            if stem_type in ['hihat', 'cymbals']:
                # Get sustain parameters from config
                sustain_window_sec = config.get('audio', {}).get('sustain_window_sec', 0.2)
                envelope_threshold = config.get('audio', {}).get('envelope_threshold', 0.1)
                smooth_kernel = config.get('audio', {}).get('envelope_smooth_kernel', 51)
                
                sustain_duration = calculate_sustain_duration(
                    audio, onset_sample, sr,
                    window_ms=sustain_window_sec * 1000,  # Convert to ms
                    envelope_threshold=envelope_threshold,
                    smooth_kernel=smooth_kernel
                )
            
            # Calculate combined metrics (functional core)
            total_energy = primary_energy + secondary_energy
            spectral_ratio = (total_energy / low_energy) if low_energy > 0 else 100.0
            body_wire_geomean = calculate_geomean(primary_energy, secondary_energy)
            
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
            
            # Determine if this onset should be kept (functional core)
            is_real_hit = should_keep_onset(
                geomean=body_wire_geomean,
                sustain_ms=sustain_duration,
                geomean_threshold=geomean_threshold,
                min_sustain_ms=min_sustain_ms,
                stem_type=stem_type
            )
            
            # In learning mode, keep ALL detections but mark them
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
            elif not learning_mode:
                ratios_rejected.append(spectral_ratio)
        
        onset_times = np.array(filtered_times)
        onset_strengths = np.array(filtered_strengths)
        peak_amplitudes = np.array(filtered_amplitudes)
        stem_geomeans = np.array(filtered_geomeans)
        hihat_sustain_durations = filtered_sustains if stem_type == 'hihat' else None
        hihat_spectral_data = filtered_spectral if stem_type == 'hihat' else None
        
        # Show ALL onset data in chronological order with multiple ratios
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
        
        print(f"\n    After spectral filtering: {len(onset_times)} hits (rejected {len(ratios_rejected)} artifacts)")
    
    if len(onset_times) == 0:
        return []
    
    # Get MIDI note number
    note = getattr(drum_mapping, stem_type)
    
    # Special handling for hi-hat open/closed/handclap detection
    if stem_type == 'hihat' and detect_hihat_open:
        # Get open_sustain_ms threshold from config (used for open/closed classification)
        hihat_config = config.get('hihat', {})
        open_sustain_threshold = hihat_config.get('open_sustain_ms', 150)
        hihat_states = detect_hihat_state(
            audio, 
            sr, 
            onset_times, 
            sustain_durations=hihat_sustain_durations,
            open_sustain_threshold_ms=open_sustain_threshold,
            spectral_data=hihat_spectral_data,
            config=config
        )
    else:
        hihat_states = ['closed'] * len(onset_times)
    
    # Calculate velocity based on spectral energy for filtered stems, amplitude for others
    if stem_type in ['snare', 'kick', 'toms'] and len(stem_geomeans) > 0:
        # For spectrally-filtered stems, use geometric mean of primary and secondary energy (better correlates with loudness)
        max_geomean = np.max(stem_geomeans)
        if max_geomean > 0:
            normalized_values = stem_geomeans / max_geomean
        else:
            normalized_values = np.ones_like(stem_geomeans)
    elif len(peak_amplitudes) > 0:
        # For other stems, use peak amplitude
        max_amp = np.max(peak_amplitudes)
        if max_amp > 0:
            normalized_values = peak_amplitudes / max_amp
        else:
            normalized_values = np.ones_like(peak_amplitudes)
    else:
        normalized_values = np.array([])
    
    # For toms: detect pitch and classify into low/mid/high
    tom_classifications = None
    if stem_type == 'toms':
        tom_config = config.get('toms', {})
        enable_pitch = tom_config.get('enable_pitch_detection', True)
        
        if enable_pitch and len(onset_times) > 0:
            print(f"\n    Detecting tom pitches...")
            pitch_method = tom_config.get('pitch_method', 'yin')
            min_pitch = tom_config.get('min_pitch_hz', 60.0)
            max_pitch = tom_config.get('max_pitch_hz', 250.0)
            
            # Detect pitch for each tom hit
            detected_pitches = []
            for onset_time in onset_times:
                pitch = detect_tom_pitch(audio, sr, onset_time, method=pitch_method, 
                                        min_hz=min_pitch, max_hz=max_pitch)
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
            
            # Show detailed pitch table
            if len(onset_times) <= 20:  # Only show table if not too many
                print(f"\n      {'Time':>8s} {'Pitch(Hz)':>10s} {'Tom':>8s}")
                for i, (time, pitch, classification) in enumerate(zip(onset_times, detected_pitches, tom_classifications)):
                    tom_name = ['Low', 'Mid', 'High'][classification]
                    pitch_str = f"{pitch:.1f}" if pitch > 0 else "N/A"
                    print(f"      {time:8.3f} {pitch_str:>10s} {tom_name:>8s}")
    
    # Create MIDI events
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
