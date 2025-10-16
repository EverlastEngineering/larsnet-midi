"""
Convert separated drum stems to MIDI tracks.

Analyzes each drum stem to detect onsets (hits) and converts them to MIDI notes
with velocity based on the hit intensity.

Architecture: Functional Core, Imperative Shell
- Pure functions in stems_to_midi_helpers.py (testable, no side effects)
- I/O and orchestration in this file (thin imperative shell)
"""

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from midiutil import MIDIFile
import mido
import argparse
from typing import Union, List, Tuple, Dict, Optional

# Import functional core (pure helper functions)
from stems_to_midi_helpers import (
    ensure_mono,
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean,
    should_keep_onset,
    normalize_values
)

# Import configuration module
from stems_to_midi_config import load_config, DrumMapping
from stems_to_midi_detection import (
    detect_onsets,
    detect_tom_pitch,
    classify_tom_pitch,
    detect_hihat_state,
    estimate_velocity
)
from stems_to_midi_midi import create_midi_file, read_midi_notes


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


def learn_threshold_from_midi(
    audio_path: Union[str, Path],
    original_midi_path: Union[str, Path],
    edited_midi_path: Union[str, Path],
    stem_type: str,
    config: Dict,
    drum_mapping: DrumMapping
) -> Dict[str, float]:
    """
    Learn optimal threshold by comparing original (all detections) with user-edited MIDI.
    
    Args:
        audio_path: Path to audio file
        original_midi_path: MIDI with ALL detections (from learning mode)
        edited_midi_path: User-edited MIDI (false positives removed)
        stem_type: Type of stem ('snare', etc.)
        config: Configuration dict
        drum_mapping: MIDI note mapping
    
    Returns:
        Dictionary with suggested thresholds
    """
    print(f"\n  Learning thresholds for {stem_type}...")
    
    # Get MIDI note for this stem
    target_note = getattr(drum_mapping, stem_type)
    
    # Read both MIDI files
    original_times = read_midi_notes(original_midi_path, target_note)
    edited_times = read_midi_notes(edited_midi_path, target_note)
    
    print(f"    Original detections: {len(original_times)}")
    print(f"    User kept: {len(edited_times)}")
    print(f"    User removed: {len(original_times) - len(edited_times)}")
    
    # Load audio and re-analyze all original detections
    audio, sr = sf.read(str(audio_path))
    
    if config['audio']['force_mono'] and audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    
    # Analyze spectral properties of kept vs removed hits
    kept_geomeans = []
    removed_geomeans = []
    all_analysis = []  # Store all data for detailed output
    
    # Get match tolerance from config
    match_tolerance = config.get('learning_mode', {}).get('match_tolerance_sec', 0.05)
    
    for orig_time in original_times:
        # Check if this time exists in edited MIDI
        is_kept = any(abs(orig_time - edit_time) < match_tolerance for edit_time in edited_times)
        
        # Analyze this onset
        onset_sample = int(orig_time * sr)
        peak_window_sec = config.get('learning_mode', {}).get('peak_window_sec', 0.05)
        window_samples = int(peak_window_sec * sr)
        end_sample = min(onset_sample + window_samples, len(audio))
        segment = audio[onset_sample:end_sample]
        
        min_segment_length = config.get('audio', {}).get('min_segment_length', 512)
        if len(segment) < min_segment_length:
            continue
        
        # Calculate onset strength (similar to what detect_onsets does)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512, aggregate=np.median)
        onset_frame = librosa.time_to_frames(orig_time, sr=sr, hop_length=512)
        if onset_frame < len(onset_env):
            onset_strength = onset_env[onset_frame]
        else:
            onset_strength = 0.0
        
        # Calculate peak amplitude using functional core
        peak_amplitude = calculate_peak_amplitude(audio, onset_sample, sr, window_sec=0.01)
        
        # Get spectral configuration for this stem type using functional core
        try:
            spectral_config = get_spectral_config_for_stem(stem_type, config)
        except ValueError:
            continue  # Skip unsupported stem types
        
        # Calculate spectral energies using functional core
        energies = calculate_spectral_energies(segment, sr, spectral_config['freq_ranges'])
        primary_energy = energies.get('primary', 0.0)
        secondary_energy = energies.get('secondary', 0.0)
        
        # Calculate geomean using functional core
        geomean = calculate_geomean(primary_energy, secondary_energy)
        
        # Calculate sustain duration for cymbals/hihat using functional core
        sustain_duration = 0.0
        if stem_type in ['cymbals', 'hihat']:
            sustain_window_sec = config.get('audio', {}).get('sustain_window_sec', 0.2)
            envelope_threshold = config.get('audio', {}).get('envelope_threshold', 0.1)
            smooth_kernel = config.get('audio', {}).get('envelope_smooth_kernel', 51)
            
            sustain_duration = calculate_sustain_duration(
                audio, onset_sample, sr,
                window_ms=sustain_window_sec * 1000,
                envelope_threshold=envelope_threshold,
                smooth_kernel=smooth_kernel
            )
        
        # Calculate total energy (already have primary and secondary from above)
        total_energy = primary_energy + secondary_energy
        
        # Store for detailed output with ALL variables
        analysis_data = {
            'time': orig_time,
            'strength': onset_strength,
            'amplitude': peak_amplitude,
            'primary_energy': primary_energy,
            'secondary_energy': secondary_energy,
            'total_energy': total_energy,
            'geomean': geomean,
            'is_kept': is_kept
        }
        
        # Add sustain duration for cymbals
        if stem_type == 'cymbals':
            analysis_data['sustain_ms'] = sustain_duration
        
        all_analysis.append(analysis_data)
        
        if is_kept:
            kept_geomeans.append(geomean)
        else:
            removed_geomeans.append(geomean)
    
    # For cymbals, also collect sustain durations
    suggested_sustain_threshold = None
    if stem_type == 'cymbals':
        kept_sustains = [d['sustain_ms'] for d in all_analysis if d['is_kept']]
        removed_sustains = [d['sustain_ms'] for d in all_analysis if not d['is_kept']]
        
        if kept_sustains and removed_sustains:
            min_kept_sustain = min(kept_sustains)
            max_removed_sustain = max(removed_sustains)
            suggested_sustain_threshold = (max_removed_sustain + min_kept_sustain) / 2.0
    
    # Calculate suggested threshold first (so we can show predictions)
    if kept_geomeans and removed_geomeans:
        min_kept = min(kept_geomeans)
        max_removed = max(removed_geomeans)
        
        # Suggest threshold halfway between max removed and min kept
        suggested_threshold = (max_removed + min_kept) / 2.0
        
        # Show detailed analysis of all hits with predictions
        if all_analysis:
            print(f"\n      DETAILED ANALYSIS OF ALL DETECTIONS:")
            
            # Get current config thresholds
            current_geomean_threshold = config.get(stem_type, {}).get('geomean_threshold', 0)
            current_sustain_threshold = config.get(stem_type, {}).get('min_sustain_ms', 0) if stem_type == 'cymbals' else None
            
            # Comprehensive header with all variables
            if stem_type == 'cymbals':
                print(f"      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {'BodyE':>8s} {'BrillE':>8s} {'Total':>8s} {'GeoMean':>8s} {'Sustain':>8s} {'User':>8s} {'Current':>8s} {'Suggest':>8s} {'Result':>10s}")
                print(f"      {'(s)':>8s} {'':>6s} {'':>6s} {'(1-4k)':>8s} {'(4-10k)':>8s} {'':>8s} {'':>8s} {'(ms)':>8s} {'Action':>8s} {'Config':>8s} {'Learn':>8s} {'':>10s}")
            elif stem_type == 'snare':
                print(f"      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {'BodyE':>8s} {'WireE':>8s} {'Total':>8s} {'GeoMean':>8s} {'User':>8s} {'Current':>8s} {'Suggest':>8s} {'Result':>10s}")
                print(f"      {'(s)':>8s} {'':>6s} {'':>6s} {'(150-400)':>8s} {'(2-8k)':>8s} {'':>8s} {'':>8s} {'Action':>8s} {'Config':>8s} {'Learn':>8s} {'':>10s}")
            elif stem_type == 'kick':
                print(f"      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {'FundE':>8s} {'BodyE':>8s} {'Total':>8s} {'GeoMean':>8s} {'User':>8s} {'Current':>8s} {'Suggest':>8s} {'Result':>10s}")
                print(f"      {'(s)':>8s} {'':>6s} {'':>6s} {'(40-80)':>8s} {'(80-150)':>8s} {'':>8s} {'':>8s} {'Action':>8s} {'Config':>8s} {'Learn':>8s} {'':>10s}")
            elif stem_type == 'toms':
                print(f"      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {'FundE':>8s} {'BodyE':>8s} {'Total':>8s} {'GeoMean':>8s} {'User':>8s} {'Current':>8s} {'Suggest':>8s} {'Result':>10s}")
                print(f"      {'(s)':>8s} {'':>6s} {'':>6s} {'(60-150)':>8s} {'(150-400)':>8s} {'':>8s} {'':>8s} {'Action':>8s} {'Config':>8s} {'Learn':>8s} {'':>10s}")
            elif stem_type == 'hihat':
                print(f"      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {'BodyE':>8s} {'SizzleE':>8s} {'Total':>8s} {'GeoMean':>8s} {'User':>8s} {'Current':>8s} {'Suggest':>8s} {'Result':>10s}")
                print(f"      {'(s)':>8s} {'':>6s} {'':>6s} {'(500-2k)':>8s} {'(6-12k)':>8s} {'':>8s} {'':>8s} {'Action':>8s} {'Config':>8s} {'Learn':>8s} {'':>10s}")
            
            correct_count = 0
            suggest_correct_count = 0
            for data in all_analysis:
                user_action = 'KEPT' if data['is_kept'] else 'REMOVED'
                
                # Check against CURRENT config thresholds
                if stem_type == 'cymbals' and current_sustain_threshold is not None:
                    current_would_be = 'KEPT' if (data['geomean'] > current_geomean_threshold and 
                                                   data.get('sustain_ms', 0) > current_sustain_threshold) else 'REMOVED'
                else:
                    current_would_be = 'KEPT' if data['geomean'] > current_geomean_threshold else 'REMOVED'
                
                # Check against SUGGESTED thresholds (from learning)
                if stem_type == 'cymbals' and suggested_sustain_threshold is not None:
                    suggest_would_be = 'KEPT' if (data['geomean'] > suggested_threshold and 
                                                   data.get('sustain_ms', 0) > suggested_sustain_threshold) else 'REMOVED'
                else:
                    suggest_would_be = 'KEPT' if data['geomean'] > suggested_threshold else 'REMOVED'
                
                # Check if current config classifies correctly
                is_correct = (user_action == current_would_be)
                if is_correct:
                    correct_count += 1
                
                # Check if suggested thresholds would classify correctly
                is_suggest_correct = (user_action == suggest_would_be)
                if is_suggest_correct:
                    suggest_correct_count += 1
                
                # Show result based on suggested vs current
                if is_correct and is_suggest_correct:
                    result = '✓ Both OK'
                elif is_correct and not is_suggest_correct:
                    result = '✓ Cur OK'
                elif not is_correct and is_suggest_correct:
                    result = '✓ Sug OK'
                else:
                    result = '✗ Both Bad'
                
                # Print with all variables
                if stem_type == 'cymbals':
                    print(f"      {data['time']:8.3f} {data['strength']:6.3f} {data['amplitude']:6.3f} {data['primary_energy']:8.1f} {data['secondary_energy']:8.1f} "
                          f"{data['total_energy']:8.1f} {data['geomean']:8.1f} {data.get('sustain_ms', 0):8.1f} {user_action:>8s} {current_would_be:>8s} {suggest_would_be:>8s} {result:>10s}")
                else:
                    print(f"      {data['time']:8.3f} {data['strength']:6.3f} {data['amplitude']:6.3f} {data['primary_energy']:8.1f} {data['secondary_energy']:8.1f} "
                          f"{data['total_energy']:8.1f} {data['geomean']:8.1f} {user_action:>8s} {current_would_be:>8s} {suggest_would_be:>8s} {result:>10s}")
            
            current_accuracy = (correct_count / len(all_analysis)) * 100
            suggest_accuracy = (suggest_correct_count / len(all_analysis)) * 100
            print(f"\n      Current config accuracy: {correct_count}/{len(all_analysis)} ({current_accuracy:.1f}%)")
            print(f"      Suggested threshold accuracy: {suggest_correct_count}/{len(all_analysis)} ({suggest_accuracy:.1f}%)")
        
        print(f"\n    Analysis:")
        print(f"      Kept hits - GeoMean range: {min_kept:.1f} - {max(kept_geomeans):.1f}")
        print(f"      Removed hits - GeoMean range: {min(removed_geomeans):.1f} - {max_removed:.1f}")
        print(f"      Suggested GeoMean threshold: {suggested_threshold:.1f}")
        print(f"      (Midpoint between max removed ({max_removed:.1f}) and min kept ({min_kept:.1f}))")
        
        # Add sustain threshold info for cymbals
        if stem_type == 'cymbals' and suggested_sustain_threshold is not None:
            kept_sustains = [d['sustain_ms'] for d in all_analysis if d['is_kept']]
            removed_sustains = [d['sustain_ms'] for d in all_analysis if not d['is_kept']]
            print(f"\n      Kept hits - Sustain range: {min(kept_sustains):.1f}ms - {max(kept_sustains):.1f}ms")
            print(f"      Removed hits - Sustain range: {min(removed_sustains):.1f}ms - {max(removed_sustains):.1f}ms")
            print(f"      Suggested SustainMs threshold: {suggested_sustain_threshold:.1f}ms")
            print(f"      (Midpoint between max removed ({max(removed_sustains):.1f}ms) and min kept ({min(kept_sustains):.1f}ms))")
        
        result = {
            'geomean_threshold': suggested_threshold,
            'kept_range': (min_kept, max(kept_geomeans)),
            'removed_range': (min(removed_geomeans), max_removed)
        }
        
        # Add sustain threshold for cymbals
        if stem_type == 'cymbals' and suggested_sustain_threshold is not None:
            result['min_sustain_ms'] = suggested_sustain_threshold
        
        return result
    else:
        print("    Not enough data to suggest threshold")
        return {}


def save_calibrated_config(config: Dict, learned_thresholds: Dict[str, Dict], output_path: Union[str, Path]):
    """
    Save a new config file with learned thresholds.
    
    Args:
        config: Original configuration
        learned_thresholds: Dictionary mapping stem types to learned threshold dicts
        output_path: Where to save the calibrated config
    """
    # Deep copy config
    import copy
    calibrated_config = copy.deepcopy(config)
    
    # Update thresholds
    for stem_type, thresholds in learned_thresholds.items():
        if stem_type in calibrated_config and 'geomean_threshold' in thresholds:
            calibrated_config[stem_type]['geomean_threshold'] = thresholds['geomean_threshold']
            print(f"  Updated {stem_type} geomean_threshold: {thresholds['geomean_threshold']:.1f}")
    
    # Disable learning mode in the new config
    if 'learning_mode' in calibrated_config:
        calibrated_config['learning_mode']['enabled'] = False
    
    # Save
    with open(output_path, 'w') as f:
        yaml.dump(calibrated_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n  Saved calibrated config to: {output_path}")
    print(f"  You can now use this config for production MIDI conversion!")


def stems_to_midi(
    stems_dir: Union[str, Path],
    output_dir: Union[str, Path],
    onset_threshold: float = 0.3,
    min_velocity: int = 80,
    max_velocity: int = 110,
    tempo: float = 120.0,
    detect_hihat_open: bool = False,
    stems_to_process: List[str] = None
):
    """
    Convert separated drum stems to MIDI files.
    
    Args:
        stems_dir: Directory containing separated stems (subdirs: kick/, snare/, etc.)
        output_dir: Directory to save MIDI files
        onset_threshold: Threshold for onset detection (0-1, lower = more sensitive)
        min_velocity: Minimum MIDI velocity
        max_velocity: Maximum MIDI velocity
        tempo: Tempo in BPM (for MIDI timing)
        detect_hihat_open: Try to detect open hi-hat hits
        stems_to_process: List of stem types to process (default: all)
    """
    stems_dir = Path(stems_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    try:
        config = load_config()
        print(f"Loaded configuration from: midiconfig.yaml")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Creating default config file...")
        # Config file will be created by default in the same directory
        raise
    
    # Default stems to process
    if stems_to_process is None:
        stems_to_process = ['kick', 'snare', 'toms', 'hihat', 'cymbals']
    
    # Initialize drum mapping
    drum_mapping = DrumMapping()
    
    # Find all audio file directories (new structure: stems_dir/input_name/)
    # Each subdirectory should contain files like input_name-kick.wav, input_name-snare.wav, etc.
    audio_dirs = [d for d in stems_dir.iterdir() if d.is_dir()]
    
    if not audio_dirs:
        raise RuntimeError(f"No subdirectories found in {stems_dir}")
    
    audio_files_to_process = []
    for audio_dir in audio_dirs:
        # Check if this directory has the stem files we need
        has_stems = False
        for stem_type in stems_to_process:
            expected_file = audio_dir / f"{audio_dir.name}-{stem_type}.wav"
            if expected_file.exists():
                has_stems = True
                break
        
        if has_stems:
            audio_files_to_process.append(audio_dir)
    
    if not audio_files_to_process:
        raise RuntimeError(f"No valid stem directories found in {stems_dir} with expected naming pattern (e.g., name/name-kick.wav)")
    
    print(f"Processing {len(audio_files_to_process)} file(s)...")
    print(f"Settings:")
    print(f"  Onset threshold: {onset_threshold}")
    print(f"  Velocity range: {min_velocity}-{max_velocity}")
    print(f"  Tempo: {tempo} BPM")
    print(f"  Detect open hi-hat: {detect_hihat_open}")
    print()
    
    for audio_dir in audio_files_to_process:
        base_name = audio_dir.name
        print(f"Processing: {base_name}")
        
        events_by_stem = {}
        
        # Process each stem type
        for stem_type in stems_to_process:
            stem_file = audio_dir / f"{base_name}-{stem_type}.wav"
            
            if not stem_file.exists():
                print(f"  Warning: {stem_type} file not found, skipping...")
                continue
            
            # For hihat, check config for detect_open setting (can be overridden by command-line flag)
            hihat_detect = detect_hihat_open
            if stem_type == 'hihat' and not detect_hihat_open:
                # If not set via command-line, check config
                hihat_detect = config.get('hihat', {}).get('detect_open', False)
            
            events = process_stem_to_midi(
                stem_file,
                stem_type,
                drum_mapping,
                config,
                onset_threshold=onset_threshold,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                detect_hihat_open=hihat_detect
            )
            
            if events:
                events_by_stem[stem_type] = events
        
        # Create MIDI file
        if events_by_stem:
            # Add suffix for learning mode
            learning_mode = config.get('learning_mode', {}).get('enabled', False)
            if learning_mode:
                suffix = config['learning_mode']['learning_midi_suffix']
                midi_path = output_dir / f"{base_name}{suffix}.mid"
            else:
                midi_path = output_dir / f"{base_name}.mid"
            
            create_midi_file(
                events_by_stem,
                midi_path,
                tempo=tempo,
                track_name=f"Drums - {base_name}",
                config=config
            )
            
            if learning_mode:
                print(f"  Saved LEARNING MIDI: {midi_path}")
                print(f"  ** Load in DAW, delete false positives (velocity=1 hits), save as: {audio_file.stem}_edited.mid **\n")
            else:
                print(f"  Saved: {midi_path}\n")
        else:
            print(f"  No events detected, skipping MIDI creation\n")
    
    print(f"Done! MIDI files saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert separated drum stems to MIDI tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/
  
  # More sensitive onset detection
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/ -t 0.2
  
  # Less sensitive (fewer false positives)
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/ -t 0.5
  
  # Full velocity range
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --min-vel 1 --max-vel 127
  
  # Specific tempo
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --tempo 140

MIDI Note Mapping (General MIDI):
  Kick:    36 (C1)  - Bass Drum 1
  Snare:   38 (D1)  - Acoustic Snare
  Toms:    45 (A1)  - Low Tom
  Hi-Hat:  42 (F#1) - Closed Hi-Hat
           46 (A#1) - Open Hi-Hat
  Cymbals: 49 (C#2) - Crash Cymbal 1
        """
    )
    
    parser.add_argument('-i', '--input_dir', type=str, required=False,
                        help="Directory containing separated stems (must have kick/, snare/, etc. subdirectories).")
    parser.add_argument('-o', '--output_dir', type=str, default='midi_output',
                        help="Directory to save MIDI files (default: midi_output).")
    parser.add_argument('-t', '--threshold', type=float, default=0.3,
                        help="Onset detection threshold (0-1). Lower = more sensitive (default: 0.3).")
    parser.add_argument('--min-vel', type=int, default=40,
                        help="Minimum MIDI velocity (1-127, default: 40).")
    parser.add_argument('--max-vel', type=int, default=127,
                        help="Maximum MIDI velocity (1-127, default: 127).")
    parser.add_argument('--tempo', type=float, default=None,
                        help="Tempo in BPM for MIDI timing (default: read from midiconfig.yaml).")
    parser.add_argument('--detect-hihat-open', action='store_true',
                        help="Enable open/closed hi-hat detection (disabled by default - most hits will be closed).")
    parser.add_argument('--stems', type=str, nargs='+',
                        choices=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
                        help="Specific stems to process (default: all).")
    
    # Learning mode arguments
    learning_group = parser.add_argument_group('Threshold Learning Mode')
    learning_group.add_argument('--learn', action='store_true',
                               help="Enable learning mode (exports all detections, rejected=velocity 1).")
    learning_group.add_argument('--learn-from-midi', type=str, nargs=3, metavar=('AUDIO', 'ORIG_MIDI', 'EDITED_MIDI'),
                               help="Learn thresholds from edited MIDI. Args: audio_file original_midi edited_midi")
    learning_group.add_argument('--learn-stem', type=str, default='snare',
                               choices=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
                               help="Stem type for learning (default: snare).")
    
    args = parser.parse_args()
    
    # Validate that -i is provided unless using --learn-from-midi
    if not args.learn_from_midi and not args.input_dir:
        parser.error("-i/--input_dir is required unless using --learn-from-midi")
    
    # Load config to get default tempo if not specified
    if args.tempo is None:
        config = load_config()
        args.tempo = config['midi']['default_tempo']
        print(f"Using tempo from config: {args.tempo} BPM\n")
    
    # Validate
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be between 0.0 and 1.0")
    if not (1 <= args.min_vel <= 127):
        parser.error("--min-vel must be between 1 and 127")
    if not (1 <= args.max_vel <= 127):
        parser.error("--max-vel must be between 1 and 127")
    if args.min_vel > args.max_vel:
        parser.error("--min-vel cannot be greater than --max-vel")
    
    # Handle learning mode workflows
    if args.learn_from_midi:
        # Learn thresholds from edited MIDI
        from pathlib import Path
        
        audio_file = Path(args.learn_from_midi[0])
        orig_midi = Path(args.learn_from_midi[1])
        edited_midi = Path(args.learn_from_midi[2])
        
        if not audio_file.exists():
            parser.error(f"Audio file not found: {audio_file}")
        if not orig_midi.exists():
            parser.error(f"Original MIDI not found: {orig_midi}")
        if not edited_midi.exists():
            parser.error(f"Edited MIDI not found: {edited_midi}")
        
        # Load config
        config = load_config()
        drum_mapping = DrumMapping()
        
        # Learn thresholds
        learned = learn_threshold_from_midi(
            audio_file,
            orig_midi,
            edited_midi,
            args.learn_stem,
            config,
            drum_mapping
        )
        
        if learned:
            # Save calibrated config
            output_config = Path(config['learning_mode']['calibrated_config_output'])
            save_calibrated_config(config, {args.learn_stem: learned}, output_config)
    
    elif args.learn:
        # Enable learning mode in config temporarily
        import tempfile
        import shutil
        
        # Load and modify config
        config = load_config()
        config['learning_mode']['enabled'] = True
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            temp_config = f.name
        
        try:
            # Run conversion with learning mode
            print("=== LEARNING MODE ENABLED ===")
            print("All detections will be exported. Rejected hits have velocity=1.")
            print("Load MIDI in DAW, delete false positives, save as *_edited.mid")
            print("Then run: python stems_to_midi.py --learn-from-midi <audio> <original_midi> <edited_midi>\n")
            
            stems_to_midi(
                stems_dir=args.input_dir,
                output_dir=args.output_dir,
                onset_threshold=args.threshold,
                min_velocity=args.min_vel,
                max_velocity=args.max_vel,
                tempo=args.tempo,
                detect_hihat_open=args.detect_hihat_open,
                stems_to_process=args.stems
            )
        finally:
            # Clean up temp config
            Path(temp_config).unlink()
    
    else:
        # Normal mode
        stems_to_midi(
            stems_dir=args.input_dir,
            output_dir=args.output_dir,
            onset_threshold=args.threshold,
            min_velocity=args.min_vel,
            max_velocity=args.max_vel,
            tempo=args.tempo,
            detect_hihat_open=args.detect_hihat_open,
            stems_to_process=args.stems
        )
