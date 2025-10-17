"""
Learning Mode Module

Handles threshold learning from user-edited MIDI files.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Union, List
import yaml
import copy

# Import functional core helpers
from stems_to_midi_helpers import (
    ensure_mono,
    calculate_peak_amplitude,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean,
    calculate_sustain_duration,
    analyze_onset_spectral,
    time_to_sample
)

# Import MIDI reading
from stems_to_midi_midi import read_midi_notes

# Import data structures
from stems_to_midi_config import DrumMapping

__all__ = [
    'learn_threshold_from_midi',
    'save_calibrated_config'
]


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
        audio = ensure_mono(audio)
    
    # Analyze spectral properties of kept vs removed hits
    kept_geomeans = []
    removed_geomeans = []
    all_analysis = []  # Store all data for detailed output
    
    # Get match tolerance from config
    match_tolerance = config.get('learning_mode', {}).get('match_tolerance_sec', 0.05)
    
    for orig_time in original_times:
        # Check if this time exists in edited MIDI
        is_kept = any(abs(orig_time - edit_time) < match_tolerance for edit_time in edited_times)
        
        # Use unified spectral analysis helper (functional core)
        analysis = analyze_onset_spectral(audio, orig_time, sr, stem_type, config)
        
        if analysis is None:
            # Segment too short or invalid stem type, skip
            continue
        
        # Extract results from analysis
        onset_sample = analysis['onset_sample']
        primary_energy = analysis['primary_energy']
        secondary_energy = analysis['secondary_energy']
        total_energy = analysis['total_energy']
        geomean = analysis['geomean']
        sustain_duration = analysis['sustain_ms'] or 0.0
        
        # Calculate onset strength (similar to what detect_onsets does)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512, aggregate=np.median)
        onset_frame = librosa.time_to_frames(orig_time, sr=sr, hop_length=512)
        if onset_frame < len(onset_env):
            onset_strength = onset_env[onset_frame]
        else:
            onset_strength = 0.0
        
        # Calculate peak amplitude using functional core
        peak_amplitude = calculate_peak_amplitude(audio, onset_sample, sr, window_sec=0.01)
        
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

from pathlib import Path
from typing import Union, Dict
import numpy as np
import soundfile as sf
import librosa
import yaml

# Import functional core helpers
from stems_to_midi_helpers import (
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean
)

# Import config and detection modules
from stems_to_midi_config import DrumMapping
from stems_to_midi_midi import read_midi_notes


__all__ = ['learn_threshold_from_midi', 'save_calibrated_config']


# Placeholder functions - will be moved from stems_to_midi.py in Phase 5
def learn_threshold_from_midi(*args, **kwargs):
    """Placeholder - will be moved in Phase 5."""
    pass

def save_calibrated_config(*args, **kwargs):
    """Placeholder - will be moved in Phase 5."""
    pass
