"""
Onset Filtering and Analysis

Pure helper functions for onset filtering based on spectral content.

This module contains the core filtering logic that analyzes audio segments
at onset times and determines which onsets should be kept or filtered.

Functions:
- mark_reverb_continuations: Mark reverb continuation events in onset data
- filter_onsets_by_spectral: Filter onsets by spectral content
- analyze_onset_spectral: Perform complete spectral analysis for a single onset
"""

import numpy as np
from typing import Dict, List, Optional

from .audio_utils import (
    time_to_sample,
    extract_audio_segment,
    calculate_spectral_energies,
    calculate_sustain_duration,
    calculate_amplitude_at_time,
    calculate_attack_sharpness,
    calculate_envelope_continuity,
    calculate_peak_prominence,
    calculate_spectral_centroid,
    calculate_spectral_flux,
    detect_pitch,
    calculate_gap_from_previous,
    analyze_cymbal_decay_pattern,
)
from .spectral_utils import (
    get_spectral_config_for_stem,
    calculate_geomean,
    calculate_statistical_params,
    calculate_badness_score,
    should_keep_onset,
)


def mark_reverb_continuations(
    onset_data_list: List[Dict],
    time_margin_ms: float = 5.0,
    amplitude_margin: float = 0.001,
    attack_sharpness_threshold: float = 0.2
) -> List[Dict]:
    """
    Mark reverb continuation events in onset data.
    
    Reverb continuations are artifacts where peak-hold detection splits
    a single reverb/echo envelope into multiple events. Characteristics:
    - Next event starts exactly when previous ends (within time_margin_ms)
    - Amplitude continuity: start matches previous end (within amplitude_margin)
    - Low attack sharpness (< threshold): reverb/echo have smooth envelopes, not sharp attacks
    
    Note: Amplitude can increase or decrease with complex reverb (reflections building up).
    
    These events are marked as 'REVERB_CONTINUATION' status to preserve
    data while allowing MIDI export to filter them.
    
    Pure function - modifies input list in place but returns it for chaining.
    
    Args:
        onset_data_list: List of onset dicts (must have time, duration_sec, 
                        amplitude, amplitude_at_start, amplitude_at_end, 
                        attack_sharpness, status)
        time_margin_ms: Maximum gap between events (default 5ms)
        amplitude_margin: Maximum amplitude difference (default 0.001)
        attack_sharpness_threshold: Minimum attack sharpness for real hits (default 0.2)
    
    Returns:
        Modified onset_data_list with 'REVERB_CONTINUATION' status set
    """
    if len(onset_data_list) < 2:
        return onset_data_list
    
    # Sort by time to ensure sequential processing
    onset_data_list.sort(key=lambda e: e['time'])
    
    time_margin_sec = time_margin_ms / 1000.0
    
    for i in range(1, len(onset_data_list)):
        prev = onset_data_list[i - 1]
        curr = onset_data_list[i]
        
        # Skip if current event already rejected, or if previous is rejected
        # (but allow previous to be REVERB_CONTINUATION to catch full decay chains)
        if curr.get('status') != 'KEPT':
            continue
        if prev.get('status') not in ('KEPT', 'REVERB_CONTINUATION'):
            continue
        
        # Check required fields exist
        if ('duration_sec' not in prev or 'amplitude_at_end' not in prev or 
            'amplitude_at_start' not in curr or 'amplitude' not in prev or 
            'amplitude' not in curr):
            continue
        
        # Calculate timing
        prev_end_time = prev['time'] + prev['duration_sec']
        gap = curr['time'] - prev_end_time
        
        # Check continuation conditions
        is_adjacent = abs(gap) <= time_margin_sec
        
        # Amplitude continuity - envelope connects smoothly
        prev_end_amp = prev['amplitude_at_end']
        curr_start_amp = curr['amplitude_at_start']
        amp_diff = abs(curr_start_amp - prev_end_amp)
        is_amplitude_continuous = amp_diff <= amplitude_margin
        
        # Attack sharpness check - real hits have sharp attacks (>= threshold)
        # Reverb/echo tails have smooth envelopes (< threshold)
        # Note: Complex reverb can have increasing amplitude (reflections building up)
        curr_attack_sharpness = curr.get('attack_sharpness')
        is_smooth_envelope = (curr_attack_sharpness is not None and 
                             curr_attack_sharpness < attack_sharpness_threshold)
        
        # Mark as reverb continuation if all conditions met
        if is_adjacent and is_amplitude_continuous and is_smooth_envelope:
            curr['status'] = 'REVERB_CONTINUATION'
    
    return onset_data_list


def filter_onsets_by_spectral(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    peak_amplitudes: np.ndarray,
    audio: np.ndarray,
    sr: int,
    stem_type: str,
    config: Dict,
    learning_mode: bool = False,
    durations: Optional[np.ndarray] = None
) -> Dict:
    """
    Filter onsets by spectral content and analyze each onset.
    
    Pure function - no side effects, no I/O.
    
    Detection Output Contract (Producer):
        This function PRODUCES SpectralOnsetData for each kept onset.
        Contract defined in midi_types.py - see SpectralOnsetData TypedDict.
        filtered_onset_data contains full analysis for all KEPT onsets.
        Consumers: detect_hihat_state(), learning.py, processing_shell.py
    
    Args:
        onset_times: Array of onset times in seconds
        onset_strengths: Array of onset strengths (0-1)
        peak_amplitudes: Array of peak amplitudes
        audio: Audio signal (mono)
        sr: Sample rate
        stem_type: Type of stem ('kick', 'snare', etc.)
        config: Configuration dictionary
        learning_mode: If True, keep all onsets (don't filter)
        durations: Optional array of event durations in seconds. If provided,
                  passed to analyze_onset_spectral for Phase 2 metadata.
    
    Returns:
        Dictionary with:
        - filtered_times: np.ndarray
        - filtered_strengths: np.ndarray
        - filtered_amplitudes: np.ndarray
        - filtered_geomeans: np.ndarray
        - filtered_sustains: List (when has_sustain_analysis is True)
        - filtered_spectral: List (when has_spectral_data is True)
        - filtered_onset_data: List[SpectralOnsetData] - contract-compliant spectral data
        - all_onset_data: List[Dict] (analysis for all onsets, for debugging)
        - spectral_config: Dict (configuration used)
        - decay_analysis: Dict or None (when enable_decay_filter ran)
    """
    if len(onset_times) == 0:
        return {
            'filtered_times': np.array([]),
            'filtered_strengths': np.array([]),
            'filtered_amplitudes': np.array([]),
            'filtered_geomeans': np.array([]),
            'filtered_sustains': [],
            'filtered_spectral': [],
            'filtered_onset_data': [],  # Full spectral data for KEPT onsets
            'all_onset_data': [],
            'spectral_config': None
        }
    
    # Get spectral configuration for this stem type
    spectral_config = get_spectral_config_for_stem(stem_type, config)
    geomean_threshold = spectral_config['geomean_threshold']
    min_sustain_ms = spectral_config['min_sustain_ms']
    energy_labels = spectral_config['energy_labels']
    geomean_bands = spectral_config['geomean_bands']
    
    # Storage for filtered results
    filtered_times = []
    filtered_strengths = []
    filtered_amplitudes = []
    filtered_geomeans = []
    filtered_sustains = []  # For stems with has_sustain_analysis
    filtered_spectral = []  # For stems with has_spectral_data
    filtered_onset_data = []  # Full spectral data for KEPT onsets (for Detection Output Contract)
    
    # Store raw spectral data for ALL onsets (for debug output)
    all_onset_data = []
    
    # Handle optional durations parameter (backward compatibility)
    if durations is None:
        durations = [None] * len(onset_times)
    
    for onset_time, strength, peak_amplitude, duration in zip(onset_times, onset_strengths, peak_amplitudes, durations):
        # Use unified spectral analysis helper (now with duration)
        analysis = analyze_onset_spectral(audio, onset_time, sr, stem_type, config, duration=duration)
        
        if analysis is None:
            # Segment too short, skip
            continue
        
        # Extract results from analysis (domain-specific band names)
        low_energy = analysis['low_energy']
        total_energy = analysis['total_energy']
        geomean = analysis['geomean']
        sustain_duration = analysis['sustain_ms']
        spectral_ratio = analysis['spectral_ratio']
        
        # Extract band energies using geomean_bands order
        band_energies = {band: analysis.get(f'{band}_energy', 0.0) for band in geomean_bands}
        
        # Phase 2: Calculate extended metadata (if duration available)
        amplitude_at_start = None
        amplitude_at_end = None
        attack_sharpness = None
        envelope_continuity = None
        peak_prominence = None
        spectral_centroid_hz = None
        spectral_flux_value = None
        detected_pitch = None
        gap_from_previous = None
        
        if duration is not None:
            # Amplitude at start and end
            amplitude_at_start = calculate_amplitude_at_time(audio, onset_time, sr, window_ms=5.0)
            amplitude_at_end = calculate_amplitude_at_time(audio, onset_time + duration, sr, window_ms=5.0)
            
            # Attack characteristics
            attack_sharpness = calculate_attack_sharpness(audio, onset_time, duration, sr)
            envelope_continuity = calculate_envelope_continuity(audio, onset_time, duration, sr)
            
            # Peak prominence
            peak_prominence = calculate_peak_prominence(audio, onset_time, sr)
            
            # Spectral features
            spectral_centroid_hz = calculate_spectral_centroid(audio, onset_time, sr)
            spectral_flux_value = calculate_spectral_flux(audio, onset_time, sr)
            
            # Pitch detection - enabled for toms (fundamental is strong, good for clustering)
            # Disabled for other stems (too slow, spectral_centroid_hz is sufficient)
            # Note: Tom fundamental is typically 40-250Hz, using 40-500 range to capture fundamentals
            if stem_type == 'toms':
                detected_pitch = detect_pitch(audio, onset_time, sr, fmin=40.0, fmax=500.0)
            else:
                detected_pitch = None
            
            # Gap from previous onset
            if len(filtered_times) > 0:
                gap_from_previous = calculate_gap_from_previous(onset_time, filtered_times[-1])
        
        # Determine if this onset should be kept
        is_real_hit = should_keep_onset(
            geomean=geomean,
            sustain_ms=sustain_duration,
            geomean_threshold=geomean_threshold,
            min_sustain_ms=min_sustain_ms,
            filter_mode=spectral_config['filter_mode'],
            strength=strength,
            min_strength_threshold=spectral_config.get('min_strength_threshold')
        )
        
        # Store all data for this onset (for debug output AND sidecar v2)
        # Uses domain-specific band names (body_energy, wire_energy, etc.)
        onset_data = {
            'time': onset_time,
            'strength': strength,
            'amplitude': peak_amplitude,
            'low_energy': low_energy,
            'ratio': spectral_ratio,
            'total_energy': total_energy,
            'geomean': geomean,
            'geomean_bands': geomean_bands,
            'status': 'KEPT' if (learning_mode or is_real_hit) else 'FILTERED'
        }
        
        # Add domain-specific band energies and labels
        for band_name in geomean_bands:
            onset_data[f'{band_name}_energy'] = band_energies[band_name]
            onset_data[f'{band_name}_label'] = energy_labels.get(band_name, band_name.title())
        
        if sustain_duration is not None:
            onset_data['sustain_ms'] = sustain_duration
        
        # Add Phase 2 metadata (if calculated)
        if duration is not None:
            onset_data['duration_sec'] = duration
        if amplitude_at_start is not None:
            onset_data['amplitude_at_start'] = amplitude_at_start
        if amplitude_at_end is not None:
            onset_data['amplitude_at_end'] = amplitude_at_end
        if attack_sharpness is not None:
            onset_data['attack_sharpness'] = attack_sharpness
        if envelope_continuity is not None:
            onset_data['envelope_continuity'] = envelope_continuity
        if peak_prominence is not None:
            onset_data['peak_prominence'] = peak_prominence
        if spectral_centroid_hz is not None:
            onset_data['spectral_centroid_hz'] = spectral_centroid_hz
        if spectral_flux_value is not None:
            onset_data['spectral_flux'] = spectral_flux_value
        if detected_pitch is not None:
            onset_data['pitch_hz'] = detected_pitch
        if gap_from_previous is not None:
            onset_data['gap_from_previous_sec'] = gap_from_previous
        
        all_onset_data.append(onset_data)
        
        # In learning mode, keep ALL detections
        if learning_mode or is_real_hit:
            filtered_times.append(onset_time)
            filtered_strengths.append(strength)
            filtered_amplitudes.append(peak_amplitude)
            filtered_geomeans.append(geomean)
            # Store sustain duration and spectral data for stems that analyze sustain
            if spectral_config['has_sustain_analysis'] and sustain_duration is not None:
                filtered_sustains.append(sustain_duration)
                if spectral_config['has_spectral_data']:
                    # Per-band spectral data for classification (e.g., hihat open/closed)
                    spectral_entry = {}
                    for band_name in geomean_bands:
                        spectral_entry[f'{band_name}_energy'] = band_energies.get(band_name, 0.0)
                    filtered_spectral.append(spectral_entry)
            # Store full spectral data for this KEPT onset (Detection Output Contract)
            filtered_onset_data.append(onset_data.copy())
    
    # SECOND PASS: Remove retriggering using decay pattern analysis
    # Sustaining stems can have energy modulation during sustain that looks like new onsets
    # Analyze spectral decay pattern to distinguish true hits from decay artifacts
    stem_config_section = config.get(stem_type, {})
    enable_decay_filter = stem_config_section.get('enable_decay_filter', False)
    if enable_decay_filter and not learning_mode and len(filtered_times) > 1:
        decay_filter_window_sec = stem_config_section.get('decay_filter_window_sec', 0.5)
        
        # Build list of times to keep
        final_times = []
        final_strengths = []
        final_amplitudes = []
        final_geomeans = []
        final_sustains = []
        final_onset_data = []  # Track spectral data through decay filter
        
        # Track all decay analysis for debug output
        decay_analysis_data = []
        
        # Track active decay zones (onset_time -> decay pattern)
        active_decays = {}
        
        for i in range(len(filtered_times)):
            current_time = filtered_times[i]
            current_sample = int(current_time * sr)
            
            # Check if this onset falls within any active decay zone
            is_retrigger = False
            prev_hit_time = None
            prev_decay_rate = None
            prev_is_decaying = None
            time_since_prev = None
            
            for prev_time, decay_info in active_decays.items():
                time_diff = current_time - prev_time
                
                # If within decay window
                if 0 < time_diff < decay_filter_window_sec:
                    # Check if we're in a decaying region
                    if decay_info['is_decaying']:
                        is_retrigger = True
                        prev_hit_time = prev_time
                        prev_decay_rate = decay_info['decay_rate']
                        prev_is_decaying = decay_info['is_decaying']
                        time_since_prev = time_diff
                        break
            
            # Store analysis data
            analysis_entry = {
                'time': current_time,
                'is_retrigger': is_retrigger,
                'prev_hit_time': prev_hit_time,
                'time_since_prev': time_since_prev,
                'prev_decay_rate': prev_decay_rate,
                'prev_is_decaying': prev_is_decaying,
                'geomean': filtered_geomeans[i],
                'sustain_ms': filtered_sustains[i] if i < len(filtered_sustains) else None
            }
            
            if not is_retrigger:
                # This is a legitimate hit - keep it
                final_times.append(filtered_times[i])
                final_strengths.append(filtered_strengths[i])
                final_amplitudes.append(filtered_amplitudes[i])
                final_geomeans.append(filtered_geomeans[i])
                if i < len(filtered_sustains):
                    final_sustains.append(filtered_sustains[i])
                if i < len(filtered_onset_data):
                    final_onset_data.append(filtered_onset_data[i])
                
                # Analyze decay pattern starting from this onset
                decay_pattern = analyze_cymbal_decay_pattern(
                    audio, current_sample, sr, 
                    window_sec=decay_filter_window_sec,
                    num_windows=8
                )
                
                # Store decay pattern info in analysis entry
                analysis_entry['own_decay_rate'] = decay_pattern['decay_rate']
                analysis_entry['own_is_decaying'] = decay_pattern['is_decaying']
                
                # Store for checking subsequent onsets
                active_decays[current_time] = decay_pattern
                
                # Clean up old decays outside the window
                active_decays = {
                    t: info for t, info in active_decays.items()
                    if current_time - t < decay_filter_window_sec
                }
            
            decay_analysis_data.append(analysis_entry)
        
        # Update filtered arrays
        filtered_times = final_times
        filtered_strengths = final_strengths
        filtered_amplitudes = final_amplitudes
        filtered_geomeans = final_geomeans
        filtered_sustains = final_sustains
        filtered_onset_data = final_onset_data
        
        # Update status in all_onset_data for retriggered events
        final_times_set = set(final_times)
        for onset_data in all_onset_data:
            if onset_data['status'] == 'KEPT' and onset_data['time'] not in final_times_set:
                onset_data['status'] = 'FILTERED'  # Filtered by decay pass
    
    # THIRD PASS: Statistical outlier detection (if enabled for this stem)
    # This catches bleed that passes geomean threshold but has abnormal band ratio
    
    enable_statistical = stem_config_section.get('enable_statistical_filter', False)
    if enable_statistical and not learning_mode:
        if len(all_onset_data) > 0:
            # Calculate statistical parameters from ALL detected onsets (including rejected ones)
            # This gives us the population statistics to compare against
            statistical_params = calculate_statistical_params(all_onset_data)
            
            # Get thresholds from config (default 0.3 matches midiconfig.yaml;
            # previously defaulted to 0.6 which masked snare bleed differently
            # when the key was missing from a user's config).
            badness_threshold = stem_config_section.get('statistical_badness_threshold', 0.3)
            ratio_weight = stem_config_section.get('statistical_ratio_weight', 0.7)
            total_weight = stem_config_section.get('statistical_total_weight', 0.3)
            
            # Calculate badness scores for ALL onsets (for debug output)
            for onset_data in all_onset_data:
                badness = calculate_badness_score(
                    onset_data,
                    statistical_params,
                    ratio_weight,
                    total_weight
                )
                onset_data['badness_score'] = badness
            
            # Re-filter the already-filtered onsets (Pass 1 survivors) using statistical scores
            # Build a map of time -> onset_data for quick lookup
            onset_data_by_time = {d['time']: d for d in all_onset_data}
            
            final_times = []
            final_strengths = []
            final_amplitudes = []
            final_geomeans = []
            final_onset_data = []
            
            for i, (time, strength, amplitude, geomean) in enumerate(zip(
                filtered_times, filtered_strengths, filtered_amplitudes, filtered_geomeans
            )):
                onset_data = onset_data_by_time.get(time)
                if onset_data and onset_data.get('badness_score', 0) <= badness_threshold:
                    final_times.append(time)
                    final_strengths.append(strength)
                    final_amplitudes.append(amplitude)
                    final_geomeans.append(geomean)
                    if i < len(filtered_onset_data):
                        final_onset_data.append(filtered_onset_data[i])
            
            # Update filtered arrays with statistical filter results
            filtered_times = final_times
            filtered_strengths = final_strengths
            filtered_amplitudes = final_amplitudes
            filtered_geomeans = final_geomeans
            filtered_onset_data = final_onset_data
            
            # Update status in all_onset_data for statistically rejected events
            final_times_set = set(final_times)
            for onset_data in all_onset_data:
                if onset_data['status'] == 'KEPT' and onset_data['time'] not in final_times_set:
                    onset_data['status'] = 'FILTERED'  # Filtered by statistical pass
            
            # Store statistical info in config for debug output
            spectral_config['statistical_params'] = statistical_params
            spectral_config['statistical_enabled'] = True
            spectral_config['badness_threshold'] = badness_threshold
    
    # Prepare decay analysis data for return (present when decay filter ran)
    decay_analysis = None
    if 'decay_analysis_data' in locals():
        decay_analysis = {
            'data': decay_analysis_data,
            'window_sec': decay_filter_window_sec
        }
    
    # Mark reverb continuation events (peak-hold detection artifact filtering)
    # These are kept in the data but marked so MIDI export can filter them
    # Uses attack_sharpness to distinguish real hits from reverb tails.
    # Default 0.4 matches midiconfig.yaml; previously defaulted to 0.2 which
    # was a silent drift — the YAML value always won at runtime but missing
    # keys got a different value.
    attack_threshold = config.get('filtering', {}).get('reverb_continuation_attack_threshold', 0.4)
    all_onset_data = mark_reverb_continuations(
        all_onset_data,
        time_margin_ms=5.0,
        amplitude_margin=0.001,
        attack_sharpness_threshold=attack_threshold  # Real hits have sharper attacks (>= 0.2)
    )
    
    # Remove reverb continuations from filtered lists so they don't become MIDI notes
    reverb_times = {e['time'] for e in all_onset_data if e.get('status') == 'REVERB_CONTINUATION'}
    if reverb_times:
        # Filter out reverb continuations from all filtered arrays
        keep_indices = [i for i, t in enumerate(filtered_times) if t not in reverb_times]
        filtered_times = [filtered_times[i] for i in keep_indices]
        filtered_strengths = [filtered_strengths[i] for i in keep_indices]
        filtered_amplitudes = [filtered_amplitudes[i] for i in keep_indices]
        filtered_geomeans = [filtered_geomeans[i] for i in keep_indices]
        if filtered_sustains:
            filtered_sustains = [filtered_sustains[i] for i in keep_indices if i < len(filtered_sustains)]
        if filtered_spectral:
            filtered_spectral = [filtered_spectral[i] for i in keep_indices if i < len(filtered_spectral)]
        filtered_onset_data = [d for d in filtered_onset_data if d['time'] not in reverb_times]
    
    return {
        'filtered_times': np.array(filtered_times),
        'filtered_strengths': np.array(filtered_strengths),
        'filtered_amplitudes': np.array(filtered_amplitudes),
        'filtered_geomeans': np.array(filtered_geomeans),
        'filtered_sustains': filtered_sustains,
        'filtered_spectral': filtered_spectral,
        'filtered_onset_data': filtered_onset_data,  # Full spectral data for KEPT onsets
        'all_onset_data': all_onset_data,
        'spectral_config': spectral_config,
        'decay_analysis': decay_analysis
    }


def analyze_onset_spectral(
    audio: np.ndarray,
    onset_time: float,
    sr: int,
    stem_type: str,
    config: Dict,
    duration: Optional[float] = None
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
        duration: Optional event duration in seconds. If provided, stored
                 in result dict for downstream use in Phase 2 metadata.
    
    Returns:
        Dictionary with analysis results, or None if segment too short:
        {
            'onset_sample': int,
            'segment': np.ndarray,
            '<band>_energy': float,  # Domain-specific: body_energy, wire_energy, etc.
            'low_energy': float,
            'total_energy': float,
            'geomean': float,
            'geomean_bands': list[str],  # Band names used for geomean
            'sustain_ms': float (if calculated),
            'spectral_ratio': float,
            'duration_sec': float (if duration provided)
        }
        
        Band energy keys are domain-specific per stem type:
            snare: body_energy, wire_energy
            kick: fundamental_energy, body_energy, attack_energy
            toms: fundamental_energy, body_energy
            hihat: body_energy, sizzle_energy
            cymbals: body_energy, brilliance_energy
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
    
    # Calculate spectral energies (keys are domain-specific: body, wire, fundamental, etc.)
    energies = calculate_spectral_energies(segment, sr, spectral_config['freq_ranges'])
    
    # Extract geomean band energies in order
    geomean_bands = spectral_config['geomean_bands']
    band_energies = [energies.get(band, 0.0) for band in geomean_bands]
    low_energy = energies.get('low', 0.0)
    
    # Calculate geomean (2-way or 3-way depending on stem)
    if len(band_energies) >= 3:
        geomean = calculate_geomean(band_energies[0], band_energies[1], band_energies[2])
    else:
        geomean = calculate_geomean(band_energies[0], band_energies[1])
    
    # Calculate total energy across geomean bands
    total_energy = sum(band_energies)
    
    # Calculate spectral ratio if low energy available
    spectral_ratio = (total_energy / low_energy) if low_energy > 0 else 100.0
    
    # Calculate sustain duration if needed
    sustain_ms = None
    if stem_type in ['hihat', 'cymbals']:
        # Get stem-specific sustain analysis window, fallback to global
        stem_config = config.get(stem_type, {})
        sustain_analysis_window_sec = stem_config.get('sustain_analysis_window_sec')
        if sustain_analysis_window_sec is None:
            sustain_analysis_window_sec = config.get('audio', {}).get('sustain_window_sec', 0.2)
        
        envelope_threshold = config.get('audio', {}).get('envelope_threshold', 0.1)
        smooth_kernel = config.get('audio', {}).get('envelope_smooth_kernel', 51)
        
        sustain_ms = calculate_sustain_duration(
            audio, onset_sample, sr,
            window_ms=sustain_analysis_window_sec * 1000,
            envelope_threshold=envelope_threshold,
            smooth_kernel=smooth_kernel
        )
    
    # Build result with domain-specific energy keys
    result = {
        'onset_sample': onset_sample,
        'segment': segment,
        'low_energy': low_energy,
        'total_energy': total_energy,
        'geomean': geomean,
        'geomean_bands': geomean_bands,
        'sustain_ms': sustain_ms,
        'spectral_ratio': spectral_ratio
    }
    
    # Add each band energy with its domain-specific name
    for band_name in geomean_bands:
        result[f'{band_name}_energy'] = energies.get(band_name, 0.0)
    
    # Add duration if provided (for Phase 2 metadata enrichment)
    if duration is not None:
        result['duration_sec'] = duration
    
    return result
