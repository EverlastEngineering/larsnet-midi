"""
Convert separated drum stems to MIDI tracks.

Analyzes each drum stem to detect onsets (hits) and converts them to MIDI notes
with velocity based on the hit intensity.
"""

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from midiutil import MIDIFile
import mido
import argparse
import yaml
from typing import Union, List, Tuple, Dict, Optional
from dataclasses import dataclass


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load MIDI conversion configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'midiconfig.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


@dataclass
class DrumMapping:
    """MIDI note mappings for drums (General MIDI standard)."""
    kick: int = 36      # C1 - Bass Drum 1
    snare: int = 38     # D1 - Acoustic Snare
    toms: int = 45      # A1 - Low Tom (can be split into multiple toms)
    hihat: int = 42     # F#1 - Closed Hi-Hat
    cymbals: int = 49   # C#2 - Crash Cymbal 1
    
    # Alternative mappings
    kick_alt: int = 35      # B0 - Bass Drum 2
    snare_alt: int = 40     # E1 - Electric Snare
    tom_low: int = 45       # A1 - Low Tom
    tom_mid: int = 47       # B1 - Mid Tom
    tom_high: int = 50      # D2 - High Tom
    hihat_open: int = 46    # A#1 - Open Hi-Hat
    hihat_closed: int = 42  # F#1 - Closed Hi-Hat
    crash: int = 49         # C#2 - Crash Cymbal 1
    ride: int = 51          # D#2 - Ride Cymbal 1


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
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    
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
    window_ms: float = 150.0,
    decay_threshold: float = 0.65
) -> List[str]:
    """
    Attempt to classify hi-hat hits as open or closed based on decay and sustain.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        onset_times: Times of detected onsets
        window_ms: Window length in ms to analyze decay (longer = better detection)
        decay_threshold: Threshold for open vs closed (higher = fewer open detections)
    
    Returns:
        List of 'open' or 'closed' for each onset
    """
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    
    states = []
    window_samples = int(sr * window_ms / 1000.0)
    
    for onset_time in onset_times:
        onset_sample = int(onset_time * sr)
        end_sample = min(onset_sample + window_samples, len(audio))
        
        if end_sample - onset_sample < window_samples // 2:
            states.append('closed')
            continue
        
        # Analyze decay envelope
        segment = audio[onset_sample:end_sample]
        envelope = np.abs(segment)
        
        # Calculate multiple metrics for better classification
        if len(envelope) > 20:
            # Method 1: Energy ratio (tail vs beginning)
            first_third = envelope[:len(envelope)//3]
            last_third = envelope[len(envelope)*2//3:]
            
            first_energy = np.sum(first_third**2)
            last_energy = np.sum(last_third**2)
            
            if first_energy > 0:
                energy_ratio = last_energy / first_energy
            else:
                energy_ratio = 0
            
            # Method 2: Sustained energy over time
            # Compute RMS over sliding windows
            window_size = len(envelope) // 5
            rms_values = []
            for i in range(0, len(envelope) - window_size, window_size):
                window = envelope[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(rms)
            
            if len(rms_values) > 2:
                # Open hi-hat maintains more energy over time
                sustained_ratio = np.mean(rms_values[2:]) / (rms_values[0] + 1e-10)
            else:
                sustained_ratio = 0
            
            # Combine both metrics (open hi-hat has higher values for both)
            combined_score = (energy_ratio * 0.6) + (sustained_ratio * 0.4)
            
            # Much higher threshold - most hits are closed, only obvious sustains are open
            if combined_score > decay_threshold:
                states.append('open')
            else:
                states.append('closed')
        else:
            states.append('closed')
    
    return states


def process_stem_to_midi(
    audio_path: Union[str, Path],
    stem_type: str,
    drum_mapping: DrumMapping,
    config: Dict,
    onset_threshold: float = 0.3,
    min_velocity: int = 40,
    max_velocity: int = 127,
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
    
    # Load audio
    audio, sr = sf.read(str(audio_path))
    
    # Convert to mono if configured (and if stereo)
    if config['audio']['force_mono'] and audio.ndim == 2:
        audio = np.mean(audio, axis=1)
        print(f"    Converted stereo to mono")
    
    # Check if audio is essentially silent (max amplitude below threshold)
    max_amplitude = np.max(np.abs(audio))
    print(f"    Max amplitude: {max_amplitude:.6f}")
    
    if max_amplitude < 0.001:  # Essentially silent (less than -60dB)
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
        onset_times, onset_strengths = detect_onsets(
            audio,
            sr,
            hop_length=onset_config['hop_length'],
            threshold=onset_config['threshold'],
            delta=onset_config['delta'],
            wait=onset_config['wait']
        )
    
    print(f"    Found {len(onset_times)} hits (before filtering) -> MIDI note {getattr(drum_mapping, stem_type)}")
    
    # Calculate peak amplitudes for all stems (for velocity calculation)
    # Calculate peak amplitudes for all stems (for velocity calculation)
    # Audio is already mono at this point
    peak_amplitudes = []
    for onset_time in onset_times:
        onset_sample = int(onset_time * sr)
        peak_window = int(0.01 * sr)  # 10ms window
        peak_end = min(onset_sample + peak_window, len(audio))
        
        peak_segment = audio[onset_sample:peak_end]
        peak_amplitude = np.max(np.abs(peak_segment)) if len(peak_segment) > 0 else 0
        peak_amplitudes.append(peak_amplitude)
    
    peak_amplitudes = np.array(peak_amplitudes)
    
    # For snare and kick, filter out artifacts/bleed by checking spectral content
    if stem_type in ['snare', 'kick'] and len(onset_times) > 0:
        original_count = len(onset_times)
        # Store originals for debug output
        onset_times_orig = onset_times.copy()
        onset_strengths_orig = onset_strengths.copy()
        peak_amplitudes_orig = peak_amplitudes.copy()
        
        filtered_times = []
        filtered_strengths = []
        filtered_amplitudes = []
        filtered_geomeans = []
        ratios_kept = []
        amplitudes_kept = []
        ratios_rejected = []
        
        # Store raw spectral data for ALL onsets
        all_onset_data = []
        
        for onset_time, strength, peak_amplitude in zip(onset_times, onset_strengths, peak_amplitudes):
            onset_sample = int(onset_time * sr)
            
            # Analyze a 50ms window after onset
            # Audio is already mono at this point
            window_samples = int(0.05 * sr)
            end_sample = min(onset_sample + window_samples, len(audio))
            
            segment = audio[onset_sample:end_sample]
            
            if len(segment) < 100:
                continue
            
            # peak_amplitude already calculated above
            
            # Compute FFT
            fft = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1/sr)
            magnitude = np.abs(fft)
            
            # Use frequency ranges from config based on stem type
            if stem_type == 'snare':
                stem_config = config['snare']
                # For snare: analyze body (150-400Hz) and wires (2-8kHz)
                low_energy = np.sum(magnitude[(freqs >= stem_config['low_freq_min']) & (freqs < stem_config['low_freq_max'])])
                primary_energy = np.sum(magnitude[(freqs >= stem_config['body_freq_min']) & (freqs < stem_config['body_freq_max'])])
                secondary_energy = np.sum(magnitude[(freqs >= stem_config['wire_freq_min']) & (freqs < stem_config['wire_freq_max'])])
                energy_label_1 = 'BodyE'
                energy_label_2 = 'WireE'
            elif stem_type == 'kick':
                stem_config = config['kick']
                # For kick: analyze fundamental (40-80Hz) and body/attack (80-150Hz)
                low_energy = 0  # Not used for kick (kick IS the low energy)
                primary_energy = np.sum(magnitude[(freqs >= stem_config['fundamental_freq_min']) & (freqs < stem_config['fundamental_freq_max'])])
                secondary_energy = np.sum(magnitude[(freqs >= stem_config['body_freq_min']) & (freqs < stem_config['body_freq_max'])])
                energy_label_1 = 'FundE'  # Fundamental energy
                energy_label_2 = 'BodyE'  # Body/attack energy
            
            # Calculate combined metric
            total_energy = primary_energy + secondary_energy
            
            if low_energy > 0:
                spectral_ratio = total_energy / low_energy
            else:
                spectral_ratio = 100  # No low energy (or not used for this stem type)
            
            # Calculate geometric mean (discriminator for real hits vs artifacts)
            body_to_wire = primary_energy / secondary_energy if secondary_energy > 0 else 0
            body_times_wire = primary_energy * secondary_energy
            body_wire_geomean = np.sqrt(primary_energy * secondary_energy)
            
            # Store all data for this onset
            all_onset_data.append({
                'time': onset_time,
                'strength': strength,
                'amplitude': peak_amplitude,
                'low_energy': low_energy,
                'primary_energy': primary_energy,
                'secondary_energy': secondary_energy,
                'ratio': spectral_ratio,
                'total_energy': total_energy,
                'body_to_wire': body_to_wire,
                'body_times_wire': body_times_wire,
                'body_wire_geomean': body_wire_geomean,
                'energy_label_1': energy_label_1,
                'energy_label_2': energy_label_2
            })
            
            # Filter based on geometric mean of primary and secondary energy
            # Use threshold from config (default 100-150, can adjust per file/drum style)
            geomean_threshold = stem_config['geomean_threshold']
            is_real_hit = (body_wire_geomean > geomean_threshold)
            
            # In learning mode, keep ALL detections but mark them
            if learning_mode or is_real_hit:
                filtered_times.append(onset_time)
                filtered_strengths.append(strength)
                filtered_amplitudes.append(peak_amplitude)
                filtered_geomeans.append(body_wire_geomean)
                # Store whether this would normally be kept or rejected
                ratios_kept.append(spectral_ratio if is_real_hit else -spectral_ratio)  # Negative = would be rejected
                amplitudes_kept.append(peak_amplitude)
            elif not learning_mode:
                ratios_rejected.append(spectral_ratio)
        
        onset_times = np.array(filtered_times)
        onset_strengths = np.array(filtered_strengths)
        peak_amplitudes = np.array(filtered_amplitudes)
        stem_geomeans = np.array(filtered_geomeans)
        
        # Show ALL onset data in chronological order with multiple ratios
        print(f"\n      ALL DETECTED ONSETS - SPECTRAL ANALYSIS:")
        print(f"      Using GeoMean threshold: {geomean_threshold}")
        
        # Configure labels based on stem type
        if stem_type == 'snare':
            print(f"      Str=Onset Strength, Amp=Peak Amplitude, BodyE=Body Energy (150-400Hz), WireE=Wire Energy (2-8kHz)")
        elif stem_type == 'kick':
            print(f"      Str=Onset Strength, Amp=Peak Amplitude, FundE=Fundamental Energy (40-80Hz), BodyE=Body Energy (80-150Hz)")
        print(f"      GeoMean=sqrt({energy_label_1}*{energy_label_2}) - measures combined spectral energy")
        
        print(f"\n      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {energy_label_1+'E':>8s} {energy_label_2+'E':>8s} {'Total':>8s} {'GeoMean':>8s} {'Status':>10s}")
        for idx, data in enumerate(all_onset_data):
            is_real_hit = (data['body_wire_geomean'] > geomean_threshold)
            status = 'KEPT' if is_real_hit else 'REJECTED'
            print(f"      {data['time']:8.3f} {data['strength']:6.3f} {data['amplitude']:6.3f} {data['primary_energy']:8.1f} {data['secondary_energy']:8.1f} "
                  f"{data['total_energy']:8.1f} {data['body_wire_geomean']:8.1f} {status:>10s}")
        
        # Show summary statistics
        kept_geomeans = [d['body_wire_geomean'] for d in all_onset_data if d['body_wire_geomean'] > geomean_threshold]
        rejected_geomeans = [d['body_wire_geomean'] for d in all_onset_data if d['body_wire_geomean'] <= geomean_threshold]
        
        print(f"\n      FILTERING SUMMARY:")
        print(f"        GeoMean threshold: {geomean_threshold} (adjustable in midiconfig.yaml)")
        print(f"        Total onsets detected: {len(all_onset_data)}")
        print(f"        Kept (GeoMean > {geomean_threshold}): {len(kept_geomeans)}")
        print(f"        Rejected (GeoMean <= {geomean_threshold}): {len(rejected_geomeans)}")
        if kept_geomeans:
            print(f"        Kept GeoMean range: {min(kept_geomeans):.1f} - {max(kept_geomeans):.1f}")
        if rejected_geomeans:
            print(f"        Rejected GeoMean range: {min(rejected_geomeans):.1f} - {max(rejected_geomeans):.1f}")
        
        print(f"\n    After spectral filtering: {len(onset_times)} hits (rejected {len(ratios_rejected)} artifacts)")
        
        # ALSO show first 20 REJECTED hits to understand what we're filtering out
        if ratios_rejected and False:  # Disabled for cleaner output
            print(f"      First 20 REJECTED hits (for comparison):")
            rejected_count = 0
            for onset_time, strength, peak_amplitude in zip(onset_times_orig, onset_strengths_orig, peak_amplitudes_orig):
                onset_sample = int(onset_time * sr)
                window_samples = int(0.05 * sr)
                end_sample = min(onset_sample + window_samples, len(audio) if audio.ndim == 1 else len(audio[:, 0]))
                
                if audio.ndim == 1:
                    segment = audio[onset_sample:end_sample]
                else:
                    segment = np.mean(audio[onset_sample:end_sample], axis=1)
                
                if len(segment) < 100:
                    continue
                
                fft = np.fft.rfft(segment)
                freqs = np.fft.rfftfreq(len(segment), 1/sr)
                magnitude = np.abs(fft)
                
                low_energy = np.sum(magnitude[(freqs >= 40) & (freqs < 150)])
                snare_body_energy = np.sum(magnitude[(freqs >= 150) & (freqs < 400)])
                snare_wire_energy = np.sum(magnitude[(freqs >= 2000) & (freqs < 8000)])
                total_snare_energy = snare_body_energy + snare_wire_energy
                
                if low_energy > 0:
                    snare_ratio = total_snare_energy / low_energy
                else:
                    snare_ratio = 100
                
                is_kick_bleed = (snare_ratio < 2.0)
                
                if is_kick_bleed:
                    print(f"        Time: {onset_time:.3f}s, Ratio: {snare_ratio:.2f}, Amp: {peak_amplitude:.3f}, Reason: kick_bleed")
                    rejected_count += 1
                    if rejected_count >= 20:
                        break
    
    if len(onset_times) == 0:
        return []
    
    # Get MIDI note number
    note = getattr(drum_mapping, stem_type)
    
    # Special handling for hi-hat
    if stem_type == 'hihat' and detect_hihat_open:
        hihat_states = detect_hihat_state(audio, sr, onset_times)
    else:
        hihat_states = ['closed'] * len(onset_times)
    
    # Calculate velocity based on spectral energy for filtered stems, amplitude for others
    if stem_type in ['snare', 'kick'] and len(stem_geomeans) > 0:
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
    
    # Create MIDI events
    events = []
    for i, (time, value) in enumerate(zip(onset_times, normalized_values)):
        # In learning mode, check if this hit would normally be rejected
        is_rejected_in_learning = False
        if learning_mode and stem_type == 'snare' and i < len(ratios_kept):
            is_rejected_in_learning = (ratios_kept[i] < 0)  # Negative ratio = rejected
        
        if is_rejected_in_learning:
            # Mark rejected hits with velocity=1 so user can easily identify them
            velocity = config['learning_mode']['rejected_velocity']
        else:
            velocity = estimate_velocity(value, min_velocity, max_velocity)
        
        # Adjust note for open hi-hat
        if stem_type == 'hihat' and hihat_states[i] == 'open':
            midi_note = drum_mapping.hihat_open
        else:
            midi_note = note
        
        # Duration: until next hit or default
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - time
        else:
            duration = 0.1  # Default duration for last note
        
        duration = min(duration, 0.5)  # Cap duration at 500ms
        
        events.append({
            'time': float(time),
            'note': int(midi_note),
            'velocity': int(velocity),
            'duration': float(duration)
        })
    
    return events


def create_midi_file(
    events_by_stem: Dict[str, List[Dict]],
    output_path: Union[str, Path],
    tempo: float = 120.0,
    track_name: str = "Drums"
):
    """
    Create a MIDI file from detected drum events.
    
    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        output_path: Path to save MIDI file
        tempo: Tempo in BPM
        track_name: Name of the MIDI track
    """
    # Create MIDI file with 1 track
    midi = MIDIFile(1)
    track = 0
    channel = 9  # Channel 10 (0-indexed as 9) is typically drums in MIDI
    time = 0
    
    midi.addTrackName(track, time, track_name)
    midi.addTempo(track, time, tempo)
    
    # Add all events
    total_events = 0
    for stem_type, events in events_by_stem.items():
        for event in events:
            midi.addNote(
                track=track,
                channel=channel,
                pitch=event['note'],
                time=event['time'],
                duration=event['duration'],
                volume=event['velocity']
            )
            total_events += 1
    
    # Write to file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    print(f"  Created MIDI file with {total_events} notes")


def read_midi_notes(midi_path: Union[str, Path], target_note: int) -> List[float]:
    """
    Read note times from a MIDI file for a specific MIDI note number.
    
    Args:
        midi_path: Path to MIDI file
        target_note: MIDI note number to extract (e.g., 38 for snare)
    
    Returns:
        List of note times in seconds
    """
    midi_file = mido.MidiFile(str(midi_path))
    note_times = []
    current_time = 0.0
    
    # Get ticks per beat for time conversion
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500000  # Default tempo (120 BPM in microseconds per beat)
    
    for track in midi_file.tracks:
        current_time = 0.0
        for msg in track:
            current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
            
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.note == target_note and msg.velocity > 0:
                note_times.append(current_time)
    
    return sorted(note_times)


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
    
    # Match tolerance: 50ms
    match_tolerance = 0.05
    
    for orig_time in original_times:
        # Check if this time exists in edited MIDI
        is_kept = any(abs(orig_time - edit_time) < match_tolerance for edit_time in edited_times)
        
        # Analyze this onset
        onset_sample = int(orig_time * sr)
        window_samples = int(0.05 * sr)
        end_sample = min(onset_sample + window_samples, len(audio))
        segment = audio[onset_sample:end_sample]
        
        if len(segment) < 100:
            continue
        
        # FFT analysis
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        magnitude = np.abs(fft)
        
        # Get spectral energies
        if stem_type == 'snare':
            snare_config = config['snare']
            body_energy = np.sum(magnitude[(freqs >= snare_config['body_freq_min']) & (freqs < snare_config['body_freq_max'])])
            wire_energy = np.sum(magnitude[(freqs >= snare_config['wire_freq_min']) & (freqs < snare_config['wire_freq_max'])])
            geomean = np.sqrt(body_energy * wire_energy)
            
            if is_kept:
                kept_geomeans.append(geomean)
            else:
                removed_geomeans.append(geomean)
    
    # Calculate suggested threshold
    if kept_geomeans and removed_geomeans:
        min_kept = min(kept_geomeans)
        max_removed = max(removed_geomeans)
        
        # Suggest threshold halfway between max removed and min kept
        suggested_threshold = (max_removed + min_kept) / 2.0
        
        print(f"\n    Analysis:")
        print(f"      Kept hits - GeoMean range: {min_kept:.1f} - {max(kept_geomeans):.1f}")
        print(f"      Removed hits - GeoMean range: {min(removed_geomeans):.1f} - {max_removed:.1f}")
        print(f"      Suggested threshold: {suggested_threshold:.1f}")
        print(f"      (Midpoint between max removed ({max_removed:.1f}) and min kept ({min_kept:.1f}))")
        
        return {
            'geomean_threshold': suggested_threshold,
            'kept_range': (min_kept, max(kept_geomeans)),
            'removed_range': (min(removed_geomeans), max_removed)
        }
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
    min_velocity: int = 40,
    max_velocity: int = 127,
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
            
            events = process_stem_to_midi(
                stem_file,
                stem_type,
                drum_mapping,
                config,
                onset_threshold=onset_threshold,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                detect_hihat_open=detect_hihat_open
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
                track_name=f"Drums - {base_name}"
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
    
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Directory containing separated stems (must have kick/, snare/, etc. subdirectories).")
    parser.add_argument('-o', '--output_dir', type=str, default='midi_output',
                        help="Directory to save MIDI files (default: midi_output).")
    parser.add_argument('-t', '--threshold', type=float, default=0.3,
                        help="Onset detection threshold (0-1). Lower = more sensitive (default: 0.3).")
    parser.add_argument('--min-vel', type=int, default=40,
                        help="Minimum MIDI velocity (1-127, default: 40).")
    parser.add_argument('--max-vel', type=int, default=127,
                        help="Maximum MIDI velocity (1-127, default: 127).")
    parser.add_argument('--tempo', type=float, default=120.0,
                        help="Tempo in BPM for MIDI timing (default: 120).")
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
