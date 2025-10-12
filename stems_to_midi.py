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
    
    # Detect onsets with increased sensitivity
    # Use wait=1 for fast repeated hits (like drum rolls, double hits)
    # Lower threshold and delta for more sensitive detection
    onset_times, onset_strengths = detect_onsets(
        audio,
        sr,
        threshold=0.15,  # Lower than onset_threshold to catch more hits (default 0.3)
        delta=0.005,     # Lower delta for more sensitive peak picking (default 0.01)
        wait=1           # Allow onsets only 1 frame apart (~11ms at 44.1kHz) for fast hits
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
    
    # For snare specifically, filter out kick bleed by checking spectral content
    if stem_type == 'snare' and len(onset_times) > 0:
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
            
            # Use frequency ranges from config
            snare_config = config['snare']
            low_energy = np.sum(magnitude[(freqs >= snare_config['low_freq_min']) & (freqs < snare_config['low_freq_max'])])
            snare_body_energy = np.sum(magnitude[(freqs >= snare_config['body_freq_min']) & (freqs < snare_config['body_freq_max'])])
            snare_wire_energy = np.sum(magnitude[(freqs >= snare_config['wire_freq_min']) & (freqs < snare_config['wire_freq_max'])])
            
            # Multiple criteria for real snare:
            # 1. Spectral: Should have good mid-frequency body energy (not just highs)
            # 2. Amplitude: Should be reasonably loud (normalized to max in file)
            # 3. Balance: Should not be ONLY high frequencies (that's a click)
            
            total_snare_energy = snare_body_energy + snare_wire_energy
            
            if low_energy > 0:
                snare_ratio = total_snare_energy / low_energy
            else:
                snare_ratio = 100  # No low energy
            
            # Calculate multiple potential discriminators
            total_snare = snare_body_energy + snare_wire_energy
            body_to_wire = snare_body_energy / snare_wire_energy if snare_wire_energy > 0 else 0
            body_times_wire = snare_body_energy * snare_wire_energy
            body_wire_geomean = np.sqrt(snare_body_energy * snare_wire_energy)
            
            # Store all data for this onset
            all_onset_data.append({
                'time': onset_time,
                'strength': strength,
                'amplitude': peak_amplitude,
                'low_energy': low_energy,
                'snare_body_energy': snare_body_energy,
                'snare_wire_energy': snare_wire_energy,
                'ratio': snare_ratio,
                'total_snare': total_snare,
                'body_to_wire': body_to_wire,
                'body_times_wire': body_times_wire,
                'body_wire_geomean': body_wire_geomean
            })
            
            # Filter based on geometric mean of body and wire energy
            # Use threshold from config (default 100, can adjust per file/drum style)
            geomean_threshold = snare_config['geomean_threshold']
            is_real_snare = (body_wire_geomean > geomean_threshold)
            
            # Keep if it's a real snare
            if is_real_snare:
                filtered_times.append(onset_time)
                filtered_strengths.append(strength)
                filtered_amplitudes.append(peak_amplitude)
                filtered_geomeans.append(body_wire_geomean)
                ratios_kept.append(snare_ratio)
                amplitudes_kept.append(peak_amplitude)
            else:
                ratios_rejected.append(snare_ratio)
        
        onset_times = np.array(filtered_times)
        onset_strengths = np.array(filtered_strengths)
        peak_amplitudes = np.array(filtered_amplitudes)
        snare_geomeans = np.array(filtered_geomeans)
        
        # Show ALL onset data in chronological order with multiple ratios
        print(f"\n      ALL DETECTED ONSETS - SPECTRAL ANALYSIS:")
        print(f"      Using GeoMean threshold: {geomean_threshold}")
        print(f"      Str=Onset Strength, Amp=Peak Amplitude, BodyE=Body Energy (150-400Hz), WireE=Wire Energy (2-8kHz)")
        print(f"      GeoMean=sqrt(BodyE*WireE) - measures combined spectral energy")
        print(f"\n      {'Time':>8s} {'Str':>6s} {'Amp':>6s} {'BodyE':>8s} {'WireE':>8s} {'Total':>8s} {'GeoMean':>8s} {'Status':>10s}")
        for idx, data in enumerate(all_onset_data):
            is_real_snare = (data['body_wire_geomean'] > geomean_threshold)
            status = 'KEPT' if is_real_snare else 'REJECTED'
            print(f"      {data['time']:8.3f} {data['strength']:6.3f} {data['amplitude']:6.3f} {data['snare_body_energy']:8.1f} {data['snare_wire_energy']:8.1f} "
                  f"{data['total_snare']:8.1f} {data['body_wire_geomean']:8.1f} {status:>10s}")
        
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
        
        print(f"\n    After spectral filtering: {len(onset_times)} hits (rejected {len(ratios_rejected)} kick bleed)")
        
        # ALSO show first 20 REJECTED hits to understand what we're filtering out
        if ratios_rejected:
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
    
    # Calculate velocity based on spectral energy for snare, amplitude for others
    if stem_type == 'snare' and len(snare_geomeans) > 0:
        # For snare, use geometric mean of body and wire energy (better correlates with loudness)
        max_geomean = np.max(snare_geomeans)
        if max_geomean > 0:
            normalized_values = snare_geomeans / max_geomean
        else:
            normalized_values = np.ones_like(snare_geomeans)
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
    
    # Find all audio files (use first available stem directory as reference)
    audio_files = []
    reference_dir = None
    
    for stem_type in stems_to_process:
        stem_dir = stems_dir / stem_type
        if stem_dir.exists():
            audio_files = list(stem_dir.glob('*.wav'))
            if audio_files:
                reference_dir = stem_dir
                break
    
    if not reference_dir:
        raise RuntimeError(f"No stem directories found in {stems_dir} for: {stems_to_process}")
    
    if not audio_files:
        raise RuntimeError(f"No .wav files found in {reference_dir}")
    
    print(f"Processing {len(audio_files)} file(s)...")
    print(f"Settings:")
    print(f"  Onset threshold: {onset_threshold}")
    print(f"  Velocity range: {min_velocity}-{max_velocity}")
    print(f"  Tempo: {tempo} BPM")
    print(f"  Detect open hi-hat: {detect_hihat_open}")
    print()
    
    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")
        
        events_by_stem = {}
        
        # Process each stem type
        for stem_type in stems_to_process:
            stem_dir = stems_dir / stem_type
            stem_file = stem_dir / audio_file.name
            
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
            midi_path = output_dir / f"{audio_file.stem}.mid"
            create_midi_file(
                events_by_stem,
                midi_path,
                tempo=tempo,
                track_name=f"Drums - {audio_file.stem}"
            )
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
