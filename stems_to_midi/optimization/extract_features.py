#!/usr/bin/env python3
"""
Extract Features for Threshold Optimization

Runs detection pipeline on a stem and exports all onset candidates with features
to CSV format. This captures data BEFORE filtering so we can learn optimal thresholds.

Usage:
    python -m stems_to_midi.optimization.extract_features 4 --stem hihat
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
import soundfile as sf
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stems_to_midi.detection import detect_onsets, detect_hihat_state
from stems_to_midi.helpers import (
    ensure_mono,
    calculate_peak_amplitude,
    get_spectral_config_for_stem,
    filter_onsets_by_spectral,
    should_keep_onset
)
from project_manager import get_stem_file, get_project_by_number


def load_project_config(project_number: int) -> dict:
    """Load project configuration from midiconfig.yaml"""
    config_path = Path(f"user_files/{project_number}/midiconfig.yaml")
    
    if not config_path.exists():
        # Use default config
        config_path = Path("midiconfig.yaml")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_features_from_stem(project_number: int, stem_type: str, output_dir: Path = None):
    """
    Extract spectral features from all detected onsets in a stem.
    
    Args:
        project_number: Project number (e.g., 4 for user_files/4)
        stem_type: Stem type ('hihat', 'kick', 'snare', etc.)
        output_dir: Where to save CSV (default: user_files/{project}/optimization/)
    
    Returns:
        Path to generated CSV file
    """
    # Get project info from project_manager
    project = get_project_by_number(project_number)
    if project is None:
        raise FileNotFoundError(f"Project {project_number} not found")
    
    project_dir = project["path"]
    
    # Load project configuration
    print(f"Loading project {project_number} configuration...")
    config = load_project_config(project_number)
    
    # Find stem file using project_manager
    stem_path = get_stem_file(project_number, stem_type)
    
    if stem_path is None:
        raise FileNotFoundError(f"No {stem_type} stem file found in project {project_number}")
    
    # Create output directory
    if output_dir is None:
        output_dir = project_dir / "optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / f"{stem_type}_features.csv"
    
    print(f"Extracting features from: {stem_path}")
    print(f"Output CSV: {output_csv}")
    print()
    
    # Load audio
    print("Loading audio...")
    audio, sr = sf.read(str(stem_path))
    audio = ensure_mono(audio)
    
    # Get stem configuration
    stem_config = config.get(stem_type, {})
    hop_length = config.get('hop_length', 512)
    
    # Onset detection settings
    onset_threshold = stem_config.get('onset_threshold', 0.02)
    onset_delta = stem_config.get('onset_delta', 0.02)
    onset_wait = stem_config.get('onset_wait', 1)
    
    print(f"Detecting onsets (threshold={onset_threshold}, delta={onset_delta}, wait={onset_wait})...")
    onset_times, onset_strengths = detect_onsets(
        audio, sr,
        hop_length=hop_length,
        threshold=onset_threshold,
        delta=onset_delta,
        wait=onset_wait
    )
    
    print(f"Found {len(onset_times)} onsets")
    
    # Calculate peak amplitudes
    print("Calculating peak amplitudes...")
    peak_amplitudes = np.array([
        calculate_peak_amplitude(audio, int(t * sr), sr)
        for t in onset_times
    ])
    
    # Extract spectral features (using learning mode to keep ALL onsets)
    print("Extracting spectral features...")
    spectral_result = filter_onsets_by_spectral(
        onset_times,
        onset_strengths,
        peak_amplitudes,
        audio,
        sr,
        stem_type,
        config,
        learning_mode=True  # Keep ALL onsets, don't filter
    )
    
    all_onset_data = spectral_result['all_onset_data']
    spectral_config = spectral_result['spectral_config']
    
    print(f"Extracted features for {len(all_onset_data)} onsets")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_onset_data)
    
    # Add "Status" column showing current logic's decision
    if spectral_config:
        geomean_threshold = spectral_config['geomean_threshold']
        min_sustain_ms = spectral_config.get('min_sustain_ms')
        min_strength_threshold = spectral_config.get('min_strength_threshold')
        
        df['Status'] = df.apply(
            lambda row: 'KEPT' if should_keep_onset(
                geomean=row['body_wire_geomean'],
                sustain_ms=row.get('sustain_ms'),
                geomean_threshold=geomean_threshold,
                min_sustain_ms=min_sustain_ms,
                stem_type=stem_type,
                strength=row.get('strength'),
                min_strength_threshold=min_strength_threshold
            ) else 'REJECTED',
            axis=1
        )
    else:
        df['Status'] = 'KEPT'
    
    # Add classification columns for hihat (or other instrument-specific types)
    if stem_type == 'hihat':
        # Run the final classification logic
        print("Running hihat classification (open/closed/handclap)...")
        
        # Prepare data for detect_hihat_state
        hihat_onset_times = np.array([row['time'] for row in all_onset_data])
        sustain_durations = [row.get('sustain_ms') for row in all_onset_data]
        spectral_data = [
            {
                'primary_energy': row['primary_energy'],
                'secondary_energy': row['secondary_energy']
            }
            for row in all_onset_data
        ]
        
        # Get open threshold from config
        open_sustain_threshold = config.get('hihat', {}).get('open_sustain_ms', 150)
        
        # Classify using the actual production logic
        hihat_states = detect_hihat_state(
            audio, sr, hihat_onset_times,
            sustain_durations=sustain_durations,
            open_sustain_threshold_ms=open_sustain_threshold,
            spectral_data=spectral_data,
            config=config
        )
        
        # Add detected columns with 'x' markers
        df['detected_open'] = ['x' if state == 'open' else '' for state in hihat_states]
        df['detected_closed'] = ['x' if state == 'closed' else '' for state in hihat_states]
        
        # Add empty columns for user to fill in ground truth
        df['actual_open'] = ''
        df['actual_closed'] = ''
        
        print(f"  Detected: {sum(1 for s in hihat_states if s == 'open')} open, "
              f"{sum(1 for s in hihat_states if s == 'closed')} closed")
    
    # Reorder columns for readability
    column_order = ['time', 'strength', 'amplitude']
    
    # Add stem-specific columns
    if stem_type == 'hihat':
        column_order.extend(['primary_energy', 'secondary_energy', 'total_energy', 
                            'body_wire_geomean', 'sustain_ms', 'Status',
                            'detected_open', 'detected_closed',
                            'actual_open', 'actual_closed'])
    elif stem_type == 'kick':
        if 'tertiary_energy' in df.columns:
            column_order.extend(['primary_energy', 'secondary_energy', 'tertiary_energy', 
                                'total_energy', 'body_wire_geomean', 'Status'])
        else:
            column_order.extend(['primary_energy', 'secondary_energy', 'total_energy', 
                                'body_wire_geomean', 'Status'])
    else:
        column_order.extend(['primary_energy', 'secondary_energy', 'total_energy', 
                            'body_wire_geomean', 'Status'])
    
    # Keep only columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Rename columns to match debugging_scripts format
    rename_map = {
        'time': 'Time',
        'strength': 'Str',
        'amplitude': 'Amp',
        'primary_energy': 'BodyE' if stem_type != 'kick' else 'FundE',
        'secondary_energy': 'SizzleE' if stem_type == 'hihat' else 'WireE' if stem_type == 'snare' else 'BodyE',
        'tertiary_energy': 'AttackE',  # Kick only
        'total_energy': 'Total',
        'body_wire_geomean': 'GeoMean',
        'sustain_ms': 'SustainMs'
    }
    df = df.rename(columns=rename_map)
    
    # Remove Status column - we'll populate from actual MIDI output
    if 'Status' in df.columns:
        df = df.drop(columns=['Status'])
    
    # Remove detected columns - we'll populate from actual MIDI output
    if 'detected_open' in df.columns:
        df = df.drop(columns=['detected_open', 'detected_closed'])
    
    # Save initial CSV with all onsets
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Initial features exported to: {output_csv}")
    print(f"  Total onsets detected: {len(df)}")
    print()
    
    # Now generate MIDI using actual detection logic
    print("Running MIDI generation to capture actual detection output...")
    import subprocess
    
    midi_dir = project_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_path = midi_dir / f"06_Taylor_Swift_Ruin_The_Friendship_Drums-{stem_type}.mid"
    
    # Run the actual MIDI generation via subprocess
    cmd = [
        'python', 'stems_to_midi.py', str(project_number),
        '--stem', stem_type
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
    
    if result.returncode != 0:
        print(f"Warning: MIDI generation failed:")
        print(result.stderr)
        print("Continuing with CSV export only...")
        return output_csv
    
    print(f"  MIDI generation complete")
    
    # Find the MIDI file (might be named with or without stem type)
    midi_files = list(midi_dir.glob("*.mid"))
    if not midi_files:
        print(f"Warning: Could not find MIDI file in {midi_dir}")
        print("Continuing with CSV export only...")
        return output_csv
    
    # Use the most recently modified MIDI file
    midi_path = max(midi_files, key=lambda p: p.stat().st_mtime)
    print(f"  Reading MIDI from: {midi_path.name}")
    
    # Read back the MIDI file to see what was actually written
    import mido
    mid = mido.MidiFile(str(midi_path))
    
    # Extract note times from MIDI (convert from beats back to seconds)
    tempo = config.get('midi', {}).get('default_tempo', 120.0)
    beats_per_second = tempo / 60.0
    
    midi_note_times = []
    midi_note_types = []  # 'open' or 'closed' for hihat
    
    current_time = 0.0  # in beats
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time / mid.ticks_per_beat
            if msg.type == 'note_on' and msg.velocity > 0:
                time_seconds = current_time / beats_per_second
                midi_note_times.append(time_seconds)
                
                # Determine note type from MIDI note number
                if stem_type == 'hihat':
                    open_note = config.get('hihat', {}).get('midi_note_open', 46)
                    closed_note = config.get('hihat', {}).get('midi_note_closed', 42)
                    if msg.note == open_note:
                        midi_note_types.append('open')
                    elif msg.note == closed_note:
                        midi_note_types.append('closed')
                    else:
                        midi_note_types.append('closed')  # default
    
    print(f"  MIDI file has {len(midi_note_times)} notes")
    
    # Debug: show first few MIDI times
    if midi_note_times:
        print(f"  First 3 MIDI times: {midi_note_times[:3]}")
        print(f"  First 3 CSV times: {df['Time'].head(3).tolist()}")
    
    # Apply timing offset (MIDI times have timing_offset applied, CSV doesn't)
    timing_offset = stem_config.get('timing_offset', 0.0)
    print(f"  Applying timing offset: {timing_offset}s")
    
    # Adjust MIDI times back to match CSV times
    midi_note_times_adjusted = [t - timing_offset for t in midi_note_times]
    
    # Match CSV onsets to MIDI notes (within 20ms tolerance to account for rounding)
    df['detected_closed'] = ''
    df['detected_open'] = ''
    
    for i, row in df.iterrows():
        onset_time = row['Time']
        
        # Find closest MIDI note within tolerance
        matches = []
        for j, midi_time in enumerate(midi_note_times_adjusted):
            time_diff = abs(onset_time - midi_time)
            if time_diff < 0.020:  # 20ms tolerance
                matches.append((j, time_diff, midi_note_types[j] if j < len(midi_note_types) else 'closed'))
        
        if matches:
            # Take closest match
            best_match = min(matches, key=lambda x: x[1])
            note_type = best_match[2]
            
            if note_type == 'open':
                df.at[i, 'detected_open'] = 'x'
            else:
                df.at[i, 'detected_closed'] = 'x'
    
    # Save updated CSV
    df.to_csv(output_csv, index=False)
    
    detected_count = len(df[(df['detected_open'] == 'x') | (df['detected_closed'] == 'x')])
    print(f"\n✓ Features updated with MIDI detection results: {output_csv}")
    print(f"  Total onsets in CSV: {len(df)}")
    print(f"  Matched to MIDI notes: {detected_count}")
    print(f"  Not in MIDI: {len(df) - detected_count}")
    print()
    print(f"Next step: Label ground truth")
    print(f"  1. Open {stem_path} in your DAW")
    print(f"  2. Note timestamps of actual {stem_type} hits")
    print(f"  3. Run: python -m stems_to_midi.optimization.label {project_number} --stem {stem_type}")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from stem for threshold optimization"
    )
    parser.add_argument(
        'project_number',
        type=int,
        help="Project number (e.g., 4 for user_files/4)"
    )
    parser.add_argument(
        '--stem',
        required=True,
        help="Stem type (hihat, kick, snare, etc.)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        help="Output directory (default: user_files/{project}/optimization/)"
    )
    
    args = parser.parse_args()
    
    try:
        output_csv = extract_features_from_stem(
            args.project_number,
            args.stem,
            args.output
        )
        print(f"✓ Features ready for extraction to: {output_csv}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
