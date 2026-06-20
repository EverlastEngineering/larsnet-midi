#!/usr/bin/env python3
"""
Compare energy-based vs librosa detection for each stem individually.
Processes project 14 (Thunderstruck) one stem at a time in both modes.
"""

import json
from pathlib import Path
from project_manager import get_project_by_number, USER_FILES_DIR
from stems_to_midi.processing_shell import process_stem_to_midi
from stems_to_midi.config import load_config, DrumMapping
from stems_to_midi.midi import create_midi_file

# Stem types to test
STEMS = ['kick', 'snare', 'hihat', 'toms', 'cymbals']

def process_single_stem(project_path: Path, stem_type: str, use_librosa: bool):
    """Process a single stem and return event count and timing info."""
    config = load_config()
    
    # Override detection method
    if stem_type not in config:
        config[stem_type] = {}
    config[stem_type]['use_librosa_detection'] = use_librosa
    
    # Find stem file
    stems_dir = project_path / 'stems'
    stem_files = list(stems_dir.glob(f'*-{stem_type}.wav'))
    if not stem_files:
        return None
    
    stem_path = stem_files[0]
    
    # Create drum mapping from config
    drum_mapping = DrumMapping.from_config(config)
    
    # Get detection parameters from config
    onset_threshold = config.get(stem_type, {}).get('onset_threshold', 0.5)
    onset_delta = config.get(stem_type, {}).get('onset_delta', 0.1)
    onset_wait = config.get(stem_type, {}).get('onset_wait', 20)
    hop_length = config.get('hop_length', 512)
    
    # Process stem
    result = process_stem_to_midi(
        audio_path=stem_path,
        stem_type=stem_type,
        drum_mapping=drum_mapping,
        config=config,
    )
    
    events = result['events']
    
    if not events:
        return {
            'count': 0,
            'first_time': None,
            'last_time': None,
            'avg_velocity': None,
        }
    
    times = [e['time'] for e in events]
    velocities = [e['velocity'] for e in events]
    
    return {
        'count': len(events),
        'first_time': min(times),
        'last_time': max(times),
        'avg_velocity': sum(velocities) / len(velocities),
        'events': events,
    }

def main():
    # Get project 14
    project = get_project_by_number(14, USER_FILES_DIR)
    if project is None:
        print("ERROR: Project 14 not found")
        return
    
    project_path = Path(project['path'])
    
    print("=" * 80)
    print("STEM-BY-STEM DETECTION COMPARISON")
    print("Project: Thunderstruck (Project 14)")
    print("=" * 80)
    
    results = {}
    
    for stem_type in STEMS:
        print(f"\n{'=' * 80}")
        print(f"STEM: {stem_type.upper()}")
        print(f"{'=' * 80}\n")
        
        # Process with energy-based detection
        print(f"Processing {stem_type} with ENERGY-BASED detection...")
        energy_result = process_single_stem(project_path, stem_type, use_librosa=False)
        
        # Process with librosa detection
        print(f"\nProcessing {stem_type} with LIBROSA detection...")
        librosa_result = process_single_stem(project_path, stem_type, use_librosa=True)
        
        if energy_result is None or librosa_result is None:
            print(f"  ⚠️  Stem file not found, skipping...")
            continue
        
        # Store results
        results[stem_type] = {
            'energy': energy_result,
            'librosa': librosa_result,
        }
        
        # Compare and report
        print(f"\n{'─' * 80}")
        print(f"COMPARISON: {stem_type}")
        print(f"{'─' * 80}")
        
        energy_count = energy_result['count']
        librosa_count = librosa_result['count']
        
        print(f"\nEvent Counts:")
        print(f"  Energy-based:  {energy_count:4d} events")
        print(f"  Librosa:       {librosa_count:4d} events")
        
        if energy_count > 0 and librosa_count > 0:
            diff = energy_count - librosa_count
            diff_pct = (diff / librosa_count) * 100
            print(f"  Difference:    {diff:+4d} events ({diff_pct:+.1f}%)")
            
            # Timing comparison
            print(f"\nTiming:")
            print(f"  Energy first:  {energy_result['first_time']:.3f}s")
            print(f"  Librosa first: {librosa_result['first_time']:.3f}s")
            print(f"  Energy last:   {energy_result['last_time']:.3f}s")
            print(f"  Librosa last:  {librosa_result['last_time']:.3f}s")
            
            # Velocity comparison
            print(f"\nAverage Velocity:")
            print(f"  Energy:  {energy_result['avg_velocity']:.1f}")
            print(f"  Librosa: {librosa_result['avg_velocity']:.1f}")
            
            # Analysis
            print(f"\n{'─' * 40}")
            print("ANALYSIS:")
            print(f"{'─' * 40}")
            
            if abs(diff_pct) < 5:
                print("  ✓ Similar event counts (<5% difference)")
            elif diff_pct < -20:
                print("  ⚠️  Energy detection found significantly FEWER events")
                print("     (May be filtering out false positives)")
            elif diff_pct > 20:
                print("  ⚠️  Energy detection found significantly MORE events")
                print("     (May need threshold adjustment)")
            else:
                print("  ℹ️  Moderate difference in event counts")
        
        elif energy_count == 0 and librosa_count == 0:
            print("  ⚠️  Both methods found 0 events")
        elif energy_count == 0:
            print("  ❌ Energy detection found 0 events (PROBLEM)")
        elif librosa_count == 0:
            print("  ❌ Librosa found 0 events (PROBLEM)")
    
    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}\n")
    print(f"{'Stem':<10} {'Energy':<10} {'Librosa':<10} {'Diff':<10} {'Diff %':<10}")
    print(f"{'─' * 60}")
    
    for stem_type in STEMS:
        if stem_type not in results:
            continue
        
        energy_count = results[stem_type]['energy']['count']
        librosa_count = results[stem_type]['librosa']['count']
        diff = energy_count - librosa_count
        
        if librosa_count > 0:
            diff_pct = f"{(diff / librosa_count) * 100:+.1f}%"
        else:
            diff_pct = "N/A"
        
        print(f"{stem_type:<10} {energy_count:<10} {librosa_count:<10} {diff:<10} {diff_pct:<10}")
    
    # Overall assessment
    print(f"\n{'=' * 80}")
    print("OVERALL ASSESSMENT")
    print(f"{'=' * 80}\n")
    
    total_energy = sum(r['energy']['count'] for r in results.values())
    total_librosa = sum(r['librosa']['count'] for r in results.values())
    
    print(f"Total events (all stems):")
    print(f"  Energy-based: {total_energy}")
    print(f"  Librosa:      {total_librosa}")
    print(f"  Difference:   {total_energy - total_librosa:+d} ({((total_energy - total_librosa) / total_librosa * 100):+.1f}%)")
    
    print("\nKey findings:")
    
    # Check for major discrepancies
    problems = []
    for stem_type, data in results.items():
        energy_count = data['energy']['count']
        librosa_count = data['librosa']['count']
        
        if energy_count == 0 and librosa_count > 10:
            problems.append(f"  ❌ {stem_type}: Energy detection found 0 events (threshold too high?)")
        elif librosa_count > 0 and abs((energy_count - librosa_count) / librosa_count) > 0.5:
            problems.append(f"  ⚠️  {stem_type}: Large difference (>50%) - needs calibration")
    
    if problems:
        print("\nIssues found:")
        for problem in problems:
            print(problem)
    else:
        print("  ✓ No major discrepancies found")
        print("  ✓ Energy-based detection appears to be working correctly")
    
    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
