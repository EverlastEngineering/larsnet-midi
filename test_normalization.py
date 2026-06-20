#!/usr/bin/env python3
"""Test amplitude normalization on hihat."""

import yaml
from pathlib import Path
from stems_to_midi.processing_shell import process_stem_to_midi
from stems_to_midi.config import DrumMapping

# Load config
with open('midiconfig.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test hihat file
test_file = Path('user_files/14 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-hihat.wav')

if not test_file.exists():
    print(f"Test file not found: {test_file}")
    print("Skipping normalization test.")
else:
    print("=== Testing Hihat with Amplitude Normalization ===\n")
    
    # Ensure normalization is enabled for hihat
    if 'hihat' not in config:
        config['hihat'] = {}
    config['hihat']['normalize_amplitude'] = True
    
    print(f"Config: normalize_amplitude = {config['hihat']['normalize_amplitude']}")
    print(f"Config: target_amplitude = {config['audio']['target_amplitude']}\n")
    
    drum_mapping = DrumMapping()
    
    result = process_stem_to_midi(
        test_file,
        'hihat',
        drum_mapping,
        config,
    )
    
    print(f"\n✓ Detected {len(result['events'])} hihat events")
