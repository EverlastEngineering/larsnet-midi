#!/usr/bin/env python3
"""
Quick test to verify --maxtime feature works correctly.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import sys

# Add stems_to_midi to path
sys.path.insert(0, str(Path(__file__).parent))

from stems_to_midi.processor import _load_and_validate_audio
from stems_to_midi.config import load_config


def test_max_duration_truncates_audio():
    """Test that max_duration correctly truncates long audio."""
    
    # Create a 10-second test audio file
    sr = 22050
    duration = 10.0
    audio = np.random.randn(int(sr * duration)) * 0.1
    
    config = load_config()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = Path(f.name)
        sf.write(temp_path, audio, sr)
    
    try:
        # Test 1: Load without truncation
        print("Test 1: Loading full 10-second audio...")
        loaded_audio, loaded_sr = _load_and_validate_audio(
            temp_path, config, 'kick', max_duration=None
        )
        assert loaded_audio is not None
        assert len(loaded_audio) == int(sr * duration)
        print(f"✓ Loaded {len(loaded_audio)} samples ({len(loaded_audio)/sr:.1f} seconds)")
        
        # Test 2: Truncate to 3 seconds
        print("\nTest 2: Loading with max_duration=3.0...")
        loaded_audio, loaded_sr = _load_and_validate_audio(
            temp_path, config, 'kick', max_duration=3.0
        )
        assert loaded_audio is not None
        expected_samples = int(sr * 3.0)
        assert len(loaded_audio) == expected_samples
        print(f"✓ Truncated to {len(loaded_audio)} samples ({len(loaded_audio)/sr:.1f} seconds)")
        
        # Test 3: max_duration larger than audio (should not truncate)
        print("\nTest 3: Loading with max_duration=20.0 (larger than audio)...")
        loaded_audio, loaded_sr = _load_and_validate_audio(
            temp_path, config, 'kick', max_duration=20.0
        )
        assert loaded_audio is not None
        assert len(loaded_audio) == int(sr * duration)  # Original length
        print(f"✓ No truncation, kept {len(loaded_audio)} samples ({len(loaded_audio)/sr:.1f} seconds)")
        
        print("\n✅ All tests passed!")
        
    finally:
        temp_path.unlink()


if __name__ == '__main__':
    test_max_duration_truncates_audio()
