"""
Test suite for stems_to_midi.py

Run with: pytest test_stems_to_midi.py -v
"""

import pytest
import numpy as np
import yaml
from pathlib import Path
import tempfile
import soundfile as sf
from stems_to_midi.config import load_config, DrumMapping
from stems_to_midi.detection import (
    detect_onsets,
    estimate_velocity,
    detect_tom_pitch,
    classify_tom_pitch,
    detect_hihat_state
)
from stems_to_midi.processor import process_stem_to_midi
from stems_to_midi.midi import create_midi_file, read_midi_notes


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_config():
    """Create a minimal valid config for testing."""
    return {
        'audio': {
            'force_mono': True,
        },
        'onset_detection': {
            'threshold': 0.01,
            'delta': 0.005,
            'wait': 1,
            'hop_length': 512
        },
        'kick': {
            'midi_note': 36,
            'fundamental_freq_min': 40,
            'fundamental_freq_max': 80,
            'body_freq_min': 80,
            'body_freq_max': 150,
            'attack_freq_min': 2000,
            'attack_freq_max': 6000,
            'geomean_threshold': 150.0
        },
        'snare': {
            'midi_note': 38,
            'low_freq_min': 40,
            'low_freq_max': 150,
            'body_freq_min': 150,
            'body_freq_max': 400,
            'wire_freq_min': 2000,
            'wire_freq_max': 8000,
            'geomean_threshold': 40.0
        },
        'toms': {
            'midi_note_low': 45,
            'midi_note_mid': 47,
            'midi_note_high': 50,
            'fundamental_freq_min': 60,
            'fundamental_freq_max': 150,
            'body_freq_min': 150,
            'body_freq_max': 400,
            'enable_pitch_detection': True,
            'pitch_method': 'yin',
            'min_pitch_hz': 60,
            'max_pitch_hz': 250,
            'geomean_threshold': 80.0
        },
        'hihat': {
            'midi_note_closed': 42,
            'midi_note_open': 46,
            'midi_note': 42,
            'onset_threshold': 0.05,
            'onset_delta': 0.01,
            'onset_wait': 3,
            'body_freq_min': 500,
            'body_freq_max': 2000,
            'sizzle_freq_min': 6000,
            'sizzle_freq_max': 12000,
            'detect_open': True,
            'open_sustain_ms': 150,
            'min_sustain_ms': 25,
            'geomean_threshold': 50.0
        },
        'cymbals': {
            'midi_note': 49,
            'onset_threshold': 0.15,
            'onset_delta': 0.02,
            'onset_wait': 10,
            'min_sustain_ms': 150,
            'geomean_threshold': 10
        },
        'midi': {
            'min_velocity': 80,
            'max_velocity': 110,
            'default_tempo': 124.0,
            'max_note_duration': 0.5
        },
        'learning_mode': {
            'enabled': False
        }
    }


@pytest.fixture
def synthetic_audio():
    """Create synthetic audio with known onsets for testing."""
    sr = 22050
    duration = 2.0  # 2 seconds
    
    # Create silent audio
    audio = np.zeros(int(sr * duration))
    
    # Add 4 clear transients (impulses) at known times: 0.25s, 0.75s, 1.25s, 1.75s
    onset_times = [0.25, 0.75, 1.25, 1.75]
    onset_amplitudes = [0.8, 0.6, 0.9, 0.5]
    
    for time, amp in zip(onset_times, onset_amplitudes):
        idx = int(time * sr)
        # Create a short transient (100 samples)
        transient_length = 100
        envelope = np.exp(-np.linspace(0, 5, transient_length))
        transient = amp * envelope * np.sin(2 * np.pi * 200 * np.linspace(0, transient_length/sr, transient_length))
        audio[idx:idx+transient_length] = transient
    
    return audio, sr, onset_times, onset_amplitudes


@pytest.fixture
def temp_audio_file(synthetic_audio):
    """Create a temporary audio file for testing."""
    audio, sr, onset_times, onset_amplitudes = synthetic_audio
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = Path(f.name)
        sf.write(temp_path, audio, sr)
    
    yield temp_path, onset_times, onset_amplitudes
    
    # Cleanup
    temp_path.unlink()


@pytest.fixture
def drum_mapping():
    """Create standard drum mapping."""
    return DrumMapping()


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_load_config_default(self):
        """Test loading the default config file."""
        config = load_config()
        assert 'audio' in config
        assert 'onset_detection' in config
        assert 'kick' in config
        assert 'snare' in config
        assert 'midi' in config
    
    def test_config_has_required_fields(self, sample_config):
        """Test that config has all required fields."""
        # Audio section
        assert 'force_mono' in sample_config['audio']
        
        # Onset detection
        assert 'threshold' in sample_config['onset_detection']
        assert 'delta' in sample_config['onset_detection']
        assert 'wait' in sample_config['onset_detection']
        assert 'hop_length' in sample_config['onset_detection']
        
        # Stem-specific configs
        for stem in ['kick', 'snare', 'toms', 'hihat']:
            assert stem in sample_config
            
        # MIDI settings
        assert 'min_velocity' in sample_config['midi']
        assert 'max_velocity' in sample_config['midi']


# ============================================================================
# ONSET DETECTION TESTS
# ============================================================================

class TestOnsetDetection:
    """Test onset detection functionality."""
    
    def test_detect_onsets_synthetic(self, synthetic_audio):
        """Test onset detection on synthetic audio with known transients."""
        audio, sr, expected_times, _ = synthetic_audio
        
        onset_times, onset_strengths = detect_onsets(
            audio, sr, 
            hop_length=512,
            threshold=0.01,
            delta=0.005,
            wait=1
        )
        
        # Should detect close to 4 onsets
        assert len(onset_times) >= 3, f"Expected ~4 onsets, got {len(onset_times)}"
        
        # Check timing accuracy (within 50ms)
        for expected_time in expected_times:
            closest_detected = min(onset_times, key=lambda t: abs(t - expected_time))
            time_error = abs(closest_detected - expected_time)
            assert time_error < 0.05, f"Onset at {expected_time}s off by {time_error*1000:.1f}ms"
    
    def test_detect_onsets_silent(self):
        """Test onset detection on silent audio."""
        sr = 22050
        audio = np.zeros(sr * 1)  # 1 second of silence
        
        onset_times, onset_strengths = detect_onsets(audio, sr)
        
        # Should detect no onsets (or very few spurious ones)
        assert len(onset_times) <= 2, f"Silent audio shouldn't have onsets, got {len(onset_times)}"
    
    def test_detect_onsets_returns_normalized_strengths(self, synthetic_audio):
        """Test that onset strengths are normalized to 0-1 range."""
        audio, sr, _, _ = synthetic_audio
        
        onset_times, onset_strengths = detect_onsets(audio, sr)
        
        assert len(onset_strengths) > 0
        assert np.all(onset_strengths >= 0.0)
        assert np.all(onset_strengths <= 1.0)


# ============================================================================
# VELOCITY ESTIMATION TESTS
# ============================================================================

class TestVelocityEstimation:
    """Test MIDI velocity calculation."""
    
    def test_estimate_velocity_range(self):
        """Test velocity is in valid MIDI range."""
        velocities = [estimate_velocity(s) for s in np.linspace(0, 1, 20)]
        
        assert all(1 <= v <= 127 for v in velocities)
    
    def test_estimate_velocity_min_max(self):
        """Test min and max velocity parameters."""
        min_vel = 50
        max_vel = 100
        
        v_min = estimate_velocity(0.0, min_vel, max_vel)
        v_max = estimate_velocity(1.0, min_vel, max_vel)
        
        assert v_min == min_vel
        assert v_max == max_vel
    
    def test_estimate_velocity_monotonic(self):
        """Test velocity increases with strength."""
        strengths = np.linspace(0, 1, 10)
        velocities = [estimate_velocity(s) for s in strengths]
        
        # Check monotonically increasing
        for i in range(len(velocities) - 1):
            assert velocities[i] <= velocities[i+1]


# ============================================================================
# TOM PITCH DETECTION TESTS
# ============================================================================

class TestTomPitchDetection:
    """Test tom pitch detection and classification."""
    
    def test_classify_tom_pitch_single(self):
        """Test classification with single pitch."""
        pitches = np.array([100.0, 100.0, 100.0])
        classifications = classify_tom_pitch(pitches)
        
        # All same pitch should be classified as mid
        assert np.all(classifications == 1)
    
    def test_classify_tom_pitch_two_groups(self):
        """Test classification with two distinct pitches."""
        pitches = np.array([80.0, 80.0, 160.0, 160.0])
        classifications = classify_tom_pitch(pitches)
        
        # Should split into low (0) and high (2)
        assert np.all(classifications[:2] == 0)  # Low
        assert np.all(classifications[2:] == 2)  # High
    
    def test_classify_tom_pitch_three_groups(self):
        """Test classification with three distinct pitches."""
        pitches = np.array([70.0, 100.0, 180.0])
        classifications = classify_tom_pitch(pitches)
        
        # Should be [low, mid, high]
        assert classifications[0] == 0
        assert classifications[1] == 1
        assert classifications[2] == 2
    
    def test_classify_tom_pitch_handles_zeros(self):
        """Test that failed detections (0 Hz) are handled."""
        pitches = np.array([0.0, 100.0, 0.0, 150.0])
        classifications = classify_tom_pitch(pitches)
        
        # Should return classifications for all, even zeros
        assert len(classifications) == len(pitches)
        assert all(c in [0, 1, 2] for c in classifications)


# ============================================================================
# HI-HAT STATE DETECTION TESTS
# ============================================================================

class TestHiHatStateDetection:
    """Test hi-hat open/closed/handclap detection."""
    
    def test_detect_hihat_state_with_precalculated(self, sample_config):
        """Test hi-hat state detection with pre-calculated sustain durations."""
        onset_times = np.array([0.5, 1.0, 1.5])
        sustain_durations = [80.0, 200.0, 50.0]  # ms (open_sustain_ms is 90 in config)
        spectral_data = [
            {'primary_energy': 180, 'secondary_energy': 200},  # Closed: sustain < threshold
            {'primary_energy': 250, 'secondary_energy': 200},  # Open: high energy + long sustain
            {'primary_energy': 180, 'secondary_energy': 200}   # Closed: sustain < threshold
        ]
        
        # Dummy audio (not used when sustain_durations provided, but needs low peak amplitude)
        sr = 44100
        audio = np.zeros(sr * 2)
        # Add low amplitude transients at onset times for peak detection
        for onset_time in onset_times:
            idx = int(onset_time * sr)
            audio[idx:idx+10] = 0.05  # Very low amplitude for open detection
        
        states = detect_hihat_state(
            audio, sr, onset_times,
            sustain_durations=sustain_durations,
            open_sustain_threshold_ms=90.0,
            spectral_data=spectral_data,
            config=sample_config
        )
        
        assert len(states) == 3
        assert states[0] == 'closed'  # 80ms < 90ms
        assert states[1] == 'open'    # 200ms > 90ms + energy > 200 + low peak
        assert states[2] == 'closed'  # 50ms < 90ms
    
    def test_detect_hihat_handclap(self, sample_config):
        """Test handclap detection."""
        onset_times = np.array([0.5])
        sustain_durations = [30.0]  # Short duration (< handclap_sustain_max of 75ms)
        spectral_data = [
            {'primary_energy': 450, 'secondary_energy': 200}  # High energy (body > 400)
        ]
        
        audio = np.zeros(44100 * 2)
        sr = 44100
        
        # Enable handclap detection with current config values
        config = sample_config.copy()
        config['hihat']['detect_handclap'] = True
        config['hihat']['handclap_body_min'] = 400
        config['hihat']['handclap_sizzle_min'] = 100
        config['hihat']['handclap_sustain_max'] = 75
        
        states = detect_hihat_state(
            audio, sr, onset_times,
            sustain_durations=sustain_durations,
            spectral_data=spectral_data,
            config=config
        )
        
        assert len(states) == 1
        assert states[0] == 'handclap'


# ============================================================================
# DRUM MAPPING TESTS
# ============================================================================

class TestDrumMapping:
    """Test MIDI note mappings."""
    
    def test_drum_mapping_standard_notes(self, drum_mapping):
        """Test standard General MIDI drum notes."""
        assert drum_mapping.kick == 36
        assert drum_mapping.snare == 38
        assert drum_mapping.hihat == 42
        assert drum_mapping.hihat_open == 46
        assert drum_mapping.cymbals == 49
    
    def test_drum_mapping_tom_notes(self, drum_mapping):
        """Test tom note mappings."""
        assert drum_mapping.tom_low == 45
        assert drum_mapping.tom_mid == 47
        assert drum_mapping.tom_high == 50


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestProcessStemToMidi:
    """Integration tests for full stem processing."""
    
    def test_process_stem_returns_events(self, temp_audio_file, sample_config, drum_mapping):
        """Test that processing a stem returns MIDI events."""
        temp_path, expected_times, _ = temp_audio_file
        
        # Disable spectral filtering for this test (set threshold to None)
        test_config = sample_config.copy()
        test_config['kick']['geomean_threshold'] = None
        
        # Extract onset detection parameters from config
        onset_threshold = test_config['onset_detection']['threshold']
        onset_delta = test_config['onset_detection']['delta']
        onset_wait = test_config['onset_detection']['wait']
        hop_length = test_config['onset_detection']['hop_length']
        
        events = process_stem_to_midi(
            temp_path,
            'kick',
            drum_mapping,
            test_config,
            onset_threshold=onset_threshold,
            onset_delta=onset_delta,
            onset_wait=onset_wait,
            hop_length=hop_length,
            detect_hihat_open=False
        )
        
        # Should detect some events
        assert len(events) > 0
        
        # Check event structure
        for event in events:
            assert 'time' in event
            assert 'note' in event
            assert 'velocity' in event
            assert 'duration' in event
            assert 1 <= event['velocity'] <= 127
            assert event['note'] == 36  # Kick note
    
    def test_process_stem_silent_audio(self, sample_config, drum_mapping):
        """Test processing silent audio returns no events."""
        # Create silent audio file
        sr = 22050
        audio = np.zeros(sr * 1)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            sf.write(temp_path, audio, sr)
        
        try:
            # Extract onset detection parameters from config
            onset_threshold = sample_config['onset_detection']['threshold']
            onset_delta = sample_config['onset_detection']['delta']
            onset_wait = sample_config['onset_detection']['wait']
            hop_length = sample_config['onset_detection']['hop_length']
            
            events = process_stem_to_midi(
                temp_path,
                'kick',
                drum_mapping,
                sample_config,
                onset_threshold=onset_threshold,
                onset_delta=onset_delta,
                onset_wait=onset_wait,
                hop_length=hop_length
            )
            
            # Silent audio should produce no events
            assert len(events) == 0
        finally:
            temp_path.unlink()


class TestCreateMidiFile:
    """Test MIDI file creation."""
    
    def test_create_midi_file(self, drum_mapping):
        """Test creating a MIDI file from events."""
        events_by_stem = {
            'kick': [
                {'time': 0.5, 'note': 36, 'velocity': 100, 'duration': 0.1},
                {'time': 1.0, 'note': 36, 'velocity': 90, 'duration': 0.1}
            ],
            'snare': [
                {'time': 0.75, 'note': 38, 'velocity': 110, 'duration': 0.1}
            ]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            create_midi_file(events_by_stem, temp_path, tempo=120.0)
            
            # Check file was created
            assert temp_path.exists()
            assert temp_path.stat().st_size > 0
            
            # Read back and verify
            kick_notes = read_midi_notes(temp_path, 36)
            snare_notes = read_midi_notes(temp_path, 38)
            
            assert len(kick_notes) == 2
            assert len(snare_notes) == 1
        finally:
            if temp_path.exists():
                temp_path.unlink()


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegression:
    """Regression tests to ensure refactoring doesn't break existing behavior."""
    
    def test_config_compatibility(self):
        """Test that current config file is valid and complete."""
        config = load_config()
        
        # Check all stems have required frequency ranges
        for stem in ['kick', 'snare', 'toms', 'hihat']:
            assert stem in config
            stem_config = config[stem]
            
            # All should have some frequency ranges defined
            freq_keys = [k for k in stem_config.keys() if 'freq' in k]
            assert len(freq_keys) > 0, f"{stem} missing frequency ranges"
    
    def test_cymbal_frequency_ranges_exist(self):
        """Test that cymbal config has frequency ranges (currently hardcoded)."""
        config = load_config()
        
        # This test documents that cymbals currently DON'T have freq ranges in config
        # They're hardcoded in the Python. After refactoring, this should change.
        cymbals_config = config.get('cymbals', {})
        
        # Currently these are NOT in config (hardcoded as 1000-4000, 4000-10000)
        # After refactoring, these should exist:
        # assert 'body_freq_min' in cymbals_config
        # assert 'body_freq_max' in cymbals_config
        # assert 'brilliance_freq_min' in cymbals_config
        # assert 'brilliance_freq_max' in cymbals_config
        
        # For now, just check the config loads
        assert 'cymbals' in config


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
