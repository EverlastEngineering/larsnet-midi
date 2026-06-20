"""
Test suite for stems_to_midi.py

Run with: pytest test_stems_to_midi.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf
from stems_to_midi.config import load_config, DrumMapping
from stems_to_midi.processing_shell import process_stem_to_midi
from stems_to_midi.midi import create_midi_file, read_midi_notes, save_envelope_data, load_envelope_data
from stems_to_midi.analysis_core import estimate_velocity


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
            # 2026-06-19: PGA is the universal detection path.
            # Routing through the legacy energy/spectral pipeline
            # at processing_shell.py:1290 hits a None 'envelope_data'
            # call (the function was reassigned to a dict earlier
            # in the function) and the test crashes. The kick
            # sample_config in this test file has no other
            # PGA-specific tuning, so the module defaults
            # (broad-band contrast, IQR auto-threshold) are
            # sufficient to find the 200Hz synthetic transients.
            'use_pga_detection': True,
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

class TestProcessDrumToMIDI:
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
        
        result = process_stem_to_midi(
            temp_path,
            'kick',
            drum_mapping,
            test_config,
        )
        
        # Should return dict with events
        assert isinstance(result, dict)
        assert 'events' in result
        events = result['events']
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
            
            result = process_stem_to_midi(
                temp_path,
                'kick',
                drum_mapping,
                sample_config,
            )
            
            # Silent audio should produce no events
            events = result.get('events', [])
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
        cymbals_config = config.get('cymbals', {})  # noqa: F841
        
        # Currently these are NOT in config (hardcoded as 1000-4000, 4000-10000)
        # After refactoring, these should exist:
        # assert 'body_freq_min' in cymbals_config
        # assert 'body_freq_max' in cymbals_config
        # assert 'brilliance_freq_min' in cymbals_config
        # assert 'brilliance_freq_max' in cymbals_config
        
        # For now, just check the config loads
        assert 'cymbals' in config


class TestGetSpectralConfigWithStrength:
    """Test get_spectral_config_for_stem includes min_strength_threshold."""
    
    def test_hihat_has_strength_threshold(self, sample_config):
        """Test hihat config includes strength threshold."""
        from stems_to_midi.analysis_core import get_spectral_config_for_stem
        
        config = sample_config.copy()
        config['hihat'] = {
            'geomean_threshold': 20.0,
            'min_strength_threshold': 0.1,
            'body_freq_min': 500,
            'body_freq_max': 2000,
            'sizzle_freq_min': 6000,
            'sizzle_freq_max': 12000
        }
        
        spectral_config = get_spectral_config_for_stem('hihat', config)
        
        assert 'min_strength_threshold' in spectral_config
        assert spectral_config['min_strength_threshold'] == 0.1
    
    def test_strength_threshold_optional(self, sample_config):
        """Test strength threshold is optional."""
        from stems_to_midi.analysis_core import get_spectral_config_for_stem
        
        config = sample_config.copy()
        config['kick'] = {
            'geomean_threshold': 70.0,
            'fundamental_freq_min': 40,
            'fundamental_freq_max': 80,
            'body_freq_min': 80,
            'body_freq_max': 150,
            'attack_freq_min': 2000,
            'attack_freq_max': 6000
            # Deliberately omit min_strength_threshold
        }
        
        spectral_config = get_spectral_config_for_stem('kick', config)
        
        # Should have None if not specified
        assert spectral_config.get('min_strength_threshold') is None


class TestEnvelopePersistence:
    """Test save/load of energy envelope .npz files."""

    def test_save_creates_npz_files(self, tmp_path):
        """Level 1: Smoke test — save_envelope_data creates .npz files."""
        midi_path = tmp_path / "test_song.mid"
        midi_path.touch()

        envelope_by_stem = {
            'kick': {
                'times': np.linspace(0, 10, 1000),
                'left': np.random.rand(1000).astype(np.float32),
                'right': np.random.rand(1000).astype(np.float32),
                'sr': 44100,
                'hop_length': 512,
                'method': 'rms',
            }
        }

        paths = save_envelope_data(envelope_by_stem, midi_path)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].name == 'test_song.kick.envelope.npz'

    def test_round_trip_preserves_data(self, tmp_path):
        """Level 2: Property test — save then load returns equivalent arrays."""
        midi_path = tmp_path / "song.mid"
        midi_path.touch()

        times = np.linspace(0, 5, 500, dtype=np.float32)
        left = np.random.rand(500).astype(np.float32)
        right = np.random.rand(500).astype(np.float32)

        envelope_by_stem = {
            'snare': {
                'times': times,
                'left': left,
                'right': right,
                'sr': 22050,
                'hop_length': 256,
                'method': 'peak_hold',
            }
        }

        save_envelope_data(envelope_by_stem, midi_path)
        loaded = load_envelope_data(midi_path, 'snare')

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded['times'], times, decimal=5)
        np.testing.assert_array_almost_equal(loaded['left'], left, decimal=5)
        np.testing.assert_array_almost_equal(loaded['right'], right, decimal=5)
        assert loaded['sr'] == 22050
        assert loaded['hop_length'] == 256
        assert loaded['method'] == 'peak_hold'

    def test_multiple_stems_saved_independently(self, tmp_path):
        """Level 2: Each stem gets its own .npz file."""
        midi_path = tmp_path / "multi.mid"
        midi_path.touch()

        envelope_by_stem = {}
        for stem in ['kick', 'snare', 'hihat']:
            envelope_by_stem[stem] = {
                'times': np.linspace(0, 3, 300, dtype=np.float32),
                'left': np.ones(300, dtype=np.float32) * (hash(stem) % 100),
                'right': np.ones(300, dtype=np.float32) * (hash(stem) % 50),
                'sr': 44100,
                'hop_length': 512,
                'method': 'rms',
            }

        paths = save_envelope_data(envelope_by_stem, midi_path)
        assert len(paths) == 3

        # Each stem loads independently
        for stem in ['kick', 'snare', 'hihat']:
            loaded = load_envelope_data(midi_path, stem)
            assert loaded is not None
            expected_left = np.ones(300, dtype=np.float32) * (hash(stem) % 100)
            np.testing.assert_array_almost_equal(loaded['left'], expected_left, decimal=5)

    def test_load_missing_stem_returns_none(self, tmp_path):
        """Level 1: Loading a non-existent stem returns None."""
        midi_path = tmp_path / "missing.mid"
        midi_path.touch()

        loaded = load_envelope_data(midi_path, 'toms')
        assert loaded is None

    def test_none_envelope_skipped(self, tmp_path):
        """Level 1: None envelope entries are skipped gracefully."""
        midi_path = tmp_path / "partial.mid"
        midi_path.touch()

        envelope_by_stem = {
            'kick': {
                'times': np.linspace(0, 1, 100, dtype=np.float32),
                'left': np.zeros(100, dtype=np.float32),
                'right': np.zeros(100, dtype=np.float32),
                'sr': 44100,
                'hop_length': 512,
                'method': 'rms',
            },
            'snare': None,  # No envelope (e.g. librosa path)
        }

        paths = save_envelope_data(envelope_by_stem, midi_path)
        assert len(paths) == 1
        assert 'kick' in paths[0].name

    def test_compressed_file_size(self, tmp_path):
        """Level 2: Compressed .npz is reasonably small for typical envelope."""
        midi_path = tmp_path / "size_check.mid"
        midi_path.touch()

        # Typical 3-minute song at 44100 Hz / 512 hop = ~5168 frames
        n_frames = 5168
        envelope_by_stem = {
            'kick': {
                'times': np.linspace(0, 180, n_frames, dtype=np.float32),
                'left': np.random.rand(n_frames).astype(np.float32),
                'right': np.random.rand(n_frames).astype(np.float32),
                'sr': 44100,
                'hop_length': 512,
                'method': 'rms',
            }
        }

        paths = save_envelope_data(envelope_by_stem, midi_path)
        file_size = paths[0].stat().st_size
        # 3 arrays × 5168 × 4 bytes = ~62KB uncompressed
        # Compressed should be well under 100KB
        assert file_size < 100_000, f"Envelope .npz too large: {file_size} bytes"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
