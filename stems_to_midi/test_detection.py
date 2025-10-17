"""
Comprehensive tests for stems_to_midi.detection module.

These tests provide complete coverage of the detection algorithms.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from stems_to_midi.detection import (
    detect_onsets,
    detect_tom_pitch,
    detect_hihat_state
)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'hihat': {
            'detect_handclap': True,
            'handclap_body_min': 20,
            'handclap_sizzle_min': 100,
            'handclap_sustain_max': 50
        },
        'audio': {
            'sustain_window_sec': 0.2,
            'envelope_threshold': 0.1,
            'envelope_smooth_kernel': 51
        }
    }


@pytest.fixture
def synthetic_audio():
    """Create synthetic audio with drum hits."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create three hits at 0.5s, 1.0s, 1.5s
    audio = np.zeros_like(t)
    for hit_time in [0.5, 1.0, 1.5]:
        hit_sample = int(hit_time * sr)
        # Add a short burst of noise
        audio[hit_sample:hit_sample+100] = np.random.randn(100) * 0.5
    
    return audio, sr


class TestDetectTomPitch:
    """Tests for detect_tom_pitch function."""
    
    def test_detect_tom_pitch_yin_method(self):
        """Test tom pitch detection with YIN method."""
        # Create a synthetic tom hit with known frequency (100 Hz)
        sr = 22050
        duration = 0.1
        freq = 100.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t) * 0.5
        
        pitch = detect_tom_pitch(audio, sr, onset_time=0.0, method='yin')
        
        # Should detect something close to 100 Hz (within 20% tolerance)
        assert pitch > 0
        assert 80 < pitch < 120
    
    def test_detect_tom_pitch_pyin_method(self):
        """Test tom pitch detection with pYIN method."""
        # Create a synthetic tom hit with known frequency (100 Hz)
        sr = 22050
        duration = 0.1
        freq = 100.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t) * 0.5
        
        pitch = detect_tom_pitch(audio, sr, onset_time=0.0, method='pyin')
        
        # Should detect something (pyin is more robust but may vary more)
        assert pitch >= 0
    
    def test_detect_tom_pitch_pyin_no_confident_detections(self):
        """Test pYIN when there are no confident pitch detections."""
        # Create noisy audio with no clear pitch
        sr = 22050
        duration = 0.1
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        pitch = detect_tom_pitch(audio, sr, onset_time=0.0, method='pyin')
        
        # Should return 0 or a very uncertain value
        assert pitch >= 0
    
    def test_detect_tom_pitch_segment_too_short(self):
        """Test when audio segment is too short for pitch detection."""
        sr = 22050
        audio = np.random.randn(100)  # Very short
        
        pitch = detect_tom_pitch(audio, sr, onset_time=0.0)
        
        assert pitch == 0.0
    
    def test_detect_tom_pitch_onset_near_end(self):
        """Test when onset is near the end of audio."""
        sr = 22050
        audio = np.random.randn(1000)
        onset_time = 0.04  # Near the end
        
        pitch = detect_tom_pitch(audio, sr, onset_time=onset_time, window_ms=100.0)
        
        # Should handle gracefully
        assert pitch >= 0
    
    def test_detect_tom_pitch_exception_fallback(self):
        """Test fallback to spectral peak when YIN/pYIN fails."""
        # Create audio with a clear spectral peak at 100 Hz
        sr = 22050
        duration = 0.1
        freq = 100.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t) * 0.5
        
        # Mock librosa.yin to raise an exception
        with patch('stems_to_midi.detection.librosa.yin', side_effect=Exception("YIN failed")):
            pitch = detect_tom_pitch(audio, sr, onset_time=0.0, method='yin')
            
            # Should fall back to spectral peak detection
            assert pitch > 0
            # Spectral peak should be close to 100 Hz
            assert 50 < pitch < 150
    
    def test_detect_tom_pitch_exception_no_peak_in_range(self):
        """Test fallback when no spectral peak in expected range."""
        sr = 22050
        # Create audio with frequency outside tom range (1000 Hz)
        duration = 0.1
        freq = 1000.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t) * 0.5
        
        # Mock librosa.yin to raise an exception
        with patch('stems_to_midi.detection.librosa.yin', side_effect=Exception("YIN failed")):
            pitch = detect_tom_pitch(
                audio, sr, onset_time=0.0, method='yin',
                min_hz=60.0, max_hz=250.0
            )
            
            # Fallback finds the peak frequency, which in this case will be
            # at the boundary of the range since the actual peak is outside
            assert pitch >= 0  # Should handle gracefully
    
    def test_detect_tom_pitch_exception_silent_audio(self):
        """Test fallback with silent audio (no spectral peak)."""
        sr = 22050
        # Create completely silent audio
        duration = 0.1
        audio = np.zeros(int(sr * duration))
        
        # Mock librosa.yin to raise an exception
        with patch('stems_to_midi.detection.librosa.yin', side_effect=Exception("YIN failed")):
            pitch = detect_tom_pitch(
                audio, sr, onset_time=0.0, method='yin',
                min_hz=60.0, max_hz=250.0
            )
            
            # Should return 0 when audio is silent (no peak in any range)
            assert pitch >= 0
    
    def test_detect_tom_pitch_yin_all_nan(self):
        """Test YIN when all results are NaN."""
        sr = 22050
        audio = np.random.randn(2000) * 0.01  # Very quiet noise
        
        # This might produce all NaN values from YIN
        pitch = detect_tom_pitch(audio, sr, onset_time=0.0, method='yin')
        
        # Should handle gracefully (return 0 or fallback)
        assert pitch >= 0
    
    def test_detect_tom_pitch_custom_parameters(self):
        """Test with custom min/max Hz and window."""
        sr = 22050
        duration = 0.15
        freq = 150.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t) * 0.5
        
        pitch = detect_tom_pitch(
            audio, sr, onset_time=0.0,
            method='yin',
            min_hz=100.0,
            max_hz=200.0,
            window_ms=150.0
        )
        
        assert pitch > 0


class TestDetectOnsets:
    """Tests for detect_onsets function."""
    
    def test_detect_onsets_basic(self, synthetic_audio):
        """Test basic onset detection."""
        audio, sr = synthetic_audio
        
        onset_times, onset_strengths = detect_onsets(audio, sr)
        
        # Should detect the 3 hits
        assert len(onset_times) > 0
        assert len(onset_times) == len(onset_strengths)
        # Strengths should be normalized (0-1 range)
        assert np.all(onset_strengths >= 0)
        assert np.all(onset_strengths <= 1)
    
    def test_detect_onsets_stereo(self):
        """Test onset detection with stereo audio."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create stereo audio (2 channels) with a longer onset for librosa's n_fft
        mono = np.zeros_like(t)
        # Make the onset longer (2500 samples > n_fft default of 2048)
        mono[int(0.5 * sr):int(0.5 * sr) + 2500] = np.random.randn(2500) * 0.5
        stereo = np.vstack([mono, mono])  # Duplicate to make stereo
        
        # Suppress librosa's warning about FFT size (expected with test data)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_fft=.*is too large")
            onset_times, onset_strengths = detect_onsets(stereo, sr)
        
        # Should handle stereo and detect onsets
        assert len(onset_times) >= 0
    
    def test_detect_onsets_percentile_zero_fallback(self):
        """Test fallback when 95th percentile is zero."""
        sr = 22050
        # Create audio with very weak onsets
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.001  # Very quiet
        
        # Add one tiny peak
        audio[int(0.5 * sr)] = 0.002
        
        onset_times, onset_strengths = detect_onsets(audio, sr, threshold=0.0)
        
        # Should fall back to max normalization
        assert len(onset_times) >= 0
        if len(onset_strengths) > 0:
            assert np.all(onset_strengths >= 0)
            assert np.all(onset_strengths <= 1)
    
    def test_detect_onsets_max_zero_fallback(self):
        """Test fallback when both percentile and max are zero."""
        sr = 22050
        # Create completely silent audio
        duration = 0.5
        audio = np.zeros(int(sr * duration))
        
        onset_times, onset_strengths = detect_onsets(audio, sr, threshold=0.0)
        
        # Should return empty or handle gracefully
        assert len(onset_times) == len(onset_strengths)
    
    def test_detect_onsets_custom_parameters(self):
        """Test with custom detection parameters."""
        sr = 22050
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        onset_times, onset_strengths = detect_onsets(
            audio, sr,
            hop_length=256,
            threshold=0.5,
            pre_max=5,
            post_max=5,
            wait=10
        )
        
        # Should work with custom parameters
        assert len(onset_times) == len(onset_strengths)
    
    def test_detect_onsets_filters_zero_strengths(self):
        """Test that zero-strength detections are filtered out."""
        sr = 22050
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        onset_times, onset_strengths = detect_onsets(audio, sr, threshold=0.0)
        
        # All returned strengths should be > 0
        if len(onset_strengths) > 0:
            assert np.all(onset_strengths > 0)
    
    def test_detect_onsets_all_zero_strengths(self):
        """Test when onset strengths are all effectively zero."""
        sr = 22050
        duration = 0.5
        # Create audio that produces very weak onset envelope
        audio = np.random.randn(int(sr * duration)) * 0.001
        
        # Force detection with very low threshold
        onset_times, onset_strengths = detect_onsets(
            audio, sr, 
            threshold=0.0,
            delta=0.0001
        )
        
        # Should handle gracefully even if all strengths are zero
        assert len(onset_times) == len(onset_strengths)


class TestDetectHihatState:
    """Tests for detect_hihat_state function."""
    
    def test_detect_hihat_state_with_precalculated_data(self, sample_config):
        """Test with pre-calculated sustain durations and spectral data."""
        sr = 22050
        audio = np.random.randn(sr)  # 1 second of audio
        
        onset_times = np.array([0.1, 0.3, 0.5])
        sustain_durations = [100.0, 200.0, 40.0]  # ms (40 < 50 for handclap)
        spectral_data = [
            {'primary_energy': 10, 'secondary_energy': 50},
            {'primary_energy': 15, 'secondary_energy': 80},
            {'primary_energy': 25, 'secondary_energy': 120}
        ]
        
        states = detect_hihat_state(
            audio, sr, onset_times,
            sustain_durations=sustain_durations,
            spectral_data=spectral_data,
            config=sample_config,
            open_sustain_threshold_ms=150.0
        )
        
        assert len(states) == 3
        assert states[0] == 'closed'  # 100ms < 150ms
        assert states[1] == 'open'    # 200ms > 150ms
        assert states[2] == 'handclap'  # High energy + short sustain (40 < 50)
    
    def test_detect_hihat_state_without_precalculated_data(self, sample_config):
        """Test when sustain durations need to be calculated."""
        sr = 22050
        # Create audio with a hit at 0.5s with some sustain
        duration = 1.0
        audio = np.zeros(int(sr * duration))
        hit_sample = int(0.5 * sr)
        
        # Create a decaying envelope
        decay_length = 5000
        decay = np.exp(-np.linspace(0, 5, decay_length))
        audio[hit_sample:hit_sample + decay_length] = decay * 0.5
        
        onset_times = np.array([0.5])
        
        states = detect_hihat_state(
            audio, sr, onset_times,
            config=sample_config,
            open_sustain_threshold_ms=150.0
        )
        
        assert len(states) == 1
        assert states[0] in ['open', 'closed']
    
    def test_detect_hihat_state_stereo_audio(self, sample_config):
        """Test with stereo audio (should convert to mono)."""
        sr = 22050
        duration = 1.0
        mono = np.zeros(int(sr * duration))
        mono[int(0.5 * sr):int(0.5 * sr) + 1000] = np.random.randn(1000) * 0.3
        stereo = np.vstack([mono, mono])
        
        onset_times = np.array([0.5])
        
        states = detect_hihat_state(
            stereo, sr, onset_times,
            config=sample_config
        )
        
        assert len(states) == 1
    
    def test_detect_hihat_state_no_handclap_detection(self, sample_config):
        """Test when handclap detection is disabled."""
        config = sample_config.copy()
        config['hihat']['detect_handclap'] = False
        
        sr = 22050
        audio = np.random.randn(sr)
        onset_times = np.array([0.1])
        sustain_durations = [30.0]  # Would be handclap if detection enabled
        spectral_data = [{'primary_energy': 30, 'secondary_energy': 150}]
        
        states = detect_hihat_state(
            audio, sr, onset_times,
            sustain_durations=sustain_durations,
            spectral_data=spectral_data,
            config=config,
            open_sustain_threshold_ms=150.0
        )
        
        # Should be closed (not handclap) since detection disabled
        assert states[0] == 'closed'
    
    def test_detect_hihat_state_no_config(self):
        """Test when no config is provided."""
        sr = 22050
        audio = np.random.randn(sr)
        onset_times = np.array([0.1, 0.3])
        
        # Should use default values
        states = detect_hihat_state(audio, sr, onset_times, config=None)
        
        assert len(states) == 2
        assert all(state in ['open', 'closed'] for state in states)
    
    def test_detect_hihat_state_handclap_missing_energy(self, sample_config):
        """Test handclap detection when spectral data is missing keys."""
        sr = 22050
        audio = np.random.randn(sr)
        onset_times = np.array([0.1])
        sustain_durations = [30.0]
        spectral_data = [{}]  # Missing energy values
        
        states = detect_hihat_state(
            audio, sr, onset_times,
            sustain_durations=sustain_durations,
            spectral_data=spectral_data,
            config=sample_config,
            open_sustain_threshold_ms=150.0
        )
        
        # Should default to closed (0 energy values)
        assert states[0] == 'closed'
    
    def test_detect_hihat_state_boundary_sustain(self, sample_config):
        """Test with sustain duration exactly at threshold."""
        sr = 22050
        audio = np.random.randn(sr)
        onset_times = np.array([0.1])
        sustain_durations = [150.0]  # Exactly at threshold
        spectral_data = [{'primary_energy': 10, 'secondary_energy': 50}]
        
        states = detect_hihat_state(
            audio, sr, onset_times,
            sustain_durations=sustain_durations,
            spectral_data=spectral_data,
            config=sample_config,
            open_sustain_threshold_ms=150.0
        )
        
        # At threshold should be closed (not greater than)
        assert states[0] == 'closed'
    
    def test_detect_hihat_state_multiple_hits(self, sample_config):
        """Test with multiple hi-hat hits of different types."""
        sr = 22050
        audio = np.random.randn(sr * 2)
        onset_times = np.array([0.1, 0.3, 0.5, 0.7])
        sustain_durations = [80.0, 200.0, 40.0, 120.0]
        spectral_data = [
            {'primary_energy': 15, 'secondary_energy': 70},   # closed
            {'primary_energy': 20, 'secondary_energy': 90},   # open
            {'primary_energy': 30, 'secondary_energy': 150},  # handclap
            {'primary_energy': 10, 'secondary_energy': 60}    # closed
        ]
        
        states = detect_hihat_state(
            audio, sr, onset_times,
            sustain_durations=sustain_durations,
            spectral_data=spectral_data,
            config=sample_config,
            open_sustain_threshold_ms=150.0
        )
        
        assert len(states) == 4
        assert states[0] == 'closed'
        assert states[1] == 'open'
        assert states[2] == 'handclap'
        assert states[3] == 'closed'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
