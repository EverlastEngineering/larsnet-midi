"""
Tests for pure helper functions (functional core).

These functions have no side effects and are easy to test.
"""

import pytest
import numpy as np
from stems_to_midi_helpers import (
    ensure_mono,
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean,
    should_keep_onset,
    normalize_values
)


class TestEnsureMono:
    """Test audio channel handling."""
    
    def test_mono_unchanged(self):
        """Test that mono audio passes through unchanged."""
        mono = np.array([1.0, 2.0, 3.0])
        result = ensure_mono(mono)
        np.testing.assert_array_equal(result, mono)
    
    def test_stereo_to_mono(self):
        """Test stereo to mono conversion averages channels."""
        stereo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = ensure_mono(stereo)
        expected = np.array([1.5, 3.5, 5.5])
        np.testing.assert_array_almost_equal(result, expected)


class TestCalculatePeakAmplitude:
    """Test peak amplitude calculation."""
    
    def test_simple_peak(self):
        """Test finding peak in a simple signal."""
        sr = 22050
        audio = np.array([0.1, 0.2, 0.8, 0.3, 0.1])
        
        peak = calculate_peak_amplitude(audio, onset_sample=0, sr=sr, window_ms=10)
        assert peak == 0.8
    
    def test_empty_segment(self):
        """Test handling of empty segment."""
        sr = 22050
        audio = np.array([0.1, 0.2, 0.3])
        
        # Onset beyond audio length
        peak = calculate_peak_amplitude(audio, onset_sample=10, sr=sr, window_ms=10)
        assert peak == 0.0
    
    def test_window_limits(self):
        """Test that window doesn't exceed audio bounds."""
        sr = 22050
        audio = np.array([0.1, 0.2, 0.9, 0.3])
        
        # Window would extend beyond array
        peak = calculate_peak_amplitude(audio, onset_sample=2, sr=sr, window_ms=1000)
        assert peak == 0.9


class TestCalculateSustainDuration:
    """Test sustain duration calculation."""
    
    def test_short_sustain(self):
        """Test detecting short sustain."""
        sr = 1000  # Simple sample rate for easy calculation
        # Create audio with 100ms sustain
        audio = np.ones(200) * 0.5  # 200ms of audio
        audio[:100] = 0.5  # First 100ms at 0.5
        audio[100:] = 0.01  # Rest at very low level
        
        sustain = calculate_sustain_duration(audio, onset_sample=0, sr=sr, window_ms=200)
        # Should detect ~100ms sustain
        assert 80 < sustain < 120
    
    def test_zero_sustain(self):
        """Test audio with no sustain."""
        sr = 1000
        audio = np.zeros(200)
        audio[0] = 0.5  # Single spike
        
        sustain = calculate_sustain_duration(audio, onset_sample=0, sr=sr, window_ms=200)
        # Should be very short
        assert sustain < 50
    
    def test_segment_too_short(self):
        """Test handling of very short segments."""
        sr = 22050
        audio = np.array([0.1, 0.2])
        
        sustain = calculate_sustain_duration(audio, onset_sample=0, sr=sr, window_ms=10)
        assert sustain == 0.0


class TestCalculateSpectralEnergies:
    """Test spectral energy calculation."""
    
    def test_simple_tone(self):
        """Test energy calculation with a pure tone."""
        sr = 1000
        duration = 0.1
        freq = 100  # 100 Hz tone
        
        t = np.linspace(0, duration, int(sr * duration))
        segment = np.sin(2 * np.pi * freq * t)
        
        freq_ranges = {
            'low': (0, 50),
            'target': (80, 120),  # Should contain most energy
            'high': (200, 400)
        }
        
        energies = calculate_spectral_energies(segment, sr, freq_ranges)
        
        # Target range should have most energy
        assert energies['target'] > energies['low']
        assert energies['target'] > energies['high']
    
    def test_empty_segment(self):
        """Test handling of empty segment."""
        sr = 22050
        segment = np.array([0.1, 0.2])  # Too short
        
        freq_ranges = {'low': (0, 100), 'high': (100, 200)}
        energies = calculate_spectral_energies(segment, sr, freq_ranges)
        
        assert energies['low'] == 0.0
        assert energies['high'] == 0.0


class TestGetSpectralConfigForStem:
    """Test spectral configuration extraction."""
    
    def test_kick_config(self):
        """Test kick configuration extraction."""
        config = {
            'kick': {
                'fundamental_freq_min': 40,
                'fundamental_freq_max': 80,
                'body_freq_min': 80,
                'body_freq_max': 150,
                'geomean_threshold': 150.0
            }
        }
        
        result = get_spectral_config_for_stem('kick', config)
        
        assert 'freq_ranges' in result
        assert 'primary' in result['freq_ranges']
        assert 'secondary' in result['freq_ranges']
        assert result['freq_ranges']['primary'] == (40, 80)
        assert result['freq_ranges']['secondary'] == (80, 150)
        assert result['geomean_threshold'] == 150.0
        assert result['energy_labels']['primary'] == 'FundE'
    
    def test_snare_config(self):
        """Test snare configuration extraction."""
        config = {
            'snare': {
                'low_freq_min': 40,
                'low_freq_max': 150,
                'body_freq_min': 150,
                'body_freq_max': 400,
                'wire_freq_min': 2000,
                'wire_freq_max': 8000,
                'geomean_threshold': 40.0
            }
        }
        
        result = get_spectral_config_for_stem('snare', config)
        
        assert 'low' in result['freq_ranges']
        assert result['energy_labels']['secondary'] == 'WireE'
    
    def test_hihat_config(self):
        """Test hihat configuration extraction."""
        config = {
            'hihat': {
                'body_freq_min': 500,
                'body_freq_max': 2000,
                'sizzle_freq_min': 6000,
                'sizzle_freq_max': 12000,
                'geomean_threshold': 50.0,
                'min_sustain_ms': 25
            }
        }
        
        result = get_spectral_config_for_stem('hihat', config)
        
        assert result['energy_labels']['secondary'] == 'SizzleE'
        assert result['min_sustain_ms'] == 25
    
    def test_unknown_stem(self):
        """Test handling of unknown stem type."""
        config = {}
        
        with pytest.raises(ValueError, match="Unknown stem type"):
            get_spectral_config_for_stem('unknown', config)


class TestCalculateGeomean:
    """Test geometric mean calculation."""
    
    def test_simple_geomean(self):
        """Test basic geometric mean."""
        result = calculate_geomean(4.0, 9.0)
        expected = np.sqrt(4.0 * 9.0)
        assert abs(result - expected) < 0.001
    
    def test_zero_values(self):
        """Test geomean with zero values."""
        result = calculate_geomean(0.0, 100.0)
        assert result == 0.0
    
    def test_equal_values(self):
        """Test geomean of equal values."""
        result = calculate_geomean(5.0, 5.0)
        assert result == 5.0


class TestShouldKeepOnset:
    """Test onset filtering logic."""
    
    def test_no_filtering(self):
        """Test that onset is kept when no thresholds set."""
        result = should_keep_onset(
            geomean=10.0,
            sustain_ms=50.0,
            geomean_threshold=None,
            min_sustain_ms=None,
            stem_type='kick'
        )
        assert result is True
    
    def test_kick_geomean_pass(self):
        """Test kick passes geomean threshold."""
        result = should_keep_onset(
            geomean=200.0,
            sustain_ms=None,
            geomean_threshold=150.0,
            min_sustain_ms=None,
            stem_type='kick'
        )
        assert result is True
    
    def test_kick_geomean_fail(self):
        """Test kick fails geomean threshold."""
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=150.0,
            min_sustain_ms=None,
            stem_type='kick'
        )
        assert result is False
    
    def test_cymbal_both_required(self):
        """Test cymbal requires both geomean AND sustain."""
        # Pass geomean but fail sustain
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=100.0,
            geomean_threshold=10.0,
            min_sustain_ms=150.0,
            stem_type='cymbals'
        )
        assert result is False
        
        # Pass both
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=200.0,
            geomean_threshold=10.0,
            min_sustain_ms=150.0,
            stem_type='cymbals'
        )
        assert result is True
    
    def test_hihat_sustain_or_geomean(self):
        """Test hihat passes with EITHER sustain OR geomean."""
        # Pass sustain, fail geomean
        result = should_keep_onset(
            geomean=30.0,
            sustain_ms=50.0,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            stem_type='hihat'
        )
        assert result is True
        
        # Fail sustain, pass geomean
        result = should_keep_onset(
            geomean=60.0,
            sustain_ms=20.0,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            stem_type='hihat'
        )
        assert result is True
        
        # Fail both
        result = should_keep_onset(
            geomean=30.0,
            sustain_ms=20.0,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            stem_type='hihat'
        )
        assert result is False


class TestNormalizeValues:
    """Test value normalization."""
    
    def test_simple_normalization(self):
        """Test basic normalization."""
        values = np.array([10.0, 20.0, 30.0, 40.0])
        result = normalize_values(values)
        
        assert result[0] == 0.25  # 10/40
        assert result[-1] == 1.0  # 40/40
    
    def test_empty_array(self):
        """Test handling of empty array."""
        values = np.array([])
        result = normalize_values(values)
        assert len(result) == 0
    
    def test_all_zeros(self):
        """Test handling of all-zero values."""
        values = np.array([0.0, 0.0, 0.0])
        result = normalize_values(values)
        np.testing.assert_array_equal(result, np.ones(3))
    
    def test_preserves_length(self):
        """Test that normalization preserves array length."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_values(values)
        assert len(result) == len(values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
