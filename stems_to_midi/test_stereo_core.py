"""
Tests for stereo_core.py - Stereo Audio Analysis

Tests pure functions for analyzing stereo audio and extracting spatial information.
"""

import pytest
import numpy as np
from .stereo_core import (
    separate_channels,
    calculate_pan_position,
    calculate_stereo_width,
    calculate_stereo_features,
    classify_onset_by_pan,
)


class TestSeparateChannels:
    """Tests for channel separation function."""
    
    def test_separate_channels_samples_first(self):
        """Test channel separation with (samples, channels) format."""
        # Shape: (samples, channels) - soundfile style
        stereo = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        left, right = separate_channels(stereo)
        
        np.testing.assert_array_equal(left, np.array([0.1, 0.3, 0.5]))
        np.testing.assert_array_equal(right, np.array([0.2, 0.4, 0.6]))
    
    def test_separate_channels_channels_first(self):
        """Test channel separation with (channels, samples) format."""
        # Shape: (channels, samples) - librosa style
        stereo = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        left, right = separate_channels(stereo)
        
        np.testing.assert_array_equal(left, np.array([0.1, 0.3, 0.5]))
        np.testing.assert_array_equal(right, np.array([0.2, 0.4, 0.6]))
    
    def test_separate_channels_mono_raises(self):
        """Test that mono audio raises ValueError."""
        mono = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="Expected 2D stereo array"):
            separate_channels(mono)
    
    def test_separate_channels_wrong_channels_raises(self):
        """Test that wrong number of channels raises ValueError."""
        # 3 channels instead of 2 (samples, channels) format
        multi = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        with pytest.raises(ValueError, match="Expected stereo audio with 2 channels"):
            separate_channels(multi)


class TestCalculatePanPosition:
    """Tests for pan position calculation."""
    
    def test_calculate_pan_full_left(self):
        """Test pan calculation for full left signal."""
        # Left channel loud, right silent
        stereo = np.zeros((1000, 2))
        stereo[:, 0] = 0.8  # Left
        stereo[:, 1] = 0.0  # Right (silent)
        
        pan = calculate_pan_position(stereo, onset_sample=500, sr=22050)
        
        assert pan < -0.8  # Should be strongly left
        assert pan > -1.1  # Within valid range
    
    def test_calculate_pan_full_right(self):
        """Test pan calculation for full right signal."""
        # Left silent, right loud
        stereo = np.zeros((1000, 2))
        stereo[:, 0] = 0.0  # Left (silent)
        stereo[:, 1] = 0.8  # Right
        
        pan = calculate_pan_position(stereo, onset_sample=500, sr=22050)
        
        assert pan > 0.8  # Should be strongly right
        assert pan < 1.1  # Within valid range
    
    def test_calculate_pan_centered(self):
        """Test pan calculation for centered signal."""
        # Equal amplitude in both channels
        stereo = np.ones((1000, 2)) * 0.5
        
        pan = calculate_pan_position(stereo, onset_sample=500, sr=22050)
        
        assert abs(pan) < 0.1  # Should be near center
    
    def test_calculate_pan_left_biased(self):
        """Test pan calculation for left-biased signal."""
        # Left slightly louder than right
        stereo = np.zeros((1000, 2))
        stereo[:, 0] = 0.7  # Left
        stereo[:, 1] = 0.3  # Right
        
        pan = calculate_pan_position(stereo, onset_sample=500, sr=22050)
        
        assert pan < 0  # Should be negative (left)
        assert pan > -1.0  # But not full left
    
    def test_calculate_pan_silent_audio(self):
        """Test pan calculation for silent audio."""
        stereo = np.zeros((1000, 2))
        
        pan = calculate_pan_position(stereo, onset_sample=500, sr=22050)
        
        assert pan == 0.0  # Silent should return centered
    
    def test_calculate_pan_edge_cases(self):
        """Test pan calculation at audio boundaries."""
        stereo = np.ones((100, 2)) * 0.5
        
        # At start
        pan = calculate_pan_position(stereo, onset_sample=0, sr=22050)
        assert isinstance(pan, float)
        
        # Near end
        pan = calculate_pan_position(stereo, onset_sample=90, sr=22050)
        assert isinstance(pan, float)
    
    def test_calculate_pan_custom_window(self):
        """Test pan calculation with custom window size."""
        stereo = np.zeros((1000, 2))
        stereo[:, 0] = 0.8
        
        pan_short = calculate_pan_position(stereo, 500, 22050, window_ms=5.0)
        pan_long = calculate_pan_position(stereo, 500, 22050, window_ms=50.0)
        
        # Both should detect left pan
        assert pan_short < 0
        assert pan_long < 0


class TestClassifyOnsetByPan:
    """Tests for pan classification function."""
    
    def test_classify_onset_left(self):
        """Test classification of left-panned onset."""
        assert classify_onset_by_pan(-0.8) == 'left'
        assert classify_onset_by_pan(-0.5) == 'left'
        assert classify_onset_by_pan(-0.2) == 'left'
    
    def test_classify_onset_right(self):
        """Test classification of right-panned onset."""
        assert classify_onset_by_pan(0.8) == 'right'
        assert classify_onset_by_pan(0.5) == 'right'
        assert classify_onset_by_pan(0.2) == 'right'
    
    def test_classify_onset_center(self):
        """Test classification of centered onset."""
        assert classify_onset_by_pan(0.0) == 'center'
        assert classify_onset_by_pan(0.1) == 'center'
        assert classify_onset_by_pan(-0.1) == 'center'
    
    def test_classify_onset_threshold_boundary(self):
        """Test classification at threshold boundaries."""
        threshold = 0.15
        
        # Just inside center
        assert classify_onset_by_pan(0.14, threshold) == 'center'
        assert classify_onset_by_pan(-0.14, threshold) == 'center'
        
        # Just outside center
        assert classify_onset_by_pan(0.16, threshold) == 'right'
        assert classify_onset_by_pan(-0.16, threshold) == 'left'
    
    def test_classify_onset_custom_threshold(self):
        """Test classification with custom threshold."""
        # Narrow threshold - more strict center
        assert classify_onset_by_pan(0.08, center_threshold=0.05) == 'right'
        assert classify_onset_by_pan(-0.08, center_threshold=0.05) == 'left'
        
        # Wide threshold - more permissive center
        assert classify_onset_by_pan(0.25, center_threshold=0.3) == 'center'
        assert classify_onset_by_pan(-0.25, center_threshold=0.3) == 'center'


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_pan_calculation_accuracy(self):
        """Test pan calculation accuracy with known signals."""
        sr = 22050
        samples = 1000
        
        # Test cases with known pan positions
        test_cases = [
            (1.0, 0.0, 'left'),   # Full left
            (0.0, 1.0, 'right'),  # Full right
            (0.5, 0.5, 'center'), # Centered
            (0.7, 0.3, 'left'),   # Left-biased
            (0.3, 0.7, 'right'),  # Right-biased
        ]
        
        for left_amp, right_amp, expected_class in test_cases:
            stereo = np.zeros((samples, 2))
            stereo[:, 0] = left_amp
            stereo[:, 1] = right_amp
            
            pan = calculate_pan_position(stereo, 500, sr)
            classification = classify_onset_by_pan(pan)
            
            assert classification == expected_class, \
                f"Expected {expected_class} for L={left_amp}, R={right_amp}, got {classification} (pan={pan:.2f})"


class TestCalculateStereoWidth:
    """Tests for stereo width calculation."""

    def test_mono_signal_zero_width(self):
        """Identical L and R channels produce width ≈ 0."""
        sr = 22050
        samples = sr  # 1 second
        signal = np.sin(2 * np.pi * 440 * np.arange(samples) / sr)
        stereo = np.stack([signal, signal], axis=0)  # identical L/R

        width = calculate_stereo_width(stereo, onset_sample=1000, sr=sr, window_ms=30.0)
        assert width < 0.05, f"Expected near-zero width for mono signal, got {width}"

    def test_full_side_signal_high_width(self):
        """L and R with opposite polarity produce width ≈ 1."""
        sr = 22050
        samples = sr
        signal = np.sin(2 * np.pi * 440 * np.arange(samples) / sr)
        stereo = np.stack([signal, -signal], axis=0)  # full side: mid=0, side=2*signal

        width = calculate_stereo_width(stereo, onset_sample=1000, sr=sr, window_ms=30.0)
        assert width > 0.95, f"Expected near-1 width for full-side signal, got {width}"

    def test_half_correlation_moderate_width(self):
        """Signal on one channel only produces width ≈ 0.5 (half-correlated)."""
        sr = 22050
        signal = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        stereo = np.zeros((2, sr))
        stereo[0, :] = signal  # left only, right silent
        # mid = L, side = L → side/(side+mid) = 0.5

        width = calculate_stereo_width(stereo, onset_sample=1000, sr=sr, window_ms=30.0)
        assert 0.45 < width < 0.55, f"Expected ~0.5 for single-channel signal, got {width}"

    def test_uncorrelated_channels_moderate_width(self):
        """Uncorrelated L/R should produce width ≈ 0.5."""
        sr = 22050
        samples = sr
        rng = np.random.RandomState(42)
        left = rng.randn(samples)
        right = rng.randn(samples)
        stereo = np.stack([left, right], axis=0)

        width = calculate_stereo_width(stereo, onset_sample=1000, sr=sr, window_ms=30.0)
        assert 0.35 < width < 0.65, f"Expected ~0.5 width for uncorrelated noise, got {width}"

    def test_silent_returns_zero(self):
        """Silent audio returns 0."""
        stereo = np.zeros((2, 22050))
        width = calculate_stereo_width(stereo, onset_sample=500, sr=22050)
        assert width == 0.0

    def test_onset_near_end(self):
        """Window truncated near end of audio doesn't crash."""
        sr = 22050
        signal = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        stereo = np.stack([signal, signal * 0.5], axis=0)

        width = calculate_stereo_width(stereo, onset_sample=sr - 10, sr=sr, window_ms=30.0)
        assert 0.0 <= width <= 1.0

    def test_capped_at_one(self):
        """Width never exceeds 1.0 even with extreme side energy."""
        sr = 22050
        samples = sr
        # Left loud, right zero → side = left, mid = left → ratio = 1.0
        stereo = np.zeros((2, samples))
        stereo[0, :] = 1.0

        width = calculate_stereo_width(stereo, onset_sample=1000, sr=sr)
        assert width <= 1.0

    def test_return_type(self):
        """Returns a plain Python float."""
        stereo = np.ones((2, 22050)) * 0.5
        width = calculate_stereo_width(stereo, onset_sample=500, sr=22050)
        assert isinstance(width, float)


class TestCalculateStereoFeatures:
    """Tests for the batch stereo features helper."""

    def test_returns_both_keys(self):
        """Each result dict contains pan_confidence and stereo_width."""
        sr = 22050
        signal = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        stereo = np.stack([signal, signal], axis=0)
        onset_times = np.array([0.1, 0.5])

        features = calculate_stereo_features(stereo, onset_times, sr)
        assert len(features) == 2
        for f in features:
            assert 'pan_confidence' in f
            assert 'stereo_width' in f

    def test_mono_fallback(self):
        """1-D mono audio returns zeros for all onsets."""
        mono = np.zeros(22050)
        features = calculate_stereo_features(mono, np.array([0.1, 0.5]), sr=22050)
        assert len(features) == 2
        for f in features:
            assert f['pan_confidence'] == 0.0
            assert f['stereo_width'] == 0.0

    def test_panned_right_positive(self):
        """Right-louder stereo gives positive pan_confidence."""
        sr = 22050
        stereo = np.zeros((2, sr))
        stereo[0, :] = 0.1  # quiet left
        stereo[1, :] = 0.9  # loud right

        features = calculate_stereo_features(stereo, np.array([0.2]), sr)
        assert features[0]['pan_confidence'] > 0.5

    def test_empty_onset_times(self):
        """Empty onset array returns empty list."""
        stereo = np.ones((2, 22050))
        features = calculate_stereo_features(stereo, np.array([]), sr=22050)
        assert features == []
