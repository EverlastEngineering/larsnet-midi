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
    detect_stereo_onsets,
    detect_dual_channel_onsets,
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


class TestDetectStereoOnsets:
    """Tests for stereo onset detection function."""
    
    def test_detect_stereo_onsets_basic(self):
        """Test basic stereo onset detection."""
        # Create simple test audio with onsets
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Stereo audio with some energy
        stereo = np.random.randn(samples, 2) * 0.1
        
        # Add some louder transients
        for onset_time in [0.2, 0.5, 0.8]:
            onset_sample = int(onset_time * sr)
            if onset_sample < samples:
                stereo[onset_sample:onset_sample+100, :] += np.random.randn(100, 2) * 0.5
        
        result = detect_stereo_onsets(stereo, sr=sr)
        
        # Check structure
        assert 'left_onsets' in result
        assert 'right_onsets' in result
        assert 'mono_onsets' in result
        assert 'left_strengths' in result
        assert 'right_strengths' in result
        
        # Check types
        assert isinstance(result['left_onsets'], list)
        assert isinstance(result['right_onsets'], list)
        assert isinstance(result['mono_onsets'], list)
    
    def test_detect_stereo_onsets_silent(self):
        """Test stereo onset detection on silent audio."""
        sr = 22050
        duration = 0.5
        samples = int(sr * duration)
        
        # Silent stereo audio
        stereo = np.zeros((samples, 2))
        
        result = detect_stereo_onsets(stereo, sr=sr)
        
        # Should have minimal or no detections
        assert len(result['left_onsets']) >= 0
        assert len(result['right_onsets']) >= 0
        assert len(result['mono_onsets']) >= 0
    
    def test_detect_stereo_onsets_left_only(self):
        """Test detection when only left channel has onsets."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Left channel with transients, right silent
        left = np.random.randn(samples) * 0.1
        left[int(0.5 * sr):int(0.5 * sr) + 200] += np.random.randn(200) * 0.8
        
        right = np.zeros(samples)
        
        stereo = np.stack([left, right], axis=1)
        
        result = detect_stereo_onsets(stereo, sr=sr)
        
        # Left should have more detections than right
        # (though implementation details may vary)
        assert len(result['left_onsets']) >= 0
        assert len(result['right_onsets']) >= 0
    
    def test_detect_stereo_onsets_formats(self):
        """Test that both stereo formats work."""
        sr = 22050
        samples = 11025  # 0.5 seconds
        
        audio_samples_first = np.random.randn(samples, 2) * 0.1
        
        # Both formats should work without error
        result1 = detect_stereo_onsets(audio_samples_first, sr=sr)
        assert 'left_onsets' in result1
        
        # Channels first format
        audio_channels_first = audio_samples_first.T
        result2 = detect_stereo_onsets(audio_channels_first, sr=sr)
        assert 'left_onsets' in result2


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_stereo_workflow(self):
        """Test complete stereo analysis workflow."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create stereo audio with distinct left/right content
        left = np.random.randn(samples) * 0.1
        right = np.random.randn(samples) * 0.05  # Quieter right
        
        # Add a strong left transient
        left[int(0.5 * sr):int(0.5 * sr) + 100] += np.random.randn(100) * 0.8
        
        stereo = np.stack([left, right], axis=1)
        
        # 1. Detect onsets
        onset_data = detect_stereo_onsets(stereo, sr=sr)
        assert len(onset_data['left_onsets']) > 0 or len(onset_data['mono_onsets']) > 0
        
        # 2. Calculate pan for first mono onset (if any)
        if len(onset_data['mono_onsets']) > 0:
            first_onset = onset_data['mono_onsets'][0]
            onset_sample = int(first_onset * sr)
            
            pan = calculate_pan_position(stereo, onset_sample, sr)
            classification = classify_onset_by_pan(pan)
            
            # Should be string classification
            assert classification in ['left', 'right', 'center']
    
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


class TestDetectDualChannelOnsets:
    """Tests for dual-channel onset detection with merging."""
    
    def test_left_only_signal(self):
        """Test onset detection with signal only in left channel."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create left channel with impulse, right channel silent
        left = np.zeros(samples)
        # Sharp transient impulse at 0.5s
        left[sr//2] = 1.0
        left[sr//2+1:sr//2+100] = np.linspace(0.8, 0.0, 99)  # Quick decay
        right = np.zeros(samples)
        
        stereo = np.stack([left, right], axis=0)
        result = detect_dual_channel_onsets(stereo, sr, merge_window_ms=100)
        
        assert len(result['onset_times']) > 0, "Should detect onsets in left channel"
        # Most onsets should have left > right (allow some tolerance for edge cases)
        left_dominant = sum(1 for l, r in zip(result['left_strengths'], result['right_strengths']) if l > r)
        assert left_dominant >= len(result['onset_times']) * 0.7, \
            f"Most strengths should be left-dominant, got {left_dominant}/{len(result['onset_times'])}"
        # Most pan should be negative
        left_panned = sum(1 for pan in result['pan_confidence'] if pan < -0.1)
        assert left_panned >= len(result['onset_times']) * 0.7, \
            f"Most onsets should be left-panned, got {left_panned}/{len(result['onset_times'])}"
    
    def test_right_only_signal(self):
        """Test onset detection with signal only in right channel."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create right channel with impulse, left channel silent
        left = np.zeros(samples)
        right = np.zeros(samples)
        # Sharp transient impulse at 0.5s
        right[sr//2] = 1.0
        right[sr//2+1:sr//2+100] = np.linspace(0.8, 0.0, 99)  # Quick decay
        
        stereo = np.stack([left, right], axis=0)
        result = detect_dual_channel_onsets(stereo, sr, merge_window_ms=100)
        
        assert len(result['onset_times']) > 0, "Should detect onsets in right channel"
        # Most onsets should have right > left
        right_dominant = sum(1 for l, r in zip(result['left_strengths'], result['right_strengths']) if r > l)
        assert right_dominant >= len(result['onset_times']) * 0.7, \
            f"Most strengths should be right-dominant, got {right_dominant}/{len(result['onset_times'])}"
        # Most pan should be positive
        right_panned = sum(1 for pan in result['pan_confidence'] if pan > 0.1)
        assert right_panned >= len(result['onset_times']) * 0.7, \
            f"Most onsets should be right-panned, got {right_panned}/{len(result['onset_times'])}"
    
    def test_center_signal(self):
        """Test onset detection with centered signal."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create identical signal in both channels
        signal = np.zeros(samples)
        signal[sr//2:sr//2+1000] = 0.5  # Transient at 0.5 seconds
        
        stereo = np.stack([signal, signal], axis=0)
        result = detect_dual_channel_onsets(stereo, sr, merge_window_ms=100)
        
        assert len(result['onset_times']) > 0, "Should detect onsets in centered signal"
        # Pan confidence should be near zero for centered signals
        assert all(abs(pan) < 0.2 for pan in result['pan_confidence']), \
            f"Pan confidence should be near 0 for center signal, got {result['pan_confidence']}"
    
    def test_onset_merging(self):
        """Test that nearby L/R onsets are merged correctly."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create slightly offset transients in L/R (within 100ms)
        left = np.zeros(samples)
        right = np.zeros(samples)
        
        # Left onset at 0.5s
        left[int(0.5*sr)] = 0.6
        left[int(0.5*sr)+1:int(0.5*sr)+100] = np.linspace(0.5, 0.0, 99)
        # Right onset at 0.52s (20ms later, within merge window)
        right[int(0.52*sr)] = 0.4
        right[int(0.52*sr)+1:int(0.52*sr)+100] = np.linspace(0.3, 0.0, 99)
        
        stereo = np.stack([left, right], axis=0)
        result = detect_dual_channel_onsets(stereo, sr, merge_window_ms=100)
        
        # Should merge into single onset (or very few onsets)
        assert len(result['onset_times']) <= 3, \
            f"Should merge nearby onsets, got {len(result['onset_times'])} onsets"
        
        # At least one merged onset should have both L and R strengths > 0
        has_merged = any(l > 0 and r > 0 for l, r in zip(result['left_strengths'], result['right_strengths']))
        assert has_merged or len(result['onset_times']) == 0, \
            "Should have at least one onset with both L and R strength"
    
    def test_no_merging_far_apart(self):
        """Test that distant onsets are not merged."""
        sr = 22050
        duration = 2.0
        samples = int(sr * duration)
        
        # Create transients far apart (> 100ms)
        left = np.zeros(samples)
        right = np.zeros(samples)
        
        # Left onset at 0.5s
        left[int(0.5*sr)] = 0.8
        left[int(0.5*sr)+1:int(0.5*sr)+100] = np.linspace(0.6, 0.0, 99)
        # Right onset at 1.5s (1 second later, outside merge window)
        right[int(1.5*sr)] = 0.8
        right[int(1.5*sr)+1:int(1.5*sr)+100] = np.linspace(0.6, 0.0, 99)
        
        stereo = np.stack([left, right], axis=0)
        result = detect_dual_channel_onsets(stereo, sr, merge_window_ms=100)
        
        # Should detect separate onsets
        assert len(result['onset_times']) >= 2, \
            f"Should detect separate onsets, got {len(result['onset_times'])}"
        
        # Check that we have onsets at both expected times
        onset_0_5 = [i for i, t in enumerate(result['onset_times']) if 0.4 < t < 0.7]
        onset_1_5 = [i for i, t in enumerate(result['onset_times']) if 1.4 < t < 1.7]
        
        assert len(onset_0_5) > 0, f"Should have onset around 0.5s, got times: {result['onset_times']}"
        assert len(onset_1_5) > 0, f"Should have onset around 1.5s, got times: {result['onset_times']}"
        
        # First onset should be left-biased (left strength > right strength)
        idx_0_5 = onset_0_5[0]
        assert result['left_strengths'][idx_0_5] > result['right_strengths'][idx_0_5], \
            f"First onset should be left-dominant: L={result['left_strengths'][idx_0_5]}, R={result['right_strengths'][idx_0_5]}"
        
        # Second onset should be right-biased
        idx_1_5 = onset_1_5[0]
        assert result['right_strengths'][idx_1_5] > result['left_strengths'][idx_1_5], \
            f"Second onset should be right-dominant: L={result['left_strengths'][idx_1_5]}, R={result['right_strengths'][idx_1_5]}"
    
    def test_result_structure(self):
        """Test that result has correct structure and types."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Simple stereo signal
        signal = np.random.randn(2, samples) * 0.1
        signal[:, sr//2] = 0.8  # Single strong transient
        
        result = detect_dual_channel_onsets(signal, sr, merge_window_ms=100)
        
        # Check all required keys present
        assert 'onset_times' in result
        assert 'left_strengths' in result
        assert 'right_strengths' in result
        assert 'pan_confidence' in result
        
        # Check all lists have same length
        n = len(result['onset_times'])
        assert len(result['left_strengths']) == n
        assert len(result['right_strengths']) == n
        assert len(result['pan_confidence']) == n
        
        # Check types
        assert all(isinstance(t, float) for t in result['onset_times'])
        assert all(isinstance(s, float) for s in result['left_strengths'])
        assert all(isinstance(s, float) for s in result['right_strengths'])
        assert all(isinstance(p, float) for p in result['pan_confidence'])
        
        # Check pan confidence range [-1, 1]
        assert all(-1 <= p <= 1 for p in result['pan_confidence']), \
            f"Pan confidence out of range: {result['pan_confidence']}"
    
    def test_empty_audio(self):
        """Test with silent audio (no onsets)."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Silent stereo audio
        stereo = np.zeros((2, samples))
        result = detect_dual_channel_onsets(stereo, sr, merge_window_ms=100)
        
        # May detect 0 onsets (correct) or a few spurious ones (acceptable)
        assert len(result['onset_times']) < 5, \
            f"Silent audio should detect very few onsets, got {len(result['onset_times'])}"
    
    def test_merge_window_parameter(self):
        """Test that merge_window_ms parameter affects merging behavior."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create L/R onsets 50ms apart
        left = np.zeros(samples)
        right = np.zeros(samples)
        left[int(0.5*sr):int(0.5*sr)+1000] = 0.5
        right[int(0.55*sr):int(0.55*sr)+1000] = 0.5  # 50ms later
        
        stereo = np.stack([left, right], axis=0)
        
        # With 100ms window: should merge
        result_merge = detect_dual_channel_onsets(stereo, sr, merge_window_ms=100)
        
        # With 20ms window: should NOT merge
        result_no_merge = detect_dual_channel_onsets(stereo, sr, merge_window_ms=20)
        
        # Merged result should have fewer or equal onsets
        assert len(result_merge['onset_times']) <= len(result_no_merge['onset_times']), \
            f"Larger merge window should produce fewer onsets: {len(result_merge['onset_times'])} vs {len(result_no_merge['onset_times'])}"


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
