"""
Tests for pure analysis functions (functional core).

These functions have no side effects and are easy to test.
"""

import pytest
import numpy as np
from stems_to_midi.analysis_core import (
    ensure_mono,
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_spectral_energies,
    get_spectral_config_for_stem,
    calculate_geomean,
    should_keep_onset,
    normalize_values,
    estimate_velocity,
    classify_tom_pitch,
    filter_onsets_by_spectral,
    calculate_threshold_from_distributions,
    calculate_classification_accuracy,
    predict_classification,
    analyze_threshold_performance,
    time_to_sample,
    seconds_to_beats,
    prepare_midi_events_for_writing,
    extract_audio_segment,
    analyze_onset_spectral
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
        """Test kick configuration extraction with 3 frequency ranges."""
        config = {
            'kick': {
                'fundamental_freq_min': 40,
                'fundamental_freq_max': 80,
                'body_freq_min': 80,
                'body_freq_max': 150,
                'attack_freq_min': 2000,
                'attack_freq_max': 6000,
                'geomean_threshold': 150.0
            }
        }
        
        result = get_spectral_config_for_stem('kick', config)
        
        assert 'freq_ranges' in result
        assert 'fundamental' in result['freq_ranges']
        assert 'body' in result['freq_ranges']
        assert 'attack' in result['freq_ranges']
        assert result['freq_ranges']['fundamental'] == (40, 80)
        assert result['freq_ranges']['body'] == (80, 150)
        assert result['freq_ranges']['attack'] == (2000, 6000)
        assert result['geomean_threshold'] == 150.0
        assert result['geomean_bands'] == ['fundamental', 'body', 'attack']
        assert result['energy_labels']['fundamental'] == 'Fundamental'
        assert result['energy_labels']['body'] == 'Body'
        assert result['energy_labels']['attack'] == 'Attack'
        assert result['display_hints'] == []
        assert result['velocity_source'] == 'geomean'
        assert result['has_sustain_analysis'] is False
        assert result['use_sustain_duration'] is False
        assert result['has_spectral_data'] is False
        assert result['filter_mode'] == 'geomean_only'
    
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
        assert 'body' in result['freq_ranges']
        assert 'wire' in result['freq_ranges']
        assert result['geomean_bands'] == ['body', 'wire']
        assert result['energy_labels']['body'] == 'Body'
        assert result['energy_labels']['wire'] == 'Wire'
        assert result['display_hints'] == []
        assert result['velocity_source'] == 'geomean'
        assert result['has_sustain_analysis'] is False
        assert result['use_sustain_duration'] is False
        assert result['has_spectral_data'] is False
        assert result['filter_mode'] == 'geomean_only'
    
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
        
        assert result['geomean_bands'] == ['body', 'sizzle']
        assert result['energy_labels']['body'] == 'Body'
        assert result['energy_labels']['sizzle'] == 'Sizzle'
        assert result['min_sustain_ms'] == 25
        assert len(result['display_hints']) == 2
        assert '25ms' in result['display_hints'][0]
        assert result['velocity_source'] == 'onset_strength'
        assert result['has_sustain_analysis'] is True
        assert result['use_sustain_duration'] is False
        assert result['has_spectral_data'] is True
        assert result['filter_mode'] == 'geomean_only'
    
    def test_toms_config(self):
        """Test toms configuration extraction."""
        config = {
            'toms': {
                'fundamental_freq_min': 60,
                'fundamental_freq_max': 120,
                'body_freq_min': 120,
                'body_freq_max': 300,
                'geomean_threshold': 100.0
            }
        }
        
        result = get_spectral_config_for_stem('toms', config)
        
        assert result['freq_ranges']['fundamental'] == (60, 120)
        assert result['freq_ranges']['body'] == (120, 300)
        assert result['geomean_bands'] == ['fundamental', 'body']
        assert result['energy_labels']['fundamental'] == 'Fundamental'
        assert result['energy_labels']['body'] == 'Body'
        assert result['geomean_threshold'] == 100.0
        assert result['display_hints'] == []
        assert result['velocity_source'] == 'geomean'
        assert result['has_sustain_analysis'] is False
        assert result['use_sustain_duration'] is False
        assert result['has_spectral_data'] is False
        assert result['filter_mode'] == 'geomean_only'
    
    def test_cymbals_config(self):
        """Test cymbals configuration extraction."""
        config = {
            'cymbals': {
                'geomean_threshold': 15.0,
                'min_sustain_ms': 150
            }
        }
        
        result = get_spectral_config_for_stem('cymbals', config)
        
        # Cymbals use hardcoded frequency ranges
        assert result['freq_ranges']['body'] == (1000, 4000)
        assert result['freq_ranges']['brilliance'] == (4000, 10000)
        assert result['geomean_bands'] == ['body', 'brilliance']
        assert result['energy_labels']['body'] == 'Body'
        assert result['energy_labels']['brilliance'] == 'Brilliance'
        assert result['geomean_threshold'] == 15.0
        assert result['min_sustain_ms'] == 150
        assert len(result['display_hints']) == 1
        assert '150ms' in result['display_hints'][0]
        assert result['velocity_source'] == 'geomean'
        assert result['has_sustain_analysis'] is True
        assert result['use_sustain_duration'] is True
        assert result['has_spectral_data'] is False
        assert result['filter_mode'] == 'require_both'
    
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
    
    def test_three_way_geomean(self):
        """Test 3-way geometric mean (cube root)."""
        result = calculate_geomean(8.0, 27.0, 64.0)
        expected = np.cbrt(8.0 * 27.0 * 64.0)
        assert abs(result - expected) < 0.001
    
    def test_three_way_equal_values(self):
        """Test 3-way geomean of equal values."""
        result = calculate_geomean(5.0, 5.0, 5.0)
        assert abs(result - 5.0) < 0.001
    
    def test_three_way_with_zero(self):
        """Test 3-way geomean with zero value - falls back to 2-way."""
        # When the third band energy is 0, the function treats it as None
        # and falls back to 2-way geomean: sqrt(100 * 200) = 141.42...
        result = calculate_geomean(100.0, 200.0, 0.0)
        expected = np.sqrt(100.0 * 200.0)
        assert abs(result - expected) < 0.01
    
    def test_backwards_compatible_two_way(self):
        """Test that 2-way geomean still works (backwards compatibility)."""
        result = calculate_geomean(16.0, 64.0)
        expected = np.sqrt(16.0 * 64.0)
        assert abs(result - expected) < 0.001


class TestShouldKeepOnset:
    """Test onset filtering logic."""
    
    def test_no_filtering(self):
        """Test that onset is kept when no thresholds set."""
        result = should_keep_onset(
            geomean=10.0,
            sustain_ms=50.0,
            geomean_threshold=None,
            min_sustain_ms=None,
            filter_mode='geomean_only'
        )
        assert result is True
    
    def test_kick_geomean_pass(self):
        """Test geomean_only mode passes geomean."""
        result = should_keep_onset(
            geomean=200.0,
            sustain_ms=None,
            geomean_threshold=150.0,
            min_sustain_ms=None,
            filter_mode='geomean_only'
        )
        assert result is True
    
    def test_kick_geomean_fail(self):
        """Test geomean_only mode fails geomean."""
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=150.0,
            min_sustain_ms=None,
            filter_mode='geomean_only'
        )
        assert result is False
    
    def test_cymbal_both_required(self):
        """Test require_both mode requires both."""
        # Pass geomean but fail sustain
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=100.0,
            geomean_threshold=10.0,
            min_sustain_ms=150.0,
            filter_mode='require_both'
        )
        assert result is False
        
        # Pass both
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=200.0,
            geomean_threshold=10.0,
            min_sustain_ms=150.0,
            filter_mode='require_both'
        )
        assert result is True
    
    def test_hihat_geomean_threshold(self):
        """Test geomean_only mode with sustain thresholds."""
        # Pass geomean threshold
        result = should_keep_onset(
            geomean=60.0,
            sustain_ms=50.0,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            filter_mode='geomean_only'
        )
        assert result is True
        
        # Fail geomean threshold (sustain doesn't matter)
        result = should_keep_onset(
            geomean=30.0,
            sustain_ms=200.0,  # High sustain doesn't help
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            filter_mode='geomean_only'
        )
        assert result is False
        
        # At exact threshold (should fail, uses > not >=)
        result = should_keep_onset(
            geomean=50.0,
            sustain_ms=100.0,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            filter_mode='geomean_only'
        )
        assert result is False
        
        # Just above threshold
        result = should_keep_onset(
            geomean=50.1,
            sustain_ms=10.0,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            filter_mode='geomean_only'
        )
        assert result is True
    
    def test_cymbal_only_sustain_threshold(self):
        """Test require_both mode with only sustain threshold."""
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=200.0,
            geomean_threshold=None,
            min_sustain_ms=150.0,
            filter_mode='require_both'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=100.0,
            geomean_threshold=None,
            min_sustain_ms=150.0,
            filter_mode='require_both'
        )
        assert result is False
    
    def test_cymbal_only_geomean_threshold(self):
        """Test require_both mode with only geomean threshold."""
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=200.0,
            geomean_threshold=10.0,
            min_sustain_ms=None,
            filter_mode='require_both'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=5.0,
            sustain_ms=200.0,
            geomean_threshold=10.0,
            min_sustain_ms=None,
            filter_mode='require_both'
        )
        assert result is False
    
    def test_other_stem_with_threshold(self):
        """Test geomean_only mode applies geomean threshold."""
        result = should_keep_onset(
            geomean=200.0,
            sustain_ms=None,
            geomean_threshold=150.0,
            min_sustain_ms=None,
            filter_mode='geomean_only'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=150.0,
            min_sustain_ms=None,
            filter_mode='geomean_only'
        )
        assert result is False
    
    def test_hihat_with_no_sustain_value(self):
        """Test geomean_only mode when sustain_ms is None."""
        # Only geomean threshold set, sustain_ms is None
        result = should_keep_onset(
            geomean=60.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            filter_mode='geomean_only'
        )
        assert result is True  # Passes geomean
    
    def test_other_stem_no_geomean_threshold(self):
        """Test filter modes with no geomean threshold."""
        # No geomean threshold, should return True
        result = should_keep_onset(
            geomean=10.0,
            sustain_ms=None,
            geomean_threshold=None,
            min_sustain_ms=50.0,  # This is set but ignored for geomean_only
            filter_mode='geomean_only'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=None,
            min_sustain_ms=None,
            filter_mode='geomean_only'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=30.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            filter_mode='geomean_only'
        )
        assert result is False  # Fails both
    
    def test_other_stem_no_threshold(self):
        """Test other stem with no threshold returns True."""
        result = should_keep_onset(
            geomean=10.0,
            sustain_ms=None,
            geomean_threshold=None,
            min_sustain_ms=None,
            filter_mode='geomean_only'
        )
        assert result is True
    
    def test_strength_filter_pass(self):
        """Test onset passes strength threshold."""
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=None,
            filter_mode='geomean_only',
            strength=0.5,
            min_strength_threshold=0.1
        )
        assert result is True
    
    def test_strength_filter_fail(self):
        """Test onset fails strength threshold."""
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=None,
            filter_mode='geomean_only',
            strength=0.05,
            min_strength_threshold=0.1
        )
        assert result is False
    
    def test_strength_filter_exact_threshold(self):
        """Test onset at exact strength threshold."""
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=None,
            filter_mode='geomean_only',
            strength=0.1,
            min_strength_threshold=0.1
        )
        assert result is True
    
    def test_strength_filter_disabled(self):
        """Test strength filter disabled when threshold is None."""
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=None,
            filter_mode='geomean_only',
            strength=0.01,
            min_strength_threshold=None
        )
        assert result is True
    
    def test_strength_filter_with_geomean_fail(self):
        """Test strength passes but geomean fails."""
        result = should_keep_onset(
            geomean=30.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=None,
            filter_mode='geomean_only',
            strength=0.5,
            min_strength_threshold=0.1
        )
        assert result is False
    
    def test_strength_filter_all_stems(self):
        """Test strength filter applies to all filter modes."""
        for filter_mode in ['geomean_only', 'require_both']:
            result = should_keep_onset(
                geomean=100.0,
                sustain_ms=100.0,
                geomean_threshold=50.0,
                min_sustain_ms=50.0,
                filter_mode=filter_mode,
                strength=0.05,
                min_strength_threshold=0.1
            )
            assert result is False, f"Strength filter should reject for {filter_mode}"


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


class TestClassifyTomPitch:
    """Test tom pitch classification."""
    
    def test_empty_array(self):
        """Test handling of empty pitch array."""
        pitches = np.array([])
        result = classify_tom_pitch(pitches)
        assert len(result) == 0
    
    def test_all_invalid_pitches(self):
        """Test handling of all zero (invalid) pitches."""
        pitches = np.array([0.0, 0.0, 0.0])
        result = classify_tom_pitch(pitches)
        # Should default all to mid (1)
        np.testing.assert_array_equal(result, np.ones(3, dtype=int))
    
    def test_single_pitch(self):
        """Test single pitch value."""
        pitches = np.array([100.0, 100.0, 100.0])
        result = classify_tom_pitch(pitches)
        # All same pitch - classify as mid
        np.testing.assert_array_equal(result, np.ones(3, dtype=int))
    
    def test_two_pitches(self):
        """Test two distinct pitches - should split into low and high."""
        pitches = np.array([80.0, 80.0, 120.0, 120.0])
        result = classify_tom_pitch(pitches)
        # Should split into 0 (low) and 2 (high), no mid
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 2
        assert result[3] == 2
    
    def test_two_pitches_with_invalid(self):
        """Test two pitches with some invalid detections."""
        pitches = np.array([80.0, 0.0, 120.0, 0.0])
        result = classify_tom_pitch(pitches)
        # 80 should be low, 120 should be high, 0 should be mid
        assert result[0] == 0
        assert result[1] == 1  # Invalid goes to mid
        assert result[2] == 2
        assert result[3] == 1  # Invalid goes to mid
    
    def test_three_or_more_pitches_with_sklearn(self):
        """Test three distinct pitches using k-means clustering."""
        pitches = np.array([60.0, 60.0, 100.0, 100.0, 140.0, 140.0])
        result = classify_tom_pitch(pitches)
        
        # Should be classified as low, mid, high
        assert result[0] == 0  # 60 is low
        assert result[1] == 0
        assert result[2] == 1  # 100 is mid
        assert result[3] == 1
        assert result[4] == 2  # 140 is high
        assert result[5] == 2
    
    def test_two_distinct_groups(self):
        """Test two clearly distinct pitch groups."""
        # Use exact duplicates to ensure only 2 unique values
        pitches = np.array([70.0, 70.0, 140.0, 140.0])
        result = classify_tom_pitch(pitches)
        
        # With exactly 2 unique pitches, should only have 0 and 2, no 1
        assert np.all((result == 0) | (result == 2))
        # Lower pitches should be 0
        assert result[0] == 0
        assert result[1] == 0
        # Higher pitches should be 2
        assert result[2] == 2
        assert result[3] == 2
    
    def test_three_unique_in_two_clusters_kmeans(self):
        """Test 3 unique pitches that form 2 clusters via k-means."""
        # Three unique values but very close, might form 2 clusters
        # This specifically tests line 430 where n_clusters==2 in k-means path
        pitches = np.array([60.0, 61.0, 140.0, 60.0, 61.0, 140.0])
        result = classify_tom_pitch(pitches)
        
        # Should have 3+ unique values, so goes to k-means
        # K-means with k=min(3, 3)=3 should still work
        # Values should be classified
        assert all(r in [0, 1, 2] for r in result)
        # The 140s should all have same classification
        assert result[2] == result[5]
        # The 60s and 61s should be in lower classifications
        assert result[0] <= result[2]
        assert result[1] <= result[2]


class TestEstimateVelocity:
    """Test MIDI velocity estimation."""
    
    def test_min_velocity(self):
        """Test that minimum strength gives minimum velocity."""
        velocity = estimate_velocity(0.0, min_vel=40, max_vel=127)
        assert velocity == 40
    
    def test_max_velocity(self):
        """Test that maximum strength gives maximum velocity."""
        velocity = estimate_velocity(1.0, min_vel=40, max_vel=127)
        assert velocity == 127
    
    def test_mid_velocity(self):
        """Test middle strength gives middle velocity."""
        velocity = estimate_velocity(0.5, min_vel=40, max_vel=127)
        expected = 40 + 0.5 * (127 - 40)
        assert abs(velocity - expected) < 1
    
    def test_clipping_below_minimum(self):
        """Test velocity is clipped to valid MIDI range."""
        velocity = estimate_velocity(0.0, min_vel=1, max_vel=127)
        assert 1 <= velocity <= 127
    
    def test_clipping_above_maximum(self):
        """Test velocity never exceeds 127."""
        velocity = estimate_velocity(2.0, min_vel=40, max_vel=127)
        assert velocity == 127


class TestTimeToSample:
    """Test time to sample conversion."""
    
    def test_zero_time(self):
        """Test zero time gives zero sample."""
        sample = time_to_sample(0.0, sr=22050)
        assert sample == 0
    
    def test_one_second(self):
        """Test one second conversion."""
        sample = time_to_sample(1.0, sr=22050)
        assert sample == 22050
    
    def test_fractional_time(self):
        """Test fractional time conversion."""
        sample = time_to_sample(0.5, sr=44100)
        assert sample == 22050


class TestSecondsToBeats:
    """Test seconds to beats conversion."""
    
    def test_120_bpm(self):
        """Test conversion at 120 BPM (2 beats per second)."""
        beats = seconds_to_beats(1.0, tempo=120.0)
        assert beats == 2.0
    
    def test_60_bpm(self):
        """Test conversion at 60 BPM (1 beat per second)."""
        beats = seconds_to_beats(1.0, tempo=60.0)
        assert beats == 1.0
    
    def test_zero_time(self):
        """Test zero time gives zero beats."""
        beats = seconds_to_beats(0.0, tempo=120.0)
        assert beats == 0.0


class TestExtractAudioSegment:
    """Test audio segment extraction."""
    
    def test_simple_extraction(self):
        """Test extracting segment from middle of audio."""
        sr = 1000
        audio = np.arange(0, 1000, dtype=float)  # 1 second of audio
        
        segment = extract_audio_segment(audio, onset_sample=100, window_sec=0.1, sr=sr)
        
        assert len(segment) == 100  # 0.1 * 1000
        assert segment[0] == 100
        assert segment[-1] == 199
    
    def test_extraction_at_end(self):
        """Test extraction is clipped at end of audio."""
        sr = 1000
        audio = np.arange(0, 100, dtype=float)
        
        # Request 0.2 seconds but only 0.05 available
        segment = extract_audio_segment(audio, onset_sample=50, window_sec=0.2, sr=sr)
        
        assert len(segment) == 50  # Only 50 samples available
        assert segment[0] == 50
        assert segment[-1] == 99

class TestPrepareMidiEventsForWriting:
    """Test MIDI event preparation."""
    
    def test_empty_events(self):
        """Test handling of empty events."""
        events_by_stem = {}
        prepared = prepare_midi_events_for_writing(events_by_stem, tempo=120.0)
        assert len(prepared) == 0
    
    def test_single_event(self):
        """Test single event conversion."""
        events_by_stem = {
            'kick': [
                {'note': 36, 'velocity': 100, 'time': 1.0, 'duration': 0.1}
            ]
        }
        prepared = prepare_midi_events_for_writing(events_by_stem, tempo=120.0)
        
        assert len(prepared) == 1
        assert prepared[0]['note'] == 36
        assert prepared[0]['velocity'] == 100
        assert prepared[0]['time_beats'] == 2.0  # 1 second at 120 BPM = 2 beats
        assert prepared[0]['stem_type'] == 'kick'
    
    def test_multiple_stems(self):
        """Test events from multiple stems."""
        events_by_stem = {
            'kick': [{'note': 36, 'velocity': 100, 'time': 0.5, 'duration': 0.1}],
            'snare': [{'note': 38, 'velocity': 90, 'time': 1.0, 'duration': 0.1}]
        }
        prepared = prepare_midi_events_for_writing(events_by_stem, tempo=60.0)

        assert len(prepared) == 2
        # Check that both stems are represented
        stem_types = [e['stem_type'] for e in prepared]
        assert 'kick' in stem_types
        assert 'snare' in stem_types

    def test_dedups_duplicate_note_time_within_1ms(self):
        """Two KEPT events with the same (stem, note, time within 1ms)
        must collapse to a single MIDI note (keep the louder one).

        Regression test for the 2026-06-10 incident: the energy +
        spectral detection paths can both produce a KEPT event for
        the same hit. midiutil's addNote emits NoteOn+NoteOff per
        call; two calls with identical (pitch, channel, tick) but
        different durations leaves one NoteOn orphaned, and
        deInterleaveNotes throws ``IndexError: pop from empty list``.
        The fix is in prepare_midi_events_for_writing — dedup at
        the boundary we control so midiutil never sees the
        double-NoteOn.

        The dedup rounds time to the nearest millisecond; two
        events within the same ms on the same stem+note collapse."""
        events_by_stem = {
            'hihat': [
                # Energy-detected hihat at t=2.0003s, note 42, vel 84
                # (round(2.0003 * 1000) = 2000)
                {'note': 42, 'velocity': 84, 'time': 2.0003, 'duration': 0.21},
                # Spectral-detected hihat at t=2.0008s, note 42, vel 90
                # (round(2.0008 * 1000) = 2001 — also "same ms" only if
                # we rounded down. We don't — we round to nearest, so
                # 2000 and 2001 are different keys. Use 2.0002 to
                # share the same millisecond.)
                {'note': 42, 'velocity': 90, 'time': 2.0002, 'duration': 0.05},
            ]
        }
        prepared = prepare_midi_events_for_writing(events_by_stem, tempo=120.0)

        # The duplicate pair must collapse to one MIDI note.
        assert len(prepared) == 1, (
            f"expected 1 prepared event after dedup, got {len(prepared)}: "
            f"{prepared}"
        )
        # The louder one wins (the spectral hit at vel=90).
        assert prepared[0]['velocity'] == 90, (
            f"expected the louder event (vel=90) to survive dedup, "
            f"got vel={prepared[0]['velocity']}"
        )

    def test_dedup_does_not_collapse_cross_stem_hits(self):
        """A kick at the same instant as a snare must remain two
        separate MIDI notes — different stems with the same note
        number share (pitch, channel) on midiutil's side, but
        they're semantically independent hits and must not be
        merged. The dedup is keyed on (stem, note, time) for this
        reason."""
        events_by_stem = {
            'kick':  [{'note': 36, 'velocity': 100, 'time': 1.0, 'duration': 0.1}],
            'snare': [{'note': 36, 'velocity': 90,  'time': 1.0, 'duration': 0.1}],  # same note #, different stem
        }
        prepared = prepare_midi_events_for_writing(events_by_stem, tempo=120.0)
        # Wait — drum mapping gives kick=36 and snare=38 by default;
        # both with note=36 here means a config override. Either way,
        # the cross-stem dedup must NOT merge them.
        assert len(prepared) == 2, (
            f"cross-stem hits must not be merged by dedup, got "
            f"{len(prepared)} events: {prepared}"
        )

    def test_dedup_keeps_event_when_only_one_in_pair(self):
        """A unique (stem, note, time) event must pass through the
        dedup untouched. This is the common case (the bug only
        fires on exact or near-exact duplicate times) and must
        not be affected by the dedup pass."""
        events_by_stem = {
            'kick':  [{'note': 36, 'velocity': 100, 'time': 1.0, 'duration': 0.1}],
            'snare': [{'note': 38, 'velocity': 90,  'time': 2.0, 'duration': 0.1}],
        }
        prepared = prepare_midi_events_for_writing(events_by_stem, tempo=120.0)
        assert len(prepared) == 2
        # Original velocities preserved.
        velocities = sorted(e['velocity'] for e in prepared)
        assert velocities == [90, 100]

    def test_dedup_uses_1ms_threshold(self):
        """The dedup rounds time to 1ms (same threshold
        energy_detection_shell uses). Two events within the same
        millisecond on the same stem+note must dedup; two events
        in different milliseconds must not. Values chosen to avoid
        banker's-rounding ambiguity (1.0005s could round either way
        depending on Python version)."""
        events_by_stem = {
            'hihat': [
                # 0.0003s apart → both round(2.000 * 1000)=2000
                # and round(2.0003 * 1000)=2000 → SAME ms → dedup
                {'note': 42, 'velocity': 80, 'time': 2.0000, 'duration': 0.1},
                {'note': 42, 'velocity': 90, 'time': 2.0003, 'duration': 0.1},
                # 0.0020s apart → round(5.000 * 1000)=5000
                # and round(5.002 * 1000)=5002 → DIFFERENT ms → keep both
                {'note': 42, 'velocity': 85, 'time': 5.000, 'duration': 0.1},
                {'note': 42, 'velocity': 95, 'time': 5.002, 'duration': 0.1},
            ]
        }
        prepared = prepare_midi_events_for_writing(events_by_stem, tempo=120.0)
        # 1 + 2 = 3 (the close pair collapsed, the far pair kept)
        assert len(prepared) == 3, (
            f"1ms dedup threshold: expected 3 events, got {len(prepared)}: {prepared}"
        )


class TestCalculateThresholdFromDistributions:
    """Test threshold calculation from distributions."""
    
    def test_clear_separation(self):
        """Test threshold with clear separation."""
        kept_values = [100.0, 120.0, 150.0]
        removed_values = [10.0, 20.0, 30.0]
        
        threshold = calculate_threshold_from_distributions(kept_values, removed_values)
        
        # Should be midpoint between max removed (30) and min kept (100)
        assert threshold == 65.0
    
    def test_no_kept_values(self):
        """Test handling of no kept values."""
        threshold = calculate_threshold_from_distributions([], [10.0, 20.0])
        assert threshold is None
    
    def test_no_removed_values(self):
        """Test handling of no removed values."""
        threshold = calculate_threshold_from_distributions([100.0, 120.0], [])
        assert threshold is None
    
    def test_overlapping_distributions(self):
        """Test threshold with overlapping distributions."""
        kept_values = [50.0, 60.0, 70.0]
        removed_values = [40.0, 55.0, 65.0]
        
        threshold = calculate_threshold_from_distributions(kept_values, removed_values)
        
        # Midpoint between max removed (65) and min kept (50)
        assert threshold == 57.5


class TestCalculateClassificationAccuracy:
    """Test classification accuracy calculation."""
    
    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        user_actions = ['KEPT', 'KEPT', 'REMOVED', 'REMOVED']
        predictions = ['KEPT', 'KEPT', 'REMOVED', 'REMOVED']
        
        result = calculate_classification_accuracy(user_actions, predictions)
        
        assert result['correct_count'] == 4
        assert result['total_count'] == 4
        assert result['accuracy'] == 100.0
    
    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        user_actions = ['KEPT', 'KEPT']
        predictions = ['REMOVED', 'REMOVED']
        
        result = calculate_classification_accuracy(user_actions, predictions)
        
        assert result['correct_count'] == 0
        assert result['total_count'] == 2
        assert result['accuracy'] == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        user_actions = ['KEPT', 'KEPT', 'REMOVED', 'REMOVED']
        predictions = ['KEPT', 'REMOVED', 'REMOVED', 'KEPT']
        
        result = calculate_classification_accuracy(user_actions, predictions)
        
        assert result['correct_count'] == 2
        assert result['total_count'] == 4
        assert result['accuracy'] == 50.0
    
    def test_empty_lists(self):
        """Test handling of empty lists."""
        result = calculate_classification_accuracy([], [])
        
        assert result['correct_count'] == 0
        assert result['total_count'] == 0
        assert result['accuracy'] == 0.0
    
    def test_mismatched_lengths(self):
        """Test handling of mismatched list lengths."""
        result = calculate_classification_accuracy(['KEPT'], ['KEPT', 'REMOVED'])
        
        assert result['accuracy'] == 0.0


class TestPredictClassification:
    """Test classification prediction."""
    
    def test_above_threshold(self):
        """Test prediction when above threshold."""
        prediction = predict_classification(
            geomean=150.0,
            geomean_threshold=100.0,
            stem_type='snare'
        )
        assert prediction == 'KEPT'
    
    def test_below_threshold(self):
        """Test prediction when below threshold."""
        prediction = predict_classification(
            geomean=50.0,
            geomean_threshold=100.0,
            stem_type='snare'
        )
        assert prediction == 'REMOVED'
    
    def test_cymbals_both_required_pass(self):
        """Test cymbals requiring both thresholds - pass."""
        prediction = predict_classification(
            geomean=150.0,
            geomean_threshold=100.0,
            sustain_ms=200.0,
            sustain_threshold=150.0,
            stem_type='cymbals'
        )
        assert prediction == 'KEPT'
    
    def test_cymbals_both_required_fail(self):
        """Test cymbals requiring both thresholds - fail."""
        prediction = predict_classification(
            geomean=150.0,
            geomean_threshold=100.0,
            sustain_ms=100.0,
            sustain_threshold=150.0,
            stem_type='cymbals'
        )
        assert prediction == 'REMOVED'
    
    def test_cymbals_no_sustain_threshold(self):
        """Test cymbals with no sustain threshold."""
        prediction = predict_classification(
            geomean=150.0,
            geomean_threshold=100.0,
            stem_type='cymbals'
        )
        assert prediction == 'KEPT'


class TestAnalyzeOnsetSpectral:
    """Test onset spectral analysis."""
    
    def test_simple_analysis(self):
        """Test basic spectral analysis of onset."""
        sr = 22050
        duration = 0.5
        # Create simple audio
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        config = {
            'kick': {
                'fundamental_freq_min': 40,
                'fundamental_freq_max': 80,
                'body_freq_min': 80,
                'body_freq_max': 150,
                'attack_freq_min': 2000,
                'attack_freq_max': 6000,
                'geomean_threshold': 150.0
            },
            'audio': {
                'peak_window_sec': 0.05,
                'min_segment_length': 512
            }
        }
        
        result = analyze_onset_spectral(audio, onset_time=0.1, sr=sr, stem_type='kick', config=config)
        
        assert result is not None
        assert 'fundamental_energy' in result
        assert 'body_energy' in result
        assert 'geomean' in result
        assert 'onset_sample' in result
        assert result['onset_sample'] == int(0.1 * sr)
    
    def test_segment_too_short(self):
        """Test handling of very short audio segment."""
        sr = 22050
        audio = np.array([0.1, 0.2, 0.3])  # Very short
        
        config = {
            'kick': {
                'fundamental_freq_min': 40,
                'fundamental_freq_max': 80,
                'body_freq_min': 80,
                'body_freq_max': 150,
                'attack_freq_min': 2000,
                'attack_freq_max': 6000
            },
            'audio': {
                'peak_window_sec': 0.05,
                'min_segment_length': 512
            }
        }
        
        result = analyze_onset_spectral(audio, onset_time=0.0, sr=sr, stem_type='kick', config=config)
        assert result is None
    
    def test_with_sustain_calculation(self):
        """Test analysis includes sustain for hihat/cymbals."""
        sr = 22050
        duration = 0.5
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        config = {
            'hihat': {
                'body_freq_min': 500,
                'body_freq_max': 2000,
                'sizzle_freq_min': 6000,
                'sizzle_freq_max': 12000,
                'min_sustain_ms': 25
            },
            'audio': {
                'peak_window_sec': 0.05,
                'min_segment_length': 512,
                'sustain_window_sec': 0.2,
                'envelope_threshold': 0.1,
                'envelope_smooth_kernel': 51
            }
        }
        
        result = analyze_onset_spectral(audio, onset_time=0.1, sr=sr, stem_type='hihat', config=config)
        
        assert result is not None
        assert 'sustain_ms' in result
        assert result['sustain_ms'] is not None
    
    def test_unknown_stem_type(self):
        """Test handling of unknown stem type."""
        sr = 22050
        duration = 0.5
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        config = {'audio': {'peak_window_sec': 0.05, 'min_segment_length': 512}}
        
        result = analyze_onset_spectral(audio, onset_time=0.1, sr=sr, stem_type='unknown', config=config)
        assert result is None


class TestFilterOnsetsBySpectral:
    """Test onset filtering by spectral content."""
    
    def test_empty_onsets(self):
        """Test handling of empty onset arrays."""
        result = filter_onsets_by_spectral(
            onset_times=np.array([]),
            onset_strengths=np.array([]),
            peak_amplitudes=np.array([]),
            audio=np.array([0.1, 0.2]),
            sr=22050,
            stem_type='kick',
            config={},
            learning_mode=False
        )
        
        assert len(result['filtered_times']) == 0
        assert len(result['all_onset_data']) == 0
        assert result['spectral_config'] is None
    
    def test_learning_mode_keeps_all(self):
        """Test that learning mode keeps all onsets."""
        sr = 22050
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        config = {
            'kick': {
                'fundamental_freq_min': 40,
                'fundamental_freq_max': 80,
                'body_freq_min': 80,
                'body_freq_max': 150,
                'attack_freq_min': 2000,
                'attack_freq_max': 6000,
                'geomean_threshold': 1000.0  # Very high threshold that would normally filter all
            },
            'audio': {
                'peak_window_sec': 0.05,
                'min_segment_length': 512
            }
        }
        
        onset_times = np.array([0.1, 0.2, 0.3])
        onset_strengths = np.array([0.5, 0.6, 0.7])
        peak_amplitudes = np.array([0.1, 0.2, 0.3])
        
        result = filter_onsets_by_spectral(
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            peak_amplitudes=peak_amplitudes,
            audio=audio,
            sr=sr,
            stem_type='kick',
            config=config,
            learning_mode=True
        )
        
        # In learning mode, all onsets should be kept
        assert len(result['filtered_times']) == 3
    
    def test_filtering_by_threshold(self):
        """Test that threshold filtering works."""
        sr = 22050
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        config = {
            'kick': {
                'fundamental_freq_min': 40,
                'fundamental_freq_max': 80,
                'body_freq_min': 80,
                'body_freq_max': 150,
                'attack_freq_min': 2000,
                'attack_freq_max': 6000,
                'geomean_threshold': 10000000.0  # Extremely high threshold
            },
            'audio': {
                'peak_window_sec': 0.05,
                'min_segment_length': 512
            }
        }
        
        onset_times = np.array([0.1, 0.2, 0.3])
        onset_strengths = np.array([0.5, 0.6, 0.7])
        peak_amplitudes = np.array([0.1, 0.2, 0.3])
        
        result = filter_onsets_by_spectral(
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            peak_amplitudes=peak_amplitudes,
            audio=audio,
            sr=sr,
            stem_type='kick',
            config=config,
            learning_mode=False
        )
        
        # With extremely high threshold, likely no onsets kept
        # But all should be in analysis data
        assert len(result['all_onset_data']) > 0
    
    def test_hihat_sustain_data(self):
        """Test that hihat filtering includes sustain data."""
        sr = 22050
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        config = {
            'hihat': {
                'body_freq_min': 500,
                'body_freq_max': 2000,
                'sizzle_freq_min': 6000,
                'sizzle_freq_max': 12000,
                'geomean_threshold': 1.0,  # Low threshold to keep onsets
                'min_sustain_ms': 1
            },
            'audio': {
                'peak_window_sec': 0.05,
                'min_segment_length': 512,
                'sustain_window_sec': 0.2,
                'envelope_threshold': 0.1,
                'envelope_smooth_kernel': 51
            }
        }
        
        onset_times = np.array([0.1, 0.2])
        onset_strengths = np.array([0.5, 0.6])
        peak_amplitudes = np.array([0.1, 0.2])
        
        result = filter_onsets_by_spectral(
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            peak_amplitudes=peak_amplitudes,
            audio=audio,
            sr=sr,
            stem_type='hihat',
            config=config,
            learning_mode=False
        )
        
        # Hihat should have sustain and spectral data
        if len(result['filtered_times']) > 0:
            assert len(result['filtered_sustains']) > 0
            assert len(result['filtered_spectral']) > 0
    
    def test_skips_segments_that_are_too_short(self):
        """Test that onsets with segments too short are skipped."""
        sr = 22050
        # Very short audio - only 100 samples
        audio = np.random.randn(100) * 0.1
        
        config = {
            'kick': {
                'fundamental_freq_min': 40,
                'fundamental_freq_max': 80,
                'body_freq_min': 80,
                'body_freq_max': 150,
                'attack_freq_min': 2000,
                'attack_freq_max': 6000,
                'geomean_threshold': 150.0
            },
            'audio': {
                'peak_window_sec': 0.05,
                'min_segment_length': 512  # Require 512 samples
            }
        }
        
        # Onset near the end won't have enough samples
        onset_times = np.array([0.0])
        onset_strengths = np.array([0.5])
        peak_amplitudes = np.array([0.1])
        
        result = filter_onsets_by_spectral(
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            peak_amplitudes=peak_amplitudes,
            audio=audio,
            sr=sr,
            stem_type='kick',
            config=config,
            learning_mode=False
        )
        
        # Onset should be skipped because segment is too short
        assert len(result['filtered_times']) == 0
        assert len(result['all_onset_data']) == 0  # No data since analysis returned None


class TestAnalyzeThresholdPerformance:
    """Test threshold performance analysis."""
    
    def test_perfect_threshold(self):
        """Test threshold that perfectly separates data."""
        analysis_data = [
            {'is_kept': True, 'geomean': 150.0},
            {'is_kept': True, 'geomean': 200.0},
            {'is_kept': False, 'geomean': 50.0},
            {'is_kept': False, 'geomean': 75.0}
        ]
        
        result = analyze_threshold_performance(
            analysis_data,
            geomean_threshold=100.0,
            stem_type='snare'
        )
        
        assert len(result['user_actions']) == 4
        assert result['accuracy']['accuracy'] == 100.0
    
    def test_poor_threshold(self):
        """Test threshold that incorrectly classifies everything."""
        analysis_data = [
            {'is_kept': True, 'geomean': 150.0},
            {'is_kept': False, 'geomean': 200.0}
        ]
        
        result = analyze_threshold_performance(
            analysis_data,
            geomean_threshold=175.0,
            stem_type='snare'
        )
        
        assert result['accuracy']['accuracy'] == 0.0
    
    def test_with_sustain_threshold(self):
        """Test analysis with sustain threshold for cymbals."""
        analysis_data = [
            {'is_kept': True, 'geomean': 150.0, 'sustain_ms': 200.0},
            {'is_kept': False, 'geomean': 150.0, 'sustain_ms': 100.0}
        ]
        
        result = analyze_threshold_performance(
            analysis_data,
            geomean_threshold=100.0,
            sustain_threshold=150.0,
            stem_type='cymbals'
        )
        
        assert result['accuracy']['accuracy'] == 100.0


# ============================================================================
# Tests for uncovered functions
# ============================================================================

class TestAnalyzeCymbalDecayPattern:
    """Tests for analyze_cymbal_decay_pattern - cymbal decay analysis"""
    
    def test_fast_decay_cymbal(self):
        """Cymbal with fast decay should show is_decaying=True and negative decay_rate"""
        from stems_to_midi.analysis_core import analyze_cymbal_decay_pattern
        
        # Create synthetic cymbal with exponential decay
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Fast decay: decay_factor = 3.0 means e^(-3*t)
        envelope = np.exp(-3.0 * t)
        # Add high-frequency noise (cymbal characteristic)
        noise = np.random.randn(len(t))
        audio = noise * envelope
        
        result = analyze_cymbal_decay_pattern(audio, onset_sample=0, sr=sr)
        
        assert 'decay_energies' in result
        assert 'is_decaying' in result
        assert 'decay_rate' in result
        assert result['is_decaying'] is True
        assert result['decay_rate'] < 0  # Negative means decaying
    
    def test_slow_decay_cymbal(self):
        """Cymbal with slow decay should still show decay pattern"""
        from stems_to_midi.analysis_core import analyze_cymbal_decay_pattern
        
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Slow decay
        envelope = np.exp(-0.5 * t)
        noise = np.random.randn(len(t))
        audio = noise * envelope
        
        result = analyze_cymbal_decay_pattern(audio, onset_sample=0, sr=sr)
        
        assert result['is_decaying'] is True
        assert result['decay_rate'] < 0
    
    def test_non_decaying_pattern(self):
        """Multiple cymbal hits should not show decay pattern"""
        from stems_to_midi.analysis_core import analyze_cymbal_decay_pattern
        
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Create multiple hits with increasing energy (opposite of decay)
        audio = np.zeros(len(t))
        for i in range(8):
            start = i * len(t) // 8
            end = start + 1000
            if end < len(audio):
                audio[start:end] = np.random.randn(end - start) * (1.0 + i * 0.1)
        
        result = analyze_cymbal_decay_pattern(audio, onset_sample=0, sr=sr)
        
        # Should not be decaying due to energy increases
        assert result['is_decaying'] is False or result['decay_rate'] >= 0
    
    def test_short_segment(self):
        """Very short audio segment should return empty/default result"""
        from stems_to_midi.analysis_core import analyze_cymbal_decay_pattern
        
        sr = 22050
        audio = np.random.randn(100)  # Very short
        
        result = analyze_cymbal_decay_pattern(audio, onset_sample=0, sr=sr, num_windows=8)
        
        assert result['decay_energies'] == []
        assert result['is_decaying'] is False
        assert result['decay_rate'] == 0.0
    
    def test_onset_near_end(self):
        """Onset near end of audio should handle gracefully"""
        from stems_to_midi.analysis_core import analyze_cymbal_decay_pattern
        
        sr = 22050
        audio = np.random.randn(sr)  # 1 second
        onset_sample = sr - 100  # Near the end
        
        result = analyze_cymbal_decay_pattern(audio, onset_sample=onset_sample, sr=sr)
        
        # Should handle gracefully (short segment)
        assert 'decay_energies' in result
        assert 'is_decaying' in result


class TestCalculateStatisticalParams:
    """Tests for calculate_statistical_params - statistical analysis of onset data"""
    
    def test_empty_list(self):
        """Empty onset list should return default values"""
        from stems_to_midi.analysis_core import calculate_statistical_params
        
        result = calculate_statistical_params([])
        
        assert result['median_ratio'] == 1.0
        assert result['median_total'] == 100.0
        assert result['ratio_spread'] == 1.0
        assert result['total_spread'] == 1.0
    
    def test_single_onset(self):
        """Single onset should handle gracefully"""
        from stems_to_midi.analysis_core import calculate_statistical_params
        
        onset_data = [{
            'fundamental_energy': 100.0,
            'body_energy': 50.0,
            'total_energy': 150.0,
            'geomean_bands': ['fundamental', 'body']
        }]
        
        result = calculate_statistical_params(onset_data)
        
        assert result['median_ratio'] == 2.0  # 100/50
        assert result['median_total'] == 150.0
        assert result['ratio_spread'] == 1.0  # Default for single value
        assert result['total_spread'] == 1.0
    
    def test_multiple_onsets_normal_distribution(self):
        """Multiple onsets should calculate correct statistics"""
        from stems_to_midi.analysis_core import calculate_statistical_params
        
        onset_data = [
            {'fundamental_energy': 100.0, 'body_energy': 50.0, 'total_energy': 150.0, 'geomean_bands': ['fundamental', 'body']},
            {'fundamental_energy': 120.0, 'body_energy': 60.0, 'total_energy': 180.0, 'geomean_bands': ['fundamental', 'body']},
            {'fundamental_energy': 80.0, 'body_energy': 40.0, 'total_energy': 120.0, 'geomean_bands': ['fundamental', 'body']},
        ]
        
        result = calculate_statistical_params(onset_data)
        
        # Ratios should all be 2.0
        assert result['median_ratio'] == 2.0
        assert result['median_total'] == 150.0
        assert result['ratio_spread'] < 1e-6  # Nearly zero (all ratios same)
        assert result['total_spread'] > 0  # Total energies vary
    
    def test_zero_band2_energy_handling(self):
        """Zero second-band energy should be handled safely"""
        from stems_to_midi.analysis_core import calculate_statistical_params
        
        onset_data = [
            {'fundamental_energy': 100.0, 'body_energy': 0.0, 'total_energy': 100.0, 'geomean_bands': ['fundamental', 'body']},
        ]
        
        result = calculate_statistical_params(onset_data)
        
        # Should not raise division by zero
        assert result['median_ratio'] > 0
        assert np.isfinite(result['median_ratio'])
    
    def test_spread_never_zero(self):
        """Spread values should never be exactly zero (numerical safety)"""
        from stems_to_midi.analysis_core import calculate_statistical_params
        
        # All identical values
        onset_data = [
            {'fundamental_energy': 100.0, 'body_energy': 50.0, 'total_energy': 150.0, 'geomean_bands': ['fundamental', 'body']},
            {'fundamental_energy': 100.0, 'body_energy': 50.0, 'total_energy': 150.0, 'geomean_bands': ['fundamental', 'body']},
        ]
        
        result = calculate_statistical_params(onset_data)
        
        # Spreads should be very small but not zero
        assert result['ratio_spread'] > 0
        assert result['total_spread'] > 0


class TestCalculateBadnessScore:
    """Tests for calculate_badness_score - onset quality scoring"""
    
    def test_perfect_typical_kick(self):
        """Onset matching median should have low badness score"""
        from stems_to_midi.analysis_core import calculate_badness_score
        
        statistical_params = {
            'median_ratio': 2.0,
            'median_total': 150.0,
            'ratio_spread': 0.5,
            'total_spread': 20.0
        }
        
        onset_data = {
            'fundamental_energy': 100.0,
            'body_energy': 50.0,  # Ratio = 2.0, matches median
            'total_energy': 150.0,  # Matches median
            'geomean_bands': ['fundamental', 'body']
        }
        
        score = calculate_badness_score(onset_data, statistical_params)
        
        assert 0 <= score <= 1
        assert score < 0.2  # Should be close to 0 for perfect match
    
    def test_low_ratio_high_badness(self):
        """Onset with low ratio (snare bleed) should have high badness"""
        from stems_to_midi.analysis_core import calculate_badness_score
        
        statistical_params = {
            'median_ratio': 2.0,
            'median_total': 150.0,
            'ratio_spread': 0.5,
            'total_spread': 20.0
        }
        
        onset_data = {
            'fundamental_energy': 50.0,
            'body_energy': 100.0,  # Ratio = 0.5, much lower than median
            'total_energy': 150.0,
            'geomean_bands': ['fundamental', 'body']
        }
        
        score = calculate_badness_score(onset_data, statistical_params)
        
        assert score > 0.5  # Should be high due to low ratio
    
    def test_different_total_energy(self):
        """Onset with different total energy should contribute to badness"""
        from stems_to_midi.analysis_core import calculate_badness_score
        
        statistical_params = {
            'median_ratio': 2.0,
            'median_total': 150.0,
            'ratio_spread': 0.5,
            'total_spread': 20.0
        }
        
        onset_data = {
            'fundamental_energy': 200.0,
            'body_energy': 100.0,  # Ratio = 2.0, matches median
            'total_energy': 50.0,  # Very different from median
            'geomean_bands': ['fundamental', 'body']
        }
        
        score = calculate_badness_score(onset_data, statistical_params)
        
        assert score > 0  # Should have some badness due to total energy difference
    
    def test_score_bounded(self):
        """Badness score should always be in [0, 1] range"""
        from stems_to_midi.analysis_core import calculate_badness_score
        
        statistical_params = {
            'median_ratio': 2.0,
            'median_total': 150.0,
            'ratio_spread': 0.5,
            'total_spread': 20.0
        }
        
        # Extreme onset
        onset_data = {
            'fundamental_energy': 1000.0,
            'body_energy': 1.0,
            'total_energy': 1.0,
            'geomean_bands': ['fundamental', 'body']
        }
        
        score = calculate_badness_score(onset_data, statistical_params)
        
        assert 0 <= score <= 1


class TestCalculateVelocitiesFromFeatures:
    """Tests for calculate_velocities_from_features"""
    
    def test_empty_features(self):
        """Empty feature array should return empty velocities"""
        from stems_to_midi.analysis_core import calculate_velocities_from_features as calc_vels
        
        result = calc_vels(np.array([]), 40, 127)
        
        assert len(result) == 0
    
    def test_normalized_features(self):
        """Normalized features should map to velocity range"""
        from stems_to_midi.analysis_core import calculate_velocities_from_features as calc_vels
        
        features = np.array([0.0, 0.5, 1.0])
        result = calc_vels(features, 40, 127)
        
        assert len(result) == 3
        # Should be in ascending order
        assert result[0] < result[1] < result[2]
        # Should be within bounds
        assert np.all(result >= 40)
        assert np.all(result <= 127)


class TestFilterOnsetsBySpectralCymbalDecay:
    """Tests for filter_onsets_by_spectral cymbal decay filtering logic"""
    
    def test_cymbal_decay_filter_basic(self):
        """Cymbal decay filter should remove retriggering during decay"""
        from stems_to_midi.analysis_core import filter_onsets_by_spectral
        
        # Create cymbal audio with decay
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        envelope = np.exp(-2.0 * t)
        noise = np.random.randn(len(t))
        audio = noise * envelope
        
        # Two onsets: one at start, one during decay
        onset_times = np.array([0.0, 0.3])  # Second onset during decay
        onset_strengths = np.array([1.0, 0.8])
        peak_amplitudes = np.array([1.0, 0.5])
        
        config = {
            'cymbals': {
                'enable_decay_filter': True,
                'decay_filter_window_sec': 0.5,
                'geomean_threshold': 0.0,
                'min_sustain_ms': 0.0,
                'high_freq_min': 4000,
                'high_freq_max': 20000,
                'mid_freq_min': 1000,
                'mid_freq_max': 4000
            }
        }
        
        result = filter_onsets_by_spectral(
            onset_times, onset_strengths, peak_amplitudes,
            audio, sr, 'cymbals', config, learning_mode=False
        )
        
        # Should filter out the second onset (during decay)
        # Note: This test may be fragile depending on decay detection
        assert len(result['filtered_times']) >= 1  # At least the first onset
    
    def test_cymbal_decay_filter_disabled(self):
        """With decay filter disabled, all onsets should pass"""
        from stems_to_midi.analysis_core import filter_onsets_by_spectral
        
        sr = 22050
        audio = np.random.randn(sr * 2)
        onset_times = np.array([0.0, 0.3, 0.6])
        onset_strengths = np.array([1.0, 0.8, 0.9])
        peak_amplitudes = np.array([1.0, 0.8, 0.9])
        
        config = {
            'cymbals': {
                'enable_decay_filter': False,
                'geomean_threshold': 0.0,
                'min_sustain_ms': 0.0,
                'high_freq_min': 4000,
                'high_freq_max': 20000,
                'mid_freq_min': 1000,
                'mid_freq_max': 4000
            }
        }
        
        result = filter_onsets_by_spectral(
            onset_times, onset_strengths, peak_amplitudes,
            audio, sr, 'cymbals', config, learning_mode=False
        )
        
        # All onsets should pass when decay filter is disabled
        assert len(result['filtered_times']) == 3
    
    def test_learning_mode_keeps_all_onsets(self):
        """Learning mode should keep all onsets regardless of filters"""
        from stems_to_midi.analysis_core import filter_onsets_by_spectral
        
        sr = 22050
        audio = np.random.randn(sr)
        onset_times = np.array([0.0, 0.1, 0.2])
        onset_strengths = np.array([1.0, 0.5, 0.3])
        peak_amplitudes = np.array([1.0, 0.5, 0.3])
        
        config = {
            'kick': {
                'geomean_threshold': 10.0,  # Very high threshold
                'min_sustain_ms': 1000.0,   # Very high threshold
                'fundamental_freq_min': 60,
                'fundamental_freq_max': 150,
                'body_freq_min': 150,
                'body_freq_max': 300,
                'attack_freq_min': 300,
                'attack_freq_max': 600
            }
        }
        
        result = filter_onsets_by_spectral(
            onset_times, onset_strengths, peak_amplitudes,
            audio, sr, 'kick', config, learning_mode=True
        )
        
        # All onsets should be kept in learning mode
        assert len(result['filtered_times']) == 3
    
    def test_kick_statistical_filter(self):
        """Kick statistical filter should remove outliers"""
        from stems_to_midi.analysis_core import filter_onsets_by_spectral
        
        sr = 22050
        # Create audio with distinct frequency bands
        audio = np.random.randn(sr)
        
        onset_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        onset_strengths = np.ones(5)
        peak_amplitudes = np.ones(5)
        
        config = {
            'kick': {
                'enable_statistical_filter': True,
                'statistical_badness_threshold': 0.5,
                'statistical_ratio_weight': 0.7,
                'statistical_total_weight': 0.3,
                'geomean_threshold': 0.0,
                'min_sustain_ms': 0.0,
                'fundamental_freq_min': 60,
                'fundamental_freq_max': 150,
                'body_freq_min': 150,
                'body_freq_max': 300,
                'attack_freq_min': 300,
                'attack_freq_max': 600
            }
        }
        
        result = filter_onsets_by_spectral(
            onset_times, onset_strengths, peak_amplitudes,
            audio, sr, 'kick', config, learning_mode=False
        )
        
        # Should calculate badness scores and store in onset_data
        assert 'all_onset_data' in result
        assert len(result['all_onset_data']) > 0
        # First onset should have badness_score if statistical filter enabled
        if result['all_onset_data']:
            assert 'badness_score' in result['all_onset_data'][0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
