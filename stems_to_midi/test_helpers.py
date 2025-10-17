"""
Tests for pure helper functions (functional core).

These functions have no side effects and are easy to test.
"""

import pytest
import numpy as np
from stems_to_midi.helpers import (
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
    calculate_velocities_from_features,
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
        
        assert result['freq_ranges']['primary'] == (60, 120)
        assert result['freq_ranges']['secondary'] == (120, 300)
        assert result['energy_labels']['primary'] == 'FundE'
        assert result['energy_labels']['secondary'] == 'BodyE'
        assert result['geomean_threshold'] == 100.0
    
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
        assert result['freq_ranges']['primary'] == (1000, 4000)
        assert result['freq_ranges']['secondary'] == (4000, 10000)
        assert result['energy_labels']['primary'] == 'BodyE'
        assert result['energy_labels']['secondary'] == 'BrillE'
        assert result['geomean_threshold'] == 15.0
        assert result['min_sustain_ms'] == 150
    
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
    
    def test_cymbal_only_sustain_threshold(self):
        """Test cymbal with only sustain threshold set."""
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=200.0,
            geomean_threshold=None,
            min_sustain_ms=150.0,
            stem_type='cymbals'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=100.0,
            geomean_threshold=None,
            min_sustain_ms=150.0,
            stem_type='cymbals'
        )
        assert result is False
    
    def test_cymbal_only_geomean_threshold(self):
        """Test cymbal with only geomean threshold set."""
        result = should_keep_onset(
            geomean=20.0,
            sustain_ms=200.0,
            geomean_threshold=10.0,
            min_sustain_ms=None,
            stem_type='cymbals'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=5.0,
            sustain_ms=200.0,
            geomean_threshold=10.0,
            min_sustain_ms=None,
            stem_type='cymbals'
        )
        assert result is False
    
    def test_other_stem_with_threshold(self):
        """Test other stems (snare, toms) use geomean only."""
        result = should_keep_onset(
            geomean=200.0,
            sustain_ms=None,
            geomean_threshold=150.0,
            min_sustain_ms=None,
            stem_type='snare'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=150.0,
            min_sustain_ms=None,
            stem_type='toms'
        )
        assert result is False
    
    def test_hihat_with_no_sustain_value(self):
        """Test hihat when sustain_ms is None."""
        # Only geomean threshold set, sustain_ms is None
        result = should_keep_onset(
            geomean=60.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            stem_type='hihat'
        )
        assert result is True  # Passes geomean
    
    def test_other_stem_no_geomean_threshold(self):
        """Test other stems with no geomean threshold (line 312)."""
        # No geomean threshold, should return True
        result = should_keep_onset(
            geomean=10.0,
            sustain_ms=None,
            geomean_threshold=None,
            min_sustain_ms=50.0,  # This is set but ignored for non-hihat/cymbal
            stem_type='snare'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=100.0,
            sustain_ms=None,
            geomean_threshold=None,
            min_sustain_ms=None,
            stem_type='toms'
        )
        assert result is True
        
        result = should_keep_onset(
            geomean=30.0,
            sustain_ms=None,
            geomean_threshold=50.0,
            min_sustain_ms=25.0,
            stem_type='hihat'
        )
        assert result is False  # Fails both
    
    def test_other_stem_no_threshold(self):
        """Test other stem with no threshold returns True."""
        result = should_keep_onset(
            geomean=10.0,
            sustain_ms=None,
            geomean_threshold=None,
            min_sustain_ms=None,
            stem_type='snare'
        )
        assert result is True


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


class TestCalculateVelocitiesFromFeatures:
    """Test velocity calculation from feature values."""
    
    def test_empty_array(self):
        """Test handling of empty feature array."""
        velocities = calculate_velocities_from_features(
            np.array([]), min_velocity=40, max_velocity=127
        )
        assert len(velocities) == 0
    
    def test_single_value(self):
        """Test single feature value."""
        velocities = calculate_velocities_from_features(
            np.array([0.5]), min_velocity=40, max_velocity=127
        )
        assert len(velocities) == 1
        assert 40 <= velocities[0] <= 127
    
    def test_multiple_values(self):
        """Test multiple feature values."""
        features = np.array([0.0, 0.5, 1.0])
        velocities = calculate_velocities_from_features(
            features, min_velocity=40, max_velocity=127
        )
        
        assert len(velocities) == 3
        assert velocities[0] == 40  # Min
        assert velocities[2] == 127  # Max
        assert 40 < velocities[1] < 127  # Mid


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
                'geomean_threshold': 150.0
            },
            'audio': {
                'peak_window_sec': 0.05,
                'min_segment_length': 512
            }
        }
        
        result = analyze_onset_spectral(audio, onset_time=0.1, sr=sr, stem_type='kick', config=config)
        
        assert result is not None
        assert 'primary_energy' in result
        assert 'secondary_energy' in result
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
                'body_freq_max': 150
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
