"""
Tests for Note Classification Core — Pure Functional Tests

Tests the two-pass note classification system that assigns MIDI notes
based on stored spectral features. No audio, no I/O.
"""

import numpy as np
import pytest

from stems_to_midi.note_classification_core import (
    classify_hihat_notes,
    classify_tom_notes,
    classify_cymbal_notes,
    classify_snare_notes,
    classify_notes,
    analyze_clusters,
    _cluster_values,
    _extract_feature_values,
    _map_note,
    _resolve_cluster_feature,
)
from stems_to_midi.config import DrumMapping


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def drum_mapping():
    """Default General MIDI drum mapping."""
    return DrumMapping()


@pytest.fixture
def default_config():
    """Minimal config with hihat thresholds."""
    return {
        'hihat': {
            'open_geomean_min': 262.0,
            'open_sustain_ms': 150.0,
        },
    }


def _make_event(**kwargs):
    """Helper to create an event dict with defaults."""
    base = {'time': 0.0, 'status': 'KEPT', 'strength': 0.5}
    base.update(kwargs)
    return base


# ============================================================================
# _cluster_values tests
# ============================================================================


class TestClusterValues:
    """Tests for the k-means clustering helper."""

    def test_empty_array(self):
        result = _cluster_values(np.array([]), k=3)
        assert len(result) == 0

    def test_single_value(self):
        result = _cluster_values(np.array([100.0]), k=3)
        assert result.tolist() == [0]

    def test_two_values_k3(self):
        """Two unique values with k=3 should spread across range."""
        result = _cluster_values(np.array([100.0, 500.0, 100.0, 500.0]), k=3)
        # 100 should map to 0, 500 should map to 2
        assert result[0] == 0
        assert result[1] == 2
        assert result[2] == 0
        assert result[3] == 2

    def test_three_distinct_clusters(self):
        """Three clearly separated groups should cluster correctly."""
        values = np.array([100, 105, 110, 300, 310, 305, 600, 610, 605])
        result = _cluster_values(values, k=3)
        # Low group → 0, mid group → 1, high group → 2
        assert all(result[:3] == 0)
        assert all(result[3:6] == 1)
        assert all(result[6:] == 2)

    def test_all_identical(self):
        """All identical values should classify as 0."""
        values = np.array([200.0, 200.0, 200.0])
        result = _cluster_values(values, k=3)
        assert all(result == 0)

    def test_k4_with_four_clusters(self):
        """Four clusters for snare-type classification."""
        values = np.array([100, 200, 400, 800, 105, 205, 395, 810])
        result = _cluster_values(values, k=4)
        # Should separate into 4 ordered groups
        assert result[0] == result[4]  # both ~100
        assert result[1] == result[5]  # both ~200
        assert result[2] == result[6]  # both ~400
        assert result[3] == result[7]  # both ~800
        # And should be ordered
        assert result[0] < result[1] < result[2] < result[3]


class TestExtractFeatureValues:
    """Tests for feature extraction helper."""

    def test_extracts_valid_values(self):
        events = [
            {'spectral_centroid_hz': 500.0},
            {'spectral_centroid_hz': 0},
            {'spectral_centroid_hz': 800.0},
            {},
        ]
        values, indices = _extract_feature_values(events, 'spectral_centroid_hz')
        assert values.tolist() == [500.0, 800.0]
        assert indices == [0, 2]

    def test_no_valid_values(self):
        events = [{'spectral_centroid_hz': 0}, {}]
        values, indices = _extract_feature_values(events, 'spectral_centroid_hz')
        assert len(values) == 0
        assert indices == []

    def test_none_values_excluded(self):
        events = [{'spectral_centroid_hz': None}, {'spectral_centroid_hz': 300.0}]
        values, indices = _extract_feature_values(events, 'spectral_centroid_hz')
        assert values.tolist() == [300.0]
        assert indices == [1]

    def test_allow_zero_false_excludes_zero(self):
        events = [{'stereo_width': 0.0}, {'stereo_width': 0.5}]
        values, indices = _extract_feature_values(events, 'stereo_width')
        assert values.tolist() == [0.5]
        assert indices == [1]

    def test_allow_zero_true_includes_zero(self):
        events = [{'stereo_width': 0.0}, {'stereo_width': 0.5}]
        values, indices = _extract_feature_values(events, 'stereo_width', allow_zero=True)
        assert values.tolist() == [0.0, 0.5]
        assert indices == [0, 1]


# ============================================================================
# Hihat Classification Tests
# ============================================================================


class TestClassifyHihatNotes:
    """Tests for hihat open/closed classification."""

    def test_open_hihat(self, default_config):
        """High geomean + long sustain → open."""
        events = [_make_event(
            body_energy=300.0,
            sizzle_energy=300.0,
            sustain_ms=200.0,
        )]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'open'

    def test_closed_hihat_low_geomean(self, default_config):
        """Low geomean → closed regardless of sustain."""
        events = [_make_event(
            body_energy=10.0,
            sizzle_energy=10.0,
            sustain_ms=200.0,
        )]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'closed'

    def test_closed_hihat_short_sustain(self, default_config):
        """Short sustain → closed regardless of geomean."""
        events = [_make_event(
            body_energy=300.0,
            sizzle_energy=300.0,
            sustain_ms=50.0,
        )]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'closed'

    def test_uses_stored_geomean(self, default_config):
        """Prefers stored geomean over computing from energies."""
        events = [_make_event(
            geomean=500.0,
            sustain_ms=200.0,
            # energies would produce geomean=10 if used
            body_energy=1.0,
            sizzle_energy=100.0,
        )]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'open'

    def test_missing_energy_fields(self, default_config):
        """Missing energies → geomean=0 → closed."""
        events = [_make_event(sustain_ms=200.0)]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'closed'

    def test_mixed_events(self, default_config):
        """Mix of open and closed events."""
        events = [
            _make_event(geomean=500.0, sustain_ms=200.0),
            _make_event(geomean=100.0, sustain_ms=50.0),
            _make_event(geomean=300.0, sustain_ms=160.0),
            _make_event(geomean=260.0, sustain_ms=200.0),  # just below threshold
        ]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'open'
        assert events[1]['hihat_state'] == 'closed'
        assert events[2]['hihat_state'] == 'open'
        assert events[3]['hihat_state'] == 'closed'

    def test_empty_events(self, default_config):
        """Empty list returns empty list."""
        result = classify_hihat_notes([], default_config)
        assert result == []

    def test_boundary_geomean_exact(self, default_config):
        """Geomean exactly at threshold → open (>=)."""
        events = [_make_event(geomean=262.0, sustain_ms=150.0)]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'open'

    def test_custom_thresholds(self):
        """Custom thresholds from config override defaults."""
        config = {
            'hihat': {
                'open_geomean_min': 100.0,
                'open_sustain_ms': 50.0,
            },
        }
        events = [_make_event(geomean=150.0, sustain_ms=60.0)]
        classify_hihat_notes(events, config)
        assert events[0]['hihat_state'] == 'open'

    def test_preserves_stored_open_state(self, default_config):
        """Stored hihat_state='open' is preserved on rebuild (parity with other stems).

        Regression for bug A4: hihat classification was always overwriting
        hihat_state even when the event already had one, which meant a
        reconvert could silently flip a previously-classified event.
        """
        events = [
            _make_event(geomean=10.0, sustain_ms=10.0, hihat_state='open'),
            _make_event(geomean=10.0, sustain_ms=10.0, hihat_state='closed'),
        ]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'open'   # Preserved
        assert events[1]['hihat_state'] == 'closed'  # Preserved

    def test_force_reclassify_overrides_stored(self, default_config):
        """force_reclassify=True re-runs classification ignoring stored state."""
        events = [
            # Stored 'open' but the new thresholds would say 'closed'
            _make_event(geomean=10.0, sustain_ms=10.0, hihat_state='open'),
            # Stored 'closed' but the new thresholds would say 'open'
            _make_event(geomean=500.0, sustain_ms=200.0, hihat_state='closed'),
        ]
        classify_hihat_notes(events, default_config, force_reclassify=True)
        assert events[0]['hihat_state'] == 'closed'  # Reclassified
        assert events[1]['hihat_state'] == 'open'    # Reclassified

    def test_unset_state_always_classified(self, default_config):
        """Events without a stored hihat_state get classified fresh."""
        events = [
            _make_event(geomean=500.0, sustain_ms=200.0),
            _make_event(geomean=10.0, sustain_ms=10.0),
        ]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'open'
        assert events[1]['hihat_state'] == 'closed'

    def test_stored_state_not_treated_as_truth_when_invalid(self, default_config):
        """Stored hihat_state with unknown value is reclassified."""
        events = [
            _make_event(geomean=500.0, sustain_ms=200.0, hihat_state='handclap'),
        ]
        classify_hihat_notes(events, default_config)
        # 'handclap' is not in the ('open', 'closed') truth set → reclassify
        assert events[0]['hihat_state'] == 'open'

    def test_none_sustain_treated_as_zero(self, default_config):
        """None sustain_ms should not crash, treat as 0."""
        events = [_make_event(geomean=500.0, sustain_ms=None)]
        classify_hihat_notes(events, default_config)
        assert events[0]['hihat_state'] == 'closed'


# ============================================================================
# Tom Classification Tests
# ============================================================================


class TestClassifyTomNotes:
    """Tests for tom low/mid/high classification."""

    def test_three_distinct_toms(self, default_config):
        """Three clearly different centroids → low/mid/high."""
        events = [
            _make_event(spectral_centroid_hz=200.0),
            _make_event(spectral_centroid_hz=500.0),
            _make_event(spectral_centroid_hz=1000.0),
        ]
        classify_tom_notes(events, default_config)
        assert events[0]['classification'] == 0  # low
        assert events[1]['classification'] == 1  # mid
        assert events[2]['classification'] == 2  # high

    def test_single_tom(self, default_config):
        """Single event → defaults to mid (1) since only one unique value."""
        events = [_make_event(spectral_centroid_hz=500.0)]
        classify_tom_notes(events, default_config)
        # Single value clusters as 0 (lowest), which is the only cluster
        assert events[0]['classification'] == 0

    def test_no_centroid_data(self, default_config):
        """No spectral_centroid_hz → default to mid."""
        events = [_make_event(), _make_event()]
        classify_tom_notes(events, default_config)
        assert all(e['classification'] == 1 for e in events)

    def test_mixed_valid_invalid(self, default_config):
        """Events without centroid data get default, valid ones get classified."""
        events = [
            _make_event(spectral_centroid_hz=200.0),
            _make_event(),  # no centroid
            _make_event(spectral_centroid_hz=1000.0),
        ]
        classify_tom_notes(events, default_config)
        assert events[0]['classification'] == 0  # low
        assert events[1]['classification'] == 1  # default mid
        # Only 2 unique values → k=min(3,2)=2, so max classification is 1
        assert events[2]['classification'] == 1  # high (but only 2 clusters)

    def test_all_same_centroid(self, default_config):
        """All same centroid → all get classification 0."""
        events = [
            _make_event(spectral_centroid_hz=500.0),
            _make_event(spectral_centroid_hz=500.0),
        ]
        classify_tom_notes(events, default_config)
        assert events[0]['classification'] == events[1]['classification']

    def test_empty_events(self, default_config):
        result = classify_tom_notes([], default_config)
        assert result == []


# ============================================================================
# Cymbal Classification Tests
# ============================================================================


class TestClassifyCymbalNotes:
    """Tests for cymbal crash/ride/chinese classification."""

    def test_three_distinct_cymbals(self, default_config):
        """Three clearly different centroids → crash/ride/chinese with 3 clusters."""
        config = {**default_config, 'cymbals': {'expected_clusters': 3}}
        events = [
            _make_event(spectral_centroid_hz=2000.0),  # crash (lowest)
            _make_event(spectral_centroid_hz=5000.0),  # ride (mid)
            _make_event(spectral_centroid_hz=8000.0),  # chinese (highest)
        ]
        classify_cymbal_notes(events, config)
        assert events[0]['classification'] == 0  # crash
        assert events[1]['classification'] == 1  # ride
        assert events[2]['classification'] == 2  # chinese

    def test_no_centroid_data(self, default_config):
        """No centroid data → default to crash."""
        events = [_make_event(), _make_event()]
        classify_cymbal_notes(events, default_config)
        assert all(e['classification'] == 0 for e in events)

    def test_single_cymbal(self, default_config):
        events = [_make_event(spectral_centroid_hz=3000.0)]
        classify_cymbal_notes(events, default_config)
        assert events[0]['classification'] == 0

    def test_empty_events(self, default_config):
        result = classify_cymbal_notes([], default_config)
        assert result == []


# ============================================================================
# Snare Classification Tests
# ============================================================================


class TestClassifySnareNotes:
    """Tests for snare type classification using stereo_width."""

    def test_two_distinct_types_stereo_width(self, default_config):
        """Mono snare vs wide clap using stereo_width with default clusters=2."""
        config = {**default_config, 'snare': {'expected_clusters': 2}}
        events = [
            _make_event(stereo_width=0.03),  # mono snare (narrowest → 0)
            _make_event(stereo_width=0.35),  # wide clap (widest → 1)
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1

    def test_three_distinct_types_stereo_width(self, default_config):
        """Three stereo width clusters."""
        config = {**default_config, 'snare': {'expected_clusters': 3}}
        events = [
            _make_event(stereo_width=0.01),  # narrow
            _make_event(stereo_width=0.20),  # mid
            _make_event(stereo_width=0.60),  # wide
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1
        assert events[2]['classification'] == 2

    def test_expected_clusters_1_all_snare(self, default_config):
        """expected_clusters=1 → all events classification=0."""
        config = {**default_config, 'snare': {'expected_clusters': 1}}
        events = [
            _make_event(stereo_width=0.03),
            _make_event(stereo_width=0.40),
        ]
        classify_snare_notes(events, config)
        assert all(e['classification'] == 0 for e in events)

    def test_expected_clusters_3_three_groups(self, default_config):
        """expected_clusters=3 → three stereo width clusters."""
        config = {**default_config, 'snare': {'expected_clusters': 3}}
        events = [
            _make_event(stereo_width=0.01),
            _make_event(stereo_width=0.20),
            _make_event(stereo_width=0.50),
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1
        assert events[2]['classification'] == 2

    def test_expected_clusters_clamped_high(self, default_config):
        """expected_clusters > 3 clamped to 3."""
        config = {**default_config, 'snare': {'expected_clusters': 10}}
        events = [
            _make_event(stereo_width=0.01),
            _make_event(stereo_width=0.20),
            _make_event(stereo_width=0.60),
        ]
        classify_snare_notes(events, config)
        # Should behave like expected_clusters=3
        assert events[2]['classification'] == 2

    def test_reduces_k_for_few_unique(self, default_config):
        """Two unique values with expected_clusters=3 → k reduced to 2."""
        config = {**default_config, 'snare': {'expected_clusters': 3}}
        events = [
            _make_event(stereo_width=0.02),
            _make_event(stereo_width=0.02),
            _make_event(stereo_width=0.40),
        ]
        classify_snare_notes(events, config)
        classes = [e['classification'] for e in events]
        assert classes[0] == classes[1]  # same width → same class
        assert classes[0] != classes[2]  # different width → different class

    def test_fallback_to_spectral_centroid(self, default_config):
        """No stereo_width data → falls back to spectral_centroid_hz."""
        config = {**default_config, 'snare': {'expected_clusters': 2}}
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1

    def test_no_feature_data(self, default_config):
        """No stereo_width or centroid → all default snare (0)."""
        config = {**default_config, 'snare': {'expected_clusters': 3}}
        events = [_make_event()]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0

    def test_zero_stereo_width_is_valid(self, default_config):
        """stereo_width=0.0 (mono) is a valid value, not skipped."""
        config = {**default_config, 'snare': {'expected_clusters': 2}}
        events = [
            _make_event(stereo_width=0.0),
            _make_event(stereo_width=0.4),
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0  # narrow
        assert events[1]['classification'] == 1  # wide

    def test_empty_events(self, default_config):
        result = classify_snare_notes([], default_config)
        assert result == []

    def test_default_config_no_snare_key(self):
        """Config without 'snare' key defaults to expected_clusters=2."""
        events = [
            _make_event(stereo_width=0.02),
            _make_event(stereo_width=0.40),
        ]
        classify_snare_notes(events, {})
        # Default is 2 clusters, so should split
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1


# ============================================================================
# _resolve_cluster_feature Tests
# ============================================================================


class TestResolveClusterFeature:
    """Tests for the shared cluster feature resolution helper."""

    def test_auto_snare_prefers_stereo_width(self):
        """Auto mode for snare picks stereo_width when available."""
        events = [
            _make_event(stereo_width=0.1, spectral_centroid_hz=300.0),
            _make_event(stereo_width=0.4, spectral_centroid_hz=5000.0),
        ]
        values, indices = _resolve_cluster_feature(events, 'snare', {})
        assert len(values) == 2
        # Should be stereo_width values
        assert np.isclose(values[0], 0.1)
        assert np.isclose(values[1], 0.4)

    def test_auto_toms_prefers_spectral_centroid(self):
        """Auto mode for toms picks spectral_centroid_hz first."""
        events = [
            _make_event(spectral_centroid_hz=200.0, stereo_width=0.1),
            _make_event(spectral_centroid_hz=800.0, stereo_width=0.3),
        ]
        values, indices = _resolve_cluster_feature(events, 'toms', {})
        assert len(values) == 2
        assert np.isclose(values[0], 200.0)
        assert np.isclose(values[1], 800.0)

    def test_explicit_feature_override(self):
        """Explicit cluster_feature overrides auto priority."""
        events = [
            _make_event(stereo_width=0.1, spectral_centroid_hz=300.0),
            _make_event(stereo_width=0.4, spectral_centroid_hz=5000.0),
        ]
        config = {'toms': {'cluster_feature': 'stereo_width'}}
        values, indices = _resolve_cluster_feature(events, 'toms', config)
        # Should use stereo_width despite toms default being centroid
        assert np.isclose(values[0], 0.1)
        assert np.isclose(values[1], 0.4)

    def test_explicit_feature_falls_back_when_missing(self):
        """Explicit feature with no data falls back to next priority."""
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        # Request stereo_width but events don't have it → falls back to centroid
        config = {'snare': {'cluster_feature': 'stereo_width'}}
        values, indices = _resolve_cluster_feature(events, 'snare', config)
        assert len(values) == 2
        assert np.isclose(values[0], 300.0)

    def test_auto_with_no_data_returns_empty(self):
        """No feature data available → empty arrays."""
        events = [_make_event(), _make_event()]
        values, indices = _resolve_cluster_feature(events, 'snare', {})
        assert len(values) == 0
        assert len(indices) == 0

    def test_zero_stereo_width_is_valid(self):
        """stereo_width=0.0 is valid (mono signal), not treated as missing."""
        events = [
            _make_event(stereo_width=0.0),
            _make_event(stereo_width=0.3),
        ]
        values, indices = _resolve_cluster_feature(events, 'snare', {})
        assert len(values) == 2
        assert np.isclose(values[0], 0.0)

    def test_auto_selects_first_feature_with_data(self):
        """Auto mode for cymbals: centroid first, then stereo_width."""
        # Only stereo_width available, not centroid
        events = [
            _make_event(stereo_width=0.1),
            _make_event(stereo_width=0.5),
        ]
        values, indices = _resolve_cluster_feature(events, 'cymbals', {})
        assert len(values) == 2
        # Fell back to stereo_width since centroid missing
        assert np.isclose(values[0], 0.1)


# ============================================================================
# cluster_feature Config Override Tests
# ============================================================================


class TestClusterFeatureOverride:
    """Tests for cluster_feature config override on classify functions."""

    def test_tom_explicit_stereo_width(self, default_config):
        """Toms with cluster_feature='stereo_width' clusters by width."""
        config = {
            **default_config,
            'toms': {'expected_clusters': 2, 'cluster_feature': 'stereo_width'},
        }
        events = [
            _make_event(stereo_width=0.05, spectral_centroid_hz=500.0),
            _make_event(stereo_width=0.40, spectral_centroid_hz=500.0),
        ]
        classify_tom_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1

    def test_cymbal_explicit_stereo_width(self, default_config):
        """Cymbals with cluster_feature='stereo_width' clusters by width."""
        config = {
            **default_config,
            'cymbals': {'expected_clusters': 2, 'cluster_feature': 'stereo_width'},
        }
        events = [
            _make_event(stereo_width=0.05, spectral_centroid_hz=3000.0),
            _make_event(stereo_width=0.40, spectral_centroid_hz=3000.0),
        ]
        classify_cymbal_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1

    def test_snare_explicit_spectral_centroid(self, default_config):
        """Snare with cluster_feature='spectral_centroid_hz' uses centroid."""
        config = {
            **default_config,
            'snare': {
                'expected_clusters': 2,
                'cluster_feature': 'spectral_centroid_hz',
            },
        }
        events = [
            _make_event(spectral_centroid_hz=300.0, stereo_width=0.3),
            _make_event(spectral_centroid_hz=5000.0, stereo_width=0.3),
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1


# ============================================================================
# expected_clusters=None (YAML null) Tests
# ============================================================================


class TestExpectedClustersNull:
    """Tests that expected_clusters=None (YAML null) doesn't crash."""

    def test_tom_null_clusters_uses_default(self, default_config):
        """Toms with expected_clusters=None uses default of 3."""
        config = {**default_config, 'toms': {'expected_clusters': None}}
        events = [
            _make_event(spectral_centroid_hz=200.0),
            _make_event(spectral_centroid_hz=500.0),
            _make_event(spectral_centroid_hz=1000.0),
        ]
        classify_tom_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1
        assert events[2]['classification'] == 2

    def test_cymbal_null_clusters_uses_default(self, default_config):
        """Cymbals with expected_clusters=None uses default of 2."""
        config = {**default_config, 'cymbals': {'expected_clusters': None}}
        events = [
            _make_event(spectral_centroid_hz=2000.0),
            _make_event(spectral_centroid_hz=8000.0),
        ]
        classify_cymbal_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1

    def test_snare_null_clusters_uses_default(self, default_config):
        """Snare with expected_clusters=None uses default of 2."""
        config = {**default_config, 'snare': {'expected_clusters': None}}
        events = [
            _make_event(stereo_width=0.02),
            _make_event(stereo_width=0.40),
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1


# ============================================================================
# expected_clusters for toms/cymbals Tests
# ============================================================================


class TestExpectedClustersTomsAndCymbals:
    """Tests for configurable expected_clusters on toms and cymbals."""

    def test_toms_expected_clusters_1(self, default_config):
        """expected_clusters=1 → all toms get classification 0."""
        config = {**default_config, 'toms': {'expected_clusters': 1}}
        events = [
            _make_event(spectral_centroid_hz=200.0),
            _make_event(spectral_centroid_hz=1000.0),
        ]
        classify_tom_notes(events, config)
        assert all(e['classification'] == 0 for e in events)

    def test_toms_expected_clusters_4(self, default_config):
        """expected_clusters=4 → four tom groups."""
        config = {**default_config, 'toms': {'expected_clusters': 4}}
        events = [
            _make_event(spectral_centroid_hz=150.0),
            _make_event(spectral_centroid_hz=400.0),
            _make_event(spectral_centroid_hz=700.0),
            _make_event(spectral_centroid_hz=1200.0),
        ]
        classify_tom_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[3]['classification'] == 3

    def test_cymbals_expected_clusters_1(self, default_config):
        """expected_clusters=1 → all cymbals get classification 0."""
        config = {**default_config, 'cymbals': {'expected_clusters': 1}}
        events = [
            _make_event(spectral_centroid_hz=2000.0),
            _make_event(spectral_centroid_hz=8000.0),
        ]
        classify_cymbal_notes(events, config)
        assert all(e['classification'] == 0 for e in events)

    def test_cymbals_expected_clusters_3(self, default_config):
        """expected_clusters=3 → three cymbal groups."""
        config = {**default_config, 'cymbals': {'expected_clusters': 3}}
        events = [
            _make_event(spectral_centroid_hz=2000.0),
            _make_event(spectral_centroid_hz=5000.0),
            _make_event(spectral_centroid_hz=8000.0),
        ]
        classify_cymbal_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1
        assert events[2]['classification'] == 2

    def test_toms_clusters_clamped_to_max_4(self, default_config):
        """expected_clusters > 4 clamped to 4."""
        config = {**default_config, 'toms': {'expected_clusters': 10}}
        events = [
            _make_event(spectral_centroid_hz=150.0),
            _make_event(spectral_centroid_hz=400.0),
            _make_event(spectral_centroid_hz=700.0),
            _make_event(spectral_centroid_hz=1200.0),
        ]
        classify_tom_notes(events, config)
        assert events[3]['classification'] == 3  # max classification = k-1 = 3


# ============================================================================
# Cluster Analysis Tests
# ============================================================================


class TestAnalyzeClusters:
    """Tests for analyze_clusters metadata extraction."""

    def test_returns_cluster_info_per_classification(self, drum_mapping):
        """Each unique classification gets a cluster info entry."""
        events = [
            _make_event(classification=0, stereo_width=0.03, note=38),
            _make_event(classification=0, stereo_width=0.04, note=38),
            _make_event(classification=1, stereo_width=0.35, note=37),
        ]
        result = analyze_clusters(events, 'snare', drum_mapping)
        assert len(result) == 2
        # Sorted by count descending
        assert result[0]['classification'] == 0
        assert result[0]['count'] == 2
        assert result[1]['classification'] == 1
        assert result[1]['count'] == 1

    def test_cluster_info_has_required_fields(self, drum_mapping):
        """Each cluster info dict has the expected keys."""
        events = [
            _make_event(classification=0, stereo_width=0.03, note=38),
            _make_event(classification=1, stereo_width=0.35, note=37),
        ]
        result = analyze_clusters(events, 'snare', drum_mapping)
        required_keys = {
            'classification', 'note', 'note_label', 'count',
            'features', 'distinguishing_feature', 'distinguishing_label',
            'description',
        }
        for cluster in result:
            assert required_keys.issubset(cluster.keys())

    def test_feature_stats_computed(self, drum_mapping):
        """Feature stats include mean, min, max."""
        events = [
            _make_event(classification=0, stereo_width=0.02, note=38),
            _make_event(classification=0, stereo_width=0.06, note=38),
        ]
        result = analyze_clusters(events, 'snare', drum_mapping)
        features = result[0]['features']
        assert 'stereo_width' in features
        sw = features['stereo_width']
        assert sw['mean'] == pytest.approx(0.04, abs=0.001)
        assert sw['min'] == pytest.approx(0.02, abs=0.001)
        assert sw['max'] == pytest.approx(0.06, abs=0.001)

    def test_single_cluster_has_description(self, drum_mapping):
        """Single cluster still returns valid description."""
        events = [
            _make_event(classification=0, stereo_width=0.03, note=38),
        ]
        result = analyze_clusters(events, 'snare', drum_mapping)
        assert len(result) == 1
        assert result[0]['description']  # non-empty string

    def test_empty_events(self, drum_mapping):
        """Empty events list returns empty result."""
        result = analyze_clusters([], 'snare', drum_mapping)
        assert result == []


# ============================================================================
# Note Mapping Tests
# ============================================================================


class TestMapNote:
    """Tests for mapping classification to MIDI note numbers."""

    def test_hihat_open(self, drum_mapping):
        event = {'hihat_state': 'open'}
        assert _map_note(event, 'hihat', drum_mapping) == 46

    def test_hihat_closed(self, drum_mapping):
        event = {'hihat_state': 'closed'}
        assert _map_note(event, 'hihat', drum_mapping) == 42

    def test_hihat_default(self, drum_mapping):
        """Missing hihat_state → closed."""
        event = {}
        assert _map_note(event, 'hihat', drum_mapping) == 42

    def test_tom_low(self, drum_mapping):
        event = {'classification': 0}
        assert _map_note(event, 'toms', drum_mapping) == 45

    def test_tom_mid(self, drum_mapping):
        event = {'classification': 1}
        assert _map_note(event, 'toms', drum_mapping) == 47

    def test_tom_high(self, drum_mapping):
        event = {'classification': 2}
        assert _map_note(event, 'toms', drum_mapping) == 50

    def test_cymbal_crash(self, drum_mapping):
        event = {'classification': 0}
        assert _map_note(event, 'cymbals', drum_mapping) == 49

    def test_cymbal_ride(self, drum_mapping):
        event = {'classification': 1}
        assert _map_note(event, 'cymbals', drum_mapping) == 51

    def test_cymbal_chinese(self, drum_mapping):
        event = {'classification': 2}
        assert _map_note(event, 'cymbals', drum_mapping) == 52

    def test_snare_types(self, drum_mapping):
        assert _map_note({'classification': 0}, 'snare', drum_mapping) == 38
        assert _map_note({'classification': 1}, 'snare', drum_mapping) == 37
        assert _map_note({'classification': 2}, 'snare', drum_mapping) == 39

    def test_kick_fallback(self, drum_mapping):
        """Kick has no sub-classification, uses fallback."""
        event = {}
        assert _map_note(event, 'kick', drum_mapping) == 36

    def test_cluster_note_map_overrides_default(self, drum_mapping):
        """Config with cluster_note_map overrides default note mapping."""
        config = {'snare': {'cluster_note_map': {0: 39, 1: 38}}}
        event0 = {'classification': 0}
        event1 = {'classification': 1}
        assert _map_note(event0, 'snare', drum_mapping, config) == 39  # clap instead of snare
        assert _map_note(event1, 'snare', drum_mapping, config) == 38  # snare instead of rimshot

    def test_cluster_note_map_string_keys(self, drum_mapping):
        """cluster_note_map works with string keys (from JSON deserialization)."""
        config = {'snare': {'cluster_note_map': {'0': 40, '1': 37}}}
        event = {'classification': 0}
        assert _map_note(event, 'snare', drum_mapping, config) == 40

    def test_cluster_note_map_missing_key_falls_through(self, drum_mapping):
        """Classification not in cluster_note_map falls back to default map."""
        config = {'snare': {'cluster_note_map': {0: 39}}}
        event = {'classification': 1}
        # classification 1 not in map, so falls back to default: rimshot=37
        assert _map_note(event, 'snare', drum_mapping, config) == 37

    def test_no_config_uses_default(self, drum_mapping):
        """No config parameter uses default note mapping."""
        event = {'classification': 0}
        assert _map_note(event, 'snare', drum_mapping) == 38


# ============================================================================
# classify_notes Integration Tests
# ============================================================================


class TestClassifyNotes:
    """Tests for the main classify_notes dispatcher."""

    def test_kick_all_same_note(self, drum_mapping, default_config):
        """Kick events should all get note 36."""
        events = [_make_event(), _make_event(), _make_event()]
        classify_notes(events, 'kick', drum_mapping, default_config)
        assert all(e['note'] == 36 for e in events)

    def test_hihat_mixed(self, drum_mapping, default_config):
        """Hihat open/closed based on geomean + sustain."""
        events = [
            _make_event(geomean=500.0, sustain_ms=200.0),
            _make_event(geomean=100.0, sustain_ms=50.0),
        ]
        classify_notes(events, 'hihat', drum_mapping, default_config)
        assert events[0]['note'] == 46  # open
        assert events[1]['note'] == 42  # closed

    def test_toms_classified(self, drum_mapping, default_config):
        """Toms get low/mid/high notes based on centroid clustering."""
        events = [
            _make_event(spectral_centroid_hz=200.0),
            _make_event(spectral_centroid_hz=500.0),
            _make_event(spectral_centroid_hz=1000.0),
        ]
        classify_notes(events, 'toms', drum_mapping, default_config)
        assert events[0]['note'] == 45  # low tom
        assert events[1]['note'] == 47  # mid tom
        assert events[2]['note'] == 50  # high tom

    def test_cymbals_classified(self, drum_mapping, default_config):
        """Cymbals get crash/ride/chinese based on centroid clustering (3 clusters)."""
        config = {**default_config, 'cymbals': {'expected_clusters': 3}}
        events = [
            _make_event(spectral_centroid_hz=2000.0),
            _make_event(spectral_centroid_hz=5000.0),
            _make_event(spectral_centroid_hz=8000.0),
        ]
        classify_notes(events, 'cymbals', drum_mapping, config)
        assert events[0]['note'] == 49  # crash
        assert events[1]['note'] == 51  # ride
        assert events[2]['note'] == 52  # chinese

    def test_snare_default_two_clusters(self, drum_mapping, default_config):
        """Snare with default expected_clusters=2 → snare + rimshot by stereo_width."""
        events = [
            _make_event(stereo_width=0.03),
            _make_event(stereo_width=0.35),
        ]
        classify_notes(events, 'snare', drum_mapping, default_config)
        assert events[0]['note'] == 38  # snare (narrow)
        assert events[1]['note'] == 37  # rimshot (wide)

    def test_snare_single_cluster_all_snare(self, drum_mapping, default_config):
        """Snare with expected_clusters=1 → all note 38."""
        config = {**default_config, 'snare': {'expected_clusters': 1}}
        events = [
            _make_event(stereo_width=0.03),
            _make_event(stereo_width=0.35),
        ]
        classify_notes(events, 'snare', drum_mapping, config)
        assert all(e['note'] == 38 for e in events)

    def test_snare_classified_with_three_clusters(self, drum_mapping, default_config):
        """Snare with expected_clusters=3 → varied notes by stereo_width."""
        config = {**default_config, 'snare': {'expected_clusters': 3}}
        events = [
            _make_event(stereo_width=0.01),
            _make_event(stereo_width=0.20),
            _make_event(stereo_width=0.60),
        ]
        classify_notes(events, 'snare', drum_mapping, config)
        assert events[0]['note'] == 38  # snare
        assert events[1]['note'] == 37  # rimshot
        assert events[2]['note'] == 39  # clap

    def test_snare_with_cluster_note_map(self, drum_mapping, default_config):
        """Snare with cluster_note_map overrides default note assignment."""
        config = {**default_config, 'snare': {
            'expected_clusters': 2,
            'cluster_note_map': {0: 39, 1: 38},  # swap: narrow=clap, wide=snare
        }}
        events = [
            _make_event(stereo_width=0.03),
            _make_event(stereo_width=0.40),
        ]
        classify_notes(events, 'snare', drum_mapping, config)
        assert events[0]['note'] == 39  # narrow → clap (overridden)
        assert events[1]['note'] == 38  # wide → snare (overridden)

    def test_empty_events(self, drum_mapping, default_config):
        """Empty list returns empty list without error."""
        result = classify_notes([], 'hihat', drum_mapping, default_config)
        assert result == []

    def test_custom_drum_mapping(self, default_config):
        """Custom drum mapping overrides default MIDI notes."""
        custom_mapping = DrumMapping(tom_low=41, tom_mid=43, tom_high=48)
        events = [
            _make_event(spectral_centroid_hz=200.0),
            _make_event(spectral_centroid_hz=500.0),
            _make_event(spectral_centroid_hz=1000.0),
        ]
        classify_notes(events, 'toms', custom_mapping, default_config)
        assert events[0]['note'] == 41
        assert events[1]['note'] == 43
        assert events[2]['note'] == 48

    def test_realistic_hihat_population(self, drum_mapping, default_config):
        """Realistic scenario: mostly closed with some opens."""
        events = []
        # 20 closed hits (low geomean, short sustain)
        for i in range(20):
            events.append(_make_event(
                time=i * 0.5,
                geomean=100.0 + i * 2,
                sustain_ms=40.0 + i,
            ))
        # 3 open hits (high geomean, long sustain)
        for i in range(3):
            events.append(_make_event(
                time=10.0 + i * 2,
                geomean=400.0 + i * 50,
                sustain_ms=200.0 + i * 20,
            ))
        classify_notes(events, 'hihat', drum_mapping, default_config)
        closed_count = sum(1 for e in events if e['note'] == 42)
        open_count = sum(1 for e in events if e['note'] == 46)
        assert closed_count == 20
        assert open_count == 3
