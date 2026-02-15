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
    _cluster_values,
    _extract_feature_values,
    _map_note,
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
        assert events[2]['classification'] == 2  # high

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
        """Three clearly different centroids → crash/ride/chinese."""
        events = [
            _make_event(spectral_centroid_hz=2000.0),  # crash (lowest)
            _make_event(spectral_centroid_hz=5000.0),  # ride (mid)
            _make_event(spectral_centroid_hz=8000.0),  # chinese (highest)
        ]
        classify_cymbal_notes(events, default_config)
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
    """Tests for snare type classification."""

    def test_four_distinct_types(self, default_config):
        """Four clearly different centroids with expected_clusters=4."""
        config = {**default_config, 'snare': {'expected_clusters': 4}}
        events = [
            _make_event(spectral_centroid_hz=300.0),   # snare (lowest)
            _make_event(spectral_centroid_hz=800.0),   # rimshot
            _make_event(spectral_centroid_hz=2000.0),  # clap
            _make_event(spectral_centroid_hz=5000.0),  # clap+snare (highest)
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1
        assert events[2]['classification'] == 2
        assert events[3]['classification'] == 3

    def test_expected_clusters_1_all_snare(self, default_config):
        """expected_clusters=1 (default) → all events classification=0."""
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=800.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        classify_snare_notes(events, default_config)
        assert all(e['classification'] == 0 for e in events)

    def test_expected_clusters_2_two_groups(self, default_config):
        """expected_clusters=2 → split into snare (0) and rimshot (1)."""
        config = {**default_config, 'snare': {'expected_clusters': 2}}
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1

    def test_expected_clusters_3_three_groups(self, default_config):
        """expected_clusters=3 → snare/rimshot/clap."""
        config = {**default_config, 'snare': {'expected_clusters': 3}}
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=1500.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0
        assert events[1]['classification'] == 1
        assert events[2]['classification'] == 2

    def test_expected_clusters_clamped_high(self, default_config):
        """expected_clusters > 4 clamped to 4."""
        config = {**default_config, 'snare': {'expected_clusters': 10}}
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=800.0),
            _make_event(spectral_centroid_hz=2000.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        classify_snare_notes(events, config)
        # Should behave like expected_clusters=4
        assert events[3]['classification'] == 3

    def test_reduces_k_for_few_unique(self, default_config):
        """Two unique values with expected_clusters=4 → k reduced to 2."""
        config = {**default_config, 'snare': {'expected_clusters': 4}}
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=800.0),
        ]
        classify_snare_notes(events, config)
        # Should produce only classifications 0 and a higher value
        classes = [e['classification'] for e in events]
        assert classes[0] == classes[1]  # same centroid → same class
        assert classes[0] != classes[2]  # different centroid → different class

    def test_no_centroid_data(self, default_config):
        """No centroid with expected_clusters > 1 → default snare (0)."""
        config = {**default_config, 'snare': {'expected_clusters': 4}}
        events = [_make_event()]
        classify_snare_notes(events, config)
        assert events[0]['classification'] == 0

    def test_empty_events(self, default_config):
        result = classify_snare_notes([], default_config)
        assert result == []

    def test_default_config_no_snare_key(self):
        """Config without 'snare' key defaults to expected_clusters=1."""
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        classify_snare_notes(events, {})
        assert all(e['classification'] == 0 for e in events)


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
        assert _map_note({'classification': 3}, 'snare', drum_mapping) == 40

    def test_kick_fallback(self, drum_mapping):
        """Kick has no sub-classification, uses fallback."""
        event = {}
        assert _map_note(event, 'kick', drum_mapping) == 36


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
        """Cymbals get crash/ride/chinese based on centroid clustering."""
        events = [
            _make_event(spectral_centroid_hz=2000.0),
            _make_event(spectral_centroid_hz=5000.0),
            _make_event(spectral_centroid_hz=8000.0),
        ]
        classify_notes(events, 'cymbals', drum_mapping, default_config)
        assert events[0]['note'] == 49  # crash
        assert events[1]['note'] == 51  # ride
        assert events[2]['note'] == 52  # chinese

    def test_snare_default_all_snare(self, drum_mapping, default_config):
        """Snare with default expected_clusters=1 → all note 38."""
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=800.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        classify_notes(events, 'snare', drum_mapping, default_config)
        assert all(e['note'] == 38 for e in events)

    def test_snare_classified_with_clusters(self, drum_mapping, default_config):
        """Snare with expected_clusters=4 → varied notes by centroid."""
        config = {**default_config, 'snare': {'expected_clusters': 4}}
        events = [
            _make_event(spectral_centroid_hz=300.0),
            _make_event(spectral_centroid_hz=800.0),
            _make_event(spectral_centroid_hz=2000.0),
            _make_event(spectral_centroid_hz=5000.0),
        ]
        classify_notes(events, 'snare', drum_mapping, config)
        assert events[0]['note'] == 38  # snare
        assert events[1]['note'] == 37  # rimshot
        assert events[2]['note'] == 39  # clap
        assert events[3]['note'] == 40  # clap+snare

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
