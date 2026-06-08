"""
Tests for the cluster-feature resolver in note_classification_core.

Locks down the contract that `_resolve_cluster_feature` returns a
3-tuple (values, valid_indices, actual_feature) where `actual_feature`
is the feature that was actually used (possibly different from the
user's explicit choice if their choice has no data). This makes the
silent fallback observable: the caller can log a warning when
`actual_feature != chosen`.

User report (2026-06-08): picking "Pitch" in the snare Cluster By
dropdown did nothing visible. Root cause was that
`enable_pitch_detection: false` meant no `pitch_hz` data was
computed; `_resolve_cluster_feature` then walked the priority chain
and silently used `stereo_width` instead. Same result as the
default — looked like "doesn't work."

These tests catch any future regression that swallows the fallback
silently again.
"""

import pytest

from stems_to_midi.note_classification_core import _resolve_cluster_feature


# ─── _resolve_cluster_feature contract ─────────────────────────────────


class TestResolveClusterFeatureContract:
    """`_resolve_cluster_feature` must return a 3-tuple and surface
    the actual feature used so callers can detect silent fallback.
    """

    def test_returns_three_tuple(self):
        """The function must return (values, valid_indices, actual_feature).
        Pinning the shape here — call sites in classify_tom_notes,
        classify_cymbal_notes, classify_snare_notes all unpack the
        return value, so adding a third element is an API change
        that must be loud (this test fails on the old 2-tuple)."""
        events = [{'stereo_width': 0.1}, {'stereo_width': 0.5}]
        result = _resolve_cluster_feature(events, 'snare', {})
        assert isinstance(result, tuple), (
            f"_resolve_cluster_feature must return a tuple, got {type(result).__name__}"
        )
        assert len(result) == 3, (
            f"_resolve_cluster_feature must return a 3-tuple "
            f"(values, valid_indices, actual_feature), got len={len(result)}: {result!r}"
        )
        values, valid_indices, actual_feature = result
        assert actual_feature == 'stereo_width', (
            f"With no cluster_feature config, 'snare' defaults to "
            f"stereo_width per the priority chain. Got: {actual_feature!r}"
        )

    def test_actual_feature_matches_chosen_when_data_present(self):
        """When the user picks 'pitch_hz' and pitch_hz data is present,
        the resolver must use pitch_hz — not fall back to anything else."""
        events = [
            {'pitch_hz': 220.0, 'stereo_width': 0.1},
            {'pitch_hz': 440.0, 'stereo_width': 0.5},
        ]
        config = {'snare': {'cluster_feature': 'pitch_hz'}}
        values, valid_indices, actual_feature = _resolve_cluster_feature(
            events, 'snare', config,
        )
        assert actual_feature == 'pitch_hz', (
            f"User chose 'pitch_hz' and data is present — must use "
            f"pitch_hz, not fall back. Got: {actual_feature!r}"
        )
        assert list(values) == [220.0, 440.0]

    def test_actual_feature_reveals_fallback_when_chosen_has_no_data(self):
        """The bug: when the user picks 'pitch_hz' but no events have
        pitch_hz, the resolver falls back to stereo_width. The new
        contract surfaces the actual feature so callers can warn."""
        # Events have stereo_width but NOT pitch_hz
        events = [
            {'stereo_width': 0.1, 'spectral_centroid_hz': 1500.0},
            {'stereo_width': 0.5, 'spectral_centroid_hz': 3000.0},
        ]
        config = {'snare': {'cluster_feature': 'pitch_hz'}}
        values, valid_indices, actual_feature = _resolve_cluster_feature(
            events, 'snare', config,
        )
        # User chose pitch_hz; resolver must walk the chain and use
        # stereo_width (next in the priority list for snare).
        assert actual_feature == 'stereo_width', (
            f"Fallback must surface as actual_feature='stereo_width' "
            f"so the caller can log a warning. Got: {actual_feature!r}"
        )
        assert list(values) == [0.1, 0.5]

    def test_actual_feature_is_stereo_width_for_snare_auto(self):
        """Snare's default ('auto') priority chain starts with
        stereo_width, then spectral_centroid_hz."""
        events = [
            {'stereo_width': 0.1, 'spectral_centroid_hz': 1500.0},
        ]
        config = {'snare': {}}  # no cluster_feature → 'auto'
        values, valid_indices, actual_feature = _resolve_cluster_feature(
            events, 'snare', config,
        )
        assert actual_feature == 'stereo_width'

    def test_actual_feature_is_pitch_hz_for_toms_auto(self):
        """Toms' default priority chain starts with pitch_hz."""
        events = [
            {'pitch_hz': 110.0, 'spectral_centroid_hz': 500.0},
        ]
        config = {'toms': {}}
        _, _, actual_feature = _resolve_cluster_feature(
            events, 'toms', config,
        )
        assert actual_feature == 'pitch_hz'

    def test_actual_feature_is_none_when_no_data_at_all(self):
        """If no events have any feature data, the resolver returns
        empty arrays. The actual_feature should be None (sentinel)
        so callers know there's no fallback to warn about."""
        events = [{'time': 1.0}, {'time': 2.0}]  # no feature data
        config = {'snare': {'cluster_feature': 'pitch_hz'}}
        values, valid_indices, actual_feature = _resolve_cluster_feature(
            events, 'snare', config,
        )
        assert len(values) == 0
        assert actual_feature is None, (
            f"When no features are available, actual_feature must be None, "
            f"got: {actual_feature!r}"
        )
