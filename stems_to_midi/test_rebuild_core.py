"""
Tests for rebuild_core.py — Rebuild MIDI from Analysis.

Tests the pure functional core that re-filters cached detection results
and produces MIDI-ready events without audio I/O.
"""

import copy
import pytest
from unittest.mock import patch

from .rebuild_core import (
    _merge_event_pools,
    _apply_overrides,
    _refilter_events,
    _events_to_midi,
    rebuild_events_from_analysis,
)
from .config import DrumMapping


# ============================================================================
# Fixtures
# ============================================================================


def _make_event(time, geomean=100.0, strength=0.5, amplitude=0.3,
                status='KEPT', sustain_ms=None, **kwargs):
    """Helper to build a minimal analysis event dict."""
    event = {
        'time': time,
        'geomean': geomean,
        'strength': strength,
        'amplitude': amplitude,
        'status': status,
    }
    if sustain_ms is not None:
        event['sustain_ms'] = sustain_ms
    event.update(kwargs)
    return event


def _make_analysis_data(stems_dict):
    """Build a minimal v3 analysis.json structure."""
    return {
        'version': '3.0',
        'tempo_bpm': 120.0,
        'stems': stems_dict,
    }


def _make_config(stem_type='kick', geomean_threshold=50.0, **overrides):
    """Build a minimal config dict with required sections."""
    base = {
        'midi': {'default_tempo': 120.0, 'max_note_duration': 0.5, 'min_velocity': 80, 'max_velocity': 110},
        'audio': {'default_note_duration': 0.1},
        'onset_detection': {'hop_length': 512, 'threshold': 0.3, 'delta': 0.01, 'wait': 3},
        'drum_mapping': {},
    }
    # Add stem section with geomean_threshold + required freq range fields
    stem_config = {'geomean_threshold': geomean_threshold}
    if stem_type == 'kick':
        stem_config.update({
            'fundamental_freq_min': 30, 'fundamental_freq_max': 80,
            'body_freq_min': 100, 'body_freq_max': 300,
            'attack_freq_min': 2000, 'attack_freq_max': 5000,
        })
    elif stem_type == 'snare':
        stem_config.update({
            'low_freq_min': 100, 'low_freq_max': 300,
            'body_freq_min': 200, 'body_freq_max': 800,
            'wire_freq_min': 4000, 'wire_freq_max': 8000,
        })
    elif stem_type == 'hihat':
        stem_config.update({
            'body_freq_min': 500, 'body_freq_max': 4000,
            'sizzle_freq_min': 8000, 'sizzle_freq_max': 16000,
            'min_sustain_ms': 25,
            'open_sustain_ms': 150,
        })
    elif stem_type == 'cymbals':
        stem_config.update({
            'body_freq_min': 500, 'body_freq_max': 4000,
            'brilliance_freq_min': 8000, 'brilliance_freq_max': 16000,
            'min_sustain_ms': 150,
        })
    elif stem_type == 'toms':
        stem_config.update({
            'fundamental_freq_min': 80, 'fundamental_freq_max': 300,
            'body_freq_min': 2000, 'body_freq_max': 6000,
        })
    stem_config.update(overrides)
    base[stem_type] = stem_config
    return base


# ============================================================================
# _merge_event_pools tests
# ============================================================================


class TestMergeEventPools:
    def test_configured_only(self):
        configured = [_make_event(1.0), _make_event(2.0)]
        result = _merge_event_pools(configured, [])
        assert len(result) == 2
        assert all(e['_source'] == 'configured' for e in result)

    def test_sensitive_only(self):
        sensitive = [_make_event(1.0), _make_event(2.0)]
        result = _merge_event_pools([], sensitive)
        assert len(result) == 2
        assert all(e['_source'] == 'sensitive' for e in result)

    def test_deduplicates_by_time(self):
        configured = [_make_event(1.0), _make_event(2.0)]
        sensitive = [_make_event(1.005), _make_event(3.0)]  # 1.005 overlaps with 1.0
        result = _merge_event_pools(configured, sensitive, merge_window_sec=0.015)
        assert len(result) == 3  # 1.0 (configured), 2.0 (configured), 3.0 (sensitive)
        times = [e['time'] for e in result]
        assert 1.005 not in times  # Deduplicated

    def test_non_overlapping_sensitive_included(self):
        configured = [_make_event(1.0)]
        sensitive = [_make_event(5.0)]
        result = _merge_event_pools(configured, sensitive)
        assert len(result) == 2
        assert result[1]['_source'] == 'sensitive'

    def test_empty_both(self):
        result = _merge_event_pools([], [])
        assert result == []

    def test_sorted_by_time(self):
        configured = [_make_event(3.0)]
        sensitive = [_make_event(1.0), _make_event(2.0)]
        result = _merge_event_pools(configured, sensitive)
        times = [e['time'] for e in result]
        assert times == sorted(times)


# ============================================================================
# _apply_overrides tests
# ============================================================================


class TestApplyOverrides:
    def test_override_kept(self):
        events = [_make_event(1.2345, status='FILTERED')]
        _apply_overrides(events, {'1.2345': 'KEPT'})
        assert events[0]['status'] == 'KEPT'
        assert events[0]['override'] is True

    def test_override_filtered(self):
        events = [_make_event(1.2345, status='KEPT')]
        _apply_overrides(events, {'1.2345': 'FILTERED'})
        assert events[0]['status'] == 'FILTERED'
        assert events[0]['override'] is True

    def test_no_matching_override(self):
        events = [_make_event(1.2345, status='KEPT')]
        _apply_overrides(events, {'9.9999': 'FILTERED'})
        assert events[0]['status'] == 'KEPT'
        assert 'override' not in events[0]

    def test_empty_overrides(self):
        events = [_make_event(1.0, status='KEPT')]
        _apply_overrides(events, {})
        assert events[0]['status'] == 'KEPT'
        assert 'override' not in events[0]


# ============================================================================
# _refilter_events tests
# ============================================================================


class TestRefilterEvents:
    def test_filters_below_threshold(self):
        events = [
            _make_event(1.0, geomean=100.0, status='KEPT'),
            _make_event(2.0, geomean=10.0, status='KEPT'),
        ]
        spectral_config = {'geomean_threshold': 50.0, 'filter_mode': 'geomean_only'}
        _refilter_events(events, spectral_config)
        assert events[0]['status'] == 'KEPT'
        assert events[1]['status'] == 'FILTERED'

    def test_promotes_above_threshold(self):
        events = [_make_event(1.0, geomean=100.0, status='FILTERED')]
        spectral_config = {'geomean_threshold': 50.0, 'filter_mode': 'geomean_only'}
        _refilter_events(events, spectral_config)
        assert events[0]['status'] == 'KEPT'

    def test_override_survives_strict_threshold(self):
        """Override KEPT event stays KEPT even when geomean is below threshold."""
        events = [_make_event(1.0, geomean=10.0, status='KEPT')]
        events[0]['override'] = True
        spectral_config = {'geomean_threshold': 500.0, 'filter_mode': 'geomean_only'}
        _refilter_events(events, spectral_config)
        assert events[0]['status'] == 'KEPT'

    def test_override_filtered_survives_permissive_threshold(self):
        """Override FILTERED event stays FILTERED even when geomean is above threshold."""
        events = [_make_event(1.0, geomean=1000.0, status='FILTERED')]
        events[0]['override'] = True
        spectral_config = {'geomean_threshold': 50.0, 'filter_mode': 'geomean_only'}
        _refilter_events(events, spectral_config)
        assert events[0]['status'] == 'FILTERED'

    def test_no_threshold_keeps_all(self):
        events = [_make_event(1.0, geomean=1.0, status='FILTERED')]
        spectral_config = {'geomean_threshold': None, 'filter_mode': 'geomean_only'}
        _refilter_events(events, spectral_config)
        assert events[0]['status'] == 'KEPT'

    def test_require_both_mode(self):
        events = [
            _make_event(1.0, geomean=100.0, sustain_ms=200.0, status='FILTERED'),
            _make_event(2.0, geomean=100.0, sustain_ms=10.0, status='KEPT'),
        ]
        spectral_config = {
            'geomean_threshold': 50.0,
            'min_sustain_ms': 100.0,
            'filter_mode': 'require_both',
        }
        _refilter_events(events, spectral_config)
        assert events[0]['status'] == 'KEPT'   # Both pass
        assert events[1]['status'] == 'FILTERED'  # Sustain too short


# ============================================================================
# rebuild_events_from_analysis tests
# ============================================================================


class TestRebuildEventsFromAnalysis:
    def test_basic_rebuild(self):
        """Rebuild with same thresholds produces same KEPT/FILTERED statuses."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [
                    _make_event(1.0, geomean=100.0, status='KEPT'),
                    _make_event(2.0, geomean=10.0, status='FILTERED'),
                ],
                'events_sensitive': [],
            }
        })
        config = _make_config('kick', geomean_threshold=50.0)

        updated, midi_events = rebuild_events_from_analysis(analysis, {}, config)

        # KEPT event should produce MIDI, FILTERED should not
        assert len(midi_events['kick']) == 1
        assert midi_events['kick'][0]['time'] == 1.0

    def test_lowered_threshold_promotes_events(self):
        """Lowering threshold promotes previously-FILTERED events."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [
                    _make_event(1.0, geomean=100.0, status='KEPT'),
                    _make_event(2.0, geomean=30.0, status='FILTERED'),
                ],
                'events_sensitive': [],
            }
        })
        config = _make_config('kick', geomean_threshold=20.0)

        updated, midi_events = rebuild_events_from_analysis(analysis, {}, config)

        assert len(midi_events['kick']) == 2  # Both now pass

    def test_raised_threshold_demotes_events(self):
        """Raising threshold demotes previously-KEPT events."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [
                    _make_event(1.0, geomean=100.0, status='KEPT'),
                    _make_event(2.0, geomean=60.0, status='KEPT'),
                ],
                'events_sensitive': [],
            }
        })
        config = _make_config('kick', geomean_threshold=80.0)

        updated, midi_events = rebuild_events_from_analysis(analysis, {}, config)

        assert len(midi_events['kick']) == 1  # Only geomean=100 passes

    def test_override_kept_survives_strict_threshold(self):
        """Manual override KEPT survives even when threshold would filter."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [
                    _make_event(1.0, geomean=10.0, status='FILTERED'),
                ],
                'events_sensitive': [],
            }
        })
        overrides = {'kick': {'1.0000': 'KEPT'}}
        config = _make_config('kick', geomean_threshold=500.0)

        updated, midi_events = rebuild_events_from_analysis(analysis, overrides, config)

        assert len(midi_events['kick']) == 1

    def test_override_filtered_survives_permissive_threshold(self):
        """Manual override FILTERED keeps event out even when threshold would keep."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [
                    _make_event(1.0, geomean=1000.0, status='KEPT'),
                ],
                'events_sensitive': [],
            }
        })
        overrides = {'kick': {'1.0000': 'FILTERED'}}
        config = _make_config('kick', geomean_threshold=1.0)

        updated, midi_events = rebuild_events_from_analysis(analysis, overrides, config)

        assert len(midi_events['kick']) == 0

    def test_per_stem_rebuild(self):
        """Rebuilding a single stem leaves other stems unchanged."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [_make_event(1.0, geomean=100.0)],
                'events_sensitive': [],
            },
            'snare': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [_make_event(2.0, geomean=100.0)],
                'events_sensitive': [],
            },
        })
        config = _make_config('kick', geomean_threshold=50.0)
        config['snare'] = {
            'geomean_threshold': 50.0,
            'low_freq_min': 100, 'low_freq_max': 300,
            'body_freq_min': 200, 'body_freq_max': 800,
            'wire_freq_min': 4000, 'wire_freq_max': 8000,
        }

        updated, midi_events = rebuild_events_from_analysis(
            analysis, {}, config, stem_types=['kick']
        )

        assert 'kick' in midi_events
        assert 'snare' not in midi_events  # Not rebuilt

    def test_sensitive_events_promoted(self):
        """Sensitive-only events can be promoted when thresholds lowered."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [_make_event(1.0, geomean=100.0, status='KEPT')],
                'events_sensitive': [_make_event(3.0, geomean=30.0, status='KEPT')],
            }
        })
        config = _make_config('kick', geomean_threshold=20.0)

        updated, midi_events = rebuild_events_from_analysis(analysis, {}, config)

        assert len(midi_events['kick']) == 2
        times = [e['time'] for e in midi_events['kick']]
        assert 3.0 in times

    def test_updated_analysis_has_new_statuses(self):
        """Updated analysis data reflects new filter results."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [
                    _make_event(1.0, geomean=100.0, status='KEPT'),
                    _make_event(2.0, geomean=60.0, status='KEPT'),
                ],
                'events_sensitive': [],
            }
        })
        config = _make_config('kick', geomean_threshold=80.0)

        updated, _ = rebuild_events_from_analysis(analysis, {}, config)

        configured = updated['stems']['kick']['events_configured']
        assert configured[0]['status'] == 'KEPT'
        assert configured[1]['status'] == 'FILTERED'

    def test_version_mismatch_raises(self):
        analysis = {'version': '1.0', 'stems': {}}
        config = _make_config('kick')

        with pytest.raises(ValueError, match="not supported"):
            rebuild_events_from_analysis(analysis, {}, config)

    def test_empty_analysis_raises(self):
        with pytest.raises(ValueError, match="No analysis data"):
            rebuild_events_from_analysis(None, {}, {})

    def test_no_stems_raises(self):
        analysis = {'version': '3.0', 'stems': {}}
        config = _make_config('kick')

        with pytest.raises(ValueError, match="no stem data"):
            rebuild_events_from_analysis(analysis, {}, config)

    def test_missing_stem_raises(self):
        analysis = _make_analysis_data({
            'kick': {
                'logic': {},
                'events_configured': [],
                'events_sensitive': [],
            }
        })
        config = _make_config('kick')

        with pytest.raises(ValueError, match="Stems not in analysis"):
            rebuild_events_from_analysis(
                analysis, {}, config, stem_types=['snare']
            )

    def test_does_not_mutate_input(self):
        """Rebuild must not mutate the input analysis_data dict."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [
                    _make_event(1.0, geomean=100.0, status='KEPT'),
                    _make_event(2.0, geomean=60.0, status='KEPT'),
                ],
                'events_sensitive': [],
            }
        })
        original = copy.deepcopy(analysis)
        config = _make_config('kick', geomean_threshold=80.0)

        rebuild_events_from_analysis(analysis, {}, config)

        # Input should be unchanged
        assert analysis == original

    def test_multi_stem_rebuild(self):
        """Rebuild works across multiple stems simultaneously."""
        analysis = _make_analysis_data({
            'kick': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [_make_event(1.0, geomean=100.0)],
                'events_sensitive': [],
            },
            'snare': {
                'logic': {'geomean_threshold': 50.0},
                'events_configured': [_make_event(2.0, geomean=100.0)],
                'events_sensitive': [],
            },
        })
        config = _make_config('kick', geomean_threshold=50.0)
        config['snare'] = {
            'geomean_threshold': 50.0,
            'low_freq_min': 100, 'low_freq_max': 300,
            'body_freq_min': 200, 'body_freq_max': 800,
            'wire_freq_min': 4000, 'wire_freq_max': 8000,
        }

        updated, midi_events = rebuild_events_from_analysis(analysis, {}, config)

        assert 'kick' in midi_events
        assert 'snare' in midi_events
        assert len(midi_events['kick']) == 1
        assert len(midi_events['snare']) == 1
