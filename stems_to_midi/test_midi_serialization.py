"""
Tests for save_analysis_sidecar / _serialize_onset_events.

Focuses on bug B: pan_confidence / stereo_width / pitch_hz must always
appear in the JSON output, with null when the value is not available
(mono audio, pitch detection disabled, or missing in legacy data).
This contract lets downstream consumers (WebUI tuning panel, JSON-driven
analysis scripts) treat the keys as always-present schema fields.
"""

import json
import tempfile
from pathlib import Path

import pytest

from stems_to_midi.midi import (
    _serialize_onset_events,
    save_analysis_sidecar,
    load_analysis_sidecar,
)


# ============================================================================
# _serialize_onset_events: always-present fields (bug B)
# ============================================================================


class TestSerializeOnsetEventsAlwaysPresentFields:
    """Bug B: pan_confidence / stereo_width / pitch_hz must always be present."""

    def test_pan_confidence_present_when_none(self):
        """pan_confidence key always appears, even when value is None."""
        onset = {'time': 1.0, 'status': 'KEPT'}
        events = _serialize_onset_events([onset])
        assert 'pan_confidence' in events[0]
        assert events[0]['pan_confidence'] is None

    def test_stereo_width_present_when_none(self):
        """stereo_width key always appears, even when value is None."""
        onset = {'time': 1.0, 'status': 'KEPT'}
        events = _serialize_onset_events([onset])
        assert 'stereo_width' in events[0]
        assert events[0]['stereo_width'] is None

    def test_pitch_hz_present_when_none(self):
        """pitch_hz key always appears, even when value is None."""
        onset = {'time': 1.0, 'status': 'KEPT'}
        events = _serialize_onset_events([onset])
        assert 'pitch_hz' in events[0]
        assert events[0]['pitch_hz'] is None

    def test_pan_confidence_value_preserved(self):
        """Numeric pan_confidence is rounded and written."""
        onset = {'time': 1.0, 'status': 'KEPT', 'pan_confidence': -0.3456789}
        events = _serialize_onset_events([onset])
        assert events[0]['pan_confidence'] == -0.3457  # 4-decimal rounding

    def test_stereo_width_value_preserved(self):
        """Numeric stereo_width is rounded and written."""
        onset = {'time': 1.0, 'status': 'KEPT', 'stereo_width': 0.1234567}
        events = _serialize_onset_events([onset])
        assert events[0]['stereo_width'] == 0.1235

    def test_pitch_hz_value_preserved(self):
        """Numeric pitch_hz is rounded and written."""
        onset = {'time': 1.0, 'status': 'KEPT', 'pitch_hz': 245.6789}
        events = _serialize_onset_events([onset])
        assert events[0]['pitch_hz'] == 245.6789

    def test_all_three_present_simultaneously(self):
        """All three stem-relevant fields are present at once."""
        onset = {'time': 1.0, 'status': 'KEPT',
                 'pan_confidence': 0.1, 'stereo_width': 0.2, 'pitch_hz': 100.0}
        events = _serialize_onset_events([onset])
        for key in ('pan_confidence', 'stereo_width', 'pitch_hz'):
            assert key in events[0]

    def test_optional_phase2_fields_still_omitted_when_missing(self):
        """Optional Phase 2 fields are still omitted (not forced to null)."""
        onset = {'time': 1.0, 'status': 'KEPT'}
        events = _serialize_onset_events([onset])
        for optional in ('duration_sec', 'attack_sharpness',
                         'spectral_centroid_hz', 'spectral_flux'):
            assert optional not in events[0], (
                f"Optional field {optional!r} should be omitted, not null"
            )

    def test_basic_fields_still_present(self):
        """time, status, and computed features still work as before."""
        onset = {
            'time': 1.2345, 'status': 'KEPT', 'strength': 0.8,
            'amplitude': 0.3, 'geomean': 50.0, 'sustain_ms': 200.0,
        }
        events = _serialize_onset_events([onset])
        assert events[0]['time'] == 1.2345
        assert events[0]['status'] == 'KEPT'
        assert events[0]['strength'] == 0.8
        assert events[0]['geomean'] == 50.0


# ============================================================================
# save_analysis_sidecar: roundtrip
# ============================================================================


class TestSaveAnalysisSidecarAlwaysPresentFields:
    """End-to-end: save and load a sidecar with the always-present fields."""

    def _make_midi_path(self):
        """Return a temp MIDI path (sidecar derives name from it)."""
        tmp = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
        tmp.close()
        return Path(tmp.name)

    def test_sidecar_has_pan_confidence_for_all_events(self):
        """Every event in sidecar has the pan_confidence key (null or value)."""
        midi_path = self._make_midi_path()
        try:
            analysis_by_stem = {
                'hihat': {
                    'all_onset_data': [
                        {'time': 1.0, 'status': 'KEPT', 'pan_confidence': 0.1, 'stereo_width': 0.05, 'pitch_hz': None},
                        {'time': 2.0, 'status': 'FILTERED'},  # No stereo data
                    ],
                    'sensitive_onset_data': [],
                    'spectral_config': {'geomean_threshold': 50.0, 'min_sustain_ms': 25},
                }
            }
            events_by_stem = {
                'hihat': [
                    {'time': 1.0, 'note': 42, 'velocity': 80, 'duration': 0.1},
                ]
            }
            config = {
                'hihat': {
                    'open_geomean_min': 262.0, 'open_sustain_ms': 100.0,
                    'midi_note_closed': 42, 'midi_note_open': 46,
                }
            }
            save_analysis_sidecar(
                events_by_stem, midi_path, tempo=120.0,
                analysis_by_stem=analysis_by_stem, config=config,
            )
            sidecar_path = midi_path.with_suffix('.analysis.json')
            with open(sidecar_path) as f:
                data = json.load(f)
            hihat_events = data['stems']['hihat']['events_configured']
            for ev in hihat_events:
                assert 'pan_confidence' in ev
                assert 'stereo_width' in ev
                assert 'pitch_hz' in ev
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

    def test_sidecar_null_for_missing_pan_confidence(self):
        """Events without pan_confidence data get null, not key omission."""
        midi_path = self._make_midi_path()
        try:
            analysis_by_stem = {
                'kick': {
                    'all_onset_data': [
                        {'time': 1.0, 'status': 'KEPT'},  # No pan_confidence (mono)
                    ],
                    'sensitive_onset_data': [],
                    'spectral_config': {'geomean_threshold': 50.0},
                }
            }
            events_by_stem = {
                'kick': [{'time': 1.0, 'note': 36, 'velocity': 80, 'duration': 0.1}],
            }
            config = {'kick': {'geomean_threshold': 50.0}}
            save_analysis_sidecar(
                events_by_stem, midi_path, tempo=120.0,
                analysis_by_stem=analysis_by_stem, config=config,
            )
            sidecar_path = midi_path.with_suffix('.analysis.json')
            with open(sidecar_path) as f:
                data = json.load(f)
            kick_event = data['stems']['kick']['events_configured'][0]
            # Keys must exist, value null when not available
            assert 'pan_confidence' in kick_event
            assert kick_event['pan_confidence'] is None
            assert 'stereo_width' in kick_event
            assert kick_event['stereo_width'] is None
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

    def test_sensitive_events_also_have_always_present_fields(self):
        """events_sensitive follows the same null-or-value contract."""
        midi_path = self._make_midi_path()
        try:
            analysis_by_stem = {
                'snare': {
                    'all_onset_data': [
                        {'time': 1.0, 'status': 'KEPT', 'pan_confidence': 0.0, 'stereo_width': 0.0},
                    ],
                    'sensitive_onset_data': [
                        {'time': 1.5, 'status': 'KEPT'},  # No pan_confidence
                    ],
                    'spectral_config': {'geomean_threshold': 50.0},
                }
            }
            events_by_stem = {
                'snare': [{'time': 1.0, 'note': 38, 'velocity': 80, 'duration': 0.1}],
            }
            config = {'snare': {'geomean_threshold': 50.0, 'expected_clusters': 2}}
            save_analysis_sidecar(
                events_by_stem, midi_path, tempo=120.0,
                analysis_by_stem=analysis_by_stem, config=config,
            )
            sidecar_path = midi_path.with_suffix('.analysis.json')
            with open(sidecar_path) as f:
                data = json.load(f)
            sensitive = data['stems']['snare']['events_sensitive']
            assert len(sensitive) == 1
            assert 'pan_confidence' in sensitive[0]
            assert 'stereo_width' in sensitive[0]
            assert 'pitch_hz' in sensitive[0]
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)


# ============================================================================
# load_analysis_sidecar: subset validation (bug C)
# ============================================================================


class TestLoadAnalysisSidecarValidation:
    """Bug C: events_configured must be a subset of events_sensitive by time."""

    def _write_sidecar(self, midi_path, data):
        sidecar_path = midi_path.with_suffix('.analysis.json')
        with open(sidecar_path, 'w') as f:
            json.dump(data, f)
        return sidecar_path

    def _make_midi_path(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
        tmp.close()
        return Path(tmp.name)

    def test_valid_subset_no_warnings(self):
        """All configured events have a matching sensitive time → no warning."""
        midi_path = self._make_midi_path()
        try:
            sidecar = {
                'version': '3.0',
                'stems': {
                    'kick': {
                        'events_configured': [
                            {'time': 1.0, 'status': 'KEPT'},
                            {'time': 2.0, 'status': 'KEPT'},
                        ],
                        'events_sensitive': [
                            {'time': 1.0, 'status': 'KEPT'},
                            {'time': 1.5, 'status': 'KEPT'},
                            {'time': 2.0, 'status': 'KEPT'},
                        ],
                    }
                }
            }
            self._write_sidecar(midi_path, sidecar)
            data = load_analysis_sidecar(midi_path)
            assert 'data_integrity_warnings' not in data or not data['data_integrity_warnings']
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

    def test_missing_event_emits_warning(self):
        """Configured event with no matching sensitive time → warning."""
        midi_path = self._make_midi_path()
        try:
            sidecar = {
                'version': '3.0',
                'stems': {
                    'snare': {
                        'events_configured': [
                            {'time': 1.0, 'status': 'KEPT'},
                            {'time': 5.0, 'status': 'KEPT'},  # Not in sensitive
                        ],
                        'events_sensitive': [
                            {'time': 1.0, 'status': 'KEPT'},
                        ],
                    }
                }
            }
            self._write_sidecar(midi_path, sidecar)
            data = load_analysis_sidecar(midi_path)
            warnings = data.get('data_integrity_warnings', [])
            assert len(warnings) == 1
            assert "snare" in warnings[0]
            assert "1" in warnings[0]  # one missing
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

    def test_time_tolerance_within_1ms(self):
        """Two events within 1ms are considered the same."""
        midi_path = self._make_midi_path()
        try:
            sidecar = {
                'version': '3.0',
                'stems': {
                    'hihat': {
                        'events_configured': [{'time': 1.0, 'status': 'KEPT'}],
                        'events_sensitive': [{'time': 1.0005, 'status': 'KEPT'}],
                    }
                }
            }
            self._write_sidecar(midi_path, sidecar)
            data = load_analysis_sidecar(midi_path)
            warnings = data.get('data_integrity_warnings', [])
            assert warnings == []
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

    def test_empty_sensitive_with_nonempty_configured_warns(self):
        """Edge case: events_configured present, events_sensitive missing."""
        midi_path = self._make_midi_path()
        try:
            sidecar = {
                'version': '3.0',
                'stems': {
                    'kick': {
                        'events_configured': [{'time': 1.0, 'status': 'KEPT'}],
                        'events_sensitive': [],
                    }
                }
            }
            self._write_sidecar(midi_path, sidecar)
            data = load_analysis_sidecar(midi_path)
            warnings = data.get('data_integrity_warnings', [])
            assert len(warnings) == 1
            assert "no events_sensitive" in warnings[0]
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

    def test_returns_none_when_sidecar_missing(self):
        """Missing sidecar → None (not an empty dict)."""
        midi_path = self._make_midi_path()
        try:
            data = load_analysis_sidecar(midi_path)
            assert data is None
        finally:
            midi_path.unlink(missing_ok=True)

    def test_warning_per_stem(self):
        """Multiple stems with violations → one warning per stem."""
        midi_path = self._make_midi_path()
        try:
            sidecar = {
                'version': '3.0',
                'stems': {
                    'kick': {
                        'events_configured': [{'time': 99.0, 'status': 'KEPT'}],
                        'events_sensitive': [{'time': 1.0, 'status': 'KEPT'}],
                    },
                    'snare': {
                        'events_configured': [{'time': 88.0, 'status': 'KEPT'}],
                        'events_sensitive': [{'time': 2.0, 'status': 'KEPT'}],
                    },
                }
            }
            self._write_sidecar(midi_path, sidecar)
            data = load_analysis_sidecar(midi_path)
            warnings = data.get('data_integrity_warnings', [])
            assert len(warnings) == 2
            joined = ' '.join(warnings)
            assert 'kick' in joined
            assert 'snare' in joined
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)
