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

    def test_spectral_fields_preserved_when_present(self):
        """Bug (2026-06-09, second iteration): _serialize_onset_events
        was dropping band_powers, band_max_idx, and band_max_ratio
        from spectral events that survive into events_configured.
        The WebUI tooltip (webui-tooltip-bands task) wants to show
        the per-band profile for spectral events, but if the sidecar
        strips the field, the tooltip can't render it.
        _build_events_configured produces spectral events with these
        fields set; the serializer must round and write them through.

        (Earlier this test asserted on bins_above_floor / max_db;
        those were superseded by the per-band profile on 2026-06-09.)
        """
        onset = {
            'time': 1.2345, 'status': 'KEPT', 'method': 'spectral',
            'strength': 0.95,
            'band_powers': [1.0e+00, 5.0e-04, 1.0e-04, 2.0e-05, 1.0e-05],
            'band_max_idx': 0,
            'band_max_ratio': 2000.0,
        }
        events = _serialize_onset_events([onset])
        assert 'band_powers' in events[0], (
            "band_powers was dropped by the serializer; "
            "spectral events in events_configured will not have the "
            "data the WebUI tooltip needs"
        )
        assert 'band_max_idx' in events[0], (
            "band_max_idx was dropped by the serializer"
        )
        assert 'band_max_ratio' in events[0], (
            "band_max_ratio was dropped by the serializer"
        )
        # band_max_idx is an int (0-4)
        assert events[0]['band_max_idx'] == 0
        # band_powers is a 5-list of floats; verify shape and a value
        assert len(events[0]['band_powers']) == 5
        assert events[0]['band_powers'][0] == pytest.approx(1.0, rel=1e-5)
        # band_max_ratio is a float, 4-decimal rounded (the previous
        # 2dp was lossy — e.g. 459.12 became 459.12 but a 4-decimal
        # round preserves the full precision the user needs)
        assert events[0]['band_max_ratio'] == 2000.0

    def test_snap_delta_high_precision_preserved(self):
        """Regression test (2026-06-10): the JSON serializer was
        rounding snap_delta to 2 decimal places, which collapsed
        every small-magnitude signal to 0.00. The user's
        calibration case had snap_delta values in the 0.0001-0.001
        range (real toms hits) and they were all serialized as
        0.0 in the sidecar — so the user thought the spectral
        detector was broken. The fix: snap_delta, band_delta,
        snap_to_ring_ratio, snap_to_top_ratio get 6-decimal
        precision in the JSON (same as band_powers)."""
        onset = {
            'time': 14.722, 'status': 'KEPT', 'method': 'spectral',
            # The user's actual small snap_delta value
            'snap_delta': 0.000352,
            # The corresponding large band_delta
            'band_delta': 809.3249,
            # A small snap_to_ring_ratio
            'snap_to_ring_ratio': 0.000352 / 809.3249,
            'snap_to_top_ratio': 0.000352 / 36.37,
            'band_max_ratio': 36.37,
            'band_max_idx': 0,
            'band_powers': [1.0, 0.5, 0.1, 0.05, 0.01],
        }
        events = _serialize_onset_events([onset])

        # The signal MUST survive serialization. Pre-fix this was
        # 0.0 in the JSON, which made the user think the detector
        # was broken when it was actually working correctly.
        assert events[0]['snap_delta'] == pytest.approx(0.000352, rel=1e-5), (
            f"snap_delta=0.000352 must survive serialization. "
            f"Got: {events[0]['snap_delta']!r}"
        )
        assert events[0]['band_delta'] == pytest.approx(809.3249, rel=1e-5), (
            f"band_delta=809.3249 must survive serialization. "
            f"Got: {events[0]['band_delta']!r}"
        )
        # The ratios also need precision — at 2dp they'd be 0.0
        ratio = 0.000352 / 809.3249
        assert events[0]['snap_to_ring_ratio'] == pytest.approx(ratio, rel=1e-5), (
            f"snap_to_ring_ratio must survive serialization. "
            f"Got: {events[0]['snap_to_ring_ratio']!r}"
        )

    def test_snap_to_ring_ratio_small_value_survives(self):
        """Regression test (2026-06-10): snap_to_ring_ratio values
        in the 1e-7 to 1e-5 range must NOT round to 0.0. The
        serializer uses significant-figure rounding (6 sig figs)
        for these fields rather than fixed decimal places, so
        4.35e-7 stays as 4.35e-7 instead of disappearing.

        This is the same data the user's calibration case hit:
        ring=665, snap=0.01 → ratio 1.5e-5 (locked in
        test_spectral_transient_core.py). Pre-fix, the JSON
        rounded it to 0.0."""
        onset = {
            'time': 14.0, 'status': 'KEPT', 'method': 'spectral',
            'snap_delta': 0.01,
            'band_delta': 665.0,
            'snap_to_ring_ratio': 0.01 / 665.0,  # 1.5e-5
            'band_max_ratio': 2.0,
            'band_powers': [1, 1, 1, 1, 1],
        }
        events = _serialize_onset_events([onset])
        expected = 0.01 / 665.0  # ~1.5038e-5
        assert events[0]['snap_to_ring_ratio'] != 0.0, (
            f"snap_to_ring_ratio={expected:.3e} must NOT round to 0.0. "
            f"Got: {events[0]['snap_to_ring_ratio']!r}"
        )
        assert events[0]['snap_to_ring_ratio'] == pytest.approx(
            expected, rel=1e-5
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


# ============================================================================
# Validator tolerance — bug C round 2 (2026-06-08)
# ============================================================================


class TestValidateEventsSubsetHopTolerance:
    """Round 2 of bug C (2026-06-08). The configured and sensitive
    detection runs are two separate calls to detect_onsets_energy_based
    with different thresholds. For stereo stems, the merge in
    energy_detection_core.py:507 picks ``min(left_peak_time,
    right_peak_time)``. Because the two passes find different sets of
    L/R peaks (the sensitive pass catches quieter hits the configured
    pass missed), the merged onset time can land on a different hop
    for the same physical hit. The hop duration at 512 samples /
    44.1kHz is 11.61ms.

    The validator's old 1ms tolerance was tighter than the actual
    quantization step, so it produced false-positive toasters for
    legitimate stereo events. The fix widens the tolerance to ~12ms
    (one hop). Real data-integrity issues (hand-edited analysis.json,
    events written to the wrong array) still trigger warnings — the
    gap from those is at least hundreds of ms.

    User report (2026-06-08): "I get a bunch of toasters when I
    convert a MIDI with this in it" — snare had 7-11 missing events,
    all with status=KEPT, all off by exactly 11.61ms from the
    nearest sensitive event.
    """

    def _write_sidecar(self, midi_path, data):
        sidecar_path = midi_path.with_suffix('.analysis.json')
        with open(sidecar_path, 'w') as f:
            json.dump(data, f)
        return sidecar_path

    def _make_midi_path(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
        tmp.close()
        return Path(tmp.name)

    def test_11ms_gap_within_tolerance_no_warning(self):
        """An 11.61ms gap (one hop at 512/44100) is the maximum the
        detection pipeline can produce. The validator must NOT warn
        about it. (Old 1ms tolerance produced a false-positive
        toaster.)"""
        midi_path = self._make_midi_path()
        try:
            sidecar = {
                'version': '3.0',
                'stems': {
                    'snare': {
                        'events_configured': [
                            {'time': 1.0,        'status': 'KEPT'},
                            {'time': 2.0,        'status': 'KEPT'},
                            {'time': 3.0,        'status': 'KEPT'},
                        ],
                        'events_sensitive': [
                            # Each configured event is exactly one hop
                            # away from a sensitive event.
                            {'time': 1.0 + 0.01161, 'status': 'KEPT'},
                            {'time': 2.0 - 0.01161, 'status': 'KEPT'},
                            {'time': 3.0 + 0.01161, 'status': 'KEPT'},
                        ],
                    }
                }
            }
            self._write_sidecar(midi_path, sidecar)
            data = load_analysis_sidecar(midi_path)
            warnings = data.get('data_integrity_warnings', [])
            assert warnings == [], (
                f"Validator should not warn about one-hop gap (11.61ms). "
                f"This is the maximum the stereo merge can produce. "
                f"Got warnings: {warnings}"
            )
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

    def test_20ms_gap_still_warns(self):
        """A 20ms gap is well past one hop and indicates a real
        data-integrity issue (event written to the wrong array,
        hand-edited analysis.json, etc.). The validator must
        still warn."""
        midi_path = self._make_midi_path()
        try:
            sidecar = {
                'version': '3.0',
                'stems': {
                    'snare': {
                        'events_configured': [{'time': 1.0, 'status': 'KEPT'}],
                        'events_sensitive':  [{'time': 1.020, 'status': 'KEPT'}],
                    }
                }
            }
            self._write_sidecar(midi_path, sidecar)
            data = load_analysis_sidecar(midi_path)
            warnings = data.get('data_integrity_warnings', [])
            assert len(warnings) == 1, (
                f"20ms gap should still produce a warning (real "
                f"data-integrity issue). Got: {warnings}"
            )
            assert 'snare' in warnings[0]
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

    def test_1ms_gap_still_passes(self):
        """Sub-hop gap (1ms) is the original test case and must
        continue to pass — the validator didn't get looser about
        tight timings, only about the hop quantization."""
        midi_path = self._make_midi_path()
        try:
            sidecar = {
                'version': '3.0',
                'stems': {
                    'hihat': {
                        'events_configured': [{'time': 1.0,   'status': 'KEPT'}],
                        'events_sensitive':  [{'time': 1.001, 'status': 'KEPT'}],
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

    def test_tolerance_default_includes_hop_duration(self):
        """The validator's default tolerance should be >= one hop
        duration. Locks the contract that future maintainers see
        the rationale: 11.61ms = 512 / 44100."""
        from stems_to_midi.midi import _validate_events_subset
        # The function signature exposes time_tolerance_sec as the
        # second positional arg. Call it with an explicit tolerance
        # of 0.012 (12ms) and assert a synthetic one-hop gap
        # produces no warnings.
        data = {
            'stems': {
                'snare': {
                    'events_configured': [{'time': 1.0,        'status': 'KEPT'}],
                    'events_sensitive':  [{'time': 1.0 + 0.01161, 'status': 'KEPT'}],
                }
            }
        }
        # 12ms tolerance must accept a 11.61ms gap
        warnings = _validate_events_subset(data, time_tolerance_sec=0.012)
        assert warnings == [], (
            f"12ms tolerance should accept a 11.61ms gap. Got: {warnings}"
        )

    def test_legacy_1ms_tolerance_still_catches_5ms_gap(self):
        """Pin the old behavior so a future widening doesn't
        accidentally re-introduce this false-positive. The legacy
        1ms tolerance correctly catches a 5ms gap (which is well
        past one hop and indicates a real bug)."""
        from stems_to_midi.midi import _validate_events_subset
        data = {
            'stems': {
                'snare': {
                    'events_configured': [{'time': 1.0,   'status': 'KEPT'}],
                    'events_sensitive':  [{'time': 1.005, 'status': 'KEPT'}],
                }
            }
        }
        warnings = _validate_events_subset(data, time_tolerance_sec=0.001)
        assert len(warnings) == 1
        assert 'snare' in warnings[0]


# ============================================================================
# hihat_state serialization (T2 follow-up, 2026-06-08)
# ============================================================================


class TestSerializeHihatState:
    """T2 A4 added 'preserve stored hihat_state on rebuild' — but
    hihat_state is only ever set in the in-memory event dict (by
    classify_hihat_notes in note_classification_core.py). It is never
    propagated to onset_data, so _serialize_onset_events never writes
    it to the JSON. Result: sidecar has no hihat_state field, T3 e2e
    found 'hihat_state field missing from all 13 hihat KEPT events in
    fresh conversion (baseline had 13/13)'.

    The fix: _serialize_onset_events must surface hihat_state when the
    midi_event param carries it. These tests assert the contract."""

    def test_hihat_state_in_serialized_event_when_provided(self):
        """When midi_events[i] has hihat_state='open', the serialized
        event must include hihat_state='open'."""
        onset = {
            'time': 1.0,
            'status': 'KEPT',
            'sustain_ms': 200,
            'geomean': 400.0,
            'pan_confidence': 0.0,
            'stereo_width': 0.1,
            'pitch_hz': None,
        }
        midi_event = {
            'time': 1.0,
            'note': 46,
            'velocity': 100,
            'hihat_state': 'open',
        }
        events = _serialize_onset_events([onset], midi_events=[midi_event])
        assert events[0].get('hihat_state') == 'open', (
            f"hihat_state='open' was passed in midi_events but is missing "
            f"from serialized event: {events[0]}"
        )

    def test_hihat_state_closed_propagates(self):
        """closed state propagates too — not just open."""
        onset = {
            'time': 2.0, 'status': 'KEPT',
            'sustain_ms': 80, 'geomean': 300.0,
            'pan_confidence': 0.0, 'stereo_width': 0.1, 'pitch_hz': None,
        }
        midi_event = {
            'time': 2.0, 'note': 42, 'velocity': 100,
            'hihat_state': 'closed',
        }
        events = _serialize_onset_events([onset], midi_events=[midi_event])
        assert events[0].get('hihat_state') == 'closed'

    def test_hihat_state_handclap_propagates(self):
        """handclap state propagates (T2 added a separate MIDI note for
        handclap bleed from hihat)."""
        onset = {
            'time': 3.0, 'status': 'KEPT',
            'sustain_ms': 20, 'geomean': 200.0,
            'pan_confidence': 0.0, 'stereo_width': 0.1, 'pitch_hz': None,
        }
        midi_event = {
            'time': 3.0, 'note': 39, 'velocity': 80,
            'hihat_state': 'handclap',
        }
        events = _serialize_onset_events([onset], midi_events=[midi_event])
        assert events[0].get('hihat_state') == 'handclap'

    def test_hihat_state_omitted_when_not_a_hihat_event(self):
        """For non-hihat events (or when midi_event lacks hihat_state),
        the field is not written. This keeps the schema clean for
        stems that don't use the field."""
        onset = {
            'time': 1.0, 'status': 'KEPT',
            'sustain_ms': 80, 'geomean': 200.0,
            'pan_confidence': 0.0, 'stereo_width': 0.1, 'pitch_hz': None,
        }
        midi_event = {'time': 1.0, 'note': 36, 'velocity': 99}
        # No hihat_state in midi_event
        events = _serialize_onset_events([onset], midi_events=[midi_event])
        # hihat_state should not be present (or should be None / absent)
        assert events[0].get('hihat_state') in (None, '', 'closed'), (
            f"hihat_state should not be set when midi_event lacks it: {events[0]}"
        )

    def test_hihat_state_round_trip_through_save_analysis_sidecar(self, tmp_path):
        """End-to-end: save the sidecar to disk, reload, hihat_state
        is preserved. This is the real contract — the JSON file must
        contain the field so the WebUI reclassify + tuning flows see it.
        """
        from stems_to_midi.midi import save_analysis_sidecar, load_analysis_sidecar

        onset = {
            'time': 1.0, 'status': 'KEPT', 'strength': 2.0,
            'sustain_ms': 200, 'geomean': 400.0,
            'pan_confidence': 0.0, 'stereo_width': 0.1, 'pitch_hz': None,
        }
        midi_events = [
            {'time': 1.0, 'note': 46, 'velocity': 110, 'hihat_state': 'open'},
        ]

        midi_path = tmp_path / 'song.mid'
        midi_path.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x00')

        try:
            save_analysis_sidecar(
                events_by_stem={'hihat': midi_events},
                midi_path=midi_path,
                tempo=120.0,
                analysis_by_stem={
                    'hihat': {
                        'all_onset_data': [onset],
                        'sensitive_onset_data': [],
                        'spectral_config': None,
                    },
                },
                config={},
            )
            data = load_analysis_sidecar(midi_path)
            hihat_events = data['stems']['hihat']['events_configured']
            # Find the event with hihat_state
            states = [e.get('hihat_state') for e in hihat_events]
            assert 'open' in states, (
                f"hihat_state='open' should round-trip through the sidecar. "
                f"Got states: {states}"
            )
        finally:
            midi_path.unlink(missing_ok=True)
            midi_path.with_suffix('.analysis.json').unlink(missing_ok=True)

