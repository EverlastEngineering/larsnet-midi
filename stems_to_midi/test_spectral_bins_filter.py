"""
TDD test for the spectral bins filter in _build_events_configured.

Background (2026-06-09, project 4 toms 73-77s):
The spectral detector found 11 events in 73-77s. 6 are real hits
(bins >= 150, mostly 167), 5 are tail false positives (bins < 150).
The 'both' mode in _build_events_configured currently adds ALL
spectral events to events_configured, polluting the sidecar with
5 false positives per run.

Fix: drop spectral events with bins_above_floor < 150 before adding
them to events_configured. NO energy-coupling — the spectral detector
runs in isolation; this is purely a quality floor on the spectral
signal itself.

User direction (2026-06-09): "spectral data should NOT use ANY data
from the original system. for now, JUST use spectral data for the
spectral events. I want to develop this system in isolation from
the other for now."

The threshold (150 in SPECTRAL_BINS_FLOOR; the 159 number in test
data is the data-driven minimum for a real hit in project 4 toms
73-77s) is a quality floor on the spectral signal: only events
where a strong majority of the 167 high-freq bins crossed the -50dB
floor count as hits.
"""

import pytest

from stems_to_midi.processing_shell import _build_events_configured


class TestSpectralBinsFilterInBothMode:
    """The bins >= 150 filter drops weak spectral events (the tail
    false positives) without coupling to the energy detector.
    """

    def test_both_mode_drops_weak_spectral_events(self):
        """With detection_method='both', spectral events with
        bins_above_floor < 150 must be dropped from events_configured
        (no coupling to energy events)."""
        # 6 strong spectral (the real hits) + 5 weak (the false positives)
        spectral_onset_data = [
            # Real hits
            {"time": 73.700, "bins_above_floor": 159, "max_db": -13.4, "strength": 0.95},
            {"time": 73.868, "bins_above_floor": 167, "max_db": -0.4,  "strength": 1.00},
            {"time": 74.066, "bins_above_floor": 167, "max_db": -14.4, "strength": 1.00},
            {"time": 74.234, "bins_above_floor": 167, "max_db": -12.9, "strength": 1.00},
            {"time": 74.420, "bins_above_floor": 167, "max_db": -9.3,  "strength": 1.00},
            {"time": 74.600, "bins_above_floor": 167, "max_db": -0.7,  "strength": 1.00},
            # False positives (post-hit tail)
            {"time": 74.797, "bins_above_floor": 120, "max_db": -24.6, "strength": 0.72},
            {"time": 74.931, "bins_above_floor": 99,  "max_db": -29.3, "strength": 0.59},
            {"time": 75.053, "bins_above_floor": 19,  "max_db": -41.3, "strength": 0.11},
            {"time": 75.175, "bins_above_floor": 16,  "max_db": -45.0, "strength": 0.10},
            {"time": 75.355, "bins_above_floor": 41,  "max_db": -44.6, "strength": 0.25},
        ]
        # No energy events at all — pure isolation test
        result = _build_events_configured(
            all_onset_data=[],
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method="both",
        )

        result_times = sorted(ev["time"] for ev in result)
        expected_strong_times = sorted([
            73.700, 73.868, 74.066, 74.234, 74.420, 74.600,
        ])
        assert result_times == expected_strong_times, (
            "expected only the 6 strong (bins >= 150) spectral events, "
            "got {} events at times: {}".format(
                len(result_times),
                [(t, e.get("bins_above_floor")) for t, e in zip(result_times, result)],
            )
        )

    def test_both_mode_does_not_couple_to_energy_events(self):
        """The bins filter applies to spectral events INDEPENDENTLY
        of the energy detector's output. A weak spectral event must
        be dropped even if there's a strong energy event nearby, and
        a strong spectral event must be added even if there's a
        strong energy event nearby."""
        spectral_onset_data = [
            # Strong spectral + nearby energy (would have been deduped before)
            {"time": 5.000, "bins_above_floor": 167, "max_db": -5.0, "strength": 1.0},
            # Weak spectral + nearby energy (no promotion)
            {"time": 6.000, "bins_above_floor": 50,  "max_db": -30.0, "strength": 0.3},
        ]
        # Energy events at almost the same times as the spectral ones
        all_onset_data = [
            {"time": 5.005, "status": "KEPT", "method": "peak_hold", "note": 45, "pitch_hz": 100.0},
            {"time": 6.005, "status": "KEPT", "method": "peak_hold", "note": 45, "pitch_hz": 100.0},
        ]
        result = _build_events_configured(
            all_onset_data=list(all_onset_data),
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method="both",
        )
        # Should contain BOTH energy events (passed through) plus
        # the strong spectral event at 5.000 (passes bins filter).
        # The weak spectral event at 6.000 is dropped (bins < 150).
        result_times = sorted(ev["time"] for ev in result)
        # The energy events at 5.005 and 6.005 pass through. The
        # strong spectral at 5.000 adds a new entry; the weak
        # spectral at 6.000 is filtered.
        assert any(abs(t - 5.000) < 0.001 for t in result_times), (
            "expected strong spectral at 5.000 in result, got {}".format(result_times)
        )
        assert any(abs(t - 5.005) < 0.001 for t in result_times), (
            "expected energy event at 5.005 in result, got {}".format(result_times)
        )
        assert any(abs(t - 6.005) < 0.001 for t in result_times), (
            "expected energy event at 6.005 in result, got {}".format(result_times)
        )
        assert not any(abs(t - 6.000) < 0.001 for t in result_times), (
            "weak spectral at 6.000 (bins=50) should be dropped, got {}".format(result_times)
        )

    def test_spectral_mode_also_applies_bins_filter(self):
        """The bins filter must apply in 'spectral' mode too (the
        user wants to see only strong spectral events, not the
        weak ones)."""
        spectral_onset_data = [
            {"time": 1.0, "bins_above_floor": 167, "max_db": -5.0, "strength": 1.0},
            {"time": 2.0, "bins_above_floor": 30,  "max_db": -40.0, "strength": 0.2},
        ]
        result = _build_events_configured(
            all_onset_data=[],
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method="spectral",
        )
        result_times = [ev["time"] for ev in result]
        assert 1.0 in result_times
        assert 2.0 not in result_times, (
            "weak spectral at 2.0 (bins=30) should be dropped even in 'spectral' mode"
        )

    def test_energy_mode_unchanged(self):
        """The bins filter is spectral-only. Energy-mode events_configured
        must not be affected."""
        all_onset_data = [
            {"time": 1.0, "status": "KEPT", "method": "peak_hold", "note": 36, "pitch_hz": 50.0},
            {"time": 2.0, "status": "REVERB_CONTINUATION", "method": "peak_hold"},
        ]
        result = _build_events_configured(
            all_onset_data=list(all_onset_data),
            spectral_onset_data=[],
            midi_events=[],
            detection_method="energy",
        )
        assert len(result) == 2
        assert result[0]["time"] == 1.0
        assert result[1]["time"] == 2.0
