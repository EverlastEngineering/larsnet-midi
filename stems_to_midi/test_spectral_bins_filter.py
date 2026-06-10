"""
TDD test for the band-ratio quality filter in _build_events_configured.

Background (2026-06-09, project 4 toms 73-77s, second iteration):
The spectral detector found 11 events in 73-77s. 6 are real hits
(band_max_ratio >= 2.0, mostly >= 100), 5 are tail false positives
(band_max_ratio < 2.0).
The 'both' mode in _build_events_configured adds ALL spectral events
to events_configured, polluting the sidecar with 5 false positives per
run.

Fix: drop spectral events with band_max_ratio < 2.0 before adding
them to events_configured. NO energy-coupling — the spectral detector
runs in isolation; this is purely a quality floor on the spectral
signal itself.

User direction (2026-06-09): "spectral data should NOT use ANY data
from the original system. for now, JUST use spectral data for the
spectral events. I want to develop this system in isolation from the
other for now."

Replaces the earlier bins-floor (SPECTRAL_BINS_FLOOR=159) filter from
2026-06-09 (bins-above-floor superseded by per-band power profile).
"""

import pytest

from stems_to_midi.processing_shell import _build_events_configured


class TestSpectralBinsFilterInBothMode:
    """The band_max_ratio >= 2.0 filter drops weak spectral events
    (the tail false positives) without coupling to the energy detector.
    """

    def test_both_mode_drops_weak_spectral_events(self):
        """With detection_method='both', spectral events with
        band_max_ratio < 1.2 must be dropped from events_configured
        (no coupling to energy events)."""
        # 6 strong spectral (the real hits) + 5 weak (the false positives)
        spectral_onset_data = [
            # Real hits — high band_max_ratio (a real hit lights up one band
            # much more than the others)
            {"time": 73.700, "band_powers": [9.5e+00, 9.1e-03, 1.5e-03, 9.8e-04, 1.9e-03],
             "band_max_idx": 0, "band_max_ratio": 1050.0, "strength": 1.00},
            {"time": 73.868, "band_powers": [2.2e+02, 6.4e-01, 1.9e-02, 2.2e-03, 9.1e-03],
             "band_max_idx": 0, "band_max_ratio": 340.0, "strength": 1.00},
            {"time": 74.066, "band_powers": [1.4e+02, 3.4e-01, 1.1e-02, 6.9e-03, 9.2e-03],
             "band_max_idx": 0, "band_max_ratio": 397.0, "strength": 1.00},
            {"time": 74.234, "band_powers": [5.7e+02, 3.5e+01, 7.8e-02, 4.0e-02, 5.8e-02],
             "band_max_idx": 0, "band_max_ratio": 16.0, "strength": 1.00},
            {"time": 74.420, "band_powers": [3.3e+02, 3.1e+00, 1.7e-02, 9.6e-03, 2.0e-02],
             "band_max_idx": 0, "band_max_ratio": 108.0, "strength": 1.00},
            {"time": 74.600, "band_powers": [9.6e+02, 4.1e+00, 5.3e-02, 3.2e-02, 2.4e-02],
             "band_max_idx": 0, "band_max_ratio": 232.0, "strength": 1.00},
            # False positives (post-hit tail) — low band_max_ratio (the
            # spectral energy is spread across bands, no clear winner)
            # All below the 1.0 floor.
            {"time": 74.797, "band_powers": [1.5e-02, 3.2e-03, 4.2e-03, 1.1e-02, 5.2e-03],
             "band_max_idx": 3, "band_max_ratio": 0.92, "strength": 0.11},
            {"time": 74.931, "band_powers": [1.2e-02, 5.0e-03, 6.0e-03, 8.0e-03, 4.0e-03],
             "band_max_idx": 3, "band_max_ratio": 0.95, "strength": 0.12},
            {"time": 75.053, "band_powers": [3.0e-04, 1.0e-04, 4.0e-04, 1.5e-03, 8.0e-04],
             "band_max_idx": 3, "band_max_ratio": 0.85, "strength": 0.10},
            {"time": 75.175, "band_powers": [1.5e-04, 8.0e-05, 3.0e-04, 1.2e-03, 7.0e-04],
             "band_max_idx": 3, "band_max_ratio": 0.87, "strength": 0.10},
            {"time": 75.355, "band_powers": [4.0e-04, 2.0e-04, 5.0e-04, 1.8e-03, 1.0e-03],
             "band_max_idx": 3, "band_max_ratio": 0.92, "strength": 0.11},
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
            "expected only the 6 strong (band_max_ratio >= 1.0) spectral "
            "events, got {} events at times: {}".format(
                len(result_times),
                [(t, ev.get("band_max_ratio")) for t, ev in zip(result_times, result)],
            )
        )

    def test_both_mode_does_not_couple_to_energy_events(self):
        """The band-ratio filter applies to spectral events INDEPENDENTLY
        of the energy detector's output. A weak spectral event must be
        dropped even if there's a strong energy event nearby, and a
        strong spectral event must be added even if there's a strong
        energy event nearby."""
        spectral_onset_data = [
            # Strong spectral + nearby energy (would have been deduped before)
            {"time": 5.000, "band_powers": [1.0, 0.001, 0.001, 0.001, 0.001],
             "band_max_idx": 0, "band_max_ratio": 1000.0, "strength": 1.0},
            # Weak spectral + nearby energy (no promotion); ratio=0.95 below 1.0
            {"time": 6.000, "band_powers": [0.5, 0.45, 0.3, 0.2, 0.1],
             "band_max_idx": 0, "band_max_ratio": 0.95, "strength": 0.11},
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
        result_times = sorted(ev["time"] for ev in result)
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
            "weak spectral at 6.000 (ratio=0.95) should be dropped, got {}".format(result_times)
        )

    def test_spectral_mode_also_applies_band_ratio_filter(self):
        """The band-ratio filter must apply in 'spectral' mode too (the
        user wants to see only strong spectral events, not the weak ones)."""
        spectral_onset_data = [
            {"time": 1.0, "band_powers": [1.0, 0.001, 0.001, 0.001, 0.001],
             "band_max_idx": 0, "band_max_ratio": 1000.0, "strength": 1.0},
            # ratio=0.95 below the 1.0 floor
            {"time": 2.0, "band_powers": [0.5, 0.45, 0.3, 0.2, 0.1],
             "band_max_idx": 0, "band_max_ratio": 0.95, "strength": 0.11},
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
            "weak spectral at 2.0 (ratio=0.95) should be dropped even in 'spectral' mode"
        )

    def test_energy_mode_unchanged(self):
        """The band-ratio filter is spectral-only. Energy-mode events_configured
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
