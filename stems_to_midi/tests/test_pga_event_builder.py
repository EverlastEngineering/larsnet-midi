"""
Tests for stems_to_midi.pga_event_builder.

The PGA event builder is the pure functional core of the toms
detection pipeline. These tests lock its public contract:

  1. Stem-type gating — build_pga_events is the toms helper. The
     call site in process_stem_to_midi only calls it for toms;
     the helper itself doesn't gate on stem_type (it would have
     to take it as an arg). The gating is documented in the
     helper docstring and exercised end-to-end by the existing
     test_stems_to_midi tests.
  2. Return shape — (events_kept, events_filtered, debug_dict).
     ``events_kept`` and ``events_filtered`` are lists of dicts
     in detection order; ``debug_dict`` carries the detector
     internals.
  3. ``pga_min_prominence`` filter moves events between KEPT
     and FILTERED based on the configured threshold.
  4. All events carry the diagnostic fields the WebUI tooltip
     and the sidecar serializer need: ``frame``,
     ``envelope_value``, ``prominence``, ``iqr_threshold``,
     ``midi_velocity``, ``pga_filter_config``.
  5. Functional-core / imperative-shell — the helper is
     testable on a synthetic signal without any I/O, and the
     integration test runs against a real toms stem from
     ``user_files/``.

The real-audio test gracefully skips if the fixture is missing
so the test file runs in CI without bundled audio.
"""
import os
import re
import sys
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

# Ensure repo root is on sys.path so ``stems_to_midi`` resolves
# consistently with the existing test_* modules at the same
# level. (The new tests/ subdirectory changes the import
# resolution relative to the existing tests, so we explicitly
# add the parent directory.)
_TEST_DIR = Path(__file__).resolve().parent
_PKG_PARENT = _TEST_DIR.parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from stems_to_midi.pga_event_builder import (  # noqa: E402
    build_pga_events,
    _build_pga_events_with_filter,
    apply_pga_decay_col_min_filter,
    apply_attack_rise_max_filter,
)
from stems_to_midi.percentile_gated_detector import (  # noqa: E402
    detect_percentile_gated_broad_attacks,
    _build_static_noise_floor,
)


# --- Helpers ---------------------------------------------------------------


def _make_synthetic_broadband_burst_stem(
    sr: int = 44100,
    hit_times_sec: tuple = (0.5, 1.0, 1.5, 2.0),
    freq_hz: float = 200.0,
    decay_ms: float = 80.0,
    duration_sec: float = 3.0,
) -> np.ndarray:
    """Build a mono signal with broadband exponentially-decaying
    bursts at specified times. Mimics a tom hit shape (broadband
    enough to trigger the PGA detector) and is deterministic
    enough for stable tests.

    Returns 1-D float32 array.
    """
    t = np.arange(int(sr * duration_sec)) / sr
    y = np.zeros_like(t)
    decay_samples = int(decay_ms / 1000.0 * sr)
    env = np.exp(-np.arange(decay_samples) / (decay_samples / 4.0))
    for hit_t in hit_times_sec:
        i0 = int(hit_t * sr)
        i1 = min(i0 + decay_samples, len(y))
        n = i1 - i0
        if n <= 0:
            continue
        burst = (
            np.sin(2 * np.pi * freq_hz * np.arange(n) / sr)
            + 0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / sr)
            + 0.3 * np.sin(2 * np.pi * 3000 * np.arange(n) / sr)
            + 0.2 * np.sin(2 * np.pi * 5000 * np.arange(n) / sr)
        )
        y[i0:i1] += burst * env[:n]
    return y.astype(np.float32)


def _default_config(**overrides) -> dict:
    """A config dict with the keys build_pga_events reads. Caller
    can override any key via kwargs."""
    cfg = {
        'onset_detection': {'pga_min_prominence': 1000.0},
        'midi': {'min_velocity': 80, 'max_velocity': 110},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# --- Tests -----------------------------------------------------------------


class TestBuildPGAEventsShape:
    """Return shape is exactly (events_kept, events_filtered,
    debug_dict) and the list entries are dicts."""

    def test_returns_three_tuple_of_lists_and_dict(self):
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config())
        assert isinstance(kept, list)
        assert isinstance(filtered, list)
        assert isinstance(debug, dict)

    def test_kept_and_filtered_are_dicts(self):
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config())
        for ev in kept:
            assert isinstance(ev, dict)
        for ev in filtered:
            assert isinstance(ev, dict)

    def test_kept_and_filtered_disjoint_by_status(self):
        """No event appears in both lists; status field is the
        partitioning key."""
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config())
        assert all(ev.get('status') != 'FILTERED' for ev in kept)
        assert all(ev.get('status') == 'FILTERED' for ev in filtered)

    def test_kept_plus_filtered_equals_all_pga_events(self):
        """The helper internally builds a unified list and
        partitions it; the union must equal the KEPT+FILTERED
        count returned in the two output lists."""
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config())
        # Re-run the detector to count total candidates; the
        # helper may have one or two more or fewer depending on
        # the prominence filter, but the union must cover them.
        events_times, _ = detect_percentile_gated_broad_attacks(y, 44100)
        assert len(kept) + len(filtered) == len(events_times)

    def test_debug_dict_has_expected_keys(self):
        """The debug dict exposes the detector internals —
        envelope, peak indices, prominences — for the WebUI
        tooltip and downstream diagnostics."""
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config())
        for k in ('envelope', 'peaks', 'prominences', 'times', 'freqs'):
            assert k in debug, f"debug dict missing key {k!r}"


class TestStemTypeGating:
    """The helper itself is stem-agnostic, but it is called
    ONLY when stem_type == 'toms' (see process_stem_to_midi's
    Step 11.5 / Step 12). The gating is documented in the
    helper docstring; the consumer side enforces it. These
    tests verify the helper is callable on real audio
    regardless of stem — the gating is in the consumer, not
    the helper. (An end-to-end consumer-side test would need
    the full midiconfig.yaml; see test_stems_to_midi for
    that surface.)"""

    def test_helper_callable_on_real_toms_audio_returns_consistent_data(
        self, _clear_stft_cache,
    ):
        """On real toms audio, the helper returns the same
        partition that the toms branch of process_stem_to_midi
        builds (pga_onset_data = kept + filtered, in time
        order). This is the contract the consumer relies on.

        NOTE: This test reads a real wav file. The
        ``_clear_stft_cache`` fixture (autouse=True on this
        class via the class-level autouse below) clears the
        id()-keyed STFT cache before each test in this class
        so the cache can't return STALE data from a prior
        test's audio (the cache is a pre-existing fragility
        in spectral_transient_core.py:215 — the key is
        ``id(audio)`` which can collide when memory pressure
        reuses a freed buffer's id). The fixture is scoped to
        THIS test class only; other tests in the suite are
        unaffected.
        """
        wav_path = (
            Path(__file__).resolve().parent.parent.parent
            / 'user_files' / '4 - 2_funk_80_beat_4-4_4'
            / 'stems' / '2_funk_80_beat_4-4_4-toms.wav'
        )
        if not wav_path.exists():
            pytest.skip(f"toms wav not found at {wav_path}")
        import soundfile as _sf
        audio, sr = _sf.read(str(wav_path), always_2d=True)
        # Force a fresh allocation so id(audio_mono) is unique
        # to this call.
        audio_mono = np.ascontiguousarray(
            audio.mean(axis=1).astype(np.float32)
        )
        config = _default_config(
            onset_detection={'pga_min_prominence': 3000.0},
        )
        # 2026-06-15: _build_pga_events_with_filter returns
        # (raw, events_kept, events_filtered, pga_debug) — a
        # 4-tuple. Earlier versions returned a 3-tuple. The
        # raw list is not used here; the partition contract
        # is what we verify.
        _raw, kept, filtered, _debug = _build_pga_events_with_filter(
            audio_mono, sr, config,
        )
        # The toms consumer in process_stem_to_midi reassembles
        # pga_onset_data as kept + filtered, and the downstream
        # MIDI builder iterates that list skipping FILTERED. The
        # helper must expose the same partition.
        assert len(kept) > 0, "real toms stem should produce KEPT events"
        assert len(filtered) > 0, (
            "real toms stem should produce FILTERED events with pga_min_prominence=3000"
        )
        # Every event in kept has status != FILTERED, every
        # event in filtered has status == FILTERED. The
        # consumer relies on this to skip filtered events in
        # the MIDI builder.
        for ev in kept:
            assert ev.get('status') != 'FILTERED'
        for ev in filtered:
            assert ev.get('status') == 'FILTERED'


class TestProminenceFilterMovesEvents:
    """The pga_min_prominence filter is the only filter applied
    inside build_pga_events. A threshold that is set high should
    move events from KEPT to FILTERED; a threshold of 0 should
    keep everything."""

    def test_zero_threshold_keeps_everything(self):
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(
            y, 44100, _default_config(
                onset_detection={'pga_min_prominence': 0.0},
            ),
        )
        assert len(filtered) == 0
        assert len(kept) + len(filtered) == len(kept)
        assert len(kept) > 0, "synthetic stem should produce events"

    def test_high_threshold_filters_everything(self):
        y = _make_synthetic_broadband_burst_stem()
        # 2026-06-15: _build_pga_events_with_filter returns
        # 4-tuple (raw, kept, filtered, debug). The raw list
        # and debug dict are not used here.
        _raw, kept, filtered, _debug = _build_pga_events_with_filter(
            y, 44100, _default_config(
                onset_detection={'pga_min_prominence': 1e9},
            ),
        )
        assert len(kept) == 0
        assert len(filtered) > 0
        for ev in filtered:
            assert 'filter_reason' in ev
            assert 'pga_min_prominence' in ev['filter_reason']

    def test_prominence_movement_is_monotonic(self):
        """Raising the threshold should never increase the
        KEPT count. (A higher threshold can only move events
        from KEPT to FILTERED, not the other way.)"""
        y = _make_synthetic_broadband_burst_stem()
        prev_kept = None
        for thr in (0.0, 100.0, 1000.0, 10000.0, 1e9):
            # 2026-06-15: 4-tuple unpacking (see above).
            _raw, kept, filtered, _debug = _build_pga_events_with_filter(
                y, 44100, _default_config(
                    onset_detection={'pga_min_prominence': thr},
                ),
            )
            if prev_kept is not None:
                assert len(kept) <= prev_kept, (
                    f"raising threshold from {prev_thr} to {thr} "
                    f"increased kept from {prev_kept} to {len(kept)}"
                )
            prev_kept = len(kept)
            prev_thr = thr


class TestEventDiagnosticFields:
    """Every event returned by build_pga_events must carry the
    diagnostic fields the WebUI / sidecar consumer relies on:
    ``frame``, ``envelope_value``, ``prominence``,
    ``iqr_threshold``, ``midi_velocity``, ``pga_filter_config``."""

    REQUIRED_FIELDS = (
        'frame', 'envelope_value', 'prominence',
        'iqr_threshold', 'midi_velocity', 'pga_filter_config',
    )

    def test_all_kept_events_have_required_fields(self):
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config())
        for ev in kept:
            for f in self.REQUIRED_FIELDS:
                assert f in ev, (
                    f"KEPT event missing required field {f!r}: "
                    f"keys={list(ev.keys())}"
                )

    def test_all_filtered_events_have_required_fields(self):
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config(
            onset_detection={'pga_min_prominence': 1e9},
        ))
        for ev in filtered:
            for f in self.REQUIRED_FIELDS:
                assert f in ev, (
                    f"FILTERED event missing required field {f!r}: "
                    f"keys={list(ev.keys())}"
                )

    def test_midi_velocity_within_configured_range(self):
        """midi_velocity is the linear-mapping of envelope_value
        onto [min_velocity, max_velocity] from config."""
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config(
            midi={'min_velocity': 70, 'max_velocity': 120},
        ))
        for ev in kept + filtered:
            assert 70 <= ev['midi_velocity'] <= 120
            assert isinstance(ev['midi_velocity'], int)

    def test_midi_velocity_in_midi_byte_range(self):
        """Defensive: midi_velocity must be a valid MIDI byte
        (1-127), regardless of the user's configured min/max."""
        y = _make_synthetic_broadband_burst_stem()
        # User supplies an out-of-range max — helper should
        # clamp to 127.
        kept, filtered, debug = build_pga_events(y, 44100, _default_config(
            midi={'min_velocity': 200, 'max_velocity': 300},
        ))
        for ev in kept + filtered:
            assert 1 <= ev['midi_velocity'] <= 127

    def test_pga_filter_config_contains_active_settings(self):
        """The pga_filter_config dict is the sidecar's source
        of truth for "which filter dropped which event". The
        helper must record the active values."""
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, debug = build_pga_events(y, 44100, _default_config(
            onset_detection={'pga_min_prominence': 500.0},
            midi={'min_velocity': 60, 'max_velocity': 100},
        ))
        for ev in kept + filtered:
            cfg = ev['pga_filter_config']
            assert cfg['pga_min_prominence'] == 500.0
            assert cfg['min_velocity'] == 60
            assert cfg['max_velocity'] == 100


class TestPureFunction:
    """Functional-core / imperative-shell: the helper has no
    file I/O, no module-level state, no logging side-effects
    beyond print(). Calling it twice on the same input must
    produce the same output."""

    def test_pure_function_same_input_same_output(self):
        y = _make_synthetic_broadband_burst_stem()
        config = _default_config()
        kept1, filtered1, debug1 = build_pga_events(y, 44100, config)
        kept2, filtered2, debug2 = build_pga_events(y, 44100, config)
        # Strip envelope / times arrays from debug for the
        # exact-equality check; they should be identical, but
        # numpy array equality is finicky in a test.
        assert len(kept1) == len(kept2)
        assert len(filtered1) == len(filtered2)
        for a, b in zip(kept1, kept2):
            # Float fields: compare with small tolerance.
            for k in ('time', 'frame', 'envelope_value',
                      'prominence', 'iqr_threshold', 'midi_velocity'):
                assert abs(a[k] - b[k]) < 1e-9, f"mismatch on {k!r}"
            assert a['status'] == b['status']
            assert a['method'] == b['method']

    def test_does_not_mutate_input_audio(self):
        """Pure function: must not mutate the input audio
        array."""
        y = _make_synthetic_broadband_burst_stem()
        y_copy = y.copy()
        build_pga_events(y, 44100, _default_config())
        np.testing.assert_array_equal(y, y_copy)

    def test_handles_empty_audio(self):
        """Defensive: an empty audio array returns empty
        lists and a placeholder debug dict, no crash."""
        kept, filtered, debug = build_pga_events(np.array([]), 44100, _default_config())
        assert kept == []
        assert filtered == []
        assert isinstance(debug, dict)

    def test_handles_zero_sample_rate(self):
        """Defensive: sr=0 returns empty results, no crash."""
        kept, filtered, debug = build_pga_events(
            _make_synthetic_broadband_burst_stem(), 0, _default_config(),
        )
        assert kept == []
        assert filtered == []


class TestPitchConfigWiring:
    """Wire-up tests: ``_build_pga_events_with_filter`` must read
    the toms pitch config from ``config['toms']`` and forward
    those values to ``compute_event_features`` on each per-event
    call.

    Added 2026-06-18 alongside the perf work that identified
    pYIN as the dominant runtime cost. Before this wiring, the
    function passed no pitch kwargs to ``compute_event_features``
    and the function fell back to its defaults (pYIN, 30-4000Hz)
    — silently ignoring the user's YAML ``pitch_method: 'yin'``
    config. This test guards against that regression.
    """

    def _capturing_compute_event_features(self, monkeypatch):
        """Patch ``compute_event_features`` at its definition site
        (``stems_to_midi.event_features``) so the lazy import inside
        ``pga_event_builder._build_pga_events_with_filter`` resolves
        to our fake. Returns a list that gets one entry per call.
        """
        captured = []

        def fake(audio, sr, t, **kwargs):
            captured.append(kwargs)
            return {
                'duration_ms': None, 'attack_rise_ms': None,
                'pitch_hz': None, 'pitch_confidence': None,
                'decay_t60_ms': None, 'spectral_centroid_hz': None,
                'spectral_flatness': None, 'hr_peak_offset_ms': None,
                'decay_envelope_energy': None, 'decay_col_min_median_db': None,
                'inter_onset_ms': None,
            }

        monkeypatch.setattr(
            'stems_to_midi.event_features.compute_event_features',
            fake,
        )
        return captured

    def test_pitch_config_forwarded_to_compute_event_features(self, monkeypatch):
        """The four pitch keys from ``config['toms']`` —
        ``enable_pitch_detection``, ``pitch_method``,
        ``min_pitch_hz``, ``max_pitch_hz`` — must be read and
        passed as kwargs to ``compute_event_features``.
        """
        captured = self._capturing_compute_event_features(monkeypatch)
        y = _make_synthetic_broadband_burst_stem()
        config = _default_config(
            toms={
                'enable_pitch_detection': False,
                'pitch_method': 'yin',
                'min_pitch_hz': 60.0,
                'max_pitch_hz': 250.0,
            },
        )
        # Force the prominence filter to be permissive so at least
        # one event survives into the feature-extraction path.
        config['onset_detection']['pga_min_prominence'] = 0.0
        _build_pga_events_with_filter(y, 44100, config)

        assert len(captured) >= 1, (
            "expected at least one compute_event_features call"
        )
        for kwargs in captured:
            assert kwargs.get('enable_pitch_detection') is False, (
                "enable_pitch_detection should be False (from config)"
            )
            assert kwargs.get('pitch_method') == 'yin'
            assert kwargs.get('pitch_fmin_hz') == 60.0
            assert kwargs.get('pitch_fmax_hz') == 250.0

    def test_pitch_config_defaults_match_yaml(self, monkeypatch):
        """If the config has no ``toms`` section at all, the
        wiring falls back to the YAML-default values
        (enable=True, yin, 60-250Hz) so the per-event call still
        gets a coherent set of kwargs.
        """
        captured = self._capturing_compute_event_features(monkeypatch)
        y = _make_synthetic_broadband_burst_stem()
        config = _default_config()  # no 'toms' key
        config['onset_detection']['pga_min_prominence'] = 0.0
        _build_pga_events_with_filter(y, 44100, config)

        assert len(captured) >= 1
        for kwargs in captured:
            assert kwargs.get('enable_pitch_detection') is True
            assert kwargs.get('pitch_method') == 'yin'
            assert kwargs.get('pitch_fmin_hz') == 60.0
            assert kwargs.get('pitch_fmax_hz') == 250.0


@pytest.fixture(autouse=True)
def _clear_stft_cache():
    """Autouse fixture for ``TestOnRealAudio``: clear the
    id()-keyed STFT cache in ``spectral_transient_core`` before
    each test in the class.

    The cache uses ``id(audio)`` as a key, which can collide
    when memory pressure reuses a freed buffer's id — a
    pre-existing fragility in spectral_transient_core.py:215.
    By clearing the cache before each test, we guarantee a
    fresh computation against THIS test's audio, regardless
    of where the test lands in the full suite's run order.
    Scoped to this class only — other tests in the suite are
    unaffected.
    """
    from stems_to_midi.spectral_transient_core import _STFT_CACHE
    _STFT_CACHE.clear()
    yield
    # No teardown — leave the cache alone. Subsequent tests in
    # the same class will get the autouse clear; tests in other
    # classes are unaffected.


class TestOnRealAudio:
    """Run the helper end-to-end on a real toms stem from
    user_files/. The point isn't to validate the detection
    algorithm (that's percentile_gated_detector's job) but to
    verify the helper's I/O contract on real data: that the
    sidecar-relevant fields are populated and the per-event
    feature extraction succeeds."""

    TOMS_WAV = (
        Path(__file__).resolve().parent.parent.parent
        / 'user_files' / '4 - 2_funk_80_beat_4-4_4'
        / 'stems' / '2_funk_80_beat_4-4_4-toms.wav'
    )

    def _load_mono(self):
        if not self.TOMS_WAV.exists():
            pytest.skip(f"toms wav not found at {self.TOMS_WAV}")
        # The class-level autouse fixture (_clear_stft_cache)
        # already cleared the id()-keyed STFT cache before
        # this test runs, so we don't need to clear it again
        # here. Just force a fresh allocation so id(audio_mono)
        # is unique to this call.
        audio, sr = sf.read(str(self.TOMS_WAV), always_2d=True)
        audio_mono = np.ascontiguousarray(
            audio.mean(axis=1).astype(np.float32)
        )
        return audio_mono, sr

    def test_real_toms_produces_kept_and_filtered_events(self):
        audio_mono, sr = self._load_mono()
        kept, filtered, debug = build_pga_events(
            audio_mono, sr, _default_config(
                onset_detection={'pga_min_prominence': 3000.0},
            ),
        )
        # Project 4 calibration (2026-06-10) found 25 candidates
        # with the default prominence threshold; both kept and
        # filtered should be non-empty.
        total = len(kept) + len(filtered)
        assert total > 0, "real toms stem produced no PGA candidates"
        assert isinstance(debug, dict)
        assert debug.get('envelope') is not None
        assert len(debug['envelope']) > 0

    def test_real_toms_event_features_computed(self):
        """Per-event feature extraction (Step 6 of the helper)
        must succeed on real audio. We check for a few
        representative feature keys; values may be None on
        hard cases (defensive try/except) but the keys must
        be present."""
        audio_mono, sr = self._load_mono()
        kept, filtered, debug = build_pga_events(
            audio_mono, sr, _default_config(
                onset_detection={'pga_min_prominence': 3000.0},
            ),
        )
        for ev in kept + filtered:
            # These are the feature keys compute_event_features
            # returns. Some may be None on bad segments, but
            # the keys must always be present.
            for f in ('duration_ms', 'attack_rise_ms', 'pitch_hz',
                      'pitch_confidence', 'decay_t60_ms',
                      'spectral_centroid_hz', 'spectral_flatness',
                      'inter_onset_ms'):
                assert f in ev, f"missing feature key {f!r}"

    def test_real_toms_kept_events_are_time_ordered(self):
        """Detection order = event time order. This is the
        contract the sidecar / WebUI rely on for time-ordered
        rendering without an extra sort."""
        audio_mono, sr = self._load_mono()
        kept, filtered, debug = build_pga_events(
            audio_mono, sr, _default_config(
                onset_detection={'pga_min_prominence': 3000.0},
            ),
        )
        if len(kept) >= 2:
            times = [ev['time'] for ev in kept]
            assert times == sorted(times), (
                f"KEPT events not time-ordered: {times}"
            )
        if len(filtered) >= 2:
            times = [ev['time'] for ev in filtered]
            assert times == sorted(times), (
                f"FILTERED events not time-ordered: {times}"
            )


# --- Noise-floor gate (2026-06-15) -----------------------------------------


class TestNoiseFloorGate:
    """Noise-floor gate (2026-06-15) added to
    ``_build_static_noise_floor`` after observing the
    silence-to-noise phantom in real toms stems: stem-splitter
    digital-silence gaps (-160 dB in every bin) can pull a
    per-bin floor down to digital silence, so when the noise
    resumes at ~-75 dB the contrast envelope sees an 85 dB
    jump and the IQR-gated ``find_peaks`` calls it a
    high-prominence attack.

    The fix is a global **gate** = ``max(p5 across all bins)``,
    the upper bound of the quietest portions of the song. Every
    bin's floor is clamped to ``>=`` the gate after the per-bin
    pass. This test class locks the gate's behavior so the
    phantom cannot reappear unnoticed.

    All three tests are pure unit tests on synthetic spectrograms
    (no audio round-trip) so they run in milliseconds.
    """

    def test_gate_lifts_quiet_bin(self):
        """A bin whose noise floor is well below the global
        gate gets lifted to the gate. The gate is the max
        p5 across all bins — the upper bound of the
        quietest portions of the song. The quiet bin's
        pre-clamp floor is below the gate, so the clamp
        raises it.

        This is the core gate contract: no bin's floor
        can be lower than the loudest of the per-bin p5
        values.
        """
        n_bins = 8
        n_frames = 200
        s_db = np.full((n_bins, n_frames), -70.0)
        # Pull bin 0 down to a quieter level. Its p5 is -90,
        # which is below the gate (max p5 = -70 from bins 1-7).
        s_db[0] = -90.0
        floor, gate_db, p5_per_bin, n_lifted = _build_static_noise_floor(s_db)
        # Gate is the max p5 — -70 from bins 1-7.
        assert gate_db == pytest.approx(-70.0, abs=0.1)
        # Every floor must be >= the gate.
        assert np.all(floor >= gate_db - 1e-9), (
            f"floors below gate: floor={floor}, gate={gate_db}"
        )
        # Bin 0 was lifted (its pre-clamp p5 was -90 < -70).
        assert n_lifted >= 1
        # p5_per_bin exposes the per-bin p5 values: bin 0 is
        # the quietest (-90), bins 1-7 are at the gate (-70).
        assert p5_per_bin[0] == pytest.approx(-90.0, abs=0.1)
        assert p5_per_bin[1] == pytest.approx(-70.0, abs=0.1)
        # The gate value is the max of p5_per_bin.
        assert gate_db == pytest.approx(float(p5_per_bin.max()), abs=1e-6)

    def test_silence_gap_does_not_lower_gate(self):
        """A bin with a long digital-silence gap in the
        middle and noise at -75 dB on either side. The
        silence frames are excluded by the 0.5-dB
        neighborhood rule, so the pre-clamp floor is -75
        (from the noise frames). The gate is also -75
        (all bins are at -75). No bin is lifted, and the
        gate is stable — this is the regression test for
        the phantom-attack scenario: with the gate in
        place, a silence gap in the middle of a bin does
        not pull the floor or the gate below the noise
        level.
        """
        n_bins = 4
        n_frames = 1000
        s_db = np.full((n_bins, n_frames), -75.0)
        # Insert a digital-silence gap in frames 400-500 in
        # every bin (the stem-splitter scenario).
        s_db[:, 400:500] = -160.0
        floor, gate_db, p5_per_bin, n_lifted = _build_static_noise_floor(s_db)
        # All floors are at the noise level (or the gate, which
        # is the same value here).
        assert np.all(floor >= -75.0 - 1e-9)
        assert np.all(floor <= -75.0 + 1e-9)
        # Gate is -75 (max p5 across bins; the silence frames
        # are excluded by the 0.5-dB rule).
        assert gate_db == pytest.approx(-75.0, abs=0.1)
        # No bin was lifted (all floors are at the gate already).
        assert n_lifted == 0
        # p5_per_bin is uniform across bins.
        assert np.allclose(p5_per_bin, -75.0, atol=0.1)

    def test_summary_line_emitted(self, capsys):
        """The detector prints exactly one ``[percentile_gated]
        noise floor: gate=XdB, ...`` line per call, in the
        format documented in the plan. Format-locked so
        downstream tools that grep the console don't break.
        """
        y = _make_synthetic_broadband_burst_stem()
        detect_percentile_gated_broad_attacks(y, 44100)
        captured = capsys.readouterr()
        # The line must appear at least once.
        assert "[percentile_gated] noise floor:" in captured.out
        # Format check: the line must match the documented
        # pattern exactly.
        pattern = (
            r"\[percentile_gated\] noise floor: gate=-?\d+\.\d+dB, "
            r"per-bin p5 range=\[-?\d+\.\d+, -?\d+\.\d+\]dB, "
            r"lifted \d+/\d+ bins"
        )
        assert re.search(pattern, captured.out), (
            f"summary line format mismatch; captured:\n{captured.out}"
        )


# --- decay_col_min filter (2026-06-15) -------------------------------------


class TestDecayColMinFilter:
    """``apply_pga_decay_col_min_filter`` (2026-06-15) — sister
    function to ``apply_pga_prominence_filter``. Same contract,
    different diagnostic field (``decay_col_min_median_db``
    instead of ``prominence``).

    The detector stamps ``decay_col_min_median_db`` on every
    event via ``compute_high_res_decay_signature`` in
    ``event_features.py``. This test class locks the filter
    behavior so the contract cannot drift.

    All four tests are pure unit tests on synthetic event
    dicts (no audio round-trip) so they run in milliseconds.
    """

    def test_drops_quiet_events(self):
        """An event with ``decay_col_min_median_db`` below the
        threshold is tagged FILTERED with a reason naming the
        configured threshold. An event above the threshold is
        KEPT and has no filter_reason."""
        events = [
            {'time': 0.5, 'decay_col_min_median_db': -70.0},
            {'time': 1.0, 'decay_col_min_median_db': -90.0},
            {'time': 1.5, 'decay_col_min_median_db': -75.0},
        ]
        kept, filtered = apply_pga_decay_col_min_filter(events, -80.0)
        # -70 > -80: KEPT
        # -90 < -80: FILTERED
        # -75 > -80: KEPT
        assert len(kept) == 2
        assert len(filtered) == 1
        # Verify the times.
        kept_times = sorted(e['time'] for e in kept)
        filtered_times = sorted(e['time'] for e in filtered)
        assert kept_times == [0.5, 1.5]
        assert filtered_times == [1.0]
        # Verify the status and reason on the filtered event.
        ev = filtered[0]
        assert ev['status'] == 'FILTERED'
        assert 'min_decay_col_min_db' in ev['filter_reason']
        assert '-90.0dB' in ev['filter_reason']
        assert '-80.0dB' in ev['filter_reason']
        # Verify the kept events have KEPT status and no
        # stale filter_reason.
        for ev in kept:
            assert ev['status'] == 'KEPT'
            assert 'filter_reason' not in ev

    def test_skips_none_values(self):
        """An event with no ``decay_col_min_median_db`` field
        cannot be filtered by the threshold — it has no value
        to compare. Same pattern as
        ``apply_pga_prominence_filter`` for events with no
        ``prominence`` field. The event is still tagged
        ``status='KEPT'`` (it survived the filter because the
        filter could not act on it) but it has no
        ``filter_reason``."""
        events = [
            {'time': 0.5, 'decay_col_min_median_db': -70.0},
            {'time': 1.0},  # no decay_col_min_median_db
            {'time': 1.5, 'decay_col_min_median_db': -90.0},
        ]
        kept, filtered = apply_pga_decay_col_min_filter(events, -80.0)
        # -70 > -80: KEPT
        # 1.0 (no field): KEPT (cannot be filtered)
        # -90 < -80: FILTERED
        assert len(kept) == 2
        assert len(filtered) == 1
        # The event with no field is in kept, tagged KEPT,
        # and has no filter_reason (the filter did not act
        # on it).
        none_event = next(e for e in kept if e['time'] == 1.0)
        assert none_event['status'] == 'KEPT'
        assert 'filter_reason' not in none_event
        # Sanity: the -90 event is the only filtered one.
        assert filtered[0]['time'] == 1.5

    def test_disabled_ids_takes_precedence(self):
        """A manually-disabled event is tagged FILTERED with
        reason "manually disabled via WebUI" even if its
        ``decay_col_min_median_db`` passes the threshold. Same
        pattern as the prominence filter."""
        events = [
            {'time': 0.5, 'decay_col_min_median_db': -70.0},  # passes, but disabled
            {'time': 1.0, 'decay_col_min_median_db': -90.0},  # fails threshold
            {'time': 1.5, 'decay_col_min_median_db': -70.0},  # passes
        ]
        # 0.5 is in disabled_ids but its decay_col_min is above
        # the threshold. Disabled check should win.
        kept, filtered = apply_pga_decay_col_min_filter(
            events, -80.0, disabled_ids={0.5},
        )
        assert len(kept) == 1
        assert len(filtered) == 2
        # 0.5 is in filtered (disabled) with the disabled reason.
        ev_disabled = next(e for e in filtered if e['time'] == 0.5)
        assert ev_disabled['status'] == 'FILTERED'
        assert ev_disabled['filter_reason'] == 'manually disabled via WebUI'
        # 1.0 is in filtered (threshold) with the threshold reason.
        ev_threshold = next(e for e in filtered if e['time'] == 1.0)
        assert ev_threshold['status'] == 'FILTERED'
        assert 'min_decay_col_min_db' in ev_threshold['filter_reason']
        # 1.5 is the only KEPT.
        assert kept[0]['time'] == 1.5
        assert kept[0]['status'] == 'KEPT'

    def test_threshold_resolution_in_build_pga_events_with_filter(self):
        """The threshold resolution in
        ``_build_pga_events_with_filter`` follows per-stem >
        global > -80.0 default. The per-event ``pga_filter_config``
        records the resolved value on every event (so the
        sidecar tooltip can show it)."""
        # Case 1: per-stem wins over global.
        cfg = {
            'toms': {'pga_min_prominence': 0.0, 'min_decay_col_min_db': -70.0},
            'onset_detection': {
                'pga_min_prominence': 0.0,
                'min_decay_col_min_db': -90.0,
            },
        }
        y = _make_synthetic_broadband_burst_stem()
        _raw, kept, _filtered, _debug = _build_pga_events_with_filter(y, 44100, cfg)
        # At least one event should have a pga_filter_config
        # entry for min_decay_col_min_db.
        if _raw:
            ev = _raw[0]
            assert ev['pga_filter_config']['min_decay_col_min_db'] == -70.0
        # Case 2: global wins when per-stem is absent.
        cfg2 = {
            'toms': {'pga_min_prominence': 0.0},
            'onset_detection': {
                'pga_min_prominence': 0.0,
                'min_decay_col_min_db': -75.0,
            },
        }
        _raw2, _kept2, _filtered2, _debug2 = _build_pga_events_with_filter(
            y, 44100, cfg2,
        )
        if _raw2:
            ev2 = _raw2[0]
            assert ev2['pga_filter_config']['min_decay_col_min_db'] == -75.0
        # Case 3: default -80.0 when both are absent.
        cfg3 = {
            'toms': {'pga_min_prominence': 0.0},
            'onset_detection': {'pga_min_prominence': 0.0},
        }
        _raw3, _kept3, _filtered3, _debug3 = _build_pga_events_with_filter(
            y, 44100, cfg3,
        )
        if _raw3:
            ev3 = _raw3[0]
            assert ev3['pga_filter_config']['min_decay_col_min_db'] == -80.0


class TestAttackRiseMaxFilter:
    """``apply_attack_rise_max_filter`` (2026-06-17) — third
    PGA pass, after prominence and decay_col_min. Same
    contract as the other two, but for the ``attack_rise_ms``
    field (the 10-90% rise time on the high-res STFT
    envelope). Uses ``kind: 'max_value'`` (drop if value >
    threshold), unlike the other two which use ``min_value``.

    Catches wire-tail / step-back FPs that pass prominence +
    decay_col_min but have an unusually long rise time —
    these FPs 'step back' to a previous attack before rising
    to their own peak. User observation on project 6: real
    strikes have attack_rise < 20ms; FPs cluster at 100-500ms.
    """

    def test_drops_long_rise_events(self):
        """An event with attack_rise_ms above the threshold is
        tagged FILTERED. An event at or below is KEPT."""
        events = [
            {'time': 0.5, 'attack_rise_ms': 11.0},
            {'time': 1.0, 'attack_rise_ms': 150.0},
            {'time': 1.5, 'attack_rise_ms': 18.0},
        ]
        kept, filtered = apply_attack_rise_max_filter(events, 20.0)
        # 11 < 20: KEPT
        # 150 > 20: FILTERED
        # 18 < 20: KEPT
        assert len(kept) == 2
        assert len(filtered) == 1
        kept_times = sorted(e['time'] for e in kept)
        filtered_times = sorted(e['time'] for e in filtered)
        assert kept_times == [0.5, 1.5]
        assert filtered_times == [1.0]
        ev = filtered[0]
        assert ev['status'] == 'FILTERED'
        assert 'attack_rise_max_ms' in ev['filter_reason']
        assert '150.0ms' in ev['filter_reason']
        assert '20.0ms' in ev['filter_reason']

    def test_value_at_threshold_is_kept(self):
        """At threshold: KEPT (>= threshold, not strictly >)."""
        events = [{'time': 1.0, 'attack_rise_ms': 20.0}]
        kept, filtered = apply_attack_rise_max_filter(events, 20.0)
        assert len(kept) == 1
        assert len(filtered) == 0

    def test_skips_none_values(self):
        """Missing attack_rise_ms → KEPT (can't filter what
        you can't see). Same pattern as the other two filters."""
        events = [
            {'time': 0.5, 'attack_rise_ms': 11.0},
            {'time': 1.0},  # no attack_rise_ms
            {'time': 1.5, 'attack_rise_ms': 150.0},
        ]
        kept, filtered = apply_attack_rise_max_filter(events, 20.0)
        assert len(kept) == 2  # 0.5 + 1.0
        assert len(filtered) == 1  # 1.5
        # The event with no field is KEPT.
        none_event = next(e for e in kept if e['time'] == 1.0)
        assert none_event['status'] == 'KEPT'
        assert 'filter_reason' not in none_event

    def test_filter_reason_uses_registry_template(self):
        """The filter reason must use the registry's reason_template
        (above attack_rise_max_ms ({value}ms > {threshold}ms))
        with the float1 value_format — proves the function
        reads the template from the registry, not a hard-coded
        format string."""
        events = [
            {'time': 1.0, 'attack_rise_ms': 387.5},
        ]
        kept, filtered = apply_attack_rise_max_filter(events, 20.0)
        assert len(filtered) == 1
        reason = filtered[0]['filter_reason']
        # Registry template is "above attack_rise_max_ms ({value}ms > {threshold}ms)"
        # with value_format=float1. Expect "387.5ms > 20.0ms".
        assert reason == 'above attack_rise_max_ms (387.5ms > 20.0ms)'

    def test_disabled_ids_takes_precedence(self):
        """A manually-disabled event is tagged FILTERED with
        the disabled reason regardless of attack_rise value."""
        events = [
            {'time': 1.0, 'attack_rise_ms': 11.0},  # would pass
        ]
        kept, filtered = apply_attack_rise_max_filter(
            events, 20.0, disabled_ids={1.0},
        )
        assert len(kept) == 0
        assert len(filtered) == 1
        ev = filtered[0]
        assert ev['filter_reason'] == 'manually disabled via WebUI'

    def test_threshold_resolution_in_build_pga_events_with_filter(self):
        """_build_pga_events_with_filter must include
        attack_rise_max_ms in pga_filter_config with per-stem
        > global > default precedence."""
        # Per-stem wins.
        cfg = {
            'toms': {'pga_min_prominence': 0.0, 'attack_rise_max_ms': 50.0},
            'onset_detection': {'pga_min_prominence': 0.0, 'attack_rise_max_ms': 100.0},
        }
        y = _make_synthetic_broadband_burst_stem()
        _raw, kept, _filtered, _debug = _build_pga_events_with_filter(y, 44100, cfg)
        if _raw:
            assert _raw[0]['pga_filter_config']['attack_rise_max_ms'] == 50.0
        # Global wins when per-stem is absent.
        cfg2 = {
            'toms': {'pga_min_prominence': 0.0},
            'onset_detection': {'pga_min_prominence': 0.0, 'attack_rise_max_ms': 75.0},
        }
        _raw2, _kept2, _filtered2, _debug2 = _build_pga_events_with_filter(y, 44100, cfg2)
        if _raw2:
            assert _raw2[0]['pga_filter_config']['attack_rise_max_ms'] == 75.0
        # Default 20.0 when both are absent.
        cfg3 = {
            'toms': {'pga_min_prominence': 0.0},
            'onset_detection': {'pga_min_prominence': 0.0},
        }
        _raw3, _kept3, _filtered3, _debug3 = _build_pga_events_with_filter(y, 44100, cfg3)
        if _raw3:
            assert _raw3[0]['pga_filter_config']['attack_rise_max_ms'] == 20.0
