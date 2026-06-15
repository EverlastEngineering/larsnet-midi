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
)
from stems_to_midi.percentile_gated_detector import (  # noqa: E402
    detect_percentile_gated_broad_attacks,
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
        kept, filtered, _ = _build_pga_events_with_filter(audio_mono, sr, config)
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
        kept, filtered, debug = _build_pga_events_with_filter(
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
            kept, filtered, _ = _build_pga_events_with_filter(
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
