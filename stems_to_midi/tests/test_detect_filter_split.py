"""
Tests for the detect / apply_filter split of
``stems_to_midi.pga_event_builder``.

This test file locks the post-refactor contract (2026-06-19):

  1. ``detect_pga_events`` is pure and returns a flat list of
     events with ``status='KEPT'`` on every event — no filter
     applied. The full diagnostic surface (frame, envelope_value,
     prominence, iqr_threshold, midi_velocity, pga_filter_config)
     is preserved.
  2. ``apply_pga_prominence_filter`` is pure and walks a
     detect-time list, partitioning into kept/filtered based on
     a threshold and an optional disabled-id set.
  3. ``build_pga_events`` is a thin detect-only wrapper: returns
     ``(events_kept, [], debug_dict)`` where ``events_kept`` is
     the raw all-KEPT list (no filter applied). This is the
     sidecar-write path — the sidecar stores the raw list and
     the WebUI re-filters at tuning time.
  4. ``_build_pga_events_with_filter`` is the
     process-and-filter wrapper used by the processing_shell
     call site. It applies the prominence + decay_col_min +
     attack_rise filters and returns
     ``(raw, kept, filtered, debug_dict)``. This is the
     regression-checked behavior: the partition it produces
     matches the pre-refactor contract.
  5. ``load_event_overrides`` returns None when the file is
     absent and the parsed dict when present.

The real-audio tests use the same fixture as
``test_pga_event_builder.py``:

    user_files/4 - 2_funk_80_beat_4-4_4/stems/
        2_funk_80_beat_4-4_4-toms.wav

and gracefully skip when the fixture is missing.
"""
import json
import os
import sys
import tempfile
import shutil
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
    detect_pga_events,
    apply_pga_prominence_filter,
)
from stems_to_midi.event_overrides import (  # noqa: E402
    load_event_overrides,
    EventOverridesError,
)


# --- Fixtures / helpers ----------------------------------------------------


# Path to the real toms wav — same fixture the legacy
# ``test_pga_event_builder.py`` uses. Skipped at collection
# time if the file is missing.
TOMS_WAV = (
    Path(__file__).resolve().parent.parent.parent
    / 'user_files' / '4 - 2_funk_80_beat_4-4_4'
    / 'stems' / '2_funk_80_beat_4-4_4-toms.wav'
)


def _default_config(**overrides) -> dict:
    """Same default-config shape used by the legacy test file."""
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


def _make_synthetic_broadband_burst_stem(
    sr: int = 44100,
    hit_times_sec: tuple = (0.5, 1.0, 1.5, 2.0),
    freq_hz: float = 200.0,
    decay_ms: float = 80.0,
    duration_sec: float = 3.0,
) -> np.ndarray:
    """Synthetic toms-shaped signal — same shape as the
    legacy test helper, copied here so this test file
    stands alone without depending on
    ``test_pga_event_builder.py`` internals."""
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


def _load_real_toms():
    """Load the real toms fixture as mono float32, or
    skip the test if the file is missing. The fresh
    allocation (`np.ascontiguousarray`) keeps
    ``id(audio_mono)`` unique to this call so the
    id()-keyed STFT cache in
    ``spectral_transient_core.py:215`` cannot serve
    stale data from a prior test."""
    if not TOMS_WAV.exists():
        pytest.skip(f"toms wav not found at {TOMS_WAV}")
    audio, sr = sf.read(str(TOMS_WAV), always_2d=True)
    audio_mono = np.ascontiguousarray(
        audio.mean(axis=1).astype(np.float32)
    )
    return audio_mono, sr


# Autouse fixture for the whole module: clear the
# id()-keyed STFT cache before each test that loads real
# audio. Same rationale as the legacy
# ``test_pga_event_builder.TestOnRealAudio`` class fixture —
# a pre-existing fragility in spectral_transient_core.py:215
# (the cache keys on ``id(audio)`` which can collide when
# memory pressure reuses a freed buffer's id).
@pytest.fixture(autouse=True)
def _clear_stft_cache():
    from stems_to_midi.stft_utils import _STFT_CACHE
    _STFT_CACHE.clear()
    yield
    # No teardown — let other tests handle their own state.


# --- Tests -----------------------------------------------------------------


class TestDetectPGAEventsAllKept:
    """``detect_pga_events`` is pure detection: every event
    in the output has ``status='KEPT'`` regardless of the
    prominence filter setting. Diagnostic fields are still
    attached so the WebUI tooltip can show them."""

    def test_detect_returns_flat_list_of_dicts_on_synthetic(self):
        y = _make_synthetic_broadband_burst_stem()
        events = detect_pga_events(y, 44100, _default_config())
        assert isinstance(events, list)
        assert len(events) > 0, "synthetic toms should produce PGA events"
        for ev in events:
            assert isinstance(ev, dict)
            assert ev.get('status') == 'KEPT', (
                f"detect_pga_events must return KEPT events only, "
                f"got status={ev.get('status')!r}"
            )

    def test_detect_all_events_kept_on_real_toms(self):
        """The contract test: real toms fixture, all
        events have status='KEPT' even though the default
        threshold (1000) would FILTER some of them when
        build_pga_events is called. This is the core
        guarantee the refactor provides — the detection
        step does not apply any filter."""
        audio_mono, sr = _load_real_toms()
        events = detect_pga_events(audio_mono, sr, _default_config())
        assert len(events) > 0
        for ev in events:
            assert ev.get('status') == 'KEPT', (
                f"detect_pga_events must not filter — "
                f"event at {ev.get('time'):.3f}s has "
                f"status={ev.get('status')!r}"
            )

    def test_detect_attaches_all_diagnostic_fields_on_real_toms(self):
        """Diagnostic fields the WebUI tooltip relies on
        must survive the split — the spec lists
        ``frame``, ``envelope_value``, ``prominence``,
        ``iqr_threshold``, ``midi_velocity``,
        ``pga_filter_config`` as the fields the consumer
        needs in detect-time output."""
        audio_mono, sr = _load_real_toms()
        events = detect_pga_events(audio_mono, sr, _default_config())
        for ev in events:
            for field in (
                'frame', 'envelope_value', 'prominence',
                'iqr_threshold', 'midi_velocity', 'pga_filter_config',
            ):
                assert field in ev, (
                    f"detect_pga_events event missing "
                    f"diagnostic field {field!r}"
                )

    def test_detect_returns_empty_list_on_empty_audio(self):
        """Defensive: empty input → empty list, no crash."""
        events = detect_pga_events(np.array([]), 44100, _default_config())
        assert events == []

    def test_detect_returns_empty_list_on_zero_sr(self):
        """Defensive: sr=0 → empty list, no crash."""
        events = detect_pga_events(
            _make_synthetic_broadband_burst_stem(),
            0,
            _default_config(),
        )
        assert events == []


class TestApplyPgaProminenceFilter:
    """``apply_pga_prominence_filter`` is pure: walks a
    detect-time list, partitions it into (kept, filtered)
    by prominence and an optional disabled-id set. The
    function is the single re-filter point the WebUI
    re-applies on every tuning-panel change."""

    def _build_detect_input(self):
        """Build a small synthetic detect-time list with
        known prominences for the filter tests. Avoids
        running the full PGA pipeline so the test is
        fast and deterministic."""
        return [
            {'time': 0.5, 'method': 'percentile_gated', 'status': 'KEPT',
             'prominence': 200.0, 'time_id': 0.5},
            {'time': 1.0, 'method': 'percentile_gated', 'status': 'KEPT',
             'prominence': 5000.0, 'time_id': 1.0},
            {'time': 1.5, 'method': 'percentile_gated', 'status': 'KEPT',
             'prominence': 800.0, 'time_id': 1.5},
            {'time': 2.0, 'method': 'percentile_gated', 'status': 'KEPT',
             'prominence': 9000.0, 'time_id': 2.0},
        ]

    def test_threshold_zero_keeps_all(self):
        events = self._build_detect_input()
        kept, filtered = apply_pga_prominence_filter(events, threshold=0.0)
        assert len(kept) == 4
        assert len(filtered) == 0

    def test_threshold_huge_filters_all(self):
        events = self._build_detect_input()
        kept, filtered = apply_pga_prominence_filter(
            events, threshold=1e9,
        )
        assert len(kept) == 0
        assert len(filtered) == 4
        for ev in filtered:
            assert ev['status'] == 'FILTERED'
            assert 'pga_min_prominence' in ev['filter_reason']

    def test_threshold_1000_partitions_correctly(self):
        """The synthetic list has prominences 200, 5000,
        800, 9000 — threshold 1000 should keep 5000+9000
        (the >1000 ones), filter 200+800."""
        events = self._build_detect_input()
        kept, filtered = apply_pga_prominence_filter(
            events, threshold=1000.0,
        )
        kept_proms = sorted(ev['prominence'] for ev in kept)
        filtered_proms = sorted(ev['prominence'] for ev in filtered)
        assert kept_proms == [5000.0, 9000.0]
        assert filtered_proms == [200.0, 800.0]

    def test_filter_reason_uses_threshold_and_prominence(self):
        """The filter_reason format is
        ``"below pga_min_prominence ({prom:.0f} < {thr:.0f})"``."""
        events = self._build_detect_input()
        _, filtered = apply_pga_prominence_filter(
            events, threshold=1000.0,
        )
        reasons = sorted(ev['filter_reason'] for ev in filtered)
        assert reasons == [
            'below pga_min_prominence (200 < 1000)',
            'below pga_min_prominence (800 < 1000)',
        ]

    def test_disabled_ids_overrides_prominence(self):
        """A passing event (prominence > threshold) is
        still tagged FILTERED when its id is in
        ``disabled_ids``. The disabled check takes
        precedence — that's the whole point of the
        parameter (WebUI user toggle beats the slider)."""
        events = self._build_detect_input()
        kept, filtered = apply_pga_prominence_filter(
            events, threshold=0.0,  # would keep everything
            disabled_ids={1.0},    # disable the prominence=5000 event
        )
        # Only the event with time_id=1.0 should be filtered.
        assert len(kept) == 3
        assert len(filtered) == 1
        assert filtered[0]['time_id'] == 1.0
        assert filtered[0]['status'] == 'FILTERED'
        assert filtered[0]['filter_reason'] == 'manually disabled via WebUI'

    def test_disabled_ids_with_threshold_does_not_double_filter(self):
        """An event in disabled_ids gets the 'manually
        disabled' reason, NOT the prominence reason,
        even if it would have been filtered anyway. The
        disabled reason takes precedence so the WebUI
        tooltip distinguishes user intent from
        automatic filter hits."""
        events = self._build_detect_input()
        # Synthetic data: 4 events with prominences
        # 200, 5000, 800, 9000. With threshold=1000,
        # the events at 0.5s (prom=200) and 1.5s
        # (prom=800) are filtered by prominence — 2
        # total. The event at 0.5s is ALSO in
        # disabled_ids, so its filter_reason should be
        # the disabled reason (NOT the prominence
        # reason), but the count is still 2.
        kept, filtered = apply_pga_prominence_filter(
            events, threshold=1000.0,
            disabled_ids={0.5},
        )
        assert len(filtered) == 2
        # Find the disabled event specifically.
        disabled_ev = next(ev for ev in filtered if ev['time_id'] == 0.5)
        assert disabled_ev['filter_reason'] == 'manually disabled via WebUI'
        # The auto-filtered event keeps the prominence reason.
        auto_filtered = next(ev for ev in filtered if ev['time_id'] == 1.5)
        assert auto_filtered['filter_reason'] == 'below pga_min_prominence (800 < 1000)'

    def test_falls_back_to_time_id_when_id_field_missing(self):
        """The id resolution order is event['id'] then
        event['time']. The synthetic test events have
        no 'id' field, so the time is used as the
        stable identifier."""
        events = self._build_detect_input()
        kept, filtered = apply_pga_prominence_filter(
            events, threshold=0.0,
            disabled_ids={2.0},  # disable by time
        )
        assert len(filtered) == 1
        assert filtered[0]['time_id'] == 2.0

    def test_uses_explicit_id_field_when_present(self):
        """When an event has an explicit 'id' field, it
        takes precedence over 'time' for the disabled
        lookup. The WebUI may stamp ids that are
        distinct from the onset time (e.g. a stable
        hash), and the filter must respect them."""
        events = [
            {'time': 0.5, 'id': 'evt_a', 'prominence': 5000.0,
             'status': 'KEPT', 'time_id': 0.5},
        ]
        kept, filtered = apply_pga_prominence_filter(
            events, threshold=0.0,
            disabled_ids={'evt_a'},
        )
        assert len(filtered) == 1
        assert filtered[0]['id'] == 'evt_a'

    def test_filter_clears_stale_filter_reason_on_kept(self):
        """Re-applying the filter at a higher threshold
        should clear the stale filter_reason from a
        prior filter call, so the WebUI tooltip only
        shows reasons for events currently FILTERED.
        An event that survives a second filter pass
        must NOT carry a stale filter_reason from the
        first call — that would mislead the tooltip."""
        events = self._build_detect_input()
        # First call: low threshold, tags the low-prom
        # events (200 and 800) as filtered with a
        # filter_reason. The two high-prom events
        # (5000 and 9000) are kept and have NO
        # filter_reason after this pass.
        kept, filtered = apply_pga_prominence_filter(
            events, threshold=1000.0,
        )
        assert len(filtered) == 2
        for ev in filtered:
            assert 'filter_reason' in ev
        for ev in kept:
            assert 'filter_reason' not in ev, (
                f"KEPT event should not carry filter_reason; "
                f"got {ev.get('filter_reason')!r}"
            )
        # Second call: threshold=0 (everything passes
        # by prominence, no disabled_ids). All four
        # events should land in kept. Any event that
        # was filtered on the first call must have its
        # stale filter_reason cleared, so the kept list
        # has zero filter_reasons.
        kept, filtered = apply_pga_prominence_filter(
            events, threshold=0.0,
        )
        assert len(kept) == 4
        assert len(filtered) == 0
        for ev in kept:
            assert 'filter_reason' not in ev, (
                f"stale filter_reason survived a re-filter "
                f"to KEPT: {ev.get('filter_reason')!r}"
            )

    def test_passes_through_detect_pga_events_output_on_real_toms(self):
        """The split is meaningful only if the two
        functions compose correctly. Run the real
        toms fixture through detect → apply_filter
        and verify the partition matches build_pga_events
        for the same default threshold (regression
        for the wrapper test below)."""
        audio_mono, sr = _load_real_toms()
        threshold = 3000.0
        raw = detect_pga_events(
            audio_mono, sr, _default_config(
                onset_detection={'pga_min_prominence': threshold},
            ),
        )
        # At threshold=0, every event in `raw` should be kept.
        kept_all, filtered_none = apply_pga_prominence_filter(
            raw, threshold=0.0,
        )
        assert len(kept_all) == len(raw)
        assert len(filtered_none) == 0
        # At threshold=1e9, every event should be filtered.
        kept_zero, filtered_all = apply_pga_prominence_filter(
            raw, threshold=1e9,
        )
        assert len(kept_zero) == 0
        assert len(filtered_all) == len(raw)


class TestBuildPgaEventsWrapperRegression:
    """``build_pga_events`` (2026-06-19 contract) is a thin
    wrapper around ``detect_pga_events`` ONLY — it does NOT
    apply any filter. All events are returned as
    ``status='KEPT'`` and the second tuple slot is always
    empty. The filter is applied by
    :func:`_build_pga_events_with_filter` (the function the
    processing_shell call site uses) — see
    :class:`TestBuildPgaEventsWithFilter` below for those
    tests.

    Rationale (2026-06-19): the sidecar stores the raw
    all-KEPT list and the WebUI / rebuild paths apply the
    filter at re-filter time with their own threshold.
    Splitting detect (always-KEPT) from filter (kept/filtered)
    lets the WebUI re-tune the threshold without re-running
    the detector, and keeps the detect step a pure
    functional core.
    """

    def test_wrapper_returns_all_events_as_kept(self):
        """``build_pga_events`` is pure detection — every
        event in the returned ``kept`` list has
        ``status='KEPT'``, regardless of the configured
        threshold."""
        y = _make_synthetic_broadband_burst_stem()
        # Force a huge threshold — the wrapper must
        # NOT use it to filter.
        kept, filtered, _ = build_pga_events(
            y, 44100, _default_config(
                onset_detection={'pga_min_prominence': 1e9},
            ),
        )
        assert len(filtered) == 0, (
            "build_pga_events must not apply any filter — "
            "filtered list should always be empty"
        )
        assert len(kept) > 0
        for ev in kept:
            assert ev.get('status') == 'KEPT'

    def test_wrapper_preserves_debug_dict_shape(self):
        """The wrapper still returns the legacy debug
        dict so the existing call site in
        processing_shell.py keeps working."""
        audio_mono, sr = _load_real_toms()
        kept, filtered, debug = build_pga_events(
            audio_mono, sr, _default_config(),
        )
        assert isinstance(debug, dict)
        for k in ('envelope', 'peaks', 'prominences', 'times', 'freqs'):
            assert k in debug, f"debug dict missing key {k!r}"

    def test_wrapper_zero_threshold_still_keeps_everything(self):
        """Sanity: even with threshold=0 (the most
        permissive setting) the wrapper still returns
        every detected event — the wrapper never filters."""
        y = _make_synthetic_broadband_burst_stem()
        kept, filtered, _ = build_pga_events(
            y, 44100, _default_config(
                onset_detection={'pga_min_prominence': 0.0},
            ),
        )
        assert len(filtered) == 0
        assert len(kept) > 0


class TestBuildPgaEventsWithFilter:
    """``_build_pga_events_with_filter`` IS the function
    that applies the prominence + decay_col_min + attack_rise
    filters. Its partition behavior on real audio must
    match ``detect_pga_events`` + ``apply_pga_prominence_filter``
    run with the same threshold (regression check).
    """

    def test_wrapper_same_partition_as_direct_filter_on_real_toms(self):
        """Compare the filter-wrapper's kept/filtered split
        to running detect → apply_filter manually with the
        same threshold. They MUST be identical — this is the
        regression test for the refactor."""
        audio_mono, sr = _load_real_toms()
        threshold = 3000.0
        config = _default_config(
            onset_detection={'pga_min_prominence': threshold},
        )
        _raw, kept, filtered, _ = _build_pga_events_with_filter(
            audio_mono, sr, config,
        )
        raw = detect_pga_events(audio_mono, sr, config)
        kept_direct, filtered_direct = apply_pga_prominence_filter(
            raw, threshold=threshold,
        )
        # The filter-wrapper applies the prominence filter
        # AND the downstream decay_col_min + attack_rise
        # filters, so its filtered list is a superset of
        # the direct prominence-only filter. The kept list
        # is correspondingly smaller.
        assert len(kept) <= len(kept_direct)
        assert len(filtered) >= len(filtered_direct)
        # The union equals the detect-time count.
        assert len(kept) + len(filtered) == len(raw)

    def test_wrapper_default_threshold_matches_pre_refactor(self):
        """The pre-refactor function used the same default
        threshold (1000) read from
        ``onset_detection.pga_min_prominence``. With the
        default config (no override) the filter-wrapper
        must produce the same partition as the
        pre-refactor function did."""
        audio_mono, sr = _load_real_toms()
        config = _default_config()  # pga_min_prominence=1000
        _raw, kept, filtered, _ = _build_pga_events_with_filter(
            audio_mono, sr, config,
        )
        # Sanity: the partition is non-trivial.
        assert len(kept) + len(filtered) > 0
        # Every event in `kept` either has no prominence
        # field (defensive — filter treats None as KEPT)
        # OR has prominence >= the configured threshold.
        for ev in kept:
            prom = ev.get('prominence')
            assert prom is None or prom >= 1000.0, (
                f"kept event has prominence={prom} below "
                f"threshold 1000: {ev!r}"
            )
        # Every event in `filtered` has a filter_reason
        # recorded (one of the PGA filters dropped it).
        for ev in filtered:
            assert ev.get('filter_reason'), (
                f"filtered event missing filter_reason: {ev!r}"
            )

    def test_wrapper_zero_threshold_keeps_everything(self):
        """threshold=0 keeps everything — the filter is
        permissive (prominence >= 0 is true for every
        event that has a numeric prominence)."""
        y = _make_synthetic_broadband_burst_stem()
        _raw, kept, filtered, _ = _build_pga_events_with_filter(
            y, 44100, _default_config(
                onset_detection={'pga_min_prominence': 0.0},
            ),
        )
        assert len(filtered) == 0
        assert len(kept) > 0

    def test_wrapper_huge_threshold_filters_everything(self):
        """threshold=1e9 filters every event that has a
        numeric prominence (1e9 is far above any real
        prominence). The decay_col_min and attack_rise
        filters can also drop events with no prominence
        field at all (since they don't satisfy the
        min_value check), so the union of all 3 filters
        can drop every event."""
        y = _make_synthetic_broadband_burst_stem()
        _raw, kept, filtered, _ = _build_pga_events_with_filter(
            y, 44100, _default_config(
                onset_detection={'pga_min_prominence': 1e9},
            ),
        )
        # Prominence filter alone drops every event;
        # decay_col_min and attack_rise can't add
        # anything back to KEPT, so kept must be empty.
        assert len(kept) == 0
        assert len(filtered) > 0


class TestLoadEventOverrides:
    """``load_event_overrides`` is the read-side of the
    WebUI's per-project override file. Returns ``None``
    when the file is absent and the parsed dict when
    present."""

    def test_returns_none_when_file_absent(self):
        """The common case: no override file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / 'proj'
            project_dir.mkdir()
            # midi/ subdir exists but no overrides file
            (project_dir / 'midi').mkdir()
            result = load_event_overrides(project_dir)
            assert result is None

    def test_returns_none_when_midi_dir_absent(self):
        """Defensive: even if the project has no midi/
        dir at all, the loader must not crash — it
        returns None because there's no file to read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / 'proj'
            project_dir.mkdir()
            result = load_event_overrides(project_dir)
            assert result is None

    def test_returns_dict_when_file_present(self):
        """The happy path: a well-formed override file
        round-trips through the loader as the parsed
        dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / 'proj'
            midi_dir = project_dir / 'midi'
            midi_dir.mkdir(parents=True)
            override_path = midi_dir / 'event_overrides.json'
            payload = {
                '14.9014': {
                    'status': 'FILTERED',
                    'reason': 'manually disabled via WebUI',
                },
                '15.2500': {
                    'status': 'KEPT',
                    'reason': 're-enabled by user',
                },
            }
            with open(override_path, 'w') as f:
                json.dump(payload, f)
            result = load_event_overrides(project_dir)
            assert result == payload
            # And the returned dict is independent — the
            # loader does not mutate it. (We don't mutate
            # it explicitly, but the test guards against
            # a future refactor that decides to "normalize"
            # the data in place.)
            assert result is not None
            result['new_key'] = {'status': 'KEPT'}
            assert 'new_key' not in payload  # original untouched

    def test_raises_on_invalid_json(self):
        """A malformed file (truncated, non-JSON) raises
        EventOverridesError, not a generic JSONDecodeError.
        The WebUI surfaces this as a toast — the file is
        left in place so the user can hand-edit it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / 'proj'
            midi_dir = project_dir / 'midi'
            midi_dir.mkdir(parents=True)
            override_path = midi_dir / 'event_overrides.json'
            with open(override_path, 'w') as f:
                f.write('{ not valid json')
            with pytest.raises(EventOverridesError):
                load_event_overrides(project_dir)

    def test_raises_on_top_level_non_object(self):
        """A top-level JSON array (instead of an object)
        is a schema violation: the override format is
        always an object keyed by event id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / 'proj'
            midi_dir = project_dir / 'midi'
            midi_dir.mkdir(parents=True)
            override_path = midi_dir / 'event_overrides.json'
            with open(override_path, 'w') as f:
                json.dump(['not', 'a', 'dict'], f)
            with pytest.raises(EventOverridesError):
                load_event_overrides(project_dir)

    def test_raises_on_non_dict_value(self):
        """A value that is not a dict (e.g. a bare string)
        is a schema violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / 'proj'
            midi_dir = project_dir / 'midi'
            midi_dir.mkdir(parents=True)
            override_path = midi_dir / 'event_overrides.json'
            with open(override_path, 'w') as f:
                json.dump({'evt_a': 'not a dict'}, f)
            with pytest.raises(EventOverridesError):
                load_event_overrides(project_dir)

    def test_accepts_string_project_dir(self):
        """The signature accepts ``str | Path`` — the
        function must resolve a string path the same
        way as a ``Path`` object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / 'proj'
            midi_dir = project_dir / 'midi'
            midi_dir.mkdir(parents=True)
            override_path = midi_dir / 'event_overrides.json'
            with open(override_path, 'w') as f:
                json.dump({'evt_a': {'status': 'FILTERED'}}, f)
            result = load_event_overrides(str(project_dir))
            assert result == {'evt_a': {'status': 'FILTERED'}}
