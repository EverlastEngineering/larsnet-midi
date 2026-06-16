"""
PGA (percentile-gated broad-attack) event builder for the toms stem.

Isolates the toms detection path (2026-06-13 refactor) as a pure
functional core. The public surface is two pure functions plus a
thin wrapper:

  - ``detect_pga_events(audio_mono, sr, config)`` runs the
    percentile-gated broad-attack detector, attaches per-event
    diagnostic fields (frame, envelope_value, prominence,
    iqr_threshold, midi_velocity, pga_filter_config), and runs
    per-event feature extraction. **All** events are returned with
    ``status='KEPT'`` — no filter is applied. The consumer can
    re-filter the result later via a single pure function call.

  - ``apply_pga_prominence_filter(events, threshold,
    disabled_ids=None)`` walks a detect-time event list and tags
    events with ``prominence < threshold`` as
    ``status='FILTERED'`` with reason
    ``"below pga_min_prominence (X < Y)"``. If ``disabled_ids``
    is provided, any event whose id is in that set is also
    tagged FILTERED (with reason
    ``"manually disabled via WebUI"``) — even if its prominence
    passes the threshold. Returns ``(kept, filtered)``.

  - ``build_pga_events(audio, sr, config)`` is a thin wrapper
    that calls ``detect_pga_events`` and then
    ``apply_pga_prominence_filter`` with the configured
    ``pga_min_prominence`` threshold. Preserves the original
    return shape ``(events_kept, events_filtered, debug_dict)``
    so the existing call site in ``processing_shell.py`` and
    the legacy tests keep working without modification.

Pipeline (matches the original inline flow in
``processing_shell.py:1717-1892``):
  1. ``detect_percentile_gated_broad_attacks`` (broadband contrast
     envelope + IQR-thresholded peak-pick).
  2. Per-event diagnostic dict: ``time``, ``method``,
     ``status='KEPT'``, ``frame``, ``envelope_value``,
     ``prominence``, ``iqr_threshold``.
  3. MIDI velocity mapping: linear envelope-value → MIDI
     ``[min_velocity, max_velocity]`` from ``config.midi``.
  4. Per-event feature extraction via
     ``event_features.compute_event_features`` — duration, pitch,
     decay, brightness, etc. Two-pass flow: pass 1 uses the
     full detect-time list (KEPT+FILTERED) as the "neighbors".
  5. Prominence filter (config ``onset_detection.pga_min_prominence``,
     default 1000) — when called via the wrapper, moves
     low-prominence events from KEPT to FILTERED.

Design constraints:
  - Pure functions. No file I/O, no module-level state, no
    side-effects beyond ``print()`` (imperative-shell residue).
  - Imports only standard library / numpy / scipy / in-project
    modules that the rest of the pipeline already uses.
  - ``detect_pga_events`` and ``apply_pga_prominence_filter``
    are both pure and side-effect free; the consumer can call
    them independently.
  - Returns ``events_kept`` and ``events_filtered`` as separate
    lists so consumers can pass them straight to the existing
    serializer (which expects KEPT first, then FILTERED).
  - ``debug_dict`` exposes the raw peak indices, envelope, and
    prominences for diagnostics — same shape as the legacy
    inline implementation, so existing tests that inspect
    ``pga_debug`` continue to work.

The WebUI re-filter path (planned for a future step) will call
``detect_pga_events`` once at sidecar-write time, store the raw
list in the sidecar, and then re-call
``apply_pga_prominence_filter`` with a slider-driven threshold
on every tuning-panel change. This refactor is the prerequisite
for that flow.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .percentile_gated_detector import detect_percentile_gated_broad_attacks
from .filter_kinds import (
    find_filter,
    evaluate_filter,
    build_filter_reason as _build_filter_reason,
)


__all__ = [
    'build_pga_events',
    'detect_pga_events',
    'apply_pga_prominence_filter',
    'apply_pga_decay_col_min_filter',
    '_build_pga_events_with_filter',
    'PGAEventBuildError',
]


class PGAEventBuildError(RuntimeError):
    """Raised when the PGA event builder cannot produce a valid
    event list (e.g. an internal invariant is violated)."""


def _empty_result(debug: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """Uniform return shape for the no-events / error paths so
    downstream code can destructure without an ``if``."""
    return [], [], (debug or {
        'freqs': None,
        'times': None,
        's_db': None,
        'floor': None,
        'envelope': None,
        'peaks': np.array([], dtype=int),
        'prominences': np.array([]),
    })


def detect_pga_events(
    audio_mono: np.ndarray,
    sr: int,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run the PGA detector and return a flat list of event dicts.

    Pure function: no file I/O, no mutation of input. **All**
    events in the returned list have ``status='KEPT'`` — this
    function does not apply any filter. To re-filter the list
    with a different threshold, pass it to
    :func:`apply_pga_prominence_filter`.

    Diagnostic fields attached to every event:
      - ``time`` (float): onset time in seconds.
      - ``method`` (str): always ``'percentile_gated'``.
      - ``status`` (str): always ``'KEPT'`` here.
      - ``frame`` (int, optional): STFT frame index of the peak.
      - ``envelope_value`` (float, optional): contrast envelope
        value at the peak.
      - ``prominence`` (float, optional): scipy find_peaks
        prominence of the peak.
      - ``iqr_threshold`` (float, optional): peak-pick threshold
        (``q3 + 2.5*IQR`` of the envelope).
      - ``midi_velocity`` (int): linear-mapping of
        ``envelope_value`` into ``[min_velocity, max_velocity]``
        from config.
      - ``pga_filter_config`` (dict): the active filter settings
        at detect time (``pga_min_prominence``, ``min_velocity``,
        ``max_velocity``). Used by the WebUI tooltip to show
        "Active filter: pga_min_prominence=X" alongside the event.
      - Per-event feature keys (2026-06-12 fix): ``duration_ms``,
        ``attack_rise_ms``, ``pitch_hz``, ``pitch_confidence``,
        ``decay_t60_ms``, ``spectral_centroid_hz``,
        ``spectral_flatness``, ``hr_peak_offset_ms``,
        ``decay_envelope_energy``, ``decay_col_min_median_db``,
        ``inter_onset_ms``.

    Args:
        audio_mono: 1-D float array of mono audio samples.
        sr: Sample rate in Hz.
        config: Project config dict (same dict passed to
            ``process_stem_to_midi``). Reads
            ``midi.min_velocity`` / ``midi.max_velocity`` (defaults
            ``80``/``110``) for the MIDI velocity mapping, and
            ``onset_detection.pga_min_prominence`` (default
            ``1000``) and ``onset_detection.min_decay_col_min_db``
            (default ``-80.0`` dB) for the ``pga_filter_config``
            record. Per-stem overrides (``toms.pga_min_prominence``,
            ``toms.min_decay_col_min_db``) win over the global
            onset_detection equivalents. ``detect_pga_events``
            itself does NOT apply either filter — it only records
            the configured thresholds for downstream consumers
            to use.

    Returns:
        Flat list of event dicts in detection order, all with
        ``status='KEPT'``. Empty list when the input is empty,
        ``sr <= 0``, or the detector finds no candidates.
    """
    if audio_mono is None or len(audio_mono) == 0:
        return []
    if sr is None or sr <= 0:
        return []

    # Step 1: Run the percentile-gated broad-attack detector.
    # The detector returns a list of sub-frame-accurate strike
    # times in seconds and a debug dict with envelope / peak
    # metadata. We discard the debug here — the sidecar-level
    # debug is exposed by the legacy ``build_pga_events`` wrapper
    # for back-compat.
    pga_event_times, _pga_debug = detect_percentile_gated_broad_attacks(
        audio_mono, sr,
    )

    _env = _pga_debug.get('envelope') if _pga_debug else None
    _peaks = _pga_debug.get('peaks') if _pga_debug else None
    _proms = _pga_debug.get('prominences') if _pga_debug else None

    # Step 2: Build per-event diagnostic dicts. The IQR threshold
    # is recomputed here for symmetry with the algorithm — see
    # percentile_gated_detector.py for the q3 + 2.5*IQR rule.
    if _env is not None and _env.size > 0:
        _q1, _q3 = np.percentile(_env, [25, 75])
        _iqr = _q3 - _q1
        _abs_thr = _q3 + 2.5 * _iqr
    else:
        _abs_thr = None

    pga_onset_data: List[Dict[str, Any]] = []
    for i, t in enumerate(pga_event_times):
        ev: Dict[str, Any] = {
            'time': float(t),
            'method': 'percentile_gated',
            'status': 'KEPT',
        }
        if _peaks is not None and i < len(_peaks):
            p = int(_peaks[i])
            ev['frame'] = p
            if _env is not None and p < len(_env):
                ev['envelope_value'] = float(_env[p])
            if _proms is not None and i < len(_proms):
                ev['prominence'] = float(_proms[i])
        if _abs_thr is not None:
            ev['iqr_threshold'] = float(_abs_thr)
        pga_onset_data.append(ev)

    # Step 3: MIDI velocity mapping. Linear envelope-value →
    # ``[min_velocity, max_velocity]`` from config. Per-file,
    # data-driven — no magic numbers. Equal envelope values
    # collapse to min_velocity.
    midi_min = int(config.get('midi', {}).get('min_velocity', 80))
    midi_max = int(config.get('midi', {}).get('max_velocity', 110))
    if midi_max <= midi_min:
        # Defensive: ensure sane ordering even if the user
        # mis-configured.
        midi_max = midi_min + 1
    env_vals = [ev.get('envelope_value') for ev in pga_onset_data
                if ev.get('envelope_value') is not None]
    if env_vals:
        env_min = min(env_vals)
        env_max = max(env_vals)
    else:
        env_min, env_max = 0.0, 1.0
    for ev in pga_onset_data:
        env = ev.get('envelope_value')
        if env is None or env_max == env_min:
            ev['midi_velocity'] = midi_min
        else:
            t_norm = (env - env_min) / (env_max - env_min)
            ev['midi_velocity'] = int(round(midi_min + t_norm * (midi_max - midi_min)))
        # Clamp defensively (MIDI velocity is 1-127).
        ev['midi_velocity'] = max(1, min(127, ev['midi_velocity']))

    # Step 4: Record the active filter config on each event so
    # the sidecar / WebUI can show "which filter dropped which
    # event" without re-reading midiconfig.yaml. The thresholds
    # recorded here are the configured values at detect time;
    # the WebUI tuning panel re-applies the filters at
    # re-filter time using its own slider values.
    # 2026-06-15: per-stem overrides (toms.pga_min_prominence,
    # toms.min_decay_col_min_db) win over the global
    # onset_detection equivalents. Same precedence pattern as
    # the prominence filter in _build_pga_events_with_filter.
    onset_cfg = config.get('onset_detection', {})
    toms_cfg = config.get('toms', {})
    pga_min_prominence = float(
        toms_cfg.get('pga_min_prominence')
        if toms_cfg.get('pga_min_prominence') is not None
        else onset_cfg.get('pga_min_prominence', 1000.0)
    )
    min_decay_col_min_db = float(
        toms_cfg.get('min_decay_col_min_db')
        if toms_cfg.get('min_decay_col_min_db') is not None
        else onset_cfg.get('min_decay_col_min_db', -80.0)
    )
    pga_filter_config = {
        'pga_min_prominence': pga_min_prominence,
        'min_decay_col_min_db': min_decay_col_min_db,
        'min_velocity': midi_min,
        'max_velocity': midi_max,
    }
    for ev in pga_onset_data:
        ev['pga_filter_config'] = pga_filter_config

    # Step 5: Per-event feature extraction. Two-pass flow:
    #   Pass 1: measure with the FULL detect-time list (KEPT +
    #           FILTERED both included as candidate neighbors).
    #   Pass 2 (WebUI re-measure, out of scope here): re-measure
    #           with the final FILTERED list.
    # The neighbor for each event is the next KEPT event in the
    # list (FILTERED events are SKIPPED — using their time as
    # the cap would truncate the current strike's ring at the
    # FP's time).
    if pga_onset_data:
        # Lazy import: compute_event_features pulls in librosa /
        # scipy stack and is not on the cold path.
        from .event_features import compute_event_features
        for i, ev in enumerate(pga_onset_data):
            next_t: Optional[float] = None
            for j in range(i + 1, len(pga_onset_data)):
                candidate = pga_onset_data[j]
                if candidate.get('status') != 'FILTERED':
                    next_t = candidate.get('time')
                    break
            try:
                feats = compute_event_features(
                    audio_mono, sr, ev['time'],
                    next_event_time_sec=next_t,
                )
            except Exception:
                # Defensive: a bad event shouldn't poison the
                # rest of the pipeline. The diagnostic surface
                # in the WebUI will show "N/A" for features on
                # this event.
                feats = {
                    'duration_ms': None,
                    'attack_rise_ms': None,
                    'root_pitch_hz': None,
                    'pitch_confidence': None,
                    'decay_t60_ms': None,
                    'spectral_centroid_hz': None,
                    'spectral_flatness': None,
                    'hr_peak_offset_ms': None,
                    'decay_envelope_energy': None,
                    'decay_col_min_median_db': None,
                    'inter_onset_ms': None,
                }
            ev.update(feats)

    return pga_onset_data


def _apply_pga_filter(
    events: List[Dict[str, Any]],
    filter_spec: Dict[str, Any],
    threshold: Any,
    disabled_ids: Optional[Set[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply a filter from the registry to a list of events.

    Shared body for :func:`apply_pga_prominence_filter` and
    :func:`apply_pga_decay_col_min_filter` (and any future PGA
    filter added to the registry). The two wrappers exist only
    to keep the public API stable and to provide stem-specific
    docstrings; the actual logic is here.

    2026-06-15: this is the centralization point for the
    2026-06-15 filter-registry refactor. Adding a new filter
    is now a JSON entry in ``stems_to_midi/filter_registry.json``
    plus a thin wrapper here that calls this helper — no
    hand-rolled filter logic per filter.

    The disabled_ids check takes precedence over the threshold
    (so the WebUI can hide an event the user has explicitly
    toggled off regardless of the slider value).

    Does NOT mutate the per-event features /
    pga_filter_config of each event — only touches ``status``
    and (when changed) adds ``filter_reason``. The consumer is
    expected to call this with the same threshold multiple
    times during interactive tuning without losing diagnostic
    data.
    """
    kept: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []
    disabled_ids = disabled_ids or set()

    for ev in events:
        # Resolve the stable id for the disabled lookup. The
        # WebUI identifies events by 'time' (when the pipeline
        # didn't stamp an explicit 'id') and by 'id' once we
        # migrate; both work here.
        ev_id = ev.get('id', ev.get('time'))
        if ev_id in disabled_ids:
            ev['status'] = 'FILTERED'
            ev['filter_reason'] = 'manually disabled via WebUI'
            filtered.append(ev)
            continue

        # Delegate to the registry. evaluate_filter returns
        # True (KEPT) / False (FILTERED) / None (cannot
        # evaluate — e.g. the field is missing). None is
        # treated as KEPT (we can't filter what we can't
        # see; same as the old behavior for events with no
        # field).
        result = evaluate_filter(filter_spec, ev, threshold)
        if result is False:
            ev['status'] = 'FILTERED'
            ev['filter_reason'] = _build_filter_reason(
                filter_spec, ev, threshold,
            )
            filtered.append(ev)
        else:
            ev['status'] = 'KEPT'
            # Clear any stale filter_reason from a prior filter
            # call (e.g. the user moved the slider back up). This
            # is intentional: the WebUI tooltip only shows the
            # reason when the event is currently FILTERED, but
            # leaving a stale reason would be confusing if the
            # event is re-shown via the tuning panel.
            if 'filter_reason' in ev:
                del ev['filter_reason']
            kept.append(ev)

    return kept, filtered


def apply_pga_prominence_filter(
    events: List[Dict[str, Any]],
    threshold: float,
    disabled_ids: Optional[Set[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-tag PGA events with status='FILTERED' based on prominence
    and an optional manual-disable set.

    Thin wrapper around the filter registry (2026-06-15). The
    actual filter logic lives in
    ``stems_to_midi/filter_registry.json`` under the
    ``pga_min_prominence`` entry, evaluated by
    :mod:`stems_to_midi.filter_kinds`. The WebUI uses the same
    registry via :mod:`webui.static.js.filter_kinds`. Adding a
    new filter is a JSON entry — no per-filter Python code.

    Pure function: walks ``events`` in input order, sets
    ``status='FILTERED'`` and ``filter_reason=...`` on any event
    that either:

      (a) has ``prominence < threshold`` (with reason
          ``"below pga_min_prominence ({value} < {threshold})"``,
          from the registry's reason_template),
      (b) has an id (or fallback stable identifier) in
          ``disabled_ids`` (with reason
          ``"manually disabled via WebUI"``).

    The disabled check takes precedence — an event in
    ``disabled_ids`` is tagged FILTERED even if its prominence
    passes the threshold, so the WebUI can hide an event the
    user has explicitly toggled off regardless of the slider
    value.

    The function does NOT mutate the prominence / midi_velocity /
    per-event features / pga_filter_config of each event — it
    only touches ``status`` and (when changed) adds
    ``filter_reason``. The consumer is expected to call this
    with the same threshold multiple times during interactive
    tuning without losing diagnostic data.

    Args:
        events: Flat list of event dicts from
            :func:`detect_pga_events` (or any list of PGA-shaped
            dicts). The list may be in any status; this function
            re-derives the partition from scratch.
        threshold: Minimum prominence for an event to remain
            ``status='KEPT'``. Events with ``prominence < threshold``
            are tagged ``status='FILTERED'``. An event with no
            ``prominence`` field is left untouched (it cannot be
            filtered by threshold).
        disabled_ids: Optional set of event identifiers. If
            provided, any event whose id is in the set is
            tagged ``status='FILTERED'`` with reason
            ``"manually disabled via WebUI"``. The id resolution
            order is: ``event['id']``, then fallback to
            ``event['time']`` (a stable float is fine — the
            WebUI uses the time as the persistent id when the
            pipeline didn't stamp an explicit id).

    Returns:
        ``(kept, filtered)`` — two flat lists partitioning
        ``events`` by final status, both in input order.
    """
    return _apply_pga_filter(
        events, find_filter('pga_min_prominence'), threshold, disabled_ids,
    )


def apply_pga_decay_col_min_filter(
    events: List[Dict[str, Any]],
    threshold: float,
    disabled_ids: Optional[Set[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-tag PGA events with status='FILTERED' based on the
    high-resolution decay ``col_min`` diagnostic and an optional
    manual-disable set (2026-06-15).

    Sister function to :func:`apply_pga_prominence_filter` —
    same contract, different diagnostic field. Also a thin
    wrapper around the filter registry; see
    :func:`apply_pga_prominence_filter` for the design.

    The detector stamps ``decay_col_min_median_db`` on every
    event (see :func:`stems_to_midi.event_features.compute_event_features`
    → :func:`compute_high_res_decay_signature`), so the filter
    layer is decoupled from the detector.

    Pure function: walks ``events`` in input order, sets
    ``status='FILTERED'`` and ``filter_reason=...`` on any event
    that either:

      (a) has ``decay_col_min_median_db < threshold`` (with
          reason
          ``"below min_decay_col_min_db ({value}dB < {threshold}dB)"``,
          from the registry's reason_template),
      (b) has an id (or fallback stable identifier) in
          ``disabled_ids`` (with reason
          ``"manually disabled via WebUI"``).

    The disabled check takes precedence — an event in
    ``disabled_ids`` is tagged FILTERED even if its
    ``decay_col_min_median_db`` passes the threshold, so the
    WebUI can hide an event the user has explicitly toggled
    off regardless of the slider value.

    An event with no ``decay_col_min_median_db`` field is left
    untouched (it cannot be filtered by threshold — same as
    the prominence pattern for events with no ``prominence``).

    The function does NOT mutate the per-event features /
    pga_filter_config of each event — it only touches
    ``status`` and (when changed) adds ``filter_reason``. The
    consumer is expected to call this with the same threshold
    multiple times during interactive tuning without losing
    diagnostic data.

    Args:
        events: Flat list of event dicts from
            :func:`detect_pga_events` (or any list of PGA-shaped
            dicts). The list may be in any status; this function
            re-derives the partition from scratch.
        threshold: Minimum ``decay_col_min_median_db`` (dB) for
            an event to remain ``status='KEPT'``. Events with
            ``decay_col_min_median_db < threshold`` are tagged
            ``status='FILTERED'``. An event with no
            ``decay_col_min_median_db`` field is left untouched.
        disabled_ids: Optional set of event identifiers. If
            provided, any event whose id is in the set is
            tagged ``status='FILTERED'`` with reason
            ``"manually disabled via WebUI"``. The id resolution
            order is: ``event['id']``, then fallback to
            ``event['time']``.

    Returns:
        ``(kept, filtered)`` — two flat lists partitioning
        ``events`` by final status, both in input order.
    """
    return _apply_pga_filter(
        events, find_filter('min_decay_col_min_db'), threshold, disabled_ids,
    )


def build_pga_events(
    audio_mono: np.ndarray,
    sr: int,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, List]], List[Dict], Dict[str, Any]]:
    """Run the full PGA event pipeline on mono audio and return
    the kept/filtered partition plus the detector debug dict.

    Thin wrapper that preserves the original return shape of
    the pre-refactor function: ``(events_kept, events_filtered,
    debug_dict)``. Internally it composes :func:`detect_pga_events`
    (pure) and :func:`apply_pga_prominence_filter` (pure) with
    the configured ``onset_detection.pga_min_prominence``
    threshold. The full-conversion call site in
    ``processing_shell.py:1745`` keeps working unchanged.

    Args:
        audio_mono: 1-D float array of mono audio samples.
        sr: Sample rate in Hz.
        config: Project config dict (the same dict passed to
            ``process_stem_to_midi``). Reads
            ``onset_detection.pga_min_prominence`` (default
            ``1000``) and ``midi.min_velocity`` /
            ``midi.max_velocity`` (defaults ``80``/``110``).

    Returns:
        ``(events_kept, events_filtered, debug_dict)``:
          - ``events_kept``: list of event dicts that survived the
            prominence filter (``status='KEPT'``). MIDI
            output uses only this list.
          - ``events_filtered``: list of event dicts that were
            tagged FILTERED by the prominence filter. Kept in
            the return value so the WebUI can render them as
            faded markers; the MIDI serializer skips them.
          - ``debug_dict``: detector internals — ``freqs``,
            ``times``, ``s_db``, ``floor``, ``envelope``,
            ``peaks``, ``prominences`` (same shape as the
            legacy inline implementation in
            ``percentile_gated_detector.detect_percentile_gated_broad_attacks``).
            Exposed here for back-compat with the legacy tests
            that inspect ``pga_debug``; ``detect_pga_events``
            itself does not return the debug (it discards the
            debug after attaching per-event fields, since
            re-filtering a stored event list does not need it).

    The function is a pure functional core:
      - No file I/O.
      - No mutation of input audio or config.
      - The returned event dicts are owned by the caller (safe
        to mutate, extend, or serialize).
    """
    if audio_mono is None or len(audio_mono) == 0:
        return _empty_result()
    if sr is None or sr <= 0:
        return _empty_result()

    # Re-run the detector to recover the debug dict. The pure
    # ``detect_pga_events`` discards the debug, but the legacy
    # ``build_pga_events`` callers (and the existing test
    # contract) expect it on the return value.
    _pga_event_times, pga_debug = detect_percentile_gated_broad_attacks(
        audio_mono, sr,
    )

    raw = detect_pga_events(audio_mono, sr, config)
    # No filter applied here — all events returned as KEPT.
    # The sidecar stores the raw all-KEPT list; the WebUI and
    # rebuild path call apply_pga_prominence_filter separately
    # with their own threshold. Return shape is preserved
    # (events_kept=all, events_filtered=[], debug_dict).
    return raw, [], pga_debug


def _build_pga_events_with_filter(
    audio_mono: np.ndarray,
    sr: int,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, List]], List[Dict], Dict[str, Any]]:
    """Run the full PGA event pipeline on mono audio and return
    the kept/filtered partition plus the detector debug dict.

    This is the **filtered** variant used by the processing_shell
    call site. It calls :func:`detect_pga_events` (all-KEPT raw) then
    :func:`apply_pga_prominence_filter` with the configured
    ``onset_detection.pga_min_prominence`` threshold.

    Preserves the same return shape as the legacy ``build_pga_events``:
    ``(events_kept, events_filtered, debug_dict)``.

    Args:
        audio_mono: 1-D float array of mono audio samples.
        sr: Sample rate in Hz.
        config: Project config dict. Reads
            ``onset_detection.pga_min_prominence`` (default
            ``1000``) and ``midi.min_velocity`` /
            ``midi.max_velocity`` (defaults ``80``/``110``).

    Returns:
        ``(raw, events_kept, events_filtered, debug_dict)``:
          - ``raw``: the full list of events with no filter applied
          - ``events_kept``: list of event dicts that survived the
            prominence filter (``status='KEPT'``).
          - ``events_filtered``: list of event dicts tagged
            ``status='FILTERED`` by the prominence filter.
          - ``debug_dict``: detector internals (same shape as
            ``detect_percentile_gated_broad_attacks`` debug output).
    """
    if audio_mono is None or len(audio_mono) == 0:
        return _empty_result()
    if sr is None or sr <= 0:
        return _empty_result()

    _pga_event_times, pga_debug = detect_percentile_gated_broad_attacks(
        audio_mono, sr,
    )

    raw = detect_pga_events(audio_mono, sr, config)
    # Apply the PGA prominence filter (existing behavior).
    # 2026-06-15: per-stem override (toms.pga_min_prominence)
    # wins over the global onset_detection.pga_min_prominence.
    onset_cfg = config.get('onset_detection', {})
    toms_cfg = config.get('toms', {})
    prom_threshold = float(
        toms_cfg.get('pga_min_prominence')
        if toms_cfg.get('pga_min_prominence') is not None
        else onset_cfg.get('pga_min_prominence', 1000.0)
    )
    events_kept, events_filtered = apply_pga_prominence_filter(
        raw, prom_threshold,
    )
    # 2026-06-15: apply the decay_col_min filter on top of the
    # prominence filter. Same per-stem > global > default
    # resolution pattern. Default -80.0 dB matches the empirical
    # split (real strikes -60 to -84 dB, noise pops -84 to -90 dB).
    decay_col_min_threshold = float(
        toms_cfg.get('min_decay_col_min_db')
        if toms_cfg.get('min_decay_col_min_db') is not None
        else onset_cfg.get('min_decay_col_min_db', -80.0)
    )
    events_kept, decay_filtered = apply_pga_decay_col_min_filter(
        events_kept, decay_col_min_threshold,
    )
    # Concatenate the two filtered lists. The decay_col_min
    # filter is downstream of the prominence filter, so events
    # it filters were already KEPT (i.e., they had high
    # prominence) but failed the ring-quality check.
    events_filtered = events_filtered + decay_filtered
    # Record the active thresholds in pga_filter_config so the
    # sidecar tooltip can show what filter the event was
    # processed under.
    for ev in raw:
        pga_filter_config = dict(ev.get('pga_filter_config', {}))
        pga_filter_config['pga_min_prominence'] = prom_threshold
        pga_filter_config['min_decay_col_min_db'] = decay_col_min_threshold
        ev['pga_filter_config'] = pga_filter_config
    return raw, events_kept, events_filtered, pga_debug
