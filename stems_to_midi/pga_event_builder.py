"""
PGA (percentile-gated broad-attack) event builder for the toms stem.

Isolates the toms detection path (2026-06-13 refactor) as a pure
functional core. The function ``build_pga_events`` takes
``(audio_mono, sr, config)`` and returns a 3-tuple
``(events_kept, events_filtered, debug_dict)`` suitable for direct
consumption by ``process_stem_to_midi``'s toms branch and the
``_serialize_onset_events`` sidecar path.

Pipeline (matches the original inline flow in
``processing_shell.py:1717-1892``):
  1. ``detect_percentile_gated_broad_attacks`` (broadband contrast
     envelope + IQR-thresholded peak-pick).
  2. Per-event diagnostic dict: ``time``, ``method``,
     ``status='KEPT'``, ``frame``, ``envelope_value``,
     ``prominence``, ``iqr_threshold``.
  3. Prominence filter (config ``onset_detection.pga_min_prominence``,
     default 1000) moves low-prominence events from KEPT to FILTERED.
  4. MIDI velocity mapping: linear envelope-value → MIDI
     ``[min_velocity, max_velocity]`` from ``config.midi``.
  5. Per-event feature extraction via
     ``event_features.compute_event_features`` — duration, pitch,
     decay, brightness, etc. Two-pass flow: pass 1 uses the
     post-prominence-filter list as the "neighbors" (the events the
     user is most likely to keep); pass 2 is the WebUI re-measure
     path that runs against the final FILTERED list.

Design constraints:
  - Pure function. No file I/O, no module-level state, no
    side-effects beyond ``print()`` (imperative-shell residue).
  - Imports only standard library / numpy / scipy / in-project
    modules that the rest of the pipeline already uses.
  - Returns ``events_kept`` and ``events_filtered`` as separate
    lists so consumers can pass them straight to the existing
    serializer (which expects KEPT first, then FILTERED).
  - ``debug_dict`` exposes the raw peak indices, envelope, and
    prominences for diagnostics — same shape as the legacy
    inline implementation, so existing tests that inspect
    ``pga_debug`` continue to work.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .percentile_gated_detector import detect_percentile_gated_broad_attacks


__all__ = [
    'build_pga_events',
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


def build_pga_events(
    audio_mono: np.ndarray,
    sr: int,
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """Run the full PGA event pipeline on mono audio.

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

    # Step 1: Run the percentile-gated broad-attack detector.
    # The detector returns a list of sub-frame-accurate strike
    # times in seconds and a debug dict with envelope /
    # peak metadata.
    pga_event_times, pga_debug = detect_percentile_gated_broad_attacks(
        audio_mono, sr,
    )

    _env = pga_debug.get('envelope') if pga_debug else None
    _peaks = pga_debug.get('peaks') if pga_debug else None
    _proms = pga_debug.get('prominences') if pga_debug else None

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

    # Step 3: Prominence filter. Tag events with prominence
    # below ``pga_min_prominence`` as FILTERED. The default 1000
    # was chosen empirically on project 4 calibration (2026-06-10)
    # — real toms strikes had prominence 2000-15000, the 14.84/14.97
    # soft hits had 2127-2727 (so they survive the default — the
    # duration feature catches them), and the 74.748/74.925 FPs
    # had 127-432 (so the default kills them). See
    # midiconfig.yaml's ``onset_detection.pga_min_prominence``.
    pga_min_prominence = float(
        config.get('onset_detection', {}).get('pga_min_prominence', 1000.0)
    )
    pga_filtered_count = 0
    for ev in pga_onset_data:
        prom = ev.get('prominence')
        if prom is not None and prom < pga_min_prominence:
            ev['status'] = 'FILTERED'
            ev['filter_reason'] = (
                f"below pga_min_prominence ({prom:.0f} < {pga_min_prominence:.0f})"
            )
            pga_filtered_count += 1
    if pga_filtered_count:
        print(f"    PGA prominence filter (min={pga_min_prominence}): "
              f"tagged {pga_filtered_count} events as FILTERED")

    # Step 4: MIDI velocity mapping. Linear envelope-value →
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

    # Step 5: Record the active filter config on each event so
    # the sidecar / WebUI can show "which filter dropped which
    # event" without re-reading midiconfig.yaml.
    pga_filter_config = {
        'pga_min_prominence': pga_min_prominence,
        'min_velocity': midi_min,
        'max_velocity': midi_max,
    }
    for ev in pga_onset_data:
        ev['pga_filter_config'] = pga_filter_config

    # Step 6: Per-event feature extraction. Two-pass flow:
    #   Pass 1: measure with the FULL detected list (the
    #           post-prominence-filter list, so neighbors are
    #           events the user is most likely to keep).
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

    # Step 7: Split the unified list into KEPT / FILTERED for
    # the consumer. Both lists preserve detection order so the
    # sidecar / WebUI can render them in time order without
    # re-sorting.
    events_kept = [ev for ev in pga_onset_data if ev.get('status') != 'FILTERED']
    events_filtered = [ev for ev in pga_onset_data if ev.get('status') == 'FILTERED']

    return events_kept, events_filtered, pga_debug
