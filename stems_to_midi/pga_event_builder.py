"""
PGA (percentile-gated broad-attack) event builder for the toms stem.

Isolates the toms detection path (2026-06-13 refactor) as a pure
functional core. The public surface is two pure functions plus a
thin wrapper:

  - ``detect_pga_events(audio_mono, sr, config)`` runs the
    percentile-gated broad-attack detector and attaches
    per-event diagnostic fields (frame, envelope_value,
    prominence, iqr_threshold, midi_velocity,
    pga_filter_config). **All** events are returned with
    ``status='KEPT'`` — no filter is applied, and (2026-06-19)
    no per-event features are computed. The consumer can
    re-filter the result later via a single pure function call
    and re-run feature extraction against the post-filter
    neighbor set via ``_compute_features_for_filtered_events``.

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
    that calls ``detect_pga_events``, runs the configured
    ``pga_min_prominence`` filter, and (2026-06-19) re-runs
    feature extraction on the post-filter list. Preserves the
    original return shape ``(events_kept, events_filtered,
    debug_dict)`` so the existing call site in
    ``processing_shell.py`` and the legacy tests keep working
    without modification.

Pipeline (matches the original inline flow in
``processing_shell.py:1717-1892``, restructured 2026-06-19 so
feature extraction is a post-filter pass):
  1. ``detect_percentile_gated_broad_attacks`` (broadband contrast
     envelope + IQR-thresholded peak-pick).
  2. Per-event diagnostic dict: ``time``, ``method``,
     ``status='KEPT'``, ``frame``, ``envelope_value``,
     ``prominence``, ``iqr_threshold``.
  3. MIDI velocity mapping: linear envelope-value → MIDI
     ``[min_velocity, max_velocity]`` from ``config.midi``.
  4. Prominence filter (config ``onset_detection.pga_min_prominence``,
     default 1000) — moves low-prominence events from KEPT
     to FILTERED. (Was step 5; reordered 2026-06-19.)
  4b. Decay-col-min filter (``onset_detection.min_decay_col_min_db``,
     default -80.0 dB) — moves noise-pop events to FILTERED.
  4c. Attack-rise filter (``onset_detection.attack_rise_max_ms``,
     default 20.0 ms) — moves long-rise FPs to FILTERED.
  5. Per-event feature extraction via
     ``event_features.compute_event_features`` — duration, pitch,
     decay, brightness, etc. Now runs AGAINST the post-filter
     neighbor set, so ``duration_ms`` /
     ``duration_to_valley_ms`` / ``attack_rise_ms`` /
     ``inter_onset_ms`` reflect the kept event neighborhood.
     A filtered-out FP between two kept strikes no longer caps
     the prior strike's ring at the FP's time.

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

The WebUI re-filter path (planned follow-up to this refactor)
will call ``detect_pga_events`` once at sidecar-write time,
store the raw list in the sidecar, and then on every
tuning-panel change re-apply the filter and re-run
``_compute_features_for_filtered_events`` so the WebUI's
"why was this dropped" tooltip and the per-event feature
columns always reflect the post-filter neighbor set.
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import math

import numpy as np

from .percentile_gated_detector import detect_percentile_gated_broad_attacks
from .percentile_gated_detector import (
    DEFAULT_MAX_FLOOR_GATE_DB,
    DEFAULT_BROAD_FREQ_MIN_HZ,
    DEFAULT_BROAD_FREQ_MAX_HZ,
    DEFAULT_DB_RISE_THRESHOLD,
    DEFAULT_NMS_MIN_FRAMES,
    DEFAULT_STRIKE_OFFSET_SEC,
)
from .filter_kinds import (
    find_filter,
    evaluate_filter,
    build_filter_reason as _build_filter_reason,
)
# 2026-06-29: hihat "openness" score stamping (TEST viability hook).
# Stamp the per-event openness score on raw PGA events BEFORE any
# filter runs, so the user can see the score distribution across
# KEPT/FILTERED events in the sidecar. Lazy-imports librosa inside
# the wrapper so non-hihat stems don't pay the cold-path cost.
from .note_classification_core import stamp_hihat_openness_score


__all__ = [
    'detect_pga_events',
    'apply_pga_prominence_filter',
    'apply_pga_decay_col_min_filter',
    'apply_attack_rise_max_filter',
    '_build_pga_events_with_filter',
    '_compute_walk_features_for_filtered_events',
    '_resolve_max_floor_gate_db',
    '_resolve_pga_abs_envelope_threshold',
    '_resolve_pga_detector_param',
    'apply_pga_min_envelope_value',
]


# ---------------------------------------------------------------------------
# Envelope walk — open/closed hihat discriminator (2026-06-19)
# ---------------------------------------------------------------------------
# The PGA detector computes a broadband contrast envelope over the
# whole stem (see percentile_gated_detector._broad_attack_envelope).
# At detect time we cache that envelope on the debug dict. For every
# detected event, we walk the envelope in two directions from the
# peak frame and record the dB-domain and linear-domain slopes, the
# pct_at_stop (where the forward walk ended as a fraction of peak),
# and the onset cross status (whether the envelope drops below the
# back-stop threshold within a tight backward window).
#
# These fields are baked into events_pga in the sidecar so the
# hihat_state classifier (note_classification_core.classify_hihat_notes)
# can consume them at rebuild time without re-running the detector.
# The WebUI's slope_threshold slider (settings_schema.hihat_slope_threshold)
# controls the cut between closed (steep slope, clean decay) and open
# (shallow slope, ring-out before next hit).

# Forward (decay) walk stops at one of:
#   * envelope drops to DECAY_PCT_THRESHOLD of peak (clean ring-out), or
#   * another KEPT event's peak frame is hit (next hit cuts in), or
#   * DECAY_MAX_FRAMES reached (~1.16s — long-ringing open hihat).
DECAY_PCT_THRESHOLD = 0.50
DECAY_MAX_FRAMES = 200

# Backward (onset) walk stops at one of:
#   * envelope drops to ONSET_PCT_THRESHOLD of peak (clean pre-strike silence), or
#   * another KEPT event's peak frame is hit, or
#   * ONSET_MAX_FRAMES reached (~35ms — hihat attacks are short).
# Onset is walked on the RAW envelope (not rolling-mean-smoothed) so
# the silence floor before a strike is correctly captured — smoothing
# pads 1-2 frames out and stops us crossing.
ONSET_PCT_THRESHOLD = 0.20
ONSET_MAX_FRAMES = 6

# dB-domain floor for log(0) — same as midi.py. We use this when
# averaging per-frame dB deltas so a true zero doesn't poison the mean.
_DB_FLOOR = 1e-9


def _resolve_max_floor_gate_db(
    config: Dict[str, Any],
    stem_type: Optional[str] = None,
) -> float:
    """Resolve the global noise-floor gate cap (dB) with the
    standard per-stem > global > default precedence.

    Resolution order (2026-06-18, generalized 2026-06-19 to
    any stem — was previously hardcoded to ``toms`` only,
    which meant a ``snare.pga_max_floor_gate_db`` override
    was silently ignored, falling through to the global or
    module default):
      1. ``<stem_type>.pga_max_floor_gate_db`` — per-stem
         override. When ``stem_type`` is provided, ONLY that
         stem is checked. When ``stem_type`` is None
         (legacy / test fallback), all known stems are
         walked and the first non-None value wins.
      2. ``onset_detection.pga_max_floor_gate_db`` (the global
         setting in midiconfig.yaml).
      3. :data:`percentile_gated_detector.DEFAULT_MAX_FLOOR_GATE_DB`
         (currently -60.0 dB — see that module's docstring
         for the rationale).

    Set the per-stem or global value to a very large positive
    number (e.g. ``1000``) to effectively disable the cap
    (the implementation does ``min(raw_gate, cap)``, so a
    cap larger than any plausible raw gate is a no-op).
    Set to a value BELOW the raw max p5 to force the floor
    to that value (useful for diagnostic / tuning runs).

    ``None`` values are skipped at every level so YAML
    ``null`` keeps the default. Non-numeric values fall back
    to the module default. The returned value is always a
    finite float.
    """
    onset_cfg = config.get('onset_detection', {}) or {}
    # Per-stem overrides.
    if stem_type is not None:
        stem_cfgs = [config.get(stem_type, {}) or {}]
    else:
        # Fallback: walk all known stems. The first non-None
        # wins. Same caveat as the other resolvers —
        # production callers should pass stem_type.
        stem_cfgs = [config.get(s, {}) or {} for s in _PGA_STEM_NAMES]
    for stem_cfg in stem_cfgs:
        raw = stem_cfg.get('pga_max_floor_gate_db')
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    # Global override.
    raw = onset_cfg.get('pga_max_floor_gate_db', DEFAULT_MAX_FLOOR_GATE_DB)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_FLOOR_GATE_DB


# Stems known to the PGA pipeline. Used by
# ``_resolve_pga_abs_envelope_threshold`` to walk per-stem
# overrides in priority order. The list is intentionally
# small — only the stems the PGA detector is calibrated
# for. The function takes the FIRST non-None value it
# finds, so the iteration order matters only when more
# than one stem sets the override (which doesn't happen in
# practice: each project config has at most one stem on
# the PGA path).
_PGA_STEM_NAMES = ('toms', 'snare', 'kick', 'hihat', 'cymbals')


def _resolve_pga_abs_envelope_threshold(
    config: Dict[str, Any],
    stem_type: Optional[str] = None,
) -> Optional[float]:
    """Resolve the absolute envelope threshold for the PGA
    detector's ``find_peaks`` step, with per-stem > global
    > IQR-auto precedence.

    Resolution order (2026-06-18):
      1. ``<stem_type>.pga_abs_envelope_threshold`` — per-stem
         override. When ``stem_type`` is provided, ONLY that
         stem is checked. When ``stem_type`` is None (the
         default — e.g. from a test that doesn't know which
         stem is being processed), all known stems are
         walked and the first non-None value wins. This
         fallback is unsafe when MULTIPLE stems in the
         config carry the override at the same time
         (e.g. toms + snare both set it), so production
         callers should always pass ``stem_type``.
      2. ``onset_detection.pga_abs_envelope_threshold`` —
         global override.
      3. ``None`` — the detector falls back to its
         IQR-based auto-threshold (``q3 + 2.5*IQR`` of the
         envelope). This is the default behavior; it
         works for stems with a wide dynamic range
         (e.g. toms — quiet frames vs. loud strikes) but
         FAILS for stems with a narrow dynamic range
         (e.g. snare — many similar-loudness hits), where
         the IQR ends up small and the auto-threshold
         lands above the envelope maximum. For those
         stems, set a fixed value.

    ``None`` values are skipped at every level so YAML
    ``null`` keeps the IQR-auto default. Non-numeric
    values fall back to ``None`` (auto) rather than
    crashing the pipeline. A returned float is always
    finite and ``> 0`` (a threshold of 0 is meaningless
    for ``find_peaks``).
    """
    onset_cfg = config.get('onset_detection', {}) or {}
    # Per-stem overrides.
    if stem_type is not None:
        stem_cfg = config.get(stem_type, {}) or {}
        raw = stem_cfg.get('pga_abs_envelope_threshold')
        if raw is not None:
            try:
                val = float(raw)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
    else:
        # Fallback: walk all known stems. The first non-None
        # wins. This is the legacy behavior — kept so
        # existing tests that don't pass stem_type still
        # work, but production callers (process_stem_to_midi,
        # process_percentile_gated) should always pass
        # stem_type to avoid one stem's threshold leaking
        # into another's detection.
        for stem_name in _PGA_STEM_NAMES:
            stem_cfg = config.get(stem_name, {}) or {}
            raw = stem_cfg.get('pga_abs_envelope_threshold')
            if raw is not None:
                try:
                    val = float(raw)
                    if val > 0:
                        return val
                except (TypeError, ValueError):
                    pass
    # Global override.
    raw = onset_cfg.get('pga_abs_envelope_threshold')
    if raw is not None:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    # IQR-auto fallback.
    return None


def _resolve_pga_detector_param(
    config: Dict[str, Any],
    key: str,
    default: Union[float, int],
    stem_type: Optional[str] = None,
) -> Union[float, int]:
    """Resolve a PGA detector tuning parameter with per-stem
    > global > default precedence.

    Args:
        config: Project config dict.
        key: The YAML key to look up (see "Available keys"
            below).
        default: The value to return when neither the per-stem
            nor the global override is set. The TYPE of
            ``default`` is preserved — if default is ``int``,
            the returned value is ``int``; if default is
            ``float``, the returned value is ``float``. This
            matches the type the detector expects.
        stem_type: If provided, ONLY this stem's section is
            checked. If ``None`` (the default — e.g. from a
            test that doesn't know which stem is being
            processed), all known stems are walked and the
            first non-None value wins. Production callers
            (process_stem_to_midi, process_percentile_gated)
            should always pass ``stem_type`` to avoid one
            stem's threshold leaking into another's
            detection.

    Returns:
        The resolved value as the same type as ``default``.
        Falls back to ``default`` on any error (missing
        key, unparseable string, etc.) so the detector
        always gets a usable value.

    Available keys (2026-06-18 — wired in this commit):
        - ``pga_broad_freq_min_hz`` (float, default
          :data:`DEFAULT_BROAD_FREQ_MIN_HZ` = 600.0):
          inclusive lower bound of the broadband frequency
          range summed for the contrast envelope. Default
          excludes the 0-600 Hz low bands that saturate on
          toms strikes. Set lower (e.g. 200) for stems
          whose attack energy lives in the body range
          (snare, kick).
        - ``pga_broad_freq_max_hz`` (float, default
          :data:`DEFAULT_BROAD_FREQ_MAX_HZ` = 8000.0):
          inclusive upper bound of the broadband frequency
          range. Default 8000 captures cymbal/shaker
          sizzle; set higher (e.g. 12000) for stems with
          high-frequency attack content.
        - ``pga_db_rise_threshold`` (float, default
          :data:`DEFAULT_DB_RISE_THRESHOLD` = 10.0):
          per-bin contrast threshold in dB. 10 dB is
          "an order of magnitude above noise"; set lower
          (e.g. 5) to recover quieter hits at the cost
          of more noise FPs.
        - ``pga_nms_min_frames`` (int, default
          :data:`DEFAULT_NMS_MIN_FRAMES` = 20): minimum
          STFT frames between peaks (~116ms at hop=256).
          Default is the "safe NMS floor" — shorter would
          merge flams, longer would split sixteenths.
          Set to 0 to disable NMS (every peak kept).
        - ``pga_strike_offset_sec`` (float, default
          :data:`DEFAULT_STRIKE_OFFSET_SEC` = 0.008):
          Hann window center-of-bin bias correction in
          seconds. The contrast envelope peaks a few ms
          after the true strike onset; this offset shifts
          every event time backward by this amount.
          Default 8ms is calibrated on toms; tune per
          -stem if the bias is different (e.g. tighter
          attacks may need a smaller offset).
    """
    onset_cfg = config.get('onset_detection', {}) or {}
    # Pick the per-stem search strategy.
    if stem_type is not None:
        stem_cfgs = [config.get(stem_type, {}) or {}]
    else:
        # Fallback: walk all known stems. The first non-None
        # wins. Same caveat as _resolve_pga_abs_envelope_threshold
        # — production callers should pass stem_type to
        # avoid one stem's threshold leaking into another's
        # detection.
        stem_cfgs = [config.get(s, {}) or {} for s in _PGA_STEM_NAMES]
    # Per-stem override.
    for stem_cfg in stem_cfgs:
        raw = stem_cfg.get(key)
        if raw is not None:
            try:
                return type(default)(raw)
            except (TypeError, ValueError):
                pass
    # Global override.
    raw = onset_cfg.get(key)
    if raw is not None:
        try:
            return type(default)(raw)
        except (TypeError, ValueError):
            pass
    return default


def _resolve_pga_detection_method(
    config: Dict[str, Any],
    stem_type: Optional[str] = None,
) -> str:
    """Resolve the PGA detection method (delta | peak) with
    per-stem > global > default precedence. (2026-06-26)

    Args:
        config: Project config dict.
        stem_type: If provided, ONLY this stem's section is
            checked. If ``None`` (e.g. from a test that doesn't
            know which stem is being processed), all known
            PGA stems are walked and the first non-None value
            wins. Production callers should always pass
            ``stem_type``.

    Returns:
        ``'delta'`` or ``'peak'`` — case-insensitive. ``'delta'``
        uses the 5-frame rate-of-change signal (Δ5) for
        peak detection; ``'peak'`` uses the raw contrast
        envelope. Defaults to ``'delta'`` (matches the most
        recent experiment code state; hihat specifically
        benefits from delta because the hihat strike's
        "stick + mass" double-transient registers as two
        delta peaks but one envelope peak).

    Precedence:
        1. ``<stem_type>.pga_detection_method`` — per-stem
        2. ``onset_detection.pga_detection_method`` — global
        3. ``'delta'`` — default

    Unknown values fall through to the default rather than
    raising — this way a typo in midiconfig.yaml degrades
    gracefully to delta (the safer fallback) rather than
    crashing the whole pipeline.
    """
    VALID = ('delta', 'peak')

    def _normalize(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        return s if s in VALID else None

    onset_cfg = config.get('onset_detection', {}) or {}
    # 1. Per-stem override
    if stem_type is not None:
        raw = (config.get(stem_type, {}) or {}).get('pga_detection_method')
        v = _normalize(raw)
        if v is not None:
            return v
    else:
        # Walk all known PGA stems
        for stem_name in ('hihat', 'cymbals', 'snare', 'kick', 'toms'):
            raw = (config.get(stem_name, {}) or {}).get('pga_detection_method')
            v = _normalize(raw)
            if v is not None:
                return v
    # 2. Global override
    v = _normalize(onset_cfg.get('pga_detection_method'))
    if v is not None:
        return v
    # 3. Default
    return 'delta'


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
    stem_type: Optional[str] = None,
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

    Note: as of 2026-06-19, this function does NOT attach the
    per-event feature keys (``duration_ms``, ``attack_rise_ms``,
    ``pitch_hz``, ``pitch_confidence``, ``decay_t60_ms``,
    ``spectral_centroid_hz``, ``spectral_flatness``,
    ``hr_peak_offset_ms``, ``decay_envelope_energy``,
    ``decay_col_min_median_db``, ``inter_onset_ms``). Those are
    attached in a post-filter pass via
    :func:`_compute_features_for_filtered_events` so the
    neighbor-dependent fields (``duration_ms``,
    ``duration_to_valley_ms``, ``attack_rise_ms``,
    ``inter_onset_ms``) reflect the KEPT event set, not the
    pre-filter list. Callers that need the back-compat
    "events with features" contract should use
    :func:`build_pga_events` instead, or call
    :func:`_compute_features_for_filtered_events` themselves
    after applying their own filter.

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
    # 2026-06-18: pass the max_floor_gate_db cap (per-stem >
    # global > default, see _resolve_max_floor_gate_db) so the
    # detector's global noise-floor gate cannot over-lift on
    # dense/saturated mixes. See percentile_gated_detector.py
    # for the algorithm and midiconfig.yaml's
    # ``onset_detection.pga_max_floor_gate_db`` for the user
    # override.
    # 2026-06-18: also pass an optional absolute envelope
    # threshold override (per-stem > global > IQR-auto, see
    # _resolve_pga_abs_envelope_threshold). The default
    # IQR-based threshold (``q3 + 2.5*IQR``) works for
    # stems with wide dynamic range (toms) but FAILS for
    # narrow-range stems (snare) where the IQR is small
    # and the auto-threshold lands above the envelope
    # max, suppressing every peak. Setting
    # ``<stem>.pga_abs_envelope_threshold`` in the project
    # midiconfig forces a fixed value instead.
    # 2026-06-18: also pass the detector's other tuning
    # parameters through the same per-stem > global >
    # default resolution path (see
    # _resolve_pga_detector_param). Each value falls
    # back to the module's hard-coded default when
    # unset, so the pipeline behavior is identical to
    # before when no per-project overrides exist.
    # 2026-06-19: _resolve_max_floor_gate_db is now
    # stem_type-aware (generalized to drop the toms-only
    # hardcode that was leaking toms values into kick
    # processing). Per-stem overrides for any of the
    # 5 known PGA stems are honored.
    max_floor_gate_db = _resolve_max_floor_gate_db(config, stem_type)
    abs_envelope_threshold = _resolve_pga_abs_envelope_threshold(config, stem_type)
    broad_freq_min_hz = _resolve_pga_detector_param(
        config, 'pga_broad_freq_min_hz', DEFAULT_BROAD_FREQ_MIN_HZ, stem_type,
    )
    broad_freq_max_hz = _resolve_pga_detector_param(
        config, 'pga_broad_freq_max_hz', DEFAULT_BROAD_FREQ_MAX_HZ, stem_type,
    )
    db_rise_threshold = _resolve_pga_detector_param(
        config, 'pga_db_rise_threshold', DEFAULT_DB_RISE_THRESHOLD, stem_type,
    )
    nms_min_frames = _resolve_pga_detector_param(
        config, 'pga_nms_min_frames', DEFAULT_NMS_MIN_FRAMES, stem_type,
    )
    strike_offset_sec = _resolve_pga_detector_param(
        config, 'pga_strike_offset_sec', DEFAULT_STRIKE_OFFSET_SEC, stem_type,
    )
    # 2026-06-26: per-stem detection method (delta | peak).
    # 'delta' uses the 5-frame rate-of-change signal for peak
    # detection (hihat benefits from this; the stick + mass
    # double-transient registers as two distinct delta peaks).
    # 'peak' uses the raw contrast envelope (the original
    # behavior; cleaner for sustained stems like cymbals
    # and toms where delta would over-fragment the signal).
    # Resolved per-stem > global > 'delta' default.
    detection_method = _resolve_pga_detection_method(config, stem_type)
    pga_event_times, _pga_debug = detect_percentile_gated_broad_attacks(
        audio_mono, sr,
        broad_freq_min_hz=broad_freq_min_hz,
        broad_freq_max_hz=broad_freq_max_hz,
        db_rise_threshold=db_rise_threshold,
        abs_envelope_threshold=abs_envelope_threshold,
        nms_min_frames=nms_min_frames,
        strike_offset_sec=strike_offset_sec,
        max_floor_gate_db=max_floor_gate_db,
        detection_method=detection_method,
    )

    _env = _pga_debug.get('envelope') if _pga_debug else None
    _peaks = _pga_debug.get('peaks') if _pga_debug else None
    _proms = _pga_debug.get('prominences') if _pga_debug else None
    # Peak bases (2026-06-19): frame indices of the left/right
    # valley around each peak. Populated even when prominence=0
    # is passed to find_peaks. We add the per-event gap
    # (right_base - peak) in STFT frames + ms for downstream
    # hihat open/closed exploration.
    _lbases = _pga_debug.get('left_bases') if _pga_debug else None
    _rbases = _pga_debug.get('right_bases') if _pga_debug else None
    # Peak widths (2026-06-19): scipy.peak_widths at
    # rel_height=0.9. Bounded to a 10% slice around the peak,
    # so unlike left_bases/right_bases the measurements don't
    # run off to a distant baseline. left_ips / right_ips are
    # floating-point frame indices; we round to int and
    # compute the per-event attack/decay frame split.
    _lips = _pga_debug.get('peak_left_ips') if _pga_debug else None
    _rips = _pga_debug.get('peak_right_ips') if _pga_debug else None
    # 2026-06-26: per-event Δ1/Δ2/Δ5 stability + combined score
    # (sign-bearing warble filter). These were computed in
    # percentile_gated_detector.py but were only in the debug
    # dict, never surfaced in the sidecar. The WebUI's
    # warble-robustness filter needs them on each event; this
    # is the canonical place to copy them from the per-peak
    # debug arrays to the per-event dicts that the sidecar
    # serializes. All three ratios share the same definitions
    # and thresholds as in the detector (see the comments
    # there). -1.0 is a sentinel for "undefined" (no forward
    # window or zero peak). Empty arrays when ``peaks`` is
    # empty (no events detected) — all KEPT events still get
    # values, so the sidecar's per-event array will be the
    # same length as ``pga_onset_data``.
    _d1s = _pga_debug.get('delta1_stability') if _pga_debug else None
    _d2s = _pga_debug.get('delta2_stability') if _pga_debug else None
    _d5s = _pga_debug.get('delta5_stability') if _pga_debug else None
    _cs  = _pga_debug.get('combined_score') if _pga_debug else None

    # Step 2: Build per-event diagnostic dicts. The IQR threshold
    # is recomputed here for symmetry with the algorithm — see
    # percentile_gated_detector.py for the q3 + 2.5*IQR rule.
    if _env is not None and _env.size > 0:
        _q1, _q3 = np.percentile(_env, [25, 75])
        _iqr = _q3 - _q1
        _abs_thr = _q3 + 2.5 * _iqr
    else:
        _abs_thr = None

    # PGA STFT hop (samples) — must match the standard PGA
    # detector call (n_fft=1024, hop=256). Used to convert
    # frame-index gaps into ms. See spectral_transient_core
    # DEFAULT_STFT_PARAMS.
    _pga_hop_samples = 256
    _pga_hop_ms = _pga_hop_samples / 44100.0 * 1000.0  # ≈5.804ms

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
            if _lbases is not None and i < len(_lbases):
                ev['left_base_frame'] = int(_lbases[i])
            if _rbases is not None and i < len(_rbases):
                ev['right_base_frame'] = int(_rbases[i])
                # The gap from the peak to the right valley is
                # a candidate open/closed hihat discriminator:
                # closed hits have a tight valley close to the
                # peak; open hits have the right valley pushed
                # out by the long ring. See 2026-06-19
                # open-hihat-detection-2026-06.md.
                ev['right_base_minus_peak_frames'] = int(_rbases[i]) - p
                ev['right_base_minus_peak_ms'] = (
                    ev['right_base_minus_peak_frames'] * _pga_hop_ms
                )
            # Peak widths (2026-06-19): peak_widths(rel_height=0.9)
            # gives a tight, bounded measurement of how wide the
            # peak is. left_ips/right_ips are floating-point
            # frame indices — attack_frames is "frames from left
            # intercept to peak", decay_frames is "frames from
            # peak to right intercept". For hihat open vs
            # closed, decay_frames is the candidate
            # discriminator (open rings longer → right_ips
            # pushes further out).
            if _lips is not None and _rips is not None and i < len(_lips) and i < len(_rips):
                li = float(_lips[i])
                ri = float(_rips[i])
                ev['peak_width_left_ip_frame'] = li
                ev['peak_width_right_ip_frame'] = ri
                ev['attack_frames'] = float(p) - li
                ev['decay_frames'] = ri - float(p)
                ev['attack_ms'] = (float(p) - li) * _pga_hop_ms
                ev['decay_ms'] = (ri - float(p)) * _pga_hop_ms
        # 2026-06-26: per-event stability + combined score.
        # Same index alignment as ``prominences`` above: i is
        # the index into pga_onset_data, which corresponds to
        # the same index into the debug arrays (sorted in the
        # same order as pga_event_times). Defensive bounds
        # checks because the arrays can be empty if there are
        # no peaks.
        if _d1s is not None and i < len(_d1s):
            ev['delta1_stability'] = float(_d1s[i])
        if _d2s is not None and i < len(_d2s):
            ev['delta2_stability'] = float(_d2s[i])
        if _d5s is not None and i < len(_d5s):
            ev['delta5_stability'] = float(_d5s[i])
        if _cs is not None and i < len(_cs):
            ev['combined_score'] = float(_cs[i])
        if _abs_thr is not None:
            ev['iqr_threshold'] = float(_abs_thr)
        pga_onset_data.append(ev)

    # Step 3: MIDI velocity mapping. Linear envelope-value →
    # ``[min_velocity, max_velocity]`` from config. Per-file,
    # data-driven — no magic numbers. Equal envelope values
    # collapse to min_velocity.
    #
    # 2026-06-26: REVERTED to ``envelope_value`` basis after
    # brief experiment with ``prominence``. Prominence did
    # widen the dynamic range, but the user reported the
    # resulting velocity values were unintuitive for hihat
    # specifically (the Δ5 prominence has different perceptual
    # character than absolute envelope amplitude). Sticking
    # with envelope_value for now; the user is investigating
    # a better basis (per-stem, or dB-scale) separately.
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
    # 2026-06-18: was hardcoded to read ONLY from
    # ``toms.<key>``, which meant a ``snare.pga_min_prominence``
    # override in a project config was silently ignored —
    # snare always saw the toms value (or the global). Fixed
    # to use ``_resolve_pga_detector_param`` which honors
    # ``stem_type`` (passed in by the call site) and walks
    # the per-stem > global > default precedence. Same fix
    # applies to the pitch-detection block below.
    pga_min_prominence = _resolve_pga_detector_param(
        config, 'pga_min_prominence', 1000.0, stem_type,
    )
    min_decay_col_min_db = _resolve_pga_detector_param(
        config, 'min_decay_col_min_db', -80.0, stem_type,
    )
    pga_filter_config = {
        'pga_min_prominence': pga_min_prominence,
        'min_decay_col_min_db': min_decay_col_min_db,
        'min_velocity': midi_min,
        'max_velocity': midi_max,
    }
    for ev in pga_onset_data:
        ev['pga_filter_config'] = pga_filter_config

    # 2026-06-19: per-event feature extraction moved out of
    # ``detect_pga_events`` and into a post-filter pass
    # (``_compute_features_for_filtered_events``). Reasons:
    #   - Neighbor-dependent features (``duration_ms``,
    #     ``duration_to_valley_ms``, ``attack_rise_ms``,
    #     ``inter_onset_ms``) were bounded against the
    #     pre-filter list — a filtered-out FP between two
    #     kept strikes capped the prior strike's ring at
    #     the FP's time. The WebUI tuning panel would then
    #     show a ring that was always too short.
    #   - Re-measuring features on the final KEPT set after
    #     the filter is the only way to get post-filter
    #     neighbors. This function now returns the raw
    #     detect-time list (no features) and lets
    #     ``_build_pga_events_with_filter`` / the WebUI
    #     re-filter path own the feature pass.
    return pga_onset_data


def _walk_event_envelope(
    envelope: np.ndarray,
    peak_frame: int,
    direction: int,
    pct_threshold: float,
    stop_frames: set,
    max_frames: int,
) -> tuple[int, str, float, float, float | None]:
    """Walk a PGA contrast envelope from ``peak_frame`` in ``direction``
    (+1 forward / -1 backward) and return per-step statistics that
    discriminate open vs closed hihats.

    The walk stops at the first of:
      - envelope drops to ``pct_threshold * peak_value`` (clean ring-out
        forward, clean pre-strike silence backward),
      - a frame in ``stop_frames`` is hit (another KEPT event's peak
        frame — the next strike cuts in forward, or a prior strike is
        still ringing backward),
      - ``max_frames`` is reached (cap — typically open hihat's long
        ring forward, or short hihat attack backward).

    Returns
    -------
    frames_walked : int
        Number of frames advanced. ``0`` means we never moved.
    stop_reason : str
        ``'normal'`` (crossed the threshold), ``'hit_other_event'``
        (blocked by a neighbor KEPT event), ``'max_walk'`` (capped),
        ``'no_peak'`` (peak_value <= 0).
    avg_db_per_frame : float
        Mean per-frame dB delta across the walked window.
        Forward decay: positive number = env dropped below peak.
        Backward attack: positive number = env rose up to peak.
        Computed in log space, so high-amplitude steps near the peak
        dominate (the slope "looks steep" early, "shallow" near the
        floor). See ``avg_linear_per_frame`` for the log-free version.
    avg_linear_per_frame : float
        Mean per-frame LINEAR delta across the walked window,
        normalized to peak_value so the result is in (0, 1] for forward
        decay and [-1, 0) for backward attack. No log quirk; for
        comparison and population statistics.
    pct_at_stop : float | None
        Where we ended up, as a fraction of peak_value
        (``env[final] / peak_value``). For ``'normal'`` ≈
        ``pct_threshold``. For ``'hit_other_event'`` / ``'max_walk'``
        this is the residual — how far down (or up, backward) we got
        before being cut off. ``None`` if peak_value is zero.
    """
    n = len(envelope)
    peak_val = float(envelope[peak_frame]) if 0 <= peak_frame < n else 0.0
    if peak_val <= 0:
        return 0, "no_peak", 0.0, 0.0, None
    peak_db = 20.0 * math.log10(max(peak_val, _DB_FLOOR))

    threshold = pct_threshold * peak_val
    last_step = max_frames
    last_reason = "max_walk"
    for step in range(1, max_frames + 1):
        f_next = peak_frame + direction * step
        if f_next < 0 or f_next >= n:
            last_step = step
            last_reason = "edge"
            break
        if f_next in stop_frames:
            last_step = step
            last_reason = "hit_other_event"
            break
        v = float(envelope[f_next])
        if v <= threshold:
            last_step = step
            last_reason = "normal"
            break

    # pct_at_stop — sample at the final walked frame (clamped into
    # bounds; if last_reason == 'edge' the raw frame may be off the
    # end of the envelope).
    f_final = peak_frame + direction * last_step
    if 0 <= f_final < n:
        pct_at_stop = float(envelope[f_final] / peak_val)
    else:
        pct_at_stop = None

    # Walked window: peak_frame+direction .. peak_frame+direction*last_step.
    # Forward: delta = peak_db - v_db (positive if env dropped).
    # Backward: delta = v_db - peak_db (positive if env rose up to peak).
    # Sign-convention matches the dB-domain interpretation of
    # "how loud is the ring" / "how loud was the silence".
    deltas_db: List[float] = []
    deltas_lin: List[float] = []
    for step in range(1, last_step + 1):
        f_next = peak_frame + direction * step
        if f_next < 0 or f_next >= n:
            break
        v = float(envelope[f_next])
        v_db = 20.0 * math.log10(max(v, _DB_FLOOR))
        if direction > 0:
            deltas_db.append(peak_db - v_db)
            deltas_lin.append((peak_val - v) / peak_val)
        else:
            deltas_db.append(v_db - peak_db)
            deltas_lin.append((v - peak_val) / peak_val)
    if not deltas_db:
        return last_step, last_reason, 0.0, 0.0, pct_at_stop
    avg_db = float(sum(deltas_db) / len(deltas_db))
    avg_lin = float(sum(deltas_lin) / len(deltas_lin))
    return last_step, last_reason, avg_db, avg_lin, pct_at_stop


def _compute_walk_features_for_filtered_events(
    events: List[Dict[str, Any]],
    envelope: Optional[np.ndarray],
    sr: int,
    hop_samples: int = 256,
) -> None:
    """Attach the per-event broadband-envelope walk features to every
    event in ``events``. Mutates each event in place.

    The walk fields are the open/closed hihat discriminator:

      - ``decay_slope_db``        : mean per-frame dB drop over the
        forward walk window (positive = env dropped, larger = sharper
        decay → closed hihat). Forward walk: 50% threshold, ≤200
        frames (~1.16s) cap, stops at next KEPT event.
      - ``decay_slope_linear``    : same as above but in linear units
        (peak-fraction lost per frame, in (0, 1]). No log quirk.
      - ``decay_frames_walked``   : frames advanced before stopping.
        Always populated; the longest walks (200 frames) are open hihats
        that get blocked by a neighbor KEPT event or hit the cap.
      - ``decay_pct_at_stop``     : envelope value at the stop frame
        as a fraction of peak. Open hihats typically sit at 0.7-1.0+
        (next hit cut in); closed hihats reliably cross to ~0.49.
      - ``decay_stop_reason``     : ``'normal'``, ``'hit_other_event'``,
        ``'max_walk'``, ``'edge'``, ``'no_peak'``.
      - ``onset_crossed``         : True iff the backward walk reached
        ``ONSET_PCT_THRESHOLD`` of peak within ``ONSET_MAX_FRAMES``
        (~35ms). True → clean pre-strike silence (typical of a closed
        hihat). False → sitting on top of a prior hit's ring or the
        strike itself is loud enough that the envelope doesn't dip.
      - ``onset_cross_ms``        : ms-from-peak to the 20% crossing
        point (negative direction, reported as magnitude). ``None``
        if the walk did not cross.

    The walk is run against the post-filter KEPT event set so that
    ``decay_stop_reason`` and ``decay_pct_at_stop`` reflect what the
    user sees after filtering, not the pre-filter list. FILTERED
    events also get the fields attached (the diagnostic surface in
    the WebUI shows them too) but their ``decay_stop_reason`` walks
    against the same set — the FILTERED status doesn't change which
    frames are valid stopping points, only which events get emitted.

    ``envelope`` may be None if the detector was run without
    producing a debug dict (defensive — the detector always populates
    it in practice). When None, every event gets ``None`` fields
    and the classifier falls back to the existing geomean+sustain rule.

    Pure function. No file I/O. No mutation of input audio.
    """
    if not events or envelope is None or len(envelope) == 0:
        for ev in events:
            ev['decay_slope_db'] = None
            ev['decay_slope_linear'] = None
            ev['decay_frames_walked'] = None
            ev['decay_pct_at_stop'] = None
            ev['decay_stop_reason'] = None
            ev['onset_crossed'] = False
            ev['onset_cross_ms'] = None
        return

    # Build the set of KEPT peak frames once. Used by both walks as
    # the neighbor-event stop set (every other KEPT peak frame is
    # a valid cut-off point). We exclude the event's own frame so
    # the walk doesn't immediately stop on itself.
    kept_frames: Set[int] = set()
    for ev in events:
        if ev.get('status') != 'FILTERED':
            f = ev.get('frame')
            if isinstance(f, (int, np.integer)):
                kept_frames.add(int(f))

    hop_ms = hop_samples / float(sr) * 1000.0

    for ev in events:
        peak_frame = ev.get('frame')
        if not isinstance(peak_frame, (int, np.integer)):
            # No frame — this event came from a detector that
            # doesn't expose STFT frames. Skip the walk; the
            # classifier will fall back to geomean+sustain.
            ev['decay_slope_db'] = None
            ev['decay_slope_linear'] = None
            ev['decay_frames_walked'] = None
            ev['decay_pct_at_stop'] = None
            ev['decay_stop_reason'] = None
            ev['onset_crossed'] = False
            ev['onset_cross_ms'] = None
            continue
        peak_frame = int(peak_frame)

        # Forward (decay) walk: stop at any OTHER KEPT event.
        stop = kept_frames - {peak_frame}
        dec_n, dec_reason, dec_db, dec_lin, dec_pct = _walk_event_envelope(
            envelope, peak_frame, +1,
            DECAY_PCT_THRESHOLD, stop, DECAY_MAX_FRAMES,
        )

        # Backward (onset) walk: same neighbor set. Onset uses
        # min_frames=1 semantically (we allow crossing at step 1)
        # because the 20% threshold with a 6-frame cap is already
        # the discriminator — min_frames should not suppress
        # crossings at step=1. The walk helper itself doesn't
        # enforce min_frames (the threshold does the gating).
        onset_n, onset_reason, _onset_db, _onset_lin, onset_pct = (
            _walk_event_envelope(
                envelope, peak_frame, -1,
                ONSET_PCT_THRESHOLD, stop, ONSET_MAX_FRAMES,
            )
        )

        ev['decay_slope_db'] = round(dec_db, 4) if dec_reason != 'no_peak' else None
        ev['decay_slope_linear'] = round(dec_lin, 4) if dec_reason != 'no_peak' else None
        ev['decay_frames_walked'] = dec_n
        ev['decay_pct_at_stop'] = round(dec_pct, 4) if dec_pct is not None else None
        ev['decay_stop_reason'] = dec_reason
        ev['onset_crossed'] = (onset_reason == 'normal')
        ev['onset_cross_ms'] = (
            round(onset_n * hop_ms, 2) if onset_reason == 'normal' else None
        )
        ev['onset_cross_ms'] = (
            round(onset_n * hop_ms, 2) if onset_reason == 'normal' else None
        )


def _find_prev_next_kept(
    events: List[Dict[str, Any]],
    i: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Find the previous and next non-FILTERED event times
    around index ``i`` in ``events``.

    Used to bound the ``duration_ms`` / ``duration_to_valley_ms``
    / ``attack_rise_ms`` / ``inter_onset_ms`` features to the
    actual KEPT-event neighborhood — without this skip, a
    filtered-out FP between two strikes would truncate the prior
    strike's ring at the FP's time and stretch the next strike's
    attack across the gap.

    Returns ``(prev_t, next_t)``; each is ``None`` for the first
    / last event (or when no kept neighbor exists in that
    direction).
    """
    prev_t: Optional[float] = None
    for j in range(i - 1, -1, -1):
        candidate = events[j]
        if candidate.get('status') != 'FILTERED':
            prev_t = candidate.get('time')
            break
    next_t: Optional[float] = None
    for j in range(i + 1, len(events)):
        candidate = events[j]
        if candidate.get('status') != 'FILTERED':
            next_t = candidate.get('time')
            break
    return prev_t, next_t


def _compute_features_for_filtered_events(
    events: List[Dict[str, Any]],
    audio_mono: np.ndarray,
    sr: int,
    config: Dict[str, Any],
    stem_type: Optional[str],
) -> None:
    """Compute per-event features for events that have already
    been through the filter step.

    Mutates each event in-place by adding the per-event feature
    keys (``duration_ms``, ``duration_to_valley_ms``,
    ``attack_rise_ms``, ``pitch_hz``, ...). Neighbor-dependent
    features (``duration_*``, ``attack_rise_ms``,
    ``inter_onset_ms``) are bounded to the *kept* event
    neighborhood via :func:`_find_prev_next_kept` — so an FP
    that was just dropped by the filter no longer truncates the
    prior strike's ring at the FP's time.

    This is the post-filter feature pass that fixes the
    "duration was capped at the filtered event" bug (the
    pre-filter list was used as neighbors before this pass
    existed, so a filtered event between two kept strikes
    silently capped the prior strike's ``duration_ms`` at the
    filtered event's time).

    Per-event exceptions are swallowed: a bad event shouldn't
    poison the rest of the pipeline. The WebUI shows "N/A" for
    the feature on that event.
    """
    if not events:
        return
    # Lazy import: compute_event_features pulls in librosa /
    # scipy stack and is not on the cold path.
    from .event_features import compute_event_features
    # Read pitch config once (not per-event). These keys are
    # declared in the YAML under each stem section (e.g.
    # ``toms.enable_pitch_detection``, ``toms.pitch_method``,
    # ``toms.min_pitch_hz``, ``toms.max_pitch_hz``). Defaults
    # match the user's toms config — YIN (5-10× faster than
    # pYIN), 60-250Hz search range (toms fundamentals).
    # 2026-06-18: was hardcoded to read ONLY from the
    # ``toms`` section, so a snare stem would silently
    # see toms pitch config. Fixed to use the call
    # site's ``stem_type`` for per-stem overrides.
    stem_cfg = config.get(stem_type, {}) or {}
    enable_pitch_detection = bool(
        stem_cfg.get('enable_pitch_detection', True)
    )
    pitch_method = stem_cfg.get('pitch_method', 'yin')
    pitch_fmin_hz = float(
        _resolve_pga_detector_param(
            config, 'min_pitch_hz', 60.0, stem_type,
        )
    )
    pitch_fmax_hz = float(
        _resolve_pga_detector_param(
            config, 'max_pitch_hz', 250.0, stem_type,
        )
    )
    for i, ev in enumerate(events):
        # 2026-06-18: prev_event lookup added so
        # ``attack_rise_ms`` is bounded by the previous
        # event's time — without it, a ringing previous
        # hit keeps the envelope above 10% of the new
        # peak all the way back into the previous hit's
        # body, producing ``attack_rise_ms`` ≈
        # ``inter_onset_ms`` on snare / dense hihats
        # (see bug-tracking.md "attack_rise_ms unbounded
        # by previous event").
        prev_t, next_t = _find_prev_next_kept(events, i)
        try:
            feats = compute_event_features(
                audio_mono, sr, ev['time'],
                enable_pitch_detection=enable_pitch_detection,
                pitch_method=pitch_method,
                pitch_fmin_hz=pitch_fmin_hz,
                pitch_fmax_hz=pitch_fmax_hz,
                next_event_time_sec=next_t,
                prev_event_time_sec=prev_t,
            )
        except Exception:
            # Defensive: a bad event shouldn't poison the
            # rest of the pipeline. The diagnostic surface
            # in the WebUI will show "N/A" for features on
            # this event.
            feats = {
                'duration_ms': None,
                'duration_to_valley_ms': None,
                'attack_rise_ms': None,
                'pitch_hz': None,
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


def apply_attack_rise_max_filter(
    events: List[Dict[str, Any]],
    threshold: float,
    disabled_ids: Optional[Set[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-tag PGA events with status='FILTERED' based on the
    attack_rise_ms diagnostic (2026-06-17).

    Third pass of the PGA filter chain (after prominence and
    decay_col_min). Catches wire-tail / step-back FPs that
    have real-looking attack (high prominence, good ring
    quality) but the high-res STFT detects they 'step back'
    to a previous attack — producing an unusually long
    10-90% rise time. Default 20 ms (rational cut between
    real strikes at 11-18 ms and FPs at 100-500 ms on
    project 6 Taylor Swift toms).

    Pure function: returns ``(kept, filtered)``. Mirrors
    :func:`apply_pga_prominence_filter` and
    :func:`apply_pga_decay_col_min_filter` — same registry-
    driven pattern. The filter spec lives in
    ``stems_to_midi/filter_registry.json`` under the
    ``attack_rise_max_ms`` entry.

    Layering: call this on the events that PASSED both
    prominence and decay_col_min — passing the full events
    list would overwrite the prior filters' FILTERED status
    with KEPT for events that pass attack_rise_ms (the
    composition bug 2026-06-17 was about this exact issue).
    """
    return _apply_pga_filter(
        events, find_filter('attack_rise_max_ms'), threshold, disabled_ids,
    )


def apply_pga_min_envelope_value(
    events: List[Dict[str, Any]],
    threshold: float,
    disabled_ids: Optional[Set[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-tag PGA events with status='FILTERED' based on
    the envelope_value diagnostic (2026-06-17).

    Registry-driven wrapper (filter kind=min_value).
    Reads the filter spec from
    ``stems_to_midi/filter_registry.json`` under the
    ``pga_min_envelope_value`` entry; the predicate is evaluated by
    the shared :func:`evaluate_filter` in
    :mod:`stems_to_midi.filter_kinds`. Mirrors the
    pattern of :func:`apply_pga_prominence_filter`,
    :func:`apply_pga_decay_col_min_filter`, and
    :func:`apply_attack_rise_max_filter` — same
    `_apply_pga_filter` helper.

    Returns ``(kept, filtered)``. Layered composition:
    pass the events that PASSED the previous filter,
    not the full events list — otherwise this filter
    overwrites the previous filter's FILTERED status
    with KEPT (the 2026-06-17 composition bug).
    """
    return _apply_pga_filter(
        events, find_filter('pga_min_envelope_value'), threshold, disabled_ids,
    )


def apply_pga_min_combined_score(
    events: List[Dict[str, Any]],
    threshold: float,
    disabled_ids: Optional[Set[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Warble filter (2026-06-26): drop events whose combined_score
    is below the threshold. combined_score = prominence ×
    delta5_stability — sign-bearing. Positive means a real
    sustained strike; negative means a warble spike (single-
    frame Δ5 transient, no sustained rise). On Metallica hihat
    (project 10): threshold 0 = 100% precision separator
    (528 FPs below 0, 225 real hits above 0). Default 0.

    Sister filter to :func:`apply_pga_prominence_filter` and
    :func:`apply_pga_min_envelope_value`. Same registry-driven
    pattern (filter kind=min_value, field=combined_score,
    reason_template etc.). Mirrors the layered composition: pass
    the events that PASSED the previous filter, not the full
    list, to avoid overwriting the previous filter's status.

    Use cases:
      - In place of pga_min_envelope_value for stems where
        the envelope filter is killing real-but-quiet onsets.
      - In combination with pga_min_prominence — both can
        run, since combined_score already includes prominence
        and the warble dimension adds independent information.
    """
    return _apply_pga_filter(
        events, find_filter('pga_min_combined_score'), threshold, disabled_ids,
    )

def _build_pga_events_with_filter(
    audio_mono: np.ndarray,
    sr: int,
    config: Dict[str, Any],
    stem_type: Optional[str] = None,
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

    # 2026-06-18: pass the max_floor_gate_db cap. See
    # detect_pga_events for the rationale.
    max_floor_gate_db = _resolve_max_floor_gate_db(config, stem_type)
    abs_envelope_threshold = _resolve_pga_abs_envelope_threshold(config, stem_type)
    broad_freq_min_hz = _resolve_pga_detector_param(
        config, 'pga_broad_freq_min_hz', DEFAULT_BROAD_FREQ_MIN_HZ, stem_type,
    )
    broad_freq_max_hz = _resolve_pga_detector_param(
        config, 'pga_broad_freq_max_hz', DEFAULT_BROAD_FREQ_MAX_HZ, stem_type,
    )
    db_rise_threshold = _resolve_pga_detector_param(
        config, 'pga_db_rise_threshold', DEFAULT_DB_RISE_THRESHOLD, stem_type,
    )
    nms_min_frames = _resolve_pga_detector_param(
        config, 'pga_nms_min_frames', DEFAULT_NMS_MIN_FRAMES, stem_type,
    )
    strike_offset_sec = _resolve_pga_detector_param(
        config, 'pga_strike_offset_sec', DEFAULT_STRIKE_OFFSET_SEC, stem_type,
    )
    _pga_event_times, pga_debug = detect_percentile_gated_broad_attacks(
        audio_mono, sr,
        broad_freq_min_hz=broad_freq_min_hz,
        broad_freq_max_hz=broad_freq_max_hz,
        db_rise_threshold=db_rise_threshold,
        abs_envelope_threshold=abs_envelope_threshold,
        nms_min_frames=nms_min_frames,
        strike_offset_sec=strike_offset_sec,
        max_floor_gate_db=max_floor_gate_db,
    )

    raw = detect_pga_events(audio_mono, sr, config, stem_type=stem_type)
    # 2026-06-29: TEST-only — hihat openness score (stamped BEFORE
    # the filter chain so KEPT and FILTERED events both carry it,
    # letting the user see how noise / FPs score on the openness
    # axis). Stamps `hihat_openness_score` (float in [0, 1]) on
    # every event that has a 'frame' field. Recomputes the mel-spec
    # internally — duplicative with the KMeans classifier's mel-spec
    # (which runs in classify_hihat_by_kmeans downstream), but the
    # classifier only stamps KEPT events so this can't be reused.
    # Future: cache the mel-spec in pga_debug and pass it through
    # to avoid the duplicate compute — only worth doing once the
    # test confirms the score is viable.
    if stem_type == 'hihat':
        stamp_hihat_openness_score(raw, audio_mono, sr)
    # 2026-06-22: envelope_value filter (Pass 0.4). Sister
    # to the prominence filter but uses the linear
    # envelope_value at the peak frame (set by
    # detect_pga_events on every event) instead of the
    # scipy prominence. Runs BEFORE the prominence
    # filter so low-energy FPs are dropped first, then
    # the relative-prominence comparison culls the
    # remaining noise. Same per-stem > global > default
    # resolution pattern.
    envelope_value_threshold = _resolve_pga_detector_param(
        config, 'pga_min_envelope_value', 1000.0, stem_type,
    )
    envelope_value_threshold = float(envelope_value_threshold)
    events_kept, envelope_value_filtered = apply_pga_min_envelope_value(
        raw, envelope_value_threshold,
    )
    events_filtered = list(envelope_value_filtered)
    # Apply the PGA prominence filter (existing behavior).
    # 2026-06-15: per-stem override (toms.pga_min_prominence)
    # wins over the global onset_detection.pga_min_prominence.
    # 2026-06-18: was hardcoded to read ONLY from the
    # ``toms`` section. Now uses the call site's stem_type
    # via the resolver.
    pga_min_prominence = _resolve_pga_detector_param(
        config, 'pga_min_prominence', 1000.0, stem_type,
    )
    prom_threshold = float(pga_min_prominence)
    events_kept, prom_filtered = apply_pga_prominence_filter(
        events_kept, prom_threshold,
    )
    events_filtered = events_filtered + prom_filtered
    # 2026-06-15: apply the decay_col_min filter on top of the
    # prominence filter. Same per-stem > global > default
    # resolution pattern. Default -80.0 dB matches the empirical
    # split (real strikes -60 to -84 dB, noise pops -84 to -90 dB).
    # 2026-06-18: was hardcoded to read ONLY from the
    # ``toms`` section. Now uses the call site's stem_type
    # via the resolver.
    decay_col_min_threshold = _resolve_pga_detector_param(
        config, 'min_decay_col_min_db', -80.0, stem_type,
    )
    events_kept, decay_filtered = apply_pga_decay_col_min_filter(
        events_kept, decay_col_min_threshold,
    )
    # Concatenate the two filtered lists. The decay_col_min
    # filter is downstream of the prominence filter, so events
    # it filters were already KEPT (i.e., they had high
    # prominence) but failed the ring-quality check.
    events_filtered = events_filtered + decay_filtered
    # 2026-06-17: attack_rise filter (third PGA pass). Catches
    # wire-tail / step-back FPs that pass prominence +
    # decay_col_min but have an unusually long 10-90% rise
    # time. Layered on top of the previous filters; events
    # passing both are KEPT, events failing this are
    # FILTERED.
    # 2026-06-18: was hardcoded to read ONLY from the
    # ``toms`` section. Now uses the call site's stem_type
    # via the resolver.
    attack_rise_threshold = _resolve_pga_detector_param(
        config, 'attack_rise_max_ms', 20.0, stem_type,
    )
    events_kept, attack_filtered = apply_attack_rise_max_filter(
        events_kept, attack_rise_threshold,
    )
    events_filtered = events_filtered + attack_filtered
    # 2026-06-26: warble filter. Drops events whose combined_score
    # (= prominence × delta5_stability) is below the threshold.
    # Sign-bearing: positive = real sustained strike, negative =
    # warble spike from stem-splitter demuxing. Default 0 keeps
    # all positive (real) hits and drops all negative (FP) hits
    # per the data explored on the cymbals and hihat stems; the
    # per-stem key is pga_min_combined_score in midiconfig.
    combined_score_threshold = _resolve_pga_detector_param(
        config, 'pga_min_combined_score', 0.0, stem_type,
    )
    events_kept, cs_filtered = apply_pga_min_combined_score(
        events_kept, combined_score_threshold,
    )
    events_filtered = events_filtered + cs_filtered
    # Record the active thresholds in pga_filter_config so the
    # sidecar tooltip can show what filter the event was
    # processed under.
    for ev in raw:
        pga_filter_config = dict(ev.get('pga_filter_config', {}))
        pga_filter_config['pga_min_envelope_value'] = envelope_value_threshold
        pga_filter_config['pga_min_prominence'] = prom_threshold
        pga_filter_config['min_decay_col_min_db'] = decay_col_min_threshold
        pga_filter_config['attack_rise_max_ms'] = attack_rise_threshold
        pga_filter_config['pga_min_combined_score'] = combined_score_threshold
        ev['pga_filter_config'] = pga_filter_config
    # 2026-06-19: post-filter feature pass. Now that the final
    # KEPT/FILTERED partition is set, compute per-event features
    # against the post-filter neighbor set (so ``duration_ms``,
    # ``duration_to_valley_ms``, ``attack_rise_ms``, and
    # ``inter_onset_ms`` reflect the KEPT event set — a
    # filtered-out FP no longer caps the prior strike's ring).
    # Features are attached to ALL events in ``raw`` (both KEPT
    # and FILTERED); the FILTERED list goes into the sidecar's
    # diagnostic surface so the WebUI can show "why this was
    # dropped" with the actual feature values, not None.
    _compute_features_for_filtered_events(
        raw, audio_mono, sr, config, stem_type,
    )
    # 2026-06-19: per-event broadband-envelope walk. See
    # build_pga_events above for rationale.
    _envelope = pga_debug.get('envelope') if pga_debug else None
    _compute_walk_features_for_filtered_events(
        raw, _envelope, sr, hop_samples=256,
    )
    return raw, events_kept, events_filtered, pga_debug
