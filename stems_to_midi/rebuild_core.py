"""
Rebuild MIDI from Analysis — PGA-only

2026-06-22: rewritten as a PGA-only surface. The legacy
energy/spectral branch (geomean / sustain / strength / reverb
continuation / snap-mask / band_max_ratio / sensitive-merge
filter chain) was deleted along with the modules that
implemented it (``analysis_core.spectral_utils``,
``analysis_core.onset_filtering``, ``analysis_core.threshold_learning``,
``analysis_core.classification``, ``stems_to_midi.learning``,
``stems_to_midi.optimization_core``,
``stems_to_midi.energy_detection_core``,
``stems_to_midi.energy_detection_shell``,
``stems_to_midi.detection_shell``). Stems that previously used
the energy/spectral pipeline now go through the PGA path
exclusively; the only way to opt in to PGA is
``<stem>.use_pga_detection: true`` in the project's midiconfig
(``process_stem_to_midi`` was already a thin shim that
short-circuits to ``process_percentile_gated`` for any stem
with that flag set).

Public entry point: :func:`rebuild_events_from_analysis`.
"""

from typing import Any, Dict, List, Optional, Tuple

from .config import DrumMapping
from .note_classification_core import classify_notes


def _classification_thresholds_changed(
    config: Dict,
    stem_type: str,
) -> bool:
    """
    Determine whether any classification threshold (e.g. hihat
    ``open_decay_slope_max``, cluster ``expected_clusters``,
    per-class ``midi_note_*``) has been touched by the user
    since the last conversion.

    The Pass 2 classifier (``note_classification_core.classify_notes``)
    preserves stored ``hihat_state`` / ``classification`` to avoid
    silently flipping a previously-classified event when nothing
    about its inputs has changed. When the user moves a
    classification slider, we need to force reclassification so the
    new threshold takes effect.

    The PGA-only path has no sidecar logic block, so we can't
    diff against the previously-stored values. Instead we return
    True whenever ANY classification threshold is configured for
    the stem in yaml. The rebuild will re-apply the current rule
    from scratch, which is the correct behavior for a config
    change. The classification is cheap (a per-event rule check
    on the kept set).
    """
    stem_config = config.get(stem_type, {})

    # Hihat: open_decay_slope_max is the live classifier.
    if stem_type == 'hihat':
        if stem_config.get('open_decay_slope_max') is not None:
            return True
        return False

    # Snare/toms/cymbals: cluster count and per-class MIDI notes
    # are the live classification inputs. Any change forces
    # reclassification.
    classification_keys = (
        'expected_clusters', 'cluster_feature', 'cluster_note_map',
    )
    if any(stem_config.get(k) is not None for k in classification_keys):
        return True
    if any(k.startswith('midi_note_') for k in stem_config):
        return True

    return False


def _format_time_key(t: float) -> str:
    """Format a song time as the 4-decimal key used by overrides."""
    return f"{t:.4f}"


def _apply_overrides(
    events: List[Dict],
    overrides: Dict[str, Dict[str, Any]],
) -> List[Dict]:
    """
    Apply manual overrides to event statuses (legacy path).

    Override keys are time strings rounded to 4 decimals. Each
    override value is a record: ``{ status: "KEPT"|"FILTERED",
    [classification]: int }`` — ``status`` is required;
    ``classification`` is optional. Mutates and returns the same
    list.

    The modern rebuild path uses ``_move_overridden_events``
    instead, which is a post-filter veto on the KEPT/FILTERED
    split (so the override can un-FILTER events the filter
    dropped — this function only mutates events in the input
    list, which the legacy call site passed as ``pga_kept``).
    This function is kept for the no-PGA fallback path in
    ``rebuild_events_from_analysis`` (legacy projects that
    only have events_configured, no events_pga).
    """
    if not overrides:
        return events

    for event in events:
        time_key = _format_time_key(event['time'])
        override = overrides.get(time_key)
        if not override:
            continue
        event['status'] = override['status']
        if override.get('classification') is not None:
            event['classification'] = override['classification']
        event['override'] = True

    return events


def _move_overridden_events(
    pga_kept: List[Dict],
    pga_filtered: List[Dict],
    overrides: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Treat the user's override as a post-filter veto on the event's
    KEPT/FILTERED status. The PGA prominence filter ran first
    and split events into ``pga_kept`` (passed) and ``pga_filtered``
    (dropped). If the user has clicked an event to a state that
    disagrees with the filter's decision, we move the event to
    the matching list. This is the fix for the "MIDI has the
    FILTERED note anyway" bug — the override's KEPT wins over
    the filter's FILTERED (and vice versa).

    Args:
        pga_kept: Events that passed the filter.
        pga_filtered: Events that the filter dropped.
        overrides: The full per-stem override dict, keyed by
            time string.

    Returns:
        Tuple of (new_pga_kept, new_pga_filtered) — both new
        lists, with events moved between them per the override.
        The override's status is applied to the moved event
        (and to events that already match the override's
        status, for consistency).
    """
    if not overrides:
        return pga_kept, pga_filtered

    new_pga_kept = []
    new_pga_filtered = []

    for event in pga_kept:
        time_key = f"{event['time']:.4f}"
        override = overrides.get(time_key)
        if override is None:
            new_pga_kept.append(event)
            continue
        new_status = override.get('status')
        if new_status == 'FILTERED':
            # Filter let it through, override says drop. Move
            # to filtered, and apply the override's
            # classification (if any) so downstream sees it.
            event['status'] = 'FILTERED'
            if override.get('classification') is not None:
                event['classification'] = override['classification']
            new_pga_filtered.append(event)
        else:
            # Filter and override agree (KEPT) or override has
            # no status. Keep in pga_kept; apply the
            # classification so the override's class wins.
            if override.get('classification') is not None:
                event['classification'] = override['classification']
            new_pga_kept.append(event)

    for event in pga_filtered:
        time_key = f"{event['time']:.4f}"
        override = overrides.get(time_key)
        if override is None:
            new_pga_filtered.append(event)
            continue
        new_status = override.get('status')
        if new_status == 'KEPT':
            # Filter dropped it, override says keep. This is
            # the bug fix — move to kept, apply override.
            event['status'] = 'KEPT'
            if override.get('classification') is not None:
                event['classification'] = override['classification']
            new_pga_kept.append(event)
        else:
            new_pga_filtered.append(event)

    return new_pga_kept, new_pga_filtered


def rebuild_events_from_analysis(
    analysis_data: Dict,
    overrides: Dict[str, Dict[str, str]],
    config: Dict,
    stem_types: Optional[List[str]] = None,
) -> Tuple[Dict, Dict[str, List[Dict]]]:
    """
    Re-filter and rebuild MIDI events from cached analysis.json data
    (PGA-only).

    This is the primary entry point for the rebuild-from-analysis
    pipeline. It replaces the full detection pipeline when the
    user moves a PGA filter slider or a manual override changes.

    For every stem with ``events_pga`` in the sidecar, the PGA
    prominence filter is re-applied from scratch using the
    threshold from yaml (per-stem > global > 1000.0). Manual
    overrides from ``event_overrides.json`` are applied after
    filtering. The kept set is passed through
    :func:`note_classification_core.classify_notes` (hihat
    open/closed, cluster-based note assignment) and then turned
    into MIDI events.

    Args:
        analysis_data: Parsed analysis.json dict (v3 format).
        overrides: Per-stem manual overrides from
            event_overrides.json. Format: ``{stem_type: {time_key:
            'KEPT'|'FILTERED'}}``.
        config: Parsed midiconfig.yaml dict with current
            thresholds.
        stem_types: Optional list of stems to rebuild (None = all
            stems present in analysis_data).

    Returns:
        Tuple of:
        - updated_analysis: Copy of analysis_data with event
          statuses reflecting the new filter results. Overridden
          events carry an 'override' flag.
        - midi_events_by_stem: Dict mapping stem_type to lists of
          MIDI event dicts ready for ``create_midi_file()``.

    Raises:
        ValueError: If analysis_data is missing, wrong version, or
            has no stems.
    """
    if not analysis_data:
        raise ValueError("No analysis data provided")

    version = analysis_data.get('version', '')
    if not version.startswith('3'):
        raise ValueError(
            f"Analysis data version '{version}' is not supported. "
            f"Re-run full detection to generate v3 format."
        )

    stems_data = analysis_data.get('stems', {})
    if not stems_data:
        raise ValueError("Analysis data contains no stem data")

    available_stems = list(stems_data.keys())
    if stem_types is None:
        stem_types = available_stems
    else:
        missing = set(stem_types) - set(available_stems)
        if missing:
            raise ValueError(f"Stems not in analysis data: {missing}")

    drum_mapping = DrumMapping.from_config(config)

    # Deep copy the stems section so we don't mutate the input
    import copy
    updated_stems = copy.deepcopy(stems_data)

    midi_events_by_stem = {}

    for stem_type in stem_types:
        stem_data = updated_stems[stem_type]
        raw_pga_events = list(stem_data.get('events_pga', []))
        pga_rebuild_active = len(raw_pga_events) > 0
        stem_overrides = overrides.get(stem_type, {})

        # No events_pga in the sidecar — this stem never went
        # through the PGA pipeline (legacy project that pre-dates
        # PGA-universal, or the sidecar was built by the energy
        # pipeline). Skip it; the user must re-run full detection
        # to populate events_pga. Trusting the empty
        # events_configured would silently produce a no-MIDI
        # result, which is the bug we just fixed at the
        # spectral_config level.
        if not pga_rebuild_active:
            # Apply overrides to whatever configured events the
            # sidecar carries, so the user-visible statuses
            # reflect their WebUI toggles. Don't try to build
            # MIDI events (the legacy filter chain is gone).
            configured = stem_data.get('events_configured', [])
            if configured and stem_overrides:
                _apply_overrides(configured, stem_overrides)
            continue

        # ----------------------------------------------------------------
        # PGA prominence re-filter (2026-06-15, generalized 2026-06-18).
        # events_pga carries ALL detected events (status='KEPT' at
        # detect time, no filter applied). We re-apply the
        # prominence filter here using the threshold from yaml
        # (per-stem > global > 1000.0 hard floor), so the slider
        # value lands regardless of what the original detection
        # pass stamped.
        # ----------------------------------------------------------------
        stem_pga = config.get(stem_type, {}).get('pga_min_prominence')
        global_pga = config.get('onset_detection', {}).get('pga_min_prominence')
        pga_threshold = float(
            stem_pga if stem_pga is not None
            else global_pga if global_pga is not None
            else 1000.0
        )

        # Optional second + third PGA passes — only when the user
        # has configured thresholds. Mirrors the WebUI panel
        # behavior: no slider exposed → no filter applied.
        # 2026-06-22: envelope_value filter (Pass 0.4) runs
        # BEFORE prominence in the chain. envelope_value
        # measures the absolute height of the peak in the
        # broadband contrast envelope, prominence measures
        # the peak's vertical distance to the local contour.
        # envelope_value is the "is this a real strike at
        # all" test; prominence is the "is this a clean
        # isolated strike" test. Running envelope_value
        # first culls low-energy FPs before the more
        # expensive relative comparison.
        stem_envelope_value = config.get(stem_type, {}).get('pga_min_envelope_value')
        global_envelope_value = config.get('onset_detection', {}).get('pga_min_envelope_value')
        envelope_value_threshold = (
            float(stem_envelope_value) if stem_envelope_value is not None
            else float(global_envelope_value) if global_envelope_value is not None
            else None
        )
        stem_col_min = config.get(stem_type, {}).get('min_decay_col_min_db')
        global_col_min = config.get('onset_detection', {}).get('min_decay_col_min_db')
        col_min_threshold = (
            float(stem_col_min) if stem_col_min is not None
            else float(global_col_min) if global_col_min is not None
            else None
        )
        stem_attack_rise = config.get(stem_type, {}).get('attack_rise_max_ms')
        global_attack_rise = config.get('onset_detection', {}).get('attack_rise_max_ms')
        attack_rise_threshold = (
            float(stem_attack_rise) if stem_attack_rise is not None
            else float(global_attack_rise) if global_attack_rise is not None
            else None
        )

        from .pga_event_builder import (
            apply_pga_prominence_filter,
            apply_pga_decay_col_min_filter,
            apply_attack_rise_max_filter,
            apply_pga_min_envelope_value,
            apply_pga_min_combined_score,
        )
        # Pass 0.4: envelope_value filter. Runs first in
        # the chain (before prominence) so low-energy
        # FPs are dropped before the relative-prominence
        # comparison. Mirrors the JS applyTuningFilter
        # chain order.
        if envelope_value_threshold is not None:
            pga_kept, envelope_value_filtered = apply_pga_min_envelope_value(
                raw_pga_events,
                envelope_value_threshold,
            )
            pga_filtered = envelope_value_filtered
        else:
            pga_kept = list(raw_pga_events)
            pga_filtered = []
        # Pass 0.5: prominence filter. Layered on the
        # kept list from Pass 0.4 (envelope_value), not
        # the raw events — otherwise this filter would
        # overwrite envelope_value's FILTERED status
        # with KEPT (the 2026-06-17 composition bug).
        pga_kept, prominence_filtered = apply_pga_prominence_filter(
            pga_kept,
            pga_threshold,
        )
        pga_filtered = pga_filtered + prominence_filtered
        if col_min_threshold is not None:
            pga_kept, col_min_filtered = apply_pga_decay_col_min_filter(
                pga_kept,
                col_min_threshold,
            )
            pga_filtered = pga_filtered + col_min_filtered
        if attack_rise_threshold is not None:
            pga_kept, attack_filtered = apply_attack_rise_max_filter(
                pga_kept,
                attack_rise_threshold,
            )
            pga_filtered = pga_filtered + attack_filtered
        # 2026-06-26: warble filter. Drops events whose
        # combined_score (prominence × delta5_stability) is
        # below the threshold. This is the last filter in the
        # PGA chain because it has a sign-bearing signature:
        # positive = real sustained strike, negative = warble
        # spike. Mirrors the Python pipeline chain order
        # so the server result matches what the user sees in
        # the tuning panel. Per-stem > global > 0.0 default.
        stem_combined_score = config.get(stem_type, {}).get('pga_min_combined_score')
        global_combined_score = config.get('onset_detection', {}).get('pga_min_combined_score')
        combined_score_threshold = (
            float(stem_combined_score) if stem_combined_score is not None
            else float(global_combined_score) if global_combined_score is not None
            else 0.0
        )
        pga_kept, cs_filtered = apply_pga_min_combined_score(
            pga_kept,
            combined_score_threshold,
        )
        pga_filtered = pga_filtered + cs_filtered

        # 2026-06-30: treat the override as a POST-FILTER VETO on
        # status. The PGA filter chain ran above and split events
        # into pga_kept (passed) and pga_filtered (dropped). The
        # override wins: an event with override.status='KEPT' that
        # the filter dropped gets moved to pga_kept; an event with
        # override.status='FILTERED' that the filter kept gets
        # moved to pga_filtered. This is the fix for the "MIDI
        # has the FILTERED note anyway" bug — the user explicitly
        # clicked the event to a state, and the filter must not
        # silently override that decision.
        #
        # The override's classification (if set) is applied here
        # too, so the downstream MIDI loop picks it up via the
        # existing _map_note path.
        pga_kept, pga_filtered = _move_overridden_events(
            pga_kept, pga_filtered, stem_overrides,
        )

        # Stamp filter_reason on the filtered events for the sidecar.
        # Skip events the user has overridden — their status is
        # already correct (from _move_overridden_events above), and
        # the filter_reason would be misleading.
        kept_times = {round(e['time'], 4) for e in pga_kept}
        filtered_times = {round(e['time'], 4) for e in pga_filtered}
        override_time_keys = {
            round(float(t), 4) for t in (stem_overrides or {}).keys()
        }
        for ev in raw_pga_events:
            t = round(ev['time'], 4)
            if t in override_time_keys:
                # User override. The status was set correctly
                # by _move_overridden_events (or by the override
                # record's `status` field). Drop any existing
                # filter_reason — the override is the new "why".
                ev['status'] = (
                    stem_overrides[_format_time_key(t)]['status']
                )
                ev.pop('filter_reason', None)
                continue
            if t in kept_times:
                ev['status'] = 'KEPT'
                ev.pop('filter_reason', None)
            elif t in filtered_times:
                ev['status'] = 'FILTERED'
                existing_reason = ev.get('filter_reason', '')
                if (
                    'min_decay_col_min_db' not in existing_reason
                    and 'attack_rise_max_ms' not in existing_reason
                    and 'pga_min_envelope_value' not in existing_reason
                ):
                    prom = ev.get('prominence')
                    ev['filter_reason'] = (
                        f"below pga_min_prominence ({prom:.0f} < {pga_threshold:.0f})"
                        if prom is not None else 'below pga_min_prominence'
                    )
            # Reflect the active filter thresholds in the sidecar
            ev['pga_filter_config'] = dict(ev.get('pga_filter_config', {}))
            ev['pga_filter_config']['pga_min_prominence'] = pga_threshold
            if envelope_value_threshold is not None:
                ev['pga_filter_config']['pga_min_envelope_value'] = envelope_value_threshold
            if col_min_threshold is not None:
                ev['pga_filter_config']['min_decay_col_min_db'] = col_min_threshold
            if attack_rise_threshold is not None:
                ev['pga_filter_config']['attack_rise_max_ms'] = attack_rise_threshold

        events = list(pga_kept)
        updated_stems[stem_type]['events_pga'] = raw_pga_events

        # Re-run classification so a hihat open_decay_slope_max
        # slider change (or any per-stem classification threshold)
        # takes effect on Save & Reconvert. force_reclassify=True
        # every time a classification threshold is configured —
        # the PGA path has no stored logic to diff against, and
        # the classifier is cheap.
        classification_changed = _classification_thresholds_changed(config, stem_type)
        classify_notes(
            events, stem_type, drum_mapping, config,
            force_reclassify=classification_changed,
        )

        # 2026-06-30: classification override application. The
        # _move_overridden_events call above already wrote the
        # override's classification onto the events in pga_kept.
        # _map_note (called inside classify_notes) reads the event's
        # classification and emits the right MIDI note. So the
        # override's per-event note flows through naturally. We
        # only need to mirror the override's classification onto the
        # raw_pga_events entries (so the sidecar reflects it) and
        # the same for hihat_state if the user toggled it.
        time_to_kept = {round(e['time'], 4): e for e in events}
        for ev in raw_pga_events:
            if ev.get('status') != 'KEPT':
                continue
            t = round(ev['time'], 4)
            cls_ev = time_to_kept.get(t)
            if cls_ev is not None:
                if 'hihat_state' in cls_ev:
                    ev['hihat_state'] = cls_ev['hihat_state']
                if 'classification' in cls_ev:
                    ev['classification'] = cls_ev['classification']
                if 'note' in cls_ev:
                    ev['note'] = cls_ev['note']

        # Build MIDI events directly from PGA kept events. The
        # PGA events already carry midi_velocity from the
        # detector's linear envelope mapping. The note,
        # timing_offset, and max_note_duration are read per-stem.
        #
        # 2026-06-30: ``stem_note`` is now only a FALLBACK for
        # events that lack a per-event ``note`` field (kick, or
        # classification skipped). For hihat/toms/snare/cymbals,
        # ``classify_notes`` above stamps ``ev['note']`` per
        # event (open vs closed for hihat; low/mid/high for toms;
        # snare/rimshot/clap for snare; crash/ride/chinese for
        # cymbals). The MIDI loop reads ``ev.get('note')`` first
        # and falls back to ``stem_note`` otherwise.
        stem_note = getattr(drum_mapping, stem_type)
        stem_timing_offset = config.get(stem_type, {}).get('timing_offset', 0.0)
        stem_max_duration = config.get(stem_type, {}).get(
            'max_note_duration',
            config.get('midi', {}).get('max_note_duration', 0.5),
        )
        midi_events = []
        for i, ev in enumerate(events):
            midi_time = float(ev['time']) + stem_timing_offset
            velocity = int(ev.get('midi_velocity', 80))
            if ev.get('duration_ms') is not None:
                duration = min(ev['duration_ms'] / 1000.0, stem_max_duration)
            elif i < len(events) - 1:
                duration = min(events[i + 1]['time'] - ev['time'], stem_max_duration)
            else:
                duration = config.get('audio', {}).get('default_note_duration', 0.1)
            ev_note = ev.get('note') or stem_note
            midi_events.append({
                'time': float(midi_time),
                'note': int(ev_note),
                'velocity': int(velocity),
                'duration': float(duration),
                'hihat_state': ev.get('hihat_state'),
            })
        midi_events_by_stem[stem_type] = midi_events

        # PGA-only stems have no sidecar logic block. Strip any
        # stale one the user might have carried over from a
        # pre-PGA conversion of the same project.
        stem_data.pop('logic', None)

    updated_analysis = dict(analysis_data)
    updated_analysis['stems'] = updated_stems
    return updated_analysis, midi_events_by_stem
