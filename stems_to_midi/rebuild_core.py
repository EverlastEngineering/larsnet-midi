"""
Rebuild MIDI from Analysis — Functional Core

Re-filters cached detection results from analysis.json and produces
MIDI-ready events without re-running audio detection. This enables
sub-second parameter tuning iteration.

The rebuild operates in two modes:
- **Same thresholds**: Trust stored statuses from analysis.json exactly.
  The full pipeline applied multi-pass filtering (geomean, decay, statistical,
  reverb continuation) that cannot be replicated without audio.
- **Changed thresholds**: Re-apply geomean/sustain filtering (Pass 1) to
  events_configured. Merge sensitive events only when thresholds are lowered
  to discover events the original pipeline would not have found.

After filtering, note classification (Pass 2) runs on the final KEPT set
using stored spectral features (spectral_centroid_hz, sustain_ms, energy
bands). This ensures note assignments (open/closed hihat, crash/ride/chinese,
low/mid/high tom, snare types) reflect the actual event population.

Pure functions — no I/O, no side effects.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .analysis_core import (
    get_spectral_config_for_stem,
    mark_reverb_continuations,
    should_keep_onset,
    normalize_values,
    estimate_velocity,
)
from .config import DrumMapping
from .note_classification_core import classify_notes


# ============================================================================
# Threshold Comparison
# ============================================================================


def _thresholds_changed(
    spectral_config: Dict,
    stored_logic: Dict,
) -> bool:
    """
    Determine if current config thresholds differ from stored analysis logic.

    Compares geomean_threshold, min_sustain_ms, min_strength_threshold,
    and the onset_events_enabled master gate (2026-06-10) — the
    parameters that the user can tune via sliders.

    Args:
        spectral_config: Current config from get_spectral_config_for_stem().
        stored_logic: The 'logic' block from analysis.json for this stem.

    Returns:
        True if any threshold has changed, False if identical.
    """
    current_geomean = spectral_config.get('geomean_threshold')
    stored_geomean = stored_logic.get('geomean_threshold')

    current_sustain = spectral_config.get('min_sustain_ms')
    stored_sustain = stored_logic.get('min_sustain_ms')

    current_strength = spectral_config.get('min_strength_threshold')
    stored_strength = stored_logic.get('min_strength_threshold')

    if current_geomean != stored_geomean:
        return True
    if current_sustain != stored_sustain:
        return True
    if current_strength != stored_strength:
        return True

    # Master onset-filter gate (2026-06-10). When this flips, we
    # need to re-run the rebuild path even if no threshold number
    # changed — the master gate is the path that bypasses
    # _refilter_events entirely. Use the spectral_config's value
    # (which carries the per-stem gate via get_spectral_config_for_stem).
    current_master = spectral_config.get('onset_events_enabled', None)
    stored_master = stored_logic.get('onset_events_enabled', None)
    # Both None → unchanged (legacy project, gate is missing on
    # both sides, _refilter_events runs as before). Any other
    # combination → changed.
    if current_master is not None or stored_master is not None:
        if current_master != stored_master:
            return True

    return False


def _classification_thresholds_changed(
    spectral_config: Dict,
    stored_logic: Dict,
    config: Dict,
    stem_type: str,
) -> bool:
    """
    Determine if hihat open/closed (or any future threshold-based classification)
    thresholds have changed since the last rebuild.

    The Pass 2 classifier (note_classification_core.classify_notes) preserves
    stored ``hihat_state`` / ``classification`` to avoid silently flipping
    a previously-classified event when nothing about its inputs has changed.
    When the user moves an open_geomean_min / open_sustain_ms / cluster
    threshold slider, we need to force reclassification so the new
    thresholds take effect.

    Args:
        spectral_config: Current config from get_spectral_config_for_stem().
        stored_logic: The 'logic' block from analysis.json for this stem.
        config: Full config dict (for hihat.* / per-stem classification keys).
        stem_type: Stem type to inspect.

    Returns:
        True if any classification threshold has changed, False if identical.
    """
    if stem_type == 'hihat':
        hihat_config = config.get('hihat', {})
        # 2026-06-19: open_decay_slope_max replaces the legacy
        # open_geomean_min + open_sustain_ms pair. The legacy keys
        # remain in the loop defensively (some pre-2026-06-19
        # analysis.json sidecars may have them stored, and removing
        # them here would silently skip reclassification when an
        # older project is rebuilt). The slope key is the primary
        # gate — without it, dragging the slope slider would save
        # to YAML but never trigger force_reclassify, leaving the
        # rebuilt events with stale hihat_state values.
        for key in ('open_decay_slope_max', 'open_geomean_min', 'open_sustain_ms'):
            current = hihat_config.get(key)
            stored = stored_logic.get(key)
            # Only treat as a real change when both sides are known and
            # disagree. If the stored logic has no record of the key (older
            # analysis.json, or key was added after the save), assume the
            # stored event classifications are still authoritative — the
            # rebuild path will repopulate the logic block for the next save.
            if current is not None and stored is not None and current != stored:
                return True
        return False

    # For snare/toms/cymbals, classification is driven by
    # expected_clusters, cluster_feature, and the per-class MIDI
    # notes (midi_note, midi_note_rimshot, midi_note_clap, midi_note_crash,
    # midi_note_ride, midi_note_chinese). Any change to these forces
    # reclassification so the new labels take effect.
    stem_config = config.get(stem_type, {})
    classification_keys = (
        'expected_clusters', 'cluster_feature', 'midi_note',
    )
    # Per-class notes: any midi_note_* key in this stem.
    for key in stem_config:
        if key.startswith('midi_note_'):
            classification_keys = classification_keys + (key,)
            break  # only need to detect a change in any of them
    for key in classification_keys:
        # The thresholds come from the stem config (config[stem_type]),
        # NOT the per-stem spectral_config. spectral_config is the
        # geomean/sustain/strength threshold bundle, not the
        # classification input. We read both: the stem config has
        # the value the user just set, the stored_logic has what the
        # previous conversion used.
        current = stem_config.get(key)
        stored = stored_logic.get(key)
        if current is not None and stored is not None and current != stored:
            return True
        # Also check the spectral_config (covers cluster_feature which
        # some flows put there). This is a defensive read.
        spectral_value = spectral_config.get(key)
        if spectral_value is not None and stored is not None and spectral_value != stored:
            return True
    return False


def _thresholds_lowered(
    spectral_config: Dict,
    stored_logic: Dict,
) -> bool:
    """
    Determine if thresholds were lowered (more permissive), requiring
    sensitive events to fill in newly-qualifying candidates.

    Args:
        spectral_config: Current config from get_spectral_config_for_stem().
        stored_logic: The 'logic' block from analysis.json for this stem.

    Returns:
        True if geomean threshold was lowered, sustain threshold was lowered, 
        or strength threshold was lowered.
    """
    current_geomean = spectral_config.get('geomean_threshold', 0)
    stored_geomean = stored_logic.get('geomean_threshold', 0)

    if current_geomean < stored_geomean:
        return True

    current_sustain = spectral_config.get('min_sustain_ms')
    stored_sustain = stored_logic.get('min_sustain_ms')

    # If sustain filter was added or threshold raised, that's more restrictive
    # If sustain filter was removed or threshold lowered, that's more permissive
    if stored_sustain is not None and current_sustain is not None:
        if current_sustain < stored_sustain:
            return True
    elif stored_sustain is not None and current_sustain is None:
        # Sustain filter removed = more permissive
        return True

    # Check min_strength_threshold
    current_strength = spectral_config.get('min_strength_threshold')
    stored_strength = stored_logic.get('min_strength_threshold')
    
    if stored_strength is not None and current_strength is not None:
        if current_strength < stored_strength:
            return True
    elif stored_strength is not None and current_strength is None:
        # Strength filter removed = more permissive
        return True
    elif stored_strength is None and current_strength is not None:
        # Strength filter added = more restrictive (not lowered)
        pass

    return False


# ============================================================================
# Event Pool Construction
# ============================================================================


def _merge_sensitive_events(
    configured_events: List[Dict],
    sensitive_events: List[Dict],
    merge_window_sec: float = 0.015,
) -> List[Dict]:
    """
    Add sensitive-only events to the configured pool for re-filtering.

    Only called when thresholds have been lowered, to find events that
    the original pipeline would not have detected at configured sensitivity.
    Configured events are authoritative; sensitive events fill gaps only.

    Args:
        configured_events: Events from configured-sensitivity detection.
        sensitive_events: Events from max-sensitivity detection.
        merge_window_sec: Time window for considering events as duplicates.

    Returns:
        Combined event list sorted by time.
    """
    configured_times = {round(e['time'], 4) for e in configured_events}

    merged = list(configured_events)

    for event in sensitive_events:
        t = event['time']
        is_duplicate = any(
            abs(t - ct) < merge_window_sec for ct in configured_times
        )
        if not is_duplicate:
            entry = dict(event)
            entry['_source'] = 'sensitive'
            merged.append(entry)

    merged.sort(key=lambda e: e['time'])
    return merged


# ============================================================================
# Re-filtering
# ============================================================================


def _apply_overrides(
    events: List[Dict],
    overrides: Dict[str, str],
) -> List[Dict]:
    """
    Apply manual overrides to event statuses.

    Override keys are time strings rounded to 4 decimals.
    Override values are 'KEPT' or 'FILTERED'.

    Args:
        events: Event dicts (mutated in place for efficiency).
        overrides: {time_key: 'KEPT'|'FILTERED'} from event_overrides.json.

    Returns:
        Same event list with override flags applied.
    """
    if not overrides:
        return events

    for event in events:
        time_key = f"{event['time']:.4f}"
        if time_key in overrides:
            event['status'] = overrides[time_key]
            event['override'] = True

    return events


def _refilter_events(
    events: List[Dict],
    spectral_config: Dict,
) -> List[Dict]:
    """
    Re-apply Pass 1 filtering thresholds (geomean/sustain) to events.

    Only called when thresholds have changed. Events with 'override' flag
    retain their status regardless.

    Note: This only applies Pass 1. Passes 2-4 (decay, statistical, reverb
    continuation) from the full pipeline require audio and cannot be replicated.
    The reverb continuation filter is applied separately as a post-pass since
    it operates on stored metadata.

    Spectral events (method='spectral') are EXEMPT from the geomean / sustain /
    strength filters (Bug D, 2026-06-09). Those signals are properties of
    the energy-detector output; spectral events have band_powers /
    band_max_ratio / band_delta / snap_delta instead. The previous
    implementation filtered them out whenever geomean was None (which
    ``event.get('geomean', 0.0)`` produces), which silently destroyed all
    magenta events when the user dragged the geomean slider. The
    show-only-snap pass (``_apply_show_only_snap_events``) handles
    low-snap-delta spectral FPs.

    Onset events visibility gate (2026-06-10 round 2): the
    ``onset_events_enabled`` bool does NOT change the behavior of
    this function — the geomean/sustain/strength filter still
    runs and produces the same statuses. The gate's effect
    happens later in ``rebuild_events_from_analysis``: after
    all filter passes, energy events (``method != 'spectral'``)
    are dropped from the events list entirely. This keeps the
    per-event filter logic simple and idempotent (re-evaluating
    the same inputs gives the same outputs) while letting the
    user "show only spectral" by flipping the gate.

    Args:
        events: Event dicts with pre-computed geomean, sustain_ms, strength, etc.
        spectral_config: From get_spectral_config_for_stem() with current thresholds.

    Returns:
        Same event list with updated 'status' fields.
    """
    geomean_threshold = spectral_config.get('geomean_threshold')
    min_sustain_ms = spectral_config.get('min_sustain_ms')
    filter_mode = spectral_config.get('filter_mode', 'geomean_only')
    min_strength_threshold = spectral_config.get('min_strength_threshold')

    for event in events:
        # Skip overridden events — user decision is authoritative
        if event.get('override'):
            continue

        # Spectral events are not subject to the energy-derived
        # filters (geomean / sustain / strength). They have their
        # own quality signal (band_max_ratio) which the server-side
        # band-ratio quality floor already enforces. Keep them
        # unconditionally here; the snap-mask pass handles the
        # low-snap-delta FPs.
        #
        # PGA events (method='percentile_gated') are likewise exempt
        # (2026-06-11). They have no geomean/sustain/strength values —
        # the prominence-vs-threshold gate is the PGA quality signal
        # and it was already applied at detection time
        # (see processing_shell step 11.6 / ``pga_min_prominence``).
        # Sending them through ``should_keep_onset`` would default
        # ``geomean`` to 0.0 and unconditionally FILTER every PGA
        # event, silently dropping the entire toms stream on rebuild
        # when ``geomean_threshold`` is non-zero.
        if event.get('method') in ('spectral', 'percentile_gated'):
            continue

        is_kept = should_keep_onset(
            geomean=event.get('geomean', 0.0),
            sustain_ms=event.get('sustain_ms'),
            geomean_threshold=geomean_threshold,
            min_sustain_ms=min_sustain_ms,
            filter_mode=filter_mode,
            strength=event.get('strength'),
            min_strength_threshold=min_strength_threshold,
        )
        event['status'] = 'KEPT' if is_kept else 'FILTERED'

    return events


def _apply_reverb_continuation_filter(
    events: List[Dict],
    config: Dict,
) -> List[Dict]:
    """
    Apply reverb continuation detection as a post-filter pass.

    Uses stored metadata (attack_sharpness, duration_sec, amplitude_at_start,
    amplitude_at_end) to identify events that are reverb/decay artifacts
    rather than real hits. This replicates the final pass from
    filter_onsets_by_spectral() without requiring audio.

    Args:
        events: Event dicts with status field. Only KEPT events are evaluated.
        config: Full config dict for reverb continuation threshold.

    Returns:
        Same event list with REVERB_CONTINUATION statuses applied.
    """
    # Only process events that have the required metadata
    kept_events = [e for e in events if e.get('status') == 'KEPT']
    if len(kept_events) < 2:
        return events

    # Check if events have the required metadata fields
    has_metadata = all(
        'duration_sec' in e and 'amplitude_at_start' in e
        for e in kept_events
    )
    if not has_metadata:
        return events

    # mark_reverb_continuations modifies in place — operates on KEPT events.
    # Default 0.4 matches midiconfig.yaml (previously 0.2 — silent drift).
    attack_threshold = config.get('filtering', {}).get(
        'reverb_continuation_attack_threshold', 0.4
    )
    mark_reverb_continuations(
        kept_events,
        time_margin_ms=5.0,
        amplitude_margin=0.001,
        attack_sharpness_threshold=attack_threshold,
    )

    # Transfer status changes back to the main events list
    # (mark_reverb_continuations modified the kept_events in place,
    # and they reference the same dicts as the events list)
    return events


def _apply_show_only_snap_events(
    events: List[Dict],
    config: Dict,
    stem_type: str,
) -> List[Dict]:
    """
    Drop spectral events whose snap_delta is zero (2026-06-10).

    Replaces the 2026-06-09 snap-mask + 2026-06-10 advanced-filter
    chain with a single, easy-to-reason-about toggle: "Show Only Snap
    Events" — when on, only events with snap_delta > 0 survive. When
    off (default), the filter is a no-op.

    Rationale: snap_delta > 0 means the broadband attack signal
    fired in the snap bands (see spectral_transient_core snap_delta
    definition). snap_delta == 0 typically indicates a wire-tail /
    decay event where the per-band-dominant ring has outlasted the
    broadband attack — these were the events the old snap-mask was
    catching. With the ratio slider below handling the "extreme
    dominance" FP case (events like band_max_ratio 459), this
    filter is sufficient for the user's typical tom tuning.

    Gate: respects ``config[stem_type].show_only_snap_events``.
    Missing key (legacy projects) → False (no-op). Explicit True
    applies the filter. The filter is idempotent across rebuilds
    (it doesn't add a status that another pass would then need to
    undo).

    Only events with a non-None ``snap_delta`` are evaluated. Energy
    events (no snap_delta) and overridden events are untouched.

    Args:
        events: Event dicts with status field. Mutated in place.
        config: Full config dict (with per-stem sections).
        stem_type: Stem name (e.g. 'toms', 'snare').

    Returns:
        Same event list with snap-zero events set to FILTERED.
    """
    stem_cfg = config.get(stem_type, {})
    enabled = stem_cfg.get('show_only_snap_events', False)
    if not enabled:
        return events

    for event in events:
        if event.get('override'):
            continue
        if event.get('method') != 'spectral':
            continue
        sd = event.get('snap_delta')
        if sd is None or sd <= 0:
            event['status'] = 'FILTERED'

    return events


def _apply_band_max_ratio_max(
    events: List[Dict],
    config: Dict,
    stem_type: str,
) -> List[Dict]:
    """
    Drop spectral events whose band_max_ratio exceeds a user-set
    ceiling (2026-06-10).

    Replaces the old "Filter High-Strength FPs" stage 3 of the
    advanced filter. The previous implementation was lossy — it
    operated on the clamp-to-1.0 `strength` field, so a band_max_ratio
    of 11 and a band_max_ratio of 459 were indistinguishable. This
    filter reads the RAW band_max_ratio, so the user can finally
    tell those events apart and keep the real one (18.99) while
    dropping the FP (459.12).

    Gate: respects ``config[stem_type].band_max_ratio_max``.
    Missing key (legacy projects) → 0 (no-op, disabled). The slider
    in the WebUI sidecar labels 0 as "Off" / "Disabled" so the user
    can confirm the filter is inactive. Any positive value is the
    ceiling — events with band_max_ratio strictly greater than the
    threshold are FILTERED. The slider max is the dataset's actual
    max ratio (computed at UI build time) so the user can express
    the full range without losing precision.

    Only spectral events with a non-None ``band_max_ratio`` are
    evaluated. Energy events and overridden events are untouched.
    Other event statuses (REVERB_CONTINUATION, etc.) are preserved
    when not in the "above ceiling" bucket.

    Args:
        events: Event dicts with status field. Mutated in place.
        config: Full config dict (with per-stem sections).
        stem_type: Stem name (e.g. 'toms', 'snare').

    Returns:
        Same event list with above-ceiling events set to FILTERED.
    """
    stem_cfg = config.get(stem_type, {})
    threshold = stem_cfg.get('band_max_ratio_max', None)
    if threshold is None:
        return events
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return events
    # 0 or negative = disabled. The UI slider labels this "Off" so
    # the user always knows the filter is inactive.
    if threshold <= 0:
        return events

    for event in events:
        if event.get('override'):
            continue
        if event.get('method') != 'spectral':
            continue
        ratio = event.get('band_max_ratio')
        if ratio is None:
            continue
        if ratio > threshold:
            event['status'] = 'FILTERED'

    return events


# ============================================================================
# MIDI Event Creation from Analysis Events
# ============================================================================


def _events_to_midi(
    kept_events: List[Dict],
    stem_type: str,
    drum_mapping: DrumMapping,
    config: Dict,
    spectral_config: Dict,
) -> List[Dict]:
    """
    Convert kept analysis events to MIDI event dicts.

    Handles velocity normalization and note assignment (including
    pitch classification for toms, cymbals, snare, hihat state).

    This replicates the logic in processing_shell._create_midi_events()
    but operates on pre-computed analysis data rather than raw arrays.

    Args:
        kept_events: Analysis events with status == 'KEPT'.
        stem_type: Stem type for note/classification routing.
        drum_mapping: MIDI note mapping.
        config: Full config dict.
        spectral_config: Spectral config for this stem.

    Returns:
        List of MIDI event dicts with time, note, velocity, duration.
    """
    if not kept_events:
        return []

    stem_config = config.get(stem_type, {})
    min_velocity = stem_config.get('min_velocity', config.get('midi', {}).get('min_velocity', 80))
    max_velocity = stem_config.get('max_velocity', config.get('midi', {}).get('max_velocity', 110))
    timing_offset = stem_config.get('timing_offset', 0.0)
    default_note = getattr(drum_mapping, stem_type)
    use_sustain_duration = spectral_config.get('use_sustain_duration', False)
    max_note_duration = stem_config.get('max_note_duration', config.get('midi', {}).get('max_note_duration', 0.5))
    default_duration = config.get('audio', {}).get('default_note_duration', 0.1)

    # Determine velocity source
    velocity_source = spectral_config.get('velocity_source', 'peak_amplitude')

    # Extract velocity feature values
    if velocity_source == 'geomean':
        raw_values = np.array([e.get('geomean', 0.0) for e in kept_events])
    elif velocity_source == 'onset_strength':
        raw_values = np.array([e.get('strength', 0.0) for e in kept_events])
    else:
        raw_values = np.array([e.get('amplitude', 0.0) for e in kept_events])

    normalized = normalize_values(raw_values)

    midi_events = []
    for i, event in enumerate(kept_events):
        # For PGA events (toms), use the pre-computed midi_velocity from
        # the detector's linear envelope-value mapping. For other stems,
        # compute from the configured velocity source.
        if event.get('method') == 'percentile_gated' and event.get('midi_velocity') is not None:
            velocity = int(event['midi_velocity'])
        else:
            velocity = estimate_velocity(float(normalized[i]), min_velocity, max_velocity)
        midi_note = _resolve_note(event, i, stem_type, drum_mapping, config)

        # Duration: sustain-based or time-to-next
        if use_sustain_duration and event.get('sustain_ms') is not None:
            duration = event['sustain_ms'] / 1000.0
            stem_max = stem_config.get('max_note_duration', 2.0)
            duration = min(duration, stem_max)
        elif i < len(kept_events) - 1:
            duration = kept_events[i + 1]['time'] - event['time']
            duration = min(duration, max_note_duration)
        else:
            duration = default_duration

        midi_time = event['time'] + timing_offset

        midi_events.append({
            'time': float(midi_time),
            'note': int(midi_note),
            'velocity': int(velocity),
            'duration': float(duration),
        })

        # Generate foot-close for open hihats
        if stem_type == 'hihat' and event.get('hihat_state') == 'open':
            generate_foot_close = stem_config.get('generate_foot_close', False)
            if generate_foot_close and event.get('sustain_ms') is not None:
                foot_close_note = stem_config.get('midi_note_foot_close', 44)
                sustain_sec = event['sustain_ms'] / 1000.0
                foot_close_time = midi_time + sustain_sec
                foot_close_vel = max(40, min(100, int(velocity * 0.7)))
                midi_events.append({
                    'time': float(foot_close_time),
                    'note': int(foot_close_note),
                    'velocity': int(foot_close_vel),
                    'duration': 0.05,
                })

    return midi_events


def _resolve_note(
    event: Dict,
    index: int,
    stem_type: str,
    drum_mapping: DrumMapping,
    config: Dict,
) -> int:
    """
    Resolve the MIDI note number for an event based on classification data.

    Analysis events may carry classification metadata (note field from prior
    processing, or hihat_state/classification fields). Uses these when
    available, otherwise falls back to default stem note.
    """
    # If the event already has a note assignment from prior processing, use it
    if 'note' in event and event['note'] is not None:
        return event['note']

    # Hihat state classification
    if stem_type == 'hihat':
        hihat_state = event.get('hihat_state', 'closed')
        if hihat_state == 'handclap':
            return drum_mapping.handclap
        elif hihat_state == 'open':
            return drum_mapping.hihat_open
        return drum_mapping.hihat_closed

    # Use default note for stem type
    return getattr(drum_mapping, stem_type)


# ============================================================================
# Main Rebuild Function
# ============================================================================


def rebuild_events_from_analysis(
    analysis_data: Dict,
    overrides: Dict[str, Dict[str, str]],
    config: Dict,
    stem_types: Optional[List[str]] = None,
) -> Tuple[Dict, Dict[str, List[Dict]]]:
    """
    Re-filter and rebuild MIDI events from cached analysis.json data.

    This is the primary entry point for the rebuild-from-analysis pipeline.
    It replaces the full detection pipeline when only filtering thresholds
    or manual overrides have changed.

    Operating modes per stem:
    - **Same thresholds, no overrides**: Trust stored statuses from the full
      pipeline. Events already went through multi-pass filtering (geomean,
      decay, statistical, reverb continuation). Just reconstruct MIDI.
    - **Same thresholds, with overrides**: Apply overrides to stored events,
      then reconstruct MIDI.
    - **Changed thresholds**: Re-apply Pass 1 (geomean/sustain) filtering.
      If thresholds lowered, merge sensitive events to find new candidates.
      Apply reverb continuation filter as post-pass.

    Args:
        analysis_data: Parsed analysis.json dict (v3 format).
        overrides: Per-stem manual overrides from event_overrides.json.
            Format: {stem_type: {time_key: 'KEPT'|'FILTERED'}}.
        config: Parsed midiconfig.yaml dict with current thresholds.
        stem_types: Optional list of stems to rebuild (None = all stems
            present in analysis_data).

    Returns:
        Tuple of:
        - updated_analysis: Copy of analysis_data with event statuses
          reflecting the new filter results. Overridden events carry
          an 'override' flag.
        - midi_events_by_stem: Dict mapping stem_type to lists of MIDI
          event dicts ready for create_midi_file().

    Raises:
        ValueError: If analysis_data is missing, wrong version, or has no stems.
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

    # Determine which stems to rebuild
    available_stems = list(stems_data.keys())
    if stem_types is None:
        stem_types = available_stems
    else:
        # Validate requested stems exist
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
        configured_events = stem_data.get('events_configured', [])
        sensitive_events = stem_data.get('events_sensitive', [])
        # 2026-06-18: ``pga_rebuild_active`` is the trigger for
        # the PGA-only rebuild path. It's true when the sidecar
        # carries ``events_pga`` (i.e. ``process_percentile_gated``
        # produced the sidecar — true for toms today, and true
        # for any other stem that opts into PGA via
        # ``<stem>.use_pga_detection: true`` in the project
        # midiconfig). The flag drives all the conditional
        # branches below; previously they were hard-coded to
        # ``stem_type == 'toms'``.
        raw_pga_events = list(stem_data.get('events_pga', []))
        pga_rebuild_active = len(raw_pga_events) > 0
        # 2026-06-15: toms (and any other PGA-only stem) no
        # longer reads or writes a sidecar logic block. The
        # PGA-only rebuild path resolves its threshold directly
        # from yaml; the comparison helpers
        # (_thresholds_changed / _thresholds_lowered /
        # _classification_thresholds_changed) below are only
        # needed for the energy/spectral stems that still use
        # the logic block for change detection.
        stored_logic = {} if pga_rebuild_active else stem_data.get('logic', {})
        stem_overrides = overrides.get(stem_type, {})

        # Get current spectral config for this stem (reads thresholds from config)
        spectral_config = get_spectral_config_for_stem(stem_type, config)

        # Determine rebuild strategy based on threshold changes.
        # Skipped for PGA-only stems: the PGA prominence filter
        # always re-runs and reads its threshold from yaml
        # directly (see below), so the change-detection helpers
        # are not needed.
        if pga_rebuild_active:
            changed = False
            lowered = False
            # 2026-06-20: the classification thresholds (e.g. hihat
            # open_decay_slope_max) are independent of the prominence
            # filter — they live on the events themselves, not on
            # the spectral_config. PGA stems still need to detect
            # a slope change so Save & Reconvert forces
            # classify_notes to re-stamp hihat_state on every KEPT
            # event. Without this, the stored hihat_state from the
            # last conversion is preserved unchanged and the user's
            # slope slider value sits in yaml with no visible
            # effect on the sidecar.
            #
            # PGA stems have no sidecar logic block (stored_logic
            # is empty here), so the standard diff against
            # stored_logic can't run — every key would compare
            # None vs current. We force classification any time a
            # classification threshold is configured for this stem
            # in yaml. Cheap (a per-event rule check on the kept
            # set) and correct: every rebuild re-applies the
            # current rule from scratch. The previous behavior
            # (preserve stored hihat_state forever) was a bug —
            # the user's slope change had no visible effect on
            # Save & Reconvert.
            if stem_type == 'hihat' and config.get('hihat', {}).get('open_decay_slope_max') is not None:
                classification_changed = True
            elif _classification_thresholds_changed(
                spectral_config, stored_logic, config, stem_type,
            ):
                classification_changed = True
            else:
                classification_changed = False
        else:
            changed = _thresholds_changed(spectral_config, stored_logic)
            lowered = changed and _thresholds_lowered(spectral_config, stored_logic)
            classification_changed = _classification_thresholds_changed(
                spectral_config, stored_logic, config, stem_type,
            )

        # ------------------------------------------------------------------
        # PGA prominence re-filter for PGA-only stems (2026-06-15,
        # generalized 2026-06-18 to any stem with ``events_pga``
        # in the sidecar, not just toms).
        #
        # After the pga_event_builder refactor, events_pga carries ALL
        # detected events (status='KEPT', no filter applied at detect
        # time). We ALWAYS re-apply the prominence filter for any
        # PGA-only stem on every rebuild — using the threshold from
        # yaml. This produces the correct KEPT/filtered split
        # regardless of whether the user moved the WebUI slider.
        #
        # This branch runs BEFORE the geomean/sustain filter path
        # (which exempts method='percentile_gated' at line 365) so
        # the PGA re-filter is the primary filter for these stems.
        # ------------------------------------------------------------------
        if pga_rebuild_active:
            if raw_pga_events:
                # yaml is the single source of truth for PGA
                # prominence. The per-event pga_filter_config
                # threshold in the sidecar is the detect-time
                # value (informational; not used as a fallback)
                # and the sidecar's logic block is no longer
                # emitted for PGA-only stems. Priority:
                # stem-specific YAML > global onset_detection
                # YAML > hard default. The hard default (1000)
                # is never reached in practice because the
                # project yaml always carries a value, but we
                # keep it as a defensive floor.
                stem_pga = config.get(stem_type, {}).get('pga_min_prominence')
                global_pga = config.get('onset_detection', {}).get('pga_min_prominence')
                pga_threshold = float(
                    stem_pga if stem_pga is not None
                    else global_pga if global_pga is not None
                    else 1000.0
                )
            else:
                pga_threshold = None

        if pga_rebuild_active and pga_threshold is not None:
            from .pga_event_builder import (
                apply_pga_prominence_filter,
                apply_pga_decay_col_min_filter,
                apply_attack_rise_max_filter,
            )
            raw_pga = list(stem_data.get('events_pga', []))
            if raw_pga:
                # 2026-06-15: also resolve the decay_col_min
                # threshold (per-stem > global > -80.0 default).
                # Mirrors the resolution in
                # _build_pga_events_with_filter and
                # detect_pga_events.
                stem_col_min = config.get(stem_type, {}).get('min_decay_col_min_db')
                global_col_min = config.get('onset_detection', {}).get('min_decay_col_min_db')
                # 2026-06-20: only apply decay_col_min / attack_rise
                # when the threshold is explicitly configured for
                # this stem or globally. The hard-coded -80.0 default
                # was calibrated for TOMS strikes (range -60 to -84 dB
                # per filter_registry.json) and silently filters every
                # cymbals / hihat / snare event whose decay_col_min is
                # in the -95 to -120 dB range. The WebUI panel
                # skips these filters when they're not in
                # tuningConfig (no slider exposed), so the rebuild
                # path must mirror that — otherwise the panel shows
                # N kept events and Save & Reconvert wipes them to 0.
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
                # Run the prominence filter first.
                pga_kept, pga_filtered = apply_pga_prominence_filter(
                    raw_pga,
                    pga_threshold,
                )
                # 2026-06-15: run the decay_col_min filter on
                # top of the prominence filter. Events that
                # passed the prominence check but failed the
                # ring-quality check (decay_col_min_median_db
                # below the threshold) are tagged FILTERED.
                # 2026-06-20: only runs when col_min_threshold is
                # not None — see comment above on WebUI/server
                # agreement.
                if col_min_threshold is not None:
                    pga_kept, col_min_filtered = apply_pga_decay_col_min_filter(
                        pga_kept,
                        col_min_threshold,
                    )
                    pga_filtered = pga_filtered + col_min_filtered
                # 2026-06-17: attack_rise filter (third PGA
                # pass). Catches wire-tail / step-back FPs
                # that pass prominence + decay_col_min but
                # have an unusually long 10-90% rise time.
                # Layered on top of the previous filters.
                # 2026-06-20: only runs when attack_rise_threshold
                # is not None — same rationale.
                if attack_rise_threshold is not None:
                    pga_kept, attack_filtered = apply_attack_rise_max_filter(
                        pga_kept,
                        attack_rise_threshold,
                    )
                    pga_filtered = pga_filtered + attack_filtered
                # Build time-keyed lookup for status assignment
                kept_times = {round(e['time'], 4) for e in pga_kept}
                filtered_times = {round(e['time'], 4) for e in pga_filtered}
                for ev in raw_pga:
                    t = round(ev['time'], 4)
                    if t in kept_times:
                        ev['status'] = 'KEPT'
                        ev.pop('filter_reason', None)
                    elif t in filtered_times:
                        ev['status'] = 'FILTERED'
                        # Determine which filter dropped this event
                        # based on its filter_reason (set by the
                        # filter that tagged it).
                        existing_reason = ev.get('filter_reason', '')
                        if (
                            'min_decay_col_min_db' in existing_reason
                            or 'attack_rise_max_ms' in existing_reason
                        ):
                            # Already has the decay_col_min or
                            # attack_rise reason from the
                            # corresponding filter. The Python
                            # wrapper sets the reason via
                            # build_filter_reason from the
                            # registry's reason_template.
                            pass
                        else:
                            prom = ev.get('prominence')
                            ev['filter_reason'] = (
                                f"below pga_min_prominence ({prom:.0f} < {pga_threshold:.0f})"
                                if prom is not None else 'below pga_min_prominence'
                            )
                    # Update pga_filter_config to reflect the new
                    # thresholds. 2026-06-20: only stamp the keys
                    # whose filter is actually active — a stem that
                    # never had decay_col_min configured should not
                    # gain a stale -80.0 in its sidecar metadata.
                    ev['pga_filter_config'] = dict(ev.get('pga_filter_config', {}))
                    ev['pga_filter_config']['pga_min_prominence'] = pga_threshold
                    if col_min_threshold is not None:
                        ev['pga_filter_config']['min_decay_col_min_db'] = col_min_threshold
                    if attack_rise_threshold is not None:
                        ev['pga_filter_config']['attack_rise_max_ms'] = attack_rise_threshold

                # events_configured for PGA-only stems is EMPTY —
                # events_pga is the single source of truth. rebuild_core
                # writes back the re-filtered statuses to events_pga
                # only. Generalized 2026-06-18 from toms-only to any
                # stem with events_pga in the sidecar.
                events = list(pga_kept)
                updated_stems[stem_type]['events_pga'] = raw_pga
                # events_configured intentionally absent for PGA-only
                # stems.

                # 2026-06-20: re-run classify_notes on the KEPT PGA
                # events so a hihat open_decay_slope_max slider
                # change (or any other per-stem classification
                # threshold) takes effect on Save & Reconvert.
                # Without this, the stored hihat_state / classification
                # from the last conversion is preserved unchanged and
                # the rebuilt MIDI + sidecar silently keep stale
                # labels. force_reclassify=True mirrors the reclassify
                # endpoint's contract — the user explicitly moved a
                # classification slider, so the rule must re-fire.
                # Note: classify_notes is imported at module top
                # (line 36). A local re-import here would shadow the
                # module-level binding and break the non-PGA
                # classify_notes call later in this function (Python
                # would treat the symbol as a local that hasn't been
                # bound on the non-PGA path).
                classify_notes(
                    events, stem_type, drum_mapping, config,
                    force_reclassify=classification_changed,
                )
                # Also stamp the new classification back onto the
                # raw_pga entries (events_configured is empty for
                # PGA-only stems, so the sidecar reader falls
                # through to events_pga — the source of truth).
                time_to_kept = {round(e['time'], 4): e for e in events}
                for ev in raw_pga:
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

                # Build MIDI events for PGA-only stems directly from
                # PGA kept events. These stems use ONLY events_pga —
                # skip the energy/spectral path entirely. The PGA
                # events already carry midi_velocity from the
                # detector's linear envelope mapping. The note,
                # timing_offset, and max_note_duration are read
                # per-stem (so snare uses drum_mapping.snare and
                # snare.timing_offset, not the hard-coded toms
                # equivalents).
                stem_note = getattr(drum_mapping, stem_type)
                # 2026-06-19: hihat open/closed -> MIDI note flip.
                # The PGA detector stamps hihat_state on every event
                # (slope rule primary, geomean+sustain fallback). For
                # hihat specifically, route open hihats to
                # drum_mapping.hihat_open (46) instead of the default
                # closed-hihat note (42). Other stems keep the
                # default stem note. Mirrors the live-detect path in
                # processing_shell_percentile_gated.process_percentile_gated.
                stem_note_open = getattr(
                    drum_mapping, 'hihat_open', stem_note
                ) if stem_type == 'hihat' else stem_note
                stem_timing_offset = config.get(stem_type, {}).get('timing_offset', 0.0)
                stem_max_duration = config.get(stem_type, {}).get('max_note_duration',
                                         config.get('midi', {}).get('max_note_duration', 0.5))
                midi_events = []
                for i, ev in enumerate(pga_kept):
                    midi_time = float(ev['time']) + stem_timing_offset
                    velocity = int(ev.get('midi_velocity', 80))
                    # Duration: use stored duration_ms or time-to-next
                    if ev.get('duration_ms') is not None:
                        duration = min(ev['duration_ms'] / 1000.0, stem_max_duration)
                    elif i < len(pga_kept) - 1:
                        duration = min(pga_kept[i + 1]['time'] - ev['time'], stem_max_duration)
                    else:
                        duration = config.get('audio', {}).get('default_note_duration', 0.1)
                    # 2026-06-19: open hihats use the open note (46).
                    ev_note = (
                        stem_note_open
                        if ev.get('hihat_state') == 'open'
                        else stem_note
                    )
                    midi_events.append({
                        'time': float(midi_time),
                        'note': int(ev_note),
                        'velocity': int(velocity),
                        'duration': float(duration),
                        'hihat_state': ev.get('hihat_state'),
                    })
                midi_events_by_stem[stem_type] = midi_events
                # PGA-only stem: the early-return `continue` below
                # skips the legacy branch that would have written the
                # logic block and updated events_configured /
                # events_sensitive. The PGA-only stems use ONLY
                # events_pga, so we still need to strip any stale
                # logic block from the loaded sidecar here (the
                # non-PGA cleanup at the end of the loop body
                # doesn't run for these stems). 2026-06-15.
                stem_data.pop('logic', None)
                continue
            else:
                events = list(configured_events)
        elif changed:
            # Thresholds changed — need to re-filter
            if lowered:
                # Thresholds lowered — merge sensitive events to find new candidates
                events = _merge_sensitive_events(
                    configured_events, sensitive_events,
                )
            else:
                # Thresholds raised — only re-filter configured events
                events = list(configured_events)

            # Apply manual overrides before re-filtering
            _apply_overrides(events, stem_overrides)

            # Re-apply Pass 1 filtering with new thresholds
            _refilter_events(events, spectral_config)

            # Apply reverb continuation filter (post-pass, uses stored metadata)
            _apply_reverb_continuation_filter(events, config)

            # 2026-06-10: the legacy snap-mask + advanced-filter chain
            # is replaced by the two simple toggles below. Both are
            # idempotent and run last so the user can flip either one
            # without re-running the detector. The order doesn't
            # matter — they act on disjoint attributes (snap_delta vs
            # band_max_ratio) — but snap-first matches the WebUI order
            # for predictability.
            _apply_show_only_snap_events(events, config, stem_type)
            _apply_band_max_ratio_max(events, config, stem_type)
        else:
            # Thresholds unchanged — trust stored statuses from full pipeline
            events = list(configured_events)

            # Still apply manual overrides (user may have toggled individual events)
            _apply_overrides(events, stem_overrides)

            # 2026-06-10: same two new filters as above — re-evaluate
            # them on every rebuild so toggles / slider changes the
            # user makes in the WebUI sidecar land in the saved MIDI
            # even when no other threshold changed.
            _apply_show_only_snap_events(events, config, stem_type)
            _apply_band_max_ratio_max(events, config, stem_type)

        # Onset events visibility gate (2026-06-10 round 2). When
        # the user has turned the toms onset_events toggle OFF,
        # energy-detected events are removed from the
        # events_configured list entirely. Spectral and PGA events
        # are unaffected. This is the "I want a spectral-only toms
        # view" path. The dropped events are NOT in events_sensitive
        # either — they're gone from the saved MIDI and the view.
        # The user must re-run full detection to restore them.
        #
        # 2026-06-11: PGA events (method='percentile_gated') are now
        # also exempt. Toms switched to PGA-only on 2026-06-11; the
        # gate previously kept only ``method == 'spectral'`` events
        # which silently dropped every PGA event when
        # ``onset_events_enabled: false`` was set in midiconfig.yaml
        # (the legacy default in many user projects). Keep both
        # spectral and PGA when the gate is closed.
        if spectral_config.get('onset_events_enabled') is False:
            events = [
                e for e in events
                if e.get('method') in ('spectral', 'percentile_gated')
            ]

        # Extract kept events for MIDI generation
        kept_events = [e for e in events if e.get('status') == 'KEPT']

        # Pass 2: Classify notes on the final KEPT set using stored features.
        # Force re-classification only when a classification threshold has
        # actually changed (e.g. user moved the open_geomean_min slider) —
        # otherwise preserve the stored hihat_state/classification so a
        # simple rebuild does not silently re-classify events.
        classify_notes(
            kept_events, stem_type, drum_mapping, config,
            force_reclassify=classification_changed,
        )

        # Generate MIDI events
        midi_events = _events_to_midi(
            kept_events, stem_type, drum_mapping, config, spectral_config,
        )
        midi_events_by_stem[stem_type] = midi_events

        # Update analysis data with new statuses
        # Separate back into configured vs sensitive-sourced for storage
        updated_configured = [
            e for e in events if e.get('_source') != 'sensitive'
        ]
        updated_sensitive = [
            e for e in events if e.get('_source') == 'sensitive'
        ]

        # Clean internal fields before storage
        for event_list in [updated_configured, updated_sensitive]:
            for event in event_list:
                event.pop('_source', None)

        # Attach note/velocity/classification to KEPT events via time-based matching.
        # Index-based pairing breaks when sensitive events are merged in, because
        # kept_midi includes entries for sensitive-sourced KEPT events that don't
        # appear in updated_configured.
        kept_midi = [e for e in midi_events if e.get('note') != 44]  # Exclude foot-close
        midi_by_time = {round(e['time'], 4): e for e in kept_midi}
        # classify_notes sets classification on kept_events in-place; build lookup
        kept_by_time = {round(e['time'], 4): e for e in kept_events}
        for event in updated_configured:
            if event.get('status') == 'KEPT':
                t = round(event['time'], 4)
                midi_ev = midi_by_time.get(t)
                if midi_ev:
                    event['note'] = midi_ev['note']
                    event['velocity'] = midi_ev['velocity']
                # classification/hihat_state are set in-place by classify_notes
                # on the kept_events refs, which are the same dicts as in events/
                # updated_configured — so they're already present.
            else:
                # Clear stale note/velocity/classification from previously-KEPT events
                event.pop('note', None)
                event.pop('velocity', None)
                event.pop('hihat_state', None)
                event.pop('classification', None)

        stem_data['events_configured'] = updated_configured
        if updated_sensitive:
            # Only update sensitive if we merged them in
            stem_data['events_sensitive'] = sensitive_events  # Keep original

        # Update stored logic to reflect current thresholds and classification
        # params (2026-06-15: skipped for toms — toms no longer emits a
        # logic block. The WebUI reads yaml directly via the
        # tuning-config endpoint, and the rebuild path's PGA
        # prominence filter reads yaml directly too. The other stems
        # still use the logic block for their change-detection
        # comparison against the live yaml; drop this guard for them
        # when they migrate to PGA).
        if stem_type != 'toms':
            stem_data['logic'] = _build_logic_block(
                spectral_config, stored_logic, stem_type, config,
            )
        else:
            # Toms: strip any stale logic block from the loaded
            # sidecar (older projects still have it). The next save
            # will also not emit one (see midi.py save_analysis_sidecar).
            stem_data.pop('logic', None)

    # Build updated analysis output
    updated_analysis = dict(analysis_data)
    updated_analysis['stems'] = updated_stems

    return updated_analysis, midi_events_by_stem


def _build_logic_block(
    spectral_config: Dict,
    stored_logic: Dict,
    stem_type: str = '',
    config: Optional[Dict] = None,
) -> Dict:
    """
    Build updated logic block reflecting current thresholds.

    Preserves non-threshold fields (freq_bands, passes, decay_filter_enabled)
    from the stored logic while updating threshold values and classification
    thresholds (e.g., hihat open/closed boundaries).
    """
    logic = dict(stored_logic)
    logic['geomean_threshold'] = spectral_config.get('geomean_threshold')
    logic['min_sustain_ms'] = spectral_config.get('min_sustain_ms')

    # Include global filtering thresholds so the frontend can read them
    if config:
        filtering_config = config.get('filtering', {})
        logic['reverb_continuation_attack_threshold'] = filtering_config.get(
            'reverb_continuation_attack_threshold', 0.4
        )

    # Include classification thresholds so the frontend can read them
    if config:
        stem_config = config.get(stem_type, {})
        if stem_type == 'hihat':
            logic['open_geomean_min'] = stem_config.get('open_geomean_min', 262.0)
            logic['open_sustain_ms'] = stem_config.get('open_sustain_ms', 100.0)
        if stem_type in ('snare', 'toms', 'cymbals'):
            defaults = {'snare': 2, 'toms': 3, 'cymbals': 2}
            raw = stem_config.get('expected_clusters')
            logic['expected_clusters'] = int(raw) if raw is not None else defaults[stem_type]
            logic['cluster_feature'] = stem_config.get('cluster_feature', 'auto')
            cluster_note_map = stem_config.get('cluster_note_map')
            if cluster_note_map:
                logic['cluster_note_map'] = cluster_note_map

            # Persist the user's tuning panel choices back to the
            # sidecar (logic block). The WebUI's tuning panel reads
            # the user's last choice from the logic block, so anything
            # that can be set in the sidecar must be persisted here
            # for the next page load to see it.
            #
            # 2026-06-10: the snap-mask + advanced-filter chain was
            # replaced by `show_only_snap_events` and
            # `band_max_ratio_max`. We persist the new keys (so the
            # user's slider position + toggle state survive a
            # rebuild) and also continue to persist the legacy keys
            # for any older project that still has them set — those
            # legacy keys are simply ignored by the new filter chain.
            if stem_type == 'toms':
                # Onset events visibility gate (2026-06-10 round 2).
                # Default is True (the schema default); missing key
                # means "use the schema default" so we only persist
                # when the user has set it explicitly.
                master = stem_config.get('onset_events_enabled', None)
                if master is not None:
                    logic['onset_events_enabled'] = bool(master)
                # New spectrogram filters (2026-06-10).
                show_only_snap = stem_config.get('show_only_snap_events', None)
                if show_only_snap is not None:
                    logic['show_only_snap_events'] = bool(show_only_snap)
                # Persist ratio_max even when it's 0 (the "Off"
                # sentinel). The slider's value of 0 is a meaningful
                # user choice — "I checked, the filter is off" — and
                # must survive a rebuild. The fall-through `is not
                # None` check below would also let 0 through, but we
                # be explicit.
                if 'band_max_ratio_max' in stem_config:
                    logic['band_max_ratio_max'] = float(
                        stem_config['band_max_ratio_max']
                    )

    return logic
