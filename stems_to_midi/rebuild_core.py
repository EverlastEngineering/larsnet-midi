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

    Compares geomean_threshold, min_sustain_ms, and min_strength_threshold —
    the parameters that the user can tune via sliders.

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

    # mark_reverb_continuations modifies in place — operates on KEPT events
    attack_threshold = config.get('filtering', {}).get(
        'reverb_continuation_attack_threshold', 0.2
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
        stored_logic = stem_data.get('logic', {})
        stem_overrides = overrides.get(stem_type, {})

        # Get current spectral config for this stem (reads thresholds from config)
        spectral_config = get_spectral_config_for_stem(stem_type, config)

        # Determine rebuild strategy based on threshold changes
        changed = _thresholds_changed(spectral_config, stored_logic)
        lowered = changed and _thresholds_lowered(spectral_config, stored_logic)

        if changed:
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
        else:
            # Thresholds unchanged — trust stored statuses from full pipeline
            events = list(configured_events)

            # Still apply manual overrides (user may have toggled individual events)
            _apply_overrides(events, stem_overrides)

        # Extract kept events for MIDI generation
        kept_events = [e for e in events if e.get('status') == 'KEPT']

        # Pass 2: Classify notes on the final KEPT set using stored features
        classify_notes(kept_events, stem_type, drum_mapping, config)

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

        # Update stored logic to reflect current thresholds and classification params
        stem_data['logic'] = _build_logic_block(
            spectral_config, stored_logic, stem_type, config,
        )

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
            logic['open_sustain_ms'] = stem_config.get('open_sustain_ms', 150.0)
        if stem_type in ('snare', 'toms', 'cymbals'):
            defaults = {'snare': 2, 'toms': 3, 'cymbals': 2}
            raw = stem_config.get('expected_clusters')
            logic['expected_clusters'] = int(raw) if raw is not None else defaults[stem_type]
            logic['cluster_feature'] = stem_config.get('cluster_feature', 'auto')
            cluster_note_map = stem_config.get('cluster_note_map')
            if cluster_note_map:
                logic['cluster_note_map'] = cluster_note_map

    return logic
