"""
Rebuild MIDI from Analysis — Functional Core

Re-filters cached detection results from analysis.json and produces
MIDI-ready events without re-running audio detection. This enables
sub-second parameter tuning iteration.

Pure functions — no I/O, no side effects.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .analysis_core import (
    get_spectral_config_for_stem,
    should_keep_onset,
    normalize_values,
    estimate_velocity,
)
from .config import DrumMapping


# ============================================================================
# Event Pool Construction
# ============================================================================


def _merge_event_pools(
    configured_events: List[Dict],
    sensitive_events: List[Dict],
    merge_window_sec: float = 0.015,
) -> List[Dict]:
    """
    Merge configured and sensitive event pools, deduplicating by time.

    Configured events take precedence when times overlap within the merge
    window. Sensitive-only events fill gaps (events detected at max
    sensitivity but not at configured settings).

    Args:
        configured_events: Events from configured-sensitivity detection.
        sensitive_events: Events from max-sensitivity detection.
        merge_window_sec: Time window for considering events as duplicates.

    Returns:
        Merged event list sorted by time, each annotated with 'source'.
    """
    # Index configured event times for fast lookup
    configured_times = {round(e['time'], 4) for e in configured_events}

    merged = []

    # Add all configured events (they are authoritative)
    for event in configured_events:
        entry = dict(event)
        entry['_source'] = 'configured'
        merged.append(entry)

    # Add sensitive events that don't overlap with configured events
    for event in sensitive_events:
        t = event['time']
        # Check if any configured event is within merge window
        is_duplicate = any(
            abs(t - ct) < merge_window_sec for ct in configured_times
        )
        if not is_duplicate:
            entry = dict(event)
            entry['_source'] = 'sensitive'
            merged.append(entry)

    # Sort by time
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
    Re-apply filtering thresholds to events using pre-computed spectral features.

    Events with 'override' flag retain their status regardless of thresholds.

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
        stem_overrides = overrides.get(stem_type, {})

        # Get current spectral config for this stem (reads thresholds from config)
        spectral_config = get_spectral_config_for_stem(stem_type, config)

        # Step 1: Merge event pools (configured + sensitive, deduplicated)
        merged = _merge_event_pools(configured_events, sensitive_events)

        # Step 2: Apply manual overrides
        _apply_overrides(merged, stem_overrides)

        # Step 3: Re-filter with current thresholds
        _refilter_events(merged, spectral_config)

        # Step 4: Extract kept events for MIDI generation
        kept_events = [e for e in merged if e.get('status') == 'KEPT']

        # Step 5: Generate MIDI events
        midi_events = _events_to_midi(
            kept_events, stem_type, drum_mapping, config, spectral_config,
        )
        midi_events_by_stem[stem_type] = midi_events

        # Step 6: Update analysis data with new statuses
        # Split merged back into configured and sensitive for storage
        updated_configured = [e for e in merged if e.get('_source') == 'configured']
        updated_sensitive = [e for e in merged if e.get('_source') == 'sensitive']

        # Clean internal fields before storage
        for event_list in [updated_configured, updated_sensitive]:
            for event in event_list:
                event.pop('_source', None)

        # Attach note/velocity to KEPT events in configured list
        midi_idx = 0
        kept_midi = [e for e in midi_events if e.get('note') != 44]  # Exclude foot-close
        for event in updated_configured:
            if event.get('status') == 'KEPT' and midi_idx < len(kept_midi):
                event['note'] = kept_midi[midi_idx]['note']
                event['velocity'] = kept_midi[midi_idx]['velocity']
                midi_idx += 1
            elif event.get('status') != 'KEPT':
                # Clear stale note/velocity from previously-KEPT events
                event.pop('note', None)
                event.pop('velocity', None)

        stem_data['events_configured'] = updated_configured
        stem_data['events_sensitive'] = updated_sensitive

    # Build updated analysis output
    updated_analysis = dict(analysis_data)
    updated_analysis['stems'] = updated_stems

    return updated_analysis, midi_events_by_stem
