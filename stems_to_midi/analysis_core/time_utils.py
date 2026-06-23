"""
Time and MIDI Conversion Utilities

Pure helper functions for time conversion and MIDI event preparation.

Functions:
- seconds_to_beats: Convert seconds to beats based on tempo
- prepare_midi_events_for_writing: Convert times to beats and prepare for MIDI writing
"""

from typing import Dict, List


def seconds_to_beats(time_sec: float, tempo: float) -> float:
    """
    Convert time in seconds to beats based on tempo.

    Pure function - no side effects.

    Args:
        time_sec: Time in seconds
        tempo: Tempo in BPM (beats per minute)

    Returns:
        Time in beats
    """
    beats_per_second = tempo / 60.0
    return time_sec * beats_per_second


def prepare_midi_events_for_writing(
    events_by_stem: Dict[str, List[Dict]],
    tempo: float
) -> List[Dict]:
    """
    Prepare MIDI events for writing by converting times to beats.

    Pure function - no side effects.

    Deduplicates events with the same (stem, note, time-ms) tuple
    by keeping the loudest velocity. This collapses true
    simultaneous double-detections (the energy and spectral paths
    can both produce a KEPT event for the same hit) into a single
    MIDI note, preventing midiutil's `deInterleaveNotes` from
    orphaning NoteOns and crashing on a NoteOff pop.

    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        tempo: Tempo in beats per minute

    Returns:
        List of events with times converted to beats, flattened from all stems
    """
    # Minimum duration to prevent MIDI library errors
    # At 120 BPM, 0.01 beats = 5ms
    MIN_DURATION_BEATS = 0.01

    seen: dict = {}
    for stem_type, events in events_by_stem.items():
        for event in events:
            duration_beats = seconds_to_beats(event['duration'], tempo)

            # BUGFIX: Enforce minimum duration to prevent midiutil errors
            if duration_beats < MIN_DURATION_BEATS:
                duration_beats = MIN_DURATION_BEATS

            prepared_event = {
                'note': event['note'],
                'velocity': event['velocity'],
                'time_beats': seconds_to_beats(event['time'], tempo),
                'duration_beats': duration_beats,
                'stem_type': stem_type
            }

            # 1ms-rounded key per (stem, note). Different stems
            # with the same note number share pitch+channel on
            # midiutil's side, so we'd otherwise collapse
            # legitimately-coincident cross-stem hits.
            key = (stem_type, event['note'], round(event['time'] * 1000))
            existing = seen.get(key)
            if existing is None or prepared_event['velocity'] > existing['velocity']:
                seen[key] = prepared_event

    return list(seen.values())
