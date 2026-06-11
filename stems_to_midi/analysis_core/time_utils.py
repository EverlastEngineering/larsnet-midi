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
    
    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        tempo: Tempo in beats per minute

    Returns:
        List of events with times converted to beats, flattened from all stems
    """
    # Minimum duration to prevent MIDI library errors
    # At 120 BPM, 0.01 beats = 5ms
    MIN_DURATION_BEATS = 0.01

    prepared_events = []

    # BUGFIX (2026-06-10): the energy + spectral detection paths can
    # both produce a KEPT event for the same hit (same stem, same
    # time, same drum note). When that happens the events_configured
    # list has two events with identical (time, note) — and
    # midiutil's addNote emits a NoteOn+NoteOff pair for each one.
    # The two NoteOns share (pitch, channel, tick) but the two
    # NoteOffs go to different ticks (different durations). The
    # sort + deInterleaveNotes pass then loses track: it pairs the
    # first NoteOff with the first NoteOn, leaving the second
    # NoteOn orphaned, and the NEXT NoteOff for the same key tries
    # to pop an empty stack and dies with `IndexError: pop from
    # empty list`. The cleanest fix is to dedupe at the boundary
    # we control: keep the loudest velocity for each
    # (time, note) pair and drop the rest. Both events are KEPT
    # (we already filtered), so we're not losing signal — we're
    # collapsing a true simultaneous double-detection into a single
    # MIDI note. Without this the WebUI sees the toast "Failed:
    # pop from empty list" and the user loses the entire MIDI file
    # even though 99% of the events are valid.
    #
    # The dedup rounds time to 1ms (the same threshold
    # energy_detection_shell uses for raw-onset dedup). Anything
    # within the same millisecond on the same note is treated as a
    # duplicate. We key by (stem, note, rounded_time) so events
    # from different stems at the same instant (e.g. a kick+snare
    # hit) are NOT collapsed — they're independent MIDI notes.
    seen: dict = {}
    for stem_type, events in events_by_stem.items():
        for event in events:
            duration_beats = seconds_to_beats(event['duration'], tempo)

            # BUGFIX: Enforce minimum duration to prevent midiutil errors
            # Zero or negative durations cause "pop from empty list" in deInterleaveNotes
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
            # legitimately-coincident cross-stem hits. Keying on
            # stem_type keeps the MIDI semantically correct.
            key = (stem_type, event['note'], round(event['time'] * 1000))
            existing = seen.get(key)
            if existing is None or prepared_event['velocity'] > existing['velocity']:
                seen[key] = prepared_event

    prepared_events = list(seen.values())
    return prepared_events
