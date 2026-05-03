# MIDI Timing Analysis - Working vs Broken

## The Problem
MIDI files play at wrong tempo in DAWs - notes are stretched/compressed.

## Old Working Code (One Month Ago)

### Writing MIDI (`midi.py` - `create_midi_file`)
- Uses **`midiutil.MIDIFile`** library
- Adds tempo: `midi.addTempo(track, time, tempo)` where tempo=120.0
- Converts times: `seconds_to_beats(time, tempo) = time_sec * (tempo / 60)`
  - At 120 BPM: 1 second = 2 beats
- Notes stored as **beats** (not ticks)

### Key Conversion Function (`analysis_core.py`)
```python
def seconds_to_beats(time_sec: float, tempo: float) -> float:
    beats_per_second = tempo / 60.0
    return time_sec * beats_per_second
```

### Reading MIDI (`midi.py` - `read_midi_notes`)
- Uses **`mido.MidiFile`** library
- Default tempo: `tempo = 500000` microseconds (=120 BPM)
- Uses `mido.tick2second()` to convert ticks → seconds

## Why Old Works
1. **Write**: Convert seconds → beats using formula: `beats = seconds * (tempo/60)`
2. **Store**: Notes in beats with tempo event (120 BPM)
3. **Read**: DAW uses stored tempo (120 BPM) to convert beats back to seconds
4. **Result**: Original timing preserved

## Current Broken Code
- Uses **`mido.MIDIFile`** with ticks_per_beat=480
- Conversion: `ticks = seconds * 480 * (tempo/60)`
- Added tempo event back but still has issues in DAWs

## The Fix
Revert to using **`midiutil.MIDIFile`** (like old working code).
