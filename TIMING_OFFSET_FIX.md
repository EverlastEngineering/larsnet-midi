# Timing Offset Fix - Apply at MIDI Creation, Not Detection

**Date**: October 18, 2025  
**Status**: ✅ Fixed  
**Issue**: Large timing offsets (> ±0.2s) caused notes to disappear

## Problem

When `timing_offset` was applied during onset detection (in `detect_onsets()`), it shifted the onset times BEFORE spectral analysis happened. This caused:

1. **Onset detection** finds kick at 1.0s (correct audio location)
2. **timing_offset: 0.30** shifts onset time to 1.30s
3. **Spectral analysis** tries to analyze audio at 1.30s (wrong location - no kick energy there!)
4. **Filtering** sees low energy → rejects as artifact → **note disappears**

With large offsets (±0.2s or more), most/all notes disappeared because spectral analysis was looking at the wrong part of the audio.

## Solution

Move `timing_offset` application from **onset detection** to **MIDI event creation**.

### Architecture Flow

**OLD (Broken)**:
```
Onset Detection → Apply timing_offset → Spectral Analysis → MIDI Creation
                   (shifts audio analysis location!)
```

**NEW (Fixed)**:
```
Onset Detection → Spectral Analysis → MIDI Creation → Apply timing_offset
                                                        (only shifts MIDI timing)
```

### Key Principle

- **Audio analysis** (spectral energy, peak amplitude) must happen at the **actual audio location** where the hit occurs
- **MIDI timing** can be shifted independently to compensate for perceived timing issues

## Changes Made

### 1. Removed `timing_offset` from `detect_onsets()`

**File**: `stems_to_midi/detection.py`

- Removed `timing_offset` parameter from function signature (line ~131)
- Removed timing offset application code (was on line ~238)
- Added comment explaining timing offset is applied later

### 2. Removed `timing_offset` from `detect_onsets()` call

**File**: `stems_to_midi/processor.py`

- Removed `timing_offset` argument from `detect_onsets()` call (line ~362)

### 3. Added `timing_offset` to `_create_midi_events()`

**File**: `stems_to_midi/processor.py`

- Reads `timing_offset` from stem config (line ~260)
- Applies offset when setting MIDI event time: `midi_time = float(time) + timing_offset` (line ~297)

## Configuration

Settings remain in `midiconfig.yaml` per stem type:

```yaml
kick:
  timing_offset: 0.030    # Shift MIDI events 30ms later
  
snare:
  timing_offset: 0.0      # No shift (default)
  
hihat:
  timing_offset: -0.010   # Shift MIDI events 10ms earlier
```

## Benefits

1. ✅ **Large offsets work** - Can now use values like ±0.3s without losing notes
2. ✅ **Spectral analysis accuracy** - Always analyzes audio at correct location
3. ✅ **Independent timing** - Can fine-tune MIDI timing without affecting detection
4. ✅ **Clean separation** - Audio analysis separate from MIDI timing adjustment

## Testing

After this fix, you should be able to:
- Use `timing_offset: 0.30` without notes disappearing
- Use `timing_offset: -0.30` without notes disappearing  
- Adjust timing independently per stem type
- Maintain full spectral filtering accuracy

## Typical Values

- **Kick**: `0.020 - 0.040` (20-40ms later, compensates for slow attack)
- **Snare**: `0.0 - 0.020` (0-20ms, usually accurate)
- **Hi-hat**: `-0.010 - 0.010` (fine-tune ±10ms)
- **Toms**: `0.0 - 0.030` (similar to kick)
- **Cymbals**: `0.0 - 0.020` (usually accurate)

Extreme values (> 0.1s) are possible but unusual - check your detection settings first!
