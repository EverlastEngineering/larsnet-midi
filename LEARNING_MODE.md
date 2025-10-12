# Threshold Learning Mode

This workflow helps you calibrate the detection thresholds for your specific drum recordings.

## Why Use Learning Mode?

Different drum recordings, drum types, and playing styles have different spectral characteristics. Learning mode helps you find the optimal `geomean_threshold` for your recordings by:

1. Exporting ALL detected onsets (even ones that would normally be rejected)
2. Marking rejected hits with velocity=1 (so you can easily see them)
3. Letting you manually verify/delete false positives in your DAW
4. Analyzing which hits you kept vs deleted to suggest optimal thresholds

## Workflow

### Step 1: Generate Learning MIDI

Run stems_to_midi in learning mode with ultra-sensitive detection:

```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_learning/ --learn --stems snare
```

This will:
- Use very sensitive onset detection (catches everything)
- Export ALL detections to `drums_learning.mid`
- Mark rejected hits (based on current threshold) with velocity=1
- Normal hits have normal velocity

### Step 2: Edit in DAW

1. Load `midi_learning/drums_learning.mid` into your DAW alongside the audio
2. You'll see:
   - **Velocity=1 hits**: Currently rejected (shown in red/low in most DAWs)
   - **Normal velocity hits**: Currently kept
3. Listen through and **delete any false positives**:
   - Delete hits that aren't real drum hits
   - Keep only the hits that match actual snare strikes
4. Save the edited MIDI as `drums_edited.mid` (in the same folder)

### Step 3: Learn Optimal Threshold

Analyze the differences between original and edited MIDI:

```bash
python stems_to_midi.py --learn-from-midi \\
    cleaned_stems/snare/drums.wav \\
    midi_learning/drums_learning.mid \\
    midi_learning/drums_edited.mid \\
    --learn-stem snare
```

This will:
- Compare which hits you kept vs deleted
- Analyze spectral characteristics of each group
- Suggest an optimal `geomean_threshold`
- Save a calibrated config to `midiconfig_calibrated.yaml`

### Step 4: Use Calibrated Config

```bash
# Rename or use the calibrated config
cp midiconfig_calibrated.yaml midiconfig.yaml

# Now process normally with optimized thresholds
python stems_to_midi.py -i cleaned_stems/ -o midi_final/ --stems snare
```

## Understanding the Output

When learning from MIDI, you'll see output like:

```
Learning thresholds for snare...
  Original detections: 450
  User kept: 380
  User removed: 70

  Analysis:
    Kept hits - GeoMean range: 285.3 - 1224.8
    Removed hits - GeoMean range: 15.2 - 245.7
    Suggested threshold: 265.5
    (Midpoint between max removed (245.7) and min kept (285.3))
```

**Interpretation:**
- **Real snare hits** have GeoMean 285-1225
- **False positives** have GeoMean 15-246
- **Suggested threshold: 265.5** - safely between the two groups

## Configuration

In `midiconfig.yaml`:

```yaml
learning_mode:
  enabled: false                    # Set to true or use --learn flag
  export_all_detections: true       # Export even rejected hits
  rejected_velocity: 1               # Velocity for rejected hits
  
  # Ultra-sensitive detection settings for learning
  learning_onset_threshold: 0.05    # Very low (normal is 0.15)
  learning_delta: 0.002              # Very sensitive (normal is 0.005)
  learning_wait: 1                   # Minimum gap between hits
```

## Tips

1. **Start conservative**: It's easier to remove false positives than to add missing hits
2. **Listen carefully**: Some quiet ghost notes might look like artifacts but are real
3. **Multiple passes**: You can iterate - run learning mode multiple times with different settings
4. **Per-song calibration**: Different songs may need different thresholds
5. **Save configs**: Keep calibrated configs for different drum styles/recordings

## Troubleshooting

**Too many false positives in learning mode?**
- The ultra-sensitive settings are intentional - you'll manually remove them
- If it's overwhelming, adjust `learning_onset_threshold` in config (try 0.10)

**Missing real hits in learning MIDI?**
- Lower `learning_onset_threshold` even more (try 0.03)
- Check `learning_delta` - lower = more sensitive (try 0.001)

**Learned threshold doesn't work well?**
- Check if there's overlap in GeoMean ranges (kept vs removed)
- May need to adjust spectral frequency ranges in config
- Try learning from multiple songs and averaging thresholds
