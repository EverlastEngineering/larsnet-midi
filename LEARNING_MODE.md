# Threshold Learning Mode

This workflow helps you calibrate the spectral filtering thresholds for your specific drum recordings.

## Why Use Learning Mode?

Different drum recordings, drum types, and playing styles have different spectral characteristics. Learning mode helps you find the optimal `geomean_threshold` for your recordings by:

1. Exporting ALL detected onsets (even ones that would normally be rejected)
2. Marking rejected hits with velocity=1 (so you can easily see them in your DAW)
3. Letting you manually verify/delete false positives in your DAW
4. Analyzing which hits you kept vs deleted to suggest optimal thresholds

## The Spectral Filtering Methodology

### Problem: Drum Bleed and Artifacts

When separating drums with neural networks, you get:
- **Real hits**: The actual drum you want
- **Bleed**: Other drums bleeding through (e.g., kick hitting on snare track)
- **Artifacts**: Neural network processing artifacts

### Solution: Spectral Analysis

Each drum has a unique spectral signature. We analyze the frequency content of each detected onset:

#### Snare Drum
- **Body energy** (150-400 Hz): The fundamental and body resonance
- **Wire energy** (2-8 kHz): The snare wires rattling
- **GeoMean** = √(Body × Wire): Combined discriminator
- **Real snares**: High body AND high wire energy (GeoMean 250-1200)
- **Kick bleed**: High body, LOW wire energy (GeoMean 20-100)

#### Kick Drum  
- **Fundamental energy** (40-80 Hz): The deep bass thump
- **Body energy** (80-150 Hz): The beater attack and shell resonance
- **Attack energy** (2-6 kHz): The high-frequency beater click/snap
- **GeoMean** = ∛(Fundamental × Body × Attack): Combined discriminator (3-way)
- **Real kicks**: High fundamental AND high body AND high attack (GeoMean 100-400)
- **Artifacts**: May have some energy but not all three (GeoMean 10-80)

### Why Geometric Mean?

The geometric mean (√(A × B)) requires BOTH frequency ranges to have energy:
- If either frequency range is weak, GeoMean is low → REJECT
- Only when BOTH ranges are strong does GeoMean get high → KEEP

This is more effective than:
- Simple ratio (A/B): Can be fooled by just one strong frequency
- Sum (A+B): Doesn't require both to be present
- Product (A×B): Too sensitive to scale differences

## Workflow

### Step 1: Generate Learning MIDI

Run stems_to_midi in learning mode with ultra-sensitive detection:

```bash
# For snare
python stems_to_midi.py -i cleaned_stems/ -o midi_learning/ --learn --stems snare

# For kick
python stems_to_midi.py -i cleaned_stems/ -o midi_learning/ --learn --stems kick

# Or process multiple stems
python stems_to_midi.py -i cleaned_stems/ -o midi_learning/ --learn --stems snare kick

# For long tracks (10+ minutes), use --maxtime to analyze only first N seconds
python stems_to_midi.py -i cleaned_stems/ -o midi_learning/ --learn --stems snare --maxtime 50
```

This will:
- Use very sensitive onset detection (catches everything, even noise)
- Export ALL detections to `drums_learning.mid`
- Mark rejected hits (based on current threshold) with velocity=1
- Real hits have normal velocity (40-127)
- Show spectral analysis for each onset in the console

**Tip**: The `--maxtime` option (in seconds) makes learning much faster for long tracks. Usually 30-60 seconds is enough to get representative samples of your drum hits.

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
python stems_to_midi.py --learn-from-midi \
    cleaned_stems/snare/drums.wav \
    midi_learning/drums_learning.mid \
    midi_learning/drums_edited.mid \
    --learn-stem snare

# For long tracks, use --maxtime to analyze only the same portion you used in Step 1
python stems_to_midi.py --learn-from-midi \
    cleaned_stems/snare/drums.wav \
    midi_learning/drums_learning.mid \
    midi_learning/drums_edited.mid \
    --learn-stem snare \
    --maxtime 50
```

This will:
- Compare which hits you kept vs deleted
- Analyze spectral characteristics of each group
- Suggest an optimal `geomean_threshold`
- Save a calibrated config to `midiconfig_calibrated.yaml`

**Important**: If you used `--maxtime` in Step 1, use the same value in Step 3 to analyze the same audio segment.

### Step 4: Use Calibrated Config

```bash
# Rename or use the calibrated config
cp midiconfig_calibrated.yaml midiconfig.yaml

# Now process normally with optimized thresholds
python stems_to_midi.py -i cleaned_stems/ -o midi_final/ --stems snare
```

## Understanding the Output

### During Learning Mode (Step 1)

You'll see spectral analysis for each onset:

```
ALL DETECTED ONSETS - SPECTRAL ANALYSIS:
Using GeoMean threshold: 100
Str=Onset Strength, Amp=Peak Amplitude, BodyE=Body Energy (150-400Hz), WireE=Wire Energy (2-8kHz)
GeoMean=sqrt(BodyE*WireE) - measures combined spectral energy

    Time      Str    Amp    BodyE    WireE    Total  GeoMean     Status
   0.245    0.834  0.156   856.3  1247.8   2104.1    1033.2       KEPT
   0.456    0.412  0.089   124.5    45.2    169.7      74.5   REJECTED
   0.789    0.923  0.201  1124.7  1856.3   2981.0    1441.3       KEPT
```

**What to look for:**
- **KEPT hits**: Should be real drum hits (high GeoMean)
- **REJECTED hits**: Should be artifacts/bleed (low GeoMean)
- If many REJECTED look real, threshold is too high
- If many KEPT look fake, threshold is too low

### After Learning from MIDI (Step 3)

You'll see analysis of what you kept vs removed:

```
Learning thresholds for snare...
  Original detections: 450
  User kept: 380
  User removed: 70

  Spectral Analysis:
    Kept hits - GeoMean range: 285.3 - 1224.8 (mean: 654.2)
    Removed hits - GeoMean range: 15.2 - 245.7 (mean: 87.3)
    
    Suggested threshold: 265.5
    (Midpoint between max removed (245.7) and min kept (285.3))
    
    Separation quality: EXCELLENT (19.8 gap between groups)

  Saved calibrated configuration to: midiconfig_calibrated.yaml
```

**Interpretation:**
- **Real snare hits** have GeoMean 285-1225
- **False positives** have GeoMean 15-246
- **Suggested threshold: 265.5** - safely between the two groups
- **Gap of 19.8**: Good separation between kept/removed (bigger = better)

### Typical GeoMean Ranges

Based on analysis of various recordings:

| Drum  | Real Hits | Bleed/Artifacts | Typical Threshold |
|-------|-----------|-----------------|-------------------|
| Snare | 250-1200  | 20-100          | 100-150           |
| Kick  | 200-800   | 20-150          | 150-200           |
| Toms  | TBD       | TBD             | TBD               |

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

## Tips & Best Practices

### General Workflow Tips

1. **Start conservative**: It's easier to remove false positives than to add missing hits
2. **Listen carefully**: Some quiet ghost notes might look like artifacts but are real
3. **Multiple passes**: You can iterate - run learning mode multiple times with different settings
4. **Per-song calibration**: Different songs may need different thresholds
5. **Save configs**: Keep calibrated configs for different drum styles/recordings

### When to Adjust Spectral Ranges

Usually you don't need to change the frequency ranges, but if you have unusual drums:

**Snare with different tuning:**
- Higher tuned snare: Increase `body_freq_min` and `body_freq_max` (try 200-500 Hz)
- Lower tuned snare: Decrease ranges (try 120-350 Hz)

**Kick with different characteristics:**
- Sub kick/808: Lower fundamental range (try 35-70 Hz)
- Punchy rock kick: Focus on attack (increase `body_freq_max` to 200 Hz)

**Edit these in `midiconfig.yaml`:**
```yaml
snare:
  body_freq_min: 150    # Lower = deeper snares
  body_freq_max: 400    # Higher = higher tuned snares
  wire_freq_min: 2000   # Wire rattle start
  wire_freq_max: 8000   # Wire rattle end
```

### Interpreting Separation Quality

After learning, look at the gap between max removed and min kept:

- **Gap > 50**: Excellent separation, threshold will work great
- **Gap 20-50**: Good separation, threshold should work well  
- **Gap 5-20**: Moderate separation, may have edge cases
- **Gap < 5**: Poor separation, consider adjusting frequency ranges or using different approach

### Working with Multiple Recordings

If you process multiple songs with different drum sounds:

1. Run learning mode on each song separately
2. Note the suggested thresholds for each
3. Use the average as your general threshold
4. Or keep separate configs for different genres/drum types

## Troubleshooting

### Too many false positives in learning mode?
- **This is normal!** The ultra-sensitive settings are intentional - you'll manually remove them in your DAW
- If it's overwhelming (thousands of hits), adjust `learning_onset_threshold` in config (try 0.10 instead of 0.05)
- Remember: velocity=1 hits are already flagged as "probably bad" - focus on removing those first

### Missing real hits in learning MIDI?
- Lower `learning_onset_threshold` even more (try 0.03 or 0.02)
- Check `learning_delta` - lower = more sensitive (try 0.001)
- Reduce `learning_wait` (try 0 for very fast passages)
- Verify the audio file has good separation (run LarsNet separation again if needed)

### Learned threshold doesn't work well?
- **Overlap in ranges**: Check if kept and removed GeoMeans overlap significantly
  - If yes: Spectral characteristics may not be distinctive enough
  - Try adjusting frequency ranges in config
  - May need different filtering approach for this recording
  
- **Threshold too sensitive**: Many real hits being rejected
  - Lower the threshold manually in `midiconfig.yaml`
  - Or run learning mode again with more careful editing in DAW
  
- **Threshold too loose**: Many artifacts still getting through
  - Raise the threshold manually
  - Or learn from multiple songs and use highest threshold that keeps all real hits

### Spectral analysis shows unexpected values?
- **All energies very low**: Audio might be too quiet (check levels, normalize if needed)
- **All energies very high**: Audio might be clipping (check source files)
- **Frequency ranges seem wrong**: You may have unusual drum tuning - adjust ranges in config

### Learning from MIDI fails?
- **"MIDI notes don't match"**: Make sure you're analyzing the correct stem type with `--learn-stem`
- **"Not enough data"**: Need at least a few kept and removed hits to calculate threshold
- **File not found**: Check paths to original and edited MIDI files

## Advanced: Understanding the Math

The geometric mean is calculated as:

```
GeoMean = √(Primary_Energy × Secondary_Energy)
```

Why this works:
- If either energy is 0, GeoMean = 0 (instant rejection)
- If one energy is weak (10) and other strong (1000), GeoMean = 100 (still relatively weak)
- Only when both are strong (500 × 800) does GeoMean = 632 (strong signal)

For comparison with other metrics:
- **Arithmetic mean**: (A + B) / 2 = Can be high even if one is 0
- **Ratio**: A / B = Can be any value, doesn't indicate magnitude
- **Product**: A × B = Too large, hard to set thresholds
- **Geometric mean**: √(A × B) = Balanced discriminator that requires both

## Next Steps

After calibrating thresholds for snare and kick:
1. Apply same methodology to toms (different frequency ranges needed)
2. Work on hi-hat open/closed detection (already implemented)
3. Separate crash vs ride cymbals (spectral analysis of decay/sustain)
4. Export to sheet music (MusicXML format) - see TODO.md
