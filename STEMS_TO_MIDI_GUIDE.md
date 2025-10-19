# Stems to MIDI Conversion Guide

**Convert your drum stems to MIDI in 30 seconds** — retrigger with better samples, quantize timing, or study real drum performances.

---

## Quick Start (The TL;DR)

**Got cleaned drum stems? Run this:**

```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_output/
```

**That's it!** Import the MIDI files into your DAW and load your favorite drum samples.

### What Just Happened?

The script analyzed your stems, detected every drum hit, estimated velocities, and created MIDI files using the General MIDI standard. Your files are in `midi_output/`.

### MIDI Note Mapping (What You Need for Your DAW)

| Drum | MIDI Note | Note Name |
|------|-----------|-----------|
| **Kick** | 36 | C1 |
| **Snare** | 38 | D1 |
| **Hi-Hat (Closed)** | 42 | F#1 |
| **Hi-Hat (Open)** | 46 | A#1 |
| **Toms** | 45 | A1 (Low Tom) |
| **Cymbals** | 49 | C#2 (Crash) |

Most drum VSTs use this General MIDI mapping by default.

---

## Common Adjustments

**Missing quiet hits?** Lower the sensitivity:
```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_output/ -t 0.2
```

**Too many false triggers?** Raise the threshold:
```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_output/ -t 0.5
```

**Want more dynamics?** Expand velocity range:
```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --min-vel 1 --max-vel 127
```

**Need consistent levels?** Narrow velocity range:
```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --min-vel 80 --max-vel 100
```

---

## Configuration-Driven Processing

The script uses `midiconfig.yaml` for advanced control over detection and filtering. **No command-line editing needed** — just edit the YAML file!

### Key Configuration Sections:

**1. Audio Processing (`audio`):**
- Silence threshold, segment lengths, envelope detection
- All hardcoded values now configurable

**2. Per-Stem Settings (`kick`, `snare`, `toms`, `hihat`, `cymbals`):**
- Custom onset detection thresholds
- Spectral filtering ranges (frequency analysis)
- Geomean thresholds for artifact rejection

**3. Learning Mode (`learning_mode`):**
- Calibrate thresholds from your edited MIDI
- See `LEARNING_MODE.md` for details

### Architecture: Functional Core, Imperative Shell

The codebase follows a clean architecture pattern:

- **`stems_to_midi_helpers.py`**: Pure, testable functions (86% coverage)
  - Audio analysis, spectral filtering, sustain detection
  - Zero side effects, fully deterministic
  
- **`stems_to_midi.py`**: Thin coordination layer
  - Handles I/O, prints, MIDI creation
  - Orchestrates the functional core

**Benefits:** Easy to test, maintain, and extend. Changes to helpers automatically benefit all functions that use them.

---

## Prerequisites

The script requires these Python packages:

```bash
pip install librosa midiutil numpy soundfile scipy
```

**Check your installation:**
```bash
pip list | grep -E "(librosa|midiutil|scipy|soundfile)"
```

---

## Why Convert to MIDI?

✅ **Retriggering**: Use your favorite drum samples  
✅ **Quantization**: Fix timing issues  
✅ **Editing**: Easy to adjust individual hits  
✅ **Mixing**: Replace or blend with original audio  
✅ **MIDI Learn**: Trigger lights, visuals, or other devices  
✅ **Drum Programming**: Study and learn from real performances

---

## Complete Workflow Example

Here's the full pipeline from audio file to MIDI:

```bash
# Step 1: Separate drums with LarsNet
python separate.py -i input/ -o separated_stems/ -w 1.5

# Step 2: Clean up kick/snare bleed
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/

# Step 3: Convert to MIDI
python stems_to_midi.py -i cleaned_stems/ -o midi_output/

# Step 4: Import MIDI into your DAW and load drum samples!
```

---

## Command Options Reference

```bash
python stems_to_midi.py -i <input_dir> -o <output_dir> [OPTIONS]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i` / `--input_dir` | *required* | Directory with separated stems |
| `-o` / `--output_dir` | `midi_output` | Where to save MIDI files |
| `-t` / `--threshold` | `0.3` | Detection sensitivity (0-1): lower = more sensitive |
| `--min-vel` | `40` | Minimum MIDI velocity (1-127) |
| `--max-vel` | `127` | Maximum MIDI velocity (1-127) |
| `--tempo` | `120.0` | Tempo in BPM (metadata only) |
| `--detect-hihat-open` | *off* | Try to detect open hi-hats |
| `--stems` | *all* | Process specific stems only |

---

## Real-World Examples

### Test Multiple Settings (Recommended)

When unsure, try all three and pick the best:

```bash
# Conservative (fewer notes, cleaner)
python stems_to_midi.py -i cleaned_stems/ -o midi_conservative/ -t 0.5

# Balanced (default - start here)
python stems_to_midi.py -i cleaned_stems/ -o midi_balanced/ -t 0.3

# Sensitive (more notes, catches ghost notes)
python stems_to_midi.py -i cleaned_stems/ -o midi_sensitive/ -t 0.2
```

### Specific Use Cases

**Specific tempo:**
```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --tempo 140
```

**Process only kick and snare:**
```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --stems kick snare
```

**Enable open/closed hi-hat detection:**
```bash
python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --detect-hihat-open
```

**Per-stem processing with different settings:**
```bash
# Kick: Less sensitive (avoid false triggers)
python stems_to_midi.py -i cleaned_stems/ -o midi_kick/ --stems kick -t 0.4

# Snare: Balanced
python stems_to_midi.py -i cleaned_stems/ -o midi_snare/ --stems snare -t 0.3

# Hi-hat: More sensitive (catch ghost notes)
python stems_to_midi.py -i cleaned_stems/ -o midi_hihat/ --stems hihat -t 0.2
```

Then manually combine the MIDI files in your DAW.

---

## Complete Workflow

### Full Pipeline: Separation → Cleanup → MIDI

```bash
# Step 1: Separate drums with Wiener filter
python separate.py -i input/ -o separated_stems/ -w 1.5

# Step 2: Clean up kick/snare bleed with sidechain
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/

# Step 3: Convert to MIDI
python stems_to_midi.py -i cleaned_stems/ -o midi_output/
```

### Testing Different Detection Settings

When you're unsure of the best settings, try multiple configurations:

```bash
# Conservative (fewer notes)
python stems_to_midi.py -i cleaned_stems/ -o midi_conservative/ -t 0.5

# Balanced (default)
python stems_to_midi.py -i cleaned_stems/ -o midi_balanced/ -t 0.3

# Sensitive (more notes)
python stems_to_midi.py -i cleaned_stems/ -o midi_sensitive/ -t 0.2
```

Then compare the results and pick the best one!

---

## Troubleshooting Quick Reference

| Problem | Quick Fix |
|---------|-----------|
| **Missing quiet hits** | Lower threshold: `-t 0.2` |
| **Too many false triggers** | Raise threshold: `-t 0.5` |
| **No velocity dynamics** | Expand range: `--min-vel 1 --max-vel 127` |
| **Wrong hi-hat open/closed** | Disable detection: `--detect-hihat-open` (off by default) |
| **Timing doesn't match** | Set correct tempo: `--tempo 140` |
| **Wrong drum sounds** | Check your VST uses General MIDI mapping |

### Detailed Solutions

**Missing drum hits:**
1. Lower threshold: `-t 0.2` or `-t 0.15`
2. Verify the stem actually contains those hits
3. Check if heavy compression reduced transients

**Too many false notes:**
1. Raise threshold: `-t 0.4` or `-t 0.5`
2. Use sidechain cleanup first
3. Check for bleed from other instruments

**No velocity variation:**
1. Increase velocity range: `--min-vel 1 --max-vel 127`
2. Check if source audio is heavily compressed
3. Verify your drum sampler is velocity-sensitive

---

## Using MIDI in Your DAW

### Quick Import

1. Drag MIDI file into your DAW
2. Load a drum sampler/VST (ensure it uses General MIDI mapping)
3. Done! Play and adjust as needed

### DAW-Specific Tips

| DAW | Recommended Drum Plugin |
|-----|------------------------|
| **Ableton Live** | Drum Rack (map to General MIDI) |
| **Logic Pro** | Drum Kit Designer / Drum Machine Designer |
| **Pro Tools** | Strike or any GM-compatible plugin |
| **FL Studio** | FPC or any drum sampler |

### Common Edits

- **Quantize** to fix timing
- **Adjust velocities** for dynamics
- **Delete** false positives
- **Draw in** missed hits
- **Layer** with original audio for hybrid sound

### Blending Strategies

1. **Layering**: Keep audio + MIDI-triggered samples
2. **Replacement**: Mute audio, use only samples
3. **Reinforcement**: Blend (e.g., add sub to kick)
4. **Parallel processing**: Different effects on each

---

## Best Practices

**🎯 For Best Results:**
- Always use cleaned stems (after sidechain cleanup)
- Start with default settings, adjust only if needed
- Test multiple threshold values and compare
- Import MIDI with audio to visually verify alignment

**📁 Recommended Folder Structure:**
```
project/
├── input/               # Original mixes
├── separated_stems/     # LarsNet output
├── cleaned_stems/       # After sidechain cleanup
├── midi_output/         # Generated MIDI
└── final_mix/           # Completed project
```

**⚡ Performance:**
- Processing is typically real-time or faster
- CPU-based (no GPU required)
- Batch process entire albums with ease
- MIDI files are tiny (< 10 KB each)

---

## FAQ

**Q: Do I need to know the tempo?**  
A: No! MIDI events use absolute time. Tempo is just metadata.

**Q: Can I use this on live drum recordings?**  
A: Yes, but separate them with LarsNet first.

**Q: Will this work with electronic drums?**  
A: Yes! Often works even better due to cleaner transients.

**Q: How do I handle drum rolls?**  
A: Lower threshold (`-t 0.15`) to catch fast notes. May need manual cleanup.

**Q: Timing is off in my DAW?**  
A: Verify sample rate (44.1kHz) and DAW tempo match the MIDI file.

**Q: Can I merge multiple MIDI files?**  
A: Yes, in any DAW or MIDI editor.

**Q: How do I fix wrong velocities?**  
A: Adjust `--min-vel` and `--max-vel`, or edit in your DAW.

---

## How It Works (Technical Details)

### The Algorithm

1. **Load audio** from each stem
2. **Detect onsets** using librosa (spectral flux + peak picking)
3. **Estimate velocities** from transient energy
4. **Classify hi-hats** by analyzing decay envelope (if enabled)
5. **Generate MIDI** with accurate timing and velocities

### Threshold Parameter Explained

The `-t` threshold controls sensitivity:

| Value | Sensitivity | Use Case |
|-------|-------------|----------|
| **0.1-0.2** | High | Catches ghost notes, may have false positives |
| **0.3-0.4** | Medium | ✅ **Recommended starting point** |
| **0.5-0.7** | Low | Only strong hits, fewer errors |

### Accuracy

- ✅ **90-95%** for clean, well-separated stems
- ⚠️ **70-85%** for stems with bleed or complex rhythms
- 📝 Manual editing may be needed for perfection

### Limitations

- Can't detect hits that LarsNet didn't separate
- Velocity estimation is approximate
- Hi-hat open/closed detection not 100% reliable
- Very fast rolls (32nd notes+) may be incomplete
- Best with percussive, transient-rich drums

---

## Advanced Features

### Spectral Filtering

The detection system uses multi-band spectral analysis to distinguish real drum hits from artifacts and bleed:

**Kick Drum - 3-Way Filtering:**
- **Fundamental** (40-80 Hz): Deep bass thump
- **Body** (80-150 Hz): Shell resonance and punch  
- **Attack** (2000-6000 Hz): Beater click
- Calculates: `GeoMean = ∛(Fundamental × Body × Attack)`
- Typical threshold: 70-150

**Snare Drum - 2-Way Filtering:**
- **Body** (150-400 Hz): Shell resonance
- **Wires** (2000-8000 Hz): Snare buzz
- Calculates: `GeoMean = √(Body × Wires)`
- Typical threshold: 20-50

**Configuration:** Edit `midiconfig.yaml` to adjust frequency ranges and thresholds per drum type.

### Statistical Outlier Detection (Kick Only)

An optional second-pass filter that catches snare bleed by analyzing spectral signatures:

1. Calculates median FundE/BodyE ratio across all detected kicks
2. Scores each onset based on deviation from population
3. Rejects statistical outliers (likely bleed or artifacts)

**Enable in `midiconfig.yaml`:**
```yaml
kick:
  enable_statistical_filter: true
  statistical_badness_threshold: 0.6  # 0-1, higher = more permissive
```

**When to use:** Tracks with heavy snare bleed in the kick channel that passes basic geomean filtering.

### Timing Offset Correction

Per-stem timing adjustment to compensate for perceived latency:

```yaml
kick:
  timing_offset: 0.030    # Shift MIDI 30ms later (kick often feels early)
hihat:
  timing_offset: -0.010   # Shift 10ms earlier (hi-hat often feels late)
```

**Applied during MIDI creation** - does not affect spectral analysis location. Can use large values (±0.3s) without breaking detection.

### Learning Mode

Automatically calibrate thresholds from user-edited MIDI:

```bash
# 1. Generate initial MIDI
python stems_to_midi.py -i stems/ -o initial_midi/

# 2. Edit in DAW - mark correct hits, delete false positives

# 3. Re-run with learning mode
python stems_to_midi.py -i stems/ -o improved_midi/ --learn-from initial_midi/edited.mid
```

The system analyzes which onsets you kept/deleted and suggests optimal threshold values. See `LEARNING_MODE.md` for details.

---

## Advanced Customization

### Custom MIDI Note Mapping

Edit `DrumMapping` class in `stems_to_midi.py`:

```python
@dataclass
class DrumMapping:
    kick: int = 36      # Change to your preferred MIDI note
    snare: int = 38
    # ... etc
```

### Custom Onset Detection

Modify `detect_onsets()` function parameters:

```python
onset_frames = librosa.onset.onset_detect(
    pre_max=3,      # Frames before peak
    post_max=3,     # Frames after peak
    delta=0.07,     # Peak picking threshold
    wait=10         # Min frames between peaks
)
```

### Velocity Curve Adjustment

Modify `estimate_velocity()` for different dynamics:

```python
# More aggressive dynamics
curved_strength = strength ** 1.5

# Compressed/leveled dynamics
curved_strength = np.sqrt(strength)
```

---

## Additional MIDI Note Reference

The script defines these General MIDI notes (can be customized):

- **Kick Alt**: 35 (Bass Drum 2)
- **Snare Alt**: 40 (Electric Snare)
- **Tom Low**: 45, **Tom Mid**: 47, **Tom High**: 50
- **Ride Cymbal**: 51
- **China Cymbal**: 52
- **Handclap**: 39

---

**Happy drum programming! 🥁🎹**
