# Sidechain Cleanup Guide

## Overview

When separating drum tracks with LarsNet, you may experience **bleed** between stems - particularly snare frequencies bleeding into the kick track. This guide explains how to use **sidechain compression** to dramatically reduce this bleed and get cleaner separated stems.

## The Problem

Neural network source separation models sometimes struggle with frequency overlap between instruments:
- **Kick and snare** share similar frequency ranges (especially in the low-mids: 80-250 Hz)
- The snare's "body" can bleed into the kick track
- Traditional EQ can help but may remove important kick frequencies too

## The Solution: Sidechain Compression

**Sidechain compression** uses the separated snare track as a trigger to automatically "duck" (reduce) the kick track whenever the snare plays. Since kick and snare hits typically don't occur at the exact same moment, this effectively removes snare bleed without affecting the actual kick hits.

### Why This Works

1. ✅ **Temporal separation**: Kick and snare hits occur at different times
2. ✅ **Surgical precision**: Only ducks when snare is actually present
3. ✅ **Preserves kick**: Real kick hits remain untouched
4. ✅ **Automatic**: No manual editing required

---

## Quick Start

### Step 1: Separate Drums

First, separate your drum mix using LarsNet with Wiener filtering:

```bash
python separate.py -i input/ -o separated_stems/ -w 1.5
```

**Parameters:**
- `-i`: Input directory containing your drum mix WAV files
- `-o`: Output directory for separated stems
- `-w`: Wiener filter exponent (1.0-2.0 recommended, higher = more aggressive separation)
- `-d`: Device (`cpu` or `cuda` for GPU)

### Step 2: Apply Sidechain Cleanup

Run the sidechain cleanup on your separated stems:

```bash
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/
```

**That's it!** Your cleaned stems will be in `cleaned_stems/` with snare bleed removed from the kick track.

---

## Advanced Usage

### Basic Parameters

```bash
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/ [OPTIONS]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i` / `--input_dir` | *required* | Directory with separated stems (must contain track subdirs with stem files) |
| `-o` / `--output_dir` | `cleaned_stems` | Where to save cleaned stems |
| `-t` / `--threshold` | `-30.0` | Sidechain threshold in dB (lower = more sensitive) |
| `-r` / `--ratio` | `10.0` | Compression ratio (higher = more aggressive ducking) |
| `--attack` | `1.0` | Attack time in ms (how fast compression kicks in) |
| `--release` | `100.0` | Release time in ms (how long kick stays ducked) |
| `--dry-wet` | `1.0` | Mix: 0.0 = original, 1.0 = fully processed |

### Preset Configurations

#### 🔥 **Aggressive (Heavy Bleed)**
For tracks with severe snare bleed in kick:

```bash
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/ \
  -t -40 -r 20 --attack 0.5 --release 80
```

#### 🎯 **Balanced (Recommended)**
Default settings work well for most tracks:

```bash
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/
```

#### 🎵 **Gentle (Preserve Dynamics)**
For subtle cleanup while maintaining natural sound:

```bash
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/ \
  -t -25 -r 4 --release 150 --dry-wet 0.7
```

#### ⚡ **Fast & Tight**
For quick transients and minimal ducking time:

```bash
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/ \
  --attack 0.3 --release 50
```

---

## Parameter Guide

### Threshold (`-t` / `--threshold`)

**What it does:** Determines how loud the snare needs to be to trigger ducking.

- **Lower values** (e.g., `-40 dB`): More sensitive, ducks on quieter snare hits
- **Higher values** (e.g., `-25 dB`): Less sensitive, only ducks on loud snare hits

**Recommendations:**
- Start with **-30 dB** (default)
- If you still hear snare bleed: try **-35 to -40 dB**
- If kick sounds overly ducked: try **-25 to -28 dB**

### Ratio (`-r` / `--ratio`)

**What it does:** Controls how much the kick is reduced when snare is present.

- **Higher ratios** (e.g., `20:1`): More aggressive ducking
- **Lower ratios** (e.g., `4:1`): Gentler, more subtle reduction

**Recommendations:**
- **10:1** (default) - Good starting point for bleed removal
- **15-20:1** - Heavy bleed situations
- **4-6:1** - Gentle, musical ducking

### Attack (`--attack`)

**What it does:** How quickly compression kicks in when snare is detected.

- **Faster** (e.g., `0.5 ms`): Instant response, catches sharp transients
- **Slower** (e.g., `5 ms`): More gentle, lets some initial transient through

**Recommendations:**
- **0.5-1 ms** - Best for snare transients (default: 1 ms)
- **2-5 ms** - More musical/natural sounding
- Snare has fast attack, so keep this low!

### Release (`--release`)

**What it does:** How long the kick stays ducked after snare stops.

- **Faster** (e.g., `50 ms`): Quick recovery, kick comes back quickly
- **Slower** (e.g., `150 ms`): Longer ducking, more sustained reduction

**Recommendations:**
- **80-100 ms** (default: 100 ms) - Balanced, natural
- **50-70 ms** - Fast recovery for tight grooves
- **120-150 ms** - Longer ducking for sustained snare bleed

### Dry/Wet (`--dry-wet`)

**What it does:** Blends processed (wet) with original (dry) signal.

- **1.0**: 100% processed (fully ducked)
- **0.7**: 70% processed, 30% original
- **0.5**: 50/50 blend
- **0.0**: 100% original (no processing)

**Recommendations:**
- **1.0** (default) - Maximum bleed removal
- **0.7-0.8** - More natural while still removing most bleed
- **0.5** - Subtle cleanup, very natural

---

## Complete Workflow Example

### Full Pipeline with Optimal Settings

```bash
# Step 1: Separate drums with Wiener filtering
python separate.py -i input/ -o separated_stems/ -w 1.5 -d cuda

# Step 2: Apply sidechain cleanup with balanced settings
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/

# Alternative: If still hearing bleed, use aggressive settings
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems_v2/ \
  -t -40 -r 15 --attack 0.5
```

### Testing Different Settings

When experimenting, use descriptive output directory names:

```bash
# Test 1: Default
python sidechain_cleanup.py -i separated_stems/ -o test_default/

# Test 2: Aggressive
python sidechain_cleanup.py -i separated_stems/ -o test_aggressive/ \
  -t -40 -r 20 --attack 0.5

# Test 3: Gentle
python sidechain_cleanup.py -i separated_stems/ -o test_gentle/ \
  -t -25 -r 4 --dry-wet 0.7
```

Compare the results and choose the best one!

---

## Troubleshooting

### Problem: Kick sounds "pumping" or unnatural

**Cause:** Compression is too aggressive or release is too fast.

**Solutions:**
- Increase threshold: `-t -25` (less sensitive)
- Decrease ratio: `-r 6` (less aggressive)
- Increase release: `--release 150` (smoother recovery)
- Use dry/wet blend: `--dry-wet 0.7`

### Problem: Still hearing snare bleed in kick

**Cause:** Compression not aggressive enough or threshold too high.

**Solutions:**
- Lower threshold: `-t -40` (more sensitive)
- Increase ratio: `-r 15` or `-r 20` (more aggressive)
- Faster attack: `--attack 0.5` (catch transients quicker)
- Check that snare file exists and is aligned with kick

### Problem: Kick loses punch or dynamics

**Cause:** Over-compression or attack too fast.

**Solutions:**
- Use dry/wet blend: `--dry-wet 0.6` (preserve more original)
- Slower attack: `--attack 3` (let initial transient through)
- Lower ratio: `-r 6` (less aggressive)
- Higher threshold: `-t -28` (only duck on louder snare hits)

### Problem: Inconsistent results across different sections

**Cause:** Threshold may need adjustment for varying dynamics.

**Solutions:**
- Lower threshold to catch quieter snare: `-t -35`
- Use a blend to maintain consistency: `--dry-wet 0.8`
- Consider applying compression/limiting to snare track first to even out dynamics

---

## Technical Details

### How It Works

1. **Envelope Following**: Analyzes the snare track's amplitude envelope using configurable attack/release times
2. **Threshold Detection**: When snare envelope exceeds threshold, triggers gain reduction
3. **Gain Calculation**: Calculates how much to reduce kick based on compression ratio
4. **Application**: Applies time-varying gain reduction to kick track
5. **Preservation**: Other stems (toms, hihat, cymbals) are copied unchanged

### Signal Flow

```
Snare Track → Envelope Follower → Threshold Detection → Gain Calculation
                                                              ↓
Kick Track  ────────────────────────────────────→  Apply Gain → Output
```

### Processing

- Uses **soft-knee compression** for smooth transitions
- Envelope follower with independent attack/release
- Stereo-aware (applies same gain to both channels)
- Sample-accurate processing

---

## Tips & Best Practices

### 🎯 General Tips

1. **Always use Wiener filtering** (`-w 1.5`) when separating - it provides better initial separation
2. **Start with defaults** and only adjust if needed
3. **Use descriptive output folders** when testing multiple settings
4. **Listen in context** - check how stems sound together, not just solo
5. **GPU acceleration** (`-d cuda`) makes separation much faster

### 🎚️ Mixing Tips

After cleanup, you can further enhance your stems:
- **EQ**: Gentle high-pass on kick at 30-40 Hz removes rumble
- **Compression**: Apply standard compression to each stem for consistency
- **Reverb/Effects**: Treat each stem independently in your DAW
- **Layering**: Blend with original samples if needed

### 📁 File Organization

Recommended directory structure:
```
project/
├── input/                  # Original drum mixes
│   └── drums.wav
├── separated_stems/        # Initial LarsNet output
│   └── drums/
│       ├── drums-kick.wav
│       ├── drums-snare.wav
│       ├── drums-toms.wav
│       ├── drums-hihat.wav
│       └── drums-cymbals.wav
├── cleaned_stems/          # After sidechain cleanup
│   └── drums/
│       ├── drums-kick.wav      (cleaned)
│       ├── drums-snare.wav
│       ├── drums-toms.wav
│       ├── drums-hihat.wav
│       └── drums-cymbals.wav
└── final_stems/           # After any additional processing
```

### ⚡ Performance

- Processing is real-time or faster on most systems
- No GPU required for sidechain cleanup (only for separation)
- Can batch process multiple files automatically
- Output files maintain same quality as input (no re-encoding artifacts)

---

## Combining with Other Techniques

### Option 1: Sidechain + EQ

For maximum cleanup:

```bash
# Separate with EQ post-processing
python separate_with_eq.py -i input/ -o separated_stems/ -w 1.5

# Then apply sidechain cleanup
python sidechain_cleanup.py -i separated_stems/ -o cleaned_stems/
```

### Option 2: Iterative Refinement

```bash
# Initial separation
python separate.py -i input/ -o step1/ -w 1.5

# Aggressive sidechain
python sidechain_cleanup.py -i step1/ -o step2/ -t -40 -r 15

# Gentle EQ on result if needed
python separate_with_eq.py -i step2/ -o final/ --no-eq  # Just copies with option for EQ
```

---

## Examples & Results

### Example 1: Standard Rock Drum Mix

**Input:** Complex drum mix with heavy snare/kick overlap

```bash
python separate.py -i input/ -o separated/ -w 1.5
python sidechain_cleanup.py -i separated/ -o cleaned/
```

**Result:** Clean kick and snare separation, suitable for remixing

### Example 2: Electronic/Programmed Drums

**Input:** Layered electronic drums with heavy processing

```bash
python separate.py -i input/ -o separated/ -w 2.0
python sidechain_cleanup.py -i separated/ -o cleaned/ -t -35 -r 12 --attack 0.3
```

**Result:** Tight, punchy separated stems with minimal bleed

### Example 3: Live/Jazz Drums (Preserve Dynamics)

**Input:** Natural, dynamic drum performance

```bash
python separate.py -i input/ -o separated/ -w 1.2
python sidechain_cleanup.py -i separated/ -o cleaned/ -t -28 -r 6 --dry-wet 0.7
```

**Result:** Natural-sounding separation maintaining original dynamics

---

## FAQ

**Q: Does this work on all drum mixes?**  
A: Yes! It works on any separated stems that have temporal separation between kick and snare hits (which is most music).

**Q: Will this affect the actual kick hits?**  
A: No! Since kick and snare rarely hit at exactly the same time, your kick transients remain intact.

**Q: Can I use this for other bleed issues?**  
A: Yes! You could use hihat track to sidechain cymbals, or vice versa. The principle is the same.

**Q: Does this require GPU?**  
A: No, sidechain cleanup runs on CPU. GPU is only beneficial for the initial LarsNet separation.

**Q: What if kick and snare play at the same time?**  
A: In those cases, the ducking will affect the kick slightly, but typically the kick transient is stronger and comes through. You can use `--dry-wet 0.7` to preserve more of the original.

**Q: Can I automate this in a batch process?**  
A: Yes! Both scripts automatically process all WAV files in the input directory.

**Q: What sample rates are supported?**  
A: Any sample rate supported by LarsNet (typically 44.1kHz). The sidechain script preserves whatever sample rate is in the files.

---

## Credits & References

- **LarsNet**: Neural network drum separation by polimi-ispl
- **Sidechain Compression**: Classic audio engineering technique adapted for bleed removal
- **Implementation**: Uses scipy for signal processing, soundfile for I/O

---

## Version History

- **v1.0** (2025-10-12): Initial release with envelope follower and soft-knee compression

---

## Support & Contribution

Found a bug or have suggestions? Feel free to open an issue or submit a pull request!

**Enjoy your clean drum stems! 🥁🎵**
