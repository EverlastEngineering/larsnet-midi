# Detection Output Contract

This document defines the standardized output format for stem-to-MIDI detection. The goal is to preserve all spectral analysis data that is currently calculated but discarded.

## Current Problem

Detection calculates rich spectral data, prints it to terminal, then discards everything except:
```python
{time, note, velocity, duration}
```

This document defines what data exists and how to preserve it.

## Per-Stem Spectral Features

### Common Fields (All Stems)

| Field | Type | Description |
|-------|------|-------------|
| `time` | float | Onset time in seconds |
| `onset_strength` | float | librosa onset strength (0.0-1.0) |
| `peak_amplitude` | float | Peak waveform amplitude near onset |
| `total_energy` | float | Sum of all frequency band energies |
| `geomean` | float | Geometric mean of energies (threshold metric) |
| `status` | str | "KEPT" or "REJECTED" |

### Snare

| Field | Hz Range | Description |
|-------|----------|-------------|
| `body_energy` | 150-400 Hz | Low/mid drum body |
| `wire_energy` | 2-8 kHz | Snare wire sizzle |

**GeoMean Formula:** `sqrt(body_energy * wire_energy)`  
**Default Threshold:** 40.0

### Kick

| Field | Hz Range | Description |
|-------|----------|-------------|
| `fundamental_energy` | 40-80 Hz | Fundamental thump |
| `body_energy` | 80-150 Hz | Body resonance |
| `attack_energy` | 150-400 Hz | Beater attack |

**GeoMean Formula:** `cbrt(fundamental_energy * body_energy * attack_energy)`  
**Default Threshold:** 70.0

### Hi-Hat

| Field | Hz Range | Description |
|-------|----------|-------------|
| `body_energy` | 500-2000 Hz | Metal body |
| `sizzle_energy` | 6-12 kHz | High sizzle |
| `sustain_ms` | - | Decay duration (open vs closed) |

**GeoMean Formula:** `sqrt(body_energy * sizzle_energy)`  
**Default Threshold:** 20.0  
**Open/Closed Threshold:** 100ms (≥100ms = open)

### Cymbals

| Field | Hz Range | Description |
|-------|----------|-------------|
| `body_energy` | 1-4 kHz | Metal body |
| `brilliance_energy` | 4-10 kHz | Brilliance |
| `sustain_ms` | - | Decay duration |

**GeoMean Formula:** `sqrt(body_energy * brilliance_energy)`  
**Default Threshold:** 30.0

### Toms

| Field | Hz Range | Description |
|-------|----------|-------------|
| `fundamental_energy` | 60-150 Hz | Fundamental pitch |
| `body_energy` | 150-400 Hz | Body resonance |
| `detected_pitch_hz` | - | Pitch detection result |
| `tom_class` | - | "Low", "Mid", or "High" |

**GeoMean Formula:** `sqrt(fundamental_energy * body_energy)`  
**Default Threshold:** 80.0

## Extended Output Format

### Current Format (4 fields)
```python
events = [
    {'time': 0.5, 'note': 38, 'velocity': 100, 'duration': 0.1},
]
```

### Extended Format (backwards compatible)
```python
events = [
    {
        # MIDI essentials (unchanged)
        'time': 0.5,
        'note': 38,
        'velocity': 100,
        'duration': 0.1,
        
        # Spectral features (NEW - MIDI export ignores these)
        'onset_strength': 0.85,
        'peak_amplitude': 0.42,
        'geomean': 185.3,
        'status': 'KEPT',
        
        # Stem-specific (varies by instrument)
        'body_energy': 64.0,
        'wire_energy': 1684.7,
        'total_energy': 1748.8,
    },
]
```

## Implementation Notes

1. **MIDI Export:** The `midi.py` module only reads `time`, `note`, `velocity`, `duration`. Extra fields are ignored.

2. **Learning Tools:** The `learning.py` and `optimization/` modules can use the extra fields for threshold optimization.

3. **Confidence Scoring:** `geomean / threshold` gives a confidence ratio. Values near 1.0 are marginal decisions.

4. **Debug Output:** The terminal output columns map directly to these field names (BodyE → body_energy, etc.)

5. **Anchor Note (27):** All MIDI files contain a technical anchor note at time 0 with MIDI note 27 and velocity 1. This ensures proper DAW alignment when importing. It is NOT actual drum content - ignore note 27 when analyzing MIDI files.

## CLI Commands for Per-Stem Testing

```bash
# Run individual stems with debug output
python stems_to_midi_cli.py <project> --stems snare --maxtime 10
python stems_to_midi_cli.py <project> --stems kick --maxtime 60
python stems_to_midi_cli.py <project> --stems hihat --maxtime 10
python stems_to_midi_cli.py <project> --stems toms --maxtime 60
python stems_to_midi_cli.py <project> --stems cymbals --maxtime 60

# Config flags for verbose output
# midiconfig.yaml:
#   debug:
#     show_all_onsets: true
#     show_spectral_data: true
```

## Learning Mode (High-Sensitivity Troubleshooting)

Learning mode runs detection with ultra-sensitive settings to catch ALL possible onsets, including false positives. This is essential for:

1. **Threshold calibration** - See what the detector CAN detect, then tune thresholds
2. **Debugging missed hits** - Determine if an onset was detected but filtered out
3. **Training data generation** - Export all candidates for ML labeling

### CLI Usage

```bash
# Enable learning mode via CLI flag
python stems_to_midi_cli.py <project> --learn

# Learning mode + individual stem
python stems_to_midi_cli.py <project> --learn --stems hihat --maxtime 30
```

### Config Settings

From `midiconfig.yaml`:

```yaml
learning_mode:
  enabled: false                      # Enable learning mode
  export_all_detections: true         # Export ALL onsets (even rejected)
  rejected_velocity: 1                # Velocity for rejected hits (visible in DAW)
  kept_velocity_normal: true          # Use normal velocity for kept hits
  
  # Ultra-sensitive detection parameters
  learning_onset_threshold: 0.0001    # Very low - catches everything
  learning_delta: 0.002               # Very sensitive peak picking
  learning_wait: 1                    # Allow very close hits (~11ms)
  
  # Output locations
  learning_midi_suffix: "_learning"   # MIDI file suffix
  calibrated_config_output: "midiconfig_calibrated.yaml"
```

### Learning Mode Output

In learning mode, ALL detected onsets are exported to MIDI:
- **Kept hits**: Normal velocity (40-127 based on amplitude)
- **Rejected hits**: Velocity = 1 (visible in DAW but quiet)

This allows visual inspection in a DAW to see what the detector found and why certain hits were rejected.

### Workflow for Threshold Calibration

1. Run in learning mode: `python stems_to_midi_cli.py <project> --learn`
2. Open the `_learning.mid` file in DAW
3. Delete false positives, add missed hits
4. Save edited MIDI
5. Run threshold learning: `learn_threshold_from_midi()` compares original vs edited
6. New optimal thresholds saved to `midiconfig_calibrated.yaml`

## Known Limitations

### --maxtime Truncation Pitfall

The `--maxtime` flag truncates audio for faster testing, but **may miss content that appears later in the track**.

**Example:** Thunderstruck cymbals don't appear until ~90 seconds (intro is all hi-hat):
```
0-90s:  max amplitude 0.0005 (effectively silent)
90s+:   max amplitude 0.31-0.56 (actual content)
```

Running `--maxtime 60` shows "Audio is silent, skipping..." but the full track has 77 cymbal hits.

**Best Practice:**
1. Run full conversion first (no maxtime) to confirm what content exists
2. Use `--maxtime` only for faster iteration after confirming content location
3. For songs with long intros, content may not appear until minutes in

### Stem Source Folders

CLI prefers `cleaned/` over `stems/` folders. When debugging, verify which source is used:
```
Using stems from: /path/to/project/cleaned  # or /path/to/project/stems
```

## Migration Path

1. Add fields to output dicts in `filter_onsets_by_spectral()` 
2. Pipe through `_create_midi_events()` unchanged
3. MIDI export ignores extra fields (no changes needed)
4. Update learning tools to use new fields
5. Add JSON sidecar export for analysis tooling
