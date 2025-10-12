# TODO: MIDI Conversion Improvements

## High Priority

### 1. Kick Drum - Spectral Filtering
**Status**: Not Started  
**Complexity**: Easy (copy snare logic)

Apply the same spectral filtering approach used for snare to kick drum:
- [x] Snare uses geometric mean of body (150-400Hz) + wires (2-8kHz)
- [ ] Kick should use fundamental (40-80Hz) + body/attack (80-150Hz)
- [ ] Define frequency ranges in config (kick_freq_min, kick_body_min, etc.)
- [ ] Calculate kick_geomean = sqrt(fundamental_energy * body_energy)
- [ ] Add kick.geomean_threshold to midiconfig.yaml
- [ ] Support learning mode for kick
- [ ] Test and document typical threshold values

**Rationale**: Kick has similar issues with snare bleed and artifacts. Low-frequency fundamental distinguishes real kicks.

**Config additions needed**:
```yaml
kick:
  midi_note: 36
  geomean_threshold: 150.0  # TBD from testing
  fundamental_freq_min: 40
  fundamental_freq_max: 80
  body_freq_min: 80
  body_freq_max: 150
```

---

### 2. Toms - Pitch Detection & Multi-Note Mapping
**Status**: Not Started  
**Complexity**: Medium (pitch detection + note mapping)

Toms span a wide pitch range (low/mid/high). Need to:
- [ ] Implement pitch detection (librosa.piptrack or autocorrelation)
- [ ] Detect fundamental frequency of each tom hit
- [ ] Classify into low/mid/high based on frequency ranges
- [ ] Map to different MIDI notes:
  - Low tom: 45 (A1) or 41 (F1)
  - Mid tom: 47 (B1) or 48 (C2)
  - High tom: 50 (D2) or 43 (G#1)
- [ ] Add configurable pitch ranges to config
- [ ] Support learning mode (learns pitch ranges from user edits)
- [ ] Optional: Support 2-tom, 3-tom, 4-tom setups

**Challenges**:
- Tom pitch can vary within a hit (pitch bend)
- Need to distinguish toms from kick/snare
- Some setups have 2 toms, others have 4+

**Config additions needed**:
```yaml
toms:
  # Multi-pitch detection
  detect_multiple_pitches: true
  
  # Pitch ranges (Hz) - adjust based on your drum kit
  low_tom_min: 80
  low_tom_max: 120
  mid_tom_min: 120
  mid_tom_max: 180
  high_tom_min: 180
  high_tom_max: 300
  
  # MIDI note mapping
  low_tom_note: 45    # A1
  mid_tom_note: 47    # B1
  high_tom_note: 50   # D2
  
  # Filtering
  geomean_threshold: 100.0  # Similar to snare/kick
  body_freq_min: 80
  body_freq_max: 300
  attack_freq_min: 1000
  attack_freq_max: 5000
```

---

### 3. Hi-Hat & Cymbals - Spectral Separation
**Status**: Partially Implemented (open/closed detection exists)  
**Complexity**: High (very similar spectral content)

**Current State**:
- Open/closed hi-hat detection exists (decay-based)
- No filtering for artifacts
- Cymbals use separate stem but no intelligence

**Improvements Needed**:

#### Hi-Hat:
- [ ] Apply spectral filtering (reject kicks/snare bleed)
- [ ] Improve open/closed detection (currently decay-based)
- [ ] Add foot hi-hat detection (different spectral signature)
- [ ] Add geomean_threshold for hi-hat
- [ ] Frequency ranges: focus on 5-15kHz (bright metal sounds)

#### Cymbals:
- [ ] Separate crash vs ride detection
- [ ] Pitch-based classification (ride ~400Hz, crash ~550Hz fundamentals)
- [ ] Decay analysis (crashes sustain longer)
- [ ] Map to different notes:
  - Crash 1: 49 (C#2)
  - Crash 2: 57 (A2)
  - Ride: 51 (D#2)
  - Ride bell: 53 (F2)
  - China: 52 (E2)
- [ ] Apply spectral filtering

**Challenges**:
- Hi-hat and cymbals have very similar bright/high-frequency content
- Overlapping hits (crash while hi-hat playing)
- Many cymbal types with subtle differences
- User might want to manually adjust in DAW anyway

**Config additions needed**:
```yaml
hihat:
  midi_note: 42              # Closed hi-hat
  midi_note_open: 46         # Open hi-hat
  midi_note_foot: 44         # Pedal hi-hat
  
  geomean_threshold: 80.0    # Lower than snare (lighter hits)
  
  # Spectral ranges (hi-hat is bright, 5-15kHz dominant)
  body_freq_min: 5000
  body_freq_max: 10000
  bright_freq_min: 10000
  bright_freq_max: 15000
  
  # Open/closed detection
  detect_open: true
  decay_threshold: 0.65
  open_decay_min: 0.3        # Minimum decay for open hi-hat

cymbals:
  # Cymbal type detection
  detect_cymbal_types: true
  
  # Pitch ranges for classification (fundamental frequencies)
  ride_pitch_min: 350
  ride_pitch_max: 450
  crash_pitch_min: 500
  crash_pitch_max: 650
  china_pitch_min: 400
  china_pitch_max: 550
  
  # MIDI note mapping
  crash1_note: 49
  crash2_note: 57
  ride_note: 51
  ride_bell_note: 53
  china_note: 52
  
  # Filtering
  geomean_threshold: 100.0
  body_freq_min: 400
  body_freq_max: 1000
  bright_freq_min: 5000
  bright_freq_max: 15000
```

---

## Medium Priority

### 4. User-Friendly Threshold Adjustment
**Status**: Not Started  
**Complexity**: Easy to Medium

Make it easier for users to adjust thresholds without learning mode:

- [ ] Add interactive threshold adjustment mode
- [ ] `--preview` flag: Shows all detections with thresholds in terminal
- [ ] Real-time threshold adjustment: user types new value, see results immediately
- [ ] Visual histogram of GeoMean values (ASCII art in terminal)
- [ ] Percentile-based suggestions (e.g., "90% of hits are above 250")
- [ ] Save adjusted thresholds to config

**Example workflow**:
```bash
python stems_to_midi.py --preview -i cleaned_stems/ --stems snare

# Shows:
# GeoMean distribution:
# 0-100:   ████████ (45 hits) - likely artifacts
# 100-200: ███ (18 hits) - borderline
# 200-300: ██████████████ (120 hits) - real snares
# 300-400: ████████████████ (180 hits) - real snares
# 400+:    ████████ (67 hits) - loud snares
# 
# Current threshold: 100
# Suggested: 180 (excludes bottom 10%)
# 
# Enter new threshold (or press Enter to accept): _
```

---

### 5. Multi-Stem Coordination
**Status**: Not Started  
**Complexity**: Medium

Sometimes onsets overlap across stems. Coordinate detection:

- [ ] Detect simultaneous hits across stems (within 20ms)
- [ ] Handle overlapping hits (kick + snare on same beat)
- [ ] Ensure MIDI timing is synchronized across stems
- [ ] Option to "lock" timing to kick (everything aligns to kick hits)
- [ ] Crossfade analysis (does kick have snare frequencies? Reduce both)

---

### 6. Ghost Note Detection
**Status**: Not Started  
**Complexity**: Medium

Quiet articulations between main hits:

- [ ] Detect very low amplitude hits (currently filtered out)
- [ ] Separate threshold for ghost notes
- [ ] Map to lower velocity (20-40)
- [ ] User option to include/exclude ghost notes
- [ ] Special handling for snare ghost notes

---

### 7. Flam & Drag Detection
**Status**: Not Started  
**Complexity**: High

Drum rudiments with multiple very close hits:

- [ ] Detect flams (2 hits <30ms apart)
- [ ] Detect drags (3+ hits <20ms apart)
- [ ] Option to merge into single MIDI note or keep separate
- [ ] Grace note MIDI notation
- [ ] Velocity adjustment for grace notes

---

## Low Priority / Nice to Have

### 8. Sheet Music Export
**Status**: Not Started  
**Complexity**: Medium to High

Export detected drum patterns as readable sheet music:

- [ ] Generate drum notation (standard 5-line staff)
- [ ] MusicXML export (can be opened in MuseScore, Finale, Sibelius)
- [ ] LilyPond format export (text-based music engraving)
- [ ] PDF generation with proper drum notation
- [ ] Handle drum notation conventions:
  - Kick on bottom space (F)
  - Snare on 3rd space (C)
  - Hi-hat above staff with stem up
  - Toms on appropriate lines
  - Cymbals with x noteheads
- [ ] Tempo detection and time signature
- [ ] Automatic bar lines based on detected meter
- [ ] Grace notes for flams/drags
- [ ] Articulation marks (accents, ghost notes)
- [ ] Drum key/legend on first page
- [ ] Multi-page layout for long pieces

**Libraries to Consider**:
- `music21` - Python music analysis/generation
- `abjad` - Python interface to LilyPond
- `mingus` - Music theory and notation
- Direct MusicXML generation

**Example Usage**:
```bash
python stems_to_midi.py -i cleaned_stems/ -o output/ --export-sheet-music
# Generates: output/drums_score.pdf, output/drums_score.musicxml
```

---

### 9. Export Formats
- [ ] Export to different DAW project formats (Logic, Ableton)
- [ ] Export to Superior Drummer / EZdrummer format
- [ ] Export timing data for drum replacement plugins
- [ ] Export to Hydrogen drum machine format
- [ ] Export as Reaper notation items

### 9. Performance Metrics
- [ ] Calculate timing accuracy (deviation from grid)
- [ ] Detect tempo changes
- [ ] Generate "heat map" of where most hits occur
- [ ] Export performance statistics

### 10. Advanced Detection
- [ ] Brush vs stick detection (spectral difference)
- [ ] Rim shots vs center hits
- [ ] Cross-stick detection
- [ ] Stick click detection

### 11. Machine Learning
- [ ] Train ML model on user-edited MIDIs
- [ ] Auto-classify drum types without stems
- [ ] Learn user preferences over time
- [ ] Suggest corrections based on musical context

---

## Documentation Needed

- [ ] Update README with all features
- [ ] Video tutorial for learning mode
- [ ] Troubleshooting guide for each stem type
- [ ] Comparison with other drum-to-MIDI tools
- [ ] Best practices for recording drums for conversion
- [ ] Example configs for different drum styles (jazz, metal, rock, electronic)

---

## Testing & Validation

- [ ] Unit tests for spectral analysis functions
- [ ] Integration tests for learning mode
- [ ] Test on diverse drum recordings (genres, quality, mic setups)
- [ ] Benchmark against commercial tools (Drumgizmo, etc.)
- [ ] User testing with real drummers
- [ ] Performance optimization (currently slow for long files)

---

## Bugs / Issues

- [ ] Learning mode temporary config not working correctly
- [ ] Stereo handling in all code paths
- [ ] Velocity calculation consistency across stems
- [ ] MIDI timing precision (quantization needed?)
- [ ] Memory usage for very long audio files

---

## Cleanup / Maintenance

- [ ] Remove old requirements.txt (superseded by environment.yml)
- [ ] Clean up debug MIDI folders (midi_debug*, midi_test*, etc.)
- [ ] Add .gitignore entries for debug output folders
- [ ] Consolidate documentation (multiple guides exist)
- [ ] Remove unused imports
- [ ] Code style consistency check

---

## Current Status Summary

| Feature | Kick | Snare | Toms | Hi-Hat | Cymbals |
|---------|------|-------|------|---------|---------|
| Basic detection | ✅ | ✅ | ✅ | ✅ | ✅ |
| Spectral filtering | ❌ | ✅ | ❌ | ❌ | ❌ |
| Learning mode | ❌ | ✅ | ❌ | ❌ | ❌ |
| Velocity from spectral | ❌ | ✅ | ❌ | ❌ | ❌ |
| Multi-note mapping | N/A | N/A | ❌ | ⚠️ | ❌ |
| Artifact rejection | ❌ | ✅ | ❌ | ❌ | ❌ |

Legend: ✅ Complete | ⚠️ Partial | ❌ Not Started | N/A Not Applicable

---

## Next Steps

**Immediate (This Session)**:
1. Implement kick drum spectral filtering (copy snare approach)
2. Add kick to learning mode
3. Test kick filtering on sample audio

**Short Term**:
1. Implement tom pitch detection
2. Add multi-note mapping for toms
3. Test and tune threshold values

**Long Term**:
1. Hi-hat/cymbal improvements
2. User-friendly threshold adjustment
3. Comprehensive testing and documentation
