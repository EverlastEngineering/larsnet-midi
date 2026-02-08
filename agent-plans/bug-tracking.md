## Bugs Migrated to GitHub Issues

Bugs are now tracked in GitHub Issues: https://github.com/EverlastEngineering/DrumToMIDI/issues

- [#2](https://github.com/EverlastEngineering/DrumToMIDI/issues/2) - Missing instrument labels (Low)
- [#3](https://github.com/EverlastEngineering/DrumToMIDI/issues/3) - Text lane legend overlap (Medium)
- [#4](https://github.com/EverlastEngineering/DrumToMIDI/issues/4) - Add ModernGL renderer test (follow-up)
- [#5](https://github.com/EverlastEngineering/DrumToMIDI/issues/5) - Stale references to stems_to_midi.py

---

## Open Bugs (Not Yet in GitHub)

### 5 code default mismatches with midiconfig.yaml values
- **Status**: Open
- **Priority**: Medium
- **Description**: Five settings have YAML values that differ from the code's `.get()` fallback default. The YAML value wins at runtime, but if a key is ever missing from a user's config, they get unexpected behavior.
- **Details**:
  1. `reverb_continuation_attack_threshold`: YAML=0.4, code default=0.2
  2. `kick.statistical_badness_threshold`: YAML=0.3, code default=0.6
  3. `hihat.detect_open`: YAML=true, code default=False
  4. `hihat.open_sustain_ms`: YAML=100, code default=150
  5. `snare.enable_pitch_detection`: YAML=false, code default=True
- **Expected Behavior**: Code fallback defaults match the documented YAML values
- **Actual Behavior**: Code fallbacks differ, creating silent behavioral drift
- **Suggested Fix**: Align all code `.get()` fallbacks to match YAML values. See [midi-yaml-settings-suggestions.md](../docs/midi-yaml-settings-suggestions.md).

### 11 dead config keys in midiconfig.yaml
- **Status**: Open
- **Priority**: Low
- **Description**: Eleven settings in midiconfig.yaml are never read by the processing pipeline. They add confusion and maintenance burden.
- **Details**: `onset_merge_window_ms` (5 stems), `hihat.enable_amplitude_refinement`, `hihat.decay_threshold`, `threshold_optimization.initial_threshold_step`, `threshold_optimization.convergence_patience`, `clustering.features`, plus 4 `learning_mode.*` settings.
- **Expected Behavior**: All config keys should be consumed by code
- **Actual Behavior**: These keys are silently ignored
- **Suggested Fix**: Remove dead keys. See [deprecations.md](../docs/deprecations.md) for full list.

### Missing MIDI note mappings in config for multi-type classification
- **Status**: Open
- **Priority**: High
- **Description**: Code supports multiple MIDI notes per stem type (snare: 4 types, cymbal: 3 types) but config only exposes single `midi_note` field
- **Details**:
  - **Snare**: Code classifies into 4 types but config only has `midi_note: 38`
    - Snare (38), Rimshot (37), Clap (39), Clap+Snare (40) - hardcoded in `DrumMapping`
    - Config should expose: `midi_note_rimshot`, `midi_note_clap`, `midi_note_clap_snare`
  - **Cymbals**: Code classifies into 3 types but config only has `midi_note: 57`
    - Crash (49), Ride (51), Chinese (52) - hardcoded in `DrumMapping`
    - Config should expose: `midi_note_crash`, `midi_note_ride`, `midi_note_chinese`
  - **Hihat**: Has proper config for closed/open/foot-close, plus handclap (39) hardcoded
    - Config should expose: `midi_note_handclap`
  - **Toms**: Properly exposed with `midi_note_low`, `midi_note_mid`, `midi_note_high` ✓
- **Expected Behavior**: All MIDI note mappings should be configurable via midiconfig.yaml
- **Actual Behavior**: Most mappings are hardcoded in `stems_to_midi/config.py::DrumMapping`
- **Impact**: Users cannot customize MIDI note mappings for different drum maps or standards
- **Suggested Fix**: Add all missing `midi_note_*` fields to midiconfig.yaml snare/cymbals/hihat sections

### Missing stereo width measurement for event classification
- **Status**: Open
- **Priority**: Medium
- **Description**: No measurement of stereo "width" to distinguish mono events (snare) from stereo events (clap)
- **Details**:
  - Current metrics: pan_confidence (L/R balance) but not stereo width (L vs R difference)
  - Stereo width measures how different L and R channels are (phase inversion comparison)
  - **Mono events**: L≈R (snare, kick) → low width
  - **Stereo events**: L≠R (handclap, room ambience) → high width
  - This metric would improve classification accuracy for snare vs clap distinction
- **Expected Behavior**: Calculate stereo width metric during detection and include in feature set
- **Actual Behavior**: Only pan position (balance) is measured, not channel difference (width)
- **Impact**: Cannot distinguish between mono-centered and stereo-centered events
- **Suggested Implementation**: 
  - Calculate correlation or RMS difference between L and R channels at onset
  - Add `stereo_width` field to onset features (range 0.0=mono to 1.0=wide stereo)
  - Include in clustering features for better classification

### Missing pan_confidence data in analysis JSON output
- **Status**: Open
- **Priority**: Medium
- **Description**: Pan position data is calculated during energy-based detection but not saved to the analysis.json sidecar file
- **Details**: 
  - `energy_detection_core.py` calculates `pan_confidence` for each onset (R-L)/(R+L) ranging from -1.0 (left) to +1.0 (right)
  - `analysis_core.py::extract_onset_features()` includes pan_confidence in feature extraction
  - `midi.py::save_analysis_sidecar()` does NOT include pan_confidence in saved JSON fields
- **Expected Behavior**: Pan position data should be available in analysis JSON for visualization and analysis
- **Actual Behavior**: Pan data is calculated but discarded during JSON serialization
- **Impact**: Cannot visualize or analyze stereo positioning of detected hits
- **Suggested Fix**: Add 'pan_confidence' to the field list in `save_analysis_sidecar()` (line ~240-245)

### Missing pitch data in analysis JSON output  
- **Status**: Open
- **Priority**: Medium
- **Description**: Pitch detection is implemented for toms/cymbals/snare but pitch_hz is not saved to analysis JSON
- **Details**:
  - `detection_shell.py` has pitch detection functions: `detect_tom_pitch()`, `detect_cymbal_pitch()`, `detect_snare_pitch()`
  - Pitch is used internally for MIDI note classification (tom: low/mid/high, cymbal: crash/ride/chinese, snare: snare/rimshot/clap/clap+snare)
  - `midi.py::save_analysis_sidecar()` includes 'pitch_hz' in the field list but it's never populated
  - Snare pitch detection exists but is disabled by default (`enable_pitch_detection: true` required in config)
- **Expected Behavior**: Pitch should be detected and saved to JSON for all applicable stems
- **Actual Behavior**: Pitch is detected for classification but not saved to analysis JSON
- **Impact**: Cannot analyze pitch distribution or validate classification decisions
- **Root Cause**: Pitch values used for classification are not passed through to the events data structure

### MIDI file creation error: "pop from empty list" in midiutil
- **Status**: Fixed
- **Priority**: High
- **Description**: IndexError in midiutil.MidiFile.deInterleaveNotes() when writing MIDI files with energy-based detection
- **Root Cause**: 
  1. Energy detection creating duplicate onset times (3x duplicates at 197.242s in cymbals)
  2. Zero-duration MIDI notes when two onsets occur at nearly identical times
  3. midiutil's deInterleaveNotes() failing to match note_on/note_off pairs with duplicates
- **Steps to Reproduce**: 
  1. Run stems-to-midi on project 14 (Thunderstruck)
  2. Energy detection produces duplicate onset times within 1ms
  3. MIDI creation calculates duration = next_onset - current_onset = 0.0
  4. midiutil.writeFile() crashes with "IndexError: pop from empty list"
- **Expected Behavior**: Each detected onset creates one MIDI note with valid duration
- **Actual Behavior**: Duplicate onsets create multiple notes at same time with 0 duration, causing MIDI library error
- **Fixed**: 2026-01-27
- **Solution**: Two-part fix:
  1. **Duplicate removal** in `energy_detection_shell.py`:
     - Round onset times to nearest millisecond
     - Remove duplicates within 1ms threshold
     - Prevents duplicate detections from reaching MIDI creation
  2. **Minimum duration enforcement** in `analysis_core.prepare_midi_events_for_writing()`:
     - Set MIN_DURATION_BEATS = 0.01 (5ms at 120 BPM)
     - Ensures all MIDI notes have valid duration
     - Prevents midiutil deInterleaveNotes errors
- **Impact**: Energy-based detection now produces valid MIDI files without errors. Removed 6 duplicate events from project 14 (snare: 1, toms: 5)
- **Files Modified**: 
  - `stems_to_midi/energy_detection_shell.py` (deduplication)
  - `stems_to_midi/analysis_core.py` (minimum duration)
- **Fixed in Commit**: (pending commit)

---

### CPU underutilization during MDX23C stem separation on macOS
- **Status**: Fixed
- **Priority**: High
- **Description**: When using MDX23C model with CPU inference on macOS, only ~55% CPU utilization observed with work distributed to efficiency cores instead of performance cores
- **Root Cause**: PyTorch default threading configuration limited to 4 threads (likely defaulting to physical core count), while system has 8 cores (4 performance + 4 efficiency)
- **Expected Behavior**: Full CPU utilization across all cores, prioritizing performance cores
- **Actual Behavior**: Only 4 threads used, resulting in ~55% CPU usage with efficiency cores engaged
- **Steps to Reproduce**: 
  1. Run stem separation with MDX23C model on Mac with device=cpu
  2. Observe CPU usage in Activity Monitor
  3. Notice low utilization (~55%) and efficiency core usage
- **Fixed**: 2026-01-25
- **Solution**: Added `_configure_cpu_threading()` method to `OptimizedMDX23CProcessor.__init__()`:
  - Detects total CPU core count using `multiprocessing.cpu_count()`
  - On macOS, detects performance vs efficiency core split using `sysctl`
  - Configures PyTorch: `torch.set_num_threads()` to use all cores
  - Sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` environment variables
  - Now uses 8 threads (100% utilization) instead of 4 threads (55% utilization)
- **Impact**: Should significantly improve CPU-based stem separation performance on macOS
- **Files Modified**: `mdx23c_optimized.py`

---

## Open Bugs (Not Yet in GitHub - Original)

### Cymbals appeared missing due to --maxtime truncation
- **Status**: Closed (User Error / Testing Bug)
- **Priority**: N/A
- **Description**: Cymbals appeared silent when testing with `--maxtime 60`
- **Root Cause**: Thunderstruck cymbals don't start until ~90 seconds (intro is all hi-hat)
  - 0-90s: max amplitude 0.000488 (effectively silent)
  - 90s+: max amplitude 0.31-0.56 (actual cymbal content)
- **Resolution**: Running full conversion (no maxtime) detects 77 cymbal events
- **Lesson**: When troubleshooting, run full conversion first, then use maxtime for faster iteration only after confirming content exists
- **Note 27**: Still just a technical anchor note for DAW alignment (see `stems_to_midi/midi.py:64`)

---

## Fixed Bugs (Historical)

### Broken import after file rename - render_midi_video_shell.py
- **Fixed**: 2026-01-18
- **Root Cause**: Missing test coverage for `render_project_video()` with `use_moderngl=True`
