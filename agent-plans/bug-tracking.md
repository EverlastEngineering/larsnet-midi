## Bugs Migrated to GitHub Issues

Bugs are now tracked in GitHub Issues: https://github.com/EverlastEngineering/DrumToMIDI/issues

- [#2](https://github.com/EverlastEngineering/DrumToMIDI/issues/2) - Missing instrument labels (Low)
- [#3](https://github.com/EverlastEngineering/DrumToMIDI/issues/3) - Text lane legend overlap (Medium)
- [#4](https://github.com/EverlastEngineering/DrumToMIDI/issues/4) - Add ModernGL renderer test (follow-up)
- [#5](https://github.com/EverlastEngineering/DrumToMIDI/issues/5) - Stale references to stems_to_midi.py

---

## Open Bugs (Not Yet in GitHub)

### MIDI timing wrong on initial conversion (works on reconvert)
- **Status**: Open
- **Priority**: High
- **Description**: MIDI file plays way too fast on initial process, but works correctly after "Save & Reconvert".
- **Root Cause**: Unknown - possibly related to commit 15a5461 which changed initial conversion to use rebuild path, or different tempo handling between initial vs rebuild.
- **Expected Behavior**: Both initial conversion and reconvert should produce identical MIDI timing.
- **Actual Behavior**: Initial conversion produces MIDI that plays too fast.

### Filtering/keeping events inconsistent between UI and saved MIDI
- **Status**: Open
- **Priority**: High
- **Description**: Events show as filtered or kept in the UI, but when saving/disk they disappear or stay when they shouldn't. Also depends on whether note classification is enabled.
- **Expected Behavior**: UI display should match what's actually saved to MIDI.
- **Actual Behavior**: Inconsistency between UI filtering display and final MIDI output.

### Reverb filter adds/removes events differently in UI vs actual filter
- **Status**: Open
- **Priority**: High
- **Description**: The reverb continuation filter shows different event counts or decisions in the UI waveform than what actually gets filtered when saving MIDI.
- **Expected Behavior**: UI should accurately reflect what's being filtered.
- **Actual Behavior**: Discrepancy between UI visualization and actual filter results.

### events_configured contains events not in events_sensitive (data integrity)
- **Status**: Open
- **Priority**: High
- **Description**: The events_configured array contained events that were not present in the events_sensitive array. This indicates corrupted analysis data.
- **Steps to Reproduce**: Unknown - user reported this after changing settings and re-running
- **Expected Behavior**: events_configured should always be a subset of events_sensitive
- **Actual Behavior**: events_configured contained extra events not in events_sensitive
- **Mitigation**: Deleted MIDI data and re-ran conversion to fix
- **Prevention**: Consider adding validation in analysis loading or a toast warning in UI if this data inconsistency is detected

### Testing gaps - no e2e verification of filtering/keeping/MIDI output
- **Status**: Open
- **Priority**: High
- **Description**: No automated tests verify that UI filtering/keeping decisions match what's actually saved to MIDI files. Current tests don't cover the full pipeline: detection → analysis.json → rebuild → MIDI output → verify notes match expectations.
- **Needed**: 
  - Test fixtures with known audio inputs
  - Test cases for each filtering scenario (geomean, sustain, reverb continuation, classification)
  - Read back generated MIDI files and verify note times/durations match expected values
  - A/B testing between different config settings
- **Impact**: Bugs like "filtering inconsistent between UI and MIDI" go undetected because there's no automated verification.

### Min Sustain slider has no effect on hihat filtering
- **Status**: Fixed
- **Priority**: High
- **Description**: The "Min Sustain" slider in the hihat tuning panel appeared to do nothing.
- **Root Cause**: filter_mode was set to 'geomean_only' so min_sustain_ms was ignored.
- **New Approach**: Apply min_sustain_ms as a final catch filter AFTER reverb continuation filtering (Pass 3), rather than at the same time as geomean filtering. This avoids the issue where sustain filtering at the same time as other filters causes wrong notes to be allowed/filtered.
- **Fix**: 
  - analysis_core.py: Added Pass 3 sustain filter after reverb filtering for hihat
  - threshold-tuning.js: Added applyMinSustainFilter() after reverb filter in WebUI

### Classification settings don't affect MIDI file output
- **Status**: Fixed
- **Priority**: High
- **Description**: Snare MIDI output has random, poorly classified notes regardless of cluster settings. Root cause: `_build_logic_block` crashes with `int(None)` when `expected_clusters` is null in YAML, causing rebuild to fail silently and fall back to full pipeline.
- **Root Cause**: `int(stem_config.get('expected_clusters', defaults[stem_type]))` returns `int(None)` when YAML value is explicitly null.
- **Fix**: Guard with `raw = stem_config.get('expected_clusters'); int(raw) if raw is not None else defaults[stem_type]`

### UI state inconsistent after save & reconvert
- **Status**: Fixed
- **Priority**: High
- **Description**: After save & reconvert, tuning panel closes and reopening shows stale state. `tuningSliderValues` retained stale values from before save.
- **Fix**: Clear `tuningSliderValues[stemType]` on successful rebuild, rebuild sliders from fresh logic block.

### Velocity bars flat in tuning mode
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Event bars all same height. Root cause: `int(None)` crash prevented rebuild, so events never got velocity assigned — all defaulted to 64.
- **Fix**: Same as classification bug fix. Velocity assigned correctly once rebuild succeeds.

### Events not color-coded when tuning is closed
- **Status**: Fixed
- **Priority**: Medium
- **Description**: All KEPT events green when tuning closed. Root cause: (1) rebuild crash prevented `classification` field from being set, (2) color system keyed by MIDI note number instead of classification index.
- **Fix**: Replaced `NOTE_TYPE_COLORS` (per-MIDI-note) with `CLASSIFICATION_COLORS` (per-classification-index). `getMarkerColor` now uses `event.classification`.

### Note type colors too similar
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Per-note colors too similar within groups (e.g., various purples for snare).
- **Fix**: Replaced with 4 high-contrast classification colors: green (#10b981), purple (#a855f7), cyan (#22d3ee), yellow (#eab308). Standard across all stems.

### Red/orange events visible when tuning is closed
- **Status**: Fixed
- **Priority**: Medium
- **Description**: FILTERED (red) and REVERB_CONTINUATION (orange) events visible when tuning is closed.
- **Fix**: Filter `displayEvents` to KEPT-only when `!waveformTuningActive`. Legend also hides filtered/reverb counts when tuning is closed.

### Initial MIDI conversion produces poor quality vs. "Save & Reconvert"
- **Status**: Fixed
- **Priority**: High
- **Description**: When first processing a MIDI file using the webui "Convert Stems to MIDI" button, the output MIDI file has many errors. However, clicking "Save & Reconvert" produces correct MIDI output - and even changing a setting then changing it BACK produces good output.
- **Root Cause**: TWO SEPARATE CODE PATHS for creating MIDI:
  1. **Legacy path** (initial conversion in `stems_to_midi_cli.py`): 
     - Ran detection → created MIDI directly from detection events
     - Then saved analysis.json SEPARATELY
  2. **New path** (reconvert in `rebuild_shell.py`):
     - Loaded analysis.json → ran rebuild logic → created MIDI
     - Applied additional filtering/classification from stored data
- **The Fix**: Modified `stems_to_midi_cli.py` to:
  1. Save analysis sidecar first after detection
  2. Load that sidecar and run rebuild to create MIDI
  3. Removed fallback code that created MIDI directly
  4. Now both initial conversion and reconvert use identical rebuild logic

### Hihat open/closed detection inconsistency and missing UI
- **Status**: Open
- **Priority**: High
- **Description**: Hihat open/closed detection works differently from other stems and lacks UI controls. The detection must be enabled via YAML config (`detect_open: true`) or CLI/API parameter, requiring a full reconvert to change. No interactive tuning UI exists for hihat classification like other stems have.

#### Part 1: Parameter Name Mismatch
- **Locations**: 
  - YAML config: `hihat.detect_open` (defaults to `true` in midiconfig.yaml line 250)
  - CLI parameter: `--detect-hihat-open` / `detect_hihat_open` (defaults to `False`)
  - API parameter: `detect_hihat_open` (defaults to `False`)
- **Root Cause**: The YAML uses `detect_open` but the CLI/API uses `detect_hihat_open`. The CLI tries to read `detect_open` from config as fallback (stems_to_midi_cli.py lines 247-251), but the webui sends `detect_hihat_open` which bypasses the YAML setting entirely.
- **Flow**:
  1. WebUI calls API with `detect_hihat_open: false` (or omitted → defaults to false)
  2. This overrides the YAML's `detect_open: true` setting
  3. Detection runs with open/closed disabled → all hihats marked as 'closed'
  4. Rebuild always re-runs classification using current config thresholds

#### Part 2: Rebuild Overwrites hihat_state
- **Location**: `note_classification_core.classify_hihat_notes()` (lines 25-65)
- **Behavior**: Unlike other stems that preserve stored classification, hihat's classification ALWAYS overwrites `hihat_state` based on current config thresholds (`open_geomean_min`, `open_sustain_ms`)
- **Impact**: Changing hihat classification thresholds triggers reclassification during rebuild, producing different MIDI than initial conversion

#### Part 3: Missing Hihat UI in Tuning Panel
- **Location**: `webui/static/js/threshold-tuning.js`
- **Current State**:
  - snare/toms/cymbals: Have `expected_clusters` slider and clustering visualization
  - hihat: NO clustering slider, NO cluster visualization
- **Only UI exposed**: Geomean threshold, reverb attack threshold, min sustain (via Pass 3 filter)
- **Missing from UI**: `detect_open`, `open_geomean_min`, `open_sustain_ms` settings

#### Part 4: Data Saved to JSON
- **Verified**: `hihat_state` IS saved to analysis.json (confirmed in user_files project)
- **Saved fields**: `body_energy`, `sizzle_energy`, `geomean`, `sustain_ms`, `hihat_state`
- **Issue**: Classification re-runs on rebuild even when data is already classified

#### Recommended Fixes
1. **Unify parameter names**: Change CLI/API to use `detect_open` to match YAML, OR change YAML to use `detect_hihat_open`
2. **Fix webui to not override**: Don't send `detect_hihat_open` in API call unless explicitly toggled by user, let YAML config take precedence
3. **Add hihat tuning UI**: Expose `open_geomean_min`, `open_sustain_ms` sliders in hihat tuning panel
4. **Preserve hihat_state in rebuild**: Don't re-classify hihat if `hihat_state` already exists in stored data (similar to how other stems preserve classification)

#### Affected Files
- `stems_to_midi_cli.py`: Lines 48, 247-251 (detect_hihat_open vs detect_open)
- `webui/api/operations.py`: Line 310 (detect_hihat_open parameter)
- `webui/static/js/threshold-tuning.js`: No hihat-specific classification sliders
- `stems_to_midi/note_classification_core.py`: Lines 25-65 (classify_hihat_notes always overwrites)


### Rebuild-from-analysis produces degraded MIDI output
- **Status**: Fixed
- **Priority**: High
- **Description**: Rebuild from analysis.json produces significantly worse MIDI than the full pipeline with same config. Many reverb events are unfiltered, and in some cases entire stems (kick) go missing.
- **Steps to Reproduce**: Run full conversion, then rebuild from analysis.json without changing sliders. Compare MIDI files.
- **Expected Behavior**: Identical MIDI output when config is unchanged
- **Actual Behavior**: Many more events (reverb artifacts), possible missing stems
- **Root Cause**: Two issues: (1) Unconditional merging of sensitive events regardless of whether thresholds changed, adding hundreds of extra events. (2) Missing multi-pass filters — rebuild only had Pass 1, while full pipeline has 4 passes including decay filter, statistical badness, and reverb continuation.
- **Solution**: Three-path strategy: same thresholds trust stored statuses; raised thresholds re-filter configured events only; lowered thresholds merge sensitive events then re-filter with reverb continuation post-pass.
- **Validation**: Project 21 real data — all 5 stems produce identical event counts when thresholds match.
- **Fixed in Commit**: aa96887

### Reverb threshold slider reverts to default after save
- **Status**: Fixed
- **Priority**: High
- **Description**: Changing `reverb_continuation_attack_threshold` in the tuning panel and saving always reverts the slider to the hardcoded fallback (0.4). The config YAML is updated correctly, and the rebuild uses the correct value, but the returned analysis.json logic block never includes the key. On re-render, the slider reads `logic['reverb_continuation_attack_threshold']` → undefined → falls back to 0.4.
- **Root Cause**: Neither `save_analysis_sidecar` (midi.py) nor `_build_logic_block` (rebuild_core.py) included `reverb_continuation_attack_threshold` in the logic block. This key lives in the global `[filtering]` config section rather than per-stem, so neither function picked it up.
- **Fix**: Read `reverb_continuation_attack_threshold` from `config['filtering']` in both `save_analysis_sidecar` and `_build_logic_block`, storing it in the logic block for the frontend to read.

### Detection Analysis section not visible after MIDI job completes
- **Status**: Fixed
- **Priority**: Medium
- **Description**: After running the MIDI conversion step, the Detection Analysis section does not appear. User must click to another project and come back for it to show.
- **Expected Behavior**: Detection Analysis section appears immediately when the MIDI job completes (since analysis data is now available).
- **Actual Behavior**: Section stays hidden until the project is re-selected.
- **Fix**: Added explicit show/expand logic in `onComplete` handler for 'stems-to-midi' jobs in operations.js to ensure the analysis section becomes visible and expanded after the project data is refreshed.

### Detection Analysis box doesn't expand when Sound Types slider increases content
- **Status**: Fixed
- **Priority**: Medium
- **Description**: When Sound Types (or Tune option) is enabled, the Detection Analysis collapsible box doesn't expand to fit the new content. User must click the expand arrow twice to "grow" it to the correct size.
- **Root Cause**: When cluster cards are rendered after reclassification, `updateCollapsibleHeights()` was not called to recalculate the container's max-height.
- **Fix**: Added `updateCollapsibleHeights()` call via `requestAnimationFrame` after rendering/hiding cluster cards in the reclassification flow.

### Energy envelope renders as flat line at bottom of canvas
- **Status**: Open
- **Priority**: High
- **Description**: The L/R energy envelopes should render as a traditional DAW-style waveform but appear as barely-visible purple at the floor of the graph. Onset markers are visible and color-coded, but the waveform shape is missing or too small to see.
- **Expected Behavior**: Energy envelope fills the canvas vertically like a standard audio waveform display.
- **Actual Behavior**: A thin sliver of purple is visible at the very bottom of the canvas.

### No zoom/pan on waveform canvas
- **Status**: Open
- **Priority**: High
- **Description**: Dense stems (e.g., hi-hat, cymbals) have too many events packed together to see anything useful. No way to zoom in horizontally or pan across the waveform.

### events_configured contains events not in events_sensitive
- **Status**: Open
- **Priority**: High
- **Description**: Some events appeared in `events_configured` that were not present in `events_sensitive` array. This should never happen - if an event passes the configured threshold, it should either have been detected in the sensitive pass OR the threshold is lower than what is in config.
- **Steps to Reproduce**: Unknown - user reported it after changing hihat tuning settings, fixed by deleting MIDI data and re-running
- **Expected Behavior**: events_configured should always be a subset of events + events_sensitive
- **Actual Behavior**: Some events in events_configured were not in events_sensitive
- **Suggestion**: Add validation in the analysis data to detect this inconsistency and warn user (toast notification)
- **Expected Behavior**: User can zoom in to see detail on dense passages and pan to navigate.
- **Actual Behavior**: Entire track is rendered at full width with no zoom capability.

### 5 code default mismatches with midiconfig.yaml values
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Five settings have YAML values that differ from the code's `.get()` fallback default. The YAML value wins at runtime, but if a key is ever missing from a user's config, they get unexpected behavior.
- **Details**:
  1. `reverb_continuation_attack_threshold`: YAML=0.4, code default=0.2
  2. `kick.statistical_badness_threshold`: YAML=0.3, code default=0.6
  3. `hihat.detect_open`: YAML=true, code default=False (no longer read in code; effectively auto-resolved by the rewrite of `stems_to_midi_cli.py` which removed the `detect_hihat_open` parameter path)
  4. `hihat.open_sustain_ms`: YAML=100, code default=150
  5. `snare.enable_pitch_detection`: YAML=false, code default=True
- **Expected Behavior**: Code fallback defaults match the documented YAML values
- **Actual Behavior**: Code fallbacks differ, creating silent behavioral drift
- **Suggested Fix**: Align all code `.get()` fallbacks to match YAML values. See [midi-yaml-settings-suggestions.md](../docs/midi-yaml-settings-suggestions.md).
- **Fixed**: 2026-06-06 (T1 drift-fix)
  - `reverb_continuation_attack_threshold` default raised 0.2 → 0.4 in `stems_to_midi/rebuild_core.py` and `stems_to_midi/analysis_core/onset_filtering.py`
  - `statistical_badness_threshold` default lowered 0.6 → 0.3 in `stems_to_midi/analysis_core/onset_filtering.py`
  - `hihat.open_sustain_ms` default lowered 150 → 100 in 6 files: `midi.py`, `note_classification_core.py`, `processing_shell.py`, `rebuild_core.py`, `analysis_core/spectral_utils.py`, `optimization/extract_features.py`, plus the schema
  - `snare.enable_pitch_detection` default flipped True → False in `stems_to_midi/processing_shell.py`
  - `hihat.detect_open` is no longer read by any production code (was only in the legacy `stems_to_midi_cli.py` arg-parser path, now removed). Effectively resolved.

### 11 dead config keys in midiconfig.yaml
- **Status**: Fixed
- **Priority**: Low
- **Description**: Eleven settings in midiconfig.yaml are never read by the processing pipeline. They add confusion and maintenance burden.
- **Details**: `onset_merge_window_ms` (5 stems), `hihat.enable_amplitude_refinement`, `hihat.decay_threshold`, `threshold_optimization.initial_threshold_step`, `threshold_optimization.convergence_patience`, `clustering.features`, plus 4 `learning_mode.*` settings.
- **Expected Behavior**: All config keys should be consumed by code
- **Actual Behavior**: These keys are silently ignored
- **Suggested Fix**: Remove dead keys. See [deprecations.md](../docs/deprecations.md) for full list.
- **Fixed**: 2026-06-06 (T1 drift-fix). All listed keys removed from `midiconfig.yaml` and (where present) from `webui/settings_schema.py`. Removal log added to `docs/deprecations.md`. No production code referenced any of these keys; `grep` confirmed pre-removal. The 6 schema entries for `onset_merge_window_ms` were also removed.

### Missing MIDI note mappings in config for multi-type classification
- **Status**: Open
- **Priority**: High
- **Description**: Code supports multiple MIDI notes per stem type (snare: 4 types, cymbal: 3 types) but config only exposes single `midi_note` field
- **Details**:
  - **Snare**: Code classifies into 3 types but config only has `midi_note: 38`
    - Snare (38), Rimshot (37), Clap (39) - hardcoded in `DrumMapping`
    - Config should expose: `midi_note_rimshot`, `midi_note_clap`
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

### Schema-YAML drift in per-stem `geomean_threshold` defaults
- **Status**: Fixed
- **Priority**: Medium
- **Date Found**: 2026-06-06 (T1 drift-fix)
- **Description**: `webui/settings_schema.py` had `kick_geomean_threshold` with default=70.0 (an old doc-only value) while `midiconfig.yaml` had `kick.geomean_threshold: 800.0`. The per-stem `snare/toms/hihat/cymbals.geomean_threshold` settings were missing from the schema entirely.
- **Impact**: Anyone using `--schema` or the WebUI form would see the wrong default for kick. The `stem_section.get('geomean_threshold')` calls in the pipeline use the YAML value, so behavior was correct, but the schema lied about defaults — a soft drift signal.
- **Root Cause**: Schema was never fully synchronized with `midiconfig.yaml`. The kick entry was inherited from an earlier docs-only spec; the others were never added when the YAML gained them.
- **Fix**: Added/aligned in `webui/settings_schema.py`:
  - `kick_geomean_threshold`: default 70.0 → 800.0
  - `snare_geomean_threshold`: added, default 40.0
  - `toms_geomean_threshold`: added, default 80.0
  - `hihat_geomean_threshold`: added, default 8.0
  - `cymbals_geomean_threshold`: added, default 100.0
  All five also exposed as CLI flags (`--kick-geomean`, `--snare-geomean`, `--toms-geomean`, `--hihat-geomean`, `--cymbals-geomean`).
- **Files**: `webui/settings_schema.py`

### DrumMapping.handclap hardcoded note (39)
- **Status**: Fixed
- **Priority**: Low
- **Date Found**: 2026-06-06 (T1 drift-fix)
- **Description**: `stems_to_midi/config.py::DrumMapping.handclap` was a property that hardcoded `return 39` rather than reading `hihat.midi_note_handclap` from config. Other sub-type notes (snare rimshot/clap, cymbals crash/ride/chinese) already had explicit fields populated via `from_config()`.
- **Impact**: Users could not customize the handclap note via YAML. Changing `hihat.midi_note_handclap` in `midiconfig.yaml` was silently ignored.
- **Root Cause**: Property was added in an early pass before the YAML gained `midi_note_handclap`. The YAML value existed but no code path read it for the handclap note.
- **Fix**: Replaced the `handclap` property with a dataclass field `hihat_handclap` populated by `from_config()` from `config['hihat']['midi_note_handclap']`. The `handclap` property now reads from the field, preserving the existing API. Schema entry `hihat_midi_note_handclap` already existed; verified it now flows through to the MIDI file.
- **Files**: `stems_to_midi/config.py`

### argparse dest defaults diverge from schema key names
- **Status**: Fixed
- **Priority**: Low
- **Date Found**: 2026-06-06 (T1 drift-fix)
- **Description**: When mapping schema `SettingDefinition` entries to `argparse` flags, the auto-generated `dest` name is the flag (e.g. `--kick-geomean` → `kick_geomean`), not the schema key (`kick_geomean_threshold`). The first pass of the CLI builder tried to look up values by `definition.cli_flag.lstrip('-').replace('-', '_')` which would return `kick_geomean` instead of `kick_geomean_threshold`. This silently dropped all overrides at write-back time.
- **Impact**: All 5 per-stem geomean flags and others would have been silently ignored by `apply_cli_overrides()` if not caught.
- **Root Cause**: `argparse` derives `dest` from the flag name unless an explicit `dest=` is passed. Generic schema→CLI mappers must explicitly set `dest=definition.key` (or maintain a name map).
- **Fix**: `_add_one_flag()` in `webui/cli_builder.py` now passes `dest=definition.key` to `parser.add_argument()`. `apply_cli_overrides()` and `validate_args()` use `definition.key` to look up the value on the Namespace.
- **Files**: `webui/cli_builder.py`

### /api/rebuild-midi endpoint silently dropped all WebUI slider overrides
- **Status**: Fixed
- **Priority**: High
- **Date Found**: 2026-06-06 (T2 bug-fixes)
- **Description**: When the user moved any threshold slider in the WebUI tuning panel and clicked "Save & Reconvert", the server-side rebuild produced MIDI that did not reflect the new thresholds. The UI's client-side `applyTuningFilter()` would re-filter the local snapshot so the waveform visualization looked correct, but the actual saved MIDI was filtered with the YAML defaults.
- **Root Cause**: The `/api/rebuild-midi` endpoint (`webui/api/operations.py`) only read `project_number`, `stem_types`, and `honor_overrides` from the request body — it did not accept any `config_overrides`. `rebuild_midi_for_project()` in `stems_to_midi/rebuild_shell.py` then loaded the YAML config and used it directly, so the slider values in `tuningSliderValues[stemType]` on the client side were never sent to the server and never applied. This was the same symptom as the "Reverb filter adds/removes events differently in UI vs actual filter" bug (also labeled D) but with a different fix scope — that bug was about reverb specifically; this one is the general case.
- **Impact**: The user-visible effect was that any slider move (geomean threshold, reverb_continuation_attack_threshold, expected_clusters, open_geomean_min, etc.) appeared to work in the tuning panel but was silently discarded on Save & Reconvert. The user would then see the tuning panel "reset" to the YAML values after the rebuild.
- **Fix**: 
  - `rebuild_shell.py`: added `config_overrides` kwarg to `rebuild_midi_for_project()` and a `_apply_config_overrides()` helper that writes dotted-path overrides (e.g. `filtering.reverb_continuation_attack_threshold`, `kick.geomean_threshold`, `hihat.open_geomean_min`) into the loaded config dict.
  - `webui/api/operations.py`: `/api/rebuild-midi` now reads `config_overrides` from the request body and forwards it.
  - `webui/static/js/threshold-tuning.js`: `Save & Reconvert` builds the overrides dict from `tuningSliderValues[stemType]` via a new `_buildConfigOverrides()` helper, which maps the slider keys to their dotted YAML paths (per-stem keys nest under the stem name; the global `reverb_continuation_attack_threshold` lives under `filtering`).
- **Files**: `stems_to_midi/rebuild_shell.py`, `webui/api/operations.py`, `webui/static/js/threshold-tuning.js`
- **Fix Commit**: 72e28ac

### Cymbal pitch_hz never computed (only toms in onset_filtering.py)
- **Status**: Open (partially mitigated)
- **Priority**: Medium
- **Date Found**: 2026-06-06 (T2 bug-fixes)
- **Description**: The bug B spec lists `pitch_hz` in the per-stem output for cymbals (and snare when pitch detection is on), but the pipeline's `pitch_hz` field is only populated for toms in `stems_to_midi/analysis_core/onset_filtering.py:258-261` (an `if stem_type == 'toms': ... else: detected_pitch = None` branch). Cymbals have a `detect_cymbal_pitch` function in `stems_to_midi/processing_shell.py:300` but its output is used for crash/ride/chinese classification, never written into the onset data that ends up in `analysis.json`.
- **Impact**: `analysis.json` events for cymbals (and snare with pitch detection on) have `pitch_hz: null` even when the audio would support it. The T2 fix (B) ensures the JSON key is always present with `null` when missing, but does not compute the value for these stems. A future T could extend `onset_filtering.py` to also run pitch detection for cymbals and snare-with-pitch-enabled.
- **Suggested Fix**: In `onset_filtering.py` line 258, add branches for cymbals and snare-pitch-enabled that call `detect_cymbal_pitch()` / `detect_snare_pitch()` with the right frequency ranges from the YAML config. The `detected_pitch` value should be written to `onset_data['pitch_hz']` the same way the tom path does.
- **Files**: `stems_to_midi/analysis_core/onset_filtering.py`, possibly `stems_to_midi/processing_shell.py`

### "Save & Reconvert" and reclassify all fail with 500/404 when project has no midiconfig.yaml
- **Status**: Open (regression introduced by T2)
- **Priority**: High
- **Date Found**: 2026-06-06 (T3 e2e-verify)
- **Description**: For any project that does not have a per-project `midiconfig.yaml` file in its folder, the WebUI tuning panel is completely non-functional. Three endpoints return 500/404 with `Config file not found`:
  1. `POST /api/config/<id>/midiconfig` — returns 500 (used as step 1 of "Save & Reconvert")
  2. `POST /api/reclassify` — returns 500 (used for live color updates when sliders move)
  3. `POST /api/rebuild-midi` (when config_overrides is nested) — returns 500 (unrelated path mismatch; the dotted-path form works)
- **Root Cause**: 
  - `webui/api/config.py::update_config` calls `get_config_engine(project_id, 'midiconfig')` which raises `ValueError("Config file not found")` when `<project_dir>/midiconfig.yaml` is missing. The 404/500 cascade is then triggered because the front-end does not gracefully handle a missing config.
  - `webui/api/operations.py::reclassify` line 548 does `config = load_config(project['path'] / 'midiconfig.yaml')` with no fallback.
  - The `saveTuningAndReconvert()` JS handler in `webui/static/js/threshold-tuning.js:856-924` always does `api.updateConfig()` first, so when the file is missing the entire "Save & Reconvert" flow dies before the `rebuildMidi(config_overrides=...)` step that the T2 fix added.
- **Impact**: The user's existing funk project (`user_files/1 - 2_funk_80_beat_4-4_4/`) has `validation.has_midiconfig: false`. In this state:
  - The "Tune" button is shown and looks functional
  - Moving any slider triggers a server-side `reclassify` that 500s
  - The "Save & Reconvert" button always errors — the user's slider changes are never persisted
  - The user has no way to interactively tune MIDI detection without first manually creating a `midiconfig.yaml` in the project folder
  - This is the exact workflow the T2 bug D fix was meant to enable, but the prerequisite step is broken
- **Reproduction**:
  ```bash
  ls /Users/jasoncopp/Source/GitHub/larsnet/user_files/1\ -\ 2_funk_80_beat_4-4_4/midiconfig.yaml
  # ls: ...: No such file or directory
  curl -X POST http://localhost:4915/api/reclassify -H 'Content-Type: application/json' \
    -d '{"project_number": 1, "stem_type": "toms", "config_overrides": {"toms.geomean_threshold": 200.0}}'
  # {"error":"Failed to reclassify","message":"Config file not found: .../midiconfig.yaml"}
  curl -X POST http://localhost:4915/api/config/1/midiconfig -H 'Content-Type: application/json' \
    -d '{"updates":[{"path":["toms","geomean_threshold"],"value":200.0}]}'
  # {"success":false,"error":"Config file not found: .../midiconfig.yaml"}
  ```
- **Expected Behavior**: Either (a) auto-create a default `midiconfig.yaml` from the repo's `midiconfig.yaml` on first save, (b) skip the `updateConfig` call entirely when there are no persisted overrides (rely on the rebuild's `config_overrides` only), or (c) surface a clear "create a config to start tuning" toast in the WebUI instead of a 500.
- **Suggested Fix**: 
  1. In `webui/static/js/threshold-tuning.js::saveTuningAndReconvert`, make the `updateConfig` call best-effort: if it 404s/500s with "Config file not found", log a warning and proceed to `rebuildMidi(config_overrides=...)` anyway. The rebuild endpoint already accepts overrides and does not need a pre-existing file.
  2. In `webui/api/operations.py::reclassify`, fall back to a default `config = load_config()` (loads repo-root `midiconfig.yaml`) when the project file is missing. Apply the `config_overrides` on top.
  3. In `webui/yaml_config_core.py::get_config_engine`, on `config_file.exists() == False`, call `engine = YAMLConfigEngine(default_config_path, project_dir / 'midiconfig.yaml')` and copy the default template into the project dir on first use. (This is the existing pattern in `webui/api/config.py:reset_config`.)
- **Files**: `webui/api/config.py`, `webui/api/operations.py`, `webui/yaml_config_core.py`, `webui/static/js/threshold-tuning.js`

### /api/projects/<id>/event-overrides URL pattern is broken (double "projects" prefix)
- **Status**: Open (regression)
- **Priority**: High
- **Date Found**: 2026-06-06 (T3 e2e-verify)
- **Description**: The event-override endpoints (`GET`/`PUT /event-overrides`) are unreachable from the WebUI's JavaScript. The Flask blueprint route is `/api/projects/projects/<int:project_number>/event-overrides` (with `projects` repeated twice — once in the `url_prefix='/api/projects'` and once in the route path), but the JS client in `webui/static/js/api.js:179,183` calls `/api/projects/${projectNumber}/event-overrides` (single `projects`). The result: every save/load of manual event KEPT/FILTERED overrides returns 404, and the user can click an event bar to toggle its status (the in-memory state changes), but the change is never persisted to `midi/event_overrides.json` and is lost on reload.
- **Steps to Reproduce**:
  1. Open the WebUI, select project 1, switch to a stem tab (e.g. Toms)
  2. Click an event bar in the waveform
  3. Check the network tab — see `PUT /api/projects/1/event-overrides` → 404
  4. Reload the page — the toggle is gone
  5. `ls midi/event_overrides.json` — the file is empty `{}`
- **Expected Behavior**: Clicking an event bar should persist the KEPT/FILTERED toggle to `midi/event_overrides.json` and survive a page reload.
- **Actual Behavior**: Toggles are in-memory only. The whole event-override feature is silently broken.
- **Root Cause**: Route definition mismatch. In `webui/api/projects.py:771,798`:
  ```python
  @projects_bp.route('/projects/<int:project_number>/event-overrides', methods=['GET'])
  @projects_bp.route('/projects/<int:project_number>/event-overrides', methods=['PUT'])
  ```
  Combined with the blueprint's `url_prefix='/api/projects'` (from `webui/app.py:69`), the final URL pattern is `/api/projects/projects/<n>/event-overrides`. The JS calls `/api/projects/${n}/event-overrides`. A direct curl to `/api/projects/projects/1/event-overrides` succeeds; the JS path 404s.
- **Workaround (verified)**: Hit the correct URL via curl/Python until the JS route is fixed:
  ```bash
  curl -X PUT 'http://localhost:4915/api/projects/projects/1/event-overrides' \
    -H 'Content-Type: application/json' \
    -d '{"overrides": {"toms": {"2.0782": "FILTERED"}}}'
  # {"saved": true}
  ```
- **Suggested Fix**: Either change the route to `@projects_bp.route('/<int:project_number>/event-overrides', ...)` (remove the redundant `projects/` prefix), or change the JS to call `\`/projects/${projectNumber}/event-overrides\`` (with the extra `projects/`). The first option is simpler and matches what the JS client expects.
- **Files**: `webui/api/projects.py:771,798`

### Snare reclassifies 5/10 events as rimshot on fresh conversion
- **Status**: Open
- **Priority**: High
- **Date Found**: 2026-06-06 (T3 e2e-verify)
- **Description**: On a fresh conversion of the funk project (with the current T1+T2 code), 5 of 10 snare events are classified as rimshot (note 37) when the prior baseline classified all 10 as snare (note 38). The classification index changes from 0 (snare) to 1 (rimshot) for the events at times 2.426, 2.601, 3.715, 4.272, and 4.644 seconds.
- **Steps to Reproduce**:
  1. `cp user_files/1\ -\ 2_funk_80_beat_4-4_4/midi/*.analysis.json /tmp/funk_baseline.json` (capture the prior output)
  2. Restore the baseline MIDI/analysis from the T2 deliverable
  3. `python -c "from stems_to_midi_cli import stems_to_midi_for_project; from project_manager import get_project_by_number, USER_FILES_DIR; stems_to_midi_for_project(get_project_by_number(1, USER_FILES_DIR))"`
  4. `python -c "import json; print(json.load(open('user_files/1 - 2_funk_80_beat_4-4_4/midi/2_funk_80_beat_4-4_4.analysis.json'))['stems']['snare']['events_configured'][0])"`
- **Expected Behavior**: Same classification as the prior baseline (all 10 → note 38, classification 0).
- **Actual Behavior**: 5/10 events are reclassified (note 37, classification 1).
- **Root Cause**: Likely a T2 change in the snare clustering path. The fresh conversion also writes a new `pitch_hz: null` field on snare events (T2 fix B), and the cluster feature input likely changed as a result. Without pitch detection enabled (the YAML default after T1 is `enable_pitch_detection: false`), the classifier should fall back to brightness/onset-shape features and still produce 10 snare hits. Suspect file: `stems_to_midi/note_classification_core.py::classify_snare_notes` or a new branch in `rebuild_core.py` that strips/transforms the classification.
- **Impact**: A user who runs the WebUI "Convert Stems to MIDI" button today gets a different snare output than a user who runs the same conversion before T2. The MIDI file size changes (2 fewer toms events at threshold 200, but here all snare notes change from 38 → 37 for half the events). Any DAW that depends on the snare note number being 38 (the GM standard) will mis-rout these hits to a rimshot sample.
- **Suggested Fix**: Inspect the snare cluster feature pipeline. Compare the per-event feature dicts (geomean, spectral_centroid_hz, attack_sharpness, body_energy/wire_energy ratio, etc.) between the baseline and fresh conversions to identify which feature is now in a different range. Likely culprit: the new `pitch_hz: null` field is being used as a clustering input, or the cluster_feature default changed.
- **Files**: `stems_to_midi/note_classification_core.py`, possibly `stems_to_midi/rebuild_core.py`

### analysis.json hihat events lost `hihat_state` field after fresh conversion
- **Status**: Open
- **Priority**: High
- **Date Found**: 2026-06-06 (T3 e2e-verify)
- **Description**: On a fresh conversion of the funk project, every hihat KEPT event in `analysis.json` is missing the `hihat_state` field (which used to be `'open'` or `'closed'`). The MIDI note number is still set correctly (42 = closed, 46 = open), but the human-readable state is gone. This breaks any downstream consumer that reads `hihat_state` instead of mapping note→state (the Waveform panel's open/closed cluster analysis, the rebuild path's "preserve hihat_state" logic in `rebuild_core.py`, etc.).
- **Steps to Reproduce**:
  ```python
  import json
  d = json.load(open('user_files/1 - 2_funk_80_beat_4-4_4/midi/2_funk_80_beat_4-4_4.analysis.json'))
  hh = d['stems']['hihat']['events_configured']
  kept = [e for e in hh if e['status'] == 'KEPT']
  print(sum(1 for e in kept if 'hihat_state' in e), '/', len(kept))
  # 0 / 13
  ```
- **Expected Behavior**: All 13 KEPT hihat events should have `hihat_state: 'closed'` (10 events) or `hihat_state: 'open'` (3 events), matching the baseline.
- **Actual Behavior**: 0/13 have `hihat_state` in the fresh conversion. The prior baseline had 13/13.
- **Root Cause**: `stems_to_midi/midi.py::save_analysis_sidecar` only writes the `classification` field (line 227-229) for KEPT events; it never writes `hihat_state`. The baseline analysis was created by an earlier code path that explicitly serialized `hihat_state` (probably via `processing_shell.py:1167-1181` which calls `detect_hihat_state()` and writes the result to each event). The migration to the new sidecar format in T2 did not include `hihat_state` in the always-present fields.
- **Impact**: 
  - The T2 fix A4 ("preserve hihat_state on rebuild") is a no-op in the WebUI's typical flow because the field is never written on initial conversion. A subsequent rebuild that says "preserve stored hihat_state" preserves nothing.
  - The hihat tuning panel's "Cluster By" feature depends on `hihat_state` (per `note_classification_core.py:565,646`); without it, cluster groups default to note number.
  - The "show open/closed in legend" works in the UI (it derives from the note number), but the underlying data structure is inconsistent with the T2 design.
- **Suggested Fix**: Add `hihat_state` to the ALWAYS_PRESENT_FIELDS list in `stems_to_midi/midi.py:184` (or as a new always-present field). In `_create_event` (line 194+), copy `onset_data.get('hihat_state')` to the event dict for hihat stems. Verify that the field is set in `onset_data` by tracing the call from `processing_shell.detect_hihat_state()` through to the save sidecar.
- **Files**: `stems_to_midi/midi.py:184,194-232`


---

## T2 follow-up bugs — 2026-06-08

User reported a TypeError toast: `stems_to_midi_for_project() got an unexpected keyword argument 'onset_threshold'` when clicking Convert in the WebUI on their funk project. Investigation found this was the visible tip of a larger surface: every T1/T2 fix was *partially* shipped — the API route, the work function signature, the JSON serialization, the classification preservation, and the route registration all had related gaps that the test suite didn't catch. Wrote 14 failing tests first, fixed code minimally, all green; ran the user's actual funk project end-to-end as the final smoke test, zero drift on rebuild.

### stems-to-midi route blindly splats request body into the work function (the user's reported TypeError)
- **Status**: Fixed
- **Priority**: High
- **Reported**: 2026-06-08 by user
- **Description**: WebUI toast: `stems-to-midi Failed: stems_to_midi_for_project() got an unexpected keyword argument 'onset_threshold'`. Clicking Convert in the WebUI on the user's funk project fails with a 500.
- **Root Cause**: T1 drift-fix (2026-06-06, commit `aad9836`) rewrote `stems_to_midi_for_project()`'s signature to take only `(project, config, stems_to_process, max_duration, learning_mode)`. The CLI now reads per-stem thresholds from the config dict. But the WebUI route at `webui/api/operations.py::stems-to-midi` (and its worker `run_stems_to_midi`) was never updated. It still does:
  ```python
  kwargs = {k: v for k, v in data.items() if k != 'project_number'}
  job_queue.submit(func=run_stems_to_midi, **kwargs)
  ```
  which splats every request-body field (including the now-removed `onset_threshold`, `onset_delta`, `onset_wait`, `hop_length`, `min_velocity`, `max_velocity`, `tempo`, `detect_hihat_open`, etc.) into `stems_to_midi_for_project(**kwargs)`. The function raises TypeError on the first stale kwarg.
- **Why the test suite missed it**: `webui/test_api.py:217::test_stems_to_midi` mocks `webui.api.operations.get_job_queue` and asserts the route returns 202 + a job_id. The real work function is never invoked, so the bad call-site is invisible. This is the same mock-at-queue-vs-mock-at-work-fn gap the T2 verifier flagged for rebuild-midi with "direct /api/rebuild-midi with config_overrides WORKS — proves endpoint is correct, JS orchestration is the problem."
- **Fix**:
  - Rewrote `webui/api/operations.py::run_stems_to_midi` to accept only the modern kwargs (project_number, config_overrides, stems_to_process, max_duration, learning_mode). It loads the project's midiconfig.yaml, applies `config_overrides` (dotted-YAML-path) on top, and passes the merged config dict to `stems_to_midi_for_project`.
  - Updated the route to extract only the modern kwargs from the request body.
  - **Tests written first** (TDD, per the user's hard rule):
    - `webui/test_api.py::TestStemsToMidiKwargsContract` — 4 tests, mock at the work-function level so the real call path is exercised, assert no stale kwargs leak through.
- **Files**: `webui/api/operations.py:60-148, 320-368`
- **Commit**: pending

### /api/projects/<n>/event-overrides URL is double-prefixed (T3 finding)
- **Status**: Fixed
- **Priority**: High (silent — every click-to-toggle event bar never persisted)
- **Description**: T3 e2e found GET and PUT `/api/projects/1/event-overrides` both 404. Click-to-toggle on event bars in the WebUI never persisted across reloads.
- **Root Cause**: Route registered as `/projects/<int:project_number>/event-overrides` (line 771, 798 of `webui/api/projects.py`) inside a blueprint (`projects_bp`) whose `url_prefix` is already `/api/projects` (line 10 of `webui/api/__init__.py`). The full URL became `/api/projects/projects/<n>/event-overrides`. The JS calls `/api/projects/<n>/event-overrides` (no `projects/` segment) → 404.
- **Why the test suite missed it**: Zero direct tests for `event-overrides`. T3 caught it only via Playwright e2e drive.
- **Fix**:
  - Changed the route paths from `/projects/<int:project_number>/event-overrides` to `/<int:project_number>/event-overrides`. The blueprint prefix supplies the `/api/projects/` segment.
  - **Tests written first**:
    - `webui/test_api.py::TestEventOverridesRoute` — 3 tests (GET empty, GET populated, PUT persists).
    - `webui/test_api.py::TestRouteRegistration` — route→JS URL smoke test that catalogs every URL the JS uses and asserts each resolves to a real Flask route. Prevents this class of bug from recurring when blueprints are renamed.
- **Files**: `webui/api/projects.py:771, 798`
- **Commit**: pending

### /api/config/<id>/midiconfig PUT 500s on projects without per-project midiconfig.yaml (T3 finding)
- **Status**: Fixed
- **Priority**: High
- **Description**: T3 e2e found the Save & Reconvert flow 500s with "Config file not found" when the JS calls `/api/config` first before `/api/rebuild-midi`. The whole tuning panel is non-functional for any project without a per-project config.
- **Root Cause**: `YAMLConfigEngine.load()` raises `FileNotFoundError` when the per-project config doesn't exist. The route's `except Exception` handler catches it and returns 500.
- **Why the test suite missed it**: No test exercised the no-config-file case.
- **Fix**:
  - Added a `FileNotFoundError` handler in `webui/api/config.py::update_config` (and `validate_config`) that returns 409 with a useful message and a `hint` field telling the user to create a per-project config or update the root.
  - **Test written first**: `webui/test_api.py::TestConfigUpdateMissingFile::test_config_update_missing_file_returns_clean_error` — asserts the response is 4xx, not 5xx.
- **Files**: `webui/api/config.py:210-222`

### hihat_state field never written to analysis.json (T2 A4 was a no-op, T3 finding)
- **Status**: Fixed
- **Priority**: High
- **Description**: T3 e2e found "hihat_state field missing from all 13 hihat KEPT events in fresh conversion (baseline had 13/13)". The T2 A4 "preserve hihat_state on rebuild" fix is a no-op because the field is never written initially.
- **Root Cause**: `stems_to_midi/midi.py::_serialize_onset_events` only writes fields that exist in `onset_data`. The `hihat_state` field is set on the in-memory event dict by `classify_hihat_notes` in `note_classification_core.py`, but it's never copied to `onset_data`, so the serializer drops it.
- **Fix**:
  - In `_serialize_onset_events`, after writing `classification`, also write `hihat_state` from `midi_events[i]` to the event dict. The round-trip is preserved by `save_analysis_sidecar` and `load_analysis_sidecar`.
  - **Tests written first** (TDD):
    - `stems_to_midi/test_midi_serialization.py::TestSerializeHihatState` — 5 tests covering open/closed/handclap propagation, omission rule, and full round-trip through `save_analysis_sidecar` + `load_analysis_sidecar`.
- **Files**: `stems_to_midi/midi.py:222-237`
- **Verified end-to-end**: 367/367 hihat events in the user's funk sidecar now have `hihat_state`. (Was 0/13 before the fix.)

### Snare/toms/cymbals classifications re-run on every rebuild (T3 finding)
- **Status**: Fixed
- **Priority**: High (silent — produces wrong MIDI when the user thought they were just refreshing)
- **Description**: T3 e2e found fresh conversion reclassifies 5/10 snare events as rimshot (note 37 vs baseline 38). The T2 design preserves `hihat_state` on rebuild but the other per-stem classifiers (`classify_tom_notes`, `classify_cymbal_notes`, `classify_snare_notes`) ignored the `force_reclassify` flag and always re-ran k-means.
- **Root Cause**: The `classify_*_notes` functions for snare/toms/cymbals never accepted a `force_reclassify` parameter. They always overwrote `event['classification']` with a fresh k-means result, even when the user hadn't changed anything.
- **Fix**:
  - All three classifiers (`classify_tom_notes`, `classify_cymbal_notes`, `classify_snare_notes`) now accept `force_reclassify=False` and short-circuit when all events already have stored classifications. Same pattern as `classify_hihat_notes` (T2 A4).
  - Extended `_classification_thresholds_changed` in `rebuild_core.py` to check snare/toms/cymbals classification keys (`expected_clusters`, `cluster_feature`, `midi_note`, `midi_note_*`). The previous comment claimed "already trigger reclassification via the 'classification' slider key path" but that path was unimplemented.
  - **Tests written first** (TDD):
    - `stems_to_midi/test_rebuild_core.py::TestSnareClassificationBaseline` — 3 tests:
      - `test_rebuild_preserves_snare_classification_when_thresholds_unchanged` (all-same baseline)
      - `test_rebuild_preserves_mixed_snare_classifications` (60/40 split)
      - `test_rebuild_reclassifies_when_expected_clusters_changes` (positive control — confirms `force_reclassify=True` actually reclassifies)
- **Files**: `stems_to_midi/note_classification_core.py:238-280, 294-336, 349-394`, `stems_to_midi/rebuild_core.py:120-145`
- **Verified end-to-end**: 137 snare events in the user's funk project, 60/40 split, survive rebuild with zero drift.

### /api/rebuild-midi returns 400 instead of 409 when the project has no MIDI yet (T3 finding)
- **Status**: Fixed
- **Priority**: Medium
- **Description**: T3 e2e found the rebuild endpoint returns 400 with "No MIDI file found in project" when the user clicks Save & Reconvert before any MIDI exists. The 400 vs 409 distinction matters for the WebUI to show "run full conversion first" cleanly.
- **Root Cause**: `stems_to_midi/rebuild_shell.py::rebuild_midi_for_project` returns the error dict without `requires_full_pipeline=True`. The route maps that to 400.
- **Fix**:
  - Both "No midi directory" and "No MIDI file found" early returns now set `requires_full_pipeline=True`. The route maps that to 409.
  - **Test written first**: `webui/test_rebuild_api.py::TestRebuildMidiErrors::test_missing_analysis_returns_409_or_404` (extended to 404/409; the 409 is the documented contract).
- **Files**: `stems_to_midi/rebuild_shell.py:125-141`

### /api/stems-to-midi 500s with "module 'stems_to_midi_cli' has no attribute '_load_project_config_for_project'" (T2 follow-up round 2)
- **Status**: Fixed
- **Priority**: High (WebUI Convert button non-functional — every stems-to-midi click crashed)
- **Description**: After T2 round 1, clicking the WebUI Convert button still 500'd. The route's work function `run_stems_to_midi` calls `stems_to_midi_cli._load_project_config_for_project(project)` and `stems_to_midi_cli._apply_cli_overrides_to_config(config, overrides)`, but those helpers were defined in `webui/api/operations.py` — not in the file that the importlib loader actually loaded (`stems_to_midi_cli.py`).
- **Root Cause**: `webui/api/operations.py:87` does `importlib.util.spec_from_file_location("stems_to_midi_cli", stems_to_midi_path)` then `spec.loader.exec_module(stems_to_midi_cli)`. The resulting module only has attributes defined in the loaded file. The two helpers I added in T2 round 1 (commit 1b2a348) live in `webui/api/operations.py`, so the loaded module doesn't see them — every WebUI Convert click crashes with `AttributeError: module 'stems_to_midi_cli' has no attribute '_load_project_config_for_project'`.
- **Why the test suite missed it**: My T2 round 1 tests mocked at the work-function level (or at the queue level) — they never invoked the importlib loader with the real work function, so the missing attribute never surfaced. The "Mock-at-the-job-queue hides real call-site bugs" memory entry describes the queue-mock gap; this bug is the same shape but for the importlib-loader attribute lookup.
- **Fix**:
  - Moved both helpers (`_load_project_config_for_project`, `_apply_cli_overrides_to_config`) into `stems_to_midi_cli.py`. The work function's `stems_to_midi_cli._load_project_config_for_project(...)` and `stems_to_midi_cli._apply_cli_overrides_to_config(...)` calls now resolve correctly because the helpers are on the loaded module's namespace.
  - Removed the duplicate helper definitions from `webui/api/operations.py`.
  - **Tests written first** (TDD):
    - `webui/test_api.py::TestStemsToMidiImportlibContract::test_loaded_module_exposes_config_loader` — asserts the importlib-loaded module has `_load_project_config_for_project`. (Was: red with the exact `AttributeError` the user hit. Now: green.)
    - `test_loaded_module_exposes_override_applier` — same shape for `_apply_cli_overrides_to_config`.
    - `test_run_stems_to_midi_loads_config_via_loaded_module` — end-to-end through the loaded module: invokes both helpers via the importlib-loaded namespace, asserts empty config when no midiconfig.yaml, asserts override applier merges dotted paths correctly.
- **Files**: `stems_to_midi_cli.py:421-465`, `webui/api/operations.py:60-110` (helpers removed)
- **Verified end-to-end**: User's funk project #1, WebUI Convert button click, full pipeline runs to `Completed successfully!` with 367 + 2 + … MIDI events per stem. Old behavior: instant crash. New behavior: full conversion.
- **Lesson for future code**: When a work function uses `importlib.util.spec_from_file_location(...)` to load a module, every helper it calls on the loaded module must be defined in the loaded file. Use direct `from x import y` for helpers that don't need the importlib indirection.
