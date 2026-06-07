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

