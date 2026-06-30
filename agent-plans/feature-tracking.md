# Feature & Bug Tracking — Waveform Tuning UI

## Feature: Loading indicator with percentage
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Shows a loading overlay with progress bar and percentage text in the waveform area while envelope and analysis data loads per stem.
- **Files**: `webui/static/js/waveform.js`, `webui/templates/index.html`

## Bug: Tuning events have no height (no velocity)
- **Status**: Fixed
- **Priority**: High
- **Description**: When "Tune" mode was active, all event bars were the same height because `events_sensitive` lack a `velocity` field. Fix computes velocity from `strength` field client-side using the same formula as Python's `estimate_velocity()`.
- **Root Cause**: `_serialize_onset_events()` in `stems_to_midi/midi.py` only attaches `velocity` when called with `midi_events` parameter.
- **Files**: `webui/static/js/waveform.js` (`drawEventBars`)

## Bug: Sensitive detection too sensitive
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Increased minimum detection thresholds by 10x. `SENSITIVE_THRESHOLD_DB`: 1.0 → 10.0, `SENSITIVE_MIN_ABSOLUTE_ENERGY`: 0.0001 → 0.001.
- **Files**: `stems_to_midi/processing_shell.py` (`_run_sensitive_detection`)

## Improvement: 4x waveform resolution
- **Status**: Fixed
- **Priority**: High
- **Description**: Increased envelope data resolution from 2000 to 8000 points for sharper waveform visuals.
- **Files**: `webui/api/projects.py` (`get_project_envelope`), `webui/test_analysis_api.py`

## Feature: Click-and-hold audio playback from cursor
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Click and hold on the waveform to play audio from the cursor position. Uses Web Audio API with the existing download route. Audio buffers are pre-fetched per stem. Panning takes precedence when zoomed.
- **Files**: `webui/static/js/waveform.js`

## Feature: Click event to toggle kept/removed
- **Status**: Fixed
- **Priority**: Low
- **Description**: Click on an event bar to toggle it between KEPT and FILTERED. Overrides are stored per-project in `event_overrides.json` (debounced save). Overridden events display a white diamond marker. API endpoints: GET/PUT `/api/projects/:num/event-overrides`.
- **Files**: `webui/static/js/waveform.js`, `webui/static/js/api.js`, `webui/api/projects.py`

## Feature: Time indicator on cursor position
- **Status**: Not Started
- **Priority**: Medium
- **Description**: Display the current time (in seconds or MM:SS format) for the cursor position when hovering over the waveform. Visible in a tooltip or indicator near the crosshair.
- **Files**: `webui/static/js/waveform.js`

## Feature: Two-pass note classification (post-filter)
- **Status**: Complete
- **Priority**: High
- **Description**: Move MIDI note classification to a post-filtering step so it runs on the final KEPT event set, not the pre-filtered set. Currently, note assignment (open/closed hihat, crash/ride/chinese cymbal, tom pitch, snare types) runs once during the full pipeline and is baked into `note` fields in analysis.json. When the rebuild pipeline changes which events are KEPT, note classification is stale — newly promoted events get the default stem note (e.g., all hihat → closed 42).
- **Problem**: Hihat open/closed detection depends on the sustain distribution of the *kept* event set. If filtering changes which events survive, the classification boundary shifts. Same for k-means clustering on toms/cymbals/snare — the cluster assignments depend on which events are in the pool.
- **Proposed Architecture**:
  - **Pass 1 (detection + filtering)**: Detect all onsets, compute spectral features, apply threshold filters. Store raw features in analysis.json (sustain_ms, spectral_centroid_hz, geomean, body/sizzle/wire/fundamental energy). No note assignment yet.
  - **Pass 2 (note classification)**: Run on the final KEPT event set only, using stored features. Pure function: `classify_notes(kept_events, stem_type, config) → events_with_notes`. This runs identically in both full pipeline and rebuild pipeline.
  - **Hihat**: Classify from `sustain_ms` + `geomean` (body × sizzle). Already stored. No audio needed.
  - **Toms**: Cluster from `spectral_centroid_hz` or `fundamental_energy` ratio. Already stored. No audio needed.
  - **Cymbals**: Cluster from `spectral_centroid_hz` or `brilliance_energy` ratio. Already stored. No audio needed.
  - **Snare**: Cluster from `spectral_centroid_hz` or `wire_energy` ratio. Already stored. No audio needed.
  - **Key insight**: All required features are already in analysis.json. The current pipeline uses YIN pitch detection (audio-dependent) for toms/cymbals/snare, but the spectral features already stored (centroid, energy ratios) provide equivalent clustering signals without audio.
- **Data Available Per Stem** (from analysis.json, all events):
  - hihat: `sustain_ms`, `body_energy`, `sizzle_energy`, `geomean`, `spectral_centroid_hz`
  - cymbals: `body_energy`, `brilliance_energy`, `geomean`, `spectral_centroid_hz`, `sustain_ms`
  - snare: `body_energy`, `wire_energy`, `geomean`, `spectral_centroid_hz`
  - toms: `fundamental_energy`, `body_energy`, `geomean`, `spectral_centroid_hz`
- **Files**: New `stems_to_midi/note_classification_core.py` (pure functions), updates to `rebuild_core.py` and `midi.py`

## Feature: Post-filter feature recompute (PGA)
- **Status**: Complete
- **Priority**: High
- **Description**: Move per-event feature extraction (`duration_ms`, `duration_to_valley_ms`, `attack_rise_ms`, `inter_onset_ms`, `pitch_hz`, `decay_t60_ms`, `spectral_*`, etc.) out of `detect_pga_events` and into a post-filter pass that runs against the KEPT event set, not the pre-filter detect-time list. The WebUI tuning panel re-filter path will also need to call this pass when the user changes a threshold slider.
- **Problem**: Neighbor-dependent features (`duration_ms`, `duration_to_valley_ms`, `attack_rise_ms`, `inter_onset_ms`) were bounded against the pre-filter list. A filtered-out FP between two kept strikes capped the prior strike's ring at the FP's time and stretched the next strike's attack across the gap. The WebUI tooltip then showed a ring that was always too short for the prior strike and an attack that was always too long for the next strike. The `event_features.py:1367-1371` docstring on `compute_event_features_for_list` already described the two-pass flow ("detect → filter → re-measure with filtered neighbors, then overwrite the `duration_to_valley_ms` field on the survivors") but `pga_event_builder.py:detect_pga_events` only implemented pass 1 — pass 2 was explicitly deferred as "out of scope here".
- **Architecture**:
  - **`detect_pga_events`** (pure detect, no features) returns events with `time`, `method`, `status='KEPT'`, `frame`, `envelope_value`, `prominence`, `iqr_threshold`, `midi_velocity`, `pga_filter_config`. No per-event features attached.
  - **`_build_pga_events_with_filter`** orchestrates the three-step pipeline: detect → apply prominence + decay_col_min + attack_rise filters → call `_compute_features_for_filtered_events` on the KEPT+FILTERED list. The neighbor lookup (`_find_prev_next_kept`) skips FILTERED events on both sides, so a filtered FP between two kept strikes no longer caps the prior strike's ring.
  - **`build_pga_events`** (legacy public wrapper) does the same: detect → filter → recompute features. The all-KEPT list IS the post-filter list, so the post-filter pass is still correct.
  - **`_compute_features_for_filtered_events`** is the new pure functional core that attaches the per-event features. Lazy-imports `compute_event_features` (librosa/scipy stack, not on cold path). Reads pitch config once (per-stem > global > default).
  - **Sidecar** now uses the post-filter KEPT+FILTERED list from `_build_pga_events_with_filter` instead of the all-KEPT list from `build_pga_events`. The WebUI tooltip can show "why was this dropped" with actual feature values, not None.
  - **WebUI re-filter path** (follow-up) will call `_compute_features_for_filtered_events` itself when the user changes a threshold slider — feature values stay in sync with the user's current filter.
- **Tests**: New `TestPostFilterFeatureRecompute` class in `tests/test_pga_event_builder.py` (functional-core, no detector, no audio I/O):
  - `test_inter_onset_skips_filtered_event` — `_find_prev_next_kept` returns the next KEPT event's time, not the FILTERED event's time.
  - `test_features_attached_to_kept_and_filtered_events` — the post-filter pass attaches the per-event feature keys to BOTH KEPT and FILTERED events (so the sidecar can show "why was this dropped" with actual values).
- **Files**: `stems_to_midi/pga_event_builder.py` (extract `_find_prev_next_kept`, `_compute_features_for_filtered_events`; remove per-event feature block from `detect_pga_events`; add post-filter pass in `_build_pga_events_with_filter` and `build_pga_events`); `stems_to_midi/processing_shell.py` (sidecar source switched to post-filter list); `stems_to_midi/tests/test_pga_event_builder.py` (new `TestPostFilterFeatureRecompute` class).

## Feature: Hihat open/closed via forward-decay slope (post-hoc walk)
- **Status**: In Progress
- **Priority**: High
- **Description**: Surface the broadband contrast envelope to the post-filter pipeline (cached to `{stem}.contrast_envelope.npz`), walk each KEPT event's envelope forward from its peak frame, and record (1) per-frame avg dB-slope, (2) per-frame avg linear-slope, (3) pct_at_stop — where the envelope ended up as a fraction of peak, and (4) the "ring-start" backward walk. The user will set a single threshold slider for open vs closed in the WebUI; the default value is **2.0 dB/frame** (population p50 across all KEPT hihats in the Taylor Swift test project).
- **Discriminator signal**: forward-decay walk hits another KEPT event before the envelope drops to 50% of peak — i.e., the ring is still loud enough that the next strike cuts in. Closed hihats reliably cross to ~49%; open hihats get blocked at 70-100%+ of peak. Slope alone separates cleanly in the dB-domain (closed 3.4-3.6, open 0.7) and is consistent in the linear-domain (closed 0.30, open 0.08) — no log quirk.
- **Files**: `stems_to_midi/midi.py` (`save_contrast_envelope`, `load_contrast_envelope`), `stems_to_midi/pga_event_builder.py` (return debug dict with envelope; per-event walk fields added), `stems_to_midi/processing_shell_percentile_gated.py` (forward `pga_envelope_data` to CLI), `stems_to_midi_cli.py` (save npz after analysis), `scripts/walk_kept_events.py` (post-hoc walk tool, exploratory).
- **Follow-up**: WebUI slider for the slope threshold; production `classify_hihat_by_decay_slope` rule that consumes the per-event slope fields from the sidecar.


## Bug: pga_min_combined_score at "off" position filters everything
- **Status**: Open
- **Priority**: High
- **Description**: When the slider is at the far-left (off / 0.0), the warble filter
  ends up filtering every kept event instead of being a no-op. The user
  expects: at the off position, the filter is disabled (no events filtered,
  key removed from yaml); at any positive value, the filter applies (events with
  combined_score < threshold are filtered). Currently, threshold=0 is treated as
  the strictest filter and every event with combined_score < 0 is dropped —
  but at the off position the user clearly wants no filter applied.
- **Expected behavior**:
  - Slider at off (far left): no filtering, key removed from yaml on save.
  - Slider > off: key saved to yaml with the slider value, filter applies.
  - Display: slider UI shows the off indicator when at the leftmost position.
- **Files to change**:
  - `webui/static/js/threshold-tuning.js`: SLIDER_RENDER_VALUE shows
    "off" when value=0, slider snaps to 0 with step matching the data, and
    the save call (buildConfigUpdates) omits pga_min_combined_score from
    the updates list when value=0.
  - `stems_to_midi/filter_registry.json` or `pga_event_builder.py`: when
    threshold=0 is read from yaml, skip the filter entirely. Mirrors the
    Python convention of treating 0 as a "disabled sentinel" for these
    "off / disabled" filters (see `band_max_ratio_max` and `show_only_snap_events`).
  - `stems_to_mdi/rebuild_core.py` or `rebuild_shell.py`: when reading the
    config for rebuild, treat 0 as "filter disabled" and skip the call
    (preserves current behavior for the 0.0 default).
- **Test**: 05-warble-rebuild.spec.ts add a case where threshold=0 and
  assert the Kept count matches the no-filter baseline (i.e., no events
  are dropped). 04-combined-score.spec.ts add an assertion that the
  default value when yaml omits the key is 0.0 (no filter) and the test
  confirms the spec returns the registry fallback.
- **Reproduction**:
  1. Open the project 10 tuning slideout, move combined_score slider
     to 0.0, save.
  2. Inspect sidecar: all kept events with negative combined_score are
     dropped, leaving a sparse set (should be a no-op).
  3. Move the slider to 100, save. Verify all those events return.
  4. Move the slider to 0 again. Verify no events are dropped.
- **User-visible contract**: the slider should not apply any filter at
  threshold=0; the saved yaml should not contain pga_min_combined_score
  at all when the user puts the slider to the off position.

## Bug: "Show Filtered" toggle does not trigger updateEvents when slideout is closed
- **Status**: Open
- **Priority**: High
- **Description**: The "Show Filtered" toggle (which controls whether filtered
  events appear faded in the waveform) works when the slideout is open and
  a slider value is being moved. But when the slideout is closed and
  the user toggles "Show Filtered", the events do NOT show their filtered
  state. They only update after opening the slideout and touching a
  slider — the slider change handler triggers the updateEvents function
  that the toggle click handler does not.
- **Expected behavior**: Toggling "Show Filtered" (whether the slideout is
  open or closed) should immediately trigger the events re-render so
  filtered events appear faded or full-color consistently.
- **Likely cause**: The toggle button has a click handler that
  updates tuningShowFiltered (in-memory state) but does not call
  applyTuningFilter() or updateEvents(). The slider change handler does
  call applyTuningFilter(), which is why moving a slider works.
- **Files to change**:
  - `webui/static/js/threshold-tuning.js`: the click handler for the
    show-filtered toggle (likely in toggleTuningShowFiltered or wherever
    the "Show Filtered" button is wired) must call applyTuningFilter()
    after updating tuningShowFiltered. Look for the pattern used by the
    slider oninput handler (which calls applyTuningFilter() and then
    re-applies the events) and copy that flow into the click handler.
  - Verify that any other toggle handler in the slideout (e.g. the
    snap-mask toggle) has the same issue and apply the same fix
    consistently.
- **Reproduction**:
  1. Close the slideout.
  2. Toggle "Show Filtered" on.
  3. The waveform does not update — filtered events are NOT faded
     (or are already hidden, depending on the prior state).
  4. Open the slideout, touch any slider (without changing its value).
  5. The waveform updates — events now show their filtered state.
- **Test**: write a Playwright spec (probably as part of 03- or a
  new -waveform-events.spec.ts) that:
  1. Opens project 10 hihat
  2. Toggles "Show Filtered" while the slideout is closed
  3. Asserts that events on the waveform show their filtered
     state (e.g. opacity < 1.0 for filtered events)
  4. Toggles it off and asserts opacity returns to 1.0
- **User-visible contract**: the Show Filtered toggle should trigger
  a re-render regardless of slideout state. Same fix should apply to any
  other toggle/checkbox in the slideout.
