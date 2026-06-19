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
