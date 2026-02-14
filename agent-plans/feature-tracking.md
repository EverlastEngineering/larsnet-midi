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
