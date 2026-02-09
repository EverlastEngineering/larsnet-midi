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
