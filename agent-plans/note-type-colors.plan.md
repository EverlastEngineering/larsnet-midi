# Note-Type Color Coding & Real-Time Classification Preview

## Objective

Color-code waveform event markers by note type (open/closed hihat, tom low/mid/high,
crash/ride/chinese, snare/rimshot/clap) instead of uniform green. Expose hihat
open/closed classification thresholds as tunable sliders. Use server-side Python
classification exclusively — no JS classification logic — with debounced API calls.

## Architecture

All classification logic stays in Python (`note_classification_core.py`). The frontend
is purely a renderer of server-provided data. Slider changes trigger debounced API
calls to a lightweight `/api/reclassify` endpoint that runs `classify_notes()` on
the current KEPT events and returns updated classification fields.

## Phases

### Phase 1: Settings Schema + Slider Configs
- Add `open_geomean_min`, `open_sustain_ms` to `settings_schema.py` (HIHAT category)
- Add corresponding slider entries to `STEM_SLIDER_CONFIGS` in `threshold-tuning.js`

### Phase 2: Reclassify API Endpoint
- New endpoint: `POST /api/reclassify`
- Input: `{ project_number, stem_type, config_overrides: { open_geomean_min, open_sustain_ms, ... } }`
- Runs `classify_notes()` with merged config on KEPT events from analysis.json
- Returns: `{ events: [{ time, note, hihat_state?, classification? }] }` (minimal payload)
- No MIDI rebuild, no disk write — classification preview only

### Phase 3: Note-Type Color Palette
- Define `NOTE_TYPE_COLORS` mapping MIDI note numbers → hex colors
- Distinct hues per sub-type within each stem
- Update `getMarkerColor()` to use note-based color for KEPT events
- Update legend to show note-type breakdown
- Update tooltip to show classification info

### Phase 4: Wire Sliders to Reclassify
- On classification-slider change, debounce (500ms) and call `/api/reclassify`
- Merge returned classification fields into local event copies
- Re-render waveform with note-type colors
- Clear classification preview on panel close or stem switch

### Phase 5: Persist Classification in Analysis
- Ensure rebuild writes `hihat_state`/`classification` to analysis.json for KEPT events
- Clear stale classification fields from non-KEPT events

## Risks
- k-means on <3 events: already handled (percentile fallback in `_cluster_values`)
- Debounce latency: 500ms debounce + ~50ms server round-trip = acceptable
- Color palette accessibility: use hues with good contrast on dark background

## Success Criteria
- KEPT events show distinct colors per note type in waveform
- Hihat open/closed boundary adjustable via sliders with live preview
- All classification runs server-side in Python
- No regressions in existing tests
