# Note-Type Color Coding — Results

## Phase 1: Settings Schema + Slider Configs
- [ ] `open_geomean_min` added to settings_schema.py
- [ ] `open_sustain_ms` added to settings_schema.py
- [ ] Slider configs added to STEM_SLIDER_CONFIGS

## Phase 2: Reclassify API Endpoint
- [ ] Endpoint created in operations.py
- [ ] API client method added to api.js
- [ ] Tests passing

## Phase 3: Note-Type Color Palette
- [ ] NOTE_TYPE_COLORS defined
- [ ] getMarkerColor updated
- [ ] Legend updated
- [ ] Tooltip updated

## Phase 4: Wire Sliders to Reclassify
- [ ] Debounced reclassify call on slider change
- [ ] Classification fields merged into events
- [ ] Re-render with colors

## Phase 5: Persist Classification
- [ ] hihat_state/classification written to analysis.json
- [ ] Stale fields cleared from non-KEPT events

## Decision Log
- Python-only classification — no JS duplication
- Debounced API call (500ms) for real-time preview
