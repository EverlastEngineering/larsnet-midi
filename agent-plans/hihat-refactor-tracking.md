# Hihat Refactor Tracking

## Goal
Remove separate `detect_hihat_open` toggle and integrate hihat into the same classification system as snare/toms/cymbals using threshold sliders (`open_geomean_min`, `open_sustain_ms`).

## Current State (Before)
- Hihat has separate `detect_hihat_open` checkbox in settings UI
- YAML has `detect_open: true` (but code defaults to False)
- Separate threshold sliders exist (`open_geomean_min`, `open_sustain_ms`) but aren't fully integrated
- Hihat NOT shown in tuning panel like other stems (no clustering visualization)

## Target State (After)
- Remove `detect_hihat_open` / `detect_open` entirely
- Always run hihat classification using threshold sliders
- Show hihat in tuning panel UI with classification (open/closed) like other stems
- If user wants to disable (avoid false positives on weak stems), they can set thresholds to extreme values or add a simple "enabled" toggle

## Tasks

### Phase 1: Remove Old Code
- [ ] 1. Remove `detect_hihat_open` from CLI (`stems_to_midi_cli.py`)
- [ ] 2. Remove `detect_hihat_open` from API (`operations.py`)  
- [ ] 3. Remove `detect-hihat-open` from webui settings (settings.js, settings_schema.py)
- [ ] 4. Remove `detect_open` from YAML configs (midiconfig.yaml, etc.)
- [ ] 5. Update documentation

### Phase 2: Core Changes
- [ ] 6. Modify `processing_shell.py` - always run hihat classification (remove conditional)
- [ ] 7. Ensure `classify_hihat_notes` always runs in rebuild

### Phase 3: UI Changes  
- [ ] 8. Add hihat to tuning panel UI (show open/closed like clusters)
- [ ] 9. Add slider controls for `open_geomean_min`, `open_sustain_ms`

### Phase 4: Testing
- [ ] 10. Run tests
- [ ] 11. Manual end-to-end test

## Key Files to Modify

1. `stems_to_midi_cli.py` - Remove detect_hihat_open parameter
2. `webui/api/operations.py` - Remove from docstring
3. `webui/static/js/settings.js` - Remove checkbox
4. `webui/static/js/operations.js` - Remove from API call
5. `webui/settings_schema.py` - Remove setting
6. `midiconfig.yaml` - Remove detect_open
7. `stems_to_midi/processing_shell.py` - Always run classification
8. `webui/static/js/threshold-tuning.js` - Add hihat to UI

## Notes
- The threshold sliders (`open_geomean_min`, `open_sustain_ms`) already exist and work
- The issue was the separate toggle that bypassed them
- After removal, hihat should work like other stems: sliders control classification
