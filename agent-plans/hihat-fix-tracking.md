# Hihat Detection Fix - Tracking

## Issue Summary
- detect_open (YAML) doesn't match detect_hihat_open (CLI/API)
- Rebuild always overwrites hihat_state
- Missing hihat UI controls

## Fix 1: Parameter Name Mismatch
**Status**: IN PROGRESS

**Problem**: 
- YAML: `detect_open: true` 
- CLI: `detect_hihat_open: false` (default)
- API: `detect_hihat_open: false` (default)

**Fix**: Remove detect_hihat_open from API - let YAML config take precedence

**Files to change**:
- [ ] webui/api/operations.py - remove detect_hihat_open from API call
- [ ] webui/static/js/operations.js - don't send detect_hihat_open

## Fix 2: Preserve hihat_state in Rebuild
**Status**: NOT STARTED

**Problem**: classify_hihat_notes() always overwrites hihat_state

**Fix**: Only classify if hihat_state not already in event data

**Files to change**:
- [ ] stems_to_midi/note_classification_core.py - check for existing hihat_state

## Fix 3: Add Hihat UI Controls
**Status**: NOT STARTED

**Problem**: No tuning sliders for hihat classification

**Fix**: Add open_geomean_min, open_sustain_ms sliders to hihat tuning panel

**Files to change**:
- [ ] webui/static/js/threshold-tuning.js - add hihat sliders

## Key Findings
- YAML has detect_open: true (midiconfig.yaml line 250)
- CLI defaults detect_hihat_open to False
- classify_hihat_notes ALWAYS runs on rebuild
- hihat_state IS saved to JSON correctly

## Tests to Run
- Test hihat classification with detect_open: true in YAML
- Verify rebuild doesn't change hihat_state
- Test UI sliders change hihat classification
