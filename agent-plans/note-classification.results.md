# Two-Pass Note Classification — Results

## Phase 1: Core module + tests
- [x] `note_classification_core.py` created — 5 classifiers + dispatcher + helpers
- [x] `test_note_classification_core.py` created — 52 tests across 7 test classes
- [x] All 52 classifier tests passing

## Phase 2: Rebuild integration
- [x] `classify_notes()` wired into `rebuild_core.py` before `_events_to_midi()`
- [x] `_resolve_note()` kept as fallback (classify_notes sets `note` field first)
- [x] All 47 rebuild tests passing

## Phase 3: Full pipeline integration
- [x] `processing_shell.py` updated — removed audio pitch detection steps 8/8b/8c
- [x] `classify_notes()` runs as Pass 2 after `_create_midi_events()`
- [x] Hihat `detect_hihat_state()` retained for foot-close event generation
- [x] `spectral_centroid_hz` now copied to MIDI events from spectral data
- [x] Full test suite: 871 passed, 2 pre-existing failures (unrelated audio threshold tests)

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-XX | Keep detect_hihat_state in full pipeline | Foot-close events require hihat_state before MIDI event creation. classify_notes then re-classifies consistently. |
| 2025-01-XX | Use spectral_centroid_hz for clustering instead of YIN pitch | Already stored in analysis.json for all events, no audio needed, equivalent clustering signal |
| 2025-01-XX | Keep _resolve_note as fallback | Backward compatibility for events with pre-existing note fields from old analysis.json |
