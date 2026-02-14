# Two-Pass Note Classification — Plan

## Goal

Move MIDI note classification to a post-filtering step so it runs on the final KEPT event set, using stored spectral features instead of audio-dependent pitch detection. This ensures note assignments (open/closed hihat, crash/ride/chinese cymbal, low/mid/high tom, snare types) are always based on the actual event pool — whether from the full pipeline or a rebuild with changed thresholds.

## Problem

Note classification currently runs once during the full pipeline (audio → YIN pitch → k-means) and is baked into each event's `note` field in analysis.json. When rebuild changes which events are KEPT:
- Newly promoted events have no `note` field → fall back to default stem note (all hihat → 42, all cymbals → 49)
- Classification boundaries (e.g., open/closed hihat sustain threshold) should shift based on the kept population, but don't

## Key Insight

All features needed for classification are already stored in analysis.json for every event (KEPT and FILTERED):
- **hihat**: `sustain_ms`, `body_energy`, `sizzle_energy`, `geomean`, `spectral_centroid_hz`
- **toms**: `fundamental_energy`, `body_energy`, `spectral_centroid_hz`
- **cymbals**: `body_energy`, `brilliance_energy`, `spectral_centroid_hz`, `sustain_ms`
- **snare**: `body_energy`, `wire_energy`, `spectral_centroid_hz`

## Architecture

### New Module: `stems_to_midi/note_classification_core.py`

Pure functional core — no I/O, no audio, no side effects.

**Functions:**
1. `classify_hihat_notes(events, config)` — sustain + geomean threshold (matches current `detect_hihat_state()` logic)
2. `classify_tom_notes(events, config)` — k-means on `spectral_centroid_hz` (replaces YIN pitch clustering)
3. `classify_cymbal_notes(events, config)` — k-means on `spectral_centroid_hz` (replaces YIN pitch clustering)
4. `classify_snare_notes(events, config)` — k-means on `spectral_centroid_hz` (replaces YIN pitch clustering)
5. `classify_notes(events, stem_type, drum_mapping, config)` — dispatcher that calls per-stem classifier, maps classification indices to MIDI notes

### Classification Algorithms

**Hihat** (threshold-based, not clustering):
- Compute `geomean = sqrt(body_energy * sizzle_energy)` if not stored
- `open` if `geomean >= open_geomean_min` AND `sustain_ms >= open_sustain_threshold_ms`
- Config keys: `hihat.open_geomean_min` (default 262), `hihat.open_sustain_ms` (default 150)
- Returns `hihat_state` field on each event

**Toms** (k-means k=3 on spectral_centroid_hz):
- Extract valid `spectral_centroid_hz` values from events
- k-means cluster into 3 groups sorted by centroid (0=low, 1=mid, 2=high)
- Fallback: percentile-based splitting when sklearn unavailable or < 3 unique values
- Map: 0 → `drum_mapping.tom_low`, 1 → `drum_mapping.tom_mid`, 2 → `drum_mapping.tom_high`

**Cymbals** (k-means k=3 on spectral_centroid_hz):
- Same pattern as toms but sorted 0=crash(lowest), 1=ride, 2=chinese(highest)
- Map: 0 → `drum_mapping.crash`, 1 → `drum_mapping.ride`, 2 → `drum_mapping.chinese`

**Snare** (k-means k=4 on spectral_centroid_hz):
- k-means with k = min(4, n_unique_values)
- Sorted by centroid: 0=snare, 1=rimshot, 2=clap, 3=clap+snare
- Map: 0 → `drum_mapping.snare`, 1 → `drum_mapping.snare_rimshot`, 2 → `drum_mapping.snare_clap`, 3 → `drum_mapping.snare_clap_snare`

### Integration Points

**rebuild_core.py**: In `rebuild_events_from_analysis()`, after extracting `kept_events` and before calling `_events_to_midi()`, call `classify_notes()`. Remove `_resolve_note()` fallback logic in favor of the classified note.

**processing_shell.py**: In the full pipeline, after all filtering is complete and the final KEPT set is determined, call `classify_notes()` instead of the current per-onset audio pitch detection + classification flow. The audio-based pitch detection (`detect_tom_pitch`, etc.) becomes unnecessary for note assignment.

## Phases

### Phase 1: Core module + tests
- Create `note_classification_core.py` with all classifiers
- Create `test_note_classification_core.py` with comprehensive tests
- No integration yet — standalone pure functions

### Phase 2: Rebuild integration
- Wire `classify_notes()` into `rebuild_core.py`
- Update `_events_to_midi()` to use classified notes
- Verify rebuild produces correct note variety

### Phase 3: Full pipeline integration
- Update `processing_shell.py` to call `classify_notes()` post-filtering
- Remove audio-dependent pitch classification from the main flow
- Verify full pipeline matches or improves on current output

## Risks

1. **spectral_centroid_hz clustering may not match YIN pitch clustering**: The centroid is a broader measure than fundamental pitch. Clustering results may differ. Mitigation: compare on real project data.
2. **Snare k=4 may over-cluster**: With centroid instead of pitch, 4 clusters may not be meaningful. Mitigation: auto-reduce k based on silhouette score or gap statistic.
3. **Backward compatibility**: Existing analysis.json files have `note` fields from the old pipeline. The new classifier should produce equivalent results when the kept set hasn't changed.

## Success Criteria

- All existing tests pass
- Rebuild pipeline produces MIDI with varied notes (not all default)
- Hihat open/closed classification matches current results on unchanged threshold data
- New test coverage for all classifier functions
