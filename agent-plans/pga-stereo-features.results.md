# Results: Stereo Features in PGA Pipeline

## Status

**Complete** — 2026-06-30

## Phase 1: Plumbing

- [x] `processing_shell_percentile_gated.py` — pass `audio_stereo` to builder
- [x] `pga_event_builder._build_pga_events_with_filter` — accept `audio_stereo` kwarg
- [x] `pga_event_builder._compute_features_for_filtered_events` — accept `audio_stereo` kwarg

## Phase 2: Stereo pass

- [x] Add `calculate_stereo_features` call after the existing mono feature pass
- [x] Defensive try/except for bad audio data

## Phase 3: Tests

- [x] Unit test: mono audio (`audio_stereo=None`) → no exception, fields are None
- [x] Unit test: stereo audio → stereo_width populated, in [0, 1]
- [x] Unit test: mono vs panned → discriminator works (max > min)
- [x] E2E on project 6: sidecar has stereo_width for all KEPT events (snare/toms/cymbals)

## Decision Log

- **Detector stays on mono**: The PGA detector (broadband contrast
  envelope + IQR-thresholded peak picker) is fundamentally temporal.
  Stereo info doesn't help detection — the per-event feature
  extraction is the only place stereo is needed.
- **No new config flag**: The existing `use_stereo: true` gate in
  `_load_and_validate_audio` already controls whether the audio
  stays stereo at the loading step. If the user has `use_stereo:
  false`, the audio is mono at pipeline entry and there's nothing
  to recover. The new stereo pass is a no-op when `audio_stereo is None`.
- **Helper handles 1-D defensively**: `calculate_stereo_features`
  already returns `[{pan_confidence: 0.0, stereo_width: 0.0}, ...]`
  for non-stereo input — so even if someone passes a 1-D array by
  mistake, the pipeline doesn't crash. The `_ALLOW_ZERO_FEATURES =
  {'stereo_width', 'pan_confidence'}` set in `note_classification_core`
  ensures 0.0 isn't treated as "missing data" by the cluster resolver.
- **Why the third test was tricky**: My initial test used anti-phase
  L/R panning ('wide') to maximize stereo_width, but anti-phase hits
  cancel to mono and the detector can't find them. Switched to
  'right' panning (R-only), which still gives width ≈ 0.5
  while keeping the mono mix identical to a regular hit.

## Metrics

- **Files changed**: 3 source + 1 test file + 2 plan/results markdown
- **Lines**: ~30 in source, ~270 in new test file, ~75 in plan/results
- **Tests added**: 3 (all pass)
- **Pytest**: 845 passed (up from 842 — the 3 new tests), 4 pre-existing
  failures (unchanged — cymbals kept=6 vs 48 ground-truth, etc.)

## End-to-end on project 6 (full conversion)

| Stem | KEPT | stereo_width populated | range | rel_iqr | classification | notes |
|------|------|------------------------|-------|---------|----------------|-------|
| snare | 247 | 247/247 (100%) | 0.028–0.629 | 1.281 | 121 vs 126 | 38 vs 37 |
| toms | 10 | 10/10 (100%) | 0.180–0.415 | 0.784 | 4 vs 6 | 45 vs 47 |
| cymbals | 13 | 13/13 (100%) | 0.315–0.561 | 0.243 | all 0 | all 49 |

Snare's wide spread lets k-means actually split into snare vs rimshot
(38 vs 37). Toms uses pitch_hz (priority 1) for its split; stereo_width
is now a real fallback when pitch isn't available. Cymbals still uses
spectral_centroid_hz (priority 1) and the centroid data is tight enough
to trigger the spread guardrail on the centroid axis (not the stereo
axis) — separate concern, not in scope for this commit.

## Test Results

```
stems_to_midi/tests/test_pga_stereo_features.py::TestStereoFeaturesInPGAPipeline
::test_mono_audio_audio_stereo_none PASSED
::test_stereo_audio_populates_fields  PASSED
::test_wide_stereo_has_larger_width_than_mono PASSED
3 passed in 2.12s
```

Full suite: `845 passed, 4 pre-existing failures (unchanged), 8 skipped, 29 deselected`.