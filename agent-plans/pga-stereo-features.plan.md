# Plan: Stereo Features in PGA Pipeline

## Problem

The PGA pipeline runs the detector on mono audio ([`processing_shell_percentile_gated.py:148`](stems_to_midi/processing_shell_percentile_gated.py#L148)) and the per-event feature extractor ([`_compute_features_for_filtered_events`](stems_to_midi/pga_event_builder.py#L1126)) only calls `compute_event_features(audio_mono, ...)` — there is no stereo pass. As a result, every PGA event has `stereo_width: None` and `pan_confidence: None` in the sidecar.

The classifier's feature priority for snare/toms/cymbals is stereo_width-first, so the resolver silently falls back to `spectral_centroid_hz` for snare (and never even reaches stereo for the others). The classifier's `_warn_on_cluster_feature_fallback` warning doesn't fire because `cluster_feature='auto'` is the default.

[`calculate_stereo_features(stereo_audio, onset_times, sr)`](stems_to_midi/stereo_core.py#L195) already exists, is pure, and is well-tested ([`test_stereo_core.py:296`](stems_to_midi/test_stereo_core.py#L296)). It's just never called from the PGA path.

## Approach

The detector itself doesn't need stereo — onset detection is fundamentally temporal (broadband contrast envelope + IQR-thresholded peak picker). Stereo info is only needed at **per-event feature extraction time**, where the existing `_compute_features_for_filtered_events` already runs a per-event loop over KEPT events.

The minimal change: plumb the original stereo audio (when available) through to the feature extractor, and add a stereo pass alongside the existing mono pass.

### Phase 1 — Plumbing

1. [`processing_shell_percentile_gated.py`](stems_to_midi/processing_shell_percentile_gated.py) — the call site to `_build_pga_events_with_filter` gains `audio_stereo=audio if audio.ndim == 2 else None`. The original `audio` variable is already in scope (loaded by `_load_and_validate_audio`).

2. [`pga_event_builder._build_pga_events_with_filter`](stems_to_midi/pga_event_builder.py#L1542) — accepts `audio_stereo: Optional[np.ndarray] = None` and passes it to `_compute_features_for_filtered_events`.

3. [`pga_event_builder._compute_features_for_filtered_events`](stems_to_midi/pga_event_builder.py#L1126) — accepts `audio_stereo: Optional[np.ndarray] = None` and runs the stereo pass after the existing mono pass.

### Phase 2 — Stereo pass

After the existing `compute_event_features` per-event loop in `_compute_features_for_filtered_events`, add:

```python
if audio_stereo is not None and events:
    from .stereo_core import calculate_stereo_features
    try:
        stereo_feats = calculate_stereo_features(
            audio_stereo,
            np.array([ev['time'] for ev in events]),
            sr,
        )
        for ev, sf in zip(events, stereo_feats):
            ev['pan_confidence'] = sf['pan_confidence']
            ev['stereo_width'] = sf['stereo_width']
    except Exception:
        # Defensive: bad stereo data shouldn't poison the rest
        # of the features. WebUI shows "N/A".
        for ev in events:
            ev.setdefault('pan_confidence', None)
            ev.setdefault('stereo_width', None)
```

### Phase 3 — Tests

1. **Unit test** in `test_pga_event_builder.py` (or new file): synthetic stereo audio with mono-panned and wide-stereo hits; assert `_compute_features_for_filtered_events` populates `stereo_width` correctly.

2. **Unit test** for mono audio path: pass `audio_stereo=None`, verify events get `stereo_width=None` (no exception, no compute).

3. **E2E** on project 6: re-run `python stems_to_midi_cli.py 6 --stems snare`; inspect sidecar, confirm `stereo_width` populated for KEPT snare events. Confirm `_resolve_cluster_feature` now picks `stereo_width` as the actual feature for snare.

### Phase 4 — Schema / config

**No new config keys needed.** The existing `use_stereo: true` gate (in `_load_and_validate_audio`) already controls whether the audio stays stereo at the loading step. If the user has `use_stereo: false`, the audio is mono and there's no stereo info to recover. The new stereo pass is a no-op for mono sources.

For the schema: no changes. The pre-existing `cluster_feature` setting already exposes stereo_width as a choice (it's in `_FEATURE_PRIORITIES['snare']`).

## Files Changed

1. `stems_to_midi/processing_shell_percentile_gated.py` — pass `audio_stereo` to builder (1 line)
2. `stems_to_midi/pga_event_builder.py` — accept and propagate `audio_stereo`, add stereo pass (~25 lines including docstring)
3. `stems_to_midi/test_pga_event_builder.py` (or new file) — unit test for the stereo pass (~50 lines)

## Risks

- **Performance**: `calculate_stereo_features` is O(events × window). For 10 events it's microseconds; for 2246 events it's still sub-millisecond (the dominant cost is `compute_event_features`'s pitch detection). No measurable impact.
- **Sidecar size**: ~10 extra fields per event × 2246 events ≈ 22 KB. Negligible.
- **Sidecar format**: `stereo_width` and `pan_confidence` are already in the dynamic-passthrough allowlist in [`midi.py`'s `_serialize_onset_events` / `_serialize_pga_events`](stems_to_midi/midi.py) (added in the hihat milestone, 2026-06-29). No schema migration needed for the sidecar JSON.

## Success Criteria

1. Re-run `python stems_to_midi_cli.py 6 --stems snare` on project 6. Sidecar `events_pga[*].stereo_width` is non-None for KEPT events.
2. The classifier's `_resolve_cluster_feature` reports `actual_feature='stereo_width'` for snare (verifiable via the fallback warning when `cluster_feature='stereo_width'` is explicitly set in yaml).
3. New unit tests pass; full pytest run shows no regressions beyond the 4 pre-existing cymbals failures.