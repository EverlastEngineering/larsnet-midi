# midiconfig.yaml cleanup — results

## Phase 1: Inventory ACTIVE vs DORMANT YAML keys [✅]

Subagent-driven inventory complete. 87 dead keys removed in PGA-cleanup
were counted; another ~25 dormant keys (declared but never read)
catalogued by section.

## Phase 2: Root midiconfig.yaml rewrite [✅]

`midiconfig.yaml` rewritten from 302 lines to 263 lines. Now contains:

- Every ACTIVE key with sensible defaults (audio, midi,
  onset_detection, 5 stems)
- Two DEPRECATED-but-still-wired filters with **large-bypass defaults**:
  - `onset_detection.attack_rise_max_ms: 15000` (OFF by default)
  - `onset_detection.min_decay_col_min_db: -160.0` (OFF by default)
- Commented defaults for all DORMANT keys (declarative
  discoverability without bloating the active schema) —
  fundamental/body/attack freq bands, energy_method, spectral_snap_*,
  etc.
- Removed dead sections: `filtering.*`, `clustering.method`,
  `threshold_optimization.*`, `debug.*`, `learning_mode.*` (kept in
  commented reference block at the end)
- Removed dead per-stem keys: `expected_clusters`, `threshold_db`,
  `min_peak_spacing_ms`, `min_absolute_energy`, `merge_window_ms`,
  `energy_method`, `peak_hold_ms`, `onset_threshold`, `onset_delta`,
  `onset_wait`, `min_strength_threshold`, `min_sustain_ms`,
  `enable_spectral_filter`, `reverb_continuation_attack_threshold`

## Phase 3: Project 6 midiconfig.yaml strip [✅]

Project 6: 246 → 116 lines. Stripped:

- All 14 PGA-cleanup-removed keys per stem × 5 stems = **70 dead keys**
- Full sections: `filtering.*`, `clustering.method`,
  `threshold_optimization.*`, `debug.*`, `learning_mode.*`
- `audio` and `midi` sections (defaults match root; removed)
- `onset_detection` section (no overrides; ALL match root defaults)
- `hihat.classifier_method` (dormant flag, never read by runtime)
- Spectral band keys (fundamental_freq_*, body_freq_*, attack_freq_*,
  low_freq_*, wire_freq_*, brilliance_freq_*, sizzle_freq_*)
- Legacy envelope keys (`sustain_analysis_window_sec`,
  `decay_filter_window_sec`)

Kept the live per-stem PGA-tuning overrides that project 6 actually uses:

- kick: broad bandpass 60-20000 Hz, nms 5, prominence 5800, envelope 9300
- snare: narrow bandpass 250-4000 Hz, prominence 2000, envelope 3100,
  3-cluster, pitch detection stubs
- toms: prominence 1400, envelope 10000, 2-cluster, pitch detection,
  floor gate -80
- hihat: openness_score_threshold 0.75, open_decay_slope_max 0.7, etc.
- cymbals: prominence 8300, envelope 16800, combined_score 1046, delta
  detection method, nms 25

## Phase 4: Verify both load [✅]

- `load_config(Path('midiconfig.yaml'))` loads cleanly (8 top-level keys)
- `load_config(Path('user_files/6*/midiconfig.yaml'))` loads cleanly
- `/api/rebuild-midi?project_number=6` returns success=True, all 5 stems
  rebuilt, 1255 KEPT events (same as before cleanup — behavior unchanged)
- Playwright regression tests (08b, 08d) all pass — UI behavior unchanged
- Pytest: 5 pre-existing failures in test_stems_to_midi / test_integration
  / ground_truth_e2e confirmed UNRELATED to this cleanup (reproduce
  on stash of this commit)

## Phase 5: Commit [✅]

- Two commits: root first, project 6 second. Plan/results files
  in agent-plans/.

## Final stats

| File | Before | After | Δ |
|------|--------|-------|---|
| `midiconfig.yaml` (root) | 302 lines | 263 lines | -39 / -13% |
| `user_files/6*/midiconfig.yaml` (project 6) | 246 lines | 116 lines | -130 / -53% |
| **Total** | **548** | **379** | **-169 / -31%** |
