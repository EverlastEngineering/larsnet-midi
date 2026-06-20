# PGA Cleanup Results (2026-06-20)

## Phase Status

- [ ] Phase 0 — Setup & tooling
- [ ] Phase 0.5 — Purge the 13 currently-failing tests + their WebUI features
  - [ ] 0.5a Delete `webui/test_reclassify_api.py` (7 tests reference removed `open_geomean_min` / `open_sustain_ms` schema entries; the comment at `settings_schema.py:970` documents these were removed 2026-06-19)
  - [ ] 0.5b Remove `open_geomean_min` / `open_sustain_ms` config_override paths from `webui/api/operations.py::reclassify`
  - [ ] 0.5c Delete `webui/test_spectral_overlay.py` (6 tests depend on the `method` field and `events_spectral` sidecar key, neither of which survives the spectral purge in Phase 1D)
  - [ ] 0.5d Remove the spectral overlay from `webui/static/js/waveform.js` (`markerSpectral`, `spectralOverlayActive`, `getEventColor`'s method branch)
- [ ] Phase 1 — Comment-out pass
  - [ ] 1A processing_shell.py
  - [ ] 1B stereo_core.py
  - [ ] 1C energy_detection_core.py
  - [ ] 1D spectral_transient_core.py
  - [ ] 1E detection_shell.py
- [ ] Phase 2 — Midiconfig cleanup
- [ ] Phase 3 — Settings schema cleanup
- [ ] Phase 4 — WebUI cleanup
- [ ] Phase 5 — Sidecar cleanup
- [ ] Phase 6 — Ground-truth project + e2e test
- [ ] Phase 7 — Hard-delete pass
- [ ] Phase 8 — Documentation cleanup
- [ ] Phase 9 — Final verification

## Metrics

- Baseline pytest: 1382 passed, 13 failed, 10 skipped, 20 deselected (drift from yesterday's reported 1395 — see Phase 0.5)
- Post-Phase-0.5 pytest actual: 1342 passed, 0 failed, 10 skipped, 20 deselected (deleted files had 53 test functions total, 13 of which were the failing ones — others were passing/skipped; net diff -40 from the 1382 baseline matches expectation after accounting for skipped/parameterized variants)
- Files commented: TBD
- Files hard-deleted: TBD
- Net lines removed: TBD
- Tests added: TBD
- Final pytest: TBD
