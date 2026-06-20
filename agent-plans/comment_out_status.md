# Comment-Out Status (PGA Cleanup 2026-06-20)

This file tracks every block of code that has been wrapped in
`# === CLEANUP-START: <reason> ===` / `# === CLEANUP-END ===` markers
during the comment-out pass. After verification, Phase 7 will
hard-delete each entry and update its row.

| Status | File | Lines | Reason | Marker commit | Delete commit |
|---|---|---|---|---|---|
| PENDING | stems_to_midi/processing_shell.py | 57-157 | `_load_and_validate_audio` — only called from unreachable tail | | |
| PENDING | stems_to_midi/processing_shell.py | 159-212 | `_configure_onset_detection` — only called from unreachable tail | | |
| PENDING | stems_to_midi/processing_shell.py | 214-292 | `_detect_tom_pitches` — never called | | |
| PENDING | stems_to_midi/processing_shell.py | 294-404 | `_detect_cymbal_pitches` — never called | | |
| PENDING | stems_to_midi/processing_shell.py | 406-484 | `_detect_snare_pitches` — never called | | |
| PENDING | stems_to_midi/processing_shell.py | 486-632 | `_create_midi_events` — only called from unreachable tail | | |
| PENDING | stems_to_midi/processing_shell.py | 634-727 | `_run_sensitive_detection` — only called from unreachable tail | | |
| PENDING | stems_to_midi/processing_shell.py | 729-780 | `build_spectral_config_for_stem` — only called from unreachable tail | | |
| PENDING | stems_to_midi/processing_shell.py | 782-902 | `_run_spectral_detection` — only called from unreachable tail | | |
| PENDING | stems_to_midi/processing_shell.py | 906-1111 | `_build_events_configured` — only called from unreachable tail | | |
| PENDING | stems_to_midi/processing_shell.py | 1186-1899 | unreachable tail of `process_stem_to_midi` after PGA return | | |
| PENDING | stems_to_midi/stereo_core.py | 285-380 | `detect_stereo_onsets` — librosa-based, not in main pipeline | | |
| PENDING | stems_to_midi/stereo_core.py | 392-490 | `detect_dual_channel_onsets` — librosa-based, not in main pipeline | | |
| PENDING | stems_to_midi/energy_detection_core.py | TBD | `method='spectral'` branch inside `detect_onsets_energy_based` | | |
| PENDING | stems_to_midi/spectral_transient_core.py | TBD | `SpectralTransientConfig`, `detect_spectral_transients`, `_detect_spectral_transients_impl` (keep `compute_stft_db`) | | |
| PENDING | stems_to_midi/detection_shell.py | TBD | `detect_onsets` (librosa fallback) and the 4 dead detector helpers | | |

## Phase 0.5 — Pre-cleanup test purges (2026-06-20)

The previous `test-cleanup.results.md` reported 1395 pass / 0 fail, but the
baseline has drifted. Before the comment-out pass can run, 13 currently-failing
tests must be removed AND the WebUI features they depend on must be purged
(orphaned tests with dead code dependencies are pure noise).

| Status | File | Action | Reason |
|---|---|---|---|
| DONE | `webui/test_reclassify_api.py` | DELETE | 7 tests reference `hihat_open_geomean_min` / `hihat_open_sustain_ms` schema entries that were removed 2026-06-19 (comment at `settings_schema.py:970`) | (Phase 0.5 commit) | n/a (deleted) |
| DONE | `webui/api/operations.py::reclassify` | REMOVE `open_geomean_min` / `open_sustain_ms` config_override handling | Dead config keys; reclassify endpoint continues to work via the live `open_decay_slope_max` slider | (Phase 0.5 commit) | n/a (edited) |
| DONE | `webui/test_spectral_overlay.py` | DELETE | 6 tests depend on `method` field and `events_spectral` sidecar key, both removed in Phase 1D | (Phase 0.5 commit) | n/a (deleted) |
| DONE | `webui/static/js/waveform.js` | REMOVE spectral overlay feature | `markerSpectral`, `spectralOverlayActive`, `getEventColor`'s method branch — no source data after Phase 1D | (Phase 0.5 commit) | n/a (edited) |

## Notes

- A subagent MUST append a row to this file (or update the PENDING
  row to ACTIVE) when it adds a CLEANUP marker.
- A subagent MUST update the Delete commit column in Phase 7 when
  the marked block is hard-deleted.
- Use `scripts/tools/line_range_ops.py comment <file> <start> <end>
  --reason "..."` to add the markers; this preserves indentation.
