# Results: Step 3 — Dual-Sensitivity Detection Run

## Phase Completion

- [x] Phase A: Sensitive detection run in processing_shell.py
- [x] Phase B: Plumbing — return dict & CLI
- [x] Phase C: Sidecar format v3
- [x] Phase D: Tests
- [x] Final: All tests pass, committed

## Decision Log

- Extracted `_serialize_onset_events()` helper from `save_analysis_sidecar()` to avoid duplicating serialization logic for configured vs sensitive events.
- Sensitive detection runs only for energy-based detection (skipped when `use_librosa_detection: true`).
- Sidecar key renamed from `events` to `events_configured` (no existing consumers read the old key).
- `_run_sensitive_detection()` added as a private helper in processing_shell.py to keep `process_stem_to_midi()` readable.

## Metrics

- Tests passing: 716 (701 baseline + 15 new)
- Pre-existing failures: 5 (2 integration synthetic audio, 3 webui config API)
- New tests added: 15
  - 5 `TestSerializeOnsetEvents` (serialization helper)
  - 5 `TestSidecarV3Format` (v3 JSON structure)
  - 5 `TestRunSensitiveDetection` (sensitive detection behavior)
