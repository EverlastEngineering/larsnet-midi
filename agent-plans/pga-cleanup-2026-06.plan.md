# PGA Cleanup Plan (2026-06-20)

Mirror of the session plan. Edit the session plan; treat this file as
the authoritative copy for subagent handoffs.

## TL;DR

PGA is now the only path for all 5 stems, but the legacy energy /
peak_hold / librosa / spectral / geomean code is still shipped, the
default `midiconfig.yaml` still has `use_pga_detection: false`, the
WebUI still exposes dead sliders, and there are 30+ stale
agent-plans markdown files from the legacy era. This plan removes
the dead surface in two passes — comment-out everything (no
behavioral change) then hard-delete it — so each handoff is a
verifiable no-op and easy to revert.

Ground-truth AIFF in `tests/assets/` is registered as a real
project and exercised by a new e2e test to prove the post-cleanup
pipeline still works.

## Phases

0. Setup & tooling (this plan file, results file, status file, line-range script, baseline pytest)
1. Comment-out dead code (1A-1E)
2. Midiconfig cleanup
3. Settings schema cleanup
4. WebUI cleanup
5. Sidecar cleanup
6. New ground-truth project + e2e test
7. Hard-delete commented blocks + dead scripts + dead CSVs
8. Documentation cleanup (delete superseded, update live in place)
9. Verification gates

See `/memories/session/plan.md` for the full step-by-step plan with
line ranges, file lists, and risk notes.
