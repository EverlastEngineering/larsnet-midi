# Plan: Wire Post-Filter Note Classification + Spread Guardrail

## Problem

The note classification system (`classify_tom_notes`, `classify_snare_notes`,
`classify_cymbal_notes`, `classify_hihat_notes`, and the dispatch
`classify_notes`) is implemented and unit-tested, but is NOT wired into the
live PGA pipeline. Sidecar evidence (project 6, 2026-06-30):

```
tom events_pga count: 2246
kept: 10 / 2246
classification counts: Counter({None: 10})   ← classify_tom_notes never ran
note counts:        Counter({None: 10})     ← note was never stamped
pitch_hz stats:     count=10, min=66.4, max=92.3, mean=81.0   ← data IS there
```

The bug has two halves:

1. **Live PGA path** (`processing_shell_percentile_gated.py:215-217`) only
   calls `classify_hihat_notes` for hihat stems. For toms/snare/cymbals, the
   MIDI loop hardcodes `note = drum_mapping.<stem_type>` (e.g. every tom = 47).

2. **Rebuild path** (`rebuild_core.py:409-414`) calls `classify_notes` which
   stamps `event['note']` correctly, but the MIDI loop reads
   `ev.get('hihat_state')` only and otherwise uses `ev_note = stem_note`,
   throwing away the per-event classification.

## Secondary Problem (Spread)

`classify_tom_notes` runs k-means with `expected_clusters` (default 2 or 3)
when `n_unique >= k`. If 10 pitches are 66.1–66.9 Hz (9 unique values close
together), k-means will dutifully split them into 2 clusters even though
they're semantically one tom. There is no spread guardrail — only an
"all identical" early-return when `n_unique == 1`.

## Approach

### Phase 1 — Wire classification through

**1a. Live path** — replace the hihat-only special case with a generic
`classify_notes` call covering hihat/toms/snare/cymbals. Change the MIDI
loop to read `ev.get('note') ?? stem_note` instead of the special-case
`note_open` / `note` flip.

**1b. Rebuild path** — change `ev_note = stem_note` (with `hihat_state`
override) to `ev_note = ev.get('note') or stem_note`. The classification
is already correctly stamped on `ev['note']` by the `classify_notes` call
50 lines above; we just have to consume it.

**1c. End-to-end** — re-run on project 6; expect toms sidecar to show
multiple `classification` values and the MIDI to include 45, 47, or 50.

### Phase 2 — Spread guardrail (IQR)

Add a new helper `_has_sufficient_spread(values, threshold)` that returns
True iff IQR(75th − 25th percentile) of the input array ≥ threshold.

Modify `classify_tom_notes` / `classify_snare_notes` / `classify_cymbal_notes`
to call the guardrail before invoking `_cluster_values`. When guardrail
fails, assign `classification=1` (mid) to all KEPT events and return
without clustering.

Add three per-stem config keys with sensible defaults:
- `toms.min_pitch_spread_hz` (default `5.0`, unit Hz) — IQR threshold
  for `pitch_hz`. 5 Hz ≈ a quarter-tone; two real toms in a kit are at
  least a minor third apart (~50 Hz), so 5 Hz comfortably catches
  "all close" false splits without ever rejecting real splits.
- `snare.min_stereo_width_spread` (default `0.05`, unit ratio) — IQR
  threshold for `stereo_width`. Mono snare hits cluster around 0.02–0.04;
  a layered clap is 0.3+. 0.05 separates the two cleanly.
- `cymbals.min_centroid_spread_hz` (default `500.0`, unit Hz) — IQR
  threshold for `spectral_centroid_hz`. Crash ≈ 4–5 kHz, ride ≈ 6–7 kHz,
  chinese ≈ 8–10 kHz. 500 Hz IQR is conservative.

Schema entries go in `webui/settings_schema.py` per the single-source-of-truth
pattern documented in AGENTS.md.

### Phase 3 — Outlier merge (`expected_clusters + 1`) — DEFERRED

The user's alternative idea of running KMeans with K+1 and merging the
smallest cluster back into its nearest neighbor is deferred until Phase 1+2
results are validated. The spread guardrail alone may be sufficient; if not,
this is a focused follow-up.

### Phase 4 — Tests

- **Unit (functional core, no I/O)**:
  - `_has_sufficient_spread` with flat, close-but-distinct, and well-spread
    fixtures.
  - `classify_tom_notes` with 10 events at 66.1–66.9 Hz and `expected_clusters=2`:
    all events must end up at the same classification (spread guardrail triggers).
  - `classify_tom_notes` with 5 events clearly clustered at 70 Hz and 5 at 120 Hz
    and `expected_clusters=2`: must split into 2 groups.
- **E2E (project 6)**:
  - Run `python stems_to_midi_cli.py 6`.
  - Read sidecar JSON; assert `stems.toms.events_pga[*].classification` is no
    longer `None` for KEPT events; assert `stems.toms.events_pga[*].note` is
    one of {45, 47, 50}.
  - Inspect generated MIDI; assert it contains notes 45/47/50 (not only 47).

### Files Changed

1. `stems_to_midi/processing_shell_percentile_gated.py` — Phase 1a
   (replace hihat-only block; change MIDI loop)
2. `stems_to_midi/rebuild_core.py` — Phase 1b (use ev['note'])
3. `stems_to_midi/note_classification_core.py` — Phase 2 (add
   `_has_sufficient_spread` helper; gate `classify_tom_notes`,
   `classify_snare_notes`, `classify_cymbal_notes`)
4. `webui/settings_schema.py` — three new `SettingDefinition` entries
5. `test_note_classification_core.py` — new test class
   `TestSpreadGuardrail` + 2 fixtures for tom classification
6. (optional E2E test script) `scripts/test_classification_e2e.py`

### Risks

- The behavior change for non-hihat stems means: any project that was
  depending on "all events get one MIDI note" will now get per-event
  classification. For most projects this is the desired fix, but anyone
  who deliberately relied on the bug (e.g. configured `expected_clusters=1`
  for everything) will see no change. For projects with `expected_clusters > 1`
  (e.g. toms=2, snare=2) the change is observable in the MIDI.
- Spread thresholds are calibrated from intuition + a single project
  (project 6 Taylor Swift). Worst case is the threshold is too lax and a
  real-but-narrow split gets merged. Worst-case symptom: all toms get one
  note. Worst-case fix: tune the threshold. No data corruption risk.

### Success Criteria

1. Re-run `python stems_to_midi_cli.py 6` on project 6. Sidecar toms
   `classification` field is non-None for all KEPT events; `note` field
   is one of {45, 47, 50}.
2. Generated MIDI for project 6 toms contains ≥ 2 distinct pitch classes.
3. New tests pass; full pytest run shows no regressions beyond the 4
   pre-existing cymbals failures.
4. Spread guardrail test fixture (10 values at 66.1–66.9 Hz) returns a
   single classification; 5+5 at 70/120 returns 2 classifications.