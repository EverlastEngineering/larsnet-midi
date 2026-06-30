# midiconfig.yaml cleanup — plan

## Goal

Bring the root `midiconfig.yaml` (used as the default for new projects)
back in sync with the live pipeline, and strip project 6's working
config down to active keys only.

## Scope

Two files:

1. `/Users/jasoncopp/Source/GitHub/larsnet/midiconfig.yaml` (root)
2. `/Users/jasoncopp/Source/GitHub/larsnet/user_files/6 - 01_Taylor_Swift_The_Fate_of_Ophelia_Drums/midiconfig.yaml` (project 6)

## Approach

### Root: full schema, large-bypass defaults, deprecated keys commented out

Every ACTIVE key the pipeline reads gets a sensible default value.
DEPRECATED/DORMANT keys (still wired but bypassed, or kept for compat)
get a commented `# key: default-value` line so the operator can see
the surface area without being surprised by legacy keys.

The two still-wired-but-deprecated filters,
`attack_rise_max_ms` (real toms < 20ms; FPs 100-500ms) and
`min_decay_col_min_db` (real toms -60 to -84 dB; FPs -84 to -90 dB),
default to large-bypass values so the filter is effectively OFF by
default. The user can dial the threshold down when they want the
filter active. Project 6 already shows the pattern: it sets these to
11500 / -160.0 to disable per-stem.

- root `attack_rise_max_ms`: **15000** (effectively off)
- root `min_decay_col_min_db`: **-160.0** (effectively off)

Per-stem overrides that want the filter active drop the threshold to
the empirical cluster boundary (e.g. toms.attack_rise_max_ms: 20.0).

### Root: drop dead sections that PGA-universal cleanup removed

The following sections were removed in the 2026-06-20 PGA-universal
cleanup but still linger in the root config with stale keys:

- `filtering.*` (6 dead keys)
- `clustering.method` (1)
- `threshold_optimization.*` (3)
- `debug.*` (2)
- `learning_mode.*` (5)

These are dropped from the root config. The big "removed in
PGA-universal cleanup" comment block at the end of the current root
config stays (single source of truth for that history).

### Project 6: strip dead keys per stem

Per-stem PGA-cleanup-removed keys (14 keys × 5 stems = 70 keys):

`expected_clusters`, `threshold_db`, `min_peak_spacing_ms`,
`min_absolute_energy`, `merge_window_ms`, `energy_method`,
`peak_hold_ms`, `onset_threshold`, `onset_delta`, `onset_wait`,
`min_strength_threshold`, `min_sustain_ms`, `enable_spectral_filter`,
`reverb_continuation_attack_threshold`.

Plus: `filtering.*`, `clustering.method`, `threshold_optimization.*`,
`debug.*`, `learning_mode.*` (full sections).

Plus project-specific dead keys:

- `kick.fundamental_freq_*`, `kick.body_freq_*`, `kick.attack_freq_*`
- `snare.low_freq_*`, `snare.body_freq_*`, `snare.wire_freq_*`
- `toms.fundamental_freq_*`, `toms.body_freq_*`
- `cymbals.body_freq_*`, `cymbals.brilliance_freq_*`,
  `cymbals.sustain_analysis_window_sec`, `cymbals.decay_filter_window_sec`

The spectral band keys (fundamental/body/attack/wire/low/brilliance)
are returned to a single-commented reference block so the surface is
documented but they don't bloat each project config.

### Project 6: keep per-stem PGA-tuning overrides

These are the live tuning knobs for project 6. Do NOT touch:

- kick: `pga_broad_freq_min_hz`, `pga_broad_freq_max_hz`,
  `pga_nms_min_frames`, `pga_db_rise_threshold`,
  `pga_strike_offset_sec`, `pga_max_floor_gate_db`,
  `pga_min_prominence` (5800), `pga_min_envelope_value` (9300),
  `pga_min_combined_score` (-41839)
- snare: `pga_min_prominence` (2000), `pga_broad_freq_*`,
  `pga_nms_min_frames`, `pga_db_rise_threshold`,
  `pga_max_floor_gate_db`, `pga_min_envelope_value` (3100),
  `pga_min_combined_score` (-738577),
  enable_pitch_detection / pitch_method / min/max_pitch_hz,
  `attack_rise_max_ms: 500`, `min_decay_col_min_db: -120`
  (these are tighter than the root bypass defaults — project 6
  wants the filter active for snare)
- toms: `pga_nms_min_frames`, `pga_strike_offset_sec`,
  `pga_max_floor_gate_db`, `pga_min_envelope_value` (10000),
  `pga_min_prominence` (1400), `min_decay_col_min_db: -195`,
  `attack_rise_max_ms: 500`, `pga_min_combined_score` (-37589),
  enable_pitch_detection / pitch_method / min/max_pitch_hz
- hihat: pga_* knobs, `open_decay_slope_max: 0.7`,
  `openness_score_threshold: 0.75`, `pga_min_combined_score`,
  `attack_rise_max_ms: 10000` (extreme — bypass),
  `min_decay_col_min_db: -195` (extreme — bypass),
  `classifier_method: 'kmeans'` is dropped from the project
  (the report says it's dormant, never read — better to put it
  in a commented block in the root for visibility)
- cymbals: pga_* knobs (`pga_detection_method: delta`),
  `pga_min_combined_score`, `attack_rise_max_ms: 500`,
  `min_decay_col_min_db: -195`

The user-facing open/closed hihat tuning knobs:
`open_decay_slope_max: 0.7` and `openness_score_threshold: 0.75` stay
on project 6 (these ARE live).

### Notes on hihat.classifier_method

Per the report `hihat.classifier_method` is dormant (never read by
runtime). It's a TEST-only flag that doesn't actually drive anything.
We drop it from project 6 and leave a `# hihat.classifier_method:
'slope'` commented line in the root config under a "Dormant /
research-only" block so it's discoverable.

## Risks

1. **Existing projects break if their project midiconfig.yaml relied
   on dead keys for documented behavior.** Acceptable — the dead-key
   removal is what the user wants. Recording this in the commit
   message.
2. **Root defaults diverge from project 6's per-stem overrides.**
   Intentional. The root defaults are "filter off, full data" — that
   matches what a brand-new project would want before any tuning.
3. **Snare's tight `attack_rise_max_ms: 500` / `min_decay_col_min_db:
   -120` are not bypass values** — they ARE active filters. Keep on
   project 6.

## Success criteria

- [ ] `python -c "from stems_to_midi.config import load_config; ..."` loads both files cleanly
- [ ] `pytest tests/` runs without new failures
- [ ] root `midiconfig.yaml` has every active key, large-bypass defaults for the two deprecated filters, and no dead sections
- [ ] project 6 `midiconfig.yaml` is down to < 60 lines of active config (from 246)
- [ ] Two separate commits: one for root, one for project 6

## Plan files

- plan: this file
- results: `./midiconfig-cleanup.results.md`
