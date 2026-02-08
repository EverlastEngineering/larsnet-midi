# midiconfig.yaml Improvement Suggestions

Actionable recommendations from the settings audit. Grouped by priority and type.

## High Priority — Default Mismatches

These cases have YAML values that differ from the code's fallback default. While the YAML value wins at runtime (because the config is loaded), the mismatches indicate drift between code and config that could cause surprises if a key is ever removed or renamed.

| # | Setting | YAML Value | Code Fallback | Suggested Fix |
|---|---|---|---|---|
| 1 | `filtering.reverb_continuation_attack_threshold` | `0.4` | `0.2` | Align code fallback to `0.4` |
| 2 | `kick.statistical_badness_threshold` | `0.3` | `0.6` | Align code fallback to `0.3` |
| 3 | `hihat.detect_open` | `true` | `False` | Align code fallback to `True` |
| 4 | `hihat.open_sustain_ms` | `100` | `150` | Align code fallback to `100` |
| 5 | `snare.enable_pitch_detection` | `false` | `True` | Align code fallback to `False` |

**Why this matters:** If someone runs with a minimal config (missing these keys), they get different behavior than the documented defaults.

## ~~High Priority — analysis.json Naming~~ ✅ RESOLVED

Domain-specific field names are now used end-to-end. The analysis.json sidecar writes `fundamental_energy`, `body_energy`, `wire_energy`, `sizzle_energy`, `brilliance_energy`, `attack_energy` directly, with a `freq_bands` metadata block per stem mapping each field to its Hz range. No generic `primary_energy`/`secondary_energy`/`tertiary_energy` fields remain.

## Medium Priority — Dead Config Cleanup

Remove 11 dead config keys (see [deprecations.md](deprecations.md) for full list):

1. `onset_merge_window_ms` from all 5 stem sections
2. `hihat.enable_amplitude_refinement`
3. `hihat.decay_threshold`
4. `threshold_optimization.initial_threshold_step`
5. `threshold_optimization.convergence_patience`
6. `clustering.features`
7. `learning_mode.export_all_detections`
8. `learning_mode.rejected_velocity`
9. `learning_mode.kept_velocity_normal`
10. `learning_mode.calibrated_config_output`

**Approach:** Remove in a single commit with test verification to confirm no behavioral change.

## Medium Priority — Missing Config Keys

These settings are read by code with hardcoded defaults but have no corresponding YAML entry. Adding them makes the config complete and discoverable.

| # | Setting | Code Default | Where Read | Suggested YAML Location |
|---|---|---|---|---|
| 1 | `timing_offset` | `0.0` (seconds) | `processing_shell.py` per-stem | Per-stem section |
| 2 | `enable_decay_filter` | `True` | `processing_shell.py` for cymbals | `cymbals` section |
| 3 | `cymbals.enable_pitch_detection` | `True` | `detection_shell.py` | `cymbals` section |
| 4 | `cymbals.min_pitch_hz` | `200.0` | `detection_shell.py` | `cymbals` section |
| 5 | `cymbals.max_pitch_hz` | `1000.0` | `detection_shell.py` | `cymbals` section |

## Low Priority — Naming Consistency

Minor naming inconsistencies between config and analysis.json output:

| Config Key | Output Key | Suggestion |
|---|---|---|
| `enable_statistical_filter` | `statistical_enabled` | Standardize to `enable_*` pattern |
| `decay_filter_window_sec` | `decay_window_sec` | Keep config name, add `_filter_` to output |

## Low Priority — Structural Improvements

1. **Centralize detection method config:** Currently `use_librosa_detection` is defined per-stem and also has a global default. Consider a top-level `detection_method: 'energy'` field with per-stem overrides.

2. **Add `learning_mode.match_tolerance_sec` to YAML:** This setting is read by `learning.py` but only documented in code. Default is `0.05` seconds.

3. **WebUI schema sync:** The WebUI `settings_schema.py` defines `onset_merge_window_ms` which is dead config. Sync the schema with actual processing code.

4. **Deprecate `audio.force_mono`:** Add a YAML comment marking it deprecated and pointing to per-stem `use_stereo`.
