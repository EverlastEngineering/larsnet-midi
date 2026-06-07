# Deprecated and Dead Settings

Settings that are no longer functional in the current codebase.

## Explicitly Deprecated

### `audio.force_mono`
- **Status**: Deprecated (marked in YAML comments)
- **Replaced by**: Per-stem `use_stereo` setting
- **Behavior**: When `use_stereo` is not set for a stem, the code falls back to `not force_mono`. Setting `force_mono: true` effectively sets `use_stereo: false` as a global default.
- **Recommendation**: Remove. Set `use_stereo` explicitly per stem instead.

## Removed in 2026-06 (drift-fix)

The following keys were removed from `midiconfig.yaml` and
`webui/settings_schema.py` because no production code ever read them.
Removal log entry: T1 drift-fix (settings_schema as single source of
truth).

| Key | YAML | Schema | Notes |
| --- | --- | --- | --- |
| `kick.onset_merge_window_ms` | removed | removed | confused with `merge_window_ms` |
| `snare.onset_merge_window_ms` | removed | removed | confused with `merge_window_ms` |
| `toms.onset_merge_window_ms` | removed | removed | confused with `merge_window_ms` |
| `hihat.onset_merge_window_ms` | removed | removed | confused with `merge_window_ms` |
| `cymbals.onset_merge_window_ms` | removed | removed | confused with `merge_window_ms` |
| `energy_detection.onset_merge_window_ms` | removed | n/a | not consumed by `processing_shell.py` |
| `hihat.enable_amplitude_refinement` | removed | n/a | zero references in any `.py` file |
| `hihat.decay_threshold` | removed | n/a | superseded by `open_sustain_ms` + `open_geomean_min` |
| `threshold_optimization.initial_threshold_step` | removed | n/a | `optimization_core` uses its own `threshold_step_initial=0.1` |
| `threshold_optimization.convergence_patience` | removed | n/a | `optimization_core` uses its own default `3` |
| `clustering.features` | removed | n/a | `optimization_core` builds its own feature list |
| `learning_mode.export_all_detections` | removed | n/a | always true; flag was never checked |
| `learning_mode.rejected_velocity` | removed | n/a | rejected events are excluded from MIDI entirely |
| `learning_mode.kept_velocity_normal` | removed | n/a | not read |
| `learning_mode.calibrated_config_output` | removed | n/a | `learning.py` has its own path logic |

## Legacy Section — Still Functional but Superseded

### `onset_detection` section
- **Status**: Active but legacy
- **Superseded by**: `energy_detection` section (default method)
- **Still used when**: A stem sets `use_librosa_detection: true`
- **Recommendation**: Keep for backward compatibility. Mark as legacy in comments.
