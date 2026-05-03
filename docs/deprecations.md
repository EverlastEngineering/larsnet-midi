# Deprecated and Dead Settings

Settings that are no longer functional in the current codebase. These are candidates for removal from `midiconfig.yaml`.

## Explicitly Deprecated

### `audio.force_mono`
- **Status**: Deprecated (marked in YAML comments)
- **Replaced by**: Per-stem `use_stereo` setting
- **Behavior**: When `use_stereo` is not set for a stem, the code falls back to `not force_mono`. Setting `force_mono: true` effectively sets `use_stereo: false` as a global default.
- **Recommendation**: Remove. Set `use_stereo` explicitly per stem instead.

## Dead Config — Never Read by Processing Code

These settings exist in `midiconfig.yaml` but are not consumed by any processing pipeline code.

### `onset_merge_window_ms` (all stems)
- **In YAML**: Defined under each stem section (kick, snare, toms, hihat, cymbals)
- **In code**: Only referenced in `webui/settings_schema.py` schema definition. Not read by `processing_shell.py` or any detection module.
- **Confused with**: `merge_window_ms` under `energy_detection` (which IS used for stereo L/R peak merging)
- **Recommendation**: Remove from YAML and WebUI schema, or implement the intended behavior.

### `hihat.enable_amplitude_refinement`
- **In YAML**: `false`
- **In code**: Zero references in any `.py` file.
- **Recommendation**: Remove.

### `hihat.decay_threshold`
- **In YAML**: `0.65`
- **In code**: Not read. Open/closed classification now uses `open_sustain_ms` + `open_geomean_min`.
- **Recommendation**: Remove. Document that open/closed detection uses `open_sustain_ms` and `open_geomean_min`.

### `threshold_optimization.initial_threshold_step`
- **In YAML**: `0.05`
- **In code**: `optimization_core.optimize_threshold_by_clustering()` uses `threshold_step_initial=0.1` as a function parameter default. The YAML value is never loaded.
- **Recommendation**: Remove from YAML, or wire the config value into the function call.

### `threshold_optimization.convergence_patience`
- **In YAML**: `3`
- **In code**: Passed as a function parameter with default `3`. The YAML value is never loaded into the function.
- **Recommendation**: Remove from YAML, or wire the config value into the function call.

### `clustering.features`
- **In YAML**: List of feature names
- **In code**: `optimization_core.py` builds its own internal feature list. The config value is not read.
- **Recommendation**: Remove from YAML, or refactor `optimization_core.py` to read from config.

### `learning_mode.export_all_detections`
- **In YAML**: `true`
- **In code**: Learning mode always exports all detections. The flag is not checked.
- **Recommendation**: Remove.

### `learning_mode.rejected_velocity`
- **In YAML**: `1`
- **In code**: Not read. Rejected events in learning mode are excluded from MIDI output entirely rather than being written with a low velocity.
- **Recommendation**: Remove.

### `learning_mode.kept_velocity_normal`
- **In YAML**: `true`
- **In code**: Not read.
- **Recommendation**: Remove.

### `learning_mode.calibrated_config_output`
- **In YAML**: `"midiconfig_calibrated.yaml"`
- **In code**: `learning.py` has its own `save_calibrated_config()` that constructs output paths independently.
- **Recommendation**: Remove, or refactor `learning.py` to read this value.

## Legacy Section — Still Functional but Superseded

### `onset_detection` section
- **Status**: Active but legacy
- **Superseded by**: `energy_detection` section (default method)
- **Still used when**: A stem sets `use_librosa_detection: true`
- **Recommendation**: Keep for backward compatibility. Mark as legacy in comments.
