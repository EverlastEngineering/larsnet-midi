# midiconfig.yaml Setting Tracing Report

Complete tracing of how each midiconfig.yaml setting is consumed in code.

---

## 1. `hop_length` (onset_detection section)

**YAML default:** `512`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [processing_shell.py](../stems_to_midi/processing_shell.py#L193) | 193 | `onset_config['hop_length']` | Returned in onset detection params dict; passed to `detect_onsets()` and `detect_onsets_energy_based()` as `hop_length=` | No fallback (direct key access) |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L172) | 172 | `onset_config['hop_length']` | Same key used in learning mode branch | No fallback |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L752) | 752 | `hop_length=onset_params['hop_length']` | Passed to `detect_onsets_energy_based()` for frame stepping | — |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L707) | 707 | `hop_length=hop_length` | Passed to legacy `detect_onsets()` (librosa path) | — |
| [energy_detection_shell.py](../stems_to_midi/energy_detection_shell.py#L22) | 22 | `hop_length: int = 512` | Function parameter default for `detect_onsets_energy_based()` | `512` ✅ match |
| [energy_detection_core.py](../stems_to_midi/energy_detection_core.py#L78) | 78 | `hop_length: int = 512` | Function parameter for `compute_energy_envelope()` | `512` ✅ match |
| [stems_to_midi_cli.py](../stems_to_midi_cli.py#L190) | 190 | `hop_length = config['onset_detection']['hop_length']` | CLI reads from config as fallback when CLI arg is None | No fallback |
| [learning.py](../stems_to_midi/learning.py#L123) | 123 | hardcoded `hop_length=512` | Used in `librosa.onset.onset_strength()` — does NOT read from config | hardcoded `512` |
| [test_integration.py](../test_integration.py#L295) | 295+ | `hop_length = onset_params.get('hop_length', 512)` | Tests read from config with fallback | `512` |

---

## 2. `onset_merge_window_ms` (per-stem and energy_detection sections)

**YAML defaults:** energy_detection global: `100`, kick: `100`, snare: `100`, toms: `100`, hihat: `20`, cymbals: `100`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| **⚠️ NOT read by any production code** | — | — | This setting exists in YAML per-stem configs and the energy_detection global section, but NO production `.py` file reads `onset_merge_window_ms` from config | — |
| [settings_schema.py](../webui/settings_schema.py#L666) | 666-785 | `yaml_path=['kick', 'onset_merge_window_ms']` etc. | WebUI schema defines paths for all 5 stems, used for settings display/edit only | — |

**⚠️ DEAD CONFIG**: `onset_merge_window_ms` is defined in YAML for every stem and in `energy_detection` but is never read by any processing code. The similar-sounding `merge_window_ms` (for stereo L/R merging) IS used at [processing_shell.py#L739](../stems_to_midi/processing_shell.py#L739).

---

## 3. `reverb_continuation_attack_threshold` (filtering section)

**YAML default:** `0.4`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1805) | 1805 | `attack_threshold` | Read via `config.get('filtering', {}).get('reverb_continuation_attack_threshold', 0.2)`. Passed to `mark_reverb_continuations()` as `attack_sharpness_threshold`. Events with attack sharpness below this are marked as `REVERB_CONTINUATION` and removed from MIDI output. | `0.2` ⚠️ **MISMATCH**: YAML=`0.4`, code default=`0.2` |

---

## 4. `expected_clusters` (per-stem sections)

**YAML defaults:** kick: `1`, snare: `1`, toms: `null`, hihat: `2`, cymbals: `2`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [optimization_core.py](../stems_to_midi/optimization_core.py#L188) | 188 | `expected_clusters: int` | Function parameter for `optimize_threshold_by_clustering()`. Controls target cluster count for adaptive threshold discovery. Used to determine kmeans `n_clusters`, binary search convergence target, and threshold adjustment direction. | Required param (no default) |
| [settings_schema.py](../webui/settings_schema.py#L681) | 681-800 | `yaml_path=['kick', 'expected_clusters']` etc. | WebUI schema for all stems | — |

**Note**: `expected_clusters` is read from config and passed to `optimize_threshold_by_clustering()` but the call site that reads it from per-stem YAML config is NOT in the main processing pipeline. The optimization core receives it as a parameter. The `threshold_optimization.enabled` flag (default `false`) gates whether this is used.

---

## 5. `threshold_optimization` (top-level section)

**YAML defaults:** `enabled: false`, `max_iterations: 20`, `tolerance: 0`, `initial_threshold_step: 0.05`, `convergence_patience: 3`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [settings_schema.py](../webui/settings_schema.py#L826) | 826-862 | `yaml_path=['threshold_optimization', 'enabled']` etc. | WebUI schema defines `enabled`, `max_iterations`, `tolerance` for display/edit | — |

**⚠️ DEAD CONFIG (partially)**: The `threshold_optimization` section is defined in YAML and exposed in WebUI settings schema, but NO production processing code reads `config['threshold_optimization']`. The `optimization_core.py` function `optimize_threshold_by_clustering()` accepts these as parameters but is never called from the main processing pipeline (`processing_shell.py` or `stems_to_midi_cli.py`). The settings `initial_threshold_step` and `convergence_patience` have no YAML-to-code reader at all.

---

## 6. `enable_amplitude_refinement` (hihat section)

**YAML default:** `false` (under hihat)

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| **⚠️ NOT read by any code** | — | — | — | — |

**⚠️ DEAD CONFIG**: `enable_amplitude_refinement` exists in YAML under `hihat:` but is NEVER read by any `.py` file. No `grep` matches in any production or test code.

---

## 7. `timing_offset` (per-stem sections)

**YAML:** Not present in default `midiconfig.yaml` for any stem (but documented as a per-stem override)

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [processing_shell.py](../stems_to_midi/processing_shell.py#L189) | 189 | `timing_offset` | `stem_config.get('timing_offset', 0.0)` — read from per-stem config, included in onset detection params | `0.0` |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L502) | 502 | `timing_offset` | `stem_config.get('timing_offset', 0.0)` — read again in `_create_midi_events()`, applied to MIDI event times via `midi_time = float(time) + timing_offset` | `0.0` |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L566) | 566 | `midi_time` | `midi_time = float(time) + timing_offset` — offsets MIDI timing to compensate for onset detection latency | — |
| [optimization/extract_features.py](../stems_to_midi/optimization/extract_features.py#L323) | 323 | `timing_offset` | `stem_config.get('timing_offset', 0.0)` — used to reverse-adjust MIDI times when comparing to CSV data | `0.0` |

**Note**: Not present as a key in any stem section of `midiconfig.yaml`, but code reads it with fallback `0.0`. Would be set in per-project config overrides.

---

## 8. `decay_threshold` (hihat section)

**YAML default:** `0.65` (under hihat)

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| **⚠️ NOT read by any production code** | — | — | — | — |

**⚠️ DEAD CONFIG**: `decay_threshold: 0.65` exists in YAML under `hihat:` but NO code reads `stem_config.get('decay_threshold', ...)`. The `energy_detection_core.py` has a parameter `decay_threshold_db` (in dB, default `-12.0`) which is a completely different concept (energy envelope decay, not hihat open/close detection). The hihat open/closed classification now uses `open_sustain_ms` + `open_geomean_min` instead.

---

## 9. `sustain_analysis_window_sec` (cymbals section)

**YAML default:** `2.0` (under cymbals)

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [analysis_core.py](../stems_to_midi/analysis_core.py#L2221) | 2221 | `sustain_analysis_window_sec` | `stem_config.get('sustain_analysis_window_sec')` — read from per-stem config. If None, falls back to `config.get('audio', {}).get('sustain_window_sec', 0.2)`. Converted to ms and passed to `calculate_sustain_duration()` as `window_ms=`. Controls how long to analyze audio after an onset to detect cymbal ring-out. | Fallback: `audio.sustain_window_sec` (0.2) |

---

## 10. `decay_filter_window_sec` (cymbals section)

**YAML default:** `0.5` (under cymbals)

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1627) | 1627 | `decay_filter_window_sec` | `cymbal_config.get('decay_filter_window_sec', 0.5)` — defines the lookback window for the cymbal decay retriggering filter. Onsets within this window of a previous hit's active decay are filtered as retriggering artifacts. | `0.5` ✅ match |
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1658) | 1658 | — | `if 0 < time_diff < decay_filter_window_sec:` — checks if current onset falls within decay window of a previous hit | — |
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1694) | 1694 | — | `window_sec=decay_filter_window_sec` — passed to `analyze_cymbal_decay_pattern()` | — |
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1708) | 1708 | — | Cleanup: `if current_time - t < decay_filter_window_sec` | — |
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1799) | 1799 | — | Stored in debug output dict | — |

---

## 11. `enable_decay_filter` (cymbals section)

**YAML:** Not present in `midiconfig.yaml` (but code reads it with default `True`)

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1621) | 1621 | `enable_decay_filter` | `cymbal_config.get('enable_decay_filter', True)` — if False, skips the entire cymbal decay retriggering filter (Pass 2). Allows disabling the filter without removing the code. | `True` |

**Note**: Not present in `midiconfig.yaml` — code default of `True` means decay filter is always on unless explicitly disabled in a per-project override.

---

## 12. `default_tempo` (midi section)

**YAML default:** `120.0`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [stems_to_midi_cli.py](../stems_to_midi_cli.py#L192) | 192 | `tempo` | `tempo = config['midi']['default_tempo']` — used as fallback when CLI `--tempo` arg is None. Sets the BPM for MIDI file creation. | No fallback (direct key access) |
| [optimization/extract_features.py](../stems_to_midi/optimization/extract_features.py#L290) | 290 | `tempo` | `config.get('midi', {}).get('default_tempo', 120.0)` — used for beat-time conversion in feature extraction | `120.0` ✅ match |

---

## 13. `max_note_duration` (midi section AND cymbals section)

**YAML defaults:** midi: `0.5`, cymbals: `2.0`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [processing_shell.py](../stems_to_midi/processing_shell.py#L553) | 553 | `cymbal_max` | `config.get(stem_type, {}).get('max_note_duration', 2.0)` — for cymbals, uses per-stem max allowing long ring-out | `2.0` ✅ match |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L558) | 558 | `max_duration` | `config.get('midi', {}).get('max_note_duration', 0.5)` — for non-cymbal stems, caps note duration at this value | `0.5` ✅ match |

---

## 14. `enable_pitch_detection` (snare and toms sections)

**YAML defaults:** snare: `false`, toms: `true`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [processing_shell.py](../stems_to_midi/processing_shell.py#L225) | 225 | `enable_pitch` | `tom_config.get('enable_pitch_detection', True)` — gates tom pitch detection for low/mid/high classification | `True` ⚠️ toms YAML=`true` ✅, code default=`True` |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L304) | 304 | `enable_pitch` | `cymbal_config.get('enable_pitch_detection', True)` — gates cymbal pitch detection for crash/ride/chinese classification | `True` ⚠️ cymbals: NOT in YAML, code default=`True` |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L408) | 408 | `enable_pitch` | `snare_config.get('enable_pitch_detection', True)` — gates snare pitch detection for snare/rimshot/clap classification | `True` ⚠️ **MISMATCH**: snare YAML=`false`, code default=`True` |

---

## 15. `enable_statistical_filter` (kick section)

**YAML default:** `false`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1732) | 1732 | `enable_statistical` | `stem_config.get('enable_statistical_filter', False)` — gates Pass 3 (statistical outlier detection) for kick only. If True and there are detected onsets, calculates badness scores and re-filters. | `False` ✅ match |

---

## 16. `statistical_badness_threshold` (kick section)

**YAML default:** `0.3`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1740) | 1740 | `badness_threshold` | `stem_config.get('statistical_badness_threshold', 0.6)` — onsets with badness score above this threshold are filtered out. Higher = more permissive. | `0.6` ⚠️ **MISMATCH**: YAML=`0.3`, code default=`0.6` |

---

## 17. `generate_foot_close` (hihat section)

**YAML default:** `false`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [processing_shell.py](../stems_to_midi/processing_shell.py#L505) | 505 | `generate_foot_close` | `stem_config.get('generate_foot_close', False)` — if True, generates MIDI note 44 (foot close) events at the end of open hihat sustain. Checked at [L598](../stems_to_midi/processing_shell.py#L598): `generate_foot_close and sustain_durations is not None and ...` | `False` ✅ match |

---

## 18. `detect_open` (hihat section)

**YAML default:** `true`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [stems_to_midi_cli.py](../stems_to_midi_cli.py#L248) | 248 | `hihat_detect` | `config.get('hihat', {}).get('detect_open', False)` — used as fallback when CLI `--detect-hihat-open` flag is not set. Controls whether hihat open/closed classification runs. | `False` ⚠️ **MISMATCH**: YAML=`true`, code default=`False` |

---

## 19. `open_sustain_ms` and `open_geomean_min` (hihat section)

**YAML defaults:** `open_sustain_ms: 100`, `open_geomean_min: 262.0`

### open_sustain_ms

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [processing_shell.py](../stems_to_midi/processing_shell.py#L859) | 859 | `open_sustain_ms` | `stem_config.get('open_sustain_ms', 150)` — displayed in debug output | `150` ⚠️ **MISMATCH**: YAML=`100`, code default=`150` |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L1012) | 1012 | `open_sustain_threshold` | `hihat_config.get('open_sustain_ms', 150)` — passed to `detect_hihat_state()` as `open_sustain_threshold_ms` | `150` ⚠️ **MISMATCH**: YAML=`100`, code default=`150` |
| [detection_shell.py](../stems_to_midi/detection_shell.py#L467) | 467 | `open_sustain_threshold_ms` | Used in classification: `if (geomean >= open_geomean_min and sustain_ms >= open_sustain_threshold_ms)` → 'open' | Received as param |
| [optimization/extract_features.py](../stems_to_midi/optimization/extract_features.py#L176) | 176 | `open_sustain_threshold` | `config.get('hihat', {}).get('open_sustain_ms', 150)` — for feature extraction comparison | `150` ⚠️ same mismatch |

### open_geomean_min

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [detection_shell.py](../stems_to_midi/detection_shell.py#L451) | 451 | `open_geomean_min` | `hihat_config.get('open_geomean_min', 262.0)` — used in classification logic: `if (geomean >= open_geomean_min and sustain_ms >= open_sustain_threshold_ms)` → 'open' | `262.0` ✅ match |
| [optimization/optimize.py](../stems_to_midi/optimization/optimize.py#L59) | 59+ | `open_geomean_min` | Function parameter for `evaluate_thresholds()`. Used in Bayesian optimization to find best hihat thresholds. | — |

---

## 20. `learning_mode` (top-level section)

**YAML defaults:** `enabled: false`, `export_all_detections: true`, `rejected_velocity: 1`, `kept_velocity_normal: true`, `learning_onset_threshold: 0.0001`, `learning_delta: 0.002`, `learning_wait: 1`, `learning_midi_suffix: "_learning"`, `calibrated_config_output: "midiconfig_calibrated.yaml"`

| File | Line | Variable | Usage | Code Default |
|------|------|----------|-------|-------------|
| [stems_to_midi_cli.py](../stems_to_midi_cli.py#L165) | 165-166 | `config['learning_mode']['enabled']` | Sets `enabled=True` when learning mode CLI flag is used | — |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L167) | 167 | `learning_mode` | `config.get('learning_mode', {}).get('enabled', False)` — gates ultra-sensitive detection params | `False` |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L173) | 173 | `learning_config['learning_onset_threshold']` | Ultra-low onset threshold for catching all possible hits | Direct key access |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L174) | 174 | `learning_config['learning_delta']` | Ultra-sensitive peak picking | Direct key access |
| [processing_shell.py](../stems_to_midi/processing_shell.py#L175) | 175 | `learning_config['learning_wait']` | Allow very close hits | Direct key access |
| [stems_to_midi_cli.py](../stems_to_midi_cli.py#L284) | 284 | `suffix` | `config.get('learning_mode', {}).get('learning_midi_suffix', '_learning')` — suffix for learning MIDI output files | `'_learning'` ✅ match |
| [analysis_core.py](../stems_to_midi/analysis_core.py#L1413) | 1413 | `learning_mode: bool = False` | Function parameter in `filter_onsets_by_spectral()` — if True, keeps all onsets regardless of filtering | — |

### Unused learning_mode sub-keys:
- **`export_all_detections`**: ⚠️ NOT read by any production code (only in test fixture)
- **`rejected_velocity`**: ⚠️ NOT read by any code
- **`kept_velocity_normal`**: ⚠️ NOT read by any code
- **`calibrated_config_output`**: ⚠️ NOT read by any code

---

## Summary: Default Mismatches

| Setting | YAML Default | Code Default | Risk |
|---------|-------------|-------------|------|
| `reverb_continuation_attack_threshold` | `0.4` | `0.2` | Low (YAML wins when config loaded) |
| `statistical_badness_threshold` | `0.3` | `0.6` | Low (YAML wins when config loaded) |
| `detect_open` | `true` | `False` | Low (YAML wins when config loaded) |
| `open_sustain_ms` | `100` | `150` | Low (YAML wins when config loaded) |
| `enable_pitch_detection` (snare) | `false` | `True` | Low (YAML wins when config loaded) |

These mismatches only matter if config loading fails or the key is missing.

---

## Dead Config (YAML keys never read by code)

| YAML Key | Section | Notes |
|----------|---------|-------|
| `onset_merge_window_ms` | Per-stem (kick/snare/toms/hihat/cymbals) AND energy_detection | WebUI schema only — never read by processing code. Different from `merge_window_ms` which IS used. |
| `enable_amplitude_refinement` | hihat | No code references at all |
| `decay_threshold` | hihat | Legacy key — replaced by `open_sustain_ms` + `open_geomean_min` classification |
| `threshold_optimization.initial_threshold_step` | threshold_optimization | Not read from config (hardcoded in optimization_core parameter defaults) |
| `threshold_optimization.convergence_patience` | threshold_optimization | Not read from config (hardcoded in optimization_core parameter defaults) |
| `learning_mode.export_all_detections` | learning_mode | Not read by any code |
| `learning_mode.rejected_velocity` | learning_mode | Not read by any code |
| `learning_mode.kept_velocity_normal` | learning_mode | Not read by any code |
| `learning_mode.calibrated_config_output` | learning_mode | Not read by any code |
| `clustering.features` | clustering | Not read by any code (optimization_core has its own feature list) |
| `enable_decay_filter` | cymbals (implicit — NOT in YAML) | Code reads it with default `True`, but the key is not in midiconfig.yaml |

---

## Orphaned Code References (read from config but NOT in midiconfig.yaml)

| Config Key Read | File | Line | Code Default | Notes |
|----------------|------|------|-------------|-------|
| `stem.timing_offset` | processing_shell.py | 189, 502 | `0.0` | Per-stem key, not in default YAML. Set in project overrides. |
| `cymbals.enable_decay_filter` | analysis_core.py | 1621 | `True` | Not in YAML; always on unless overridden |
| `cymbals.enable_pitch_detection` | processing_shell.py | 304 | `True` | Not in YAML; cymbals section lacks this key |
| `midi.min_velocity` / `midi.max_velocity` | — | — | — | Defined in YAML but **only read via CLI args** (not from config in production code). The CLI has its own defaults of 80/110. Test code asserts they exist in config but production code doesn't read them from config. |
