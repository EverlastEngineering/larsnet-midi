# midiconfig.yaml Settings Reference

Complete reference for all settings in `midiconfig.yaml`, organized by function (Detection vs Filtering) and traced to their code implementation.

## How Settings Flow Through the System

```
midiconfig.yaml (root or per-project)
        │
        ▼
stems_to_midi_cli.py → yaml.safe_load() → raw config dict
        │
        ├── DrumMapping.from_config(config) → MIDI note mapping
        │
        └── For each stem file:
            ├── _load_and_validate_audio()     ← audio.* settings
            ├── _configure_onset_detection()   ← onset_detection.* settings
            ├── detect_onsets_energy_based()    ← energy_detection.* + per-stem settings
            ├── filter_onsets_by_spectral()     ← filtering.* + per-stem spectral settings
            ├── detect_hihat_state()            ← hihat.* classification settings
            ├── _create_midi_events()           ← midi.* + per-stem note mappings
            └── create_midi_file()              ← midi.default_tempo, audio.very_short_duration
```

---

## Detection Settings

Detection settings control **how audio events (drum hits) are found** in the waveform. They determine sensitivity, timing resolution, and the detection algorithm used.

### `audio` — Audio Preprocessing

These settings prepare audio before any detection or filtering occurs.

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `force_mono` | `true` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L93) | **DEPRECATED.** Legacy mono conversion. Overridden by per-stem `use_stereo`. When `use_stereo` is not set for a stem, this fallback is used: `use_stereo = not force_mono`. |
| `silence_threshold` | `0.001` | float | [processing_shell.py](../stems_to_midi/processing_shell.py#L110) | Amplitude below which the entire audio file is considered silent and skipped. Corresponds to approximately -60dB. |
| `min_segment_length` | `512` | int (samples) | [analysis_core.py](../stems_to_midi/analysis_core.py#L2195) | Minimum audio segment length (in samples) for spectral analysis. Segments shorter than this are skipped. Used in `analyze_onset_spectral()`. |
| `normalize_amplitude` | `false` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L116) | Scale audio to `target_amplitude` before detection. Per-stem override available. Only scales up (quiet → louder), never scales down. |
| `target_amplitude` | `0.8` | float (0–1) | [processing_shell.py](../stems_to_midi/processing_shell.py#L120) | Target peak amplitude when `normalize_amplitude` is enabled. |
| `normalize_stereo_balance` | `false` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L125) | Equalize L/R channel RMS levels for fair stereo detection. Per-stem override available. |
| `peak_window_sec` | `0.1` | float (seconds) | [analysis_core.py](../stems_to_midi/analysis_core.py#L2193) | Window size after onset for measuring peak amplitude. Used in `analyze_onset_spectral()` to extract the audio segment for spectral analysis. |
| `sustain_window_sec` | `0.2` | float (seconds) | [analysis_core.py](../stems_to_midi/analysis_core.py#L2223), [detection_shell.py](../stems_to_midi/detection_shell.py#L467) | Window for analyzing sound decay/sustain. Used for hihat and cymbal sustain duration calculation. Per-stem override: `sustain_analysis_window_sec` (cymbals). |
| `envelope_threshold` | `0.1` | float (0–1) | [analysis_core.py](../stems_to_midi/analysis_core.py#L2224), [detection_shell.py](../stems_to_midi/detection_shell.py#L468) | Fraction of peak amplitude for sustain measurement. Sustain = how long envelope stays above `peak * envelope_threshold`. |
| `envelope_smooth_kernel` | `51` | int (odd) | [analysis_core.py](../stems_to_midi/analysis_core.py#L2225), [detection_shell.py](../stems_to_midi/detection_shell.py#L469) | Median filter kernel size for smoothing the amplitude envelope. Must be odd. Larger = smoother. |
| `default_note_duration` | `0.1` | float (seconds) | [processing_shell.py](../stems_to_midi/processing_shell.py#L567) | Duration assigned to the last MIDI note in a sequence (no next note to calculate gap). |
| `very_short_duration` | `0.01` | float (beats) | [midi.py](../stems_to_midi/midi.py#L71) | Duration of the anchor note at time 0 in the MIDI file. Used for DAW alignment. |

### `onset_detection` — Librosa Detection (Legacy)

These settings control the **old librosa-based** onset detection method. Only used when `use_librosa_detection: true` is set on a specific stem.

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `threshold` | `0.3` | float (0–1) | [processing_shell.py](../stems_to_midi/processing_shell.py#L179) | Onset strength threshold. Lower = more sensitive. Normalized onset strengths below this are discarded. Per-stem override: `onset_threshold`. |
| `delta` | `0.01` | float | [processing_shell.py](../stems_to_midi/processing_shell.py#L182) | Peak picking sensitivity for librosa. Lower = more sensitive. Per-stem override: `onset_delta`. |
| `wait` | `3` | int (frames) | [processing_shell.py](../stems_to_midi/processing_shell.py#L185) | Minimum frames between detected peaks. Each frame ≈ 11ms at hop_length=512. Per-stem override: `onset_wait`. |
| `hop_length` | `512` | int (samples) | [processing_shell.py](../stems_to_midi/processing_shell.py#L178), [energy_detection_shell.py](../stems_to_midi/energy_detection_shell.py#L46) | Samples between analysis frames. Affects time resolution for **both** detection methods. Lower = finer resolution but slower. At 44100Hz: 512 samples ≈ 11.6ms per frame. |

### `energy_detection` — Energy-Based Detection (Default)

These settings control the **new energy-based** detection method (scipy peak detection with backtracking). This is the default method.

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `use_librosa_detection` | `false` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L724) | Method selector. `false` = energy-based (default, recommended). `true` = legacy librosa. Per-stem override available. |
| `threshold_db` | `15.0` | float (dB) | [processing_shell.py](../stems_to_midi/processing_shell.py#L731), [energy_detection_core.py](../stems_to_midi/energy_detection_core.py#L213) | Prominence threshold: peaks must stand this many dB above the local minimum. Lower = more sensitive. Per-stem override available. |
| `min_peak_spacing_ms` | `100.0` | float (ms) | [processing_shell.py](../stems_to_midi/processing_shell.py#L732), [energy_detection_core.py](../stems_to_midi/energy_detection_core.py#L200) | Minimum time between detected peaks. Prevents double-detection. Per-stem override available. |
| `min_absolute_energy` | `0.01` | float | [processing_shell.py](../stems_to_midi/processing_shell.py#L733), [energy_detection_core.py](../stems_to_midi/energy_detection_core.py#L222) | Noise floor threshold. Peaks below this absolute energy are ignored. Per-stem override available. |
| `merge_window_ms` | `150.0` | float (ms) | [processing_shell.py](../stems_to_midi/processing_shell.py#L734), [energy_detection_core.py](../stems_to_midi/energy_detection_core.py#L452) | Window for merging L/R channel peaks into single events during stereo detection. Per-stem override available. |
| `use_stereo` | `true` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L91) | Detect in stereo and calculate pan position. Per-stem override available. |
| `onset_merge_window_ms` | `100` | int (ms) | — | **DEAD CONFIG.** Defined in YAML and WebUI schema but never read by the processing pipeline. The actual merge behavior is controlled by `merge_window_ms`. |
| `energy_method` | `'peak_hold'` | string | [processing_shell.py](../stems_to_midi/processing_shell.py#L735), [energy_detection_core.py](../stems_to_midi/energy_detection_core.py#L88) | Envelope calculation method: `'rms'` (root mean square), `'spectral'` (spectral flux), or `'peak_hold'` (DAW-like waveform display). `peak_hold` preserves transients best. Per-stem override available. |
| `peak_hold_ms` | `3.0` | float (ms) | [processing_shell.py](../stems_to_midi/processing_shell.py#L736), [energy_detection_core.py](../stems_to_midi/energy_detection_core.py#L123) | Smoothing window for peak_hold method. 2–5ms typical. Smaller = sharper transients, more noise. Per-stem override available. |

---

## Filter Settings

Filter settings control **which detected events are kept or rejected** based on spectral analysis, sustain duration, and statistical properties.

### `filtering` — Global Filter Settings

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `geomean_threshold` | `null` | float or null | [analysis_core.py](../stems_to_midi/analysis_core.py#L792) | Global default geometric mean threshold. Per-stem values override this. `null` = no filtering. |
| `min_strength_threshold` | `null` | float or null | [analysis_core.py](../stems_to_midi/analysis_core.py#L797) | Global minimum onset strength. Events below this are rejected before spectral analysis. Per-stem override available. |
| `min_sustain_ms` | `null` | float or null | [analysis_core.py](../stems_to_midi/analysis_core.py#L793) | Global minimum sustain duration. Per-stem values override. Only used for hihat and cymbals. |
| `expected_clusters` | `null` | int or null | — | Per-stem override. Used by optimization_core.py if `threshold_optimization.enabled` is true. `null` = disabled. |
| `enable_spectral_filter` | `true` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L800) | Master switch for spectral filtering. `false` = keep all detected onsets (no spectral analysis). Per-stem override available. |
| `reverb_continuation_attack_threshold` | `0.4` | float | [analysis_core.py](../stems_to_midi/analysis_core.py#L1805) | Attack sharpness threshold for reverb continuation filtering. Events with attack sharpness below this are marked as `REVERB_CONTINUATION` and excluded from MIDI output. Real drum hits typically have sharpness ≥ 0.4; reverb tails have < 0.4. **Note:** Code default fallback is `0.2` (mismatch with YAML value `0.4`). |

### Per-Stem Spectral Filtering

Each stem type has frequency ranges defining domain-specific energy bands. The geometric mean of these bands determines whether a detection is a real hit or an artifact.

**Filtering formula:** `geomean = sqrt(band1_energy × band2_energy)` (or `cbrt(band1 × band2 × band3)` for kick's 3-way geomean).

Events are kept when `geomean > geomean_threshold`.

#### Kick

| Setting | Default | Type | Description |
|---|---|---|---|
| `geomean_threshold` | `800.0` | float | 3-way geomean threshold: `cbrt(FundE × BodyE × AttackE)`. Real kicks: 500–2000. |
| `fundamental_freq_min` | `40` | int (Hz) | Start of fundamental range → `fundamental_energy` in analysis.json. |
| `fundamental_freq_max` | `80` | int (Hz) | End of fundamental range. |
| `body_freq_min` | `80` | int (Hz) | Start of body range → `body_energy` in analysis.json. |
| `body_freq_max` | `250` | int (Hz) | End of body range. |
| `attack_freq_min` | `1500` | int (Hz) | Start of beater attack range → `attack_energy` in analysis.json. |
| `attack_freq_max` | `5000` | int (Hz) | End of beater attack range. |
| `enable_statistical_filter` | `false` | bool | Enable second-pass statistical outlier detection (catches snare bleed). |
| `statistical_badness_threshold` | `0.3` | float (0–1) | Badness score threshold. Higher = more permissive. **Note:** Code default fallback is `0.6` (mismatch with YAML value `0.3`). |
| `statistical_ratio_weight` | `0.7` | float (0–1) | Weight for FundE/BodyE ratio deviation in badness calculation. |
| `statistical_total_weight` | `0.3` | float (0–1) | Weight for total energy deviation in badness calculation. |

#### Snare

| Setting | Default | Type | Description |
|---|---|---|---|
| `geomean_threshold` | `40.0` | float | 2-way geomean: `sqrt(BodyE × WireE)`. Real snares: 250–1200. |
| `low_freq_min` | `40` | int (Hz) | Start of low frequency range (kick bleed detection). Not used in geomean — used for spectral ratio. |
| `low_freq_max` | `150` | int (Hz) | End of low frequency range. |
| `body_freq_min` | `150` | int (Hz) | Start of snare body → `body_energy` in analysis.json. |
| `body_freq_max` | `400` | int (Hz) | End of snare body. |
| `wire_freq_min` | `2000` | int (Hz) | Start of snare wire → `wire_energy` in analysis.json. |
| `wire_freq_max` | `8000` | int (Hz) | End of snare wire. |

#### Toms

| Setting | Default | Type | Description |
|---|---|---|---|
| `geomean_threshold` | `80.0` | float | 2-way geomean: `sqrt(FundE × BodyE)`. Real toms: 150–600. |
| `fundamental_freq_min` | `60` | int (Hz) | Start of fundamental → `fundamental_energy` in analysis.json. |
| `fundamental_freq_max` | `150` | int (Hz) | End of fundamental. |
| `body_freq_min` | `150` | int (Hz) | Start of body → `body_energy` in analysis.json. |
| `body_freq_max` | `400` | int (Hz) | End of body. |

#### Hi-hat

| Setting | Default | Type | Description |
|---|---|---|---|
| `geomean_threshold` | `8.0` | float | 2-way geomean: `sqrt(BodyE × SizzleE)`. Real hihats: 15–400. |
| `min_strength_threshold` | `0.02` | float | Minimum onset strength. |
| `min_sustain_ms` | `25` | float (ms) | Minimum sustain duration. Filters handclap bleed (~20ms). |
| `body_freq_min` | `500` | int (Hz) | Start of body → `body_energy` in analysis.json. |
| `body_freq_max` | `2000` | int (Hz) | End of body. |
| `sizzle_freq_min` | `6000` | int (Hz) | Start of sizzle → `sizzle_energy` in analysis.json. |
| `sizzle_freq_max` | `12000` | int (Hz) | End of sizzle. |

#### Cymbals

| Setting | Default | Type | Description |
|---|---|---|---|
| `geomean_threshold` | `100` | float | 2-way geomean: `sqrt(BodyE × BrillianceE)`. Real cymbals: 50–500. |
| `min_strength_threshold` | `0.1` | float | Minimum onset strength. |
| `min_sustain_ms` | `150` | float (ms) | Minimum sustain duration for valid cymbal hits. |
| `body_freq_min` | `1000` | int (Hz) | Start of body/wash → `body_energy` in analysis.json. |
| `body_freq_max` | `4000` | int (Hz) | End of body. |
| `brilliance_freq_min` | `4000` | int (Hz) | Start of brilliance → `brilliance_energy` in analysis.json. |
| `brilliance_freq_max` | `10000` | int (Hz) | End of brilliance. |
| `sustain_analysis_window_sec` | `2.0` | float (seconds) | Window for measuring cymbal sustain duration. Overrides `audio.sustain_window_sec` for cymbals. |
| `decay_filter_window_sec` | `0.5` | float (seconds) | Window for decay pattern retriggering filter (Pass 2). Checks if onset occurs during decay of previous hit. |
| `max_note_duration` | `2.0` | float (seconds) | Maximum MIDI note duration for cymbal hits. Uses actual sustain, capped at this value. |

---

## Per-Stem Settings

Each stem section (`kick`, `snare`, `toms`, `hihat`, `cymbals`) shares a common structure for detection overrides plus stem-specific settings.

### Common Detection Overrides

These per-stem settings override global `energy_detection` defaults:

| Setting | Description |
|---|---|
| `use_stereo` | Override `energy_detection.use_stereo`. |
| `use_librosa_detection` | Override `energy_detection.use_librosa_detection`. |
| `threshold_db` | Override `energy_detection.threshold_db`. |
| `min_peak_spacing_ms` | Override `energy_detection.min_peak_spacing_ms`. |
| `min_absolute_energy` | Override `energy_detection.min_absolute_energy`. |
| `merge_window_ms` | Override `energy_detection.merge_window_ms`. |
| `energy_method` | Override `energy_detection.energy_method`. |
| `peak_hold_ms` | Override `energy_detection.peak_hold_ms`. |
| `onset_threshold` | Override `onset_detection.threshold` (librosa only). |
| `onset_delta` | Override `onset_detection.delta` (librosa only). |
| `onset_wait` | Override `onset_detection.wait` (librosa only). |
| `normalize_amplitude` | Override `audio.normalize_amplitude`. |
| `normalize_stereo_balance` | Override `audio.normalize_stereo_balance`. |

### MIDI Note Mappings

| Stem | Setting | Default | GM Note |
|---|---|---|---|
| **Kick** | `midi_note` | `36` | C1 - Bass Drum 1 |
| **Snare** | `midi_note` | `38` | D1 - Acoustic Snare |
| | `midi_note_rimshot` | `37` | C#1 - Side Stick |
| | `midi_note_clap` | `39` | D#1 - Hand Clap |
| **Toms** | `midi_note_low` | `45` | A1 - Low Tom |
| | `midi_note_mid` | `47` | B1 - Mid Tom |
| | `midi_note_high` | `50` | D2 - High Tom |
| **Hi-hat** | `midi_note_closed` | `42` | F#1 - Closed Hi-Hat |
| | `midi_note_open` | `46` | A#1 - Open Hi-Hat |
| | `midi_note_foot_close` | `44` | G#1 - Foot Close |
| | `midi_note_handclap` | `39` | D#1 - Hand Clap (bleed) |
| | `midi_note` | `42` | Backward compatibility alias for closed |
| **Cymbals** | `midi_note` | `57` | Backward compatibility (unused in current code) |
| | `midi_note_crash` | `49` | C#2 - Crash Cymbal 1 |
| | `midi_note_ride` | `51` | D#2 - Ride Cymbal 1 |
| | `midi_note_chinese` | `52` | E2 - Chinese Cymbal |

### Pitch Detection Settings

Used for classifying detected events into subtypes (e.g., low/mid/high tom, crash/ride, snare/rimshot).

| Stem | Setting | Default | Description |
|---|---|---|---|
| **Toms** | `enable_pitch_detection` | `true` | Enable pitch-based low/mid/high classification. |
| | `pitch_method` | `'yin'` | `'yin'` (faster) or `'pyin'` (more robust). Note: YIN is recommended for short decaying percussive sounds like toms. |
| | `min_pitch_hz` | `60` | Minimum expected tom pitch (Hz). |
| | `max_pitch_hz` | `250` | Maximum expected tom pitch (Hz). |
| | `cluster_feature` | `'pitch_hz'` | Feature to use for clustering: `'pitch_hz'` (recommended for toms), `'spectral_centroid_hz'` (brightness), `'stereo_width'`, `'pan_confidence'`, or `'auto'`. Default is `'pitch_hz'` which provides best separation for toms. |
| **Snare** | `enable_pitch_detection` | `false` | Enable pitch-based snare/rimshot/clap classification. |
| | `pitch_method` | `'yin'` | Same as toms. |
| | `min_pitch_hz` | `100.0` | Minimum expected snare pitch. |
| | `max_pitch_hz` | `500.0` | Maximum expected snare pitch. |
| **Cymbals** | `enable_pitch_detection` | (not in YAML) | Not explicitly set in YAML. Code default is `True`. Classification primarily uses pan position via `classify_cymbal_by_pan()`. |
| | `pitch_method` | `'yin'` | Same as toms. |
| | `min_pitch_hz` | `200.0` | Code default (not in YAML). |
| | `max_pitch_hz` | `1000.0` | Code default (not in YAML). |

### Hi-hat Classification Settings

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `decay_threshold` | `0.65` | float | — | **DEAD CONFIG.** Not read by current code. Open/closed detection now uses `open_sustain_ms` + `open_geomean_min` instead. |
| `open_sustain_ms` | `100` | float (ms) | [detection_shell.py](../stems_to_midi/detection_shell.py#L441), [processing_shell.py](../stems_to_midi/processing_shell.py#L1044) | Sustain threshold for open detection. Events with sustain ≥ this AND geomean ≥ `open_geomean_min` are classified as "open". **Note:** Code default fallback is `150` (mismatch with YAML `100`). |
| `open_geomean_min` | `262.0` | float | [detection_shell.py](../stems_to_midi/detection_shell.py#L442) | GeoMean threshold for open detection. Used in conjunction with `open_sustain_ms`. |
| `generate_foot_close` | `false` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L503) | Generate MIDI foot-close events (note 44) at the end of open hi-hat sustain. Velocity = 70% of open hit, capped 40–100. |
| `enable_amplitude_refinement` | `false` | bool | — | **DEAD CONFIG.** Not read by any processing code. |

---

## Output & Coordination Settings

### `clustering` — Adaptive Threshold Discovery

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `method` | `'dbscan'` | string | [clustering_core.py](../stems_to_midi/clustering_core.py#L110) | Clustering algorithm: `'dbscan'` (density-based) or `'kmeans'` (centroid-based). Used by `optimization_core.py`. |
| `features` | (list) | list[str] | — | **DEAD CONFIG.** The features list in YAML is not read by `optimization_core.py`. The code uses its own internal default feature list. |

### `threshold_optimization` — Iterative Threshold Adjustment

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `enabled` | `false` | bool | WebUI schema only | Enable adaptive threshold discovery. Not consumed by the main processing pipeline — only by `optimization_core.py` when called explicitly. |
| `max_iterations` | `20` | int | [optimization_core.py](../stems_to_midi/optimization_core.py#L193) | Maximum optimization iterations. Code receives as function parameter, not from config. |
| `tolerance` | `0` | int | [optimization_core.py](../stems_to_midi/optimization_core.py#L195) | Stop when cluster count within ±N of expected. |
| `initial_threshold_step` | `0.05` | float | — | **DEAD CONFIG.** Not read. `optimization_core.py` uses `threshold_step_initial=0.1` as function parameter default. |
| `convergence_patience` | `3` | int | [optimization_core.py](../stems_to_midi/optimization_core.py#L198) | Stop after N iterations without improvement. Code receives as function parameter, not from config. |

### `midi` — MIDI Output Settings

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `min_velocity` | `80` | int (1–127) | [stems_to_midi_cli.py](../stems_to_midi_cli.py#L394) | Minimum MIDI velocity. Overridable via CLI `--min-vel` (default 40). |
| `max_velocity` | `110` | int (1–127) | [stems_to_midi_cli.py](../stems_to_midi_cli.py#L396) | Maximum MIDI velocity. Overridable via CLI `--max-vel` (default 127). |
| `default_tempo` | `120.0` | float (BPM) | [stems_to_midi_cli.py](../stems_to_midi_cli.py#L196) | Default tempo. Overridable via CLI `--tempo`. |
| `max_note_duration` | `0.5` | float (seconds) | [processing_shell.py](../stems_to_midi/processing_shell.py#L564) | Maximum MIDI note duration for non-cymbal stems. Cymbals use their own `max_note_duration` (default 2.0s). |

### `debug` — Debug Output

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `show_all_onsets` | `true` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L798) | Print detailed spectral analysis table for ALL detected onsets (kept + filtered). |
| `show_spectral_data` | `true` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L799) | Print spectral energy values and filtering summary. |

### `learning_mode` — Threshold Learning

| Setting | Default | Type | Code Location | Description |
|---|---|---|---|---|
| `enabled` | `false` | bool | [processing_shell.py](../stems_to_midi/processing_shell.py#L167) | Enable learning mode. All detections are kept (even rejected ones). |
| `learning_onset_threshold` | `0.0001` | float | [processing_shell.py](../stems_to_midi/processing_shell.py#L172) | Ultra-sensitive threshold for catching all possible hits (librosa mode). |
| `learning_delta` | `0.002` | float | [processing_shell.py](../stems_to_midi/processing_shell.py#L173) | Ultra-sensitive peak picking (librosa mode). |
| `learning_wait` | `1` | int | [processing_shell.py](../stems_to_midi/processing_shell.py#L174) | Allow very close hits (librosa mode). |
| `learning_midi_suffix` | `"_learning"` | string | [stems_to_midi_cli.py](../stems_to_midi_cli.py#L282) | Suffix added to learning mode MIDI filenames. |
| `export_all_detections` | `true` | bool | — | **DEAD CONFIG.** Not read by code. Learning mode always exports all detections. |
| `rejected_velocity` | `1` | int | — | **DEAD CONFIG.** Not read by code. Rejected events in learning mode simply remain in the all_onset_data but are not added to MIDI output. |
| `kept_velocity_normal` | `true` | bool | — | **DEAD CONFIG.** Not read by code. |
| `calibrated_config_output` | `"midiconfig_calibrated.yaml"` | string | — | **DEAD CONFIG.** Not read by `learning.py`. The module uses its own `save_calibrated_config()` function with a different path mechanism. |

---

## analysis.json Output Mapping

The analysis.json sidecar file uses **domain-specific field names** that match the frequency band names in config settings. Each stem's `freq_bands` metadata block maps field names to Hz ranges.

### Field Names by Stem

| Stem | Energy Fields | GeoMean Formula |
|---|---|---|
| Kick | `fundamental_energy`, `body_energy`, `attack_energy` | `cbrt(FundE × BodyE × AttackE)` |
| Snare | `body_energy`, `wire_energy` | `sqrt(BodyE × WireE)` |
| Toms | `fundamental_energy`, `body_energy` | `sqrt(FundE × BodyE)` |
| Hi-hat | `body_energy`, `sizzle_energy` | `sqrt(BodyE × SizzleE)` |
| Cymbals | `body_energy`, `brilliance_energy` | `sqrt(BodyE × BrillianceE)` |

### Remaining Naming Differences

| analysis.json field | Config setting | Notes |
|---|---|---|
| `geomean` | `geomean_threshold` | Config = threshold, JSON = measured value |
| `statistical_enabled` | `enable_statistical_filter` | Naming inconsistency |
| `decay_window_sec` | `decay_filter_window_sec` | Truncated name |
| `strength` | `onset_threshold` / `threshold_db` | Config = threshold, JSON = measured value |
| `sustain_ms` | `min_sustain_ms` | Config = filter threshold, JSON = measured duration |

---

## Settings Not Read by Code (Dead Config)

These settings are defined in midiconfig.yaml but not consumed by the processing pipeline:

| Setting | Section | Notes |
|---|---|---|
| `onset_merge_window_ms` | All stems | Only in WebUI schema. Confused with `merge_window_ms` which is actually used. |
| `enable_amplitude_refinement` | hihat | Zero code references. |
| `decay_threshold` | hihat | Replaced by `open_sustain_ms` + `open_geomean_min`. |
| `initial_threshold_step` | threshold_optimization | Code uses hardcoded function default (`0.1`). |
| `convergence_patience` | threshold_optimization | Passed as function parameter, not read from config. |
| `clustering.features` | clustering | Code uses internal default feature list. |
| `export_all_detections` | learning_mode | Behavior is hardcoded. |
| `rejected_velocity` | learning_mode | Not read. |
| `kept_velocity_normal` | learning_mode | Not read. |
| `calibrated_config_output` | learning_mode | Not read. |

## Code Default Mismatches

Cases where the YAML value differs from the code's fallback default (the fallback used if the YAML key is missing):

| Setting | YAML Value | Code Default | Risk |
|---|---|---|---|
| `reverb_continuation_attack_threshold` | `0.4` | `0.2` | Low (YAML wins when config is loaded) |
| `statistical_badness_threshold` | `0.3` | `0.6` | Low (YAML wins when config is loaded) |
| `open_sustain_ms` (hihat) | `100` | `150` | Low (YAML wins when config is loaded) |
| `enable_pitch_detection` (snare) | `false` | `True` | Low (YAML wins when config is loaded) |
