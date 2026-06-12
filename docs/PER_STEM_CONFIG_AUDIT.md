# Per-stem configuration audit (2026-06-11)

## Principle

The user wants as many parameters configurable per stem as is reasonable
(their words: "Unless there's a huge reason, we should make ANY of them
configurable per stem"). This document is the audit of which parameters
are currently GLOBAL but should be per-stem, and which genuinely need
to stay global.

## Tier 1 — Per-stem (definitely)

These have well-known stem-specific values that vary by 10x or more
across stems. Defaults should match the most common case for each stem
individually. Already partially per-stem in some cases; should be
formalized.

### Onset detection (librosa + energy)
- `onset_detection.threshold` — kick 0.1, snare 0.2, toms 0.3, hihat 0.003, cymbals 0.35
  → huge stem-to-stem variation; hihats are 100x more sensitive than cymbals
- `onset_detection.delta` — kick 0.1, toms 0.05, hihat 0.005, cymbals 0.1
  → hihats need 20x finer peak picking than kicks
- `onset_detection.wait` — kick 2, snare 2, toms 3, hihat 3, cymbals 10
  → cymbals need 5x longer NMS to avoid double-counting wash
- `onset_detection.hop_length` — 256 or 512 globally; toms/snare benefit from
  256 (sub-frame timing accuracy), kick/cymbals can use 512 (4x faster)
- `pga_min_prominence` — toms 1000, others TBD per stem

### Energy detection
- `threshold_db` — already per-stem (kick 12, snare 10, toms 6, hihat 6, cymbals 15)
- `min_peak_spacing_ms` — already per-stem (kick 50, snare 80, toms 90, hihat 20, cymbals 100)
- `min_absolute_energy` — already per-stem (kick 0.02, snare 0.015, toms 0.015, hihat 0.008, cymbals 0.01)
- `merge_window_ms` — already per-stem
- `peak_hold_ms` — should be per-stem (hihat 2ms tighter than cymbals 5ms)
- `energy_method` — could be per-stem (kick='rms' vs cymbals='peak_hold')

### Spectral filter
- `geomean_threshold` — already per-stem
- `min_strength_threshold` — already per-stem
- `min_sustain_ms` — should be per-stem
- `reverb_continuation_attack_threshold` — global 0.4; could be per-stem

### PGA (percentile-gated attack) detector
- `broad_freq_min_hz` — kick 30, toms 200, snare 150, hihat 500, cymbals 1000
  → MAJOR stem difference; current global 600 makes the PGA detector
    blind to kicks (which have all their energy below 200 Hz) and
    under-sensitive to toms (whose 2nd harmonic is in 400-800 Hz)
- `broad_freq_max_hz` — toms 8000, hihat 12000, cymbals 14000
  → hihat/cymbals have key spectral content above 8 kHz
- `db_rise_threshold` — global 10dB; could be per-stem
- `nms_min_frames` — global 20 (~116ms); kicks 10 (~58ms), cymbals 50 (~290ms)
- `strike_offset_sec` — global 8ms Hann bias; per-stem (kick ~10ms, toms ~8ms, cymbals ~6ms)
- `pga_min_prominence` — global 1000; per-stem (cymbals have huge envelope swings, 1000 would miss quieter crashes)

### Spectral-flatness diagnostic (this session)
- `flat_min_hz` / `flat_max_hz` — global 600/3000
  → toms 200/3000 (f2/f3 are 200-1500 Hz), kick 30/200, hihat 2000/8000
- `body_window_ms` — global 30; per-stem (cymbals 100ms, hihats 50ms)

### High-res decay signature (this session)
- `broad_min_hz` / `broad_max_hz` — global 600/8000
  → same as PGA above
- `n_fft` / `hop` — global 128/4
  → per-stem (cymbals might want 256/8 to skip the 100ms+ attack
    tail and find the 0.5s ring; kicks might want 256/8 to fit
    the 100Hz fundamental in n_fft)

### Per-event feature extraction
- `pitch_fmin_hz` / `pitch_fmax_hz` — already per-stem
  (kick 30-80, toms 60-250, snare 100-500, hihat 500-2000, cymbals 3000-8000)
- `broad_min_hz` / `broad_max_hz` (for duration/centroid) — global 200/8000
  → per-stem (kick 30/200, cymbals 1000/10000)

### Stereo processing
- `use_stereo` — already per-stem

## Tier 2 — Per-stem (consider)

Parameters where the stem-to-stem variation is smaller (1.5-2x) but
the user might still want to tune them. Could be left global for
now and added to per-stem only if requested.

- `audio.peak_window_sec` — global 0.1s. Kicks might want 0.05s
  (sharp impulse), cymbals 0.3s (longer peak measurement).
- `audio.sustain_window_sec` — global 0.2s. Cymbals 2.0s
  (already overridden in cymbals section). Others can stay at 0.2s.
- `audio.envelope_threshold` — global 0.1 (10% of peak). Could
  be per-stem (kicks need 20%, cymbals need 5% to capture long ring).
- `audio.envelope_smooth_kernel` — global 51. Per-stem.
- `midi.min_velocity` / `midi.max_velocity` — global 80/110. Could
  be per-stem (hihats want wider range, kicks want narrower).
- `midi.max_note_duration` — global 0.5s, cymbals override 2.0s.
  Should be per-stem by default.
- `detection_method` — global 'both' but currently 'energy' in midiconfig
  per-stem override? Need to check.

## Tier 3 — Global (keep as is)

These genuinely don't have a stem-specific optimum. The "right" value
is the same for all stems.

- `audio.force_mono` — global audio quality setting, not stem-specific
- `audio.silence_threshold` — universal silence gate, no stem variation
- `audio.min_segment_length` — math/algorithm constraint, not tunable
- `audio.normalize_amplitude` / `audio.normalize_stereo_balance` —
  recording-level setting, not stem-specific
- `audio.default_note_duration` / `audio.very_short_duration` —
  MIDI output fallback, not stem-specific
- `midi.default_tempo` — song-level, not stem-specific
- `clustering.method` — algorithm choice, not stem-specific
- `learning_mode.*` — debugging workflow, not per-stem

## What to do (recommended)

1. **Formalize Tier 1 as the v1 per-stem schema.** Add per-stem
   sections under each global config block in midiconfig.yaml.
   Don't change defaults — just allow per-stem overrides.

2. **For PGA + flatness + decay signature**: these are NEW this
   session and have NEVER had per-stem tuning. The current
   600/8000Hz default is calibrated for toms. For kicks, the
   detector effectively does nothing useful. Per-stem is
   essential before shipping.

3. **For the energy/spectral detectors (already per-stem)**:
   just confirm all the per-stem keys are actually being
   read in the code. Some keys might be defined in YAML
   but not consumed by the loader.

4. **For Tier 2**: leave alone until someone asks.

## Open questions

- **Where in the config schema do per-stem overrides live?**
  Option A: nested under each stem (`toms.pga_broad_freq_min_hz`).
  Option B: top-level per-stem block (`pga_per_stem.toms.broad_freq_min_hz`).
  Option A is more conventional (each stem owns its overrides) but
  means a lot of duplication if 5 stems need to set the same key.
  The user's current midiconfig.yaml uses Option A (each stem
  has its own threshold_db, min_peak_spacing_ms, etc).

- **How to handle "no override" vs "inherit global default"?**
  Two patterns: (a) every per-stem key has its own default, and
  the global default is just the value used if no per-stem key is
  set. (b) per-stem keys are sparse — unset keys fall back to the
  global default. The user seems to prefer (b) (the comment
  "Per-stem overrides: onset_threshold, onset_delta, onset_wait,
  pga_min_prominence" on line 52 of midiconfig.yaml confirms this).

- **Should we add a validator that warns if a per-stem key is
  unrecognized?** (i.e. user typo'd `pga_broad_freq_min_hz` as
  `pga_broad_min_hz` and silently fell through to the global
  default). Currently the loader just ignores unknown keys.

## Validation needed before committing any defaults

For each new per-stem key, we need at least one example audio
recording to validate the default against. The current toms
calibration (project 4) covers toms. We do NOT have kick/snare/
hihat/cymbals calibration data — so the right move for this
session is:

1. **Wire the schema** (allow per-stem overrides, no defaults)
2. **Keep the current global defaults** (still 600/8000 for PGA)
3. **Mark as TODO in docs** that per-stem defaults need calibration

This is what the user asked for: "make ANY of them configurable
per stem" — not "calibrate them all now." Calibration can be
a future session.
