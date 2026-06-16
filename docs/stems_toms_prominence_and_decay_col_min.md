# Toms Stem: PGA Prominence + Decay Col-Min Calculation

**Date**: 2026-06-15
**Status**: Source of truth for the toms PGA prominence and
`decay_col_min_median_db` calculations. Future maintainers
should not have to dig through chat history to find these
two algorithms — they are documented here.

This file documents the two per-event diagnostic calculations
that drive the toms stem filter chain (the `pga_min_prominence`
and `min_decay_col_min_db` filters). It is the source of
truth for these two algorithms and the surrounding context
(why they exist, what they measure, how to interpret them).

---

## 1. How prominence is calculated in the toms stem

The toms stem uses the **PGA (Percentile-Gated Broad-Attack)**
detector in
[`stems_to_midi/percentile_gated_detector.py`](../stems_to_midi/percentile_gated_detector.py).
Prominence is computed per-event by
`scipy.signal.find_peaks` and stashed on each event under
the `prominence` key. The user-facing `pga_min_prominence`
knob scales against that value.

### 1.1 The detector envelope (5 steps)

1. **STFT**: `hop=256`, `n_fft=1024`, Hann window,
   log-magnitude spectrogram.
2. **Per-bin static noise floor** (with global noise gate,
   2026-06-15): for each frequency bin, take the 5th
   percentile of frames that are ≥ 0.5 dB above the bin's
   absolute minimum. The bin's floor is the mean of all
   values ≤ p5. After the per-bin pass, every bin's floor
   is clamped to `>=` the global gate = `max(p5 across all
   bins)` (the upper bound of the quietest portions of the
   song). This kills the silence-to-noise phantom that
   arises from stem-splitter digital-silence gaps.
3. **Contrast**: `max(0, spectrogram − noise_floor)`,
   shape `(n_bins, n_frames)`.
4. **Broad-frequency attack envelope**: sum of contrast over
   600–8000 Hz bins, after thresholding each bin at +10 dB
   above its floor. Excludes the saturated 0–600 Hz low
   bands on toms.
5. **Peak-pick** with `height = q3 + 2.5 * IQR` of the
   envelope (Tukey extreme-outlier rule, adapts per song)
   and `distance = 20 frames` (~116 ms NMS).

Each detected peak is then sub-frame-refined via parabolic
interpolation and shifted back by 8 ms to compensate for
the Hann window's center-of-bin bias (the contrast peaks
a few frames AFTER the actual strike onset).

### 1.2 The per-event `prominence` value

`scipy.signal.find_peaks(envelope, height=..., distance=...,
prominence=0)` is called with `prominence=0`, which is
slightly counterintuitive: it just requires "any prominence
> 0" to kill plateau/flat-top FPs. The actual `prominence`
value stored on each event is `scipy`'s standard "vertical
distance to the lowest contour line that bounds the peak
without crossing it" — calculated over the IQR-gated
detection window.

### 1.3 Threshold resolution (priority order)

1. `config.toms.pga_min_prominence` (per-stem override)
2. `config.onset_detection.pga_min_prominence` (global)
3. `1000.0` (hard floor, defensive)

The resolved value is stamped on every event in
`pga_filter_config.pga_min_prominence` so the sidecar
tooltip can show "Active filter: pga_min_prominence=X"
alongside the event.

### 1.4 Where the filter is invoked

The threshold comparison is invoked at three points:

| Call site | When | What it does |
|---|---|---|
| `pga_event_builder._build_pga_events_with_filter` | At detection time | Stamps `pga_filter_config`; tags FILTERED for the legacy return shape |
| `processing_shell_percentile_gated.process_percentile_gated` | PGA-only pipeline end-to-end | Builds MIDI events from `pga_kept` only |
| `rebuild_core._apply_toms_pga_prominence_refilter` | Every WebUI re-render | Always re-applies the filter for toms (2026-06-15), reading threshold fresh from YAML |

> **Toms-specific carve-out** at
> `rebuild_core.py:365`: `method='percentile_gated'` events
> are exempt from the geomean/sustain/strength filter that
> other stems use, because they have no geomean/sustain/strength
> values. Running them through `should_keep_onset` would
> default `geomean=0.0` and silently drop the entire toms
> stream on rebuild when `geomean_threshold` is non-zero.

### 1.5 TL;DR

For the toms stem, each event's `prominence` is **scipy's
standard prominence on a broadband contrast-summed attack
envelope** (600–8000 Hz, contrast = `spectrogram −
per_bin_noise_floor`, IQR-gated peak-pick). The user-facing
`pga_min_prominence` knob is then a simple `prominence <
threshold` test applied in a separate pure-function pass —
not during peak detection.

---

## 2. How `decay_col_min_median_db` is calculated

This is **not a filter** — it's a diagnostic feature added
on 2026-06-11 in
[`stems_to_midi/event_features.py`](../stems_to_midi/event_features.py)
under `compute_high_res_decay_signature`. It measures
broadband level in the ring of a percussive event to
distinguish real toms strikes from single-frame noise pops.

### 2.1 The high-resolution STFT

Standard pipeline STFT (`n_fft=1024, hop=256` ≈ 5.8 ms /
frame) smears the attack over 10+ ms. The signature uses a
much finer STFT:

| Param | Standard | High-res |
|---|---|---|
| `n_fft` | 1024 | **128** |
| `hop` | 256 | **4** |
| Time/frame | 5.8 ms | **0.091 ms** |
| Hz/bin | 43 | **345** |

Audio window: **10 ms before → 200 ms after** the event
time. If shorter than `n_fft` samples, returns `None`.

### 2.2 The contrast envelope

```python
freq_mask = (freqs >= broad_min_hz) & (freqs <= broad_max_hz)  # 600–8000 Hz
floor = np.percentile(band_db, 5, axis=1, keepdims=True)
contrast = np.maximum(band_db - floor, 0)
envelope = contrast.sum(axis=0)
```

The envelope is the sum of broadband-bin contrast (signal
− 5th-percentile floor) above the per-bin noise floor.

### 2.3 Peak search + attack/decay split

```python
search_start = max(0, pga_frame_local - 5)        # 5 frames ≈ 0.45 ms before PGA report
search_end   = min(len(envelope), pga_frame_local + 300)  # 300 frames ≈ 27 ms after
peak_frame   = argmax(envelope[search_start:search_end])
attack_end   = peak_frame + 30    # 30 frames ≈ 2.7 ms
decay_end    = attack_end + 200   # 200 frames ≈ 18 ms (15 ms in tests)
```

The 30-frame attack window covers the initial impulse; the
200-frame decay window covers the ring.

### 2.4 The `col_min` calculation

`col` = **column** in a spectrogram (one frame, all
frequency bins). `col_min` per frame = the **lowest-energy
frequency bin** in that frame:

```python
per_frame_min = s_db.min(axis=0)   # shape (n_frames,) — one min per frame
decay_col_min_median = float(np.median(per_frame_min[attack_end:decay_end]))
```

The stored value is the **median** (not the mean) of
per-frame mins over the decay window. Median is chosen so a
single bright harmonic bin doesn't pull the mean up — col_min
is sensitive to broadband content.

### 2.5 Interpretation (calibrated on project 4 + 6)

| Event type | `decay_col_min_median_db` |
|---|---|
| Real toms strike (sustained ring) | **-60 to -84 dB** |
| Single-frame noise pop / gap | **-84 to -90 dB** |

A real strike has a **broadband decaying body** (every bin
has some energy during the ring), so the lowest bin stays
elevated. A noise pop has a single bright impulse and then
nothing — col_min drops to the noise floor.

> **Critical for col_min to be high**: the ring needs
> **broadband content**, not just a harmonic stack. A pure
> harmonic series has quiet bins between harmonics (at the
> noise floor) which would drop col_min. Per
> `test_event_features.py:748-752`, band-limited noise is
> the right calibration signal.

### 2.6 Output fields & persistence

The full signature returns five fields:

```python
{
    'hr_peak_time':           ...,
    'hr_peak_offset_ms':      (peak_time - event_time_sec) * 1000,
    'hr_peak_envelope':       ...,
    'decay_envelope_energy':  envelope[attack_end:decay_end].sum(),  # the ring-sum twin
    'decay_col_min_median_db': decay_col_min_median,                 # the broadband-level twin
}
```

`decay_col_min_median_db` is copied to the per-event dict at
`event_features.py:1164` and persisted into the analysis
sidecar at `midi.py:475` and `midi.py:502` (rounded to 2
decimals).

### 2.7 TL;DR

`decay_col_min_median_db` = **median of `s_db.min(axis=0)`
over the 15 ms decay window after the high-res peak** (in
dB). Computed on a 128/4 high-resolution STFT (vs the
standard 1024/256) so single-frame transients and the 5–15
ms ring are visible. Higher (less negative) = real strike
with broadband ring; near -80 to -90 dB = noise pop.
Companion to `decay_envelope_energy` (the ring's
contrast-summed energy).

---

## 3. How the two calculations relate

The toms PGA filter chain uses both:

1. **Prominence** (`pga_min_prominence`, default 3000) —
   filters FPs that have low peak prominence on the
   broadband contrast envelope. Catches noise events with
   no real attack.
2. **Decay col-min** (`min_decay_col_min_db`, default
   -80.0 dB) — filters FPs that have high peak prominence
   (a real-looking attack) but no sustained ring (a
   single-frame noise pop). Catches the events the
   prominence filter misses.

The two filters are layered: events must pass BOTH to be
KEPT. This is the "two-condition" architecture the user
described — single-condition rules, with the layering
giving the "combinatory" effect without the complexity of
a full combinatory filter layer.

For the future combinatory filter layer (deferred until 5+
stems of calibration data are available), see
[calibration-data.md](../agent-plans/calibration-data.md).
