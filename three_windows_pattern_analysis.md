# Three-Window Pattern Analysis: toms 14-16s, hihat 36-38s (project 4)

**Data CSV:** `user_files/4 - 2_funk_80_beat_4-4_4/three_windows_raw.csv`

**TL;DR — three different bugs, all rooted in the same detector signal design:**

1. **Toms 14-16s**: all 3 GT hits ARE detected, but with timing
   offsets of +98ms, +9ms, +70ms (all positive — detector fires
   LATE, never early). The detector tracks the band_delta PEAK,
   which lags the attack onset by 50-100ms in a tom envelope.
2. **Hihat 36-38s**: 0 of 8 GT hits detected. Complete failure.
   The band_delta signal max for hihat is 0.36, but the detector's
   height threshold is 5.0. The hihat signal is **1000x weaker**
   than the tom signal in band_delta space.
3. **Snare 27-29s**: 2 of 2 GT hits detected (offsets +18ms,
   +24ms — good). 3rd event at 28.69s is the FP echo, 100x weaker
   than the real hits.

The unifying issue: **band_delta = max(per_bin_means) − median(per_bin_means)**
is the wrong signal. It works for toms/kicks (single-band, huge
ratios) but fails for snare (multi-band, ratios bounded near 2)
and hihat (multi-band, low absolute power).

## Toms 14-16s — timing offset (subtle but real)

User GT: 14.25, 14.45, 14.65 (eyeballed from spectrogram, 200ms apart)
Detector found: 14.35, 14.46, 14.72, 14.98
Offsets to nearest GT: **+98ms, +9ms, +70ms** (mean +59ms)

All offsets are POSITIVE — the detector fires AFTER the attack.
Looking at the band_delta signal at each GT:

```
GT 14.25:  band_delta peaks 14.35 (Δ=+100ms) — attack onset is way before the peak
GT 14.45:  band_delta peaks 14.46 (Δ=+10ms)  — well-aligned
GT 14.65:  band_delta peaks 14.70 (Δ=+50ms)  — fires during decay
```

The detector's strike definition = "band_delta PEAK". The strike
peak in a tom envelope is the post-attack buildup, not the attack
onset. The first hit (14.25) is 200ms after a previous strike
(based on 80bpm), so its attack is unambiguous, but the band_delta
peak lags by 100ms. The second hit (14.45) is well-aligned because
the previous hit's decay is still strong and the peak is at the
attack. The third hit (14.65) is offset because its attack
overlaps with the second hit's decay.

**Why the user said "missed entirely"**: the +98ms offset is at the
edge of the 100ms tolerance. In the WebUI, the magenta marker
appears 100ms to the right of the visible attack, which the user
sees as "not detected."

**Fix options (in order of effort):**
1. **Lower n_fft** to 512 or 256 for finer time resolution. With
   n_fft=256, time resolution is 5.8ms, so a 100ms offset would
   be obvious to the user. The trade-off: smaller n_fft = worse
   frequency resolution, so the band edges (60-200, etc.) would
   cover different actual frequencies. The user would need to
   re-tune the band edges.
2. **Backtrack to attack onset** — once a peak is found, walk
   backward in time to find the first frame where band_delta
   crossed some lower threshold (e.g., 50% of peak). This
   recovers the attack onset within a few ms.
3. **Lower the NMS window** to allow more peaks in fast passages.
   Currently 150ms; lower to 100ms. But this risks merging fast
   strikes (200ms apart) into one peak.

## Hihat 36-38s — 0/8 detected (the big one)

User GT: 36.116, 36.302, 36.488, 36.685 (OPEN), 37.08 (closes open), 37.428, 37.649, 37.846
Detector found: 0 events (complete miss)

The band_delta signal IS peaking at every GT hit — the detection
signal works, but the values are too small to clear the find_peaks
thresholds.

**band_delta peak values at each GT hit:**

| GT t    | band_delta peak | B4 at peak (linear power) |
|---------|-----------------|---------------------------|
| 36.116  | 0.103           | 1.13e+00                  |
| 36.302  | 0.032           | 6.79e+00                  |
| 36.488  | 0.027           | 1.10e+00                  |
| 36.685  | 0.360           | 5.09e+01 (the open one)   |
| 37.08   | 0.314           | 7.51e+00 (closes the open)|
| 37.428  | 0.127           | 7.61e+00                  |
| 37.649  | 0.019           | 2.96e+00                  |
| 37.846  | 0.018           | 3.06e+00                  |

Compare to toms 14-16s: band_delta peaks at 264, 313, 791, 18.
**Hihat band_delta is 1000x weaker than toms.**

**Why?**
- band_delta = max(per_bin_means) - median(per_bin_means)
- per_bin_means = band_powers / n_bins  (normalizes for band width)
- For toms: B0 (60-200Hz) has 7 bins at 1024 fft / 44100 sr. Toms
  have ALL their energy in B0 → per_bin_means for B0 is huge
  (300-500), B1-B4 are near zero. max = 500, median = 0.001.
  band_delta = 500.
- For hihat: B4 (2400-8000Hz) has 129 bins (40x wider than B0).
  Even a strong hihat hit only has B4 in the 1-10 range (linear).
  B0/B1/B2/B3 are near zero. per_bin_means for B4 = 10/129 = 0.08.
  max = 0.08, median = 0.001. band_delta = 0.08.

So hihat's per_bin_mean is **6000x smaller** than toms because
(a) hihat has less absolute power and (b) B4 is 40x wider so the
per-bin normalization penalizes it.

**The fundamental issue: per_bin_means is the wrong normalization
for hihat.** The intent was to prevent broadband noise (which is
spread evenly across all bins) from dominating. But hihat's
energy is genuinely concentrated in B4 — it doesn't need
normalization. The normalization is a hammer that only fits the
tob/kick use case.

**Fix options (in order of effort):**
1. **Per-stem prominence/height thresholds** in the config. Hi-hat
   uses `prominence=0.05` and `height=0.01`; toms use the current
   `prominence=2.0` and `height=5.0`. Schema-driven, no algorithm
   change. Trade-off: user needs to know what the right value is
   for each stem.
2. **Switch from per_bin_means to band_powers** for the detection
   signal. Then B4 is on the same scale as B0 (no width penalty).
   At hihat 37.08 peak: B0=9.5e-03, B1=3.08e+00, B2=6.0e-01, B3=4.5e-01,
   B4=8.28e+00. median=6.0e-01, max=8.28, band_delta=7.68. That
   passes height=5.0 cleanly. Trade-off: broadband noise would
   dominate band 4 (40x wider), so this would create new FPs
   unless paired with an absolute power floor.
3. **Combined signal: max(per_bin_means_delta, band_powers_delta).**
   Either signal can fire; the detector finds peaks in the union.
   Catches both hihat (band_powers_delta high) and kicks/toms
   (per_bin_means_delta high) without changing thresholds.
   Most robust but most code.
4. **Per-stem absolute power floor + per-stem delta floor.**
   Each stem has its own band_powers floor (e.g., hihat B4 >= 1.0)
   and its own band_delta floor (e.g., hihat band_delta >= 0.01).
   Schema-driven, configurable per project.

**Recommended**: option 4 with a sensible default per stem type.
Schema: `spectral_min_delta: float = 0.05` (per stem, default tuned
for hihat) and `spectral_min_top_band_power: float = 1.0` (per stem,
linear power floor on the top band).

## Snare 27-29s — already analyzed in detail (see prior writeup)

GT: 2 hits at 27.72 and 28.28 (energy detector ground truth)
Detector: 3 events at 27.74, 28.31, 28.69 — the 3rd is the FP
Fix: per-band absolute power floor (~5e+02 for snare)

## Summary of fixes

Three independent fixes needed:

1. **Snare**: add `spectral_min_top_band_power` per-stem setting,
   default 0 (disabled), suggested 5e+02 for snare. Schema-driven
   single number per stem.

2. **Hihat**: add `spectral_min_delta` per-stem setting, default
   0.5 (current default works for toms/kicks), suggested 0.01 for
   hihat. ALSO `spectral_min_top_band_power` for hihat (~1.0,
   since hihat's B4 power at strike is 1-10 linear). Schema-driven.

3. **Toms timing offset**: either (a) lower n_fft for finer time
   resolution, or (b) backtrack to attack onset, or (c) live with
   +50-100ms timing and add a calibration note. (c) is the cheapest
   but the user said "missed entirely" which means the offset
   hurts the WebUI UX. Recommend (b) — backtrack to attack onset
   in the detector.

All three are schema additions, no algorithm redesign. The data
model is sound; the detector just needs per-stem tuning knobs.

## Calibration targets (project 4)

After applying all three fixes, expected results in 73-77s window:
- toms: 6 GT hits all matched within 30ms (was +98ms max, target +20ms)
- snare: 3-5 events, no FP echoes (was 7, with 1 FP at 28.69s)
- hihat: 8/8 hits detected at 36-38s (was 0/8)
- cymbals: 0-3 events (no change)
- kick: 3 events (no change — already works)

## Files to modify

- `stems_to_midi/spectral_transient_core.py`:
  - Add `min_delta` and `min_top_band_power` to `SpectralTransientConfig`
  - Use `min_top_band_power` as an additional filter after peak-picking
  - Use `min_delta` as the find_peaks `height` parameter (per-stem)
  - Add `_backtrack_to_attack_onset()` helper
- `stems_to_midi/processing_shell.py`:
  - `_build_spectral_config()` reads per-stem min_delta and
    min_top_band_power from config
- `stems_to_midi/settings_schema.py`:
  - Add `spectral_min_delta_per_stem` (dict: stem_name -> float)
  - Add `spectral_min_top_band_power_per_stem` (dict: stem_name -> float)
- `stems_to_midi/cli_builder.py`:
  - Schema-driven CLI flags for both new settings
- `webui/templates/index.html`:
  - Add UI controls for per-stem spectral thresholds
