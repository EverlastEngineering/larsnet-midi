# Snare 27-29s Pattern Analysis (project 4)

**Question:** Why did the spectral detector find 3 events in snare 27-29s
when there are only 2 real hits? The 3rd event is an echo, barely audible.

**TL;DR:** The current `top/2nd ratio` signal doesn't distinguish a
loud snare hit from a quiet snare echo because **snare has spectral
energy in TWO bands (B0 + B1)**, so the ratio is bounded near 2
regardless of absolute loudness. The ratio works for toms/kicks
because those have energy in ONLY ONE band (B0); the other 4 bands
are near-silence, so the ratio is huge (100s-1000s). For snare,
you need a **per-band absolute power floor** instead.

The CSV with all the raw numbers is at
`user_files/4 - 2_funk_80_beat_4-4_4/snare_27_29s_raw.csv`.

## The raw data (snare 27-29s)

| t (s) | label          | B0 (60-200Hz) | B1 (200-600Hz) | B0/B1 | total | top/2nd |
|-------|----------------|---------------|----------------|-------|-------|---------|
| 27.74 | KNOWN_HIT      | 4.09e+03      | 2.05e+03       | 1.99  | 6.74e+03 | 1.99  |
| 28.31 | KNOWN_HIT      | 3.10e+03      | 1.46e+03       | 2.13  | 4.93e+03 | 2.13  |
| **28.69** | **FALSE_POSITIVE** | **3.01e+01** | **1.29e+01** | **2.33** | **4.63e+01** | **2.33** |

The B0/B1 ratio is **1.99, 2.13, 2.33** for the 3 events. The
false positive has the **highest** ratio of the three — exactly
opposite of what you'd want. The ratio goes UP as the absolute
power goes DOWN.

The energy detector (ground truth) confirmed only 2 hits:
- t=27.72, geomean=990, vel=92
- t=28.28, geomean=827, vel=90

## The pattern

The 28.69s event is a **38ms echo of the 28.31s hit** (28.69 - 28.31
= 0.38s — could be a snare buzz/ring, a room reflection, or a
comping artifact). The echo's spectral shape is **identical** to
the original hit (proportionally: B0/B1 ≈ 2, B2-B4 each ~5% of B1).
It's just scaled down by ~100x in absolute power.

**Key insight:** snare body energy spans TWO bands (60-200Hz and
200-600Hz). When a snare decays, ALL bands decay together at the
same rate, so the B0/B1 ratio is preserved. The same is true for
the top/2nd ratio: it's a SHAPE measurement, not a SIZE measurement.
For instruments that put energy in only one band, shape-vs-size are
the same signal (high ratio = loud hit). For instruments that span
multiple bands, shape and size decouple.

## Why the ratio works for toms and kicks (and not snare)

| stem   | B0       | B1       | B2-B4 each  | top/2nd (real hit)  | top/2nd (echo)     |
|--------|----------|----------|-------------|---------------------|---------------------|
| kick   | 3.0e+03  | 5.3e+00  | <1e+00      | **566**             | ~566 (same shape)  |
| toms   | 7.6e+02  | 1.8e+01  | <1e-01      | **42**              | ~42 (same shape)   |
| snare  | 4.1e+03  | 2.1e+03  | ~1.5e+02    | **2.0**             | **2.3** (similar!) |

Kicks and toms put ~99% of their energy in B0 and ~1% in B1-B4.
So `B0/B1` is huge (100-1000x). A real hit lights up B0 strongly
and B1-B4 stay quiet. A decay/echo would also have high B0/B1
because the SHAPE is the same — but the absolute B0 is much lower
than a real hit.

For kicks/toms, the "B1-B4 near zero" floor acts as a noise floor
that naturally suppresses quiet decay frames (because the noise
floor kicks in before B0 gets small enough to matter).

For snare, B0 and B1 are both "real" — there's no 1% floor. So the
ratio can't distinguish loud from quiet.

## What WOULD work for snare

A per-band absolute power floor. Concretely: drop any spectral
event where `top_power < ABSOLUTE_FLOOR`. Looking at the data:

- Real snare hit 1: B0 = 4.09e+03
- Real snare hit 2: B0 = 3.10e+03
- FP echo:         B0 = 3.01e+01

A floor of `top_power >= 5e+02` (500) would:
- Keep both real hits (B0 = 3100 and 4090) ✓
- Drop the echo (B0 = 30) ✓
- Have ~5x headroom for quieter-but-real snare hits

This floor is per-band, applied to the top band's power (the band
with the most energy at the strike moment). The threshold would be
in linear power (not dB) since the band_powers are already in linear
sum units.

## Why this is "good enough" for kicks and toms without a separate floor

For kicks/toms, the existing ratio filter (1.0) is already filtering
out low-power decay frames because the decay B0 value is still well
above the noise floor AND the ratio is similar to a real hit. So the
ratio-based quality signal is sufficient for those stems. Adding an
absolute floor would be redundant noise.

For snare, the absolute floor is the missing piece. It should be
applied **additionally** to the ratio filter, not as a replacement.

## Recommended implementation

In `_run_spectral_detection` (or wherever the quality floor lives),
add a per-stem absolute power threshold. The cleanest path:

1. Add a new schema setting `spectral_min_top_band_power` (linear,
   default 0 — disabled). Per-stem override possible.
2. The `SpectralTransientConfig` gets a new field `min_top_band_power`
   (default 0.0, backwards compatible).
3. After peak-picking, drop events where
   `event.band_powers[event.band_max_idx] < config.min_top_band_power`.
4. UI exposes a single slider per stem in the spectral section.

Alternative (simpler, less granular): hardcode a per-stem-type
floor based on calibration. Snare: 5e+02. Other stems: 0 (rely on
ratio). This ships faster but doesn't let the user tune per project.

I'd recommend the schema-driven version — it generalizes to any
future stem type without a code change.

## Calibration numbers to validate against

For project 4 (funk track, 80bpm):
- All known snare hits in 73-77s: B0 = 1.0e+03 to 5.0e+03 (5 hits)
- All known snare FPs (e.g. 28.69s echo): B0 < 1.0e+02

So a floor of `5e+02` would:
- Drop the 28.69s echo ✓
- Keep the 28.69s echo from polluting MIDI ✓
- Keep all 5 real snare hits in 73-77s ✓

This needs validation on more tracks. Recommend running on project 1
(the user's first funk track) and project 3 to confirm.
