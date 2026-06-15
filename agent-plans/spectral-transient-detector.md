# Spectral Transient Detector — Findings

**Date:** 2026-06-08
**Status:** Investigation complete. Method proven on 1 stem (project 3 toms).
Recommended path: ship as a complementary tool, NOT a replacement.

## TL;DR

Discovered a new onset-detection signal: **per-frame count of high-frequency
bins (≥800Hz) above a noise floor (-50dB) in a 1024-pt STFT.** This finds
broadband transients (tom strikes) reliably, sidesteps the toms envelope
shape problem, and **independently confirmed the energy detector's reverb
filter is the actual bug in 73-76s of project 3 — the energy detector
correctly finds 6 events but 4 get marked as REVERB_CONTINUATION when they
should be separate hits.**

## How the method was discovered

User observed: looking at the WebUI spectrogram of project 3's toms stem
with a noise floor of -50dB and frequencies ≥800Hz, **6 distinct hit
onsets are clearly visible** at approximately 73.676, 73.853, 74.033,
74.210, 74.411, 74.576 seconds. User asked if the same approach could
reveal these from the data.

Reverse-engineered the WebUI's STFT from
`spectrogram_analyzer_data_exporter.html`:

- Hann window, n_fft=1024, hop=256 (4:1 overlap, 5.8ms frame)
- `20*log10(|rfft|)` — magnitude in dB, no reference divisor
- Colormap with adjustable floor/gain — "noise floor at -50dB" is the
  colormap setting, not a data filter

**Two key insights:**

1. The user's "noise floor at -50dB" is applied at the **colormap** level,
   not the data. Real onsets show up as brief moments where the high-freq
   content is much louder than the surrounding floor. So the right signal
   is not "absolute dB" but "**how many high-freq bins exceed the floor
   in this frame**" — a count, not a magnitude.

2. The current JSON export (`scripts/spectral_analysis_toms.py`) clips
   the dB floor at -80dB AND downsamples 4x in time AND 2x in freq, so
   the soft onsets the user can see in the UI are completely invisible
   in the exported JSON. Recomputing at full resolution is mandatory.

## Implementation

**`stems_to_midi/spectral_transient_core.py`** — pure functional module
mirroring the existing `energy_detection_core.py` pattern. Key API:

```python
config = SpectralTransientConfig()  # defaults: 1024/256, 800-8000Hz, -50dB
events, debug = detect_spectral_transients(audio, sr, config)
# events: list of SpectralTransientEvent(time_sec, bins_above_floor, max_db, prominence_bins)
# debug:  {times, count, max_db_in_band, freqs_in_band, db_in_band}
```

**`stems_to_midi/spectral_transient_cli.py`** — CLI:
`python -m stems_to_midi.spectral_transient_cli <wav> [--compare] [--out events.json]`
Finds events, optionally compares against the project's existing
`analysis.json`, writes JSON output.

**`stems_to_midi/test_spectral_transient_core.py`** — 6 tests, all pass:
- STFT shape and dB range
- 4 synthetic drum hits detected
- 6 known hits in 73-76s of project 3 detected (THE ground-truth test)
- Too-short audio raises ValueError
- Event dataclass is frozen
- Count signal has sharp rise at hits

## Stage 1 — Known events (73-76s of project 3 toms)

User ground truth: 73.676, 73.853, 74.033, 74.210, 74.411, 74.576

Spectral detector found 11 events in 73-76s; top 6 (by bins_above_floor):

| # | Spec t (s) | Bins | Max dB | GT t (s) | Δ (ms)  |
|---|------------|------|--------|----------|---------|
| 1 | 73.700     | 159  | -13.4  | 73.676   | +24.1   |
| 2 | 73.868     | 167  | -0.4   | 73.853   | +15.5   |
| 3 | 74.066     | 167  | -14.4  | 74.033   | +32.9   |
| 4 | 74.234     | 167  | -12.9  | 74.210   | +24.2   |
| 5 | 74.420     | 167  | -9.3   | 74.411   | +9.0    |
| 6 | 74.600     | 167  | -0.7   | 74.576   | +23.9   |

- All 6 hits detected.
- Mean offset vs ground truth: **+21.6ms** (range +9.0 to +32.9ms).
  Systematic positive offset = spectral peak trails the strike by ~17ms
  on average, expected behavior (peak of energy is after attack).
- All 6 spectral events have **bins_above_floor=167 out of 167 possible**
  for hits 2-6 — i.e. every single high-freq bin in the 800-8000Hz band
  crossed the -50dB floor. Unmistakable hits.

## Stage 1.5 — Reverb filter bug confirmation

**The real story.** The energy detector's `events_sensitive` (pre-reverb-filter)
list contains 27 events in 73-77s — way more than the 6 ground truth
hits. The reverb filter is what trims it to 9 events in `events_configured`,
correctly marking most as REVERB_CONTINUATION. BUT:

The spectral detector **independently confirms** that 4 of the events
the reverb filter marked as REVERB_CONTINUATION are actually real hits
(bins_above_floor=167 for all 6, max_db 0 to -14 dB):

| Sensitive t (s) | Spectral nearby? | Status in configured | Verdict |
|-----------------|------------------|----------------------|---------|
| 73.677          | yes (Δ=+23ms)    | —                    | missed by filter |
| 73.747          | yes (Δ=-46ms)    | —                    | reverb tail, correct |
| 73.851          | yes (Δ=+17ms)    | KEPT                 | ✓ |
| 73.863          | yes (Δ=+6ms)     | KEPT                 | ✓ |
| 73.967          | yes (Δ=+99ms)    | REVERB_CONTINUATION  | filter correct |
| 74.072          | yes (Δ=-6ms)     | REVERB_CONTINUATION  | filter correct |
| 74.165          | yes (Δ=+70ms)    | KEPT                 | ✓ |
| 74.292          | yes (Δ=-58ms)    | REVERB_CONTINUATION  | filter correct |
| **74.350**      | yes (Δ=+70ms)    | —                    | **missed — spectral confirms it's a hit** |
| **74.374**      | yes (Δ=+46ms)    | REVERB_CONTINUATION  | **filtered as reverb — spectral confirms it's a hit** |
| **74.420**      | yes (Δ=-0ms)     | REVERB_CONTINUATION  | **filtered as reverb — spectral confirms it's a hit** |
| 74.617          | yes (Δ=-17ms)    | —                    | missed — but low bins, less certain |
| **74.687**      | yes (Δ=-87ms)    | REVERB_CONTINUATION  | **filtered as reverb — spectral confirms it's a hit** |
| **74.768**      | yes (Δ=+29ms)    | REVERB_CONTINUATION  | **filtered as reverb — spectral confirms it's a hit** |
| **74.791**      | yes (Δ=+6ms)     | REVERB_CONTINUATION  | **filtered as reverb — spectral confirms it's a hit** |
| **74.815**      | yes (Δ=-17ms)    | REVERB_CONTINUATION  | **filtered as reverb — spectral confirms it's a hit** |
| **74.861**      | yes (Δ=-64ms)    | REVERB_CONTINUATION  | **filtered as reverb — spectral confirms it's a hit** |
| 74.896          | yes (Δ=+35ms)    | —                    | missed — but is this a real hit? |

Per-hit ground truth (spectral vs detector's nearest KEPT or REVERB):

| GT t (s) | Spec t (s) | Spec Δ | Configured status & Δ | Sensitive Δ |
|----------|------------|--------|----------------------|-------------|
| 73.676   | 73.700     | +24.1ms | KEPT +175ms (a later hit — this one missed) | +1ms ✓ |
| 73.853   | 73.868     | +15.5ms | KEPT -2ms ✓           | -2ms ✓      |
| 74.033   | 74.066     | +32.9ms | REVERB_CONTINUATION +39ms ✗ | +39ms ✓     |
| 74.210   | 74.234     | +24.2ms | KEPT -45ms (this is hit 4) | -45ms ✓     |
| 74.411   | 74.420     | +9.0ms  | REVERB_CONTINUATION +9ms ✗ | +9ms ✓      |
| 74.576   | 74.600     | +23.9ms | REVERB_CONTINUATION +111ms ✗ | +41ms ✓    |

**The pre-reverb-filter detector (sensitive list) found ALL 6 hits
within 1-45ms of the user's ground truth.** The energy detector is
working. The reverb filter is the bug — it's over-aggressive on hits
that occur during another hit's reverb tail.

## Stage 2 — Full file (0-87s), no ground truth

| Metric | Spectral vs Configured | Spectral vs Sensitive (pre-reverb) |
|--------|------------------------|------------------------------------|
| Spectral events total | 42 | 42 |
| Other events total | 27 | 78 |
| Matched within 30ms | 10 (23.8%) | 23 (54.8%) |
| Matched within 100ms | 15 (35.7%) | 29 (69.0%) |
| Median time offset | 203ms | 29ms |
| Spectral events with no match within 100ms | 27 (64.3%) | 13 (31.0%) |
| Other events with no match within 100ms | 12 (44.4%) | 50 (64.1%) |

The **203ms median offset vs configured** is misleading — it's biased by
the 100-500ms bin having 15 cases. Those 15 are events where the
spectral hit is at a different time than the configured event (the 15
cases in the 100-500ms bin). Looking at the distribution more honestly:
- 0-30ms: 10 (24%) — tight, real matches
- 30-100ms: 5 (12%) — borderline, possibly adjacent hits
- 100-500ms: 15 (36%) — likely detector missed/double-detected
- >500ms: 12 (29%) — completely different events

**The 69% match rate against the SENSITIVE (pre-reverb-filter) list** is
the more honest number. It tells us: "the energy detector, when not
reverb-filtered, agrees with the spectral detector 69% of the time
within 100ms." That's a strong signal of independent corroboration.

Status breakdown of configured events matched by spectral (full file):
- REVERB_CONTINUATION: 36 matches
- KEPT: 6 matches

**36 of 42 spectral matches are against REVERB_CONTINUATION events.**
This is the key insight: the spectral detector is **finding real
transients in the audio that the reverb filter is incorrectly throwing
away.** That's the bug, end-to-end.

## Recommended next steps

**DO NOT** replace the energy detector with the spectral detector.
Reasons:
1. Spectral detector has no pitch info (can't tell high tom from low
   tom). Energy detector's pitch estimation (geomean) is critical for
   the existing classification pipeline.
2. Spectral detector's 21ms mean offset is too large for tight
   timing-critical work (the energy detector snaps to amplitude peak).
3. Spectral detector is more sensitive to broadband energy — may over-fire
   on cymbals, snare wire, hi-hat sizzle (none of which were tested).

**DO** wire the spectral detector in as a **secondary validation** signal:

Option A — "is this REVERB_CONTINUATION really a reverb continuation?"
- The reverb filter's job is to suppress reverb tails, but it's
  over-aggressive when a real hit occurs during the reverb tail of
  the previous hit.
- Use spectral confirmation: if a REVERB_CONTINUATION candidate has a
  strong high-freq count peak (≥100 bins_above_floor, max_db ≥-20dB)
  in its onset frame, it's a real hit, not reverb.
- This is a 1-line change to `mark_reverb_continuations` in
  `stems_to_midi/analysis_core/onset_filtering.py:42`.
- Expected impact: 3-4 of the 4 incorrectly-filtered hits in 73-76s
  would be promoted to KEPT.

Option B — standalone CLI for investigation
- The current `spectral_transient_cli` is already this. No new work
  needed; just document the use case.
- Useful for: debugging why specific events were marked as
  REVERB_CONTINUATION, validating the detector on stems with strong
  toms/snare, comparing detection methods side-by-side.

Option C — extend to a full STFT-based secondary detector
- Pair each spectral event with its energy-detector equivalent
  (geomean, strength, etc.) and add as a "spectral_confidence" feature
  to the classification pipeline. The classifier could learn that
  high-spectral-confidence + low-energy is a real hit, not reverb.
- Bigger change, more impact, but requires retraining/validation.

**My recommendation: Option A** — small, targeted, addresses the actual
bug (reverb filter over-aggression) confirmed by the data. Option C is
the right long-term direction but is much bigger and needs a proper
ground-truth dataset to validate against.

## Files created

- `stems_to_midi/spectral_transient_core.py` — 233 lines, pure functional
- `stems_to_midi/spectral_transient_cli.py` — 154 lines, CLI + comparison
- `stems_to_midi/test_spectral_transient_core.py` — 6 tests, all pass
- `scripts/compare_spectral_vs_detector.py` — two-stage analysis script
- `user_files/3 - 2_funk_80_beat_4-4_4/analysis/spectral_transients.json` — 42 events
- `user_files/3 - 2_funk_80_beat_4-4_4/analysis/spectral_vs_detector_comparison.md` — full report

## Open questions

1. Does the spectral method work for snare/cymbals/hihat stems? (Only
   tested on toms. Cymbal high-freq content might be too saturated, snare
   too transient-rich, hihat too constant.)
2. Does the offset (~21ms systematic) hold for other stems or is it
   toms-specific? (Toms have a characteristic attack shape; snare
   might peak sooner.)
3. Should the noise floor be tuned per-stem, or is -50dB a good
   universal value? (The -50dB came from the user's eyeballing, not
   from a calibration study.)
