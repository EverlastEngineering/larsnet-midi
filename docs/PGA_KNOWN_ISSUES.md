# PGA Detector Known Issues

**Status**: PGA detector (`stems_to_midi/percentile_gated_detector.py`) and feature
extraction (`stems_to_midi/event_features.py`) are wired end-to-end and producing
useful diagnostic data. The duration feature is the strongest current discriminator
(29ms for the 14.97s "soft hit" vs 145-203ms for real toms strikes, a 4-7x ratio).
**However, several known issues remain that affect filter reliability.** This doc
captures them so they aren't lost between sessions.

---

## Issue 1: `attack_rise_ms` is unreliable in tight fills

### What we see

The 5 events in the 14.25-14.97s toms fill:
- Strike 1 (14.25s): rise = 23ms — real
- Strike 2 (14.44s): rise = **203ms** — *would be flagged as FP by a 200ms cutoff, but it's a real strike*
- Strike 3 (14.62s): rise = 17ms — real
- Soft hit 1 (14.84s): rise = 610ms — actual FP
- Soft hit 2 (14.97s): rise = 714ms — actual FP

The strike 2 measurement is bad because the previous strike's ring is still loud
in the broadband envelope when the algorithm tries to find strike 2's attack peak.
The 10% point of strike 2's envelope is dominated by strike 1's residual energy,
not the actual noise floor.

### Same problem at 74-76s

Six rapidly hit toms with increasing volume in this region. Strike 3 has rise ≈
200ms, strike 4 has rise = 23ms. We're getting noise in the detector and
**sometimes getting lucky or unlucky** based on what residual ringing happens to
be in the 30ms before each strike.

### Why the proposed fix doesn't work yet

Setting `pga_max_attack_rise_ms = 200` to filter FPs would also kill the real
strike 2 at 14.44s (rise=203ms). Until the rise measurement is robust to
preceding ringing, this filter is unsafe.

---

## Issue 2: The robust "attack detector" we still need

### Proposed approach: "attack spike vs local noise floor"

The user identified the right signal:

1. Look at 2-3 STFT frames BEFORE the reported event time. This is the
   **local baseline** / **local noise floor** at the event.
2. Look at 2-3 STFT frames AFTER the reported event time (or right at
   the peak). This is the **attack spike**.
3. Compute the **delta** = (mean energy in 2-3 frames after) - (mean
   energy in 2-3 frames before). This is the per-event "attack
   contrast."
4. **Discriminator**: a real percussive attack has a LARGE delta
   (energy jumps up dramatically from the local noise floor). A
   false positive (e.g. a click, a soft hit on top of ringing, a
   noise pop) has a SMALL delta — the energy is similar before and
   after, because the "event" is just a momentary fluctuation in an
   already-active envelope.

The delta should be measured in a **specific frequency band**, not
broadband. The user specifically said: "This is especially easy to
spot below 2000Hz." Below 2000Hz is the snare/tom body band. Saturated
low bands (0-200Hz) are excluded, so the band is 200-2000Hz.

### Why this is better than attack_rise

The current `attack_rise_ms` measures the time from 10% to 90% of the
peak, where the peak is the LOCAL maximum in a ±50ms forward search.
If the peak is in a sustained ringing region (not a sharp attack),
the "rise" measurement conflates the true attack rise with the
preceding ringing's slow decline. The result: a real strike in a tight
fill can have a huge "rise" measurement.

The "attack delta" approach explicitly compares **before vs after** the
event, so it can't be fooled by sustained ringing. Even if the peak
is in a ringing region, the delta is small because the energy was
already loud before AND is loud after.

### Expected discriminator values (untested, will calibrate)

- Real strike in tight fill: delta > some threshold (e.g. 3-5x local
  baseline in 200-2000Hz band)
- Real strike isolated: delta > 10x local baseline
- Click / soft hit: delta < 2x local baseline
- Sustained ringing: delta ≈ 1x (no change)

The 2-3 frame window is tight (~12-17ms at hop=256, sr=44100). This
matches the user's "TIGHT tolerance but accurate" observation.

### Open question: how to handle the "attack peak" position

The current `compute_attack_rise_ms` searches for the envelope peak in
[event_time - 30ms, event_time + 50ms]. For the proposed delta, do we
use the envelope value at the reported event_time, or at the peak
position? Probably the peak — but with the same forward-only search
to avoid latching onto the previous strike.

---

## What to do next (not done yet)

1. **Implement `compute_attack_delta`** in `event_features.py`:
   - 200-2000Hz band envelope (not the current 200-8000Hz)
   - 3 frames before, 3 frames after
   - delta = mean(after) / mean(before) [linear ratio]
   - Returns the ratio; the user sets the threshold (default ~3.0?)

2. **Wire it into the sidecar + tooltip** as a new field.

3. **Test on the 14.25-14.97 fill and 74-76 region** to verify
   strike 2 (rise=203) gets a high attack_delta, and the 14.84/14.97
   soft hits get a low attack_delta. If the data confirms the
   hypothesis, use attack_delta as the PRIMARY filter (replacing or
   complementing the prominence filter), and deprecate
   `pga_max_attack_rise_ms`.

4. **Update this doc** with the actual measured delta values and
   the recommended threshold.

---

## Related: duration metric IS reliable

Despite the attack_rise issues, the `duration_ms` metric is a strong
discriminator. From the 14.25-14.97s data:

- Real strikes: duration 145-203ms (slope-of-decline ring time)
- Soft hits (FPs): duration 29-64ms

That's a 4-7x ratio. A `pga_min_duration_ms` filter (e.g. require
duration >= 50ms) would cleanly kill the soft hits while keeping all
real strikes. The duration measurement is robust because it's based
on the slope-of-decline, not the absolute envelope level — so
preceding ringing doesn't trick it the way it tricks attack_rise.

**Recommendation**: add `pga_min_duration_ms` to midiconfig.yaml as a
configurable filter (default 50ms) before tackling the attack_delta
work. This is a 5-line change vs the multi-day attack_delta work.

---

## Issue 3: Per-stem `broad_min_hz` override not implemented

The user identified early on that toms (200-8000Hz) and kick (30-200Hz)
need different envelope bands for accurate feature extraction. The
feature code accepts a `broad_min_hz` / `broad_max_hz` arg, but the
processor always uses 200-8000Hz. For per-stem accuracy, we need to
look up `onset_detection.pga_freq_band_min_hz` /
`pga_freq_band_max_hz` per stem_type.

This is a small wiring change. Save it for after the attack_delta work
above.
