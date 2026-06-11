# Where to adjust toms spectral detection settings

This is your reference for tuning toms detection in the WebUI. The
detector has two signals that fire independently and get unioned:

| Signal | Formula | Fires on | Default `min_*` |
|--------|---------|----------|------------------|
| **Ring** (was: only signal) | `max(per_bin_means) − median(per_bin_means)` over **all 5 bands** | per-band-dominant content (e.g., B0-dominant toms ring) | `min_band_ratio = 2.0` |
| **Snap** (new) | `min(per_bin_means)` over the **per-stem snap_bands** | broadband content in the snap range (e.g., B1+B2 for toms attack onset) | `snap_min_delta = 0.05` |

Both signals always run. The result is the union of the events they
find, with a 50ms merge window that prefers the snap time when both
fire near the same time. The wire-tail filter then drops ring
"tails" of snap events within 100ms.

## Per-stem settings (2026-06-09)

All settings live in `<project>/midiconfig.yaml` under each stem's
section. The schema (webui/settings_schema.py) drives the WebUI
form and CLI flags.

### toms (project 4 — calibrated values, 2026-06-09)

```yaml
toms:
  spectral_snap_bands: '1,2'      # B1 (200-600Hz) + B2 (600-1200Hz)
                                  # The "head snap" range — toms
                                  # attacks are broadband here.
  spectral_snap_min_delta: 0.05   # find_peaks height for the snap
                                  # signal. Lower = more sensitive.
```

Calibration results on project 4 funk track (toms 14-16s with
spectral_snap_bands: '1,2'):

| GT (s) | Detected (s) | Offset | Real? |
|--------|--------------|--------|-------|
| 14.243 | 14.269 | +26ms | ✓ |
| 14.441 | 14.449 | +8ms | ✓ |
| 14.626 | 14.640 | +14ms | ✓ |
| — | 14.901 | — | ⚠ toms decay tail, not a real hit |

Toms 73-77s (6 GT hits): all matched within 100ms. Some late-decay
events (~5 FPs) remain in 73-77s because the toms envelope has a long
ring that the detector captures.

## What you can tune

| Setting | Effect | Try this if... |
|---------|--------|-----------------|
| `spectral_snap_bands` | Which bands the SNAP signal uses | All 5 = legacy behavior (BROADBAND signal becomes noisy). (1, 2) = toms head snap. (3, 4) = cymbal-snap (mostly empty for toms). (0,) = single-band magnitude in B0. |
| `spectral_snap_min_delta` | find_peaks height for the snap signal | Lower = more snap events fire. Higher = only the strongest snaps. Default 0.05. Try 0.5 to be very strict. |

> **Note on `detection_method`**: This is for an older experimental
> method path, not the current detector. The two signals above
> (RING + SNAP) always run together and their events are unioned —
> there's no `detection_method` toggle that affects whether they
> fire. Ignore `detection_method` in `midiconfig.yaml`; it doesn't
> do anything in the current code path.

## Masking snap_delta = 0 events in the tuning view AND saved MIDI (2026-06-09)

The toms tuning panel has a **Snap Δ Mask** slider (range 0–0.5,
schema default 0.001). It's a 4th filter pass that runs after the
existing spectral / reverb / sustain filters and marks any KEPT event
with `snap_delta ≤ threshold` as `FILTERED`. The mask is INCLUSIVE —
threshold=0 filters all `snap_delta==0` events; threshold=0.05
filters everything ≤ 0.05. The slider is also forwarded via
`_buildConfigOverrides` to the server, where the same filter runs
in `_build_events_configured` (full pipeline) and
`rebuild_core._apply_snap_mask` (rebuild path), so the saved MIDI
excludes the masked events.

Why this is useful: ring-only events (high `band_delta`, low
`snap_delta`) are the typical false-positive signature — wire tails,
decay events, and post-attack ring. The 14.901s FP in 14-16s has
`Snap Δ = 0.0` and `Ring Δ = 58.9`. The default 0.001 threshold
filters this event out (and all other zero-snap events) out of the
box. Set the slider to 0 to keep all zero-snap events; set higher
(0.05, 0.1) to keep only events with strong broadband attack; set
to a negative value to disable the mask entirely.

**To use in the WebUI:**

1. Open project 4 in the WebUI
2. Click the **Tune** button to open the tuning panel
3. Switch to the **toms** stem
4. Move the **Snap Δ Mask** slider — the tuning view updates
   instantly, and clicking **Save** writes the threshold to the
   project YAML (`toms.snap_mask_threshold`) and rebuilds the MIDI
   with the mask applied
5. The events that newly appear as red are the ring-only false
   positives

**Schema / config:**

```yaml
toms:
  snap_mask_threshold: 0.001   # 0 = kill only snap_delta==0; negative = off
```

```bash
python -m stems_to_midi.cli --toms-snap-mask 0.05   # CLI override
```

## What you can read in the WebUI tooltip

Hover a magenta (spectral) event in the WebUI. The tooltip shows:

```
Time: ...
Status: KEPT
Method: spectral (magenta)
B0 60-200Hz *: 2.52e+02  (the * marks the top band)
B1 200-600Hz  : 7.04e+01
B2 600-1200Hz : 4.94e+01
B3 1200-2400Hz: 1.68e+01
B4 2400-8000Hz: 8.75e+00
Top band: B0
Top/2nd ratio: 3.58 (higher = clearer strike)
Ring Δ (max-median, all bands): 80.46   ← RING signal value at this frame
Snap Δ (min of snap_bands): 0.5987      ← SNAP signal value at this frame
Strength (ratio/10): 0.36
```

**How to read the two deltas:**

- If `Ring Δ` is high AND `Snap Δ` is low → the event was detected by
  the RING signal only (it's a per-band-dominant frame, like a decay).
  These are usually FPs.
- If `Snap Δ` is high → the event was detected by the SNAP signal
  (broadband content in the snap_bands). These are the real attacks.
- If both are high → both signals fired, merged with snap time preferred.

The 14.901s FP in 14-16s has `Snap Δ = 0.0` and `Ring Δ = 58.9` —
that's the signature of a RING-only detection (a slow toms decay
where B0 is still relatively loud but B1+B2 are dead).

## Per-frame silence mask (2026-06-09)

A pure helper in `stems_to_midi.analysis_core.spectral_utils.compute_silence_mask`
computes a per-frame boolean mask (active vs silent) from a 2D
magnitude spectrogram. The threshold is calibrated from the
**noise band** — the P5-P30 percentile of per-frame energy — using
`median + 2.5 * std`. This isolates the true background noise
floor and avoids two failure modes that bite on real drum tracks:

1. **Compressor dropouts** (long stretches of very-low-energy
   frames) drag a naive min+std threshold down to the dropout
   level, marking legitimate low-level audio as silence.
2. **High-energy transients** create a long right tail in the
   per-frame energy distribution. A naive median climbs with hit
   density and ends up above quieter hits.

The P5-P30 band sits in the noise floor (always present, never a
hit), so the threshold is invariant to hit density and to
dropout level.

**Validated against project 4 toms GT (2026-06-09):**

- 14-16s region: 3/3 hits caught on active frames
  (14.243, 14.441, 14.626)
- 73-77s region: 6/6 hits caught on active frames
  (73.676, 73.853, 74.033, 74.210, 74.411, 74.576)

**How to use it as a guard signal for low-snap-delta FPs:**

The silence mask alone doesn't distinguish a real hit from a
decay-tail FP (both have frame energy orders of magnitude above
the threshold). But the *ratio* of event energy to threshold is
a useful third signal alongside `band_delta` and `snap_delta`:

- Real hit: high band_delta, high snap_delta, **high energy/threshold ratio**
- Decay-tail FP: high band_delta, low snap_delta, **moderate energy/threshold ratio**

Downstream code can drop events whose energy ratio is below a
calibrated cutoff (typical toms: real hits ≫ 1M×, FPs ≈ 0.5M×).

**API:**

```python
from stems_to_midi.analysis_core.spectral_utils import compute_silence_mask
import librosa

spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))  # (n_freq, n_frames)
mask = compute_silence_mask(spec)  # (n_frames,) bool
active_frames = np.where(mask)[0]
```

## Settings still TODO

1. **Per-stem `spectral_min_top_band_power`** (absolute power floor) — would drop the 14.901s FP and similar decay events. Not yet implemented. Would require per-stem calibration (e.g., toms: 200, kick: 1000, snare: 500).
2. **Longer wire-tail filter window** for same-band tails (300-400ms instead of 100ms) — would catch the 14.901s FP as a tail of 14.640s. Would also affect 73-77s (might merge distinct strikes if they're <400ms apart — but toms at 80bpm are 750ms apart, so safe).
3. **Schemas for hihat/snare/kick/cymbals snap settings** — only toms has the schema entries right now. Same pattern, different snap_bands per stem.

## Per-frame CSV dump (2026-06-09)

To see **every STFT frame** (not just the peak-picked events) for the
toms stem, run:

```bash
/Users/jasoncopp/miniforge3/envs/drumtomidi/bin/python \
    scripts/dump_snap_per_frame.py \
    --project 4 \
    --stem toms
```

This writes:
- `user_files/4 - 2_funk_80_beat_4-4_4/analysis/snap_per_frame_toms.csv` (one row per ~5.8ms frame, ~15,000 rows for 87s of audio)
- `user_files/4 - 2_funk_80_beat_4-4_4/analysis/snap_per_frame_toms.json` (sidecar: config + summary stats)

The CSV columns are:

| Column | What it is |
|--------|------------|
| `time_sec`, `frame_idx` | Time in seconds, frame index |
| `band0_per_bin_mean` ... `band4_per_bin_mean` | Per-bin mean linear power in each of the 5 fixed bands (60-200, 200-600, 600-1200, 1200-2400, 2400-8000 Hz) |
| `band_delta` | RING signal: `max(per_bin_means) − median(per_bin_means)` over all 5 bands |
| `ring_pass_height` | The `min_band_ratio` threshold (2.0 by default) — frames with `band_delta >= this` are RING candidates |
| `snap_band_<i>_per_bin_mean` (one column per snap_bands entry) | The per-bin mean in each snap band — e.g., `snap_bands='2,3'` gives 2 columns |
| `snap_delta` | SNAP signal: `min(per_bin_means[snap_bands])` |
| `snap_pass_height` | The `snap_min_delta` threshold (0.0101 calibrated for project 4) |
| `band_max_idx`, `band_max_ratio`, `max_db` | Per-frame classification signals |
| `snap_bands` | The configured snap_bands (echoed for portability) |
| `is_event_peak` | 1 if this frame was chosen as a detector event, 0 otherwise |
| `event_time_sec`, `event_band_delta`, `event_snap_delta` | If `is_event_peak==1`, the event's time and the signal values at the event frame (else empty) |

**Useful sorts/filters:**

- Sort by `snap_delta` DESC — see all frames where the SNAP signal is loudest (these are the real attacks)
- Filter `is_event_peak == 1` — see only the peak-picked event frames
- Sort by `band_delta` DESC, `snap_delta` ASC — see ring-fired events with no snap confirmation (the FPs)
- Filter `snap_delta < 0.01` — see all frames where the snap was essentially silent (decay / wire-tail / silence)

**To A/B different `spectral_snap_bands` settings:** edit
`user_files/4 - 2_funk_80_beat_4-4_4/midiconfig.yaml` and change
`spectral_snap_bands: '2,3'` to `'1,2'` or `'1,2,3'`, then re-run
the script. The new CSV will have different `snap_band_*` columns
and the `snap_delta` values will be computed from the new band set.
The sidecar JSON records which `spectral_snap_bands` was active, so
the comparison is unambiguous.
