# PGA Prominence: How It's Calculated and Why It Can Be Misleading

**Date**: 2026-06-22
**Status**: Source of truth for the PGA `prominence` value attached to every
event in the analysis sidecar. Future maintainers should not have to dig through
chat history to find this algorithm — it is documented here.

This file documents the per-event `prominence` value that drives the
`pga_min_prominence` filter (the slider in the WebUI tuning panel that drops
events whose prominence is below a threshold). It is the source of truth for
how prominence is computed, what it actually measures, and a known failure mode
where prominence can be very low even for real, strong strikes.

A companion document,
[`stems_toms_prominence_and_decay_col_min.md`](stems_toms_prominence_and_decay_col_min.md),
covers the broader context of the PGA pipeline and the `decay_col_min_median_db`
diagnostic. This file is focused on the `prominence` value alone.

---

## 1. What prominence is (in larsnet)

For every event detected by the PGA (percentile-gated broad-attack) detector,
we compute a `prominence` value and stash it on the event under the
`prominence` key. The value is the standard "prominence" output of
`scipy.signal.find_peaks`:

> The prominence of a peak measures how much the peak stands out from the
> surrounding baseline of the signal. It is defined as the vertical distance
> between the peak and its lowest contour line — the lowest horizontal line
> that touches the peak but does not cross any higher peak on either side.
> ([scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html))

In larsnet, this becomes the input to the `pga_min_prominence` filter:
events with `prominence < threshold` are tagged `FILTERED`. The threshold
defaults to **1000** in the global `onset_detection.pga_min_prominence` config,
and can be overridden per-stem (e.g. `hihat.pga_min_prominence`). The project's
midiconfig.yaml may set a different value; project 4's hihat tuning set it
to **800** (visible in the sidecar's `pga_filter_config.pga_min_prominence`).

The user-facing knob is just `prominence < threshold → FILTERED`. Everything
else in this document is about where that prominence number actually comes
from.

---

## 2. How prominence is calculated in larsnet

The PGA prominence computation happens in two stages: (a) build the envelope
that gets peak-picked, then (b) ask scipy to compute prominence on each peak.

### 2.1 Stage A: build the broadband contrast envelope

The input to `find_peaks` is **not** the raw waveform or the STFT magnitude —
it is a *contrast-summed broadband attack envelope*. The pipeline is
[`stems_to_midi/percentile_gated_detector.py`](../stems_to_midi/percentile_gated_detector.py),
function `_detect_percentile_gated_broad_attacks_impl`. Steps:

1. **STFT**: `hop=256`, `n_fft=1024`, Hann window. Log-magnitude
   spectrogram `s_db` of shape `(n_bins, n_frames)`.
2. **Per-bin static noise floor** (with a global noise gate):
   for each frequency bin, take the 5th percentile of frames that are
   at least 0.5 dB above the bin's absolute minimum. The bin's floor
   is the mean of all values ≤ p5. After the per-bin pass, every
   bin's floor is clamped to `>=` the global gate (the upper bound of
   the quietest portions of the song, capped at `-60 dB`).
3. **Contrast**: `max(0, s_db − floor)`, shape `(n_bins, n_frames)`.
4. **Broad-frequency attack envelope**: sum of contrast over
   `broad_freq_min_hz..broad_freq_max_hz` (default 600–8000 Hz),
   thresholding each bin's contribution at `+10 dB` above the floor.
   Each frame now has one scalar envelope value: the total broadband
   contrast that frame, summed across all relevant bins.

The result is a 1-D array `envelope` of length `n_frames`. High values
= a frame where many broad bins rose significantly above their per-bin
noise floor (a real broadband attack). Low values = quiet or
sustained-only.

### 2.2 Stage B: peak-pick with scipy

```python
peaks, props = find_peaks(
    envelope,
    height=abs_envelope_threshold,   # q3 + 2.5 * IQR of envelope by default
    distance=nms_min_frames,          # 20 frames (~116 ms) by default
    prominence=0,                     # only require prominence > 0
)
```

The `prominence=0` parameter is a bit subtle: it tells scipy to **drop**
any peak that has zero prominence (a flat-top plateau), but it does
**not** filter out anything else. The actual per-peak prominence
values come back in `props['prominences']`.

These are the values stored on each event under the `prominence` key:

```python
# stems_to_midi/pga_event_builder.py:623
ev['prominence'] = float(_proms[i])
```

### 2.3 The IQR threshold (a related but separate value)

The detector also reports an `iqr_threshold` per event — the height
threshold the peak had to exceed to be picked at all:

```python
# stems_to_midi/pga_event_builder.py:612-657
if _env is not None and _env.size > 0:
    _q1, _q3 = np.percentile(_env, [25, 75])
    _iqr = _q3 - _q1
    _abs_thr = _q3 + 2.5 * _iqr
# ...
ev['iqr_threshold'] = float(_abs_thr)
```

This is **not** the same as `prominence`. It is the **minimum envelope
value** the peak had to exceed to be picked. In project 4's data, the
IQR threshold is around 22,211 while individual peaks have envelope
values of 11,000–13,000 — but wait, those are well below the IQR
threshold?

The peak-pick threshold uses `envelope_value` directly, which is a
**linear power sum** (broadband contrast summed across many bins,
each bin's contrast being a linear sum of itself). On dense mixes
this can reach very high values (>>10000) because many bins contribute
to the same frame. The peak-pick picks the top of this distribution.

The prominence values, by contrast, are relative-to-the-surrounding-
baseline measurements (a vertical distance, not an absolute level).
That's why they sit in the 30–15,000 range even when envelope values
sit in the 3,000–12,000 range.

The `iqr_threshold` value is recorded for diagnostics only — the
`pga_min_prominence` filter uses `prominence`, not `iqr_threshold`.

### 2.4 The threshold used by the `pga_min_prominence` filter

The filter compares `event['prominence']` against the configured
threshold (the `pga_min_prominence` setting, default **1000**).
Events with `prominence < threshold` are tagged `FILTERED` with
reason `"below pga_min_prominence ({value} < {threshold})"`. Events
above the threshold are tagged `KEPT` (subject to other filters).

The threshold resolution order is:

1. `<stem>.pga_min_prominence` (per-stem override)
2. `onset_detection.pga_min_prominence` (global default)
3. `1000.0` (hard floor, defensive)

Source: `pga_event_builder._resolve_pga_detector_param` at
`stems_to_midi/pga_event_builder.py:393`.

---

## 3. How scipy actually computes prominence

To understand why prominence can be very low for real strikes, it helps
to understand what scipy is doing under the hood.

For each peak `p` at frame `i_p`:

1. **Walk left from `i_p`**: find the first frame `i_left` to the left
   of `i_p` such that `envelope[i_left] > envelope[i_p]` (or we hit the
   edge). This is the left base candidate.
2. **Walk right from `i_p`**: find the first frame `i_right` to the
   right of `i_p` such that `envelope[i_right] > envelope[i_p]` (or we
   hit the edge). This is the right base candidate.
3. **Contour line**: `max(envelope[i_left], envelope[i_right])`. The
   peak's "lowest contour line" is the higher of the two bases — the
   smallest horizontal line that touches the peak but doesn't cross
   any higher peak on either side.
4. **Prominence**: `envelope[i_p] − contour_line_height`. If the
   envelope is monotonically descending away from `i_p` on both sides
   (no higher neighbor on either side), the contour is at the lowest
   of the surrounding envelope, and prominence = peak − that baseline.

This is why two closely-spaced peaks at similar heights can have very
different prominences. The peak whose neighbor is **closer and
similar in height** gets a tiny prominence (the contour is at the
neighbor's height, which is almost the same as the peak). The peak
whose neighbor is **further away or much lower** gets a large
prominence (the contour drops to a deep valley on one side).

---

## 4. The known failure mode: double-trigger on the rising edge

**Symptom**: an event has a real, dramatic spike (envelope jumps from
~3000 to ~11000 in ~440 ms — a 3.5× attack), but `prominence` reports
~47 (less than 5% of the typical 1000–15,000 range for real strikes).
The event is tagged `FILTERED` even though it is clearly the leading
edge of a real drum hit.

This was observed on project 4's hihat stem at time ~73.12s. The
relevant sidecar data:

```json
// event A — first peak of the strike (FILTERED)
{
  "time": 73.1218,
  "frame": 12594,
  "envelope_value": 11341.6822,
  "prominence": 46.8807,
  "iqr_threshold": 22211.2486,
  "duration_ms": null,
  "right_base_minus_peak_frames": 1,    // right valley only 1 frame away!
  "right_base_minus_peak_ms": 5.8,
  "peak_width_left_ip_frame": 12593.87,
  "peak_width_right_ip_frame": 12594.9,
  "decay_frames_walked": 2,
  "decay_stop_reason": "hit_other_event"
}

// event B — second peak of the strike, 2 frames later (KEPT)
{
  "time": 73.1339,
  "frame": 12596,
  "envelope_value": 11418.9525,
  "prominence": 11356.441,
  "duration_ms": 876.55,
  "right_base_minus_peak_frames": 155,  // right valley 155 frames away
  "right_base_minus_peak_ms": 899.77,
  "peak_width_right_ip_frame": 12719.8,
  "decay_frames_walked": 89,
  "decay_stop_reason": "normal"
}
```

Both events have nearly identical envelope values (11341 vs 11418 — a
0.7% difference). They are 12 ms apart. Event B has a real, 876 ms
ring (a clear, sustained broadband decay — exactly what a real hihat
hit looks like). Event A has no measurable ring because the algorithm's
peak-finding logic (see §5) immediately merges it into event B.

### 4.1 Why event A's prominence is so low

Looking at scipy's algorithm in §3:

- **Event A (frame 12594)**:
  - `left_base_frame = 12519` (75 frames to the left; the envelope was
    climbing from a low valley)
  - `right_base_frame = 12595` (1 frame to the right; the envelope
    is already at event B's attack level)
  - Contour line height = `max(env[12519], env[12595])` = high value
    (essentially event B's height, since B is 1 frame away)
  - Prominence = `env[12594] − env[12595]` ≈ 47 (tiny dip before B's
    attack resumes)

- **Event B (frame 12596)**:
  - `left_base_frame = 12519` (same as A)
  - `right_base_frame = 12751` (155 frames to the right; the envelope
    has decayed through event B's 876 ms ring)
  - Contour line height = `max(env[12519], env[12751])` = low value
    (the envelope has fully decayed by then)
  - Prominence = `env[12596] − env[12751]` ≈ 11356 (the full depth
    of the strike)

The contrast in computed prominence (~47 vs ~11356) does not reflect
any real difference in the strike's strength — both peaks are part of
the same physical hit. It reflects only the **topology of the envelope
landscape** around each peak. Event A happens to sit on the shoulder
of event B's rise; event B sits at the top of a long decay.

### 4.2 What prominence was supposed to capture (and didn't)

The whole point of prominence as a discriminator is: a real percussive
strike has a deep, sustained valley on either side of its peak (because
the attack is loud and the body rings while the surrounding audio is
quiet). A noise pop or ringing artifact has only a momentary spike with
no real depth below.

For event A, that depth exists in the audio — the envelope was at
~3,100 just 24 frames earlier (event at frame 12570 with envelope
3131), and much lower still 75 frames earlier (left_base_frame 12519).
A "true" prominence measure should have found that low surrounding
envelope and reported a large prominence.

scipy's algorithm cannot, because its contour line is the **maximum**
of left and right bases. When the right neighbor is at almost the same
height as the peak (a double-trigger shoulder), the contour is pinned
near the peak regardless of how deep the left valley goes.

### 4.3 The cascading effect on `duration_ms`

The problem doesn't stop at prominence. Event A's `duration_ms` is
`null` for the same reason:

In `compute_duration_ms` (event_features.py), the function first calls
`_find_attack_peak`, which searches `[-30 ms, +50 ms]` around the
event time for the local envelope maximum. For event A at 73.1218s,
the search lands on **frame 12596** (event B's peak, since 11418 >
11341). The duration calculation then walks forward from frame 12596
toward the `next_event_time_sec` cap (which is also event B's time,
73.1339s, frame 12596).

Now `i_peak == i_cap == 12596`, so the slope walk has zero frames to
traverse. `duration_sec_slope <= 0`, and the function returns `None`.

So a single upstream double-trigger produces two cascading artefacts:
1. Event A's prominence collapses to a meaningless low value.
2. Event A's duration becomes null (the algorithm thinks event A's
   ring is zero-length because the peak search merged into event B).

The downstream effect: the `pga_min_prominence` filter tags event A
as FILTERED, the sidecar tooltip says "below pga_min_prominence
(47 < 800)", and the MIDI event for event A is dropped. Event B
survives the filter, so the strike still appears in the MIDI output —
but at event B's time (12 ms late), not event A's time.

Whether this matters musically depends on the stem and the tempo.
For hihats at 80 bpm (project 4), 12 ms is well under the 16th-note
spacing (187 ms) and not noticeable. For faster passages or stems
where onset accuracy matters more, the 12 ms error compounds across
double-triggered strikes.

---

## 5. How to read prominence in the analysis sidecar

When debugging a `pga_min_prominence`-filtered event, check these
adjacent fields together:

| Field | Meaning |
|---|---|
| `envelope_value` | The envelope's actual value at the peak frame (the absolute broadband contrast). High = a frame with lots of energy in the broad band. |
| `prominence` | scipy's vertical-distance-to-contour-line measurement. **Local** to the immediate neighbor peaks. Can be very low for double-triggers even when `envelope_value` is high. |
| `iqr_threshold` | The peak-pick height threshold (`q3 + 2.5*IQR` of envelope). Diagnostic; not used by the filter. |
| `duration_ms` | Slope-of-decline ring time (peak to -10 dB/s). May be `null` for double-trigger shoulders (see §4.3). |
| `left_base_frame` / `right_base_frame` | Indices of the left and right valleys scipy picked. If `right_base_frame - frame` is small (< ~5 frames), event is part of a double-trigger. |
| `right_base_minus_peak_frames` | Distance from peak to right valley. Strong leading indicator of double-trigger scenario when < ~5. |
| `peak_width_right_ip_frame` | scipy.peak_widths at rel_height=0.9. The right intercept of the 10% slice. Should be far away for a real strike with a ring. |
| `inter_onset_ms` | Time to the next KEPT event. If this is small (< 30 ms), the event may be a double-trigger shoulder relative to a higher-prominence neighbor. |

**Heuristic**: if `envelope_value > 5000` but `prominence < 500`, the
event is very likely a double-trigger shoulder. Confirm with
`right_base_minus_peak_frames < 5` and `inter_onset_ms < 30`.

---

## 6. What this means for the filter's reliability

The `pga_min_prominence` filter has worked well in practice because:

1. **Real strikes tend to be isolated**. The majority of percussive
   events in real audio are spaced far enough apart (> 50 ms) that
   scipy's local-prominence measurement tracks the actual strike
   depth. The user's observation that prominence "generally works
   very well" is consistent with this.
2. **The filter catches the dominant FP class**: noise pops and
   ringing tails have low `envelope_value` AND low prominence (their
   envelope is a small bump on a sustained baseline, so the
   contour is close to the bump's top). They are reliably filtered.
3. **Even when the filter drops the lead-in event of a double-trigger,
   the main event survives.** The strike still appears in the MIDI,
   just at the second peak's time (a few ms late).

The failure mode (dropping a real strike's leading edge when it
appears as a double-trigger) is rare and recoverable — but it does
happen, and when it does the sidecar data clearly shows why (the
`right_base_minus_peak_frames` is small and `inter_onset_ms` is small).

The downstream `decay_col_min_median_db` filter (covered in the
companion document) and the `duration_ms` feature provide
independent signals that can confirm or reject the prominence
filter's verdict. WebUI tooltips should display all three together
when a strike is tagged `FILTERED`.

---

## 7. Where the prominence value is read and consumed

Code paths that consume `event['prominence']`:

| Location | Purpose |
|---|---|
| `stems_to_midi/filter_registry.json` | The `pga_min_prominence` filter spec (the canonical definition of the filter). |
| `stems_to_midi/pga_event_builder.py:1181` (`apply_pga_prominence_filter`) | Applies the filter, sets `status='FILTERED'` and `filter_reason`. |
| `stems_to_midi/midi.py:437` (`_round_value(ev.get('prominence'), 4)`) | Persists prominence (rounded to 4 decimals) into the analysis sidecar JSON. |
| `stems_to_midi/midi.py:587` (`logic['pga_min_prominence']`) | Persists the active filter threshold into the sidecar's logic block so the WebUI can render the slider correctly. |
| `stems_to_midi/rebuild_core.py:365` (toms PGA refilter) | Re-applies the prominence filter on rebuild using the current slider value. |
| `webui/static/js/threshold-tuning.js` | Reads the sidecar's `pga_min_prominence` from the logic block and renders the slider. Re-applies the filter when the slider moves. |
| `webui/static/js/filter_kinds.js` | Reads `filter_registry.json` (same source as Python) to evaluate `min_value` filters client-side. |

---

## 8. Future work (proposed)

The double-trigger failure mode is a real bug, but it is bounded
(see §6) and the existing layered filter chain (prominence +
decay_col_min + duration) provides independent signals that can
correct for it in the WebUI. Three approaches have been considered
for fixing the prominence calculation directly; each is deferred
until enough calibration data is collected to confirm the change
doesn't regress real-strike detection.

See [`agent-plans/prominence-improvement.plan.md`](../agent-plans/prominence-improvement.plan.md)
for the full plan and decision log.

---

## 9. TL;DR

- `event['prominence']` = scipy's vertical-distance-to-contour-line on the
  broadband contrast envelope (600–8000 Hz), computed at peak-pick time.
- The `pga_min_prominence` filter is a simple `prominence < threshold` test.
- The filter works well for isolated strikes but underestimates prominence for
  the leading edge of a double-trigger (where two envelope peaks are within
  ~12 ms of each other). The leading-edge peak gets `prominence ≈ 0` because
  scipy's contour line is pinned to the second peak's height.
- When investigating a `pga_min_prominence` rejection, check
  `right_base_minus_peak_frames` and `inter_onset_ms` to spot the
  double-trigger pattern. The strike likely still appears in the MIDI via the
  second peak.
