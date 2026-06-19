"""
Percentile-gated broad attack detector.

A complementary onsets detector to ``spectral_transient_core``. Where
the snap/ring-based detector measures *absolute* level in
hand-picked frequency bands (and gets fooled by saturated bands
where the absolute level is misleading), this one measures
*change above a per-bin noise floor* across a broad frequency
range — robust to saturation because the noise floor moves with
the signal.

Algorithm:

  1. STFT: hop=256, n_fft=1024, Hann window. Log-magnitude spectrogram.
  2. For each freq bin row:
     a. Find the bin's absolute minimum.
     b. Exclude frames within 0.5 dB of that minimum (kills the
        artificial digital-silence block; real near-silent acoustic
        noise survives).
     c. From the remaining frames, take the 5th percentile.
     d. The noise floor for the bin is the mean of all values ≤ p5
        — a robust "quiet tail" estimator that isn't fooled by brief
        transients.
  3. Contrast = max(0, spectrogram - noise_floor).
  4. Envelope = sum of contrast over broad-frequency bins
     (default 600-8000 Hz — excludes the saturated 0-600 Hz
     low-bands on toms where the broadband attack still shows up
     in the high bands even when the low bands are clipping).
  5. find_peaks on the envelope with absolute-threshold and
     minimum-distance gates.
  6. Subtract 8 ms from each peak to compensate for the Hann
     window's center-of-bin bias (calibrated against known
     ground-truth strike times).

Why this works for the toms case the existing snap_delta misses:
the snap detector measures *absolute* level in narrow bands. On
real toms audio the low bands (B0/B1) clip/saturate, so the
absolute level is unreliable. This detector measures *change
relative to the local noise floor* — saturated bands still have
a valid noise floor (the quiet frames), and the contrast (signal
- noise) is just as informative for a saturated frame as for a
quiet one.

Why it complements band_max_ratio: band_max_ratio rewards
single-band dominance (good for the RING signal, which fires on
the sustained body of a strike). The contrast envelope rewards
broadband transients (good for the ATTACK signal, which fires on
the percussive onset). A real hit has both; a wire-tail or
sustained-ringing FP has only the first.

Public entry point: :func:`detect_percentile_gated_broad_attacks`.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths
from typing import Tuple

from .spectral_transient_core import timed

# Frequency band cutoffs (Hz) — must match DEFAULT_BANDS in
# spectral_transient_core.py.
#
#   B0:   60-200
#   B1:  200-600
#   B2:  600-1200
#   B3: 1200-2400
#   B4: 2400-8000
#
# Default envelope sums B2-B4: the user observed that on toms
# audio the 0-600 Hz low bands are saturated at strike onsets,
# so absolute levels there are unreliable. The broadband
# percussive attack still shows up cleanly in the 600-8000 Hz
# range. Per-stem override is available via the function's
# ``broad_freq_min_hz`` / ``broad_freq_max_hz`` kwargs.
DEFAULT_BROAD_FREQ_MIN_HZ = 600.0
DEFAULT_BROAD_FREQ_MAX_HZ = 8000.0

# Only count contrast values strictly above this dB above the
# per-bin noise floor. 10 dB is "an order of magnitude above
# noise" — it filters out the "I'm slightly above the floor"
# bins without losing the strike signal.
DEFAULT_DB_RISE_THRESHOLD = 10.0

# Percentile used to find the per-bin noise floor. The 5th
# percentile of the bin's distribution is the "quiet tail" —
# below that is the floor; above that is signal. Smaller values
# (e.g. 1) overfit to silence; larger (e.g. 20) over-attribute
# quiet hits to the floor and lose the strikes.
DEFAULT_P5_PERCENTILE = 5.0

# Minimum absolute envelope value for a peak to be considered a
# real strike. Computed per-call as ``q3 + 2.5 * IQR`` of the
# envelope itself (a standard extreme-outlier rule, equivalent
# to ~99% of a normal distribution) — so the threshold adapts
# to the song's actual dynamic range rather than a hard-coded
# constant. Set here as a fallback for the rare case where the
# IQR-based computation can't be performed.
DEFAULT_ABS_ENVELOPE_THRESHOLD = None  # computed per-call by default

# Minimum STFT frames between peaks (~116ms at hop=256).
# 116ms is shorter than a 16th note at 130bpm (115ms) and longer
# than a typical drum flam (~30ms). It's the "safe NMS floor"
# for typical drumming — anything tighter would merge flams
# and double-triggers, anything looser would split sixteenths.
DEFAULT_NMS_MIN_FRAMES = 20

# Upper bound (dB) for the global noise-floor gate. The gate is
# ``max(p5 across all bins)`` — the loudest of the per-bin
# quietest 5% values. On dense mixes the gate can rise very
# high (e.g. -45 dB on a saturated toms stem), which would
# over-attribute real signal to "noise" and suppress the
# contrast envelope. Capping the gate at -60 dB prevents
# this over-lift: if a bin's true p5 is above -60 dB, that
# bin's contrast is already saturated and the gate value
# doesn't matter for it; the cap only matters for the bins
# that *would* otherwise be pulled up to the over-high gate.
# Configurable via ``onset_detection.pga_max_floor_gate_db``
# in midiconfig.yaml (per-stem override ``toms.pga_max_floor_gate_db``
# also accepted by ``_build_pga_events_with_filter``).
DEFAULT_MAX_FLOOR_GATE_DB = -60.0

# Hann window center bias: a transient at sample t lands in
# the STFT frame whose CENTER is at t, not the frame whose
# START is at t. For a 1024-sample Hann at sr=44100, the center
# is 512 samples = 11.6ms into the frame. But the *attack*
# energy is concentrated a few hundred samples earlier than
# the center (the rising edge of the strike reaches the
# detector before the window-center does), so the empirical
# offset is ~8ms. Calibrated against the user's 6 known GT
# toms strikes near 74s; tune per-stem if needed.
DEFAULT_STRIKE_OFFSET_SEC = 0.008


def _build_static_noise_floor(
    s_db: np.ndarray,
    max_floor_gate_db: float = DEFAULT_MAX_FLOOR_GATE_DB,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """Per-bin static noise floor + global noise gate. 2026-06-15.

    For each freq bin:
      1. Find the bin's absolute minimum value (artificial digital
         silence is usually a few identical samples at the floor).
      2. Exclude frames within 0.5 dB of that minimum — those are
         silent / not real acoustic noise.
      3. From the remaining frames, take the 5th percentile (p5).
      4. The noise floor for the bin is the MEAN of all values ≤ p5.
         This is a robust estimator: it's not fooled by a single
         loud transient pulling the mean up.

    After the per-bin pass, a **global noise gate** is applied: every
    bin's floor is clamped to ``max(floor[b], gate_db)`` where
    ``gate_db = min(max(p5 across all bins), max_floor_gate_db)``.
    The first term is the upper bound of the quietest portions of
    the song — the loudest of the per-bin p5 values. The second
    term is a safety cap: on dense/saturated mixes the unbounded
    gate can rise to e.g. -45 dB, which over-lifts the floor and
    kills real attacks in the contrast envelope. Capping at
    ``max_floor_gate_db`` (default -60 dB) prevents that over-lift
    while keeping the gate effective for the actual silence-gap
    phantom scenario it was added to solve.

    Why a gate: stem-splitter silence frames (~-160 dB in every bin)
    can pull a per-bin floor down to digital silence. When the noise
    resumes at -75 dB, the contrast ``max(0, s_db - floor)`` jumps
    85 dB and the IQR-gated ``find_peaks`` calls it a high-prominence
    attack — a phantom event. Lifting every bin's floor to the
    global gate zeros the contrast for that silence-to-noise
    transition (contrast becomes ``max(0, -75 - gate) ≈ 0`` when
    ``gate ≈ -70``). The gate is the loudest quiet bin, so it
    doesn't lift any bin above its true noise level — it only
    pushes under-estimating bins back up to the song's true quiet
    floor.

    Why a cap: a gate value above -60 dB is no longer "the quiet
    tail" of the song — it's mid-dynamic-range content. A real
    toms strike lives well above -60 dB during the attack, so
    clamping the gate to -60 dB does not erase the strike's
    contrast (the strike's contrast is still 20-40 dB above the
    floor). The cap is the threshold below which the gate stops
    being a "quiet floor estimator" and starts being an "attack
    suppressor" — the call site wants the former, not the latter.

    Args:
        s_db: log-magnitude spectrogram of shape (n_bins, n_frames).
        max_floor_gate_db: upper bound (dB) for the global gate.
            The actual gate is ``min(max(p5), max_floor_gate_db)``.
            ``None`` (or any non-numeric falsy) falls back to
            :data:`DEFAULT_MAX_FLOOR_GATE_DB`. Setting it to
            ``+inf`` (or a very large positive value) effectively
            disables the cap.

    Returns:
        (floor, gate_db_clamped, p5_per_bin, n_lifted):
          - floor: per-bin noise floor after the gate clamp,
            shape (n_bins,).
          - gate_db_clamped: the final gate value actually used
            (the post-cap value, in dB). 0.0 if the input is
            empty. Compare against ``p5_per_bin.max()`` to see
            when the cap fired.
          - p5_per_bin: the pre-clamp p5 per bin, shape (n_bins,).
            Exposed for the summary print and future WebUI use.
          - n_lifted: count of bins where the gate raised the floor
            (``floor_pre[b] < gate_db_clamped``). 0 if the input
            is empty.
    """
    n_bins = s_db.shape[0]
    floor = np.zeros(n_bins)
    p5_per_bin = np.zeros(n_bins)
    eps = 0.5
    p5_pct = DEFAULT_P5_PERCENTILE
    for b in range(n_bins):
        col = s_db[b]
        abs_min = col.min()
        real = col[col > abs_min + eps]
        if len(real) < 10:
            # All silence or nearly all silence — use the global min.
            floor[b] = abs_min
            p5_per_bin[b] = abs_min
            continue
        p5 = np.percentile(real, p5_pct)
        quiet = real[real <= p5]
        if len(quiet) == 0:
            floor[b] = p5
        else:
            floor[b] = quiet.mean()
        p5_per_bin[b] = p5
    if n_bins > 0:
        gate_db_raw = float(np.max(p5_per_bin))
        # 2026-06-18: cap the gate at max_floor_gate_db. On
        # dense/saturated mixes the raw gate can rise above the
        # song's true quiet floor (e.g. -45 dB on a saturated
        # toms stem where a high band has a non-silent p5), and
        # lifting every bin's floor to that value kills real
        # attacks in the contrast envelope. The cap keeps the
        # gate's "quiet floor" role intact while preventing the
        # over-lift. Setting max_floor_gate_db=None falls back
        # to the module default; setting it to a very large
        # positive value effectively disables the cap.
        cap = (
            max_floor_gate_db
            if max_floor_gate_db is not None
            else DEFAULT_MAX_FLOOR_GATE_DB
        )
        gate_db_clamped = min(gate_db_raw, float(cap))
        n_lifted = int(np.sum(floor < gate_db_clamped))
        floor = np.maximum(floor, gate_db_clamped)
    else:
        gate_db_clamped = 0.0
        n_lifted = 0
    return floor, gate_db_clamped, p5_per_bin, n_lifted


def _broad_attack_envelope(
    s_db: np.ndarray,
    freqs: np.ndarray,
    floor: np.ndarray,
    broad_freq_min_hz: float,
    broad_freq_max_hz: float,
    db_rise_threshold: float,
) -> np.ndarray:
    """Build the contrast-summed attack envelope.

    Args:
        s_db: log-magnitude spectrogram of shape (n_bins, n_frames).
        freqs: per-bin center frequencies (Hz), shape (n_bins,).
        floor: per-bin noise floor from ``_build_static_noise_floor``.
        broad_freq_min_hz, broad_freq_max_hz: inclusive freq range
            to sum across.
        db_rise_threshold: only count contrast above this dB.

    Returns:
        envelope: shape (n_frames,). High values = a frame where
        many broad bins rose significantly above the per-bin noise
        floor (a real broadband attack). Low values = quiet or
        sustained-only.
    """
    contrast = np.maximum(0.0, s_db - floor[:, None])
    bin_mask = (freqs >= broad_freq_min_hz) & (freqs <= broad_freq_max_hz)
    contrast_broad = contrast[bin_mask, :]
    contrast_broad = np.where(
        contrast_broad > db_rise_threshold, contrast_broad, 0.0
    )
    return contrast_broad.sum(axis=0)


def _refine_peak_time(
    p: int,
    envelope: np.ndarray,
    times: np.ndarray,
    strike_offset_sec: float,
) -> float:
    """Sub-frame refinement via parabolic interpolation.

    The peak of the contrast envelope is a real-valued maximum
    between integer frames. Fitting a parabola to the peak and its
    two neighbors gives ~0.1-frame accuracy. We then subtract
    ``strike_offset_sec`` to compensate for the Hann window's
    center-of-bin bias (the contrast peaks a few frames AFTER the
    actual strike onset).
    """
    if 1 <= p < len(envelope) - 1:
        y0, y1, y2 = envelope[p - 1], envelope[p], envelope[p + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-9:
            delta = 0.5 * (y0 - y2) / denom
        else:
            delta = 0.0
        return times[p] + delta * (times[1] - times[0]) - strike_offset_sec
    return times[p] - strike_offset_sec


def detect_percentile_gated_broad_attacks(
    audio: np.ndarray,
    sr: int,
    *args,
    **kwargs,
):
    """Wrapper around ``_detect_percentile_gated_broad_attacks_impl``
    that logs [t+Xs] timing on every call. The real implementation
    lives in the underscore-prefixed function below; this wrapper is
    kept thin so all public callers get timing for free.
    """
    with timed("detect_percentile_gated_broad_attacks"):
        return _detect_percentile_gated_broad_attacks_impl(
            audio, sr, *args, **kwargs
        )


def _detect_percentile_gated_broad_attacks_impl(
    audio: np.ndarray,
    sr: int,
    broad_freq_min_hz: float = DEFAULT_BROAD_FREQ_MIN_HZ,
    broad_freq_max_hz: float = DEFAULT_BROAD_FREQ_MAX_HZ,
    db_rise_threshold: float = DEFAULT_DB_RISE_THRESHOLD,
    abs_envelope_threshold: float = None,  # IQR-based by default
    nms_min_frames: int = DEFAULT_NMS_MIN_FRAMES,
    strike_offset_sec: float = DEFAULT_STRIKE_OFFSET_SEC,
    max_floor_gate_db: float = DEFAULT_MAX_FLOOR_GATE_DB,
    n_fft: int = 1024,
    hop: int = 256,
):
    """Detect broadband percussive attacks in an audio stem.

    See module docstring for the algorithm. Designed to complement
    :func:`stems_to_midi.spectral_transient_core.detect_spectral_transients`
    — that one measures absolute band levels (good for the RING
    signal, which develops AFTER the strike); this one measures
    broadband *change* relative to a per-bin noise floor (good
    for the ATTACK signal, which fires AT the strike).

    Args:
        audio: 1D float array of mono audio samples.
        sr: sample rate (Hz).
        broad_freq_min_hz, broad_freq_max_hz: inclusive Hz range
            to sum the contrast over. Default 600-8000 excludes the
            low bands that saturate on toms strikes.
        db_rise_threshold: only count contrast > this dB. Default
            10 dB = "an order of magnitude above noise".
        abs_envelope_threshold: only count peaks where the envelope
            value is strictly greater than this. Default 10000
            (empirically: quiet frames ~5000, smallest real strike
            ~12000, so this safely excludes noise).
        nms_min_frames: minimum STFT frames between peaks. Default
            20 = ~116ms at hop=256. Drops ringing tails without
            merging close-but-distinct strikes.
        strike_offset_sec: subtract from each peak time. Default
            8ms = Hann window center-of-bin bias for toms strikes.
            Tune per-stem if needed.
        max_floor_gate_db: upper bound (dB) for the global noise
            floor gate (2026-06-18). The gate is the max p5
            across all bins, capped at this value to prevent
            over-lift on dense/saturated mixes. Default -60 dB
            (matches ``onset_detection.pga_max_floor_gate_db``
            in midiconfig.yaml). Set to a very large positive
            value to effectively disable the cap, or ``None``
            to fall back to the module default.
        n_fft, hop: STFT parameters. Defaults match the rest of
            the larsnet pipeline (see ``compute_stft_db``).

    Returns:
        event_times: list of float strike times in seconds,
            sub-frame-accurate, Hann-bias-corrected.
        debug: dict with intermediate arrays (s_db, floor,
            envelope, peak indices, prominences) for inspection.
    """
    # Import here to avoid an import cycle at module load time.
    from .spectral_transient_core import compute_stft_db

    freqs, times, s_db = compute_stft_db(audio, sr, n_fft=n_fft, hop=hop)

    # Step 2: per-bin static noise floor + global noise gate.
    # The helper now returns (floor, gate_db_clamped, p5_per_bin, n_lifted)
    # — the floor is post-clamp, gate_db_clamped is the
    # post-cap gate value (the raw max-p5 may be higher; see
    # gate_db_raw below for the pre-cap value), p5_per_bin and
    # n_lifted are exposed for the summary print and future
    # WebUI surfacing. See _build_static_noise_floor for the
    # rationale (kills the silence-to-noise phantom that arises
    # from stem-splitter digital-silence gaps) and the
    # max_floor_gate_db cap (2026-06-18) which prevents the
    # over-lift on dense/saturated mixes.
    floor, gate_db_clamped, p5_per_bin, n_lifted = _build_static_noise_floor(
        s_db, max_floor_gate_db=max_floor_gate_db,
    )

    # Steps 3+4: foreground contrast + broad-frequency attack envelope.
    envelope = _broad_attack_envelope(
        s_db, freqs, floor,
        broad_freq_min_hz=broad_freq_min_hz,
        broad_freq_max_hz=broad_freq_max_hz,
        db_rise_threshold=db_rise_threshold,
    )

    # Step 5: peak-pick. Two thresholds — an envelope minimum and
    # a minimum-frame NMS. The envelope minimum defaults to
    # ``q3 + 2.5 * IQR`` of the envelope itself (a standard
    # extreme-outlier rule): any peak above the bulk of the
    # distribution is a candidate. This adapts to the song's
    # actual dynamic range — a loud kick track and a quiet
    # acoustic toms track both get sensible thresholds without
    # hard-coded constants. Override with ``abs_envelope_threshold``
    # for full control.
    if abs_envelope_threshold is None:
        q1, q3 = np.percentile(envelope, [25, 75])
        iqr = q3 - q1
        abs_envelope_threshold = q3 + 2.5 * iqr
    peaks, props = find_peaks(
        envelope,
        height=abs_envelope_threshold,
        distance=nms_min_frames,
        prominence=0,  # require any prominence > 0 — kills pure plateau/flat-top FPs
    )

    # Peak widths (2026-06-19): scipy.peak_widths measures the
    # horizontal extent of each peak at a configurable relative
    # height. rel_height=0.9 means "the width at 10% below the
    # peak's top" — this is bounded to a tight slice around the
    # peak, unlike left_bases/right_bases (which can travel to a
    # distant baseline if no real valley exists nearby). The
    # resulting left_ips/right_ips are floating-point frame
    # indices, so the per-event attack/decay split is sub-frame
    # accurate. For hihat open vs closed, decay_frames is the
    # candidate discriminator: closed hits have a tight
    # right_ips, open hits have a long ring pushing the
    # right_ips further out. Safe to call with an empty peaks
    # array — scipy returns empty arrays in that case.
    if len(peaks) > 0:
        widths, width_heights, left_ips, right_ips = peak_widths(
            envelope, peaks, rel_height=0.9,
        )
    else:
        widths = np.array([])
        width_heights = np.array([])
        left_ips = np.array([])
        right_ips = np.array([])

    # Step 6: sub-frame refinement + Hann-bias correction.
    event_times = [
        _refine_peak_time(p, envelope, times, strike_offset_sec)
        for p in peaks
    ]

    debug = {
        'freqs': freqs,
        'times': times,
        's_db': s_db,
        'floor': floor,
        'envelope': envelope,
        'peaks': peaks,
        'prominences': props.get('prominences', np.array([])),
        # Peak bases (2026-06-19): indices into the envelope
        # array marking the left/right "valley" of each peak —
        # the lowest envelope point on each side before the
        # peak's prominence cone closes. Populated by scipy
        # even when prominence=0 is passed; the prominence
        # values themselves are 0, but the base indices are
        # still real and reflect the natural valley around
        # the peak. Diagnostic only — used to compute
        # right_base_minus_peak_ms in the sidecar.
        'left_bases': props.get('left_bases', np.array([])),
        'right_bases': props.get('right_bases', np.array([])),
        # Peak widths (2026-06-19): scipy.peak_widths at
        # rel_height=0.9. Bounded to a tight 10% slice around
        # each peak, so unlike left_bases/right_bases the
        # measurements can't run off to a distant baseline.
        # left_ips / right_ips are floating-point frame
        # indices. Downstream pga_event_builder turns them
        # into per-event attack/decay frame splits for the
        # hihat open/closed discrimination test.
        'peak_widths': widths,
        'peak_width_heights': width_heights,
        'peak_left_ips': left_ips,
        'peak_right_ips': right_ips,
        # Noise-gate summary (2026-06-15) — exposed for the
        # end-of-detect summary print and future WebUI
        # surfacing. gate_db is the post-cap gate value
        # (what the floor was actually clamped to);
        # gate_db_raw is the pre-cap value (the max p5
        # across all bins) so the caller can see when the
        # cap fired; p5_per_bin is the pre-clamp per-bin
        # p5; n_lifted is the count of bins the gate
        # raised. See _build_static_noise_floor for the
        # algorithm and max_floor_gate_db for the cap.
        'gate_db': gate_db_clamped,
        'gate_db_raw': float(np.max(p5_per_bin)) if len(p5_per_bin) > 0 else 0.0,
        'p5_per_bin': p5_per_bin,
        'n_lifted': n_lifted,
    }
    # One-line summary of the noise-floor gate. This is the
    # imperative-shell residue; the helper is pure and
    # returns the values, this site is what the user sees
    # in the console. Format chosen to be scannable in a
    # pipeline log without scrolling.
    if len(p5_per_bin) > 0:
        p5_min = float(np.min(p5_per_bin))
        p5_max = float(np.max(p5_per_bin))
        # 2026-06-18: show the cap when it fired, so the
        # operator knows the gate was clamped (not the true
        # song quiet floor). The "(cap=XdB)" suffix only
        # appears when the cap actually changed the value;
        # otherwise the raw and clamped gates are equal and
        # the suffix is omitted to keep the line scannable.
        cap = (
            max_floor_gate_db
            if max_floor_gate_db is not None
            else DEFAULT_MAX_FLOOR_GATE_DB
        )
        cap_str = (
            f" (cap={float(cap):.1f}dB)"
            if gate_db_clamped < p5_max - 1e-9
            else ""
        )
        print(
            f"[percentile_gated] noise floor: gate={gate_db_clamped:.1f}dB{cap_str}, "
            f"per-bin p5 range=[{p5_min:.1f}, {p5_max:.1f}]dB, "
            f"lifted {n_lifted}/{len(p5_per_bin)} bins"
        )
    return event_times, debug
