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
from scipy.signal import find_peaks

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
# real strike. The toms empirical distribution: quiet frames are
# ~5000, the smallest real strike is ~12000, the loudest is
# ~17000. 10000 is the safe floor — well above the noise, well
# below the smallest strike.
DEFAULT_ABS_ENVELOPE_THRESHOLD = 10000.0

# Minimum STFT frames between peaks (~116ms at hop=256).
# Real drum strikes are 100ms+ apart; this is the NMS floor
# that drops ringing tails within a few frames of a real strike
# without merging close-but-distinct strikes (e.g. a flam at
# 50ms would merge, but a typical 16th-note at 125ms would not).
DEFAULT_NMS_MIN_FRAMES = 20

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


def _build_static_noise_floor(s_db: np.ndarray) -> np.ndarray:
    """Per-bin static noise floor. Shape (n_bins,).

    For each freq bin:
      1. Find the bin's absolute minimum value (artificial digital
         silence is usually a few identical samples at the floor).
      2. Exclude frames within 0.5 dB of that minimum — those are
         silent / not real acoustic noise.
      3. From the remaining frames, take the 5th percentile (p5).
      4. The noise floor for the bin is the MEAN of all values ≤ p5.
         This is a robust estimator: it's not fooled by a single
         loud transient pulling the mean up.

    Args:
        s_db: log-magnitude spectrogram of shape (n_bins, n_frames).

    Returns:
        floor: per-bin noise floor of shape (n_bins,).
    """
    n_bins = s_db.shape[0]
    floor = np.zeros(n_bins)
    eps = 0.5
    p5_pct = DEFAULT_P5_PERCENTILE
    for b in range(n_bins):
        col = s_db[b]
        abs_min = col.min()
        real = col[col > abs_min + eps]
        if len(real) < 10:
            # All silence or nearly all silence — use the global min.
            floor[b] = abs_min
            continue
        p5 = np.percentile(real, p5_pct)
        quiet = real[real <= p5]
        if len(quiet) == 0:
            floor[b] = p5
        else:
            floor[b] = quiet.mean()
    return floor


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
    broad_freq_min_hz: float = DEFAULT_BROAD_FREQ_MIN_HZ,
    broad_freq_max_hz: float = DEFAULT_BROAD_FREQ_MAX_HZ,
    db_rise_threshold: float = DEFAULT_DB_RISE_THRESHOLD,
    abs_envelope_threshold: float = DEFAULT_ABS_ENVELOPE_THRESHOLD,
    nms_min_frames: int = DEFAULT_NMS_MIN_FRAMES,
    strike_offset_sec: float = DEFAULT_STRIKE_OFFSET_SEC,
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

    # Step 2: per-bin static noise floor.
    floor = _build_static_noise_floor(s_db)

    # Steps 3+4: foreground contrast + broad-frequency attack envelope.
    envelope = _broad_attack_envelope(
        s_db, freqs, floor,
        broad_freq_min_hz=broad_freq_min_hz,
        broad_freq_max_hz=broad_freq_max_hz,
        db_rise_threshold=db_rise_threshold,
    )

    # Step 5: peak-pick. Two thresholds — an absolute envelope
    # minimum (drops noise-level peaks) and a minimum-frame NMS
    # (drops ringing tails within a few frames of a real strike).
    # Both are absolute values, not fractions of the envelope's
    # max, so the thresholds don't drift if the loudest hit is
    # unusually loud or quiet.
    peaks, props = find_peaks(
        envelope,
        height=abs_envelope_threshold,
        distance=nms_min_frames,
    )

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
    }
    return event_times, debug
