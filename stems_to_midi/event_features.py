"""
Event feature extraction for percussive onset classification.

This module is the per-event analysis pass that runs AFTER onsets
have been detected. It computes a battery of physically-motivated
features that downstream code (or human inspectors) can use to
classify each onset as kick / snare / hihat / toms / cymbals /
click / etc.

The features were chosen to discriminate the cases the user
identified in the false-positive triage session (2026-06-10):

  - Clicks / impulse noise: < 30ms duration, no clear pitch
  - Real toms strikes: 300-1500ms duration, 80-200Hz fundamental
  - Real snare hits: 80-300ms duration, 150-300Hz body
  - Kick: 200-800ms duration, 40-80Hz fundamental (low — needs care)
  - Open hihat: 30-100ms duration, sustained high-band content
  - Closed hihat: 30-100ms duration, fast-decaying high-band content

The classic discriminator across all these is the spectrogram
"shape": real strikes have an L-shape (broad attack, low-frequency
tail) while clicks have an I-shape (narrow vertical, no tail).
Duration captures this directly. Pitch + decay + brightness
disambiguate the cases that duration alone can't separate.

DESIGN PRINCIPLES (why this is its own module, not folded into
the existing detection_shell.py pitch functions):

  1. The existing detect_tom_pitch / detect_cymbal_pitch /
     detect_snare_pitch functions are per-stem and only return
     a single number. This module is per-event and returns a
     dict of all features, so a single call covers the whole
     feature surface.

  2. The existing functions take a fixed window from the onset
     forward. For percussive events, that window INCLUDES the
     broadband attack, which makes YIN/pYIN noisy (the attack
     is broadband noise, not pitched). The new module SKIPS the
     attack by 15-30ms (configurable) before running pitch
     detection, so it runs on the body of the sound.

  3. The existing functions use librosa.yin / librosa.pyin
     which are slow on long segments and don't expose per-frame
     confidence cleanly. This module calls the same librosa
     primitives (no point reimplementing YIN) but wraps them
     in a try/except + segment-length guard + confidence
     reporting, so a bad segment returns a clean (None, None)
     instead of throwing.

  4. New core code and tests, per the user's instruction
     (2026-06-10) — "in case there's oddball stuff in there."
     The plan is to eventually remove the other detectors and
     unify on this one feature pass.

Public entry point: :func:`compute_event_features`.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


# Default skip-past-attack time for pitch detection. The
# broadband attack is typically 5-20ms for a percussive onset;
# skipping 15ms lands us in the body where the fundamental
# is stable. Set to 0 to fall back to the existing
# detect_tom_pitch behavior (run YIN on the attack + body
# together, which is noisier).
DEFAULT_ATTACK_SKIP_MS = 15.0

# Body window length for pitch detection. The body of a
# percussive strike is 50-2000ms; 300ms gives a clean
# autocorrelation for down to ~80Hz (3.3 cycles in 300ms)
# while staying short enough to skip the decay tail.
DEFAULT_PITCH_BODY_WINDOW_MS = 300.0

# dB threshold for "ring time" / duration detection. The
# moment the broadband envelope drops to (peak - DROP_DB)
# below the peak is the end of the ring. 20 dB is
# "indistinguishable from noise floor" for typical acoustic
# content.
DEFAULT_DURATION_DROP_DB = 20.0

# pYIN voiced-probability threshold. Frames below this
# are treated as unvoiced and excluded from the median.
# 0.5 is librosa's default; raising to 0.7 reduces false
# positives on noisy segments.
DEFAULT_PYIN_VOICED_PROB = 0.5


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Squeeze stereo to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=-1)


def _envelope_at_time(
    audio: np.ndarray,
    sr: int,
    t_sec: float,
    n_fft: int = 1024,
    hop: int = 256,
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the broadband LINEAR-power envelope around an event time.

    Used by duration and decay measurements. Returns
    ``(times_sec, envelope_linear)`` arrays.

    The envelope sums LINEAR (not dB) power in the
    ``broad_min_hz..broad_max_hz`` range. Linear power is
    essential here — dB-scale envelopes span 9+ orders of
    magnitude for real percussive strikes, and "20dB below
    peak" on a dB-scale envelope corresponds to ~0.001% of
    the linear peak, which the algorithm can't easily
    reason about. Linear envelopes stay in the 0-peak range
    and 20dB below peak is just peak * 0.01.

    The default range covers toms (200-8000Hz) but excludes
    the saturated 0-200Hz sub-bass. Override for kick-specific
    work (e.g. ``broad_min_hz=30, broad_max_hz=200``).
    """
    from .spectral_transient_core import compute_stft_db
    freqs, times, s_db = compute_stft_db(audio, sr, n_fft=n_fft, hop=hop)
    # dB → linear: S = 10^(dB/20) for magnitude, but for
    # power it's 10^(dB/10). STFT output is magnitude (dB
    # amplitude), so use /20. Floor at 1e-12 to avoid log/0.
    S_linear = np.maximum(10 ** (s_db / 20.0), 1e-12)
    # Sum linear power in the broad range. This is the
    # total broadband energy per frame, in linear units.
    mask = (freqs >= broad_min_hz) & (freqs <= broad_max_hz)
    env = S_linear[mask].sum(axis=0)
    return times, env


def compute_duration_ms(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
    n_fft: int = 1024,
    hop: int = 256,
    next_event_time_sec: Optional[float] = None,
    min_slope_db_per_s: float = -10.0,
    slope_window_frames: int = 3,
) -> Optional[float]:
    """Measure the duration of a percussive event in milliseconds
    by finding the end of the ring via slope-of-decline.

    The "ring end" is the frame where the envelope's
    rate-of-decline flattens out — i.e. the strike has
    finished its rapid decay and the remaining energy is
    just sustained noise or background. The slope is
    measured in dB/s over a ``slope_window_frames``-frame
    moving window.

    Why slope-of-decline, not threshold drop:

      - Threshold drop needs to know the "noise floor" of
        this audio, which varies. A strike in a quiet
        acoustic track has a low noise floor; a strike in
        a loud rock mix has a high one.
      - Slope-of-decline is self-calibrating: the ring
        ends when the energy stops decreasing rapidly,
        regardless of what the noise floor is.
      - Slope-of-decline is unaffected by the next event's
        attack (the algorithm doesn't need a "next event"
        to find the ring end — only to handle the edge
        case where the strike rings forever and never
        flattens).
      - Slope-of-decline is unaffected by filtered-out
        events in between. If a click is filtered out,
        the surrounding strikes' ring times are still
        computed correctly (the click didn't add real
        energy to anyone's ring).

    Args:
        audio: mono or stereo audio array
        sr: sample rate
        event_time_sec: time of the onset
        broad_min_hz, broad_max_hz: frequency band for
            the envelope
        next_event_time_sec: optional cap. If the slope
            never flattens before this time (e.g. another
            strike raised the envelope back up), the
            duration is reported as "time to cap" and the
            caller can decide whether to trust it. Pass
            the time of the NEXT SURVIVING event after
            filtering.
        min_slope_db_per_s: ring is considered "ended"
            when the smoothed slope becomes less steep
            than this. Default -10 dB/s means "ring ends
            when the energy stops dropping at >=10 dB/s."
            A typical toms strike decays at 30-60 dB/s
            during its body, so -10 dB/s is a clear
            "the strike is over" threshold.
        slope_window_frames: window size for the slope
            smoothing. Default 3 frames (~17ms at
            hop=256, sr=44100). Larger values give a
            smoother slope but lag the actual ring end
            by ~half the window.

    Returns:
        Duration in milliseconds, or None if the peak
        can't be located. The reported duration is the
        time from the attack peak to the frame where the
        slope-of-decline first flattens below
        ``min_slope_db_per_s``.
    """
    times, env = _envelope_at_time(
        audio, sr, event_time_sec,
        n_fft=n_fft, hop=hop,
        broad_min_hz=broad_min_hz, broad_max_hz=broad_max_hz,
    )
    i_peak, _peak_val = _find_attack_peak(times, env, event_time_sec)
    if i_peak is None:
        return None
    # Convert envelope to dB (avoids 0/0 issues and
    # gives a linear-in-time slope for exp decay).
    env_db = 20.0 * np.log10(np.maximum(env, 1e-12))
    # Compute the slope (dB/s) using a centered difference.
    # For a frame i, slope = (env_db[i+1] - env_db[i-1]) /
    # (2*dt). The first and last frames use a one-sided
    # difference. Result is in dB per second.
    dt = times[1] - times[0]
    slope = np.zeros_like(env_db)
    slope[1:-1] = (env_db[2:] - env_db[:-2]) / (2.0 * dt)
    slope[0] = (env_db[1] - env_db[0]) / dt
    slope[-1] = (env_db[-1] - env_db[-2]) / dt
    # Smooth the slope with a moving average over
    # ``slope_window_frames`` frames. This kills the
    # frame-to-frame jitter from the STFT's spectral
    # leakage without significantly lagging the actual
    # ring end.
    kernel = np.ones(slope_window_frames) / slope_window_frames
    slope_smooth = np.convolve(slope, kernel, mode='same')
    # Walk forward from the peak. The ring ends at the
    # first frame where the smoothed slope is less steep
    # than min_slope_db_per_s. (min_slope_db_per_s is
    # negative; we want |slope| < |min_slope_db_per_s|,
    # i.e. slope > min_slope_db_per_s.)
    # Optional cap: stop at next_event_time_sec if provided.
    if next_event_time_sec is not None and next_event_time_sec > times[i_peak]:
        i_cap = int(np.argmin(np.abs(times - next_event_time_sec)))
    else:
        i_cap = len(env_db) - 1
    i_end = i_peak + slope_window_frames  # skip past the smoothing window
    while i_end < i_cap and slope_smooth[i_end] < min_slope_db_per_s:
        i_end += 1
    # The ring end is i_end (where the slope flattened)
    # or i_cap (if we hit the cap before flattening).
    duration_sec = times[min(i_end, i_cap)] - times[i_peak]
    if duration_sec <= 0:
        return None
    return float(duration_sec * 1000.0)


def compute_duration_to_valley_ms(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    next_event_time_sec: float,
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
    n_fft: int = 1024,
    hop: int = 256,
) -> Optional[float]:
    """Measure the duration of a percussive event to the envelope
    minimum before the next event.

    Where :func:`compute_duration_ms` walks until the envelope
    crosses a threshold OR the next event starts (whichever
    comes first), this function walks until the **envelope
    minimum** between this event and the next. The minimum
    is the natural "silence" between the two strikes,
    regardless of whether the next strike is loud or soft.

    Why this is useful:

      - The toms fill at 14.25/14.44/14.62 has strike 3
        ringing naturally to ~15.0s. With
        ``compute_duration_ms`` and a 14.84 next-event
        cap, strike 3 reports 180ms. With this function
        using the same 14.84 cap, it reports the time to
        the envelope MINIMUM before 14.84, which is the
        end of strike 3's natural ring.
      - If 14.84 is later filtered out as a click FP, the
        valley-finding automatically uses the next SURVIVING
        event's time as the right edge, so the duration
        extends to that next event's valley. (Note: the
        caller must re-measure with the new next_event_time
        to take advantage of this — the function itself
        just finds the valley within the [event, next_event]
        window.)

    The function returns the time from this event's peak
    to the envelope minimum, in milliseconds. Returns
    ``None`` if the peak can't be located.
    """
    times, env = _envelope_at_time(
        audio, sr, event_time_sec,
        n_fft=n_fft, hop=hop,
        broad_min_hz=broad_min_hz, broad_max_hz=broad_max_hz,
    )
    i_peak, _ = _find_attack_peak(times, env, event_time_sec)
    if i_peak is None:
        return None
    # The valley is the minimum of the envelope between
    # the peak and the next event's reported time.
    if next_event_time_sec <= event_time_sec:
        return None
    i_next = int(np.argmin(np.abs(times - next_event_time_sec)))
    i_lo = i_peak + 1
    i_hi = min(i_next, len(env) - 1)
    if i_hi <= i_lo:
        return None
    i_valley = i_lo + int(np.argmin(env[i_lo:i_hi]))
    duration_sec = times[i_valley] - times[i_peak]
    if duration_sec <= 0:
        return None
    return float(duration_sec * 1000.0)


def _find_attack_peak(
    times: np.ndarray,
    env: np.ndarray,
    event_time_sec: float,
    search_back_sec: float = 0.03,
    search_fwd_sec: float = 0.05,
) -> Tuple[Optional[int], Optional[float]]:
    """Locate the envelope peak in a small forward-biased
    window around ``event_time_sec``.

    Returns ``(i_peak, peak_val)`` or ``(None, None)`` if
    the search window is empty. The peak is the maximum
    of the linear-magnitude envelope.

    The window is intentionally asymmetric: up to 30ms
    before the event time (to handle the case where the
    detector reported the strike-onset time, which can
    precede the Hann-bias-shifted peak), and 50ms after
    (to catch the actual peak after Hann center bias).

    A SYMMETRIC search is wrong here. In a tight toms
    fill (strikes every 180ms), the previous strike's
    peak is only 100-150ms behind the current event
    time, and that previous peak is usually LARGER than
    the current one (rings haven't fully decayed). A
    symmetric ±100ms search will latch onto the
    previous strike's peak, not the current one.
    """
    dt = times[1] - times[0]
    i_center = int(np.argmin(np.abs(times - event_time_sec)))
    i_lo = max(0, i_center - int(search_back_sec / dt))
    i_hi = min(len(env), i_center + int(search_fwd_sec / dt))
    if i_hi <= i_lo:
        return None, None
    local_env = env[i_lo:i_hi]
    i_peak_local = int(np.argmax(local_env))
    i_peak = i_lo + i_peak_local
    return i_peak, float(env[i_peak])


def compute_root_pitch(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    fmin_hz: float = 60.0,
    fmax_hz: float = 2000.0,
    skip_ms: float = DEFAULT_ATTACK_SKIP_MS,
    body_window_ms: float = DEFAULT_PITCH_BODY_WINDOW_MS,
    method: str = 'pyin',
    voiced_prob: float = DEFAULT_PYIN_VOICED_PROB,
) -> Tuple[Optional[float], Optional[float]]:
    """Detect the root pitch (fundamental) of a percussive event.

    Skips the broadband attack (``skip_ms``) and runs YIN/pYIN
    on the body of the sound for ``body_window_ms``. Returns
    ``(pitch_hz, confidence)`` — confidence is 0-1 for pYIN
    (voiced probability) and a heuristic 0-1 for plain YIN
    (1.0 minus fraction of NaN frames).

    Returns ``(None, None)`` on failure (no librosa, segment
    too short, no confident pitch detected).

    The skip past the attack is critical. The attack is
    broadband noise — YIN running on the attack returns a
    random pitch (often mid-band noise) with no useful
    confidence. The body of a toms strike, 15-100ms after
    the stick hits, has a clean fundamental at the head mode
    (typically 80-200Hz for toms). Skipping 15ms lands us
    firmly in the body.

    Args:
        audio: mono audio array
        sr: sample rate
        event_time_sec: time of the onset
        fmin_hz, fmax_hz: pitch search range
        skip_ms: ms to skip past the attack (default 15)
        body_window_ms: analysis window length (default 300)
        method: 'yin' or 'pyin' (pYIN is more robust, gives
            a confidence; YIN is faster)
        voiced_prob: pYIN confidence threshold (default 0.5)

    Returns:
        (pitch_hz, confidence) — both None on failure.
        pitch_hz is the median of confident frames.
    """
    if not _HAS_LIBROSA:
        return None, None

    audio = _ensure_mono(audio)
    onset_sample = int(event_time_sec * sr) + int(skip_ms * sr / 1000.0)
    window_samples = int(body_window_ms * sr / 1000.0)
    if onset_sample + window_samples > len(audio):
        window_samples = len(audio) - onset_sample
    if window_samples < 512:
        return None, None

    segment = audio[onset_sample:onset_sample + window_samples]

    try:
        if method == 'pyin':
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment, fmin=fmin_hz, fmax=fmax_hz, sr=sr,
                frame_length=2048,
            )
            # pYIN returns NaN for unvoiced frames
            confident = f0[(voiced_flag) & (voiced_probs > voiced_prob) & (~np.isnan(f0))]
            if len(confident) == 0:
                return None, None
            pitch = float(np.median(confident))
            mean_prob = float(np.mean(voiced_probs[voiced_flag & (~np.isnan(f0))]))
            return pitch, mean_prob
        else:
            # Plain YIN: frame_length=2048 for ~21Hz resolution
            # at sr=44100 (good down to ~50Hz fundamentals).
            f0 = librosa.yin(
                segment, fmin=fmin_hz, fmax=fmax_hz, sr=sr,
                frame_length=2048,
            )
            valid = f0[~np.isnan(f0)]
            if len(valid) == 0:
                return None, None
            pitch = float(np.median(valid))
            # Heuristic confidence: fraction of valid frames.
            # YIN doesn't give a per-frame probability so this
            # is the best we can do without switching to pYIN.
            confidence = float(len(valid) / len(f0))
            return pitch, confidence
    except Exception:
        return None, None


def compute_decay_t60_ms(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    skip_ms: float = DEFAULT_ATTACK_SKIP_MS,
    body_window_ms: float = 600.0,
    fmin_hz: float = 200.0,
    fmax_hz: float = 4000.0,
) -> Optional[float]:
    """Estimate T60 (time for energy to drop 60dB) for a percussive event.

    Picks the LINEAR power in a characteristic frequency band
    (default 200-4000Hz, covers snare body and hihat), starts
    ``skip_ms`` after the onset, takes a log-linear fit
    (10 * log10(linear_power) vs time → straight line for
    exponential decay), and converts the slope to T60.

    Returns None if the fit fails (energy is constant or
    rising, the segment is too short, or the energy is
    already at the noise floor).

    Typical T60 values:
      - Closed hihat: 30-80ms (fast decay)
      - Open hihat: 200-400ms (sustained)
      - Snare: 80-250ms
      - Toms: 300-800ms
      - Cymbals: 800-2000ms
    """
    from .spectral_transient_core import compute_stft_db

    onset_sample = int(event_time_sec * sr) + int(skip_ms * sr / 1000.0)
    window_samples = int(body_window_ms * sr / 1000.0)
    if onset_sample + window_samples > len(audio):
        window_samples = len(audio) - onset_sample
    if window_samples < 512:
        return None

    audio_mono = _ensure_mono(audio)
    segment = audio_mono[onset_sample:onset_sample + window_samples]

    freqs, times, s_db = compute_stft_db(segment, sr, n_fft=1024, hop=256)
    mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    if not np.any(mask):
        return None
    # dB → linear power (use /10 for power, not /20 for
    # magnitude). Floor at 1e-20 to avoid log/0.
    S_power = np.maximum(10 ** (s_db / 10.0), 1e-20)
    # Sum power in the band — total energy per frame.
    env_power = S_power[mask].sum(axis=0)
    # Convert to dB for the log-linear fit. Skip frames
    # at the noise floor.
    env_db = 10.0 * np.log10(np.maximum(env_power, 1e-20))
    valid = env_db > -60.0
    if np.sum(valid) < 5:
        return None
    log_env = env_db[valid]
    dt = times[1] - times[0]
    t_valid = np.arange(len(log_env)) * dt

    # Linear fit: log_env = a + b * t  →  dB/s = b
    # 60dB drop takes 60/(-b) seconds
    try:
        a, b = np.polyfit(t_valid, log_env, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if b >= -0.5:
        # Energy is rising or barely falling (|slope| < 0.5 dB/s
        # means T60 > 120s — effectively sustained). Return
        # None rather than a bogus huge T60.
        return None
    t60_sec = 60.0 / (-b)
    return float(t60_sec * 1000.0)


def compute_spectral_centroid_hz(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    skip_ms: float = DEFAULT_ATTACK_SKIP_MS,
    body_window_ms: float = 200.0,
) -> Optional[float]:
    """Compute the spectral centroid (weighted-mean frequency) of the body.

    Brightness measure: high values (4-8kHz) suggest cymbals /
    hihat, low values (200-800Hz) suggest kick / toms. Useful
    as a feature for ML classification, less reliable as a
    standalone classifier (lots of overlap between instruments).

    Returns None if the segment is too short.
    """
    from .spectral_transient_core import compute_stft_db

    onset_sample = int(event_time_sec * sr) + int(skip_ms * sr / 1000.0)
    window_samples = int(body_window_ms * sr / 1000.0)
    if onset_sample + window_samples > len(audio):
        window_samples = len(audio) - onset_sample
    if window_samples < 512:
        return None

    audio_mono = _ensure_mono(audio)
    segment = audio_mono[onset_sample:onset_sample + window_samples]

    freqs, times, s_db = compute_stft_db(segment, sr, n_fft=1024, hop=256)
    # Convert dB to linear power for the centroid math
    # (centroid is a linear-power weighted mean, dB-centroid
    # would be the wrong math). Floor at 1e-12 to avoid log/0.
    S = np.maximum(10 ** (s_db / 20.0), 1e-12)
    # Sum over time to get a single mean spectrum
    mean_spectrum = S.mean(axis=1)
    if mean_spectrum.sum() <= 0:
        return None
    centroid = float(np.sum(freqs * mean_spectrum) / np.sum(mean_spectrum))
    return centroid


def compute_attack_rise_ms(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
    n_fft: int = 1024,
    hop: int = 256,
) -> Optional[float]:
    """Measure the rise time of the broadband attack (10%-90% of peak).

    Discriminates very fast transients (clicks ~1-2ms, sharp
    drumstick hits ~3-8ms) from slower onsets (mallet hits
    ~10-20ms, soft beaters ~20-50ms). Combined with
    ``duration_ms``, the (rise, ring) pair is a powerful
    classifier — clicks have low rise AND low ring, real
    strikes have variable rise but high ring.

    Returns None if the peak can't be located or the rise
    can't be bracketed (e.g. the audio starts mid-attack
    or the peak is barely above the surrounding noise).

    The math: on a LINEAR-magnitude envelope, 10% of peak
    = peak * 0.1 and 90% of peak = peak * 0.9. The
    dB equivalent is 20 dB below and 0.92 dB below peak
    respectively, but working in linear space keeps the
    thresholds intuitive.
    """
    times, env = _envelope_at_time(
        audio, sr, event_time_sec,
        n_fft=n_fft, hop=hop,
        broad_min_hz=broad_min_hz, broad_max_hz=broad_max_hz,
    )
    i_peak, peak_val = _find_attack_peak(times, env, event_time_sec)
    if i_peak is None or peak_val is None or peak_val <= 0:
        return None

    lo_thr = peak_val * 0.1   # 10% of peak
    hi_thr = peak_val * 0.9   # 90% of peak

    # Walk backward from the peak to find the 10% point.
    # The envelope is rising into the attack, so as we go
    # backward in time the envelope decreases.
    i_10 = i_peak
    while i_10 > 0 and env[i_10] > lo_thr:
        i_10 -= 1
    if i_10 == 0 and env[0] > lo_thr:
        # The 10% point is BEFORE the start of audio
        # (the envelope never drops below 10% of peak in
        # the analyzed window). Return None — we can't
        # measure rise without a starting reference.
        return None

    # Walk backward to find the 90% point.
    i_90 = i_peak
    while i_90 > i_10 and env[i_90] > hi_thr:
        i_90 -= 1
    if i_90 <= i_10:
        return None

    rise_sec = times[i_peak] - times[i_10]
    return float(rise_sec * 1000.0)


def compute_event_features(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    pitch_fmin_hz: float = 60.0,
    pitch_fmax_hz: float = 2000.0,
    pitch_method: str = 'pyin',
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
    next_event_time_sec: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Compute the full per-event feature battery.

    Convenience wrapper that runs all the individual feature
    functions and returns a flat dict suitable for direct
    attachment to a PGA (or any other) event dict. Features
    that fail return ``None`` — the caller decides what
    "missing" means (usually: don't trust the value).

    The returned dict has these keys (all may be None):
      - ``duration_ms``: ring time (peak to -20dB), capped
        by the inter-onset interval to next event. This is
        the "raw" duration measurement — accurate only when
        the next event is far enough away to let this one
        naturally decay. For tightly-clustered events, see
        ``duration_to_valley_ms`` below.
      - ``duration_to_valley_ms``: ring time to the envelope
        minimum between this event and the next. This is
        the "true physical ring" — the time until the
        silence between the two strikes. Unaffected by how
        loud or soft the next strike is. Requires
        ``next_event_time_sec`` to be set.
      - ``attack_rise_ms``: 10-90% rise time
      - ``root_pitch_hz``: fundamental via YIN/pYIN on body
      - ``pitch_confidence``: 0-1 (pYIN voiced_prob mean; YIN fraction-valid)
      - ``decay_t60_ms``: time for body energy to drop 60dB
      - ``spectral_centroid_hz``: weighted-mean frequency of body
      - ``inter_onset_ms``: time to next event (if provided);
        explicitly reported so the WebUI can show "duration
        was bounded by next event at X ms" alongside the
        measured ring time.

    Args:
        audio: mono or stereo audio array
        sr: sample rate
        event_time_sec: time of the onset in seconds
        pitch_fmin_hz, pitch_fmax_hz: pitch search range
        pitch_method: 'yin' or 'pyin'
        broad_min_hz, broad_max_hz: frequency band for
            duration/decay/centroid. Default 200-8000 covers
            toms/snare/hihat. Override for kick-specific
            work (e.g. 30-200Hz).
        next_event_time_sec: if provided, the duration
            walk-forward stops at this time and the
            ``inter_onset_ms`` field is set. Critical for
            clustered events (drum fills) where the next
            strike masks the current one before it can
            naturally decay. Also enables
            ``duration_to_valley_ms``.
    """
    audio_mono = _ensure_mono(audio)
    features: Dict[str, Optional[float]] = {
        'duration_ms': None,
        'duration_to_valley_ms': None,
        'attack_rise_ms': None,
        'root_pitch_hz': None,
        'pitch_confidence': None,
        'decay_t60_ms': None,
        'spectral_centroid_hz': None,
        'inter_onset_ms': None,
    }
    # Wrap each computation in try/except so a failure in
    # one feature doesn't poison the others. The individual
    # functions are already defensive (return None on most
    # errors), but a bug in librosa or numpy can still raise.
    try:
        features['duration_ms'] = compute_duration_ms(
            audio_mono, sr, event_time_sec,
            broad_min_hz=broad_min_hz, broad_max_hz=broad_max_hz,
            next_event_time_sec=next_event_time_sec,
        )
    except Exception:
        pass
    if next_event_time_sec is not None and next_event_time_sec > event_time_sec:
        try:
            features['duration_to_valley_ms'] = compute_duration_to_valley_ms(
                audio_mono, sr, event_time_sec,
                next_event_time_sec=next_event_time_sec,
                broad_min_hz=broad_min_hz, broad_max_hz=broad_max_hz,
            )
        except Exception:
            pass
        features['inter_onset_ms'] = float((next_event_time_sec - event_time_sec) * 1000.0)
    try:
        features['attack_rise_ms'] = compute_attack_rise_ms(
            audio_mono, sr, event_time_sec,
            broad_min_hz=broad_min_hz, broad_max_hz=broad_max_hz,
        )
    except Exception:
        pass
    try:
        pitch, conf = compute_root_pitch(
            audio_mono, sr, event_time_sec,
            fmin_hz=pitch_fmin_hz, fmax_hz=pitch_fmax_hz,
            method=pitch_method,
        )
        features['root_pitch_hz'] = pitch
        features['pitch_confidence'] = conf
    except Exception:
        pass
    try:
        features['decay_t60_ms'] = compute_decay_t60_ms(
            audio_mono, sr, event_time_sec,
            fmin_hz=broad_min_hz, fmax_hz=broad_max_hz,
        )
    except Exception:
        pass
    try:
        features['spectral_centroid_hz'] = compute_spectral_centroid_hz(
            audio_mono, sr, event_time_sec,
        )
    except Exception:
        pass
    return features


def compute_event_features_for_list(
    audio: np.ndarray,
    sr: int,
    event_times_sec: list,
    pitch_fmin_hz: float = 60.0,
    pitch_fmax_hz: float = 2000.0,
    pitch_method: str = 'pyin',
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
) -> list:
    """Compute features for a list of event times, with
    inter-onset intervals computed automatically.

    Returns a list of feature dicts, one per event time,
    in the same order. Each dict has all the fields from
    :func:`compute_event_features`, plus
    ``next_event_time_sec`` is computed from the list (the
    next event's time in the list, or ``None`` for the
    last event).

    This is the right entry point when you have a list of
    events and want the natural "next event = next item
    in list" semantics. For two-pass flows (detect →
    filter → re-measure with filtered neighbors), call
    this function once with the FULL list, then again
    with the FILTERED list, and overwrite the
    ``duration_to_valley_ms`` field on the survivors.
    """
    out = []
    for i, t in enumerate(event_times_sec):
        next_t = event_times_sec[i + 1] if i + 1 < len(event_times_sec) else None
        feats = compute_event_features(
            audio, sr, t,
            pitch_fmin_hz=pitch_fmin_hz, pitch_fmax_hz=pitch_fmax_hz,
            pitch_method=pitch_method,
            broad_min_hz=broad_min_hz, broad_max_hz=broad_max_hz,
            next_event_time_sec=next_t,
        )
        out.append(feats)
    return out
