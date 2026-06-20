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
from functools import wraps

import numpy as np
import os

from .stft_utils import timed


# Set LARSNET_TIMING=1 in environment to opt in to [t+Xs] timing logs.
_TIMING_ENABLED = os.environ.get("LARSNET_TIMING", "0") == "1"


def _log_timing(name: str):
    """Decorator: wrap the function body in a ``timed(name)`` block so
    each call is logged with [t+Xs] elapsed and cumulative stats.

    Used to find WHICH functions dominate CLI runtime — every call is
    logged, so a function called 100×/sec will produce 100 log lines.
    Quiet down by unsetting LARSNET_TIMING (the default) or removing
    the decorator if the noise is too much.
    """
    def deco(func):
        if not _TIMING_ENABLED:
            return func
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timed(name):
                return func(*args, **kwargs)
        return wrapper
    return deco

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


@_log_timing("_envelope_at_time")
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

    Note: ``t_sec`` is part of the signature for API symmetry with
    callers that pass an event time, but the returned envelope
    covers the full audio regardless of ``t_sec``. Callers index
    into the returned ``env`` array at the frame(s) near
    ``t_sec``. This means the function's output depends ONLY on
    ``(id(audio), sr, n_fft, hop, broad_min_hz, broad_max_hz)``
    — repeated calls with different ``t_sec`` produce identical
    results, so the result is memoized in ``_ENVELOPE_CACHE`` to
    avoid redoing the 7.7M-element ``10**(s_db/20)`` conversion
    on every call. Without the cache, a 47-event run does ~140
    envelope conversions (~15s of cum time) when only 1 is needed.
    """
    # Cache lookup: ``id(audio)`` alone is fragile because Python
    # reuses ids once the original object is GC'd. So we also
    # store the audio reference in the cached entry and verify
    # identity on lookup — if a different audio object happens
    # to land at the same memory address (test isolation race,
    # long-running process), we evict and recompute. Without
    # this guard the cache can return a previous test's envelope
    # with a different ``broad_*`` band, silently breaking
    # duration_ms on toms audio.
    cache_key = (id(audio), sr, n_fft, hop, broad_min_hz, broad_max_hz)
    cached = _ENVELOPE_CACHE.get(cache_key)
    if cached is not None:
        cached_audio, result = cached
        if cached_audio is audio:
            return result
    from .stft_utils import compute_stft_db
    freqs, times, s_db = compute_stft_db(audio, sr, n_fft=n_fft, hop=hop)
    # dB → linear: S = 10^(dB/20) for magnitude, but for
    # power it's 10^(dB/10). STFT output is magnitude (dB
    # amplitude), so use /20. Floor at 1e-12 to avoid log/0.
    S_linear = np.maximum(10 ** (s_db / 20.0), 1e-12)
    # Sum linear power in the broad range. This is the
    # total broadband energy per frame, in linear units.
    mask = (freqs >= broad_min_hz) & (freqs <= broad_max_hz)
    env = S_linear[mask].sum(axis=0)
    result = (times, env)
    _ENVELOPE_CACHE[cache_key] = (audio, result)
    return result


# Memoization cache for _envelope_at_time. Keyed on the args that
# actually affect output (NOT t_sec — see the function docstring).
# Each entry stores ``(audio_ref, result)`` so the lookup can
# verify the cached audio is the same object (defends against
# ``id(audio)`` reuse across tests / long-running processes).
# Without this cache, a 47-event run does ~140 STFT-derived
# envelope conversions (~15s of cum time) when only 1 is needed.
# Cleared by tests via ``_ENVELOPE_CACHE.clear()``.
_ENVELOPE_CACHE: Dict[tuple, tuple] = {}


@_log_timing("compute_duration_ms")
def compute_duration_ms(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
    duration_broad_min_hz: float = 30.0,
    duration_broad_max_hz: float = 8000.0,
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
            the envelope. Default 200-8000 (the original
            "broadband" definition used by other features
            like attack_rise/decay/centroid).
        duration_broad_min_hz, duration_broad_max_hz: SEPARATE
            frequency band used specifically for the duration
            envelope. Default 30-8000 — wide enough to see
            the toms fundamental (65-85Hz) and its sub-bass
            ring, so duration_ms reflects the actual ring
            rather than truncating at the first broadband
            zero-crossing. The duration band is decoupled
            from ``broad_*`` so other features (which benefit
            from a narrower band) are unaffected.
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
        broad_min_hz=duration_broad_min_hz, broad_max_hz=duration_broad_max_hz,
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
    duration_sec_slope = times[min(i_end, i_cap)] - times[i_peak]
    # FALLBACK for slow-decaying sub-bass rings (2026-06-12):
    # The slope-based approach exits early when the ring's
    # slope is shallower than min_slope_db_per_s (e.g. a
    # 75Hz toms ring with ~1.75s exp decay has a slope of
    # ~-5 dB/s, which is above the -10 dB/s threshold).
    # In that case, the algorithm reports just the attack
    # duration (~17ms) instead of the true ring. Detect
    # this case: if the slope-based duration is suspiciously
    # short (< 50ms) but the envelope still has significant
    # energy well past the peak, fall back to an RMS-threshold
    # approach. The RMS approach finds the frame where the
    # envelope first drops below ``rms_threshold_frac`` of
    # the peak. This is the "easy off the shelf" approach
    # the user requested — it measures the actual ring
    # length even for slow-decaying sub-bass content.
    rms_threshold_frac = 0.005  # 0.5% of peak
    if duration_sec_slope * 1000.0 < 50.0:
        # Check if there's significant energy past the peak
        post_peak = env[i_peak:]
        if len(post_peak) > 0 and post_peak.max() > 0:
            peak_val = post_peak[0]
            threshold = rms_threshold_frac * peak_val
            below = np.where(post_peak < threshold)[0]
            if len(below) > 0:
                i_end_rms = below[0]
                i_end_rms = min(i_end_rms, i_cap - i_peak)
                if i_end_rms > 0:
                    duration_sec_rms = times[i_peak + i_end_rms] - times[i_peak]
                    if duration_sec_rms > duration_sec_slope:
                        return float(duration_sec_rms * 1000.0)
    if duration_sec_slope <= 0:
        return None
    return float(duration_sec_slope * 1000.0)


@_log_timing("compute_duration_to_valley_ms")
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


@_log_timing("compute_root_pitch")
def compute_root_pitch(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    fmin_hz: float = 60.0,
    fmax_hz: float = 2000.0,
    skip_ms: float = DEFAULT_ATTACK_SKIP_MS,
    body_window_ms: float = DEFAULT_PITCH_BODY_WINDOW_MS,
    method: str = 'yin',
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

    ``method`` must be ``'yin'`` or ``'pyin'`` (anything else raises
    ``ValueError``). Default is ``'yin'`` — much faster than pYIN
    (5-10×) and produces equivalent pitch estimates for our use
    case. The user switched from pYIN to YIN after observing ~8.5s
    of cumulative pYIN time on a 47-event toms run; pitch
    resolution downstream was unaffected.

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
    if method not in ('yin', 'pyin'):
        # Previously this silently fell through to the YIN branch
        # for any method != 'pyin' (e.g. typos like 'pYIN', 'YIN',
        # ''), making config misconfigurations hard to spot. The
        # config schema documents the legal values as
        # ``pitch_method: 'yin' or 'pyin'`` — enforce that contract.
        raise ValueError(
            f"compute_root_pitch: method must be 'yin' or 'pyin', "
            f"got {method!r}"
        )
    onset_sample = int(event_time_sec * sr) + int(skip_ms * sr / 1000.0)
    window_samples = int(body_window_ms * sr / 1000.0)
    if onset_sample + window_samples > len(audio):
        window_samples = len(audio) - onset_sample
    if window_samples < 512:
        return None, None

    segment = audio[onset_sample:onset_sample + window_samples]

    # Energy pre-check: YIN doesn't have a probabilistic voiced/
    # unvoiced model like pYIN, so on a silent (or near-silent)
    # segment it returns a spurious pitch at the search-range
    # boundary (~max(fmin, fmax) — observed 2004Hz on silence
    # with fmin=60, fmax=2000). pYIN would correctly return NaN
    # and we map that to None; YIN needs an explicit RMS gate.
    # Threshold 1e-4 chosen empirically: clearly above the
    # numerical noise of float32 silence (~1e-7) and below a
    # quiet toms ring's body energy (~1e-3 for a real strike).
    if float(np.max(np.abs(segment))) < 1e-4:
        return None, None

    try:
        if method == 'pyin':
            # Adaptive frame_length: pYIN needs at least 2
            # periods of fmin to fit in the frame. At 30Hz
            # and sr=44100, that's ~2940 samples. We round
            # up to the next power of 2 (4096) to keep the
            # FFT efficient. (2026-06-12: widened from 2048
            # to support fmin down to 30Hz for low toms.)
            frame_length = max(2048, int(2 ** np.ceil(np.log2(2 * sr / fmin_hz))))
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment, fmin=fmin_hz, fmax=fmax_hz, sr=sr,
                frame_length=frame_length,
            )
            # pYIN returns NaN for unvoiced frames. Prefer
            # confident frames (voiced_prob > threshold) but
            # fall back to all voiced frames if none meet the
            # threshold — low-confidence pitches are still
            # useful for diagnostic purposes (e.g. a 75Hz
            # toms ring with slow decay may have voiced_prob
            # ~0.1-0.2 but a clear fundamental at 71-81Hz).
            # (2026-06-12: accept low confidence rather than
            # returning None.)
            voiced_mask = voiced_flag & (~np.isnan(f0))
            confident = f0[voiced_mask & (voiced_probs > voiced_prob)]
            if len(confident) > 0:
                pitch = float(np.median(confident))
            elif np.any(voiced_mask):
                # Fallback: use all voiced frames even if
                # confidence is below threshold.
                pitch = float(np.median(f0[voiced_mask]))
            else:
                return None, None
            mean_prob = float(np.mean(voiced_probs[voiced_mask]))
            return pitch, mean_prob
        else:
            # Plain YIN: same adaptive frame_length as pYIN.
            frame_length = max(2048, int(2 ** np.ceil(np.log2(2 * sr / fmin_hz))))
            f0 = librosa.yin(
                segment, fmin=fmin_hz, fmax=fmax_hz, sr=sr,
                frame_length=frame_length,
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


@_log_timing("compute_decay_t60_ms")
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
    from .stft_utils import compute_stft_db

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


@_log_timing("compute_spectral_centroid_hz")
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
    from .stft_utils import compute_stft_db

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


@_log_timing("compute_spectral_flatness")
def compute_spectral_flatness(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    flat_min_hz: float = 600.0,
    flat_max_hz: float = 3000.0,
    body_window_ms: float = 30.0,
    skip_ms: float = 0.0,
    n_fft: int = 1024,
    hop: int = 256,
) -> Optional[float]:
    """Spectral flatness of the attack region (2026-06-11).

    Flatness = geometric mean of the (linear) magnitude
    spectrum divided by the arithmetic mean, computed over
    a narrow time window centered at the event time and
    restricted to a frequency band.

    This is the textbook spectral-flatness definition
    (Johnston 1988), normalized to [0, 1]:

      * 1.0 = perfectly flat (white-noise-like spectrum)
      * ~0.0 = highly tonal (single-tone or harmonic
        stack — geometric mean of a peaked spectrum is
        much smaller than its arithmetic mean)

    For drum attacks specifically, we expect a DIFFERENT
    signature than for sustained tones:

      * Real toms/snare/kick strikes have a strong low
        fundamental + harmonics — TONAL signature, low
        flatness (geometric mean << arithmetic mean).
      * "Pop" or "click" artifacts (one-frame transient
        noise bursts) have broadband energy spread over
        many bins with no clear fundamental — high
        flatness (geometric mean ≈ arithmetic mean).

    We restrict the band to [flat_min_hz, flat_max_hz]
    (default 600-3000 Hz) to focus on the "attack
    transient" range. The full 0-22050 Hz spectrum is
    dominated by low-frequency toms rings; the 3-8 kHz
    "air" range is dominated by hihat/cymbal bleed; only
    the 600-3000 Hz band is mostly percussive attack
    content for this kind of music.

    The diagnostic value here is a PER-EVENT property
    (computed for each event that survives the prominence
    filter), not a filter threshold. Attaching the raw
    value to the sidecar lets the user (or a future
    classifier) see what the pipeline sees. No magic
    numbers — the value is whatever the math gives.

    Args:
        audio: mono or stereo audio array
        sr: sample rate
        event_time_sec: time of the onset
        flat_min_hz, flat_max_hz: inclusive Hz band for
            the flatness calculation. Default 600-3000 Hz.
        body_window_ms: length of audio segment to take
            the STFT over. Default 30 ms (~ 5 STFT frames
            at hop=256) — matches the default for
            ``compute_spectral_centroid_hz``; long enough
            to span the attack transient, short enough
            not to include the ring.
        skip_ms: skip the first N ms of the segment
            (useful for attacks with leading noise).
            Default 0 — start at the onset sample itself.
        n_fft, hop: STFT parameters. Defaults match the
            rest of the larsnet pipeline (see
            ``compute_stft_db``).

    Returns:
        float in [0, 1] or None if the segment is too
        short or the spectrum is silent.
    """
    from .stft_utils import compute_stft_db

    onset_sample = int(event_time_sec * sr) + int(skip_ms * sr / 1000.0)
    window_samples = int(body_window_ms * sr / 1000.0)
    if onset_sample + window_samples > len(audio):
        window_samples = len(audio) - onset_sample
    if window_samples < 512:
        return None

    audio_mono = _ensure_mono(audio)
    segment = audio_mono[onset_sample:onset_sample + window_samples]

    try:
        freqs, times, s_db = compute_stft_db(segment, sr, n_fft=n_fft, hop=hop)
    except ValueError:
        # Segment shorter than n_fft (e.g. very low sample
        # rate or audio at the tail). Flatness is
        # undefined here — return None rather than raise.
        return None
    # Convert dB to linear magnitude for the flatness math
    # (flatness is defined on linear magnitude). Floor at
    # 1e-12 to avoid log/0 — the geometric mean is computed
    # via exp(mean(log(x))), so a single zero would zero
    # the whole mean.
    S = np.maximum(10 ** (s_db / 20.0), 1e-12)
    # Restrict to the [flat_min_hz, flat_max_hz] band
    band_mask = (freqs >= flat_min_hz) & (freqs <= flat_max_hz)
    if not band_mask.any():
        return None
    band_spec = S[band_mask, :]
    # Mean over time to get a single mean spectrum for the
    # whole attack region
    mean_spectrum = band_spec.mean(axis=1)
    arith_mean = mean_spectrum.mean()
    # Below ~120 dB in linear terms (1e-6), the band is
    # effectively silent — the spectrum is just the
    # numerical FFT floor. Reporting flatness ≈ 1.0 here
    # would be misleading (would look like a white-noise
    # event). Return None instead.
    if arith_mean < 1e-6:
        return None
    # Flatness = (∏ x_i)^(1/N) / (1/N ∑ x_i)
    # Implemented in log space for numerical safety:
    #   log(geo) = mean(log(x_i))
    #   flatness = exp(log(geo) - log(arith))
    log_geo_mean = np.log(mean_spectrum).mean()
    log_arith_mean = np.log(arith_mean)
    flatness = float(np.exp(log_geo_mean - log_arith_mean))
    # Clamp to [0, 1] for cleanliness — by construction
    # geo ≤ arith for non-negative x with equality iff all
    # bins are equal, so the value is always in [0, 1]
    # but floating-point noise can push it slightly outside.
    if flatness < 0.0:
        flatness = 0.0
    elif flatness > 1.0:
        flatness = 1.0
    return flatness


@_log_timing("compute_high_res_decay_signature")
def compute_high_res_decay_signature(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    broad_min_hz: float = 600.0,
    broad_max_hz: float = 8000.0,
    attack_window_frames: int = 30,
    decay_window_frames: int = 200,
    n_fft: int = 128,
    hop: int = 4,
) -> Optional[Dict[str, Optional[float]]]:
    """High-resolution attack + decay signature of an event (2026-06-11).

    Computes a per-event signature on a HIGH-RESOLUTION STFT
    (default n_fft=128, hop=4 — 0.091ms / frame, ~5.8 kHz / bin
    at sr=44100) over a 200ms window centered on the event.
    The PGA detector's standard 1024/256 STFT smears the
    attack over 10+ ms and misses single-frame transients;
    this function gives sub-frame visibility into the
    attack+ring region.

    The signature has two key fields:

      ``decay_energy_15ms``: sum of the contrast envelope in
        the 15ms window starting ~3ms after the attack peak.
        A real toms strike has a sustained decaying ring
        (the toms body resonates for 100ms+); a single-frame
        noise pop has no ring, so the decay-window envelope
        is near zero. Empirically (project 4 calibration):
        FPs < 60K, real strikes > 60K. NOT a filter — a
        diagnostic.

      ``decay_col_min_median_db``: median of the per-frame
        col_min over the same 15ms decay window. col_min
        is the lowest-energy bin in the spectrum at each
        frame. A sustained broadband ring (real strike)
        keeps col_min elevated; a noise pop has col_min at
        the noise floor (~-80 to -90 dB). Empirically:
        FPs -84 to -90 dB, real strikes -60 to -84 dB.

    Also returns the high-res peak time and offset, for
    callers who want to investigate "the PGA event is at
    T but the high-res view shows a peak at T+5ms" (PGA
    envelope smearing) or "the high-res view has no
    peak in [T-1ms, T+15ms]" (PGA hallucination).

    Args:
        audio: mono or stereo audio array
        sr: sample rate
        event_time_sec: time of the onset
        broad_min_hz, broad_max_hz: frequency band for the
            contrast envelope (same as the PGA detector
            defaults).
        attack_window_frames, decay_window_frames: how many
            high-res frames after the peak to sum the
            envelope over. Defaults: 30 frames (≈2.7ms)
            attack, 200 frames (≈18ms) decay.
        n_fft, hop: high-res STFT. Defaults 128/4 are
            validated on project 4 — coarser STFT (256/64)
            re-introduces the smearing problem; finer STFT
            (64/2) is 16x slower without adding signal.

    Returns:
        dict with keys: hr_peak_time, hr_peak_offset_ms,
        hr_peak_envelope, decay_envelope_energy,
        decay_col_min_median_db. All may be None on failure.
    """
    from .stft_utils import compute_stft_db

    # Extract audio window: 10ms before, 200ms after the event
    t_start = max(0.0, event_time_sec - 0.010)
    t_end = min(len(audio) / sr, event_time_sec + 0.200)
    start_sample = int(t_start * sr)
    end_sample = int(t_end * sr)
    if end_sample - start_sample < n_fft:
        return None
    audio_mono = _ensure_mono(audio)
    segment = audio_mono[start_sample:end_sample]

    try:
        freqs, times, s_db = compute_stft_db(segment, sr, n_fft=n_fft, hop=hop)
    except ValueError:
        # Segment shorter than n_fft
        return None

    abs_times = t_start + times
    pga_frame_local = int(round((event_time_sec - t_start) * sr / hop))
    pga_frame_local = max(0, min(pga_frame_local, s_db.shape[1] - 1))

    # Build the contrast envelope in [broad_min_hz, broad_max_hz]
    # (same recipe as the PGA detector)
    freq_mask = (freqs >= broad_min_hz) & (freqs <= broad_max_hz)
    if not freq_mask.any():
        return None
    band_db = s_db[freq_mask, :]
    floor = np.percentile(band_db, 5, axis=1, keepdims=True)
    contrast = np.maximum(band_db - floor, 0)
    envelope = contrast.sum(axis=0)

    # Search for the high-res peak in [PGA - 5 frames, PGA + 200 frames]
    # (5 frames = 0.45ms before PGA time; 200 frames = 18ms after.
    #  Empirically the real strike is 5-11ms LATE vs the PGA
    #  report — see project 4 calibration. The +200 frame search
    #  bound covers the worst observed case.)
    search_start = max(0, pga_frame_local - 5)
    search_end = min(len(envelope), pga_frame_local + 300)
    if search_end <= search_start:
        return None
    search_env = envelope[search_start:search_end]
    peak_in_search = int(np.argmax(search_env))
    peak_frame = search_start + peak_in_search
    peak_time = float(abs_times[peak_frame])
    peak_env = float(envelope[peak_frame])

    # Split the post-peak region into attack (first 30 frames ≈ 2.7ms)
    # and decay (next 200 frames ≈ 18ms). The attack window covers
    # the initial impulse; the decay window covers the ring.
    attack_end = min(len(envelope), peak_frame + attack_window_frames)
    decay_end = min(len(envelope), attack_end + decay_window_frames)

    if attack_end < peak_frame or decay_end <= attack_end:
        return {
            'hr_peak_time': peak_time,
            'hr_peak_offset_ms': (peak_time - event_time_sec) * 1000.0,
            'hr_peak_envelope': peak_env,
            'decay_envelope_energy': 0.0,
            'decay_col_min_median_db': None,
        }

    attack_energy = float(envelope[peak_frame:attack_end].sum())
    decay_energy = float(envelope[attack_end:decay_end].sum())
    # col_min over the decay window — the median (not the mean)
    # is robust to a single bright bin.
    per_frame_min = s_db.min(axis=0)
    decay_col_min_median = float(np.median(per_frame_min[attack_end:decay_end]))

    return {
        'hr_peak_time': peak_time,
        'hr_peak_offset_ms': (peak_time - event_time_sec) * 1000.0,
        'hr_peak_envelope': peak_env,
        'attack_envelope_energy': attack_energy,
        'decay_envelope_energy': decay_energy,
        'decay_col_min_median_db': decay_col_min_median,
    }


@_log_timing("compute_attack_rise_ms")
def compute_attack_rise_ms(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
    n_fft: int = 1024,
    hop: int = 256,
    prev_event_time_sec: Optional[float] = None,
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

    Boundary (2026-06-18): ``prev_event_time_sec`` bounds the
    backward walk. The 10%-point search walks backward from
    the peak looking for where the envelope first drops below
    10% of the peak. Without a previous-event boundary, a
    ringing previous hit keeps the envelope above 10% of the
    new peak all the way back to the previous hit's body, and
    the 10% point gets pinned far back — producing an
    ``attack_rise_ms`` that's effectively ``inter_onset_ms``
    instead of the new hit's own rise. With ``prev_event_time_sec``
    set, the walk stops at that frame (using ``np.argmin`` to
    snap to the nearest envelope sample). If the envelope at
    that frame is STILL above 10% of the new peak (the gap
    between hits never dropped low enough), ``attack_rise_ms``
    returns ``None`` — we can't bracket the rise without a
    clear floor in the analysis window.

    The first event in a stream has no predecessor; pass
    ``prev_event_time_sec=None`` (the default) for that case.
    The function then walks all the way back to the start of
    the analyzed envelope. Callers that care about per-event
    accuracy on dense streams (snare, fast hihats) should
    always pass the previous event's time.
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

    # Resolve the backward-walk lower bound. When a previous
    # event is given, the walk stops at the envelope sample
    # nearest that time — the 10% point can't land inside
    # the previous hit's body or attack.
    if prev_event_time_sec is not None and prev_event_time_sec > 0:
        i_prev = int(np.argmin(np.abs(times - prev_event_time_sec)))
        i_prev = max(0, min(len(env) - 1, i_prev))
    else:
        i_prev = 0

    # Walk backward from the peak to find the 10% point.
    # The envelope is rising into the attack, so as we go
    # backward in time the envelope decreases.
    i_10 = i_peak
    while i_10 > i_prev and env[i_10] > lo_thr:
        i_10 -= 1
    if i_10 == i_prev and env[i_prev] > lo_thr:
        # The 10% point is BEFORE our backward-walk lower
        # bound — either the start of audio (no prev event)
        # or the previous event's time (previous hit is still
        # ringing so the envelope never dropped below 10% of
        # this hit's peak in the gap). Return None — we can't
        # measure rise without a starting reference in the
        # analysis window.
        return None

    # Walk backward to find the 90% point.
    i_90 = i_peak
    while i_90 > i_10 and env[i_90] > hi_thr:
        i_90 -= 1
    if i_90 <= i_10:
        return None

    rise_sec = times[i_peak] - times[i_10]
    return float(rise_sec * 1000.0)


@_log_timing("compute_event_features")
def compute_event_features(
    audio: np.ndarray,
    sr: int,
    event_time_sec: float,
    enable_pitch_detection: bool = True,
    pitch_fmin_hz: float = 60.0,
    pitch_fmax_hz: float = 250.0,
    pitch_method: str = 'yin',
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
    duration_broad_min_hz: float = 30.0,
    duration_broad_max_hz: float = 8000.0,
    next_event_time_sec: Optional[float] = None,
    prev_event_time_sec: Optional[float] = None,
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
        minimum between this event and the next. This is the
        "true physical ring" — the time until the
        silence between the two strikes. Unaffected by how
        loud or soft the next strike is. Requires
        ``next_event_time_sec`` to be set.
      - ``attack_rise_ms``: 10-90% rise time. When
        ``prev_event_time_sec`` is provided, the backward
        walk is bounded to that frame so a ringing previous
        hit can't stretch the rise time across hits
        (2026-06-18). Returns ``None`` when the envelope at
        the previous event is still above 10% of the new
        peak — the gap never dropped low enough to bracket
        a true new-attack rise.
      - ``pitch_hz``: fundamental via YIN/pYIN on body
      - ``pitch_confidence``: 0-1 (pYIN voiced_prob mean; YIN fraction-valid)
      - ``decay_t60_ms``: time for body energy to drop 60dB
      - ``spectral_centroid_hz``: weighted-mean frequency of body
      - ``spectral_flatness``: 0-1 broadband-flatness in 600-3000 Hz
        attack region (1=white-noise-like, 0=tonal). Diagnostic only;
        not used as a filter. Useful for spotting pop/click artifacts
        vs real strikes.
      - ``hr_peak_offset_ms``: time delta (ms) between the PGA-reported
        event time and the high-res (n_fft=128, hop=4) envelope peak.
        Real strikes tend to have offset ~5-11ms (PGA envelope smears
        the attack over 10+ ms); FPs sometimes have NO high-res peak
        at all. Diagnostic only.
      - ``decay_envelope_energy``: high-res envelope energy in the
        15ms window starting ~3ms after the high-res peak. Real
        strikes have a sustained decaying ring; noise pops have
        no ring. Empirically (project 4): FPs < 60K, real > 60K.
        Diagnostic only.
      - ``decay_col_min_median_db``: median col_min over the same
        15ms decay window. FPs sit at the noise floor (-84 to -90 dB);
        real strikes show elevated broadband energy (-60 to -84 dB).
        Diagnostic only.
      - ``inter_onset_ms``: time to next event (if provided);
        explicitly reported so the WebUI can show "duration
        was bounded by next event at X ms" alongside the
        measured ring time.

    Args:
        audio: mono or stereo audio array
        sr: sample rate
        event_time_sec: time of the onset in seconds
        pitch_fmin_hz, pitch_fmax_hz: pitch search range.
            Default 30-4000. Lower bound widened to 30Hz so
            pYIN can see low toms fundamentals (65-85Hz);
            upper bound widened to 4000Hz so the search
            covers the full body spectrum.
        pitch_method: 'yin' or 'pyin'
        broad_min_hz, broad_max_hz: frequency band used by
            attack_rise, decay, centroid, etc. Default
            200-8000 covers toms/snare/hihat. Override for
            kick-specific work (e.g. 30-200Hz). NOTE: this
            is NOT the band used for ``duration_ms`` — see
            ``duration_broad_*`` below.
        duration_broad_min_hz, duration_broad_max_hz: SEPARATE
            band for the duration envelope. Default 30-8000.
            Wider than ``broad_*`` so the duration walk-forward
            can see the toms fundamental (65-85Hz) and its
            sub-bass ring. Tuned to keep other features on the
            narrower, more meaningful band while still letting
            duration_ms reflect the true physical ring.
        next_event_time_sec: if provided, the duration
            walk-forward stops at this time and the
            ``inter_onset_ms`` field is set. Critical for
            clustered events (drum fills) where the next
            strike masks the current one before it can
            naturally decay. Also enables
            ``duration_to_valley_ms``.
        prev_event_time_sec: if provided, the
            ``attack_rise_ms`` backward walk stops at this
            frame (2026-06-18). Without it, a ringing
            previous hit can stretch the new hit's rise
            time across the entire inter-onset gap
            (symptom: ``attack_rise_ms`` ≈ ``inter_onset_ms``
            on snare / dense hihats). Pass the previous
            event's time when calling from a multi-event
            context; pass ``None`` for the first event in
            a stream (or any event whose predecessor is
            unknown).
    """
    audio_mono = _ensure_mono(audio)
    features: Dict[str, Optional[float]] = {
        'duration_ms': None,
        'duration_to_valley_ms': None,
        'attack_rise_ms': None,
        'pitch_hz': None,
        'pitch_confidence': None,
        'decay_t60_ms': None,
        'spectral_centroid_hz': None,
        'spectral_flatness': None,
        'hr_peak_offset_ms': None,
        'decay_envelope_energy': None,
        'decay_col_min_median_db': None,
        'inter_onset_ms': None,
    }
    # Wrap each computation in try/except so a failure in
    # one feature doesn't poison the others. The individual
    # functions are already defensive (return None on most
    # errors), but a bug in librosa or numpy can still raise.
    try:
        features['duration_ms'] = compute_duration_ms(
            audio_mono, sr, event_time_sec,
            broad_min_hz=duration_broad_min_hz,
            broad_max_hz=duration_broad_max_hz,
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
            prev_event_time_sec=prev_event_time_sec,
        )
    except Exception:
        pass
    # Skip pitch detection entirely when disabled (e.g. a stem
    # config sets enable_pitch_detection=false). This saves the
    # ~150ms/event YIN/pYIN call when we don't need pitch —
    # significant on a 47-event run (was ~8.5s before the default
    # switched from pYIN to YIN).
    if enable_pitch_detection:
        try:
            pitch, conf = compute_root_pitch(
                audio_mono, sr, event_time_sec,
                fmin_hz=pitch_fmin_hz, fmax_hz=pitch_fmax_hz,
                method=pitch_method,
            )
            features['pitch_hz'] = pitch
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
    try:
        features['spectral_flatness'] = compute_spectral_flatness(
            audio_mono, sr, event_time_sec,
        )
    except Exception:
        pass
    # High-res (n_fft=128, hop=4) attack+decay signature.
    # Different STFT parameters than the rest of the
    # pipeline — this one is fine enough to see single-frame
    # transients and the 5-15ms ring that distinguishes
    # real strikes from "pop" artifacts. See project 4
    # calibration for the empirical thresholds (FPs have
    # decay_envelope_energy < 60K, real strikes > 60K).
    try:
        hr_sig = compute_high_res_decay_signature(
            audio_mono, sr, event_time_sec,
        )
        if hr_sig is not None:
            features['hr_peak_offset_ms'] = hr_sig.get('hr_peak_offset_ms')
            features['decay_envelope_energy'] = hr_sig.get('decay_envelope_energy')
            features['decay_col_min_median_db'] = hr_sig.get('decay_col_min_median_db')
    except Exception:
        pass
    return features


@_log_timing("compute_event_features_for_list")
def compute_event_features_for_list(
    audio: np.ndarray,
    sr: int,
    event_times_sec: list,
    enable_pitch_detection: bool = True,
    pitch_fmin_hz: float = 60.0,
    pitch_fmax_hz: float = 250.0,
    pitch_method: str = 'yin',
    broad_min_hz: float = 200.0,
    broad_max_hz: float = 8000.0,
    duration_broad_min_hz: float = 30.0,
    duration_broad_max_hz: float = 8000.0,
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
            enable_pitch_detection=enable_pitch_detection,
            pitch_fmin_hz=pitch_fmin_hz, pitch_fmax_hz=pitch_fmax_hz,
            pitch_method=pitch_method,
            broad_min_hz=broad_min_hz, broad_max_hz=broad_max_hz,
            duration_broad_min_hz=duration_broad_min_hz,
            duration_broad_max_hz=duration_broad_max_hz,
            next_event_time_sec=next_t,
        )
        out.append(feats)
    return out
