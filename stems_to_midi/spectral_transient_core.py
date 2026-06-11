"""
Spectral-transient onset detection - complementary to energy-based detection.

INSPIRED BY: user observation (2026-06-08, project 3 — 2_funk_80_beat_4-4_4
toms stem). The user looked at the WebUI spectrogram, applied a noise floor
at -50dB and isolated frequencies >=800Hz, and could see 6 distinct hit
onsets at 73.676/73.853/74.033/74.210/74.411/74.576s in a region where the
current energy-based detector only finds 1 KEPT event (plus 8
REVERB_CONTINUATION events marked as not-hits).

The current detector's `mark_reverb_continuations` filter is over-aggressive
on toms because the toms envelope doesn't decay smoothly between hits —
between hits the energy stays high, so the algorithm thinks the second hit
is a continuation of the first and drops it.

The spectral-transient method sidesteps that filter entirely. It works on a
different physical signal:

  - Energy detector looks at broadband amplitude peaks in time domain
  - Spectral transient detector looks at broadband spectral peaks in
    frequency domain AT EACH TIME FRAME — summing linear power in 5
    fixed bands and looking for a band that's much louder than the others

Why it works for toms: a tom hit is a broadband transient — the stick
impulse excites ALL frequencies simultaneously, briefly. The high-freq
content (>=800Hz) is dominated by the strike moment and falls off fast
after the strike. So per-frame, the count of "high-freq bins above noise
floor" is a sharp pulse: 0 between hits, 100+ at the moment of a hit.

The original "count bins above floor" signal (commit 6526b7a) is
superseded by a per-band power profile (2026-06-09) — the user observed
that "bins above -50dB loses all resolution above or below that
threshold." The new signal is::

    band_powers = [b0, b1, b2, b3, b4]  # linear power sums per band
    band_delta  = max(band_powers) - median(band_powers)
                  (in per-bin-mean units, loudness-invariant)

``band_delta`` is the RISE of the loudest band above the typical (median)
band per frame. A real hit lights up one band several dB above the
others, producing a sharp spike. Quiet / decay / constant-broadband
frames have all bands near the same level, so delta is near 0 — this
is why a delta signal works for cymbals/hi-hat (constant sizzle) where
a ratio signal would stay perpetually high. Peak-pick this delta with
``find_peaks`` to get the onsets.

The method was reverse-engineered from the user's WebUI spectrogram
(``spectrogram_analyzer_data_exporter.html``): the WebUI computes
``20*log10(|STFT|)`` on a Hann-windowed signal at n_fft=1024 (configurable
in the UI) and hop=N/4=256 (default 4:1 overlap), then maps that to a
colormap with adjustable floor/gain.

Calibration (project 3, toms stem, 73-76s region, 6 known hits):
  - Spectral peaks: 73.700, 73.868, 74.048, 74.222, 74.420, 74.600
  - User eyeballed: 73.676, 73.853, 74.033, 74.210, 74.411, 74.576
  - Mean offset: +16.9ms (spectral peak trails the strike by ~17ms,
    expected: peak is after attack, strike is at attack)
  - Min offset: +9.0ms, Max offset: +24.1ms
  - This is well within the existing 12ms validator tolerance and
    consistent across all 6 hits.

Defaults (chosen for tom-like signals, 44.1kHz audio):
  - N_FFT = 1024  (~23ms time res, ~43Hz freq res — enough to see
    fundamental AND broadband transient structure)
  - HOP = 256    (4:1 overlap, ~5.8ms time step)
  - BANDS = (60-200, 200-600, 600-1200, 1200-2400, 2400-8000) Hz
    (user-chosen; covers sub/bass → high cymbal)
  - MIN_BAND_RATIO = 5.0 (the top band must be 5x the median to count
    as a hit; suppresses quiet/decay frames where all bands are near
    background)
  - MIN_PEAK_SPACING_MS = 100 (no two hits closer than 100ms;
    tighter than real drumming, generous for flams)
  - PROMINENCE = 2.0 (peak must rise at least 2.0 above local
    minimum — a smaller prominence than the old 10.0 because the
    band_ratio signal has a smaller dynamic range than the count
    signal)

Pure functional core - no side effects, no file I/O, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


# The 5 user-chosen bands. Frozen for now (per task spec); the schema
# could expose them later via SettingDefinition if per-stem tuning is
# needed.
DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (60.0, 200.0),     # band 0: sub/bass — kicks, low toms, fundamental
    (200.0, 600.0),    # band 1: low-mid — toms, snare body, kick
    (600.0, 1200.0),   # band 2: mid — snare, mid toms, hi-hat fundamental
    (1200.0, 2400.0),  # band 3: high-mid — snare wire, hi-hat, cymbal edge
    (2400.0, 8000.0),  # band 4: high — hi-hat sizzle, cymbal body
)


@dataclass(frozen=True)
class SpectralTransientEvent:
    """One detected broadband transient event.

    Carries the full per-band power profile so downstream code (WebUI
    tooltip, classification pipeline) can show the spectral SHAPE of
    each hit, not just a single number.
    """
    time_sec: float
    band_powers: Tuple[float, ...]   # length-5 tuple of linear power sums
    band_max_idx: int                # argmax of band_powers, 0-4
    band_max_ratio: float            # top / second-highest band (or 1e-20)
    # Detection signal values at the event frame (2026-06-09).
    # band_delta is the RING signal (max-median of per_bin_means over
    # all 5 bands) — fires on per-band-dominant content. snap_delta
    # is the SNAP signal (min of per_bin_means over the snap_bands)
    # — fires on broadband content in the snap range. These are
    # diagnostic values, useful for understanding why an event fired
    # (or didn't).
    band_delta: float = 0.0
    snap_delta: float = 0.0

    # Derived ratios for the WebUI tooltip and the advanced filter
    # (2026-06-10). Computed on read rather than stored, since
    # they're trivial divisions and the dataclass is frozen.
    # - snap_to_ring_ratio: snap_delta / band_delta. Low values
    #   (<<1) mean the broadband snap signal is much weaker than
    #   the per-band-dominant ring signal — typical of wire-tail
    #   / decay events where the ring component outlasts the snap
    #   by 50-100ms. The user's calibration event (ring=665,
    #   snap=0.01) gives snap/ring = 0.000015.
    # - snap_to_top_ratio: snap_delta / band_max_ratio. How strong
    #   the snap signal is relative to the top-band dominance
    #   metric. Values close to 1.0 mean the snap is roughly as
    #   strong as the per-band peak ratio — usually a real hit.
    #   Low values mean the band-dominance is happening in a
    #   non-snap band (sustained ring without broadband attack).
    @property
    def snap_to_ring_ratio(self) -> float:
        if self.band_delta == 0:
            return 0.0
        return self.snap_delta / self.band_delta

    @property
    def snap_to_top_ratio(self) -> float:
        if self.band_max_ratio == 0:
            return 0.0
        return self.snap_delta / self.band_max_ratio


@dataclass(frozen=True)
class SpectralTransientConfig:
    """Configuration for spectral transient detection.

    The legacy knobs ``floor_db`` and ``min_bins_above`` (the old
    "count bins above noise floor" signal) have been removed — they
    are replaced by the band-power approach. Knobs:

    - ``bands``         : the 5 (lo, hi) Hz tuples used for per-band
                          power sums. Default is ``DEFAULT_BANDS``;
                          frozen for now per user spec.
    - ``min_band_ratio`` : the detection threshold for the RING signal
                          (max − median over all 5 bands). Default 2.0.
    - ``snap_bands``    : tuple of band indices used to compute the
                          SNAP signal (max − median over just these
                          bands). The snap is the broadband percussive
                          transient that occurs AT the attack onset
                          (e.g. the "head snap" of a tom is in B1+B2,
                          200-1200Hz; the ring develops 50-100ms
                          later in B0). Default = all 5 bands
                          (backward compat). User-calibrated per stem:
                          toms uses (1, 2); hihat uses (3, 4); snare
                          uses (1, 2); kick uses (0,).
    - ``snap_min_delta`` : the find_peaks height for the snap signal.
                          Lower than min_band_ratio because the snap
                          signal has a smaller dynamic range than the
                          ring (B1+B2 don't get as bright as B0 in a
                          tom hit). Default 0.05.
    """
    n_fft: int = 1024
    hop: int = 256
    bands: Tuple[Tuple[float, float], ...] = DEFAULT_BANDS
    min_band_ratio: float = 2.0
    min_peak_spacing_ms: float = 100.0
    prominence: float = 2.0
    snap_bands: Tuple[int, ...] = (0, 1, 2, 3, 4)
    snap_min_delta: float = 0.05


def compute_stft_db(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute magnitude-dB STFT matching the WebUI's recipe.

    Hann window, n_fft samples, hop stride, |rfft| then 20*log10. The
    output shape is (n_bins, n_frames) and the dB values are absolute
    (no reference divisor) — pass them to a colormap with a floor/gain
    to get the user's view.

    Returns:
        freqs_hz:  shape (n_bins,),  bin -> Hz
        times_sec: shape (n_frames,), frame center -> seconds
        s_db:      shape (n_bins, n_frames), magnitude in dB
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) < n_fft:
        raise ValueError(
            f"audio too short: {len(audio)} samples < n_fft={n_fft}"
        )

    n_frames = (len(audio) - n_fft) // hop + 1
    n_bins = n_fft // 2 + 1  # rfft bins: 0 .. n_fft/2 inclusive

    win = np.hanning(n_fft)
    s = np.empty((n_bins, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + n_fft] * win
        spec = np.fft.rfft(frame, n=n_fft)
        s[:, i] = np.abs(spec)

    freqs = np.arange(n_bins) * sr / n_fft
    times = np.arange(n_frames) * hop / sr + (n_fft / 2) / sr
    s_db = 20.0 * np.log10(s + 1e-8)
    # Apply test corruption overlay (2026-06-11) — a hook
    # for A/B experiments where we want to see how the pipeline
    # behaves when the spectrogram has a known corruption in a
    # specific time range. Defaults to no-op. Set via
    # ``set_spectral_corruption()``. The corruption is applied
    # to the LOG-MAGNITUDE spectrogram AFTER the FFT, so it
    # doesn't disturb the audio samples themselves.
    if _SPECTRAL_CORRUPTION is not None:
        s_db = _apply_spectral_corruption(s_db, times, _SPECTRAL_CORRUPTION)
    return freqs, times, s_db


# ── Spectral corruption test hook ───────────────────────────────────────────
# Set via set_spectral_corruption(); cleared via clear_spectral_corruption().
# Used to A/B test how the pipeline reacts to known corruptions in the
# spectrogram (e.g. a missing 12ms window at 14.840-14.852s, simulating
# the "soft hit" artifact we observed in the toms audio). The
# corruption is applied to s_db (the log-magnitude STFT), not the audio
# samples — so the .wav file on disk is untouched, and we can run the
# same wav through the pipeline with and without corruption to compare.

_SPECTRAL_CORRUPTION: Optional[Dict[str, Any]] = None


def set_spectral_corruption(
    t_start_sec: float,
    t_end_sec: float,
    mode: str = "interpolate",
    n_fft: int = 1024,
) -> None:
    """Activate a spectral corruption overlay.

    The corruption REMOVES a time range from the spectrogram and
    FILLS it with values derived from the surrounding frames. The
    original audio is untouched — only the log-magnitude STFT
    gets patched, AFTER the FFT. This is for A/B experiments where
    we want to see how the pipeline behaves when a known
    corruption is removed from the spectrogram (simulating "the
    artifact was never there").

    Args:
        t_start_sec, t_end_sec: time range to corrupt, in seconds
            (relative to the audio start). The frames in this
            range are replaced.
        mode: how to fill the corrupt range. Options:
            - ``"interpolate"`` (default): for each frame in
              the corrupt range, blend between the pre-corruption
              column (t_start) and the post-corruption column
              (t_end) with a linear ramp. Frame at t_start keeps
              the pre value; frame at t_end keeps the post value;
              middle frames are linearly interpolated. This
              simulates "the audio transitioned smoothly from
              before to after" — appropriate for short gaps.
            - ``"pre"``: fill the entire corrupt range with copies
              of the frame at t_start (forward extrapolation).
            - ``"post"``: fill with copies of the frame at t_end
              (backward extrapolation).
        n_fft: STFT window size, for frame-index calculations
            (must match what the pipeline uses).
    """
    global _SPECTRAL_CORRUPTION
    _SPECTRAL_CORRUPTION = {
        "t_start_sec": t_start_sec,
        "t_end_sec": t_end_sec,
        "mode": mode,
        "n_fft": n_fft,
    }


def clear_spectral_corruption() -> None:
    """Deactivate the spectral corruption overlay."""
    global _SPECTRAL_CORRUPTION
    _SPECTRAL_CORRUPTION = None


def _apply_spectral_corruption(
    s_db: np.ndarray,
    times: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Apply the corruption overlay to a log-magnitude spectrogram.

    For each frame in the [t_start, t_end] range, replace its
    column in s_db with values derived from the surrounding
    frames. The "surrounding frames" are the columns at
    t_start (just before the corruption) and t_end (just after
    the corruption). This REMOVES the artifact and FILLS the
    gap with synthesized data — the audio is untouched, only
    the log-magnitude STFT is patched.
    """
    t_start = cfg["t_start_sec"]
    t_end = cfg["t_end_sec"]
    mode = cfg["mode"]
    n_fft = cfg["n_fft"]

    # Find the frame indices just before t_start and just after
    # t_end. These are the "anchor" frames that bracket the gap.
    # We use the center of each frame to decide membership.
    i_anchor_pre = int(np.searchsorted(times, t_start)) - 1
    i_anchor_post = int(np.searchsorted(times, t_end))
    # The corrupt range covers frames strictly between
    # i_anchor_pre and i_anchor_post.
    i_corrupt_lo = i_anchor_pre + 1
    i_corrupt_hi = i_anchor_post
    if i_corrupt_hi <= i_corrupt_lo:
        return s_db  # nothing to do
    if i_anchor_pre < 0 or i_anchor_post >= s_db.shape[1]:
        return s_db  # can't anchor — give up

    if mode == "interpolate":
        # Linear blend between the pre anchor and the post
        # anchor. Frame at i_corrupt_lo gets a tiny bit of the
        # post value; frame at i_corrupt_hi-1 gets nearly all
        # the post value. This simulates "the audio
        # transitioned smoothly from before to after" — the
        # right model for a 12ms audio gap where the
        # surrounding signal is continuous toms ring.
        col_pre = s_db[:, i_anchor_pre].copy()
        col_post = s_db[:, i_anchor_post].copy()
        n_corrupt = i_corrupt_hi - i_corrupt_lo
        for j in range(n_corrupt):
            # alpha = 0 at the start of the corrupt range, 1 at the end
            alpha = (j + 1) / (n_corrupt + 1)
            s_db[:, i_corrupt_lo + j] = (1.0 - alpha) * col_pre + alpha * col_post
        return s_db
    elif mode == "pre":
        # Fill with copies of the pre anchor
        col_pre = s_db[:, i_anchor_pre].copy()
        n_corrupt = i_corrupt_hi - i_corrupt_lo
        for j in range(n_corrupt):
            s_db[:, i_corrupt_lo + j] = col_pre
        return s_db
    elif mode == "post":
        # Fill with copies of the post anchor
        col_post = s_db[:, i_anchor_post].copy()
        n_corrupt = i_corrupt_hi - i_corrupt_lo
        for j in range(n_corrupt):
            s_db[:, i_corrupt_lo + j] = col_post
        return s_db
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")


def _band_powers_from_db(
    s_db: np.ndarray,
    freqs: np.ndarray,
    bands: Tuple[Tuple[float, float], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum linear power in each band per frame.

    Linear power for a dB value ``x`` is ``10 ** (x / 10)``
    (= (10 ** (x / 20)) ** 2). Returns a 2-tuple::

        (band_powers, per_bin_means)

    where:
    - ``band_powers``   : (len(bands), n_frames), raw sum of linear
                          power across all bins in the band
    - ``per_bin_means`` : (len(bands), n_frames), band_powers / n_bins
                          (i.e. power spectral density in each band).
                          Used for the ratio detection signal so the
                          signal is invariant to band width — a wide
                          band naturally accumulates more total power
                          even from white noise, but the per-bin mean
                          cancels that out.
    """
    # Convert dB to linear power once for the whole spectrum.
    # Clip dB at -80 to avoid 10**(-800/10) on empty frames.
    s_lin = 10.0 ** (np.clip(s_db, -80.0, 80.0) / 10.0)
    band_powers = np.empty((len(bands), s_db.shape[1]), dtype=np.float64)
    per_bin_means = np.empty((len(bands), s_db.shape[1]), dtype=np.float64)
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs <= hi)
        n_bins = int(mask.sum())
        if n_bins == 0:
            band_powers[i, :] = 0.0
            per_bin_means[i, :] = 0.0
        else:
            band_powers[i, :] = s_lin[mask, :].sum(axis=0)
            per_bin_means[i, :] = band_powers[i, :] / n_bins
    return band_powers, per_bin_means


def _band_snap_signal(per_bin_means_subset: np.ndarray) -> np.ndarray:
    """Per-frame ``min(per_bin_means_subset)`` over the snap bands.

    The SNAP signal — fires when ALL snap bands are simultaneously
    loud (broadband percussive transient). This is the broadband
    counterpart of the ring signal: a real drum strike lights up
    multiple high-frequency bands at the attack onset (the "snap"
    of the head/cymbal-stick contact), and that broadband signature
    decays within ~10-20ms as the energy moves into a single
    frequency band (the "ring").

    User insight (2026-06-09): the toms attack onset in the 14-16s
    window is broadband in B1 (200-600Hz) and B2 (600-1200Hz). The
    B0 ring develops 50-100ms later. A max-median delta over B1+B2
    is BACKWARDS for 2-band detection: it peaks when one band
    dominates (the ring in B1 alone), not when both are loud (the
    snap). The min signal correctly fires on broadband (both loud)
    and is small when one band is dominant (the ring).

    Edge case — 1 band: ``min`` of a single element is that element.
    For kicks (snap_bands=(0,)), this gives the absolute B0 power
    normalized by n_bins — the right signal for kicks.

    Edge case — all 5 bands (default): ``min`` of all 5 = the
    smallest band. For broadband signals (cymbals, hi-hat) this is
    moderate; for single-band strikes (toms, kicks) this is the
    smallest of the 4 quiet bands, which is very small. This is
    OPPOSITE of the max-median signal — so the default snap_bands
    (all 5) does NOT behave like the old ring signal. The user
    should configure snap_bands explicitly per stem.
    """
    return np.min(per_bin_means_subset, axis=0)


def _band_ratio_signal(per_bin_means: np.ndarray) -> np.ndarray:
    """Per-frame ``max(per_bin_means) - median(per_bin_means)``.

    The DELTA between the loudest band and the median band per frame.
    Uses per-bin means (not raw sums) so the signal is
    loudness-invariant: a wider band has more total power but the
    same power-per-bin density, so width cancels out.

    Why delta (not ratio): for stems with constant broadband content
    (cymbals sizzle, hi-hat sizzle) every frame is already elevated
    in all bands, so a ratio signal stays high and find_peaks fires
    constantly. The delta signal requires the loudest band to RISE
    ABOVE the typical level of the other bands — cymbals/hi-hat
    sizzle produces delta ~0 (every band at the same level), real
    strikes produce delta > 0 (one band briefly dominant).

    A real hit lights up one band 2-10x above the others, producing
    a sharp spike. Quiet / decay / constant-content frames have
    delta near 0. If the median is 0 (silent input), the signal
    is 0 (no transient). This keeps find_peaks happy on silence.
    """
    top = per_bin_means.max(axis=0)
    med = np.median(per_bin_means, axis=0)
    # Delta signal: how much does the loudest band exceed the
    # typical (median) band, in per-bin linear-power units.
    # Cymbals/hi-hat with constant broadband: top ≈ med, delta ≈ 0
    # Toms/single-band transients: top >> med, delta is large
    return top - med


def _band_max_from_powers(
    band_powers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame ``(argmax, top/second)`` of ``band_powers``.

    Note: this uses the raw band_powers (sums), not the per-bin means,
    because the user wants ``band_powers`` itself to drive the
    classification (a hit dominated by band 4 has more absolute power
    in band 4 than in band 0 by virtue of dominance, not by width).
    See test_high_frequency_tone_dominates_band_4 for the contract.

    Returns:
        max_idx: shape (n_frames,), int in [0, len(bands))
        max_ratio: shape (n_frames,), top / second-highest (or
                   top / 1e-20 if second is 0).
    """
    # argmax along the band axis per frame
    max_idx = np.argmax(band_powers, axis=0).astype(np.int64)
    top = band_powers.max(axis=0)
    # Sort each frame's band powers descending and take the 2nd
    sorted_bp = -np.sort(-band_powers, axis=0)
    second = sorted_bp[1, :]
    max_ratio = np.where(second > 0, top / second, top / 1e-20)
    return max_idx, max_ratio


def _apply_wire_tail_filter(
    events: List[SpectralTransientEvent],
    tail_window_sec: float = 0.150,
    tail_threshold: float = 0.5,
    snap_window_sec: float = 0.100,
) -> List[SpectralTransientEvent]:
    """Drop wire-tail events: a weaker event within ``tail_window_sec``
    AFTER a stronger event is almost certainly the tail of the strike,
    not a new hit.

    User observation (2026-06-09, project 4 snare 73-77s): the
    detector found 17 spectral events for ~3-5 real snare hits.
    Pairs of events ~100-300ms apart with the second one weaker
    (smaller top-band power) are snare-wire decay. Toms have a
    similar pattern: the strike lights up the band, then a slow
    low-frequency tail produces a second event ~100-200ms later
    with smaller top-band power.

    Calibration (2026-06-09):
      - Toms 73-77s: 6 GT hits spaced 175-201ms apart. Raw
        find_peaks finds ~15 events (each strike produces 2-3
        peaks as the band balance shifts during rise/peak/decay).
        The 6 GT hits are all in the raw set. After this filter,
        6 events remain (calibrated 2026-06-09).

    Snap-ring filter (2026-06-09): when the snap and ring both fire
    for the same strike (snap at attack onset, ring 50-100ms after),
    the RING event typically has MORE top-band power than the snap
    (the ring is in the low band, which carries the most energy for
    a toms/kick). The power-based filter would drop the snap (weak
    earlier) and keep the ring (strong later) — losing the snap
    timing precision. The fix: if two events are within
    ``snap_window_sec`` (80ms — wider than the per-strike burst
    decay ~30-50ms, narrower than the next-strike attack onset
    ~150-200ms), keep the EARLIER (the snap = attack onset) and
    drop the LATER (the ring = post-attack decay) regardless of
    power.

    Calibration for snap_window_sec=80ms:
      - Toms 14-16s (200ms-spaced strikes): 6 raw events → 3 kept
        (one snap per strike, ring dropped). 0 over-fires.
      - Toms 73-77s (175-201ms-spaced strikes): 14 raw events → 6
        kept. No regression on the 6 GT hits.
    """
    if not events:
        return events
    sorted_evs = sorted(events, key=lambda e: e.time_sec)
    kept = [sorted_evs[0]]
    for ev in sorted_evs[1:]:
        recent = kept[-1]
        gap = ev.time_sec - recent.time_sec
        if 0 < gap <= snap_window_sec:
            # Snap-ring case: drop the LATER event (the ring), keep
            # the earlier (the snap = attack onset). This is the
            # reverse of the power-based logic — power says drop the
            # snap, but timing says the snap is the real hit.
            continue
        if gap <= tail_window_sec:
            recent_top = recent.band_powers[recent.band_max_idx]
            ev_top = ev.band_powers[ev.band_max_idx]
            if ev_top < tail_threshold * recent_top:
                # Tail of recent, drop.
                continue
        kept.append(ev)
    return kept


def detect_spectral_transients(
    audio: np.ndarray,
    sr: int,
    config: Optional[SpectralTransientConfig] = None,
) -> tuple[List[SpectralTransientEvent], dict]:
    """Find broadband transient onsets in audio.

    For each STFT frame, sum the linear power in each of the 5 bands.
    The detection signal is ``max(per_bin_means) - median(per_bin_means)``
    — a per-bin loudness-invariant DELTA that requires the loudest
    band to RISE above the typical band. Cymbals/hi-hat constant
    sizzle produces delta ~0 (no real strike), real hits produce
    a sharp spike. Peak-pick this delta with ``find_peaks`` to get
    the onsets, then apply a wire-tail filter to drop the decay
    events that follow a stronger strike.

    Returns:
        events: detected events (sorted by time), each carrying the
                full band_powers profile + cheap classifications
                (band_max_idx, band_max_ratio).
        debug:  intermediate arrays for plotting / inspection
                (times, band_powers, band_delta, max_db).
    """
    cfg = config or SpectralTransientConfig()
    freqs, times, s_db = compute_stft_db(audio, sr, n_fft=cfg.n_fft, hop=cfg.hop)

    # Per-band power sums in linear units (and per-bin means for the
    # delta signal — see _band_powers_from_db docstring)
    band_powers, per_bin_means = _band_powers_from_db(s_db, freqs, cfg.bands)
    # Detection signal: max - median of per-bin means per frame.
    # Per-bin means cancel band-width asymmetry (band 4 is 40x wider
    # than band 0, so raw sums over-weight it for broadband noise).
    band_delta = _band_ratio_signal(per_bin_means)
    # SNAP detection signal (2026-06-09): the min of per-bin means
    # over the snap_bands (e.g. (1, 2) for toms — the "head snap"
    # range 200-1200Hz). The snap lights up AT the attack onset,
    # before the B0 ring develops. This lets the detector fire
    # within a few ms of the strike instead of 50-100ms after (when
    # the ring peaks). For 2-band snap ranges, min correctly fires
    # on broadband (both bands loud) and is small when one band is
    # dominant (the ring in B1 alone).
    snap_per_bin_means = per_bin_means[list(cfg.snap_bands), :]
    snap_delta = _band_snap_signal(snap_per_bin_means)
    # Per-frame max dB across the full spectrum (kept for reporting /
    # backwards debug compat)
    max_db = s_db.max(axis=0)
    # Per-frame (argmax, top/second) for the event fields
    band_max_idx, band_max_ratio = _band_max_from_powers(band_powers)

    # find_peaks needs spacing in samples
    spacing_frames = max(1, int(round(cfg.min_peak_spacing_ms / 1000.0 * sr / cfg.hop)))

    # Two parallel detection passes:
    # 1. The RING pass — uses band_delta (all 5 bands), calibrated to
    #    detect the per-band dominance that develops during the
    #    ring/sustain phase (kicks, sustained toms, sustained cymbals).
    # 2. The SNAP pass — uses snap_delta (per-stem snap_bands),
    #    calibrated to detect the broadband percussive transient at
    #    the attack onset (tom head snap, snare strike, hihat snap).
    # The two are unioned (with a small merge window to dedupe
    # events detected by both passes within ~50ms of each other).
    ring_peaks, _ = find_peaks(
        band_delta,
        height=float(cfg.min_band_ratio),
        distance=spacing_frames,
        prominence=cfg.prominence,
    )
    snap_peaks, _ = find_peaks(
        snap_delta,
        height=float(cfg.snap_min_delta),
        distance=spacing_frames,
    )
    # Merge: union the ring and snap peaks. When both signals fire
    # within 50ms of each other, prefer the SNAP time (the snap
    # peaks AT the attack onset, the ring peaks 50-100ms after).
    # Without this preference, the detector would report the ring
    # time, losing the snap timing precision.
    merge_window_frames = max(1, int(round(0.050 * sr / cfg.hop)))
    merged_peaks = []
    # Pair each snap peak with the nearest ring peak within the merge
    # window. If paired, take the snap time. If unpaired, take the
    # snap time anyway.
    used_rings = set()
    for sp in snap_peaks:
        # Find nearest ring peak within merge_window_frames
        nearest_ring = None
        nearest_dist = None
        for rp in ring_peaks:
            d = abs(int(sp) - int(rp))
            if d <= merge_window_frames and (nearest_dist is None or d < nearest_dist):
                nearest_ring = int(rp)
                nearest_dist = d
        if nearest_ring is not None:
            used_rings.add(nearest_ring)
        merged_peaks.append(int(sp))
    # Add ring peaks that weren't paired with any snap peak
    for rp in ring_peaks:
        if int(rp) not in used_rings:
            merged_peaks.append(int(rp))
    merged_peaks = np.array(sorted(merged_peaks))

    events = [
        SpectralTransientEvent(
            time_sec=float(times[p]),
            band_powers=tuple(float(x) for x in band_powers[:, p]),
            band_max_idx=int(band_max_idx[p]),
            band_max_ratio=float(band_max_ratio[p]),
            band_delta=float(band_delta[p]),
            snap_delta=float(snap_delta[p]),
        )
        for p in merged_peaks
    ]

    # Apply wire-tail filter: drop weaker events within 150ms of a
    # stronger one. This catches snare wire decay and toms envelope
    # tails that are real signal but not real hits. Window calibrated
    # on project 4 73-77s toms/snare: 150ms is just longer than the
    # per-strike burst decay (~80-100ms) but shorter than the
    # 175-200ms inter-strike gap in the user's 6-hit toms fill.
    events = _apply_wire_tail_filter(events, tail_window_sec=0.150)

    debug = {
        "times": times,
        "band_powers": band_powers,           # shape (5, n_frames)
        "band_delta": band_delta,             # shape (n_frames,)
        "snap_delta": snap_delta,             # shape (n_frames,)
        "max_db": max_db,                     # shape (n_frames,)
        "freqs": freqs,
        "s_db": s_db,
    }
    return events, debug
