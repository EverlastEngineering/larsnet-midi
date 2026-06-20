"""
STFT and timing utilities shared by event_features.py and
percentile_gated_detector.py. Extracted from the former
spectral_transient_core.py after that module's detector was
removed in the PGA-universal cleanup (2026-06-20).

Provides:
  - compute_stft_db: magnitude-dB STFT (Hann, |rfft|, 20*log10)
  - timed / get_function_timings / reset_function_timings:
    wall-time profiler used by the PGA pipeline
  - STFT cache + corruption overlay (test-only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import os
import time

import numpy as np
from scipy.signal import find_peaks

# Set LARSNET_TIMING=1 in environment to opt in to [t+Xs] timing logs.
_TIMING_ENABLED = os.environ.get("LARSNET_TIMING", "0") == "1"

# Module-level cache for compute_stft_db: keyed on (id(audio), sr, n_fft, hop).
# Same audio object + same STFT params → reuse the full-file STFT result.
# This turns N events × full-file STFT into 1 STFT total for the
# _envelope_at_time callers in event_features.py.
_STFT_CACHE: Dict[tuple, tuple] = {}

# Cumulative stats for the STFT cache. Used by get_stft_cache_stats() /
# reset_stft_cache_stats(). Populated by compute_stft_db so a CLI run can
# see how many calls hit/missed, how much time was spent in the FFT inner
# loop vs surrounding work (log10, band prep, corruption overlay), and the
# total wall time.
_STFT_CACHE_STATS: Dict[str, float] = {
    "hits": 0,
    "misses": 0,
    "fft_loop_sec": 0.0,
    "total_sec": 0.0,
}

# Wall-clock start of this Python process (set at import time). Each
# STFT log line prepends "[t+XX.Xs]" = seconds since this stamp, so the
# log can be correlated with overall CLI runtime — i.e. you can see
# whether the STFT calls are clustered early/late in the run and what
# fraction of total runtime they consume.
_PROGRAM_START = time.perf_counter()


def _elapsed_since_start() -> float:
    """Seconds since module import (used to prefix STFT log lines)."""
    return time.perf_counter() - _PROGRAM_START


# Per-function timing stats keyed by name. Populated by timed(); read by
# get_function_timings(). Use to see WHERE the time goes in the call
# stack above compute_stft_db — which functions dominate the CLI runtime
# and how many times each was called.
_FN_STATS: Dict[str, Dict[str, float]] = {}


class _TimedBlock:
    """Context manager that logs elapsed wall time on exit with [t+Xs] prefix.

    Tracks per-name call count, total wall time, and max wall time.
    Usage::

        with timed("compute_t60_ms"):
            ... function body ...

    Early returns and exceptions still trigger __exit__, so the log
    reflects true function duration even on the bail-out paths.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.t0 = 0.0

    def __enter__(self) -> "_TimedBlock":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = time.perf_counter() - self.t0
        stats = _FN_STATS.setdefault(
            self.name, {"calls": 0, "total_sec": 0.0, "max_sec": 0.0}
        )
        stats["calls"] += 1
        stats["total_sec"] += dt
        if dt > stats["max_sec"]:
            stats["max_sec"] = dt
        if _TIMING_ENABLED:
            print(
                f"[t+{_elapsed_since_start():.2f}s] {self.name} "
                f"({dt*1000:.2f}ms) — calls={stats['calls']} "
                f"cum={stats['total_sec']*1000:.1f}ms "
                f"max={stats['max_sec']*1000:.2f}ms"
            )


def timed(name: str) -> _TimedBlock:
    """Open a named timing block. See _TimedBlock for the log format."""
    return _TimedBlock(name)


def get_function_timings() -> Dict[str, Dict[str, float]]:
    """Return per-function timing stats accumulated by timed().

    Returns a copy so callers can mutate without affecting the
    underlying accumulator.
    """
    return {k: dict(v) for k, v in _FN_STATS.items()}


def reset_function_timings() -> None:
    """Reset all per-function timing stats to zero."""
    _FN_STATS.clear()


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
    t_entry = time.perf_counter()
    cache_key = (id(audio), sr, n_fft, hop, id(_SPECTRAL_CORRUPTION))
    if cache_key in _STFT_CACHE:
        dt = time.perf_counter() - t_entry
        _STFT_CACHE_STATS["hits"] += 1
        _STFT_CACHE_STATS["total_sec"] += dt
        if _TIMING_ENABLED:
            print(
                f"[t+{_elapsed_since_start():.2f}s] STFT cache hit "
                f"({dt*1000:.2f}ms) — cum_total="
                f"{_STFT_CACHE_STATS['total_sec']*1000:.1f}ms "
                f"hits={_STFT_CACHE_STATS['hits']}, "
                f"misses={_STFT_CACHE_STATS['misses']}"
            )
        return _STFT_CACHE[cache_key]

    if _TIMING_ENABLED:
        print(
            f"[t+{_elapsed_since_start():.2f}s] STFT cache miss — computing STFT "
            f"(audio={len(audio)} samples, sr={sr}, n_fft={n_fft}, hop={hop})"
        )
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
    t_fft_start = time.perf_counter()
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + n_fft] * win
        spec = np.fft.rfft(frame, n=n_fft)
        s[:, i] = np.abs(spec)
    t_fft = time.perf_counter() - t_fft_start

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

    cache_key = (id(audio), sr, n_fft, hop, id(_SPECTRAL_CORRUPTION))
    _STFT_CACHE[cache_key] = (freqs, times, s_db)

    t_total = time.perf_counter() - t_entry
    _STFT_CACHE_STATS["misses"] += 1
    _STFT_CACHE_STATS["fft_loop_sec"] += t_fft
    _STFT_CACHE_STATS["total_sec"] += t_total
    if _TIMING_ENABLED:
        print(
            f"[t+{_elapsed_since_start():.2f}s] STFT done: "
            f"total={t_total*1000:.2f}ms fft_loop={t_fft*1000:.2f}ms "
            f"frames={n_frames} — cum_total={_STFT_CACHE_STATS['total_sec']*1000:.1f}ms "
            f"cum_fft={_STFT_CACHE_STATS['fft_loop_sec']*1000:.1f}ms "
            f"hits={_STFT_CACHE_STATS['hits']} misses={_STFT_CACHE_STATS['misses']}"
        )
    return freqs, times, s_db


def get_stft_cache_stats() -> Dict[str, float]:
    """Return cumulative STFT cache statistics.

    Keys:
      - hits (int): cache hits
      - misses (int): cache misses (full STFT computed)
      - fft_loop_sec (float): total wall time in the FFT inner loop
      - total_sec (float): total wall time in compute_stft_db (includes
        audio prep, FFT, log10, corruption overlay)

    Diff total_sec − fft_loop_sec = time spent on non-FFT work
    (audio conversion, log10, corruption overlay, dict ops, prints).
    """
    return dict(_STFT_CACHE_STATS)


def reset_stft_cache_stats() -> None:
    """Reset STFT cache statistics to zero. Useful for tests
    or for isolating measurements between CLI subcommands.
    """
    _STFT_CACHE_STATS["hits"] = 0
    _STFT_CACHE_STATS["misses"] = 0
    _STFT_CACHE_STATS["fft_loop_sec"] = 0.0
    _STFT_CACHE_STATS["total_sec"] = 0.0


def reset_stft_cache() -> None:
    """Clear the STFT result cache. Useful for tests where a
    prior test's audio array is garbage-collected and the same
    ``id(audio)`` is reissued to a new, unrelated array — without
    clearing, the new array would get the old STFT and tests of
    the functional core (e.g. ``compute_spectral_centroid_hz``)
    would see phantom data from a previous test.
    """
    _STFT_CACHE.clear()


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


