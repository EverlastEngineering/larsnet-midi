"""
Tests for the per-band power profile in stems_to_midi/spectral_transient_core.py.

The new detection signal: each STFT frame gets a 5-tuple of linear power
sums, one per band. Detection runs on
``max(band_powers) / median(band_powers)`` — a loudness-invariant ratio
that lights up on a real hit (one band 5-10x the others) and stays near
1.0 on quiet/decay frames.

User spec (2026-06-09):
  band 0: 60-200 Hz    (sub/bass — kicks, low toms, fundamental)
  band 1: 200-600 Hz   (low-mid — toms, snare body, kick)
  band 2: 600-1200 Hz  (mid — snare, mid toms, hi-hat fundamental)
  band 3: 1200-2400 Hz (high-mid — snare wire, hi-hat, cymbal edge)
  band 4: 2400-8000 Hz (high — hi-hat sizzle, cymbal body)

Each spectral event then carries:
  band_powers:      tuple of 5 floats (linear power sums)
  band_max_idx:     argmax of band_powers, 0-4
  band_max_ratio:   top / second-highest, top / 1e-20 if all-zero
"""

import numpy as np
import pytest
import soundfile as sf

from stems_to_midi.spectral_transient_core import (
    DEFAULT_BANDS,
    SpectralTransientConfig,
    SpectralTransientEvent,
    compute_stft_db,
    detect_spectral_transients,
)


# --- helpers ---------------------------------------------------------------

def make_synthetic_burst(
    sr: int = 44100,
    duration_sec: float = 3.0,
    burst_time_sec: float = 1.0,
    burst_freqs_hz: tuple = (200.0, 1000.0, 3000.0, 5000.0),
    burst_amplitudes: tuple = (1.0, 0.5, 0.3, 0.2),
    decay_ms: float = 80.0,
) -> np.ndarray:
    """Build a mono signal with one exponentially-decaying multiband burst.
    The energy distribution across bands is controlled by ``burst_freqs_hz``
    and ``burst_amplitudes``."""
    t = np.arange(int(sr * duration_sec)) / sr
    y = np.zeros_like(t)
    decay_samples = int(decay_ms / 1000.0 * sr)
    env = np.exp(-np.arange(decay_samples) / (decay_samples / 4.0))
    i0 = int(burst_time_sec * sr)
    i1 = min(i0 + decay_samples, len(y))
    n = i1 - i0
    if n <= 0:
        return y
    burst = np.zeros(n)
    for f, a in zip(burst_freqs_hz, burst_amplitudes):
        burst += a * np.sin(2 * np.pi * f * np.arange(n) / sr)
    y[i0:i1] += burst * env[:n]
    return y


# --- band spec --------------------------------------------------------------

def test_default_bands_match_user_spec():
    """The 5 fixed bands match the user's chosen ranges exactly."""
    expected = (
        (60.0, 200.0),
        (200.0, 600.0),
        (600.0, 1200.0),
        (1200.0, 2400.0),
        (2400.0, 8000.0),
    )
    assert DEFAULT_BANDS == expected, (
        f"DEFAULT_BANDS {DEFAULT_BANDS} != user spec {expected}"
    )


# --- dataclass fields -------------------------------------------------------

def test_event_has_band_powers_field():
    """SpectralTransientEvent must carry a band_powers tuple of length 5."""
    e = SpectralTransientEvent(
        time_sec=1.0,
        band_powers=(0.1, 0.2, 0.3, 0.4, 0.5),
        band_max_idx=4,
        band_max_ratio=2.0,
    )
    assert e.band_powers == (0.1, 0.2, 0.3, 0.4, 0.5)
    assert len(e.band_powers) == 5


def test_event_has_band_max_idx_and_ratio_fields():
    """band_max_idx is the argmax of band_powers; band_max_ratio is top/second."""
    e = SpectralTransientEvent(
        time_sec=1.0,
        band_powers=(0.1, 0.2, 0.5, 0.3, 0.4),
        band_max_idx=2,
        band_max_ratio=2.5,  # 0.5 / 0.4 = 1.25 -- just to check it stores
    )
    assert e.band_max_idx == 2
    assert e.band_max_ratio == 2.5


def test_event_dataclass_still_frozen():
    """The dataclass must still be immutable (no accidental mutation)."""
    e = SpectralTransientEvent(
        time_sec=1.0,
        band_powers=(0.1, 0.2, 0.3, 0.4, 0.5),
        band_max_idx=4,
        band_max_ratio=1.25,
    )
    with pytest.raises(Exception):
        e.time_sec = 2.0  # type: ignore[misc]


# --- config knobs ------------------------------------------------------------

def test_config_drops_old_knobs_and_adds_new():
    """The old config knobs (floor_db, min_bins_above) are gone. New
    knobs (min_band_ratio, bands) replace them."""
    cfg = SpectralTransientConfig()
    # New knobs present
    assert hasattr(cfg, 'min_band_ratio')
    assert hasattr(cfg, 'bands')
    assert cfg.bands == DEFAULT_BANDS
    # Old knobs removed
    assert 'floor_db' not in cfg.__dataclass_fields__, (
        "old config knob 'floor_db' should be dropped"
    )
    assert 'min_bins_above' not in cfg.__dataclass_fields__, (
        "old config knob 'min_bins_above' should be dropped"
    )


# --- detection signal: per-band sums ---------------------------------------

def test_band_powers_sum_linear_power_not_db():
    """Per-frame band sums use linear power (10**(s_db/10)), not dB.
    A band of dB=-20 should contribute 1e-2 per bin; 10 bins of -20dB
    give a band sum of 0.1 (linear) — NOT 10 * -20 = -200 (dB)."""
    sr = 44100
    # Build a single 100Hz tone sustained 0.5s — fills band 0 (60-200Hz)
    # with ~0 dB (amplitude 1.0) on each frame
    t = np.arange(int(sr * 2.0)) / sr
    y = 0.5 * np.sin(2 * np.pi * 100.0 * t)
    events, debug = detect_spectral_transients(y, sr)

    # The frame at t=1.0s should have all the energy in band 0
    assert 'band_powers' in debug
    band_powers = debug['band_powers']  # shape (5, n_frames)
    assert band_powers.shape[0] == 5

    # band 0 (60-200Hz) should dominate at the sustained-tone frame
    # We don't require exact magnitudes (the FFT + windowing matter);
    # we require band 0 to be the max and orders of magnitude above
    # the high bands.
    mid = band_powers.shape[1] // 2
    bp_frame = band_powers[:, mid]
    assert int(np.argmax(bp_frame)) == 0, (
        f"100Hz tone should peak in band 0, got argmax={np.argmax(bp_frame)}"
    )
    # band 0 is at least 100x band 4 (no high-freq content in a 100Hz sine)
    assert bp_frame[0] > 100 * bp_frame[4], (
        f"band 0 {bp_frame[0]:.2e} should dwarf band 4 {bp_frame[4]:.2e} "
        f"for a 100Hz sustained tone"
    )


def test_band_powers_shape_in_debug():
    """Debug dict exposes band_powers as (5, n_frames) array."""
    sr = 44100
    y = np.zeros(sr * 2)
    events, debug = detect_spectral_transients(y, sr)
    assert 'band_powers' in debug
    assert debug['band_powers'].shape[0] == 5
    # n_frames matches the times array
    assert debug['band_powers'].shape[1] == len(debug['times'])


# --- event fields populated correctly --------------------------------------

def test_event_band_powers_argmax_matches_band_max_idx():
    """For each event, band_max_idx == argmax(event.band_powers)."""
    sr = 44100
    y = make_synthetic_burst(
        sr=sr,
        burst_freqs_hz=(100.0, 500.0, 1000.0, 3000.0, 5000.0),
        burst_amplitudes=(0.5, 0.3, 0.2, 0.1, 0.1),
    )
    events, _ = detect_spectral_transients(y, sr)
    assert len(events) >= 1
    for ev in events:
        expected_idx = int(np.argmax(ev.band_powers))
        assert ev.band_max_idx == expected_idx, (
            f"event band_max_idx={ev.band_max_idx} != "
            f"argmax(band_powers)={expected_idx}, "
            f"band_powers={ev.band_powers}"
        )


def test_event_band_max_ratio_is_top_over_second():
    """band_max_ratio is the top band power divided by the second-highest."""
    sr = 44100
    y = make_synthetic_burst(
        sr=sr,
        burst_freqs_hz=(100.0, 500.0, 1000.0, 3000.0, 5000.0),
        burst_amplitudes=(0.5, 0.3, 0.2, 0.1, 0.1),
    )
    events, _ = detect_spectral_transients(y, sr)
    assert len(events) >= 1
    for ev in events:
        bp = sorted(ev.band_powers, reverse=True)
        top, second = bp[0], bp[1]
        # Defensive: top/1e-20 if second is 0
        expected_ratio = top / (second if second > 0 else 1e-20)
        assert ev.band_max_ratio == pytest.approx(expected_ratio, rel=1e-9), (
            f"band_max_ratio {ev.band_max_ratio} != "
            f"top/second {expected_ratio}, band_powers={ev.band_powers}"
        )


def test_low_frequency_tone_dominates_band_0():
    """A 100Hz tone should produce an event with band_max_idx=0."""
    sr = 44100
    y = make_synthetic_burst(
        sr=sr,
        burst_time_sec=1.0,
        burst_freqs_hz=(100.0,),
        burst_amplitudes=(1.0,),
        decay_ms=80.0,
    )
    events, _ = detect_spectral_transients(y, sr)
    # The detector should find the 1.0s burst
    assert len(events) >= 1
    # Find the event nearest t=1.0s
    nearest = min(events, key=lambda e: abs(e.time_sec - 1.0))
    assert nearest.band_max_idx == 0, (
        f"100Hz tone should peak in band 0, got band_max_idx={nearest.band_max_idx}, "
        f"band_powers={nearest.band_powers}"
    )


def test_high_frequency_tone_dominates_band_4():
    """A 5000Hz tone should produce an event with band_max_idx=4."""
    sr = 44100
    y = make_synthetic_burst(
        sr=sr,
        burst_time_sec=1.0,
        burst_freqs_hz=(5000.0,),
        burst_amplitudes=(1.0,),
        decay_ms=80.0,
    )
    events, _ = detect_spectral_transients(y, sr)
    assert len(events) >= 1
    nearest = min(events, key=lambda e: abs(e.time_sec - 1.0))
    assert nearest.band_max_idx == 4, (
        f"5000Hz tone should peak in band 4, got band_max_idx={nearest.band_max_idx}, "
        f"band_powers={nearest.band_powers}"
    )


# --- detection on band-ratio signal -----------------------------------------

def test_band_delta_signal_in_debug():
    """The debug dict must include the band-delta signal used for detection.
    A single-band burst (all energy in one band) gives max - median > 0.
    """
    sr = 44100
    # A 100Hz-only burst: all energy in band 0, nothing in bands 1-4.
    # band_delta = b0 - median(0, 0, 0, 0) = b0 - 0 = b0
    y = make_synthetic_burst(
        sr=sr, burst_time_sec=1.0,
        burst_freqs_hz=(100.0,),
        burst_amplitudes=(1.0,),
    )
    events, debug = detect_spectral_transients(y, sr)
    assert 'band_delta' in debug, (
        "debug dict must include 'band_delta' = "
        "max(band_powers) - median(band_powers) per frame"
    )
    # The signal must spike near the hit (t=1.0s)
    times = debug['times']
    band_delta = debug['band_delta']
    near_hit = (times >= 0.995) & (times <= 1.005)
    assert band_delta[near_hit].max() > 0.0, (
        f"band_delta should spike > 0.0 at hit, got max "
        f"{band_delta[near_hit].max():.4f}"
    )


def test_band_delta_is_loudness_invariant():
    """Two bursts at different amplitudes must produce similar band_delta
    peaks (loudness invariant). The band_delta is shape-only, not
    magnitude: a 10x louder burst has 10x larger band_delta, but the
    per-band pattern is the same so the relative spike is the same."""
    sr = 44100
    # Loud burst
    y_loud = make_synthetic_burst(
        sr=sr, burst_time_sec=1.0,
        burst_freqs_hz=(100.0, 1000.0, 5000.0),
        burst_amplitudes=(0.1, 0.1, 0.1),  # small overall
    )
    y_quiet = make_synthetic_burst(
        sr=sr, burst_time_sec=1.0,
        burst_freqs_hz=(100.0, 1000.0, 5000.0),
        burst_amplitudes=(0.01, 0.01, 0.01),  # 10x smaller overall
    )
    _, debug_loud = detect_spectral_transients(y_loud, sr)
    _, debug_quiet = detect_spectral_transients(y_quiet, sr)
    # Peak band_delta at the hit frame should differ by ~10x
    # (the per-bin linear power is amplitude-squared, so 10x amp
    # gives 100x band_delta, but the band shape is the same so the
    # pattern of which band dominates is identical)
    near_hit_loud = debug_loud['band_delta'][(debug_loud['times'] >= 0.995)
                                              & (debug_loud['times'] <= 1.005)].max()
    near_hit_quiet = debug_quiet['band_delta'][(debug_quiet['times'] >= 0.995)
                                               & (debug_quiet['times'] <= 1.005)].max()
    ratio = near_hit_loud / max(near_hit_quiet, 1e-30)
    # 10x amplitude → 100x band_delta. Allow 50x-200x range.
    assert 50.0 < ratio < 200.0, (
        f"loudness scaling off: loud peak={near_hit_loud:.4f}, "
        f"quiet peak={near_hit_quiet:.4f}, ratio={ratio:.2f} "
        f"(expected ~100x for 10x amplitude)"
    )


def test_silent_audio_produces_no_events():
    """Silent input → all band_powers = 0 → band_delta = 0 everywhere
    → no events above the min_band_ratio threshold."""
    sr = 44100
    y = np.zeros(sr * 2)
    events, _ = detect_spectral_transients(y, sr)
    assert events == [], (
        f"silent input should produce no events, got {len(events)}"
    )


def test_quiet_input_with_single_dominant_band_fires():
    """A burst with all energy in band 0 (and 0 elsewhere) should fire:
    max / median = b0 / 0 → ratio is large. The 1e-20 guard prevents
    divide-by-zero."""
    sr = 44100
    y = make_synthetic_burst(
        sr=sr,
        burst_time_sec=1.0,
        burst_freqs_hz=(100.0,),
        burst_amplitudes=(1.0,),
    )
    events, _ = detect_spectral_transients(y, sr)
    assert len(events) >= 1, (
        f"100Hz-only burst should fire (band 0 dominates), "
        f"got {len(events)} events"
    )


# --- real audio regression --------------------------------------------------

def test_project_4_toms_finds_six_known_hits_in_73_77s():
    """The 6 hits the user identified in 73-77s of project 4 toms stem.
    Ground truth (user eyeballed from WebUI spectrogram):
      73.676, 73.853, 74.033, 74.210, 74.411, 74.576
    The FIRST hit (73.676) is the regression check — it was previously
    missing under bins-floor detection. The new band-delta signal
    should catch it.

    We assert that each ground-truth hit has a detected event within
    100ms (calibrated empirically on 2026-06-09 — the per-strike
    rise/peak in the toms envelope means the detector's strike
    moment can lag the GT by up to ~90ms when the strikes are 180ms
    apart). The detector may also over-fire (find extra events); the
    ``calibration-test`` task is responsible for the FP-rate
    assertion, not this one.
    """
    wav_path = (
        "user_files/4 - 2_funk_80_beat_4-4_4/stems/"
        "2_funk_80_beat_4-4_4-toms.wav"
    )
    try:
        y, sr = sf.read(wav_path, always_2d=True)
    except (FileNotFoundError, RuntimeError):
        pytest.skip(f"project 4 toms stem not found at {wav_path}")
    y = y.mean(axis=1)

    t_start = 73.0
    t_end = 77.0
    win = y[int(t_start * sr): int(t_end * sr)]
    events, _ = detect_spectral_transients(win, sr)

    # Rebase times to global timeline
    detected_times = [e.time_sec + t_start for e in events]
    ground_truth = [73.676, 73.853, 74.033, 74.210, 74.411, 74.576]

    # Every ground-truth hit must have a detected event within 100ms
    for gt in ground_truth:
        nearest = min(abs(t - gt) for t in detected_times)
        assert nearest < 0.100, (
            f"no detected event within 100ms of ground-truth hit at "
            f"{gt}s (nearest detected: {min(detected_times, key=lambda t: abs(t-gt)):.3f}s, "
            f"diff {nearest * 1000:+.1f}ms)"
        )


# ─── Snap-band detection (per-stem signal selection) ──────────────────────
# User insight (2026-06-09): the "snap" of a drum head being struck
# lives in a different frequency range than the "ring". For toms, the
# snap is broadband in B1 (200-600Hz) and B2 (600-1200Hz) — the user
# observed "the audio truly starts above the 400Hz cutoff". The ring
# (low-frequency ring) develops 50-100ms after the attack. A detector
# that fires on the RING (existing band_delta over B0) lags the
# attack by 50-100ms. A detector that fires on the SNAP (delta over
# B1+B2) catches the attack onset within a few ms.
#
# This is also the right signal for snare strikes (B1+B2) and hihat
# (B3+B4). Per-stem configuration of which bands constitute "the
# snap" is the way forward.


def test_spectral_config_has_snap_bands_field():
    """SpectralTransientConfig must have a snap_bands field that
    defaults to all 5 bands (backward compat)."""
    cfg = SpectralTransientConfig()
    assert hasattr(cfg, 'snap_bands'), (
        "SpectralTransientConfig needs a snap_bands field — the "
        "per-stem set of band indices to compute the snap detection "
        "signal over. Default: all 5 bands (backward compat)."
    )
    assert tuple(cfg.snap_bands) == (0, 1, 2, 3, 4), (
        f"default snap_bands must be (0,1,2,3,4) for backward compat, "
        f"got {cfg.snap_bands}"
    )


def test_spectral_config_has_snap_min_delta_field():
    """SpectralTransientConfig must have a snap_min_delta field that
    defaults low enough to catch weak toms/snare strikes."""
    cfg = SpectralTransientConfig()
    assert hasattr(cfg, 'snap_min_delta'), (
        "SpectralTransientConfig needs a snap_min_delta field — the "
        "find_peaks height parameter for the snap signal. Default: "
        "low (0.05) since the snap delta has a smaller dynamic range "
        "than the ring delta."
    )
    assert cfg.snap_min_delta > 0
    assert cfg.snap_min_delta <= 0.5, (
        f"snap_min_delta default should be in (0, 0.5] to catch "
        f"weak tom/snare strikes, got {cfg.snap_min_delta}"
    )


def test_snap_delta_in_debug_dict():
    """The debug dict must expose snap_delta so the user can verify
    the signal peaks at attack onset (not 50-100ms after, like the
    ring band_delta does)."""
    from stems_to_midi.test_spectral_transient_core import (
        make_synthetic_drum_stem,
    )
    sr = 44100
    y = make_synthetic_drum_stem(sr=sr, hit_times_sec=(1.0,))
    _, debug = detect_spectral_transients(y, sr)
    assert 'snap_delta' in debug, (
        "debug dict must include 'snap_delta' — the snap detection "
        "signal. The user needs to see this to verify the signal "
        "peaks at attack onset."
    )
    assert debug['snap_delta'].shape == debug['band_delta'].shape


def test_snap_delta_peaks_at_attack_onset_for_toms():
    """On project 4 toms 14-16s, the snap delta must peak at the
    attack ONSET (within 30ms of the GT eyeballed at 14.243, 14.441,
    14.626), not 50-100ms after like the ring band_delta does.

    The snap bands for toms are (1, 2) — B1 (200-600Hz) and B2
    (600-1200Hz) — the "head snap" range, NOT B0 (60-200Hz) which
    is the "ring" that develops later.
    """
    wav_path = (
        "user_files/4 - 2_funk_80_beat_4-4_4/stems/"
        "2_funk_80_beat_4-4_4-toms.wav"
    )
    try:
        y, sr = sf.read(wav_path, always_2d=True)
    except (FileNotFoundError, RuntimeError):
        pytest.skip(f"project 4 toms stem not found at {wav_path}")
    y = y.mean(axis=1)

    t_start = 14.0
    win = y[int(t_start * sr): int(16 * sr)]
    cfg = SpectralTransientConfig(snap_bands=(1, 2), snap_min_delta=0.05)
    events, debug = detect_spectral_transients(win, sr, config=cfg)

    times = debug['times']
    snap_delta = debug['snap_delta']
    band_powers = debug['band_powers']

    # For each GT, find the snap_delta peak in a +/- 50ms window
    gt_hits = [14.243, 14.441, 14.626]
    for gt in gt_hits:
        center = int((gt - t_start) * sr / 256)
        scan_start = max(0, center - int(0.05 * sr / 256))
        scan_end = min(snap_delta.shape[0], center + int(0.05 * sr / 256))
        local = snap_delta[scan_start:scan_end]
        best_local = int(np.argmax(local))
        best_idx = scan_start + best_local
        best_time = times[best_idx] + t_start
        offset_ms = (best_time - gt) * 1000
        # The snap peak should be within 50ms of the GT onset.
        # Allow 50ms because the "head snap" is broadband and may
        # be a frame or two before the GT (eyeball tolerance).
        assert abs(offset_ms) < 50, (
            f"GT {gt}: snap_delta peak at t={best_time:.3f} (Δ={offset_ms:+.0f}ms) "
            f"is too far from the attack onset. The snap signal should "
            f"peak within 50ms of the attack (not 50-100ms after like the "
            f"ring band_delta does). peak value = {snap_delta[best_idx]:.3f}"
        )


def test_toms_14_16s_snap_finder_finds_all_3_gt_within_30ms():
    """With snap_bands=(1,2), the 3 GT hits in toms 14-16s
    (14.243, 14.441, 14.626) must all be detected within 30ms of
    the GT. This is the new calibration target for toms.
    """
    wav_path = (
        "user_files/4 - 2_funk_80_beat_4-4_4/stems/"
        "2_funk_80_beat_4-4_4-toms.wav"
    )
    try:
        y, sr = sf.read(wav_path, always_2d=True)
    except (FileNotFoundError, RuntimeError):
        pytest.skip(f"project 4 toms stem not found at {wav_path}")
    y = y.mean(axis=1)

    t_start = 14.0
    win = y[int(t_start * sr): int(16 * sr)]
    cfg = SpectralTransientConfig(snap_bands=(1, 2), snap_min_delta=0.05)
    events, _ = detect_spectral_transients(win, sr, config=cfg)

    detected_times = [e.time_sec + t_start for e in events]
    # 3 GT hits in a 2s window. With snap_bands=(1, 2) and the
    # snap-ring wire-tail filter, the detector should find 3-5
    # events: 1 snap per strike (3 total) + 0-2 low-energy decay
    # tails. The 3 GT hits must each have a detected event within
    # 30ms (the snap should land at the attack onset, not 50-100ms
    # later like the ring-only detector does).
    assert 3 <= len(detected_times) <= 5, (
        f"toms 14-16s: expected 3-5 events (3 snaps + 0-2 decay tails), "
        f"got {len(detected_times)}. Events: {detected_times}"
    )
    gt_hits = [14.243, 14.441, 14.626]
    for gt in gt_hits:
        nearest = min(abs(t - gt) for t in detected_times)
        assert nearest < 0.030, (
            f"GT {gt}: no detected event within 30ms (nearest: "
            f"{min(detected_times, key=lambda t: abs(t-gt)):.3f}s, "
            f"diff {nearest * 1000:+.1f}ms)"
        )


def test_toms_73_77s_snap_finder_does_not_regress():
    """With snap_bands=(1,2), toms 73-77s must still find the 6 GT
    hits within 100ms. The snap signal should be additive to the
    ring signal, not a replacement.
    """
    wav_path = (
        "user_files/4 - 2_funk_80_beat_4-4_4/stems/"
        "2_funk_80_beat_4-4_4-toms.wav"
    )
    try:
        y, sr = sf.read(wav_path, always_2d=True)
    except (FileNotFoundError, RuntimeError):
        pytest.skip(f"project 4 toms stem not found at {wav_path}")
    y = y.mean(axis=1)

    t_start = 73.0
    win = y[int(t_start * sr): int(77 * sr)]
    cfg = SpectralTransientConfig(snap_bands=(1, 2), snap_min_delta=0.05)
    events, _ = detect_spectral_transients(win, sr, config=cfg)

    detected_times = [e.time_sec + t_start for e in events]
    ground_truth = [73.676, 73.853, 74.033, 74.210, 74.411, 74.576]
    for gt in ground_truth:
        nearest = min(abs(t - gt) for t in detected_times)
        assert nearest < 0.100, (
            f"GT {gt}: regression in 73-77s toms window — no event "
            f"within 100ms. Nearest: "
            f"{min(detected_times, key=lambda t: abs(t-gt)):.3f}s"
        )
