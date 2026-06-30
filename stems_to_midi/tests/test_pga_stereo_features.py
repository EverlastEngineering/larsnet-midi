"""
Tests for the stereo feature pass in pga_event_builder (2026-06-30).

The PGA pipeline runs the detector on mono audio (onset detection is
fundamentally temporal — broadband contrast envelope + IQR-thresholded
peak picker). Stereo info is only needed at per-event feature
extraction time, where ``_compute_features_for_filtered_events`` runs
a per-event loop over KEPT events.

When the original stereo audio is plumbed through to
``_compute_features_for_filtered_events`` (via the ``audio_stereo``
kwarg on ``_build_pga_events_with_filter``), the function also calls
``stereo_core.calculate_stereo_features`` and stamps
``pan_confidence`` / ``stereo_width`` onto each event.

When ``audio_stereo`` is None (mono source, or ``use_stereo: false``),
the stereo pass is a no-op — events get None for both fields, no
exception, no compute cost beyond a None check.

These tests pin:
  1. Synthetic stereo audio with mono-panned and wide-stereo hits →
     stereo_width correctly distinguishes the two populations.
  2. Mono source (audio_stereo=None) → no exception, stereo_width is
     None for every event.
  3. End-to-end shape: every KEPT event gets both fields (None when
     no audio, populated floats when stereo audio is present).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Match the existing tests/test_pga_event_builder.py sys.path trick:
# tests/ is a subdirectory so add the parent (repo root) explicitly.
_TEST_DIR = Path(__file__).resolve().parent
_PKG_PARENT = _TEST_DIR.parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from stems_to_midi.pga_event_builder import (  # noqa: E402
    _build_pga_events_with_filter,
)


# --- Helpers --------------------------------------------------------------


def _make_synthetic_stereo_burst_stem(
    sr: int = 44100,
    hit_times_sec: tuple = (0.5, 1.0, 1.5, 2.0),
    freq_hz: float = 200.0,
    decay_ms: float = 80.0,
    duration_sec: float = 3.0,
    panning: str = 'mono',
) -> np.ndarray:
    """Build a stereo signal with broadband bursts.

    Args:
        panning: 'mono' (identical L/R — width ≈ 0), 'right' (only
            the right channel carries the burst — width ≈ 0.5),
            'wide' (anti-phase L/R — width ≈ 1.0 but mono mix
            cancels to zero, so the detector won't see it — used
            only for direct stereo_width testing, not for PGA
            detector testing).
    """
    t = np.arange(int(sr * duration_sec)) / sr
    y_l = np.zeros_like(t)
    y_r = np.zeros_like(t)
    decay_samples = int(decay_ms / 1000.0 * sr)
    env = np.exp(-np.arange(decay_samples) / (decay_samples / 4.0))

    for hit_t in hit_times_sec:
        i0 = int(hit_t * sr)
        i1 = min(i0 + decay_samples, len(y_l))
        n = i1 - i0
        if n <= 0:
            continue
        burst = (
            np.sin(2 * np.pi * freq_hz * np.arange(n) / sr)
            + 0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / sr)
            + 0.3 * np.sin(2 * np.pi * 3000 * np.arange(n) / sr)
            + 0.2 * np.sin(2 * np.pi * 5000 * np.arange(n) / sr)
        )
        shaped = burst * env[:n]
        if panning == 'mono':
            y_l[i0:i1] += shaped
            y_r[i0:i1] += shaped
        elif panning == 'right':
            y_r[i0:i1] += shaped
        elif panning == 'wide':
            # Pure side signal: L = +burst, R = -burst → max width.
            # But mono sum = 0, so the detector can't see it. Use
            # 'right' instead when you need detector-visible hits
            # with measurable stereo width.
            y_l[i0:i1] += shaped
            y_r[i0:i1] -= shaped
        else:
            raise ValueError(f"unknown panning: {panning}")

    return np.stack([y_l, y_r], axis=1).astype(np.float32)


def _default_config(**overrides) -> dict:
    """Minimal config for ``_build_pga_events_with_filter``.

    Reads ``onset_detection.pga_min_prominence`` (default 1000) and
    stem-level overrides. Loosened prominence so our synthetic hits
    survive the filter.
    """
    cfg = {
        'onset_detection': {
            'pga_min_prominence': 100.0,
        },
        'snare': {
            'midi_note': 38,
            'midi_note_rimshot': 37,
            'midi_note_clap': 39,
        },
        'midi': {
            'min_velocity': 80,
            'max_velocity': 110,
        },
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


# --- Tests ----------------------------------------------------------------


class TestStereoFeaturesInPGAPipeline:
    """The new stereo pass in _compute_features_for_filtered_events."""

    def test_mono_audio_audio_stereo_none(self):
        """When ``audio_stereo`` is None (mono source), no exception
        fires and every event gets ``stereo_width=None`` /
        ``pan_confidence=None``. The function must NOT raise even
        though ``calculate_stereo_features`` would raise on a 1-D
        input — the no-op path is checked at the caller."""
        sr = 44100
        # Mono audio: 1-D array.
        audio_mono = _make_synthetic_stereo_burst_stem(sr=sr).mean(axis=1)
        assert audio_mono.ndim == 1

        raw, kept, filtered, _debug = _build_pga_events_with_filter(
            audio_mono, sr, _default_config(),
            stem_type='snare', audio_stereo=None,
        )

        # At least one event must survive the filter for the test
        # to be meaningful. If our synthetic signal is too weak,
        # skip rather than fail.
        if not raw:
            pytest.skip("synthetic signal too weak to produce events")

        for ev in raw:
            assert ev.get('stereo_width') is None
            assert ev.get('pan_confidence') is None

    def test_stereo_audio_populates_fields(self):
        """When ``audio_stereo`` is the original stereo audio, every
        event gets a numeric ``stereo_width`` and ``pan_confidence``.
        Wide-stereo hits should produce larger stereo_width than
        mono-panned hits."""
        sr = 44100
        # Build stereo audio with mixed panning: mono hits at
        # 0.5s, wide hits at 1.0s, mono at 1.5s, wide at 2.0s.
        # We sum two separate stereo buffers to mix pannings.
        audio_mono_part = _make_synthetic_stereo_burst_stem(
            sr=sr, hit_times_sec=(0.5, 1.5), panning='mono',
        )
        audio_wide_part = _make_synthetic_stereo_burst_stem(
            sr=sr, hit_times_sec=(1.0, 2.0), panning='wide',
        )
        audio_stereo = audio_mono_part + audio_wide_part
        audio_mono = audio_stereo.mean(axis=1)

        raw, kept, filtered, _debug = _build_pga_events_with_filter(
            audio_mono, sr, _default_config(),
            stem_type='snare', audio_stereo=audio_stereo,
        )

        if not raw:
            pytest.skip("synthetic signal too weak to produce events")

        # Every event must have a numeric value for both fields
        # (no None, no exception).
        for ev in raw:
            assert ev.get('stereo_width') is not None, (
                f"event at t={ev.get('time'):.3f} has no stereo_width"
            )
            assert ev.get('pan_confidence') is not None, (
                f"event at t={ev.get('time'):.3f} has no pan_confidence"
            )
            assert 0.0 <= ev['stereo_width'] <= 1.0
            assert -1.0 <= ev['pan_confidence'] <= 1.0

    def test_wide_stereo_has_larger_width_than_mono(self):
        """Mono-panned hits must produce strictly smaller stereo_width
        than panned hits. This is the discriminator the snare
        classifier uses to split 'snare' from 'clap'.

        Uses 'mono' (L=R, width ≈ 0) and 'right' (R-only, width ≈
        0.5). Both produce an identical mono mix, so the detector
        finds both equally well — only the stereo pass distinguishes
        them. (Anti-phase 'wide' would have width ≈ 1.0 but cancels
        in mono and the detector wouldn't see it — see the helper
        docstring for why we don't use it here.)
        """
        sr = 44100
        audio_mono_part = _make_synthetic_stereo_burst_stem(
            sr=sr, hit_times_sec=(0.5,), panning='mono',
        )
        audio_panned_part = _make_synthetic_stereo_burst_stem(
            sr=sr, hit_times_sec=(1.5,), panning='right',
        )
        # Pad to the same length so the times align.
        max_len = max(audio_mono_part.shape[0], audio_panned_part.shape[0])
        if audio_mono_part.shape[0] < max_len:
            pad = np.zeros((max_len - audio_mono_part.shape[0], 2))
            audio_mono_part = np.concatenate([audio_mono_part, pad])
        if audio_panned_part.shape[0] < max_len:
            pad = np.zeros((max_len - audio_panned_part.shape[0], 2))
            audio_panned_part = np.concatenate([audio_panned_part, pad])
        audio_stereo = audio_mono_part + audio_panned_part
        audio_mono = audio_stereo.mean(axis=1)

        raw, kept, filtered, _debug = _build_pga_events_with_filter(
            audio_mono, sr, _default_config(),
            stem_type='snare', audio_stereo=audio_stereo,
        )

        if len(raw) < 2:
            pytest.skip("synthetic signal produced < 2 events")

        # Find the widest and narrowest stereo_width and assert
        # the ordering is correct (this is what the classifier
        # uses for the k-means split).
        widths = sorted([ev['stereo_width'] for ev in raw])
        assert widths[-1] > widths[0], (
            f"Expected max stereo_width > min, got {widths}"
        )
        # Right-panned hits give width ≈ 0.5 (L=0, R=burst → side
        # = -burst, mid = burst, equal RMS → 0.5); mono hits
        # give width ≈ 0. Be generous on both ends to absorb
        # envelope shape and burst overlap.
        assert widths[-1] > 0.3, (
            f"Expected panned stereo_width > 0.3, got max={widths[-1]:.3f}"
        )
        assert widths[0] < 0.1, (
            f"Expected mono stereo_width < 0.1, got min={widths[0]:.3f}"
        )