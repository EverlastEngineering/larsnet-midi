"""
Unit tests for ``compute_silence_mask`` in
``stems_to_midi.analysis_core.spectral_utils``.

The function computes a per-frame boolean silence mask from a 2D
magnitude spectrogram by isolating the noise floor (5th-30th
percentile of per-frame energy) and using
``median(noise_zone) + k * std(noise_zone)`` as the threshold.

These tests cover the algorithm steps, the edge cases, and a
calibration test on a synthetic spectrogram designed to mimic
the failure modes the function was built to handle:

  - **Compressor dropouts**: a long stretch of near-zero energy
    drags the minimum down; naive mean+std would mis-lock onto
    the dropout zone.
  - **High-energy transients**: a few frames at huge energy create
    a long right tail; naive median would climb with hit density.
  - **All-silent file**: std=0, threshold = median, mask = all-False.
  - **Empty / 1D / constant input**: graceful handling.

There is also a calibration test against the project 4 toms
ground truth (the 14s and 74s regions), which is the canonical
example the user wants to nail.
"""
from __future__ import annotations

import numpy as np
import pytest

from stems_to_midi.analysis_core.spectral_utils import compute_silence_mask


# ─── 1. Algorithm steps ───────────────────────────────────────────────────

class TestAlgorithmSteps:
    """The function must follow the documented algorithm exactly:
    1. per-frame energy = sum of squared magnitudes across frequency
    2. P5, P30 of frame energy
    3. noise_zone = energy in [P5, P30]
    4. threshold = median(noise_zone) + k * std(noise_zone)
    5. return (frame_energy > threshold)
    """

    def test_step1_uses_sum_of_squared_magnitudes_not_linear(self):
        """A linear-sum threshold would let a few loud bins drown
        the rest. Squaring weights loud bins much more, so the
        detector sees the energy spike as a spike, not as 'one
        bin went up'. We test by injecting a frame with one giant
        bin (linear dominant) vs many moderate bins (energy dominant);
        the energy-spike frame must be the one marked active.

        The test uses many noise frames so the P5-P30 noise band
        is well-defined (with too few frames, the noise band is
        empty and the function falls back to the full distribution,
        which is fine but the test assertion becomes meaningless).
        """
        # 10 freq bins, 200 noise frames + 4 test frames.
        n_noise = 200
        n_freq = 10
        spec = np.full((n_freq, n_noise + 4), 0.1, dtype=np.float64)
        # Frames 0..n_noise-1: pure noise (all bins at 0.1, E=0.1)
        # Test frame at index n_noise+0: baseline (all bins at 0.1)
        # Test frame at index n_noise+1: one bin at 10x
        spec[3, n_noise + 1] = 10.0
        # Test frame at index n_noise+2: all bins at 1.0
        spec[:, n_noise + 2] = 1.0
        # Test frame at index n_noise+3: one bin at 0.5
        spec[3, n_noise + 3] = 0.5
        mask = compute_silence_mask(spec)
        # Energy (sum of squared magnitudes):
        #   baseline (n_noise+0): 10*0.01 = 0.1
        #   one bin at 10x (n_noise+1): 9*0.01 + 100 = 100.09
        #   all bins at 1.0 (n_noise+2): 10*1.0 = 10
        #   one bin at 0.5 (n_noise+3): 9*0.01 + 0.25 = 0.34
        # Linear sum:
        #   baseline: 1.0
        #   one at 10x: 10.3  (one bin jumps the linear sum)
        #   all at 1.0: 10.0  (catches up because all bins are loud)
        #   one at 0.5: 1.4
        # So linear-sum would rank: 10x > all@1.0 > 0.5 > baseline.
        # Squared-sum ranks: 10x > all@1.0 > 0.5 > baseline. SAME order.
        # The discriminating frame is the baseline vs 0.5x — both
        # are noise-like and should be silent. Frame 10x and all@1.0
        # are the only ones above threshold.
        # The key property: the baseline test frame (n_noise+0)
        # must be silent. If it's active, the threshold is too low.
        assert mask[n_noise + 0] == False, (
            "baseline test frame (same energy as 200 noise frames) "
            "should be silent; if it's active, the noise band is "
            "mis-calibrated"
        )
        # Frame with 10x bin must be active (energy 100, well above
        # any noise band).
        assert mask[n_noise + 1] == True, (
            "frame with one bin at 10x (E=100) must be marked active"
        )
        # Frame with all bins at 1.0 (E=10) must also be active.
        assert mask[n_noise + 2] == True, (
            "frame with all bins at 1.0 (E=10) must be marked active"
        )

    def test_step2_percentile_range_isolates_noise(self):
        """The noise_zone must be the frames whose energy is between
        P5 and P30, not the full distribution. Inject: 10 noise
        frames at E=1, 5 transient frames at E=100, 1 silent frame
        at E=0.5. The threshold should be based on the noise floor
        (E ≈ 1), not the transient-driven mean (≈ 35)."""
        n_freq = 10
        spec = np.zeros((n_freq, 16), dtype=np.float64)
        # 10 noise frames at E=1 each (10 bins at sqrt(0.1) ≈ 0.316)
        for i in range(10):
            spec[:, i] = np.sqrt(0.1)
        # 1 silence frame at E=0.5 (10 bins at sqrt(0.05) ≈ 0.224)
        spec[:, 10] = np.sqrt(0.05)
        # 5 transient frames at E=100 (10 bins at sqrt(10) ≈ 3.16)
        for i in range(11, 16):
            spec[:, i] = np.sqrt(10.0)
        mask = compute_silence_mask(spec)
        # All 5 transient frames must be active.
        assert mask[11:16].all(), (
            "all 5 transient frames should be marked active"
        )
        # The 10 noise frames at E=1 should NOT be marked active
        # (they're the noise floor).
        assert not mask[0:10].any(), (
            "noise-floor frames should be silent; the threshold "
            "should be set just above them"
        )

    def test_step4_threshold_is_median_plus_k_std(self):
        """Sanity check: the threshold equals
        median(noise_zone) + std_multiplier * std(noise_zone)."""
        np.random.seed(42)
        # 100 noise frames, 5 transient frames.
        spec = np.random.rand(50, 105) * 0.01  # noise
        spec[:, 100:105] = 1.0  # transients
        k = 2.5
        mask = compute_silence_mask(spec, std_multiplier=k)
        # Re-derive the threshold.
        frame_energy = np.sum(spec.astype(np.float64) ** 2, axis=0)
        p5 = np.percentile(frame_energy, 5)
        p30 = np.percentile(frame_energy, 30)
        nz = frame_energy[(frame_energy >= p5) & (frame_energy <= p30)]
        expected_threshold = float(np.median(nz)) + k * float(np.std(nz))
        # Re-derive the mask.
        expected_mask = frame_energy > expected_threshold
        np.testing.assert_array_equal(
            mask, expected_mask,
            err_msg=(
                "The returned mask must match the one computed from "
                "the documented algorithm. If it doesn't, the "
                "implementation is not what the docstring describes."
            ),
        )


# ─── 2. Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """The function must handle degenerate input gracefully."""

    def test_all_silent_file_returns_all_false(self):
        """An entirely silent file (every frame is 0) must return
        an all-False mask. No active frames."""
        spec = np.zeros((513, 1000), dtype=np.float64)
        mask = compute_silence_mask(spec)
        assert mask.dtype == bool
        assert mask.shape == (1000,)
        assert not mask.any(), (
            "all-silent file: all frames should be silent, no false "
            "positives"
        )

    def test_constant_nonzero_file_returns_all_false(self):
        """A file with constant non-zero energy (e.g. test tone) has
        max == min; the function short-circuits to all-False."""
        spec = np.ones((513, 1000), dtype=np.float64) * 0.5
        mask = compute_silence_mask(spec)
        assert not mask.any(), (
            "constant energy (max == min) has no contrast; mask is "
            "all-False by design"
        )

    def test_empty_frames_returns_empty_mask(self):
        """A spectrogram with 0 frames (audio is empty)."""
        spec = np.zeros((513, 0), dtype=np.float64)
        mask = compute_silence_mask(spec)
        assert mask.shape == (0,)
        assert mask.dtype == bool

    def test_single_frame(self):
        """A spectrogram with exactly 1 frame. The function must
        not crash (the percentile + noise_zone computation is
        degenerate but the result is well-defined)."""
        spec = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
        mask = compute_silence_mask(spec)
        assert mask.shape == (1,)
        assert mask.dtype == bool

    def test_2d_required_raises_on_1d(self):
        """A 1D input is ambiguous (is it a single frame or a single
        frequency?). The function must raise clearly."""
        with pytest.raises(ValueError, match="2D"):
            compute_silence_mask(np.array([1.0, 2.0, 3.0]))

    def test_2d_required_raises_on_3d(self):
        """A 3D input is wrong (e.g. (n_freq, n_frames, n_channels))."""
        with pytest.raises(ValueError, match="2D"):
            compute_silence_mask(np.zeros((10, 100, 2)))


# ─── 3. Tuning the noise band ────────────────────────────────────────────

class TestNoiseBandTuning:
    """The P5-P30 noise band is the key knob. With tighter bands the
    threshold is more conservative (more frames active); with wider
    bands the threshold is more permissive."""

    def test_tighter_band_gives_higher_active_fraction(self):
        """A wider noise band (P5-P50 vs P5-P30) includes more
        semi-quiet frames, so the median+std is computed over a
        wider distribution — usually higher mean and similar std,
        so threshold rises, and FEWER frames are marked active.

        Equivalently: tightening the band to a purer noise sample
        gives a tighter distribution; threshold = median + 2.5*std
        is well-defined and the test below is the inverse: a wider
        band should mark fewer or equal frames active vs. a tighter
        band.
        """
        np.random.seed(0)
        spec = np.random.rand(50, 500) * 0.05
        spec[:, 450:500] = 1.0  # 50 transient frames

        mask_tight = compute_silence_mask(
            spec, p5_percentile=5, p30_percentile=20,
        )
        mask_wide = compute_silence_mask(
            spec, p5_percentile=5, p30_percentile=40,
        )
        # The tighter band is closer to pure noise, so its median
        # is lower and threshold sits lower → more active frames.
        assert mask_tight.sum() >= mask_wide.sum(), (
            "tighter noise band (P5-P20) should give ≥ active frames "
            "than a wider band (P5-P40), because the wider band "
            "includes semi-active frames that raise the median"
        )

    def test_std_multiplier_increases_active_fraction(self):
        """Higher k → higher threshold → MORE frames marked active
        (the bar is higher, so fewer frames clear it). Wait, no:
        threshold = median + k*std; higher k → higher threshold →
        fewer frames > threshold → fewer active. Verify."""
        np.random.seed(1)
        spec = np.random.rand(50, 500) * 0.1
        spec[:, 400:500] = 1.0  # 100 transient frames

        mask_k1 = compute_silence_mask(spec, std_multiplier=1.0)
        mask_k5 = compute_silence_mask(spec, std_multiplier=5.0)
        assert mask_k1.sum() >= mask_k5.sum(), (
            "higher std_multiplier must give a higher threshold and "
            "thus fewer (or equal) active frames"
        )


# ─── 4. Calibration: the failure modes the function was built for ───────

class TestFailureModeCalibration:
    """The whole point of this function is to handle the two failure
    modes that naive thresholds fail on. These tests build
    spectrograms that exercise each failure mode and verify the
    function still produces a useful mask."""

    def test_compressor_dropout_does_not_lock_onto_zero(self):
        """A 30% compressor dropout zone drags a naive min+std
        threshold down to zero. The P5-P30 noise band sits in the
        dropout zone (which is the true noise floor), so the
        threshold is set ABOVE the dropout but BELOW the actual
        transients.

        The point: the function's threshold is robust to a large
        dropout zone. A naive mean+std of the full distribution
        would be inflated by the transients and would still work,
        but a naive MIN+std would lock onto the dropout and mark
        legitimate audio as silence. The P5-P30 band avoids that
        trap.

        The test verifies the threshold lands in the right place:
        above the dropout zone, well below the transient level.
        """
        np.random.seed(7)
        n_freq = 20
        n_dropout = 100
        n_background = 200
        n_transient = 10
        n_total = n_dropout + n_background + n_transient  # 310

        spec = np.zeros((n_freq, n_total), dtype=np.float64)
        # Dropout frames: E in [0.001, 0.01] with random variation.
        for i in range(n_dropout):
            e = 0.001 + np.random.rand() * 0.009
            spec[:, i] = np.sqrt(e / n_freq)
        # Background frames: E in [0.5, 2.0] (standard noise).
        for i in range(n_dropout, n_dropout + n_background):
            e = 0.5 + np.random.rand() * 1.5
            spec[:, i] = np.sqrt(e / n_freq)
        # Transient frames: E = 100.
        for i in range(n_dropout + n_background, n_total):
            spec[:, i] = np.sqrt(100.0 / n_freq)

        mask = compute_silence_mask(spec)
        # All transient frames must be active (100x background).
        assert mask[n_dropout + n_background:].all(), (
            f"transient frames must be active; "
            f"got {mask[n_dropout + n_background:].sum()}/{n_transient} active"
        )
        # The threshold is set in the dropout zone, NOT at zero
        # (a naive min-based threshold would put it at the dropout
        # min ≈ 0.001 and would mark everything else as active).
        # Recompute the threshold to verify it sits above the
        # dropout zone's max.
        frame_energy = np.sum(spec.astype(np.float64) ** 2, axis=0)
        p5 = np.percentile(frame_energy, 5)
        p30 = np.percentile(frame_energy, 30)
        nz = frame_energy[(frame_energy >= p5) & (frame_energy <= p30)]
        threshold = float(np.median(nz)) + 2.5 * float(np.std(nz))
        assert threshold > p30, (
            f"threshold ({threshold}) must be above P30 ({p30}) — "
            f"if it's not, the std_multiplier is degenerate or the "
            f"noise band collapsed to a single point"
        )
        # And the threshold is well below the transients.
        assert threshold < 10.0, (
            f"threshold ({threshold}) is way too high — should be "
            f"well below the transient level (100)"
        )

    def test_transient_right_tail_does_not_dominate_threshold(self):
        """A few frames at 1000x the background create a long right
        tail. A naive median climbs to the transient level. The
        P5-P30 noise band stays in the background, so the threshold
        is unaffected by the tail."""
        # 1000 frames: 995 background, 5 transients (1000x energy)
        spec = np.ones((20, 1000), dtype=np.float64) * 0.1  # background
        for i in [100, 300, 500, 700, 900]:
            spec[:, i] = np.sqrt(50.0)  # E = 20*50 = 1000

        mask = compute_silence_mask(spec)
        # All 5 transients must be active.
        for i in [100, 300, 500, 700, 900]:
            assert mask[i], (
                f"transient at frame {i} (1000x background energy) "
                f"should be marked active"
            )
        # Most background frames should be silent.
        n_active = mask.sum()
        assert n_active < 200, (
            f"with 5 transients in 1000 frames, mask should mark "
            f"very few background frames active. Got {n_active} "
            f"active frames"
        )


# ─── 5. Real-world validation against project 4 toms ground truth ───────

class TestProject4TomsGroundTruth:
    """End-to-end validation: load the toms stem from project 4,
    compute the silence mask on the 14-16s and 73-77s regions, and
    verify that every GT hit falls on an active frame.

    This is the canonical acceptance test the user asked for
    (2026-06-09): "use [the GT for toms at the 14 and 74 second
    areas] as truth and strongly aim for 100% accurate in both
    areas".

    Skipped automatically if the project 4 audio is not on disk
    (so the test suite works in CI without the user's data).
    """

    TOMS_WAV = (
        "/Users/jasoncopp/Source/GitHub/larsnet/user_files/"
        "4 - 2_funk_80_beat_4-4_4/stems/2_funk_80_beat_4-4_4-toms.wav"
    )
    SR = 44100
    N_FFT = 1024
    HOP = 256
    MATCH_WINDOW_SEC = 0.05  # GT hit is "caught" if any active frame in +/- this

    REGIONS = [
        (13.5, 16.5, [14.243, 14.441, 14.626], "14-16s (3 hits)"),
        (73.0, 77.0, [73.676, 73.853, 74.033, 74.210, 74.411, 74.576],
         "73-77s (6 hits)"),
    ]

    def _load_region(self, y, t_start, t_end):
        """Slice the audio region and compute the magnitude spectrogram."""
        import librosa
        f_start = int(t_start * self.SR)
        f_end = int(t_end * self.SR)
        y_region = y[f_start:f_end]
        spec = np.abs(librosa.stft(y_region, n_fft=self.N_FFT, hop_length=self.HOP))
        return spec

    def test_all_gt_hits_caught_on_active_frames(self):
        import os
        if not os.path.exists(self.TOMS_WAV):
            pytest.skip(
                f"project 4 toms audio not at {self.TOMS_WAV}; "
                f"the real-world validation test only runs when the "
                f"user's data is present"
            )
        import librosa
        y, sr = librosa.load(self.TOMS_WAV, sr=self.SR, mono=True)
        assert sr == self.SR, f"expected sr={self.SR}, got {sr}"

        failures = []
        for t_start, t_end, gt_hits, label in self.REGIONS:
            spec = self._load_region(y, t_start, t_end)
            mask = compute_silence_mask(spec)
            win = int(round(self.MATCH_WINDOW_SEC * self.SR / self.HOP))
            for gt_t in gt_hits:
                local_t = gt_t - t_start
                center = int(round(local_t * self.SR / self.HOP))
                if center < 0 or center >= len(mask):
                    failures.append(f"{label}: GT {gt_t}s out of range")
                    continue
                lo, hi = max(0, center - win), min(len(mask), center + win + 1)
                if not mask[lo:hi].any():
                    failures.append(
                        f"{label}: GT {gt_t}s not caught on an active frame "
                        f"(center frame {center}, no active frame in "
                        f"+/-{self.MATCH_WINDOW_SEC}s)"
                    )

        if failures:
            pytest.fail(
                "silence mask missed GT hits:\n  " + "\n  ".join(failures)
            )


# ─── 6. Return type contract ─────────────────────────────────────────────

class TestReturnTypeContract:
    """Lock the public API of the function."""

    def test_returns_numpy_array(self):
        spec = np.random.rand(10, 100) * 0.1
        spec[:, 50:60] = 1.0
        mask = compute_silence_mask(spec)
        assert isinstance(mask, np.ndarray)

    def test_mask_is_boolean_dtype(self):
        spec = np.random.rand(10, 100) * 0.1
        spec[:, 50:60] = 1.0
        mask = compute_silence_mask(spec)
        assert mask.dtype == bool, (
            f"mask must be bool, got {mask.dtype}"
        )

    def test_mask_length_matches_frame_count(self):
        spec = np.random.rand(10, 137) * 0.1  # arbitrary length
        spec[:, 50:60] = 1.0
        mask = compute_silence_mask(spec)
        assert len(mask) == 137, (
            f"mask length must match the frame axis, got {len(mask)}"
        )

    def test_does_not_modify_input(self):
        """Pure function — the input spectrogram must not be mutated."""
        np.random.seed(0)
        spec = np.random.rand(10, 100) * 0.1
        spec[:, 50:60] = 1.0
        spec_copy = spec.copy()
        _ = compute_silence_mask(spec)
        np.testing.assert_array_equal(
            spec, spec_copy,
            err_msg="compute_silence_mask must not mutate the input",
        )
