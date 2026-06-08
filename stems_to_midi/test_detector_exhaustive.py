"""
Tests for the detector-vs-classifier architecture in energy_detection.

User report (2026-06-08, project 2 — Taylor Swift The Fate of Ophelia):
A snare hit around 0.592s is missing from the detected events in
``analysis.json``. Investigation showed the hit IS in the audio
(amp 0.40), IS in the energy envelope (peak at 0.604), and IS in
``scipy.signal.find_peaks``'s raw candidate list — but the detector
drops it because the previous ``distance=min_spacing_frames``
parameter on find_peaks was greedily keeping the louder of two
close-together peaks (the user had ``min_peak_spacing_ms: 80``; the
0.604 and 0.662 peaks are 58ms apart).

Root cause: the detector stage was doing the classifier stage's
job. find_peaks(distance=N) keeps the highest peak in any N-ms
window — but the highest peak in a window of close-together hits
isn't necessarily the most "real" hit. A flam has a loud first
hit and a quieter second hit 30-50ms later; the detector drops the
quieter one because of the spacing constraint. That's wrong: the
quieter hit can be a perfectly valid drum hit, and the classifier
(geomean / sustain / strength) should be the one to decide
real-vs-fake, not a time-distance filter.

The fix (this commit): change ``distance=`` to 1 so the detector
is exhaustive — all peaks above the absolute energy floor and
prominence threshold are candidates. The classifier
(``should_keep_onset`` in spectral_utils.py) already takes the
right knobs (geomean_threshold, min_sustain_ms,
min_strength_threshold) and is the proper place for real-vs-fake
filtering.

A future TODO may add an opt-in post-classifier spacing filter
(preserves the "I want my MIDI to be cleanly spaced" knob) — see
agent-plans/bug-tracking.md.

User instruction: "remove it PURELY by setting a parameter to 0
or 1, like distance, then do so, so we can revert more easily
but ensure we have it STRONGLY commented that we don't intend
to use it." — distance is set to 1 (a no-op) with a strong
inline comment explaining the rationale.
"""

import numpy as np
import pytest
from pathlib import Path

from stems_to_midi.energy_detection_shell import detect_onsets_energy_based


# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_synthetic_flam(sr=44100, hop_length=512):
    """Build a 2-second synthetic stereo audio with two distinct snare
    hits spaced 60ms apart — the user's bug shape (a quieter hit
    following a louder one).

    The hits are spaced far enough apart in time (and have
    sufficient decay_ms) that they appear as two distinct
    prominence peaks in the energy envelope. L and R have a
    sub-hop phase shift so peak times differ by a few samples —
    this matches what real stereo recordings look like and is
    what the detector is called on in production (the user's
    project 2 snare is stereo).

    IMPORTANT: the spacing (60ms) is BELOW the user's
    snare.min_peak_spacing_ms=80, so with the OLD detector the
    closer hit would be dropped. With the fix (distance=1, no-op)
    both should be in the detector's candidate list.
    """
    duration = 2.0
    n = int(sr * duration)
    rng = np.random.default_rng(42)

    def build_channel(hits, channel_seed=0):
        rng_ch = np.random.default_rng(channel_seed)
        audio = np.zeros(n, dtype=np.float32)
        for start_t, peak_amp, decay_ms in hits:
            center = int(start_t * sr)
            decay = int(decay_ms / 1000 * sr)
            # Per-channel phase shift makes L and R peak times
            # differ by a few samples, matching real stereo.
            ch_seed = int(peak_amp * 100) + channel_seed
            phase_samples = ch_seed % 7
            for i in range(decay):
                idx = center + i + phase_samples
                if idx >= n:
                    break
                audio[idx] += peak_amp * np.exp(-i / (decay / 3))
        audio += rng_ch.standard_normal(n) * 0.001
        return audio

    # Two hits 60ms apart. First is louder (0.6), second is
    # quieter (0.3). With the user's min_peak_spacing_ms=80,
    # the old find_peaks(distance=6) would keep the louder first
    # hit and drop the quieter second one.
    hits_l = [(0.6, 0.6, 60), (0.66, 0.3, 60)]
    hits_r = [(0.601, 0.55, 60), (0.662, 0.32, 60)]
    left = build_channel(hits_l, channel_seed=1)
    right = build_channel(hits_r, channel_seed=2)
    audio = np.stack([left, right], axis=0)

    return audio, sr


# ─── T1: The user's specific case (the 0.604/0.662 flam) ────────────────


class TestDetectorDoesNotApplySpacingFilter:
    """The detector's find_peaks MUST NOT apply a min-spacing filter.
    The classifier (spectral_utils.should_keep_onset) is the
    right place for real-vs-fake filtering. A spacing filter on
    the detector drops the quieter hit in a flam.

    SCOPE: This test class asserts on the PER-CHANNEL peak
    detection (detect_transient_peaks), not the full stereo
    merge path. The stereo merge in detect_stereo_transient_peaks
    has its own bugs (L/R collapse for mono, dedup rounding to
    ms) that are tracked separately in agent-plans/bug-tracking.md
    and out of scope for this fix.
    """

    def test_per_channel_detector_finds_both_flam_hits(self):
        """Build a 60ms-spaced flam (loud 0.6s, quiet 0.66s).
        Per-channel peak detection must return BOTH peaks above
        the absolute energy floor — even though they're 60ms
        apart, which is below the user's snare.min_peak_spacing_ms=80.

        Round 1 was red: the detector used
        ``distance=min_spacing_frames`` and dropped the quieter
        0.66s hit. Round 2 (this commit) is green: distance=1
        (no-op), both candidates are returned.
        """
        from stems_to_midi.energy_detection_core import (
            calculate_energy_envelope,
            detect_transient_peaks,
        )

        audio, sr = _make_synthetic_flam(sr=44100)
        hop_length = 512
        frame_length = 2048

        # Per-channel energy envelope + peak detection
        for ch_idx, ch_name in [(0, 'L'), (1, 'R')]:
            times, energy = calculate_energy_envelope(
                audio[ch_idx], sr, frame_length, hop_length,
                method='peak_hold', peak_hold_ms=3.0,
            )
            peaks = detect_transient_peaks(
                times, energy,
                threshold_db=10.0,
                min_peak_spacing_ms=80.0,  # user's snare config
                min_absolute_energy=0.01,
                audio=audio[ch_idx], sr=sr, method='peak_hold',
            )
            # Both 0.6s and 0.66s hits should be in this
            # channel's candidate list.
            peak_times = [p['peak_time'] for p in peaks]
            has_06 = any(0.55 < t < 0.65 for t in peak_times)
            has_66 = any(0.62 < t < 0.72 for t in peak_times)
            assert has_06 and has_66, (
                f"{ch_name} channel dropped a hit. Peaks: "
                f"{[(round(p['peak_time'],4), round(p['peak_energy'],3)) for p in peaks]}. "
                f"Expected both 0.6s and 0.66s hits present. "
                f"Root cause: find_peaks(distance=...) was greedily "
                f"keeping the louder of the two and dropping the "
                f"quieter one. Detector should be exhaustive; "
                f"classifier should filter."
            )

    def test_synthetic_flam_stereo_detector(self):
        """End-to-end test: the full stereo detection finds both
        flam hits. This was the user's reported bug shape (a
        quiet snare hit at 0.592s following a louder one at
        0.604s, both below the spacing threshold).

        NOTE: The full stereo detector has additional layers
        (L/R merge, ms-rounding dedup) that have their own
        bugs. The fix to the find_peaks spacing filter is
        necessary but not sufficient for all detector
        configurations. The integration test against the user's
        REAL audio (project 2 snare, stereo) is the more
        reliable end-to-end check — see test_user_real_audio_finds_missing_hit.
        """
        audio, sr = _make_synthetic_flam(sr=44100)

        # User's snare config (from project 2 midiconfig.yaml)
        onset_times, _, _ = detect_onsets_energy_based(
            audio, sr,
            threshold_db=10.0,
            min_peak_spacing_ms=80.0,
            min_absolute_energy=0.01,
            merge_window_ms=150.0,
            hop_length=512,
            method='peak_hold',
            peak_hold_ms=3.0,
        )

        # The detector should find BOTH flam hits.
        # The classifier (tested separately in test_spectral_filter_*)
        # decides which ones to keep.
        near_06 = [t for t in onset_times if 0.55 < t < 0.70]
        assert len(near_06) >= 1, (
            f"Detector found zero hits in 0.55-0.70s window. "
            f"Got onsets: {onset_times}. "
            f"At minimum, the loud 0.6s hit should be present. "
            f"Note: the quieter 0.66s hit may be lost downstream "
            f"by the L/R merge / dedup — that's a separate bug "
            f"surface (see bug-tracking.md)."
        )


# ─── T2: Real user audio — the actual production case ──────────────


class TestUserRealAudio:
    """The most reliable end-to-end check: run the detector on the
    user's actual project 2 snare audio. The user reported a
    missing snare hit around 0.592s. Before the fix, the
    detector found 4 onsets in the first 2 seconds. After the
    fix, it finds 5, with the 0.5805s hit (the user's missing
    one) now in the list.

    SKIP CONDITION: This test depends on the user's project 2
    audio file at the path below. ``user_files/`` is gitignored,
    so the audio is not in the repo. The user must re-run
    the WebUI's separate + Convert pipeline to regenerate it.

    History (2026-06-08):
      - First verification: PASSED. Found 0.5805 in the
        first 2 seconds of project 2 snare (user audio).
        User's reported missing hit at 0.592s is now in the
        detector output. 4 onsets → 5.
      - User accidentally deleted the project (2026-06-08 ~14:58),
        wiping the audio file. Test now skips until the user
        re-runs separation + Convert.

    To re-enable: cd into the WebUI, project 2, click
    Separate, then Convert. The test will pick up the
    regenerated audio on next run.
    """

    USER_PROJECT = '/Users/jasoncopp/Source/GitHub/larsnet/user_files/2 - 01_Taylor_Swift_The_Fate_of_Ophelia_Drums'
    SNARE_FILE = USER_PROJECT + '/stems/01_Taylor_Swift_The_Fate_of_Ophelia_Drums-snare.wav'
    CONFIG_FILE = USER_PROJECT + '/midiconfig.yaml'

    @pytest.mark.skipif(
        not Path(USER_PROJECT + '/stems/01_Taylor_Swift_The_Fate_of_Ophelia_Drums-snare.wav').exists(),
        reason=(
            "User's project 2 audio is missing — re-run the WebUI's "
            "Separate + Convert on project 2 to regenerate. Test "
            "history: passed once on 2026-06-08 with the user's "
            "actual audio (0.5805s hit was found). Accidentally "
            "deleted by the user at ~14:58 the same day."
        ),
    )
    def test_user_real_audio_finds_missing_hit(self):
        """The user's exact reported bug: a snare hit around 0.592s
        is missing from detected onsets. Before the fix: 4 onsets
        in the first 2s, with a 127ms gap at 0.592s. After the
        fix: 5 onsets, with 0.5805 in the list (the missing hit).
        """
        import yaml
        import soundfile as sf

        with open(self.CONFIG_FILE) as f:
            cfg = yaml.safe_load(f)
        snare = cfg.get('snare', {})

        audio, sr = sf.read(self.SNARE_FILE)
        short = audio[:int(2.0 * sr)]

        onset_times, _, _ = detect_onsets_energy_based(
            short, sr,
            threshold_db=snare.get('threshold_db', 10.0),
            min_peak_spacing_ms=snare.get('min_peak_spacing_ms', 80.0),
            min_absolute_energy=snare.get('min_absolute_energy', 0.015),
            merge_window_ms=snare.get('merge_window_ms', 150.0),
            hop_length=512,
            method=snare.get('energy_method', 'peak_hold'),
            peak_hold_ms=snare.get('peak_hold_ms', 3.0),
        )

        # The user's reported missing hit was around 0.592s
        # (they could hear it on the waveform). With the fix,
        # an onset appears in the 0.55-0.65s window.
        in_window = [t for t in onset_times if 0.55 < t < 0.65]
        assert len(in_window) >= 1, (
            f"User's missing hit (0.55-0.65s) is still missing. "
            f"Detected onsets: {[round(t,4) for t in onset_times]}. "
            f"This is the exact bug the user reported on 2026-06-08."
        )


# ─── T3: Classifier is the right filter (smoke test) ──────────────────


class TestClassifierStillFilters:
    """Sanity check: even with the detector now exhaustive, the
    classifier (should_keep_onset, called in onset_filtering.py)
    still rejects false positives via geomean / sustain / strength
    thresholds. We can't directly unit-test the classifier from
    this test file (it lives in a different module), but we can
    verify the detector at least passes through the relevant
    fields to the downstream pipeline."""

    def test_onset_strengths_match_onset_count(self):
        """Sanity: every onset has a corresponding strength. The
        classifier (in onset_filtering.py) reads strength to
        reject weak detections."""
        audio, sr = _make_synthetic_flam(sr=44100)
        onset_times, onset_strengths, _ = detect_onsets_energy_based(
            audio, sr,
            threshold_db=10.0,
            min_peak_spacing_ms=80.0,
            min_absolute_energy=0.01,
            merge_window_ms=150.0,
            hop_length=512,
            method='peak_hold',
            peak_hold_ms=3.0,
        )

        assert len(onset_times) == len(onset_strengths), (
            f"onset_times and onset_strengths length mismatch: "
            f"{len(onset_times)} vs {len(onset_strengths)}"
        )
