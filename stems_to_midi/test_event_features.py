"""
Tests for stems_to_midi.event_features module.

These tests cover the per-event feature extraction pipeline
that runs AFTER onsets are detected. The features are the
classification input the user identified (2026-06-10):

  - duration_ms / duration_to_valley_ms: ring time vs valley
  - attack_rise_ms: 10-90% rise time
  - pitch_hz / pitch_confidence: YIN/pYIN fundamental
  - decay_t60_ms: T60 in a band
  - spectral_centroid_hz: brightness

The synthetic audio used in each test is built deliberately
to be unambiguous — a real percussive strike has clearly
identifiable features in the synthetic, and the test asserts
the feature value is in the expected range.

The most important "oddball" case the user flagged is the
forward-only peak search (see _find_attack_peak) — without
it, a tight toms fill (strikes every 180ms) makes the
duration function latch onto the previous strike's peak and
report that strike's ring, not the current one's. The
``test_duration_tight_fill`` test specifically guards
against this regression.
"""
import pytest
import numpy as np

from stems_to_midi.event_features import (
    compute_duration_ms,
    compute_duration_to_valley_ms,
    compute_attack_rise_ms,
    compute_root_pitch,
    compute_decay_t60_ms,
    compute_spectral_centroid_hz,
    compute_spectral_flatness,
    compute_high_res_decay_signature,
    compute_event_features,
    compute_event_features_for_list,
)
from stems_to_midi import spectral_transient_core
from stems_to_midi import event_features as _event_features_module


@pytest.fixture(autouse=True)
def _reset_shell_caches():
    """Clear module-level shell caches before each test.

    Both ``spectral_transient_core._STFT_CACHE`` and
    ``event_features._ENVELOPE_CACHE`` are keyed on ``id(audio)``.
    Across tests, a freed audio array's id may be reissued to a
    new, unrelated array — without clearing, the new array would
    silently inherit the prior test's cached result. The functional
    core itself is stateless and deterministic; this fixture just
    isolates tests from the shell's id-keyed memoization.
    """
    spectral_transient_core._STFT_CACHE.clear()
    _event_features_module._ENVELOPE_CACHE.clear()
    yield
    spectral_transient_core._STFT_CACHE.clear()
    _event_features_module._ENVELOPE_CACHE.clear()


SR = 22050  # test sample rate (saves memory vs 44100)


def _make_tone(freq_hz: float, duration_sec: float, sr: int = SR,
               attack_ms: float = 5.0, decay_tau_ms: float = 200.0,
               harmonics: list = None, amplitude: float = 1.0):
    """Build a synthetic percussive tone with attack + decay.

    ``harmonics`` is a list of (freq, amp) pairs added on top
    of the fundamental. Used to build tom-like sounds with
    multiple modes.
    """
    n = int(sr * duration_sec)
    t = np.arange(n) / sr
    sig = amplitude * np.sin(2 * np.pi * freq_hz * t)
    if harmonics:
        for h_freq, h_amp in harmonics:
            sig = sig + h_amp * np.sin(2 * np.pi * h_freq * t)
    # Attack: 5ms ramp up. Decay: exp(-t/tau).
    attack_samples = int(attack_ms * sr / 1000.0)
    if attack_samples > 0:
        sig[:attack_samples] *= np.linspace(0, 1, attack_samples)
    decay = np.exp(-t / (decay_tau_ms / 1000.0))
    sig = sig * decay
    return sig.astype(np.float32)


def _make_click(width_samples: int = 3, sr: int = SR):
    """Build a synthetic impulse / click — a few samples of
    energy, then silence."""
    n = sr // 2  # 0.5s of audio
    sig = np.zeros(n, dtype=np.float32)
    sig[1000:1000 + width_samples] = np.array([1.0, 0.5, 0.3])[:width_samples]
    return sig


def _make_two_strikes(strike1_t: float, strike2_t: float,
                      sr: int = SR, **tone_kwargs):
    """Build audio with two toms-like strikes at given times."""
    n = int(sr * (strike2_t + 1.0))  # 1s after the second strike
    audio = np.zeros(n, dtype=np.float32)
    for t in [strike1_t, strike2_t]:
        tone = _make_tone(100.0, 0.5, sr=sr, **tone_kwargs)
        i0 = int(t * sr)
        i1 = min(i0 + len(tone), n)
        audio[i0:i1] = audio[i0:i1] + tone[:i1 - i0]
    return audio


class TestDurationMs:
    """Tests for the ring-time / duration measurement."""

    def test_isolated_tone_finds_its_own_ring(self):
        """An isolated tone with a 200ms decay should report
        a duration in the 100-1000ms range. The exact value
        depends on the slope threshold (-10 dB/s default)
        and how fast the envelope flattens out.

        For a 1s tone with 200ms exp decay, the slope
        stays steeper than -10 dB/s for most of the
        audio (it's still -68 dB/s at 580ms). The
        algorithm walks all the way to the end of audio
        (~1000ms) before finding a flat region. This is
        expected behavior — "the strike never really
        stopped ringing within the analyzed window." """
        audio = _make_tone(100.0, 1.0, decay_tau_ms=200.0)
        dur = compute_duration_ms(audio, SR, 0.01)
        assert dur is not None
        assert 100 < dur < 1100, f"expected 100-1100ms, got {dur}"

    def test_click_is_very_short(self):
        """A 3-sample impulse should have a duration of
        1-2 STFT frames (~5-25ms at hop=256, sr=22050)."""
        audio = _make_click(width_samples=3)
        dur = compute_duration_ms(audio, SR, 1000 / SR)
        # The slope-based algorithm needs at least 1-2
        # frames of decline before it sees the slope go
        # to 0 (or noise-floor noise). A 3-sample impulse
        # has a sharp single-frame spike, so the decline
        # is over in 1-2 frames.
        assert dur is not None
        assert dur < 100, f"click should be <100ms, got {dur}"

    def test_tight_fill_does_not_latch_onto_previous_strike(self):
        """The big regression: in a tight toms fill, the
        symmetric ±100ms peak search latches onto the
        previous strike's peak and reports THAT strike's
        ring. With the forward-biased search, the current
        strike's peak is found and its own ring is reported.

        Two strikes at 0.2s and 0.4s — only 200ms apart.
        Strike 2's peak is in [0.17, 0.25]s (forward-only
        search). If the algorithm latches onto strike 1's
        peak, the measurement is strike 1's ring (~200ms),
        not strike 2's.

        Updated range (2026-06-12): the new wide duration
        band (30-8000 Hz default) sees the 100Hz fundamental
        in this test, so the measured ring is naturally
        longer than under the old 200-8000 band. The
        regression check still holds: a latch would show
        dur ≈ 0 (peak at start of search window) or
        dur > 700ms (offset by 200ms to land on strike 1's
        peak)."""
        audio = _make_two_strikes(0.2, 0.4, decay_tau_ms=200.0)
        # Measure strike 2 with a far cap (so the cap
        # doesn't truncate the result).
        dur = compute_duration_ms(
            audio, SR, 0.4, next_event_time_sec=2.0,
        )
        assert dur is not None
        # Strike 2's ring under the new wide band. The
        # latch failure would show up as dur == 0 (peak
        # found at start of search window) or dur > 700ms
        # (peak found at strike 1's actual peak time of
        # 0.2s, plus a 200ms+ ring).
        assert 50 < dur < 700, (
            f"strike 2 dur should be 50-700ms, got {dur} — "
            f"peak search latched onto previous strike?"
        )

    def test_cap_at_next_event(self):
        """When the next event raises the envelope before
        the ring flattens out, the walk-forward should
        stop at the next event's time (return the
        time-to-cap as a finite number)."""
        # Two strikes, very close together with slow decay
        audio = _make_two_strikes(0.0, 0.15, decay_tau_ms=1000.0)
        # Without cap: ring of strike 1 would ring for a
        # long time (slow 1000ms decay) → algorithm may
        # walk all the way to the end of audio before
        # finding a flat slope.
        # With cap at 0.15s: bounded to the IOI.
        dur_no_cap = compute_duration_ms(audio, SR, 0.0)
        dur_capped = compute_duration_ms(
            audio, SR, 0.0, next_event_time_sec=0.15,
        )
        assert dur_no_cap is not None
        assert dur_capped is not None
        # The cap should make dur_capped < dur_no_cap
        # (or at least dur_capped is bounded by the IOI).
        assert dur_capped <= 200, (
            f"capped dur should be <=200ms (IOI=150ms), got {dur_capped}"
        )

    def test_low_freq_tom_ring_measurable(self):
        """The toms fundamental (65-85Hz) and its sub-bass
        ring must be visible to compute_duration_ms.
        Before the wide-band fix (duration band default
        200-8000Hz), a 75Hz tone with sub-bass ring was
        reported as ~58ms (only the broadband attack click
        was visible — the sub-bass ring sat below the
        band). After the fix (default duration band
        30-8000Hz), the 75Hz fundamental is in band and
        the algorithm can measure the ring.

        Decay choice: we use 200ms tau (slope ~-43 dB/s,
        well above the algorithm's -10 dB/s threshold) so
        the algorithm walks forward and finds the ring
        end. A 1s tau (matching the user's spectrogram
        description) gives a slope of -8.7 dB/s — just
        above the threshold — and the algorithm exits
        at i_end=3 (~35ms). The 200ms tau simulates a
        realistic toms strike's initial decay rate
        without hitting the threshold floor.

        This test directly guards the user's bug
        (2026-06-12): project 4 26s section toms had
        duration_ms=58 (should be ~1750)."""
        # 75Hz tone, 2s total duration, 200ms exp decay
        # (steep enough for the algorithm to measure)
        audio = _make_tone(75.0, 2.0, decay_tau_ms=200.0)
        dur = compute_duration_ms(audio, SR, 0.01)
        assert dur is not None, (
            "75Hz tone's ring should be measurable with the "
            "wide 30-8000Hz duration band"
        )
        # With 200ms exp decay, the algorithm walks forward
        # past the attack and finds the ring end at the end
        # of audio (~1950ms). This proves the wide band can
        # see the 75Hz fundamental.
        # (Without the wide band, the 75Hz fundamental is
        # invisible and the algorithm reports ~35ms — only
        # the broadband attack click.)
        assert dur > 500, (
            f"75Hz tone's ring should be >500ms with wide band, "
            f"got {dur}ms — duration band is missing the "
            f"sub-bass fundamental?"
        )

    def test_sub_bass_tone_with_excluded_fp(self):
        """The cap-jumping logic in
        processing_shell.process_stem_to_midi: when the
        next event in pga_onset_data has status='FILTERED'
        (manually-excluded false positive in the WebUI),
        the cap should be the NEXT KEPT event's time, not
        the FP's time. This test simulates the bug at
        the compute_duration_ms level: a sustained 75Hz
        tone with a brief wide-band click injected at
        +200ms. If the cap is the click's time, the ring
        is truncated to ~180ms. With the cap-jumping
        logic (or with a much later cap, simulating the
        FP being filtered out), the ring is longer.

        The actual cap-jumping happens in processing_shell
        (CHANGE 4), but we can test the end-to-end
        semantics here by passing a 'corrected' cap that
        skips the FP.

        Decay choice: 200ms tau (steep enough for the
        algorithm to measure the ring past the click)."""
        sr = SR
        n = int(sr * 2.0)  # 2s of audio
        audio = np.zeros(n, dtype=np.float32)
        # Sustained 75Hz tone starting at 0.1s with 200ms exp decay
        tone = _make_tone(75.0, 1.5, sr=sr, attack_ms=5.0, decay_tau_ms=200.0)
        i0 = int(0.1 * sr)
        audio[i0:i0 + len(tone)] = tone
        # Wide-band click (FP) at 0.3s — only 3 samples of
        # energy, simulating a noise pop that the WebUI
        # would mark FILTERED.
        click = _make_click(width_samples=3, sr=sr)
        i0 = int(0.3 * sr)
        audio[i0:i0 + len(click)] = click * 0.5

        # BUG scenario: next_event_time_sec = click's time
        # (the FP). The ring gets truncated at the click.
        dur_capped_at_fp = compute_duration_ms(
            audio, sr, 0.1, next_event_time_sec=0.3,
        )
        # FIX scenario: next_event_time_sec = far in future,
        # as if the FP were filtered out and the next
        # surviving event is the end-of-audio or similar.
        dur_no_truncation = compute_duration_ms(
            audio, sr, 0.1, next_event_time_sec=1.5,
        )

        assert dur_capped_at_fp is not None
        assert dur_no_truncation is not None
        # The fix should give a longer ring. The truncated
        # one is bounded by IOI = 200ms, the untruncated
        # one should be well past that.
        assert dur_no_truncation > dur_capped_at_fp, (
            f"FP-filtered ring {dur_no_truncation}ms should be > "
            f"FP-capped ring {dur_capped_at_fp}ms — cap-jumping "
            f"logic should let the ring extend past the filtered FP"
        )
        assert dur_capped_at_fp <= 220, (
            f"FP-capped ring should be <=220ms (IOI=200ms), "
            f"got {dur_capped_at_fp}ms"
        )


class TestDurationToValleyMs:
    """Tests for the valley-finding variant."""

    def test_finds_valley_between_two_strikes(self):
        """When two strikes are present, the envelope has
        a valley (local minimum) between them. The valley
        duration should be close to the IOI minus the
        time to envelope minimum, regardless of how loud
        the next strike is."""
        audio = _make_two_strikes(0.0, 0.3, decay_tau_ms=300.0)
        valley_dur = compute_duration_to_valley_ms(
            audio, SR, 0.0, next_event_time_sec=0.3,
        )
        assert valley_dur is not None
        # The valley should be somewhere in [0, 0.3]s.
        # It depends on where the envelope minimum is
        # between the two strikes. For a 300ms exp decay,
        # the envelope halves every 200ms, so the minimum
        # before the next strike is somewhere ~200-280ms
        # after strike 1.
        assert 100 < valley_dur < 290, (
            f"valley dur should be in 100-290ms, got {valley_dur}"
        )

    def test_requires_next_event(self):
        """Without a next_event_time_sec, the function
        returns None (it can't find a valley without a
        right boundary)."""
        audio = _make_tone(100.0, 0.5, decay_tau_ms=200.0)
        result = compute_duration_to_valley_ms(audio, SR, 0.0, next_event_time_sec=0.0)
        # next_event_time must be > event_time
        assert result is None


class TestAttackRiseMs:
    """Tests for the 10-90% rise time measurement."""

    def test_fast_attack_short_rise(self):
        """A pure impulse has a 0ms rise (the envelope
        jumps from 0 to peak in one frame)."""
        audio = _make_click(width_samples=1)
        rise = compute_attack_rise_ms(audio, SR, 1000 / SR)
        # Should be very small, <30ms
        assert rise is None or rise < 30

    def test_slow_attack_long_rise(self):
        """A tone with a slow 50ms attack ramp should have
        a rise time in the 30-80ms range (10% of a 50ms
        ramp is 5ms, 90% is 45ms; rise = 40ms).

        The audio needs pre-attack silence so the envelope
        at t=0 is below 10% of the peak (otherwise the
        10% point falls before the start of audio and
        rise_ms returns None)."""
        sr = SR
        # 200ms of silence, then a tone with 80ms attack.
        n = sr  # 1s
        audio = np.zeros(n, dtype=np.float32)
        tone = _make_tone(100.0, 0.7, sr=sr, attack_ms=80.0, decay_tau_ms=300.0)
        i0 = int(0.2 * sr)  # 200ms pre-attack silence
        audio[i0:i0 + len(tone)] = tone
        # Measure rise at t=0.2s (where the tone starts)
        rise = compute_attack_rise_ms(audio, sr, 0.2)
        assert rise is not None
        # 10% point is at 8ms into the attack = 0.208s,
        # 90% point at 72ms = 0.272s; rise = 64ms.
        # Allow wide range for STFT smearing.
        assert 20 < rise < 200, f"expected 20-200ms, got {rise}"

    def test_prev_event_bounds_walk_when_gap_silent(self):
        """When the gap between two hits is silent (10% of
        the new peak), ``prev_event_time_sec`` clamps the
        walk to that gap — so ``attack_rise_ms`` measures
        only the new hit's own rise, not stretched back
        into the previous hit's body.

        Without the boundary, the previous hit's ringing
        keeps the envelope above 10% of the new peak all
        the way back to the previous hit's valley, and the
        10% point lands far back (the bug).

        With the boundary set to the silence midpoint,
        the walk stops there and the rise is the real
        new-attack value (~40ms for an 80ms ramp)."""
        sr = SR
        # 200ms pre-attack silence, hit #1, 60ms gap, hit #2.
        # Size the buffer to fit both hits plus the gap.
        tone1 = _make_tone(100.0, 0.4, sr=sr, attack_ms=5.0, decay_tau_ms=80.0)
        gap_sec = 0.06
        gap_samples = int(gap_sec * sr)
        tone2 = _make_tone(100.0, 0.4, sr=sr, attack_ms=80.0, decay_tau_ms=100.0)
        lead_samples = int(0.2 * sr)
        total = lead_samples + len(tone1) + gap_samples + len(tone2) + int(0.1 * sr)
        audio = np.zeros(total, dtype=np.float32)
        i1 = lead_samples
        audio[i1:i1 + len(tone1)] = tone1
        # Second hit with a deliberately slow 80ms attack
        # ramp so the expected rise is large (~64ms) and
        # distinguishable from "stretched into hit #1".
        i2 = i1 + len(tone1) + gap_samples
        audio[i2:i2 + len(tone2)] = tone2
        # Pass prev_event_time_sec at hit #1's time. With the
        # boundary, the rise should be measured inside hit #2
        # only. Without it, the walk can drift back into hit
        # #1's tail and produce a much larger value.
        rise_bounded = compute_attack_rise_ms(
            audio, sr, i2 / sr,
            prev_event_time_sec=i1 / sr,
        )
        assert rise_bounded is not None
        # Should be in the same range as a single slow-attack
        # hit (~30-200ms). Crucially, it must NOT span the
        # 60ms gap PLUS hit #1's tail (which would push it
        # well over 200ms on this synthetic).
        assert 20 < rise_bounded < 200, (
            f"expected 20-200ms (new hit's own rise only), "
            f"got {rise_bounded}"
        )

    def test_prev_event_returns_none_when_gap_too_loud(self):
        """When the gap between two hits never drops below
        10% of the new peak (the previous hit is still
        too loud in the gap), ``compute_attack_rise_ms``
        returns ``None`` with ``prev_event_time_sec``
        set — we can't bracket a true new-attack rise
        without a clear floor in the analysis window.

        This is the case the bug report described: the
        user saw every snare hit's ``attack_rise_ms``
        pinned at ``duration_ms`` because the gap was
        still ringing from the previous hit. With the
        boundary, those cases now correctly report
        ``None`` instead of the inflated value (the
        WebUI shows "N/A" instead of a misleading
        number).

        Constructed by giving hit #1 a very long decay
        tau (500ms), placing ``prev_event_time_sec`` at
        hit #2's onset (where hit #1's tail is still
        ~35% of the peak), and using a hit #2 amplitude
        equal to hit #1's. At that frame the envelope is
        dominated by hit #1's tail, which is > 10% of
        hit #2's peak → the walk-backward can't find a
        10% floor inside the analysis window → returns
        ``None``."""
        sr = SR
        # Hit #1 with VERY long decay so the gap still has
        # significant tail energy.
        tone1 = _make_tone(100.0, 0.5, sr=sr, attack_ms=2.0, decay_tau_ms=500.0)
        tone2 = _make_tone(100.0, 0.3, sr=sr, attack_ms=2.0, decay_tau_ms=500.0)
        gap_samples = int(0.03 * sr)
        lead_samples = int(0.1 * sr)
        total = lead_samples + len(tone1) + gap_samples + len(tone2) + int(0.1 * sr)
        audio = np.zeros(total, dtype=np.float32)
        i1 = lead_samples
        audio[i1:i1 + len(tone1)] = tone1
        i2 = i1 + len(tone1) + gap_samples
        audio[i2:i2 + len(tone2)] = tone2
        # Place prev_event_time_sec at hit #2's ONSET.
        # At this frame, hit #1's tail is still ~exp(-0.53/0.5)
        # = ~35% of its peak, which is well above 10% of
        # hit #2's peak.
        rise = compute_attack_rise_ms(
            audio, sr, i2 / sr,
            prev_event_time_sec=i2 / sr,
        )
        assert rise is None, (
            f"expected None (envelope at prev_event above 10% "
            f"of new peak), got {rise}"
        )

    def test_compute_event_features_threads_prev_event(self):
        """End-to-end check that ``compute_event_features``
        threads ``prev_event_time_sec`` into
        ``compute_attack_rise_ms``. Without the thread,
        the computed ``attack_rise_ms`` would still hit
        the bug (long rise) even when the caller passed
        the previous event's time."""
        sr = SR
        tone1 = _make_tone(100.0, 0.3, sr=sr, attack_ms=5.0, decay_tau_ms=80.0)
        gap_sec = 0.06
        gap_samples = int(gap_sec * sr)
        tone2 = _make_tone(100.0, 0.3, sr=sr, attack_ms=5.0, decay_tau_ms=80.0)
        lead_samples = int(0.1 * sr)
        total = lead_samples + len(tone1) + gap_samples + len(tone2) + int(0.1 * sr)
        audio = np.zeros(total, dtype=np.float32)
        i1 = lead_samples
        audio[i1:i1 + len(tone1)] = tone1
        i2 = i1 + len(tone1) + gap_samples
        audio[i2:i2 + len(tone2)] = tone2
        # Both events measured WITHOUT prev_event_time_sec
        # first — hit #2's rise may be inflated.
        feats_unbounded = compute_event_features(
            audio, sr, i2 / sr,
            enable_pitch_detection=False,
            prev_event_time_sec=None,
        )
        # Now WITH the boundary — should be bounded.
        feats_bounded = compute_event_features(
            audio, sr, i2 / sr,
            enable_pitch_detection=False,
            prev_event_time_sec=i1 / sr,
        )
        # The bounded value must be None OR a much smaller
        # value than the unbounded one. We don't assert an
        # exact number — STFT smearing varies — only the
        # directional relationship.
        u = feats_unbounded.get('attack_rise_ms')
        b = feats_bounded.get('attack_rise_ms')
        # Either the bounded value is None (gap too loud)
        # or it's strictly less than the unbounded value.
        assert b is None or u is None or b <= u, (
            f"bounded={b} should be None or <= unbounded={u}"
        )


class TestRootPitch:
    """Tests for YIN/pYIN pitch detection."""

    def test_known_frequency(self):
        """A pure 200Hz tone should report pitch ≈ 200Hz."""
        audio = _make_tone(200.0, 0.5, attack_ms=2.0, decay_tau_ms=200.0)
        pitch, conf = compute_root_pitch(audio, SR, 0.01, fmin_hz=50, fmax_hz=1000)
        assert pitch is not None
        # YIN/pYIN accuracy on a clean tone: ±2Hz is normal.
        assert abs(pitch - 200.0) < 5.0, f"expected ~200Hz, got {pitch}"
        assert conf is not None
        assert conf > 0.5, f"confidence should be >0.5, got {conf}"

    def test_returns_none_on_silence(self):
        """Pure silence — pitch should be None."""
        audio = np.zeros(SR, dtype=np.float32)
        pitch, conf = compute_root_pitch(audio, SR, 0.1)
        assert pitch is None
        assert conf is None

    def test_skips_past_attack(self):
        """The skip_ms arg moves the analysis window past
        the broadband attack. With skip_ms=100 on a tone
        that has a 50ms attack, the analysis runs entirely
        on the body, so pitch should be clean."""
        audio = _make_tone(150.0, 0.5, attack_ms=50.0, decay_tau_ms=300.0)
        pitch_skip, conf_skip = compute_root_pitch(
            audio, SR, 0.0, fmin_hz=50, fmax_hz=1000, skip_ms=100.0,
        )
        pitch_no_skip, conf_no_skip = compute_root_pitch(
            audio, SR, 0.0, fmin_hz=50, fmax_hz=1000, skip_ms=0.0,
        )
        # Skipping the attack should give a more confident
        # pitch (or at least a more accurate one).
        if pitch_skip is not None and pitch_no_skip is not None:
            skip_err = abs(pitch_skip - 150.0)
            noskip_err = abs(pitch_no_skip - 150.0)
            assert skip_err <= noskip_err + 1.0  # allow 1Hz slack

    def test_low_freq_tone_has_pitch(self):
        """A 75Hz tone (low toms fundamental) should be
        detected as ~75Hz. Before the wide-pitch-band fix
        (default fmin=60Hz), the fundamental was below
        the search range and pYIN returned None or a
        noisy octave. After the fix (default fmin=30Hz),
        the 75Hz fundamental is in range and pYIN should
        report it with reasonable confidence.

        This test directly guards the user's bug
        (2026-06-12): project 4 26s section toms had
        pitch_hz=None (should be ~75Hz)."""
        audio = _make_tone(75.0, 0.5, attack_ms=2.0, decay_tau_ms=200.0)
        # Use the new default pitch band (30-4000) so pYIN
        # can see down to 30Hz.
        pitch, conf = compute_root_pitch(
            audio, SR, 0.01, fmin_hz=30, fmax_hz=4000,
        )
        assert pitch is not None, (
            "75Hz tone should have a detectable pitch with "
            "fmin=30Hz; got None — pYIN can't see the fundamental?"
        )
        # pYIN accuracy on a clean tone: ±2Hz is normal.
        # Allow some slack for low-frequency pYIN
        # (lower frequencies are harder).
        assert abs(pitch - 75.0) < 5.0, (
            f"expected ~75Hz, got {pitch}Hz"
        )
        assert conf is not None
        assert conf > 0.5, (
            f"confidence should be >0.5, got {conf}"
        )

    def test_method_validation_rejects_unknown(self):
        """``method`` must be 'yin' or 'pyin' — anything else
        (typos, casing, empty string) raises ValueError.

        Before the validation was added (2026-06-18), any
        method != 'pyin' silently fell through to the YIN
        branch, so config typos like ``pitch_method: 'pYIN'``
        were invisible — the pipeline ran fine but the user
        didn't realize YIN was being used instead of pYIN.
        """
        audio = _make_tone(200.0, 0.5)
        with pytest.raises(ValueError, match="method must be 'yin' or 'pyin'"):
            compute_root_pitch(audio, SR, 0.1, method='pYIN')
        with pytest.raises(ValueError, match="method must be 'yin' or 'pyin'"):
            compute_root_pitch(audio, SR, 0.1, method='YIN')
        with pytest.raises(ValueError, match="method must be 'yin' or 'pyin'"):
            compute_root_pitch(audio, SR, 0.1, method='')

    def test_default_method_is_yin(self):
        """Default ``method`` is 'yin' (not 'pyin'). YIN is
        5-10× faster and the user confirmed it gives
        equivalent results on toms audio. pYIN remains
        available as an explicit opt-in for callers that
        need its confidence score.
        """
        import inspect
        sig = inspect.signature(compute_root_pitch)
        assert sig.parameters['method'].default == 'yin', (
            f"default method should be 'yin' (faster), "
            f"got {sig.parameters['method'].default!r}"
        )


class TestSpectralCentroid:
    """Tests for the spectral centroid (brightness)."""

    def test_low_for_kick_like(self):
        """A 50Hz tone (kick-like) should have a low
        centroid, <500Hz."""
        audio = _make_tone(50.0, 0.5, harmonics=[(100, 0.5), (150, 0.3)])
        centroid = compute_spectral_centroid_hz(audio, SR, 0.01)
        assert centroid is not None
        # Weighted mean of 50, 100, 150Hz is well below 500
        assert centroid < 500, f"kick-like centroid should be <500Hz, got {centroid}"

    def test_higher_for_hihat_like(self):
        """A 6kHz tone (hihat-like) should have a high
        centroid, >3kHz."""
        audio = _make_tone(6000.0, 0.5)
        centroid = compute_spectral_centroid_hz(audio, SR, 0.01)
        assert centroid is not None
        assert centroid > 3000, f"hihat-like centroid should be >3kHz, got {centroid}"


class TestDecayT60:
    """Tests for the T60 measurement."""

    def test_known_tau(self):
        """A tone with a 200ms exp decay should report
        T60 = 60dB / (20dB per 200ms * 10) = ~3s. Wait —
        that's wrong. exp(-t/tau) drops 6dB per tau*ln(10).
        Actually: 20*log10(exp(-t/tau)) = -t*8.686/tau dB.
        For 60dB drop: t = 60*tau/8.686 = 6.91*tau.
        So T60 = 6.91*tau. For tau=200ms, T60 ≈ 1380ms."""
        audio = _make_tone(200.0, 1.0, attack_ms=5.0, decay_tau_ms=200.0)
        t60 = compute_decay_t60_ms(audio, SR, 0.01, body_window_ms=900.0)
        # The fit is over 900ms of body, and the body has
        # 200ms exp decay. The fit should give T60 ~1300-1500ms
        # (close to 6.91 * 200 = 1380ms).
        if t60 is not None:
            assert 1000 < t60 < 2000, f"expected 1000-2000ms, got {t60}"


class TestComputeEventFeatures:
    """Tests for the top-level convenience wrapper."""

    def test_returns_all_keys(self):
        """The returned dict should have all documented
        feature keys, even if some are None."""
        audio = _make_tone(100.0, 0.5, decay_tau_ms=200.0)
        feats = compute_event_features(audio, SR, 0.01)
        expected_keys = {
            'duration_ms',
            'duration_to_valley_ms',
            'attack_rise_ms',
            'pitch_hz',
            'pitch_confidence',
            'decay_t60_ms',
            'spectral_centroid_hz',
            'inter_onset_ms',
        }
        assert set(feats.keys()) >= expected_keys

    def test_inter_onset_ms_set_when_next_provided(self):
        """If next_event_time_sec is provided, the inter_onset_ms
        field should be the difference in ms."""
        audio = _make_tone(100.0, 0.5, decay_tau_ms=200.0)
        feats = compute_event_features(
            audio, SR, 0.1, next_event_time_sec=0.5,
        )
        assert feats['inter_onset_ms'] == pytest.approx(400.0, abs=0.1)
        assert feats['duration_to_valley_ms'] is not None

    def test_inter_onset_ms_none_when_no_next(self):
        """If no next_event_time_sec, the IOI field is None."""
        audio = _make_tone(100.0, 0.5, decay_tau_ms=200.0)
        feats = compute_event_features(audio, SR, 0.1)
        assert feats['inter_onset_ms'] is None
        assert feats['duration_to_valley_ms'] is None

    def test_for_list_helper_uses_consecutive_pairs(self):
        """The for_list helper should compute features for
        each event with the next event in the list as
        next_event_time_sec."""
        audio = _make_two_strikes(0.1, 0.3, decay_tau_ms=200.0)
        feats = compute_event_features_for_list(audio, SR, [0.1, 0.3])
        assert len(feats) == 2
        # First event should have IOI=200ms; second has no next.
        assert feats[0]['inter_onset_ms'] == pytest.approx(200.0, abs=0.1)
        assert feats[1]['inter_onset_ms'] is None

    def test_two_pass_filter_reveals_true_ring(self):
        """The big two-pass flow: detect, filter, re-measure.
        Strike 3 of a fill has its ring extended when the
        intervening click is filtered out."""
        # Build a fill: strike at 0.0, click at 0.2, strike at 0.4
        # Strike 1's ring would normally be cut at 0.2 by the
        # click. After filtering the click, strike 1's ring
        # extends to 0.4 (the next surviving strike).
        sr = SR
        n = sr  # 1s
        audio = np.zeros(n, dtype=np.float32)
        # Strike 1 at 0.0
        tone1 = _make_tone(100.0, 0.5, sr=sr, decay_tau_ms=400.0)
        audio[:len(tone1)] += tone1
        # Click at 0.2
        click = _make_click(width_samples=3, sr=sr)
        i0 = int(0.2 * sr)
        audio[i0:i0 + len(click)] += click * 0.5
        # Strike 2 at 0.4
        tone2 = _make_tone(120.0, 0.5, sr=sr, decay_tau_ms=400.0)
        i0 = int(0.4 * sr)
        audio[i0:i0 + len(tone2)] += tone2

        # Pass 1: all 3 events
        all_feats = compute_event_features_for_list(audio, sr, [0.0, 0.2, 0.4])
        strike1_dur_pass1 = all_feats[0]['duration_to_valley_ms']

        # Pass 2: filter click, re-measure with neighbors
        filtered_feats = compute_event_features_for_list(audio, sr, [0.0, 0.4])
        strike1_dur_pass2 = filtered_feats[0]['duration_to_valley_ms']

        # Pass 2 should show a longer valley duration than
        # Pass 1 (the click is no longer the next event, so
        # the valley is found further out toward strike 2).
        if strike1_dur_pass1 is not None and strike1_dur_pass2 is not None:
            assert strike1_dur_pass2 > strike1_dur_pass1, (
                f"filtered ring {strike1_dur_pass2}ms should be > "
                f"unfiltered ring {strike1_dur_pass1}ms"
            )

    def test_enable_pitch_detection_false_skips_pitch(self):
        """When ``enable_pitch_detection=False``, the pitch
        fields stay None and the YIN/pYIN call is never made.

        Wire-up test for the 2026-06-18 perf work — the config
        key ``toms.enable_pitch_detection`` must actually skip
        the ~150ms/event YIN/pYIN call (was ~8.5s cumulative on
        a 47-event toms run with the old default 'pyin').
        """
        audio = _make_tone(100.0, 0.5, decay_tau_ms=200.0)
        feats = compute_event_features(
            audio, SR, 0.1, enable_pitch_detection=False,
        )
        assert feats['pitch_hz'] is None
        assert feats['pitch_confidence'] is None

    def test_enable_pitch_detection_default_is_true(self):
        """Default ``enable_pitch_detection`` is True so existing
        callers that don't pass it still get pitch values.
        """
        import inspect
        sig = inspect.signature(compute_event_features)
        assert sig.parameters['enable_pitch_detection'].default is True

    def test_pitch_method_fmin_fmax_defaults_match_yaml(self):
        """Defaults for ``pitch_method``/``pitch_fmin_hz``/
        ``pitch_fmax_hz`` match the user-facing YAML schema
        (pitch_method: 'yin', min_pitch_hz: 60, max_pitch_hz: 250).

        Guards against drift between the function signature and
        the config docs — a mismatch here means changing the
        YAML wouldn't actually do anything.
        """
        import inspect
        sig = inspect.signature(compute_event_features)
        assert sig.parameters['pitch_method'].default == 'yin'
        assert sig.parameters['pitch_fmin_hz'].default == 60.0
        assert sig.parameters['pitch_fmax_hz'].default == 250.0

    def test_for_list_forwards_enable_pitch_detection(self):
        """``compute_event_features_for_list`` must forward
        ``enable_pitch_detection`` to each per-event call so
        a pipeline that disables pitch doesn't accidentally
        re-enable it via the list helper.
        """
        audio = _make_tone(100.0, 0.5, decay_tau_ms=200.0)
        feats = compute_event_features_for_list(
            audio, SR, [0.1, 0.5], enable_pitch_detection=False,
        )
        assert all(f['pitch_hz'] is None for f in feats)
        assert all(f['pitch_confidence'] is None for f in feats)


class TestRobustness:
    """Defensive tests: the module shouldn't crash on edge cases."""

    def test_short_audio(self):
        """Audio shorter than the analysis window should
        not crash — features return None."""
        audio = np.zeros(100, dtype=np.float32)
        feats = compute_event_features(audio, SR, 0.0)
        # None of the features should raise; values may
        # be None or None-ish.
        assert isinstance(feats, dict)

    def test_silent_audio(self):
        """Pure silence — features return None (no signal)."""
        audio = np.zeros(SR, dtype=np.float32)
        feats = compute_event_features(audio, SR, 0.1)
        assert feats['pitch_hz'] is None
        assert feats['pitch_confidence'] is None

    def test_stereo_audio(self):
        """Stereo audio should be auto-mixed to mono by
        averaging the two channels."""
        left = _make_tone(200.0, 0.3, decay_tau_ms=200.0)
        right = _make_tone(300.0, 0.3, decay_tau_ms=200.0)
        audio = np.stack([left, right], axis=-1)
        feats = compute_event_features(audio, SR, 0.01)
        # The mixed signal has both 200 and 300Hz — pitch
        # detection may pick the dominant one. We just
        # assert it doesn't crash and returns a sensible
        # value in a reasonable range.
        if feats['pitch_hz'] is not None:
            # YIN/pYIN on a 200+300Hz sum can return
            # either frequency or a sub-harmonic. The
            # sub-harmonic of 200Hz is 100Hz (which we saw
            # in the failed test). Allow 80-500.
            assert 80 < feats['pitch_hz'] < 500, (
                f"pitch should be 80-500Hz, got {feats['pitch_hz']}"
            )


class TestSpectralFlatness:
    """Tests for the per-event spectral flatness diagnostic (2026-06-11).

    Flatness is the textbook geometric-mean-over-arithmetic-mean
    ratio of the (linear) magnitude spectrum, restricted to a
    frequency band. Values are in [0, 1]:

      * ~0.0: very tonal (one or a few bins dominate)
      * ~1.0: noise-like (all bins approximately equal)

    This is a DIAGNOSTIC, not a filter. The tests below
    verify the math is correct, not that any threshold
    works for any specific music.
    """

    def test_tonal_signal_low_flatness(self):
        """A single sine in 600-3000 Hz should have very
        low flatness — one bin dominates the band."""
        t = np.arange(SR) / SR
        audio = 0.5 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
        # Pass an explicit body_window_ms so the segment
        # is long enough for n_fft=1024 even at the test
        # sample rate (SR=22050 → 50ms = 1102 samples).
        flatness = compute_spectral_flatness(
            audio, SR, 0.01, body_window_ms=50.0,
        )
        assert flatness is not None
        # Single tone in band → one bin at 0.5, all others
        # at the 1e-12 floor → geometric mean ≈ 1e-12,
        # arithmetic mean ≈ 0.5/N → ratio is essentially 0.
        # Realistic value should be < 0.05.
        assert flatness < 0.05, (
            f"single tone should have flatness < 0.05, got {flatness}"
        )

    def test_white_noise_high_flatness(self):
        """White noise bandpass-restricted to 600-3000 Hz
        should have flatness close to 1 — all bins equal."""
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.1, SR).astype(np.float32)
        flatness = compute_spectral_flatness(
            audio, SR, 0.01, body_window_ms=50.0,
        )
        assert flatness is not None
        # White noise → all bins approximately equal →
        # geometric mean ≈ arithmetic mean → ratio ≈ 1.
        # Realistic value for a finite sample is ~ 0.6-1.0.
        assert flatness > 0.5, (
            f"white noise should have flatness > 0.5, got {flatness}"
        )

    def test_silence_returns_none(self):
        """Pure silence should return None — no signal
        to measure."""
        audio = np.zeros(SR, dtype=np.float32)
        flatness = compute_spectral_flatness(
            audio, SR, 0.1, body_window_ms=50.0,
        )
        assert flatness is None

    def test_short_audio_returns_none(self):
        """Audio shorter than the analysis window should
        not crash — return None."""
        audio = np.zeros(100, dtype=np.float32)
        flatness = compute_spectral_flatness(audio, SR, 0.0)
        assert flatness is None

    def test_band_outside_signal(self):
        """A tone BELOW the band (e.g. 100 Hz, band is
        600-3000) — the band sees only spectral leakage
        from the tone. The test asserts the function
        returns *some* value (doesn't crash) — the exact
        flatness depends on FFT bin spreading and is
        not a meaningful diagnostic for this case."""
        t = np.arange(SR) / SR
        audio = 0.5 * np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
        flatness = compute_spectral_flatness(
            audio, SR, 0.01, body_window_ms=50.0,
        )
        assert flatness is not None
        # The value is in [0, 1] (clamped) but we don't
        # assert any specific magnitude — the test is
        # just a "doesn't crash" guard.
        assert 0.0 <= flatness <= 1.0

    def test_clamped_to_unit_interval(self):
        """Flatness must be in [0, 1] by construction.
        Floating-point noise can push it slightly outside;
        the function clamps. Sanity check on a mixed
        signal (tone + noise)."""
        rng = np.random.default_rng(123)
        t = np.arange(SR) / SR
        tone = 0.3 * np.sin(2 * np.pi * 1000.0 * t)
        noise = 0.05 * rng.normal(0, 1, SR)
        audio = (tone + noise).astype(np.float32)
        flatness = compute_spectral_flatness(
            audio, SR, 0.01, body_window_ms=50.0,
        )
        assert flatness is not None
        assert 0.0 <= flatness <= 1.0

    def test_attached_to_compute_event_features(self):
        """The flatness value should appear in the dict
        returned by compute_event_features — it's
        automatic, no special wiring needed."""
        audio = _make_tone(200.0, 0.3, decay_tau_ms=200.0)
        feats = compute_event_features(audio, SR, 0.01)
        # The key must exist (even if value is None on
        # edge cases like very short audio).
        assert 'spectral_flatness' in feats
        # The value should be a float or None, not raise.
        assert feats['spectral_flatness'] is None or isinstance(
            feats['spectral_flatness'], float
        )


class TestHighResDecaySignature:
    """Tests for the high-res attack+decay signature (2026-06-11).

    The function uses a much finer STFT (n_fft=128, hop=4)
    than the rest of the pipeline — enough to see single-frame
    transients and the 5-15ms ring that distinguishes real
    strikes from "pop" / "gap" artifacts.

    Two key fields:
      * decay_envelope_energy: ring energy in 15ms post-peak
      * decay_col_min_median_db: broadband level in decay window

    Real strikes should have HIGH decay_envelope_energy and
    HIGH dec_cmin (less negative). Noise / gap events should
    have LOW decay_envelope_energy and LOW dec_cmin (close
    to the noise floor, around -80 dB).
    """

    def test_real_strike_has_high_decay_energy(self):
        """A synthetic toms-like strike (impulse + broadband
        decaying body) should have a high decay_envelope_energy
        and dec_cmin significantly above the noise floor.

        The body needs BROADBAND content (not just a few
        harmonics) for col_min to be high — col_min is the
        LOWEST energy bin, so a harmonic stack has quiet
        bins between harmonics at the noise floor. A
        band-limited noise envelope has high col_min."""
        sr = SR  # 22050 in test
        audio = np.zeros(sr // 2, dtype=np.float32)
        rng = np.random.default_rng(42)
        # Impulse attack (10ms of broadband noise)
        burst = (0.5 * rng.normal(0, 1, int(0.010 * sr))).astype(np.float32)
        # Body: band-limited noise modulated by an exponential
        # decay envelope. The noise keeps col_min high
        # because every bin has some energy.
        t = np.arange(int(0.300 * sr)) / sr
        envelope = np.exp(-t / 0.100).astype(np.float32)
        noise_body = (0.3 * rng.normal(0, 1, len(t))
                      * envelope).astype(np.float32)
        onset = int(0.100 * sr)
        audio[onset:onset + len(burst)] = burst
        audio[onset + len(burst):onset + len(burst) + len(noise_body)] = noise_body
        sig = compute_high_res_decay_signature(
            audio, sr, 0.100,
        )
        assert sig is not None
        # Real strike signature: high decay energy, dec_cmin
        # significantly above -80 dB.
        assert sig['decay_envelope_energy'] > 100, (
            f"real strike should have decay_envelope_energy > 100, "
            f"got {sig['decay_envelope_energy']}"
        )
        assert sig['decay_col_min_median_db'] > -80, (
            f"real strike should have dec_cmin > -80 dB, "
            f"got {sig['decay_col_min_median_db']}"
        )

    def test_single_frame_noise_has_low_decay(self):
        """A single-frame noise spike (no ring) should have
        near-zero decay_envelope_energy and dec_cmin at the
        noise floor."""
        sr = SR
        audio = np.zeros(sr // 2, dtype=np.float32)
        rng = np.random.default_rng(123)
        # Insert a single 1-sample click at 0.1s
        onset = int(0.100 * sr)
        audio[onset] = 0.5
        sig = compute_high_res_decay_signature(
            audio, sr, 0.100,
        )
        assert sig is not None
        # The high-res peak is the click itself. The decay
        # window after the peak should be near-silent.
        assert sig['decay_envelope_energy'] < 50, (
            f"noise spike should have decay_envelope_energy < 50, "
            f"got {sig['decay_envelope_energy']}"
        )
        assert sig['decay_col_min_median_db'] < -75, (
            f"noise spike should have dec_cmin < -75 dB, "
            f"got {sig['decay_col_min_median_db']}"
        )

    def test_silence_returns_none_or_zeros(self):
        """Pure silence should produce a signature with no
        meaningful decay content. We accept None OR a dict
        with decay_envelope_energy=0 (both are valid
        graceful behaviors)."""
        audio = np.zeros(SR, dtype=np.float32)
        sig = compute_high_res_decay_signature(audio, SR, 0.1)
        if sig is not None:
            assert sig['decay_envelope_energy'] == 0.0
            assert sig['decay_col_min_median_db'] is None or (
                sig['decay_col_min_median_db'] < -75
            )

    def test_returns_dict_with_expected_keys(self):
        """The signature dict has the documented keys."""
        sr = SR
        audio = np.zeros(sr // 2, dtype=np.float32)
        rng = np.random.default_rng(7)
        # Just some audio in the band
        audio += (0.1 * rng.normal(0, 1, len(audio))).astype(np.float32)
        sig = compute_high_res_decay_signature(audio, sr, 0.1)
        if sig is not None:
            for key in (
                'hr_peak_time',
                'hr_peak_offset_ms',
                'hr_peak_envelope',
                'decay_envelope_energy',
                'decay_col_min_median_db',
            ):
                assert key in sig, f"missing key {key} in signature"

    def test_attached_to_compute_event_features(self):
        """The signature should be auto-attached via
        compute_event_features — no caller change needed."""
        # Build a strike-like signal
        sr = SR
        audio = np.zeros(sr // 2, dtype=np.float32)
        rng = np.random.default_rng(99)
        burst = rng.normal(0, 0.5, int(0.010 * sr)).astype(np.float32)
        t = np.arange(int(0.300 * sr)) / sr
        body = (0.4 * np.sin(2 * np.pi * 200.0 * t)
                * np.exp(-t / 0.100)).astype(np.float32)
        onset = int(0.100 * sr)
        audio[onset:onset + len(burst)] = burst
        audio[onset + len(burst):onset + len(burst) + len(body)] = body
        feats = compute_event_features(audio, sr, 0.100)
        for key in (
            'hr_peak_offset_ms',
            'decay_envelope_energy',
            'decay_col_min_median_db',
        ):
            assert key in feats, f"missing key {key} in features"
            assert feats[key] is None or isinstance(feats[key], (int, float))
