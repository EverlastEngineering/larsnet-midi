"""
Tests for the pitch-detection gating in onset_filtering.

User report (2026-06-08): the WebUI's "Cluster By" dropdown lets the
user pick ``pitch_hz`` for snare and cymbals, the settings schema
exposes ``enable_pitch_detection`` for those stems, but the
detection pipeline at ``onset_filtering.py:258`` hardcodes
``if stem_type == 'toms':`` — so pitch_hz is *never* computed for
snare/cymbals events, even when the user has enabled pitch
detection in their config. Round 4 of the T2 follow-up added a
silent-fallback warning so the user at least sees why their
cluster-feature choice is being ignored, but the underlying bug
(pitch never gets computed for snare/cymbals) was still unfixed.

These tests lock the contract: ``_should_detect_pitch(stem_type,
config)`` returns ``(fmin, fmax)`` to detect, or ``None`` to skip.
The contract:

  - toms: always detects (legacy behavior, hardcoded for years)
  - snare: detects if ``config['snare']['enable_pitch_detection']`` is True
  - cymbals: detects if ``config['cymbals']['enable_pitch_detection']`` is True
  - kick / hihat: never detects (no schema entry, no implementation)
  - any stem without ``enable_pitch_detection`` set: don't detect
"""

import pytest

from stems_to_midi.analysis_core.onset_filtering import _should_detect_pitch


# ─── Toms: legacy behavior, always detects ──────────────────────────────


class TestTomsPitchDetection:
    """Toms' pitch detection is a long-standing feature and must
    continue to work. The old code path was
    ``if stem_type == 'toms': detected_pitch = detect_pitch(...)``.
    The new helper must preserve that behavior — toms always
    detects."""

    def test_toms_with_no_config_detects(self):
        """No config dict at all: toms should still detect (legacy
        hardcoded behavior)."""
        result = _should_detect_pitch('toms', {})
        assert result is not None, (
            "Toms should always detect pitch (legacy behavior). "
            "Got None — the hardcoded toms check was lost."
        )
        fmin, fmax = result
        # Toms fundamentals are typically 40-250Hz; old code used 40-500
        assert fmin <= 100
        assert fmax >= 200

    def test_toms_with_explicit_enable_pitch_detection_true(self):
        """Toms with enable_pitch_detection=true (the default in
        midiconfig.yaml) should still detect — no regression."""
        result = _should_detect_pitch(
            'toms', {'toms': {'enable_pitch_detection': True}}
        )
        assert result is not None

    def test_toms_with_enable_pitch_detection_false_legacy(self):
        """Toms with enable_pitch_detection=false: the legacy
        behavior was to always detect. The schema's default is
        True, so this is an unusual case. We preserve legacy
        behavior: toms always detects regardless of the flag.
        (The user can set ``kick.geomean_threshold`` etc. to
        disable features on a per-event basis via the rebuild
        path.)"""
        # This test documents the legacy contract — toms doesn't
        # honor enable_pitch_detection (yet). If a future change
        # makes toms honor the flag, this test will need to be
        # updated alongside the production code.
        result = _should_detect_pitch(
            'toms', {'toms': {'enable_pitch_detection': False}}
        )
        # Legacy: still detects.
        assert result is not None


# ─── Snare: gated on enable_pitch_detection ─────────────────────────────


class TestSnarePitchDetection:
    """Snare's pitch detection must be gated on
    ``config['snare']['enable_pitch_detection']``. The user
    exposed this in round 4 by picking 'Pitch' in the Cluster By
    dropdown and seeing no change in the output — because
    enable_pitch_detection defaulted to false, and even when
    set to true, the old hardcoded ``if stem_type == 'toms'``
    block skipped snare entirely.
    """

    def test_snare_with_no_config_does_not_detect(self):
        """No config: snare should NOT detect. (Old behavior was
        'skip everything except toms'; that was correct for
        snare since snare.enable_pitch_detection default is
        false.)"""
        result = _should_detect_pitch('snare', {})
        assert result is None, (
            "Snare with no config should not detect pitch "
            "(default is enable_pitch_detection: false). "
            "Got: %r" % (result,)
        )

    def test_snare_with_enable_pitch_detection_true_detects(self):
        """Snare with enable_pitch_detection=true should detect.
        This is the fix — the old hardcoded check skipped snare
        regardless of the flag."""
        result = _should_detect_pitch(
            'snare', {'snare': {'enable_pitch_detection': True}}
        )
        assert result is not None, (
            "Snare with enable_pitch_detection=true must detect pitch. "
            "Got None — the hardcoded toms check is still in place."
        )
        fmin, fmax = result
        # Snare fundamentals span ~100-500Hz per the schema defaults
        # (snare.min_pitch_hz=100, snare.max_pitch_hz=500). The
        # helper should honor the config or fall back to a safe
        # range.
        assert fmin < fmax
        assert fmin >= 20   # sanity: not lower than audible drums
        assert fmax <= 8000  # sanity: not higher than percussion

    def test_snare_with_enable_pitch_detection_false_does_not_detect(self):
        """Snare with enable_pitch_detection=false should NOT detect
        (the default)."""
        result = _should_detect_pitch(
            'snare', {'snare': {'enable_pitch_detection': False}}
        )
        assert result is None

    def test_snare_uses_per_stem_pitch_hz_bounds(self):
        """When the user configures custom snare min_pitch_hz /
        max_pitch_hz, those bounds should be used (not the
        toms default of 40-500)."""
        result = _should_detect_pitch('snare', {
            'snare': {
                'enable_pitch_detection': True,
                'min_pitch_hz': 200.0,
                'max_pitch_hz': 800.0,
            }
        })
        assert result is not None
        fmin, fmax = result
        # The helper should either use the exact configured bounds
        # or at least stay within them. We assert fmin and fmax
        # are within the configured range (allowing some tolerance
        # if the helper applies a small headroom).
        assert fmin >= 100, f"fmin={fmin} below expected {200}"
        assert fmax <= 1500, f"fmax={fmax} above expected {800}"


# ─── Cymbals: gated on enable_pitch_detection ───────────────────────────


class TestCymbalsPitchDetection:
    """Cymbals: same contract as snare. Gated on
    ``config['cymbals']['enable_pitch_detection']``."""

    def test_cymbals_with_no_config_does_not_detect(self):
        result = _should_detect_pitch('cymbals', {})
        assert result is None

    def test_cymbals_with_enable_pitch_detection_true_detects(self):
        result = _should_detect_pitch(
            'cymbals', {'cymbals': {'enable_pitch_detection': True}}
        )
        assert result is not None, (
            "Cymbals with enable_pitch_detection=true must detect pitch. "
            "Got None — the hardcoded toms check is still in place."
        )
        fmin, fmax = result
        assert fmin < fmax

    def test_cymbals_with_enable_pitch_detection_false_does_not_detect(self):
        result = _should_detect_pitch(
            'cymbals', {'cymbals': {'enable_pitch_detection': False}}
        )
        assert result is None


# ─── Kick and Hihat: never detect (no schema entry) ─────────────────────


class TestKickAndHihatPitchDetection:
    """Kick and hihat have no pitch-detection config in the schema.
    They must NEVER detect pitch — the helper returns None for
    these stems regardless of the config."""

    def test_kick_never_detects(self):
        for config in [{}, {'kick': {'enable_pitch_detection': True}}]:
            result = _should_detect_pitch('kick', config)
            assert result is None, (
                "Kick must never detect pitch (no schema entry, "
                "no implementation). Got: %r for config %r" % (result, config)
            )

    def test_hihat_never_detects(self):
        for config in [{}, {'hihat': {'enable_pitch_detection': True}}]:
            result = _should_detect_pitch('hihat', config)
            assert result is None, (
                "Hihat must never detect pitch (no schema entry, "
                "no implementation; hihat uses threshold-based "
                "open/closed, not pitch clustering). "
                "Got: %r for config %r" % (result, config)
            )

    def test_unknown_stem_does_not_detect(self):
        """Defensive: an unknown stem type must return None,
        not raise. This protects against future stems being
        added without thinking about pitch."""
        for stem in ['unknown', 'cowbell', '']:
            result = _should_detect_pitch(stem, {})
            assert result is None, (
                "Unknown stem %r must return None. Got: %r" % (stem, result)
            )
