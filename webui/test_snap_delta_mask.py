"""
Tests for the snap-delta mask slider in the WebUI toms tuning panel
(client-side) and the matching server-side filter in
``stems_to_midi.processing_shell._build_events_configured`` /
``stems_to_midi.rebuild_core._apply_snap_mask`` (so the saved MIDI
reflects the mask).

Added 2026-06-09 after the user asked to be able to mask events whose
``snap_delta == 0`` (or below a threshold) directly in the tuning view,
so they can sweep the cutoff and see which events get filtered, AND
have the same mask persist into the saved MIDI.

Semantics (revised 2026-06-09 after the user pointed out the original
``<`` gate made threshold=0 a no-op — they expected ``<=`` so 0 means
"kill exactly the snap_delta==0 events"):

  * The slider in the toms tuning panel runs a 4th pass in
    ``applyTuningFilter()`` that mutates ``event.status`` to 'FILTERED'
    for KEPT events with ``snap_delta <= threshold``. Events without a
    ``snap_delta`` field (non-spectral detection methods) are
    untouched.

  * The server-side mirror in ``_build_events_configured`` and
    ``_apply_snap_mask`` applies the same filter so the saved MIDI
    never contains masked events.

  * Threshold is INCLUSIVE (≤):
      - 0    → filter all snap_delta==0 events
      - 0.001 (default) → same plus a tiny epsilon to catch
                floating-point zeros
      - 0.05 → filter snap_delta ≤ 0.05
      - 0.5  → only the strongest attacks survive
      - <0   → disable the mask (back-compat / explicit opt-out)
      - None → disable the mask (key missing from config)

These tests cover:

  * The slider is registered in ``STEM_SLIDER_CONFIGS.toms`` with
    key ``snap_mask_threshold``, range 0-0.5, default 0.001.
  * The slider is NOT a classification slider (it does not need a
    server-side reclassify call).
  * The ``applySnapDeltaMask(events, threshold)`` function:
      - Marks KEPT events with snap_delta <= threshold as FILTERED
      - Leaves KEPT events with snap_delta > threshold alone
      - Leaves events without snap_delta alone
      - Leaves non-KEPT events (REVERB_CONTINUATION, FILTERED) alone
  * The slider value is forwarded via ``_buildConfigOverrides`` so
    server-side rebuilds see the same threshold the user sees in the
    tuning panel.
  * The server-side ``_build_events_configured`` and
    ``_apply_snap_mask`` apply the same filter and respect the
    override flag.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
THRESHOLD_TUNING_JS = REPO_ROOT / 'webui' / 'static' / 'js' / 'threshold-tuning.js'


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def threshold_tuning_js_text() -> str:
    return THRESHOLD_TUNING_JS.read_text()


@pytest.fixture(scope='module')
def node_available() -> bool:
    return shutil.which('node') is not None


# ─── 1. Static JS structure: the slider is registered ────────────────────

class TestSnapDeltaMaskSliderConfig:
    """The new slider must be registered in STEM_SLIDER_CONFIGS.toms
    with the documented key, range, and default."""

    def test_toms_slider_config_has_snap_mask_threshold(self, threshold_tuning_js_text):
        """STEM_SLIDER_CONFIGS.toms must contain an entry with
        key='snap_mask_threshold'. This is the slider that controls
        the snap-delta mask in the toms tuning panel."""
        # Extract the toms block from STEM_SLIDER_CONFIGS.
        m = re.search(
            r"STEM_SLIDER_CONFIGS\s*=\s*\{(.*?)\n\};",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate STEM_SLIDER_CONFIGS block"
        body = m.group(1)
        # Find the toms: [ ... ] block.
        m_toms = re.search(
            r"toms:\s*\[(.*?)\],\s*(?:hihat|cymbals|kick|snare):",
            body,
            re.DOTALL,
        )
        assert m_toms is not None, "could not locate toms: [...] block"
        toms_block = m_toms.group(1)
        assert "key: 'snap_mask_threshold'" in toms_block, (
            "STEM_SLIDER_CONFIGS.toms must include a slider with "
            "key='snap_mask_threshold' so the toms tuning panel shows "
            "the snap-delta mask slider."
        )

    def test_snap_mask_slider_range_is_zero_to_half(self, threshold_tuning_js_text):
        """The slider must have min=0, max=0.5, step=0.01. The
        fallback is the schema default 0.001 (a tiny epsilon to
        catch floating-point zeros)."""
        m = re.search(
            r"key:\s*'snap_mask_threshold',\s*label:[^}]*\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, (
            "could not locate the snap_mask_threshold slider config object"
        )
        config = m.group(0)
        assert re.search(r"min:\s*0\b", config), (
            f"snap_mask_threshold min must be 0 (the natural 'kill zeros' "
            f"value). Got: {config!r}"
        )
        assert re.search(r"max:\s*0\.5\b", config), (
            f"snap_mask_threshold max must be 0.5. Got: {config!r}"
        )
        assert re.search(r"fallback:\s*0\.001\b", config), (
            f"snap_mask_threshold fallback must be 0.001 (schema default — "
            f"clean up zero-snap events out of the box). Got: {config!r}"
        )

    def test_snap_mask_slider_is_not_classification(self, threshold_tuning_js_text):
        """The slider must NOT be marked as a classification slider
        — the snap-delta mask is a pure client-side filter, it does
        not need a server-side reclassify call. Marking it
        classification=true would trigger an unnecessary
        /api/reclassify round-trip on every slider drag."""
        m = re.search(
            r"key:\s*'snap_mask_threshold',\s*label:[^}]*\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        config = m.group(0)
        assert "classification: true" not in config, (
            "snap_mask_threshold must not be a classification slider. "
            "The mask is a pure client-side filter — no server reclassify "
            "is needed when it changes."
        )


# ─── 2. Static JS structure: the applySnapDeltaMask function exists ─────

class TestSnapDeltaMaskFunction:
    """The applySnapDeltaMask(events, threshold) function must be
    defined in threshold-tuning.js."""

    def test_apply_snap_delta_mask_function_exists(self, threshold_tuning_js_text):
        m = re.search(
            r"function\s+applySnapDeltaMask\s*\(\s*events\s*,\s*threshold\s*\)",
            threshold_tuning_js_text,
        )
        assert m is not None, (
            "expected `function applySnapDeltaMask(events, threshold)` in "
            "threshold-tuning.js. The function performs the snap-delta "
            "mask pass in applyTuningFilter()."
        )

    def test_apply_snap_delta_mask_uses_inclusive_comparison(self, threshold_tuning_js_text):
        """The mask must use ``<=`` (inclusive), not ``<``. Threshold=0
        must filter snap_delta==0 events; threshold=0.05 must filter
        snap_delta==0.05 events. The user explicitly asked for the
        inclusive semantic (2026-06-09)."""
        fn = self._extract_function(threshold_tuning_js_text)
        assert "<= threshold" in fn or "snap_delta <= threshold" in fn, (
            "applySnapDeltaMask must use the inclusive `<=` comparison. "
            "The user-facing semantic is: threshold=0 kills all "
            "snap_delta==0 events; threshold=0.05 kills everything ≤ 0.05. "
            "If you see 'snap_delta < threshold' in the function, the "
            "mask is broken — threshold=0 becomes a no-op."
        )
        # Make sure the broken `<` form is NOT there.
        assert "snap_delta < threshold" not in fn, (
            "applySnapDeltaMask must NOT use 'snap_delta < threshold'. "
            "The user expects inclusive (`<=`) semantics. The `<` form "
            "made threshold=0 a no-op, which the user flagged as a bug."
        )

    def _extract_function(self, js_text: str) -> str:
        """Pull the applySnapDeltaMask function out of threshold-tuning.js
        so it can be inspected (or eval'd) in isolation. We anchor on
        the function declaration and end at the next top-level
        `function` or end of file."""
        m = re.search(
            r"function\s+applySnapDeltaMask\s*\([^)]*\)\s*\{(.*?)\n\}\n",
            js_text,
            re.DOTALL,
        )
        if m is None:
            raise AssertionError("could not extract applySnapDeltaMask body")
        return "function applySnapDeltaMask(events, threshold) {" + m.group(1) + "\n}"

    def test_apply_snap_delta_mask_wired_into_pipeline(self, threshold_tuning_js_text):
        """applyTuningFilter must call applySnapDeltaMask when
        snap_mask_threshold is set and the stem is toms. The
        conditional check is important: the slider value lives in
        the per-stem params dict. The gate must be ``>= 0`` (not
        ``> 0``) so threshold=0 actually does something — the
        ``> 0`` form was the original bug."""
        m = re.search(
            r"function\s+applyTuningFilter\s*\(\s*\)\s*\{(.*?)\n\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate applyTuningFilter body"
        body = m.group(1)
        assert "applySnapDeltaMask" in body, (
            "applyTuningFilter must call applySnapDeltaMask. Without "
            "this call, the snap-delta mask slider would have no effect."
        )
        assert "snap_mask_threshold" in body, (
            "applyTuningFilter must read the snap_mask_threshold slider "
            "value and pass it to applySnapDeltaMask."
        )
        # The gate must be >= 0 (not > 0). The previous `> 0` form
        # silently disabled the slider at the default 0 value — the
        # whole point of adding the slider is so the user CAN set 0
        # and have the zero-snap events filtered. The `>= 0` form
        # plus the `<=` comparison in applySnapDeltaMask makes 0
        # mean "filter snap_delta==0 events".
        assert re.search(
            r"snapMaskThreshold\s*!=\s*null[^&]*&&\s*snapMaskThreshold\s*>=\s*0",
            body,
        ), (
            "The snap-delta mask pass gate must be `>= 0`, not `> 0`. "
            "The `> 0` form made threshold=0 a no-op — exactly the bug "
            "the user reported on 2026-06-09. With `>= 0` and the "
            "inclusive `<=` comparison in applySnapDeltaMask, "
            "threshold=0 means 'filter snap_delta==0 events'."
        )
        # Must also gate on stemType === 'toms' so the slider only
        # applies to the toms stem (it has a snap_delta field, others
        # don't).
        assert re.search(
            r"snapMaskThreshold[^)]*stemType\s*===\s*['\"]toms['\"]",
            body,
        ), (
            "The snap-delta mask pass must be gated on stemType === 'toms' "
            "so it doesn't run on stems that don't have snap_delta values."
        )


# ─── 3. Node-based behavioral tests: the function actually works ─────────

class TestSnapDeltaMaskBehavior:
    """Behavioral tests for applySnapDeltaMask. Eval'd inside a minimal
    browser shim so the JS runs against a real JS engine. Skipped when
    node is not available."""

    SHIM = textwrap.dedent(r"""
        const document = { getElementById: () => null };
        const window = { devicePixelRatio: 1 };
    """)

    def _run_node(self, js_source: str) -> str:
        script = self.SHIM + "\n" + js_source
        result = subprocess.run(
            ['node', '-e', script],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"node exited {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        return result.stdout

    def _extract_function(self, js_text: str) -> str:
        m = re.search(
            r"function\s+applySnapDeltaMask\s*\([^)]*\)\s*\{(.*?)\n\}\n",
            js_text,
            re.DOTALL,
        )
        if m is None:
            raise AssertionError("could not extract applySnapDeltaMask body")
        return "function applySnapDeltaMask(events, threshold) {" + m.group(1) + "\n}"

    def test_marks_low_snap_delta_kept_events_as_filtered(
        self, node_available, threshold_tuning_js_text
    ):
        """A KEPT event with snap_delta at or below the threshold must
        be marked FILTERED after the mask pass. With the new inclusive
        ``<=`` semantics: snap_delta=0 is filtered at threshold=0.05;
        snap_delta=0.5 is not."""
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract_function(threshold_tuning_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', snap_delta: 0.0 }},
                {{ time: 2.0, status: 'KEPT', snap_delta: 0.5 }},
            ];
            applySnapDeltaMask(events, 0.05);
            console.log(events[0].status + '|' + events[1].status);
        """))
        assert out.strip() == 'FILTERED|KEPT', (
            f"expected 'FILTERED|KEPT' (0.0 below 0.05, 0.5 above), got {out.strip()!r}"
        )

    def test_threshold_zero_filters_zero_snap_events(
        self, node_available, threshold_tuning_js_text
    ):
        """Threshold 0 must filter all snap_delta==0 events. This is
        the user's expected semantic (2026-06-09): 'if the threshold is
        0, I expect all events with snap delta of zero to be filtered
        out of the results'. The old `<` form made this a no-op."""
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract_function(threshold_tuning_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', snap_delta: 0.0 }},
                {{ time: 2.0, status: 'KEPT', snap_delta: 0.5 }},
                {{ time: 3.0, status: 'KEPT', snap_delta: null }},
            ];
            applySnapDeltaMask(events, 0);
            console.log(
                events[0].status + '|' +
                events[1].status + '|' +
                events[2].status
            );
        """))
        assert out.strip() == 'FILTERED|KEPT|KEPT', (
            f"threshold 0 must filter snap_delta==0 events (and leave "
            f"non-zero + null-snap events alone), got {out.strip()!r}. "
            f"This is the bug the user flagged on 2026-06-09."
        )

    def test_threshold_is_inclusive_at_boundary(
        self, node_available, threshold_tuning_js_text
    ):
        """An event with snap_delta exactly equal to the threshold must
        be filtered (inclusive `<=`). threshold=0.05 should filter
        snap_delta=0.05."""
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract_function(threshold_tuning_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', snap_delta: 0.05 }},
                {{ time: 2.0, status: 'KEPT', snap_delta: 0.0500001 }},
            ];
            applySnapDeltaMask(events, 0.05);
            console.log(events[0].status + '|' + events[1].status);
        """))
        assert out.strip() == 'FILTERED|KEPT', (
            f"threshold 0.05 must filter snap_delta==0.05 (inclusive), "
            f"got {out.strip()!r}"
        )

    def test_leaves_non_kept_events_untouched(
        self, node_available, threshold_tuning_js_text
    ):
        """Events that are already REVERB_CONTINUATION or FILTERED
        must NOT be touched by the snap mask — those statuses are
        the result of earlier filter passes and the snap mask
        doesn't have authority to change them."""
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract_function(threshold_tuning_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'REVERB_CONTINUATION', snap_delta: 0.0 }},
                {{ time: 2.0, status: 'FILTERED', snap_delta: 0.0 }},
            ];
            applySnapDeltaMask(events, 0.05);
            console.log(events[0].status + '|' + events[1].status);
        """))
        assert out.strip() == 'REVERB_CONTINUATION|FILTERED', (
            f"non-KEPT events must not be touched, got {out.strip()!r}"
        )

    def test_leaves_events_without_snap_delta_untouched(
        self, node_available, threshold_tuning_js_text
    ):
        """Events with snap_delta == null (non-spectral detection
        methods) must be left alone. The mask is a spectral-event-
        only filter."""
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract_function(threshold_tuning_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', snap_delta: null }},
                {{ time: 2.0, status: 'KEPT' }},
            ];
            applySnapDeltaMask(events, 0.5);
            console.log(events[0].status + '|' + events[1].status);
        """))
        assert out.strip() == 'KEPT|KEPT', (
            f"events without snap_delta must be untouched, got {out.strip()!r}"
        )

    def test_threshold_one_masks_everything(
        self, node_available, threshold_tuning_js_text
    ):
        """A threshold of 1.0 is an upper bound for the real snap
        signal — every spectral event's snap_delta is ≤ 1.0, so
        this threshold masks all of them. Verifies the filter is
        doing inclusive-≤-or-equal (events with snap_delta == 1.0
        are also kept)."""
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract_function(threshold_tuning_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', snap_delta: 0.0 }},
                {{ time: 2.0, status: 'KEPT', snap_delta: 0.5 }},
                {{ time: 3.0, status: 'KEPT', snap_delta: 0.99 }},
            ];
            applySnapDeltaMask(events, 1.0);
            console.log(
                events[0].status + '|' +
                events[1].status + '|' +
                events[2].status
            );
        """))
        assert out.strip() == 'FILTERED|FILTERED|FILTERED', (
            f"threshold 1.0 should mask all real snap values (≤ 1.0), "
            f"got {out.strip()!r}"
        )


# ─── 4. Server-side wiring: the slider value is forwarded on rebuild ─────

class TestSnapMaskConfigOverride:
    """The snap_mask_threshold slider value must be included in
    ``_buildConfigOverrides`` so server-side rebuilds see the same
    threshold the user sees in the tuning panel (Bug D pattern)."""

    def test_snap_mask_threshold_in_per_stem_overrides(self, threshold_tuning_js_text):
        m = re.search(
            r"function\s+_buildConfigOverrides\s*\([^)]*\)\s*\{(.*?)\n\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate _buildConfigOverrides body"
        body = m.group(1)
        # snap_mask_threshold must be in the per-stem key list.
        assert re.search(
            r"['\"]snap_mask_threshold['\"]",
            body,
        ), (
            "_buildConfigOverrides must forward snap_mask_threshold to "
            "the server so the rebuild uses the same value the user sees "
            "in the tuning panel. Without this, the client-side preview "
            "and the saved MIDI would diverge."
        )


# ─── 5. Server-side: the mask persists into the saved MIDI ───────────────

class TestServerSideSnapMask:
    """The server-side helpers must apply the same filter so the
    saved MIDI never contains masked events."""

    def test_build_events_configured_signature_accepts_threshold(self):
        """_build_events_configured must accept stem_type and config
        kwargs so it can read snap_mask_threshold."""
        from stems_to_midi.processing_shell import _build_events_configured
        import inspect
        sig = inspect.signature(_build_events_configured)
        assert 'stem_type' in sig.parameters, (
            "_build_events_configured must accept a stem_type kwarg so "
            "the snap mask can read config[stem_type].snap_mask_threshold"
        )
        assert 'config' in sig.parameters, (
            "_build_events_configured must accept a config kwarg so "
            "the snap mask can read the threshold"
        )

    def test_build_events_configured_applies_mask_to_spectral_events(self):
        """Spectral events with snap_delta ≤ threshold must be marked
        FILTERED in events_configured. The MIDI serializer skips
        FILTERED events, so the saved MIDI never contains them."""
        from stems_to_midi.processing_shell import _build_events_configured

        all_onset_data = []  # No energy events
        spectral_onset_data = [
            {'time': 1.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.0},     # Should be filtered at 0.05
            {'time': 2.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.5},     # Above threshold — kept
            {'time': 3.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.05},    # == threshold — filtered (inclusive)
        ]
        config = {'toms': {'snap_mask_threshold': 0.05}}
        result = _build_events_configured(
            all_onset_data=all_onset_data,
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method='spectral',
            stem_type='toms',
            config=config,
        )
        statuses = [e['status'] for e in result]
        # 0.0 ≤ 0.05 → FILTERED
        # 0.5 > 0.05 → KEPT
        # 0.05 == 0.05 → FILTERED (inclusive)
        assert statuses == ['FILTERED', 'KEPT', 'FILTERED'], (
            f"snap_mask=0.05 should filter snap_delta 0.0 and 0.05 "
            f"(inclusive), keep 0.5. Got: {statuses}"
        )

    def test_build_events_configured_no_mask_when_threshold_missing(self):
        """If the threshold key is missing from config, no mask
        is applied (back-compat with projects that pre-date this
        setting)."""
        from stems_to_midi.processing_shell import _build_events_configured

        spectral_onset_data = [
            {'time': 1.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.0},
        ]
        # No config at all — back-compat path.
        result = _build_events_configured(
            all_onset_data=[],
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method='spectral',
            stem_type='toms',
            config=None,
        )
        assert result[0]['status'] == 'KEPT', (
            "without a config arg, _build_events_configured must not "
            "apply the snap mask (back-compat)"
        )

    def test_build_events_configured_threshold_zero_kills_zeros(self):
        """threshold=0 must filter snap_delta==0 events (the user's
        documented expectation from 2026-06-09)."""
        from stems_to_midi.processing_shell import _build_events_configured

        spectral_onset_data = [
            {'time': 1.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.0},
            {'time': 2.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.001},
        ]
        config = {'toms': {'snap_mask_threshold': 0.0}}
        result = _build_events_configured(
            all_onset_data=[],
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method='spectral',
            stem_type='toms',
            config=config,
        )
        statuses = [e['status'] for e in result]
        assert statuses == ['FILTERED', 'KEPT'], (
            f"threshold=0 must filter snap_delta==0 only, "
            f"leave 0.001 (above 0) KEPT. Got: {statuses}"
        )

    def test_build_events_configured_negative_threshold_disables(self):
        """A negative threshold is the explicit 'off' value — no
        events are masked."""
        from stems_to_midi.processing_shell import _build_events_configured

        spectral_onset_data = [
            {'time': 1.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.0},
        ]
        config = {'toms': {'snap_mask_threshold': -1.0}}
        result = _build_events_configured(
            all_onset_data=[],
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method='spectral',
            stem_type='toms',
            config=config,
        )
        assert result[0]['status'] == 'KEPT', (
            "negative threshold must disable the mask"
        )

    def test_rebuild_core_apply_snap_mask_function_exists(self):
        """rebuild_core._apply_snap_mask must exist so the mask
        also applies in the rebuild path (after a slider change
        triggers a MIDI rebuild, not a full pipeline run)."""
        from stems_to_midi.rebuild_core import _apply_snap_mask
        assert callable(_apply_snap_mask)

    def test_rebuild_core_apply_snap_mask_marks_filtered(self):
        """_apply_snap_mask must mark KEPT events with snap_delta ≤
        threshold as FILTERED, in place, and respect the override
        flag."""
        from stems_to_midi.rebuild_core import _apply_snap_mask

        events = [
            {'time': 1.0, 'status': 'KEPT', 'snap_delta': 0.0},    # filtered
            {'time': 2.0, 'status': 'KEPT', 'snap_delta': 0.5},    # kept
            {'time': 3.0, 'status': 'KEPT', 'snap_delta': None},   # kept (no field)
            {'time': 4.0, 'status': 'KEPT', 'snap_delta': 0.0, 'override': True},  # kept (override)
            {'time': 5.0, 'status': 'FILTERED', 'snap_delta': 0.0},  # not touched
        ]
        config = {'toms': {'snap_mask_threshold': 0.05}}
        _apply_snap_mask(events, config, 'toms')
        statuses = [e['status'] for e in events]
        assert statuses == [
            'FILTERED',  # 0.0 ≤ 0.05
            'KEPT',      # 0.5 > 0.05
            'KEPT',      # no snap_delta
            'KEPT',      # override respected
            'FILTERED',  # already filtered — not promoted to KEPT
        ], f"snap mask broke invariants, got: {statuses}"

    def test_rebuild_core_apply_snap_mask_recovers_under_looser_threshold(self):
        """Bug 2 fix (2026-06-10): a previously-FILTERED spectral
        event whose snap_delta is now above the new (looser)
        threshold must recover to KEPT. The mask is idempotent
        and direction-agnostic — loosening the slider restores
        the events that the old, stricter mask had hidden."""
        from stems_to_midi.rebuild_core import _apply_snap_mask

        # Simulate state after a prior Save with snap_mask=0.5:
        # the 0.1 event was masked (0.1 ≤ 0.5), the 0.05 event
        # was masked (0.05 ≤ 0.5), the 0.6 event was kept.
        events = [
            {'time': 1.0, 'status': 'FILTERED', 'snap_delta': 0.1, 'method': 'spectral'},
            {'time': 2.0, 'status': 'FILTERED', 'snap_delta': 0.05, 'method': 'spectral'},
            {'time': 3.0, 'status': 'KEPT',     'snap_delta': 0.6, 'method': 'spectral'},
        ]
        # User loosens the threshold to 0.05 (was 0.5). The 0.1
        # event is now above the threshold and must be KEPT; the
        # 0.05 event is at the boundary (inclusive) and must be
        # FILTERED.
        config = {'toms': {'snap_mask_threshold': 0.05}}
        _apply_snap_mask(events, config, 'toms')
        statuses = [e['status'] for e in events]
        assert statuses == [
            'KEPT',      # 0.1 > 0.05 — RECOVERED
            'FILTERED',  # 0.05 ≤ 0.05 — still masked (boundary)
            'KEPT',      # 0.6 > 0.05 — unchanged
        ], (
            "Bug 2: previously-FILTERED spectral events must recover "
            "when the threshold loosens. Got: {statuses}"
        )

    def test_rebuild_core_apply_snap_mask_toggle_off_restores_all(self):
        """Bug 2 fix (2026-06-10): when the user turns the snap
        mask toggle OFF, every spectral event that was previously
        filtered by the mask must recover to KEPT (override flag
        and energy events are still respected)."""
        from stems_to_midi.rebuild_core import _apply_snap_mask

        # Prior state: threshold=0.05, toggle=on (legacy default).
        # All 0.0 events are FILTERED.
        events = [
            {'time': 1.0, 'status': 'FILTERED', 'snap_delta': 0.0, 'method': 'spectral'},
            {'time': 2.0, 'status': 'FILTERED', 'snap_delta': 0.0, 'method': 'spectral'},
            {'time': 3.0, 'status': 'KEPT',     'snap_delta': 0.5, 'method': 'spectral'},
            # Energy event: not applicable.
            {'time': 4.0, 'status': 'FILTERED', 'method': 'energy', 'geomean': 5},
        ]
        # User turns the toggle off (and keeps the threshold for
        # record — it doesn't matter, the toggle is authoritative).
        config = {'toms': {
            'snap_mask_enabled': False,
            'snap_mask_threshold': 0.05,
        }}
        _apply_snap_mask(events, config, 'toms')
        statuses = [e['status'] for e in events]
        assert statuses == [
            'KEPT',      # 0.0, was masked, now recovered
            'KEPT',      # 0.0, was masked, now recovered
            'KEPT',      # was already KEPT — unchanged
            'FILTERED',  # energy event, untouched (no snap_delta)
        ], (
            "Bug 2: snap_mask_enabled=False must restore all "
            "spectral events the mask had hidden. Got: {statuses}"
        )

    def test_rebuild_core_apply_snap_mask_does_not_touch_overrides(self):
        """An override-Kept event stays KEPT through both the
        Pass A reset and the Pass B re-evaluation. The override
        flag is the strongest signal — neither pass touches it."""
        from stems_to_midi.rebuild_core import _apply_snap_mask

        events = [
            {'time': 1.0, 'status': 'KEPT', 'snap_delta': 0.0, 'override': True},
            {'time': 2.0, 'status': 'KEPT', 'snap_delta': 0.0, 'override': True},
        ]
        # Toggle off — would normally reset FILTERED to KEPT, but
        # KEPT events with override are not touched at all.
        config = {'toms': {'snap_mask_enabled': False, 'snap_mask_threshold': 0.05}}
        _apply_snap_mask(events, config, 'toms')
        assert events[0]['status'] == 'KEPT'
        assert events[1]['status'] == 'KEPT'

    def test_rebuild_core_apply_snap_mask_no_threshold_no_op(self):
        """When config[stem_type].snap_mask_threshold is missing or
        None, the mask is a no-op (back-compat)."""
        from stems_to_midi.rebuild_core import _apply_snap_mask

        events = [
            {'time': 1.0, 'status': 'KEPT', 'snap_delta': 0.0},
        ]
        # No config — back-compat.
        _apply_snap_mask(events, {}, 'toms')
        assert events[0]['status'] == 'KEPT'
        # Negative — explicitly disabled.
        _apply_snap_mask(events, {'toms': {'snap_mask_threshold': -0.5}}, 'toms')
        assert events[0]['status'] == 'KEPT'


# ─── 6. Settings schema: the key is registered ───────────────────────────

class TestSettingsSchemaSnapMask:
    """The settings schema must include toms_snap_mask_threshold so
    it appears in the Settings form and CLI."""

    def test_settings_schema_has_toms_snap_mask_threshold(self):
        from webui.settings_schema import SETTINGS_REGISTRY
        keys = {s.key for s in SETTINGS_REGISTRY}
        assert 'toms_snap_mask_threshold' in keys, (
            "SETTINGS_REGISTRY must include toms_snap_mask_threshold "
            "so the WebUI Settings form and CLI expose the snap mask "
            "threshold"
        )

    def test_settings_schema_default_is_small_positive(self):
        """The default must be 0.001 (a small positive epsilon) so
        projects get a free cleanup of zero-snap events out of the
        box. Pure 0 would also work, but 0.001 catches floating-point
        '0.0' that should really be '1e-9'."""
        from webui.settings_schema import SETTINGS_REGISTRY
        snap = next(s for s in SETTINGS_REGISTRY if s.key == 'toms_snap_mask_threshold')
        assert snap.default == 0.001, (
            f"toms_snap_mask_threshold default must be 0.001, got {snap.default!r}"
        )
        assert snap.yaml_path == ['toms', 'snap_mask_threshold'], (
            f"yaml_path must be ['toms', 'snap_mask_threshold'], got {snap.yaml_path!r}"
        )


# ─── 7. Spectral events are exempt from the geomean filter ──────────────

class TestSpectralEventsExemptFromGeomeanFilter:
    """Bug fix (2026-06-09, Task 2): spectral events
    (method='spectral') must NOT be filtered by the geomean slider.
    Without this fix, dragging geomean to any non-default value
    silently destroys all magenta (spectral) events in the tuning
    view because spectral events have no geomean field.

    The fix: ``applySpectralFilter`` skips events with
    ``method === 'spectral'`` and lets them through to the snap
    mask pass (which is the right place to filter them — based on
    snap_delta, not geomean).
    """

    def test_apply_spectral_filter_skips_spectral_events(self, threshold_tuning_js_text):
        """The applySpectralFilter function must early-return for
        method='spectral' events so they're never touched by
        geomean / sustain / strength logic."""
        # Extract the function body so we can grep for the exemption.
        m = re.search(
            r"function\s+applySpectralFilter\s*\([^)]*\)\s*\{(.*?)\n\}\n",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not extract applySpectralFilter body"
        body = m.group(1)
        # Must have an early `continue` for spectral events.
        # The JS pattern is: `if (event.method === 'spectral') continue;`
        # so we need to allow `)` between the literal and `continue`.
        assert re.search(
            r"event\.method\s*===\s*['\"]spectral['\"]\s*\)?\s*;?\s*continue",
            body,
        ), (
            "applySpectralFilter must early-continue for "
            "method='spectral' events so the geomean / sustain / "
            "strength filters don't touch them. Without this, "
            "dragging the geomean slider silently destroys all "
            "magenta events."
        )

    def test_spectral_event_survives_extreme_geomean_in_node(
        self, node_available, threshold_tuning_js_text
    ):
        """End-to-end: a spectral event with no geomean field must
        survive an extreme geomean threshold (10000). Energy events
        get filtered; spectral events stay KEPT. This is the
        end-to-end test the user asked for ('you should be able to
        set the geomean filter super high to filter all of the
        older events out')."""
        if not node_available:
            pytest.skip('node not on PATH')
        # Extract both functions so we can call them in concert.
        m_spec = re.search(
            r"function\s+applySpectralFilter\s*\([^)]*\)\s*\{(.*?)\n\}\n",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        m_snap = re.search(
            r"function\s+applySnapDeltaMask\s*\([^)]*\)\s*\{(.*?)\n\}\n",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m_spec is not None
        assert m_snap is not None
        spec_fn = "function applySpectralFilter(events, params, filterMode) {" + m_spec.group(1) + "\n}"
        snap_fn = "function applySnapDeltaMask(events, threshold) {" + m_snap.group(1) + "\n}"
        out = self._run_node_silent(textwrap.dedent(f"""
            {spec_fn}
            {snap_fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', method: 'energy',
                   geomean: 50, strength: 0.5 }},
                {{ time: 2.0, status: 'KEPT', method: 'spectral',
                   snap_delta: 0.5 }},
            ];
            applySpectralFilter(events, {{ geomean_threshold: 10000 }}, 'geomean_only');
            console.log(events[0].status + '|' + events[1].status);
        """))
        assert out.strip() == 'FILTERED|KEPT', (
            f"with geomean=10000: energy event must be filtered, "
            f"spectral event must be KEPT. Got: {out.strip()!r}. "
            f"This is the bug — spectral events get nuked by the "
            f"geomean slider when they shouldn't be touched at all."
        )

    @staticmethod
    def _run_node_silent(js_source: str) -> str:
        """Like TestSnapDeltaMaskBehavior._run_node but with the
        document/window shim baked in so applySpectralFilter can
        run in isolation."""
        shim = "const document = { getElementById: () => null };\nconst window = { devicePixelRatio: 1 };\n"
        script = shim + "\n" + js_source
        result = subprocess.run(
            ['node', '-e', script],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"node exited {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        return result.stdout


# ─── 8. Snap-mask toggle (2026-06-10) ────────────────────────────────────
#
# The user asked for a TOGGLE in the tuning panel that controls whether
# the snap-mask is active at all. When the toggle is OFF, the slider
# disappears and the mask is skipped. When the toggle is ON, the slider
# appears and the mask runs with the chosen threshold. The toggle
# addresses two bugs from 2026-06-10:
#   1. The previous default of 0.001 silently filtered spectral events
#      on the first Save with no way to recover them.
#   2. The client-side `applyTuningFilter` always applied the mask with
#      the schema default, so the Tune view would hide events the
#      default view showed as KEPT (state desync).
#
# The toggle defaults to False (off) in the schema; legacy projects
# with no recorded bool still get the mask ON for back-compat.


class TestSnapMaskToggle:
    """The snap-mask toggle must be wired through schema, JS, and
    the server-side helpers."""

    def test_settings_schema_has_toms_snap_mask_enabled(self):
        from webui.settings_schema import SETTINGS_REGISTRY
        keys = {s.key for s in SETTINGS_REGISTRY}
        assert 'toms_snap_mask_enabled' in keys, (
            "SETTINGS_REGISTRY must include toms_snap_mask_enabled "
            "(BOOL toggle) so the WebUI Settings form and CLI expose "
            "the snap-mask on/off switch"
        )

    def test_settings_schema_snap_mask_enabled_default_is_false(self):
        from webui.settings_schema import SETTINGS_REGISTRY
        toggle = next(
            s for s in SETTINGS_REGISTRY if s.key == 'toms_snap_mask_enabled'
        )
        assert toggle.type.value == 'bool', (
            f"toms_snap_mask_enabled must be BOOL, got {toggle.type!r}"
        )
        assert toggle.default is False, (
            f"toms_snap_mask_enabled default must be False (off) — "
            f"the old default 0.001 was a one-way ratchet that hid "
            f"spectral events with no recovery path. Got: "
            f"{toggle.default!r}"
        )
        assert toggle.yaml_path == ['toms', 'snap_mask_enabled'], (
            f"yaml_path must be ['toms', 'snap_mask_enabled'], got "
            f"{toggle.yaml_path!r}"
        )

    def test_js_toms_config_has_toggle_entry(self, threshold_tuning_js_text):
        """STEM_SLIDER_CONFIGS.toms must contain a toggle entry with
        key='snap_mask_enabled' and type='toggle'."""
        m = re.search(
            r"STEM_SLIDER_CONFIGS\s*=\s*\{(.*?)\n\};",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        m_toms = re.search(
            r"toms:\s*\[(.*?)\],\s*(?:hihat|cymbals|kick|snare):",
            body,
            re.DOTALL,
        )
        assert m_toms is not None
        toms_block = m_toms.group(1)
        assert re.search(
            r"key:\s*'snap_mask_enabled'.*?type:\s*['\"]toggle['\"]",
            toms_block,
            re.DOTALL,
        ), (
            "toms slider config must include "
            "{ key: 'snap_mask_enabled', type: 'toggle', ... } so the "
            "buildSlidersForStem renderer treats it as a switch"
        )

    def test_js_threshold_slider_depends_on_toggle(
        self, threshold_tuning_js_text
    ):
        """The snap_mask_threshold slider entry must declare
        dependsOn: 'snap_mask_enabled' so the buildSlidersForStem
        renderer hides it when the toggle is off."""
        m = re.search(
            r"key:\s*'snap_mask_threshold',\s*[^}]*\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        config = m.group(0)
        assert re.search(
            r"dependsOn:\s*['\"]snap_mask_enabled['\"]",
            config,
        ), (
            "snap_mask_threshold slider must depend on snap_mask_enabled "
            "(dependsOn) so it auto-hides when the toggle is off"
        )

    def test_js_apply_tuning_filter_respects_toggle(
        self, threshold_tuning_js_text
    ):
        """applyTuningFilter must gate the snap-mask pass on
        params.snap_mask_enabled. The mask must NOT run when
        params.snap_mask_enabled === false. Missing key (undefined)
        means back-compat → ON (legacy behavior preserved)."""
        m = re.search(
            r"function\s+applyTuningFilter\s*\(\s*\)\s*\{(.*?)\n\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert re.search(
            r"snap_mask_enabled\s*!==\s*false",
            body,
        ), (
            "applyTuningFilter must check `params.snap_mask_enabled !== "
            "false` to decide whether to run the mask. The check "
            "skips the mask on explicit false (user turned the toggle "
            "off) and runs it on missing (legacy back-compat) or true."
        )

    def test_js_on_toggle_input_handler_exists(self, threshold_tuning_js_text):
        """The change-listener for the toggle must be wired up so
        flipping the switch updates tuningSliderValues and shows/
        hides the dependent slider."""
        assert re.search(
            r"function\s+onToggleInput\s*\(",
            threshold_tuning_js_text,
        ), (
            "threshold-tuning.js must define onToggleInput() to handle "
            "toggle change events on the snap_mask_enabled switch"
        )

    def test_js_build_config_overrides_includes_toggle(
        self, threshold_tuning_js_text
    ):
        """_buildConfigOverrides must forward snap_mask_enabled to
        the server so a Save with the toggle off is persisted."""
        m = re.search(
            r"function\s+_buildConfigOverrides\s*\([^)]*\)\s*\{(.*?)\n\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert re.search(
            r"['\"]snap_mask_enabled['\"]",
            body,
        ), (
            "_buildConfigOverrides must forward snap_mask_enabled so "
            "the server-side rebuild respects the user's toggle state"
        )

    def test_processing_shell_no_mask_when_toggle_off(self):
        """_build_events_configured must skip the mask when
        snap_mask_enabled === False, even if snap_mask_threshold is
        set. The toggle is authoritative — the threshold is the
        secondary gate, the toggle is the primary one."""
        from stems_to_midi.processing_shell import _build_events_configured

        spectral_onset_data = [
            {'time': 1.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.0},     # Would be filtered at threshold=0.05
        ]
        config = {'toms': {
            'snap_mask_enabled': False,
            'snap_mask_threshold': 0.05,
        }}
        result = _build_events_configured(
            all_onset_data=[],
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method='spectral',
            stem_type='toms',
            config=config,
        )
        assert result[0]['status'] == 'KEPT', (
            "snap_mask_enabled=False must skip the mask regardless of "
            "threshold. Got: {result[0]['status']!r}"
        )

    def test_processing_shell_back_compat_no_toggle_means_on(self):
        """A legacy config with snap_mask_threshold set but no
        snap_mask_enabled key must still apply the mask (back-compat
        with projects saved before the toggle existed)."""
        from stems_to_midi.processing_shell import _build_events_configured

        spectral_onset_data = [
            {'time': 1.0, 'strength': 1.0, 'band_powers': [1, 1, 1, 1, 1],
             'band_max_idx': 0, 'band_max_ratio': 5.0, 'band_delta': 0.5,
             'snap_delta': 0.0},
        ]
        # Only the threshold, no enabled key — legacy.
        config = {'toms': {'snap_mask_threshold': 0.05}}
        result = _build_events_configured(
            all_onset_data=[],
            spectral_onset_data=spectral_onset_data,
            midi_events=[],
            detection_method='spectral',
            stem_type='toms',
            config=config,
        )
        assert result[0]['status'] == 'FILTERED', (
            "Legacy config (threshold set, no enabled bool) must "
            "default the toggle to ON for back-compat"
        )

    def test_rebuild_core_no_mask_when_toggle_off(self):
        """rebuild_core._apply_snap_mask must skip the mask when
        snap_mask_enabled === False."""
        from stems_to_midi.rebuild_core import _apply_snap_mask

        events = [
            {'time': 1.0, 'status': 'KEPT', 'snap_delta': 0.0},
        ]
        config = {'toms': {
            'snap_mask_enabled': False,
            'snap_mask_threshold': 0.05,
        }}
        _apply_snap_mask(events, config, 'toms')
        assert events[0]['status'] == 'KEPT', (
            "snap_mask_enabled=False must skip the mask in rebuild path"
        )

    def test_rebuild_core_back_compat_no_toggle_means_on(self):
        """rebuild_core._apply_snap_mask back-compat: missing bool
        defaults to ON, just like processing_shell."""
        from stems_to_midi.rebuild_core import _apply_snap_mask

        events = [
            {'time': 1.0, 'status': 'KEPT', 'snap_delta': 0.0},
        ]
        # No enabled key — legacy.
        config = {'toms': {'snap_mask_threshold': 0.05}}
        _apply_snap_mask(events, config, 'toms')
        assert events[0]['status'] == 'FILTERED', (
            "Legacy config (threshold set, no enabled bool) must default "
            "the toggle to ON for back-compat in rebuild path"
        )
