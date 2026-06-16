"""
Tests for the toms sidecar's 2026-06-10 spectrogram filter replacements.

The previous test file (test_snap_delta_mask.py) covered:
  - snap_mask_enabled / snap_mask_threshold (the snap-delta mask)
  - advanced_filter_enabled / advanced_min_snap_delta /
    advanced_snap_ring_threshold / advanced_snap_ring_direction /
    advanced_filter_high_strength (the 3-stage advanced filter)

Both were replaced on 2026-06-10 with a simpler pair:
  - show_only_snap_events:   toggle. When on, drop spectral events
                             with snap_delta <= 0 (the wire-tail /
                             decay kill switch).
  - band_max_ratio_max:      slider. When > 0, drop spectral events
                             with band_max_ratio strictly greater
                             than the value. 0 = Off / Disabled
                             (a no-op).

Why the replacement: the old strength field was
``min(1.0, max(0.0, band_max_ratio/10))`` — a clamp-to-1.0
normalization that collapsed every event with band_max_ratio >= 10
to the same value, masking real differences (e.g. a real hit at
18.99 vs an FP at 459.12 both reported as strength=1.0). The new
band_max_ratio_max slider reads the RAW band_max_ratio directly so
the user can express that difference.

What this test file covers:

  1. Static JS structure: the new toggle + slider are registered in
     STEM_SLIDER_CONFIGS.toms with documented keys, ranges, defaults.
  2. Static JS structure: the applyShowOnlySnapEvents and
     applyBandMaxRatioMax functions are defined and wired into
     applyTuningFilter with the right stem gate.
  3. Node-based behavioral tests: the functions actually filter
     events the way the tooltip/help-text claims.
  4. The _buildConfigOverrides function forwards both new keys so
     the server-side rebuild sees the same values the user sees in
     the tuning panel.
  5. The settings schema registers both new keys.
  6. The server-side rebuild_core._apply_show_only_snap_events and
     rebuild_core._apply_band_max_ratio_max functions exist with
     matching semantics.

We keep the file named test_snap_delta_mask.py for the build cache
key (renaming would force a full re-collection). The contents are
intentionally rewritten from scratch for the new filter set.
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


def _toms_block(threshold_tuning_js_text: str) -> str:
    """Pull the `toms: [ ... ]` block out of STEM_SLIDER_CONFIGS so
    the slider-config tests can inspect it in isolation."""
    m = re.search(
        r"STEM_SLIDER_CONFIGS\s*=\s*\{(.*?)\n\};",
        threshold_tuning_js_text,
        re.DOTALL,
    )
    assert m is not None, "could not locate STEM_SLIDER_CONFIGS block"
    body = m.group(1)
    m_toms = re.search(
        r"toms:\s*\[(.*?)\],\s*(?:hihat|cymbals|kick|snare):",
        body,
        re.DOTALL,
    )
    assert m_toms is not None, "could not locate toms: [...] block"
    return m_toms.group(1)


# ─── 1. Static JS structure: new toggle + slider are registered ──────────

class TestTomsSpectrogramFiltersSliderConfig:
    """The 2026-06-10 replacement filter set must be registered in
    STEM_SLIDER_CONFIGS.toms with documented keys, types, defaults."""

    def test_toms_has_show_only_snap_events_toggle(self, threshold_tuning_js_text):
        """The 'Show Only Snap Events' toggle must be a toggle-type
        slider with fallback false (off by default)."""
        toms_block = _toms_block(threshold_tuning_js_text)
        m = re.search(
            r"key:\s*'show_only_snap_events'.*?type:\s*['\"]toggle['\"]",
            toms_block,
            re.DOTALL,
        )
        assert m is not None, (
            "STEM_SLIDER_CONFIGS.toms must include a toggle for "
            "{ key: 'show_only_snap_events', type: 'toggle', ... }"
        )

    def test_show_only_snap_events_default_is_off(self, threshold_tuning_js_text):
        """The toggle must default to false — opt-in, like the old
        snap_mask_enabled toggle was. Snap-zero events are kept by
        default; the user has to actively enable the filter."""
        m = re.search(
            r"key:\s*'show_only_snap_events',\s*[^}]*fallback:\s*false",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, (
            "show_only_snap_events fallback must be false "
            "(off by default, opt-in)."
        )

    def test_toms_has_band_max_ratio_max_slider(self, threshold_tuning_js_text):
        """The 'Filter Events with Top/2nd Ratio Greater Than' slider
        must be a range slider with min=0 (the 'Off' sentinel)."""
        m = re.search(
            r"key:\s*'band_max_ratio_max'",
            threshold_tuning_js_text,
        )
        assert m is not None, (
            "STEM_SLIDER_CONFIGS.toms must include a slider with "
            "key='band_max_ratio_max' so the toms tuning panel shows "
            "the band_max_ratio_max ratio slider."
        )

    def test_band_max_ratio_max_min_is_zero(self, threshold_tuning_js_text):
        """The slider min must be 0 — 0 is the 'Off / Disabled'
        sentinel that the filter treats as a no-op. The UI shows
        'Off' in the value display when the slider is at 0."""
        m = re.search(
            r"key:\s*'band_max_ratio_max',\s*[^}]*\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        config = m.group(0)
        assert re.search(r"min:\s*0\b", config), (
            f"band_max_ratio_max min must be 0 (the 'Off' sentinel). "
            f"Got: {config!r}"
        )
        assert re.search(r"fallback:\s*0\b", config), (
            f"band_max_ratio_max fallback must be 0 (the filter is "
            f"disabled by default — the user opts in). Got: {config!r}"
        )

    def test_band_max_ratio_max_is_not_classification(self, threshold_tuning_js_text):
        """The slider must NOT be marked as a classification slider
        — the band_max_ratio filter is a pure client-side filter, it
        does not need a server-side reclassify call."""
        m = re.search(
            r"key:\s*'band_max_ratio_max',\s*[^}]*\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        config = m.group(0)
        assert "classification: true" not in config, (
            "band_max_ratio_max must not be a classification slider."
        )


# ─── 2. Static JS structure: filter functions exist + are wired in ─────

class TestTomsSpectrogramFilterFunctions:
    """The 2026-06-10 replacement filter functions must exist in
    threshold-tuning.js and be wired into applyTuningFilter."""

    def test_apply_show_only_snap_events_function_exists(self, threshold_tuning_js_text):
        """applyShowOnlySnapEvents(events) must be defined."""
        m = re.search(
            r"function\s+applyShowOnlySnapEvents\s*\(\s*events\s*\)",
            threshold_tuning_js_text,
        )
        assert m is not None, (
            "expected `function applyShowOnlySnapEvents(events)` in "
            "threshold-tuning.js. The function is the 4th pass in "
            "applyTuningFilter() for the toms spectrogram."
        )

    def test_apply_band_max_ratio_max_function_exists(self, threshold_tuning_js_text):
        """applyBandMaxRatioMax(events, threshold) must be defined."""
        m = re.search(
            r"function\s+applyBandMaxRatioMax\s*\(\s*events\s*,\s*threshold\s*\)",
            threshold_tuning_js_text,
        )
        assert m is not None, (
            "expected `function applyBandMaxRatioMax(events, threshold)` "
            "in threshold-tuning.js. The function is the 5th pass in "
            "applyTuningFilter() for the toms spectrogram."
        )

    def test_apply_show_only_snap_events_filters_zero_snap(self, threshold_tuning_js_text):
        """The function must mark KEPT spectral events with snap_delta
        <= 0 as FILTERED. snap_delta == 0 is the classic wire-tail /
        decay signature — the function should drop it."""
        fn = self._extract(threshold_tuning_js_text, 'applyShowOnlySnapEvents')
        assert "snap_delta" in fn, (
            "applyShowOnlySnapEvents must read event.snap_delta "
            "to decide which events to filter."
        )
        assert re.search(r"sd\s*<=\s*0|snap_delta\s*<=\s*0", fn), (
            "applyShowOnlySnapEvents must use the inclusive <= 0 "
            "comparison: events with snap_delta == 0 are filtered."
        )

    def test_apply_band_max_ratio_max_uses_strict_greater(self, threshold_tuning_js_text):
        """The function must use STRICT > (not >=) so the threshold
        value itself is a 'keep' boundary, not a 'filter' boundary.
        This matches the user spec: 'Filter Events with Top/2nd
        Ratio GREATER THAN [value]'."""
        fn = self._extract(threshold_tuning_js_text, 'applyBandMaxRatioMax')
        assert "band_max_ratio" in fn, (
            "applyBandMaxRatioMax must read event.band_max_ratio."
        )
        # Strict > is the user-facing semantic. Make sure both
        # the right form is present and the wrong form isn't.
        assert re.search(r"ratio\s*>\s*threshold", fn), (
            "applyBandMaxRatioMax must use strict '>' comparison: "
            "events with band_max_ratio > threshold are filtered. "
            "Threshold value itself is the keep boundary."
        )
        assert "ratio >= threshold" not in fn, (
            "applyBandMaxRatioMax must use strict '>'. The '>=' "
            "form would filter the threshold value itself, which "
            "isn't what the user spec says."
        )

    def _extract(self, js_text: str, name: str) -> str:
        """Pull a named function body out of threshold-tuning.js,
        preserving the original signature so any args used inside
        the body are still bound at call time."""
        m = re.search(
            rf"function\s+{name}\s*\(([^)]*)\)\s*\{{(.*?)\n\}}\n",
            js_text,
            re.DOTALL,
        )
        if m is None:
            raise AssertionError(f"could not extract {name} body")
        args = m.group(1)
        body = m.group(2)
        return f"function {name}({args}) {{\n{body}\n}}"

    def test_filters_wired_into_applyTuningFilter(self, threshold_tuning_js_text):
        """applyTuningFilter must call both new functions, gated on
        stemType === 'toms' (the new filters only act on spectral
        events which are only present in the toms stem)."""
        m = re.search(
            r"function\s+applyTuningFilter\s*\(\s*\)\s*\{(.*?)\n\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate applyTuningFilter body"
        body = m.group(1)
        assert "applyShowOnlySnapEvents" in body, (
            "applyTuningFilter must call applyShowOnlySnapEvents."
        )
        assert "applyBandMaxRatioMax" in body, (
            "applyTuningFilter must call applyBandMaxRatioMax."
        )
        assert "show_only_snap_events" in body, (
            "applyTuningFilter must read the show_only_snap_events "
            "slider value."
        )
        assert "band_max_ratio_max" in body, (
            "applyTuningFilter must read the band_max_ratio_max "
            "slider value."
        )

    def test_band_max_ratio_filter_is_noop_at_zero(self, threshold_tuning_js_text):
        """When band_max_ratio_max is 0 (the 'Off' sentinel), the
        filter must NOT be invoked — it's a no-op. The slider
        having min=0 + fallback=0 means the user has to actively
        drag it above 0 to enable the filter."""
        m = re.search(
            r"function\s+applyTuningFilter\s*\(\s*\)\s*\{(.*?)\n\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        # The branch must be guarded with `> 0` (not `!= null`).
        # A `!= null` guard would still call applyBandMaxRatioMax
        # at value=0, which would filter nothing (since `ratio > 0`
        # is true for all real events) but it's wasteful AND
        # confusing — the user has explicitly said 0 means Off.
        assert re.search(
            r"ratioMax\s*!=\s*null[^&]*&&\s*ratioMax\s*>\s*0",
            body,
        ), (
            "The band_max_ratio filter pass must be gated on "
            "`ratioMax > 0`, not just `ratioMax != null`. The "
            "slider's 0 position is the 'Off / Disabled' sentinel "
            "— calling the filter at 0 would be a wasted no-op "
            "that obscures the user's intent."
        )


# ─── 2b. PGA filter functions (2026-06-15) ─────────────────────────────

class TestPgaFilterFunctions:
    """The 2 PGA filters (pga_min_prominence, min_decay_col_min_db)
    must exist in threshold-tuning.js and be wired into
    applyTuningFilter. This is the JS mirror of the Python
    apply_pga_prominence_filter and apply_pga_decay_col_min_filter
    in stems_to_midi.pga_event_builder — both consume the
    same filter registry (stems_to_midi/filter_registry.json).

    Bug fix (2026-06-15): the original WebUI only had
    applyPgaProminenceFilter. When the registry refactor
    added the min_decay_col_min_db slider to the toms
    STEM_SLIDER_CONFIGS, the filter function for it was
    missing — moving the slider had no effect on the live
    waveform preview, and the slider value reset to the
    default -80 on every panel open. This test class
    locks both functions exist and are wired.
    """

    def test_apply_pga_prominence_filter_function_exists(self, threshold_tuning_js_text):
        m = re.search(
            r"function\s+applyPgaProminenceFilter\s*\(\s*events\s*,\s*threshold",
            threshold_tuning_js_text,
        )
        assert m is not None, (
            "expected `function applyPgaProminenceFilter(events, "
            "threshold, disabledIds)` in threshold-tuning.js"
        )

    def test_apply_pga_decay_col_min_filter_function_exists(self, threshold_tuning_js_text):
        m = re.search(
            r"function\s+applyPgaDecayColMinFilter\s*\(\s*events\s*,\s*threshold",
            threshold_tuning_js_text,
        )
        assert m is not None, (
            "expected `function applyPgaDecayColMinFilter(events, "
            "threshold, disabledIds)` in threshold-tuning.js. This "
            "is the JS mirror of the Python "
            "apply_pga_decay_col_min_filter; both consume the same "
            "filter registry. Without it, the min_decay_col_min_db "
            "slider has no effect on the live waveform preview."
        )

    def test_pga_prominence_wired_into_apply_tuning_filter(self, threshold_tuning_js_text):
        """applyPgaProminenceFilter must be called from within
        applyTuningFilter when the stem is toms."""
        m = re.search(
            r"function\s+applyTuningFilter\s*\(\s*\)\s*\{(.*?)\n\}\n",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate applyTuningFilter block"
        body = m.group(1)
        assert "applyPgaProminenceFilter" in body, (
            "applyTuningFilter must call applyPgaProminenceFilter "
            "for the toms stem (the prominence filter is the first "
            "PGA pass)."
        )

    def test_pga_decay_col_min_wired_into_apply_tuning_filter(self, threshold_tuning_js_text):
        """applyPgaDecayColMinFilter must be called from within
        applyTuningFilter when the stem is toms. Without this, the
        min_decay_col_min_db slider has no live-preview effect."""
        m = re.search(
            r"function\s+applyTuningFilter\s*\(\s*\)\s*\{(.*?)\n\}\n",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate applyTuningFilter block"
        body = m.group(1)
        assert "applyPgaDecayColMinFilter" in body, (
            "applyTuningFilter must call applyPgaDecayColMinFilter "
            "for the toms stem (the decay_col_min filter is the "
            "second PGA pass, layered on top of the prominence "
            "filter). Without this wiring, the min_decay_col_min_db "
            "slider had no effect on the live waveform preview — "
            "the bug the user reported on 2026-06-15."
        )

    def test_decay_col_min_filter_uses_field_decay_col_min_median_db(
        self, threshold_tuning_js_text
    ):
        """The function must read event.decay_col_min_median_db
        (the field the detector stamps via
        compute_high_res_decay_signature), not some other field."""
        m = re.search(
            r"function\s+applyPgaDecayColMinFilter\s*\([^)]*\)\s*\{(.*?)\n\}\n",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "decay_col_min_median_db" in body, (
            "applyPgaDecayColMinFilter must read "
            "event.decay_col_min_median_db (the field the detector "
            "stamps). Using a different field would silently miss "
            "all events."
        )

    def test_decay_col_min_filter_updates_pga_filter_config(
        self, threshold_tuning_js_text
    ):
        """The function must update pga_filter_config.min_decay_col_min_db
        so the tooltip shows the live threshold (matches the
        prominence filter's behavior with pga_min_prominence)."""
        m = re.search(
            r"function\s+applyPgaDecayColMinFilter\s*\([^)]*\)\s*\{(.*?)\n\}\n",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "min_decay_col_min_db" in body, (
            "applyPgaDecayColMinFilter must update "
            "ev.pga_filter_config.min_decay_col_min_db so the tooltip "
            "shows the live threshold (matches the prominence "
            "filter's pga_min_prominence update)."
        )


# ─── 3. Node-based behavioral tests: functions actually work ────────────

class TestTomsSpectrogramFilterBehavior:
    """Behavioral tests for the 2026-06-10 filter functions. Eval'd
    inside a minimal browser shim so the JS runs against a real JS
    engine. Skipped when node is not available."""

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

    def _extract(self, js_text: str, name: str) -> str:
        """Extract a function from threshold-tuning.js preserving its
        original signature (so behavioral tests that call
        ``applyBandMaxRatioMax(events, 100)`` don't fail with
        ``ReferenceError: threshold is not defined``)."""
        m = re.search(
            rf"function\s+{name}\s*\(([^)]*)\)\s*\{{(.*?)\n\}}\n",
            js_text,
            re.DOTALL,
        )
        if m is None:
            raise AssertionError(f"could not extract {name} body")
        # Keep the original argument list intact.
        args = m.group(1)
        body = m.group(2)
        return f"function {name}({args}) {{\n{body}\n}}"

    # ---- applyShowOnlySnapEvents ----

    def test_snap_only_drops_zero_snap_kept_events(
        self, node_available, threshold_tuning_js_text
    ):
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract(threshold_tuning_js_text, 'applyShowOnlySnapEvents')
        script = f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', method: 'spectral', snap_delta: 0.0 }},
                {{ time: 2.0, status: 'KEPT', method: 'spectral', snap_delta: 0.5 }},
                {{ time: 3.0, status: 'KEPT', method: 'spectral', snap_delta: 1.0 }},
            ];
            applyShowOnlySnapEvents(events);
            console.log(JSON.stringify(events.map(e => ({{t: e.time, s: e.status}}))));
        """
        out = self._run_node(script)
        import json
        result = json.loads(out.strip())
        # t=1.0 (snap=0) → FILTERED, t=2.0 (snap=0.5) → KEPT, t=3.0 (snap=1) → KEPT
        assert result[0]['s'] == 'FILTERED', f"snap=0 must be filtered, got {result[0]}"
        assert result[1]['s'] == 'KEPT', f"snap=0.5 must be kept, got {result[1]}"
        assert result[2]['s'] == 'KEPT', f"snap=1.0 must be kept, got {result[2]}"

    def test_snap_only_leaves_null_snap_filtered(
        self, node_available, threshold_tuning_js_text
    ):
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract(threshold_tuning_js_text, 'applyShowOnlySnapEvents')
        script = f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', method: 'spectral', snap_delta: null }},
            ];
            applyShowOnlySnapEvents(events);
            console.log(events[0].status);
        """
        out = self._run_node(script).strip()
        assert out == 'FILTERED', (
            f"null snap_delta must be filtered (no broadband attack "
            f"signal). Got: {out}"
        )

    def test_snap_only_does_not_touch_energy_events(
        self, node_available, threshold_tuning_js_text
    ):
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract(threshold_tuning_js_text, 'applyShowOnlySnapEvents')
        script = f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', method: 'energy' }},
                {{ time: 2.0, status: 'KEPT', method: 'spectral', snap_delta: 0 }},
            ];
            applyShowOnlySnapEvents(events);
            console.log(events[0].status, events[1].status);
        """
        out = self._run_node(script).strip()
        e0_status, e1_status = out.split(' ')
        assert e0_status == 'KEPT', (
            f"energy event must be untouched. Got: {e0_status}"
        )
        assert e1_status == 'FILTERED', (
            f"spectral event with snap=0 must be filtered. Got: {e1_status}"
        )

    def test_snap_only_does_not_touch_already_filtered(
        self, node_available, threshold_tuning_js_text
    ):
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract(threshold_tuning_js_text, 'applyShowOnlySnapEvents')
        script = f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'FILTERED', method: 'spectral', snap_delta: 0.5 }},
            ];
            applyShowOnlySnapEvents(events);
            console.log(events[0].status);
        """
        out = self._run_node(script).strip()
        assert out == 'FILTERED', (
            f"already-FILTERED event must stay FILTERED. Got: {out}"
        )

    # ---- applyBandMaxRatioMax ----

    def test_ratio_max_drops_above_threshold(
        self, node_available, threshold_tuning_js_text
    ):
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract(threshold_tuning_js_text, 'applyBandMaxRatioMax')
        script = f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', method: 'spectral', band_max_ratio: 18.99 }},
                {{ time: 2.0, status: 'KEPT', method: 'spectral', band_max_ratio: 100.0 }},
                {{ time: 3.0, status: 'KEPT', method: 'spectral', band_max_ratio: 459.12 }},
            ];
            applyBandMaxRatioMax(events, 100);
            console.log(JSON.stringify(events.map(e => ({{t: e.time, s: e.status, r: e.band_max_ratio}}))));
        """
        out = self._run_node(script)
        import json
        result = json.loads(out.strip())
        # 18.99 < 100 → KEPT, 100 == 100 → KEPT (strict >), 459.12 > 100 → FILTERED
        assert result[0]['s'] == 'KEPT', f"ratio 18.99 < 100 must be kept, got {result[0]}"
        assert result[1]['s'] == 'KEPT', f"ratio 100 (== threshold) must be kept, got {result[1]}"
        assert result[2]['s'] == 'FILTERED', f"ratio 459.12 > 100 must be filtered, got {result[2]}"

    def test_ratio_max_keeps_threshold_value(
        self, node_available, threshold_tuning_js_text
    ):
        """The threshold value itself is the 'keep' boundary — events
        with band_max_ratio EXACTLY equal to the threshold are kept.
        This matches the user spec: 'Filter Events with Top/2nd Ratio
        GREATER THAN [value]'."""
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract(threshold_tuning_js_text, 'applyBandMaxRatioMax')
        script = f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', method: 'spectral', band_max_ratio: 20.0 }},
            ];
            applyBandMaxRatioMax(events, 20.0);
            console.log(events[0].status);
        """
        out = self._run_node(script).strip()
        assert out == 'KEPT', (
            f"band_max_ratio == threshold must be KEPT (strict >). "
            f"Got: {out}"
        )

    def test_ratio_max_does_not_touch_energy_events(
        self, node_available, threshold_tuning_js_text
    ):
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract(threshold_tuning_js_text, 'applyBandMaxRatioMax')
        script = f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', method: 'energy' }},
            ];
            applyBandMaxRatioMax(events, 1.0);
            console.log(events[0].status);
        """
        out = self._run_node(script).strip()
        assert out == 'KEPT', (
            f"energy event (no band_max_ratio) must be untouched. "
            f"Got: {out}"
        )

    def test_ratio_max_does_not_touch_null_ratio(
        self, node_available, threshold_tuning_js_text
    ):
        if not node_available:
            pytest.skip('node not on PATH')
        fn = self._extract(threshold_tuning_js_text, 'applyBandMaxRatioMax')
        script = f"""
            {fn}
            const events = [
                {{ time: 1.0, status: 'KEPT', method: 'spectral', band_max_ratio: null }},
            ];
            applyBandMaxRatioMax(events, 1.0);
            console.log(events[0].status);
        """
        out = self._run_node(script).strip()
        assert out == 'KEPT', (
            f"null band_max_ratio must be untouched. Got: {out}"
        )


# ─── 4. _buildConfigOverrides forwards the new keys ────────────────────

class TestConfigOverridesForwardsNewKeys:
    """_buildConfigOverrides must include both new keys so the
    server-side rebuild sees the user's saved values."""

    def test_show_only_snap_events_in_per_stem_overrides(self, threshold_tuning_js_text):
        m = re.search(
            r"function\s+_buildConfigOverrides\s*\([^)]*\)\s*\{(.*?)\n\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate _buildConfigOverrides body"
        body = m.group(1)
        assert re.search(
            r"['\"]show_only_snap_events['\"]",
            body,
        ), (
            "_buildConfigOverrides must include 'show_only_snap_events' "
            "in the per-stem key list."
        )

    def test_band_max_ratio_max_in_per_stem_overrides(self, threshold_tuning_js_text):
        m = re.search(
            r"function\s+_buildConfigOverrides\s*\([^)]*\)\s*\{(.*?)\n\}",
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert re.search(
            r"['\"]band_max_ratio_max['\"]",
            body,
        ), (
            "_buildConfigOverrides must include 'band_max_ratio_max' "
            "in the per-stem key list."
        )


# ─── 5. Settings schema registers the new keys ─────────────────────────

class TestSettingsSchemaNewKeys:
    """The settings schema must include both new keys so the project
    YAML's midiconfig.yaml section round-trips correctly."""

    def test_settings_schema_has_toms_show_only_snap_events(self):
        from webui.settings_schema import SETTINGS_REGISTRY
        keys = {s.key for s in SETTINGS_REGISTRY}
        assert 'toms_show_only_snap_events' in keys, (
            "SETTINGS_REGISTRY must include toms_show_only_snap_events "
            "so the new toggle is round-trippable through the YAML."
        )

    def test_show_only_snap_events_default_is_false(self):
        from webui.settings_schema import SETTINGS_REGISTRY
        s = next(x for x in SETTINGS_REGISTRY if x.key == 'toms_show_only_snap_events')
        assert s.default is False, (
            f"toms_show_only_snap_events default must be False "
            f"(opt-in). Got: {s.default!r}"
        )
        assert s.yaml_path == ['toms', 'show_only_snap_events'], (
            f"yaml_path must be ['toms', 'show_only_snap_events']. "
            f"Got: {s.yaml_path!r}"
        )

    def test_settings_schema_has_toms_band_max_ratio_max(self):
        from webui.settings_schema import SETTINGS_REGISTRY
        keys = {s.key for s in SETTINGS_REGISTRY}
        assert 'toms_band_max_ratio_max' in keys, (
            "SETTINGS_REGISTRY must include toms_band_max_ratio_max "
            "so the new ratio slider is round-trippable."
        )

    def test_band_max_ratio_max_default_is_zero(self):
        from webui.settings_schema import SETTINGS_REGISTRY
        s = next(x for x in SETTINGS_REGISTRY if x.key == 'toms_band_max_ratio_max')
        assert s.default == 0.0, (
            f"toms_band_max_ratio_max default must be 0.0 "
            f"(the 'Off / Disabled' sentinel). Got: {s.default!r}"
        )
        assert s.yaml_path == ['toms', 'band_max_ratio_max'], (
            f"yaml_path must be ['toms', 'band_max_ratio_max']. "
            f"Got: {s.yaml_path!r}"
        )


# ─── 6. Server-side mirror functions exist with matching semantics ─────

class TestServerSideFilterMirrors:
    """The server-side rebuild path must mirror the new JS filters
    so the saved MIDI matches what the tuning panel shows."""

    def test_rebuild_core_show_only_snap_events_exists(self):
        from stems_to_midi.rebuild_core import _apply_show_only_snap_events
        assert callable(_apply_show_only_snap_events)

    def test_show_only_snap_events_default_off(self):
        """Missing key → off (no-op). The schema default is False
        so this matches the user-facing UI default."""
        from stems_to_midi.rebuild_core import _apply_show_only_snap_events
        events = [
            {'time': 1.0, 'status': 'KEPT', 'method': 'spectral', 'snap_delta': 0.0},
        ]
        _apply_show_only_snap_events(events, {}, 'toms')
        assert events[0]['status'] == 'KEPT', (
            "show_only_snap_events with no key in config must be a no-op."
        )

    def test_show_only_snap_events_filters_zero(self):
        from stems_to_midi.rebuild_core import _apply_show_only_snap_events
        events = [
            {'time': 1.0, 'status': 'KEPT', 'method': 'spectral', 'snap_delta': 0.0},
            {'time': 2.0, 'status': 'KEPT', 'method': 'spectral', 'snap_delta': 0.5},
        ]
        _apply_show_only_snap_events(
            events, {'toms': {'show_only_snap_events': True}}, 'toms'
        )
        assert events[0]['status'] == 'FILTERED', (
            "snap_delta=0 with toggle on must be FILTERED."
        )
        assert events[1]['status'] == 'KEPT', (
            "snap_delta>0 with toggle on must be KEPT."
        )

    def test_show_only_snap_events_idempotent(self):
        """Re-running on already-filtered events must not change them."""
        from stems_to_midi.rebuild_core import _apply_show_only_snap_events
        events = [
            {'time': 1.0, 'status': 'FILTERED', 'method': 'spectral', 'snap_delta': 0.5},
        ]
        _apply_show_only_snap_events(
            events, {'toms': {'show_only_snap_events': True}}, 'toms'
        )
        assert events[0]['status'] == 'FILTERED', (
            "Already-FILTERED event must stay FILTERED (idempotent)."
        )

    def test_rebuild_core_band_max_ratio_max_exists(self):
        from stems_to_midi.rebuild_core import _apply_band_max_ratio_max
        assert callable(_apply_band_max_ratio_max)

    def test_band_max_ratio_max_zero_is_noop(self):
        from stems_to_midi.rebuild_core import _apply_band_max_ratio_max
        events = [
            {'time': 1.0, 'status': 'KEPT', 'method': 'spectral', 'band_max_ratio': 459.12},
        ]
        # 0 = Off / Disabled
        _apply_band_max_ratio_max(
            events, {'toms': {'band_max_ratio_max': 0}}, 'toms'
        )
        assert events[0]['status'] == 'KEPT', (
            "band_max_ratio_max=0 must be a no-op (the 'Off' sentinel)."
        )

    def test_band_max_ratio_max_filters_above_threshold(self):
        from stems_to_midi.rebuild_core import _apply_band_max_ratio_max
        events = [
            {'time': 1.0, 'status': 'KEPT', 'method': 'spectral', 'band_max_ratio': 18.99},
            {'time': 2.0, 'status': 'KEPT', 'method': 'spectral', 'band_max_ratio': 100.0},
            {'time': 3.0, 'status': 'KEPT', 'method': 'spectral', 'band_max_ratio': 459.12},
        ]
        _apply_band_max_ratio_max(
            events, {'toms': {'band_max_ratio_max': 100.0}}, 'toms'
        )
        assert events[0]['status'] == 'KEPT'
        assert events[1]['status'] == 'KEPT', (
            "ratio == threshold must be KEPT (strict >)."
        )
        assert events[2]['status'] == 'FILTERED', (
            "ratio > threshold must be FILTERED."
        )

    def test_band_max_ratio_max_respects_overrides(self):
        """The override flag wins — a manually-kept event must not be
        filtered by the band_max_ratio_max ceiling."""
        from stems_to_midi.rebuild_core import _apply_band_max_ratio_max
        events = [
            {'time': 1.0, 'status': 'KEPT', 'override': True,
             'method': 'spectral', 'band_max_ratio': 1000.0},
        ]
        _apply_band_max_ratio_max(
            events, {'toms': {'band_max_ratio_max': 100.0}}, 'toms'
        )
        assert events[0]['status'] == 'KEPT', (
            "Override flag must win — manually-kept event must survive."
        )

    def test_band_max_ratio_max_handles_invalid_threshold(self):
        """Defensive: invalid threshold (non-numeric) must be a no-op."""
        from stems_to_midi.rebuild_core import _apply_band_max_ratio_max
        events = [
            {'time': 1.0, 'status': 'KEPT', 'method': 'spectral', 'band_max_ratio': 1000.0},
        ]
        _apply_band_max_ratio_max(
            events, {'toms': {'band_max_ratio_max': 'not a number'}}, 'toms'
        )
        assert events[0]['status'] == 'KEPT', (
            "Invalid threshold must be a no-op (defensive)."
        )

    def test_band_max_ratio_max_ignores_energy_events(self):
        from stems_to_midi.rebuild_core import _apply_band_max_ratio_max
        events = [
            {'time': 1.0, 'status': 'KEPT', 'method': 'energy'},
        ]
        _apply_band_max_ratio_max(
            events, {'toms': {'band_max_ratio_max': 1.0}}, 'toms'
        )
        assert events[0]['status'] == 'KEPT', (
            "Energy event (no band_max_ratio) must be untouched."
        )
