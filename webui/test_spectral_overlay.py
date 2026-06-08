"""
Tests for the spectral A/B-comparison overlay in the WebUI waveform viewer.

The "spectral" color path is the new magenta treatment for events whose
``method='spectral'`` field was stamped by the pipeline
(detection_method='spectral' or 'both' with the event surviving the
12ms dedup window). The overlay is gated by a per-render flag so
single-mode projects (detection_method='energy' or 'spectral' only)
keep their original color — this is the "no regression" contract from
the task spec.

The plumbing lives in two layers:

  1. ``webui/static/js/waveform.js`` — canvas drawing side. Has a
     ``getEventColor(event, spectralOverlayActive)`` helper and a
     ``hasMixedDetectionMethods(events)`` decision function. The
     color constant lives in the ``WAVEFORM_COLORS`` dict as
     ``markerSpectral: '#ec4899'``.

  2. ``stems_to_midi/processing_shell.py`` + ``stems_to_midi/midi.py``
     — pipeline side. Energy-detected onsets are stamped with
     ``method=energy_method`` (rms / peak_hold / spectral_flux).
     Spectral-survivor onsets in 'both' or 'spectral' modes are
     stamped with ``method='spectral'`` by ``_build_events_configured``.
     The ``method`` key is on the always-present list in
     ``_serialize_onset_events`` so it survives the JSON round-trip.

This file covers BOTH layers:

  * The JS layer gets string-search tests (the existing pattern in
    test_threshold_tuning.py / test_detection_method_webui.py — there
    is no JS runtime available to pytest) plus Node-based behavioral
    tests of the helper functions so the color logic has actual
    coverage. The Node tests are skipped automatically if ``node`` is
    not on PATH.
  * The pipeline layer gets a regression test that verifies the
    ``method`` field is stamped on every energy-detected onset and
    that the sidecar round-trip preserves it.
"""

import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# ─── Constants & paths ────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
WAVEFORM_JS = REPO_ROOT / 'webui' / 'static' / 'js' / 'waveform.js'
INDEX_HTML = REPO_ROOT / 'webui' / 'templates' / 'index.html'
PROCESSING_SHELL = REPO_ROOT / 'stems_to_midi' / 'processing_shell.py'
MIDI_PY = REPO_ROOT / 'stems_to_midi' / 'midi.py'

# The spectral color choice — documented here so the test failure
# message points to the contract, not just a hex string.
EXPECTED_SPECTRAL_COLOR = '#ec4899'
EXPECTED_KEPT_COLOR = '#10b981'
EXPECTED_FILTERED_COLOR = '#ef4444'


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def waveform_js_text() -> str:
    return WAVEFORM_JS.read_text()


@pytest.fixture(scope='module')
def node_available() -> bool:
    """Skip Node-based tests when the binary isn't on PATH."""
    return shutil.which('node') is not None


# ─── 1. Static JS structure: the helper exists ───────────────────────────

class TestWaveformJSColorHelper:
    """String-search tests that confirm the new color helper is wired
    into waveform.js with the right shape and the right color."""

    def test_marker_spectral_color_constant(self, waveform_js_text):
        """WAVEFORM_COLORS.markerSpectral must be the documented
        magenta (#ec4899). The task spec recommends either orange
        (#f59e0b) or magenta (#ec4899); we picked magenta because
        #f59e0b is already taken by ``markerReverbCont``."""
        match = re.search(
            r"markerSpectral:\s*'(#[0-9a-fA-F]{3,8})'",
            waveform_js_text,
        )
        assert match is not None, (
            "WAVEFORM_COLORS.markerSpectral is missing from waveform.js — "
            "this is the dedicated magenta color for spectral-detected "
            "events in the A/B-comparison overlay."
        )
        assert match.group(1).lower() == EXPECTED_SPECTRAL_COLOR, (
            f"markerSpectral is {match.group(1)}; expected {EXPECTED_SPECTRAL_COLOR}. "
            "If the color is intentional, update EXPECTED_SPECTRAL_COLOR "
            "in this test and document the change in the deliverable."
        )

    def test_get_event_color_function_exists(self, waveform_js_text):
        """The new ``getEventColor(event, spectralOverlayActive)``
        helper must be defined. Both args are required."""
        assert re.search(
            r"function\s+getEventColor\s*\(\s*event\s*,\s*spectralOverlayActive\s*\)",
            waveform_js_text,
        ), (
            "expected ``function getEventColor(event, spectralOverlayActive)`` "
            "in waveform.js. The overlay flag is required so single-mode "
            "projects can fall through to the legacy color path."
        )

    def test_has_mixed_detection_methods_function_exists(self, waveform_js_text):
        """``hasMixedDetectionMethods(events)`` must be defined. It
        is the per-render decision function for whether to activate
        the spectral overlay."""
        assert re.search(
            r"function\s+hasMixedDetectionMethods\s*\(\s*events\s*\)",
            waveform_js_text,
        ), (
            "expected ``function hasMixedDetectionMethods(events)`` "
            "in waveform.js. The function is the single source of truth "
            "for whether the spectral overlay should be active."
        )

    def test_draw_event_bars_signature_includes_overlay_flag(self, waveform_js_text):
        """``drawEventBars`` must accept the new ``spectralOverlayActive``
        argument so it can pass it to ``getEventColor``."""
        # Find the function declaration with all 8 args.
        match = re.search(
            r"function\s+drawEventBars\s*\(\s*[^)]*spectralOverlayActive\s*\)",
            waveform_js_text,
        )
        assert match is not None, (
            "drawEventBars must accept a spectralOverlayActive argument. "
            "It is the final arg in the signature, and the renderer "
            "passes false for the sensitive background layer and the "
            "panel-level flag for the main layer."
        )

    def test_draw_event_bars_calls_get_event_color(self, waveform_js_text):
        """``drawEventBars`` must call the new ``getEventColor`` helper
        instead of the old ``getMarkerColor`` for the bar color."""
        # Find the drawEventBars body and check it doesn't call
        # getMarkerColor directly any more.
        m = re.search(
            r"function\s+drawEventBars\s*\([^)]*\)\s*\{(.*?)\n\}",
            waveform_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate drawEventBars function body"
        body = m.group(1)
        assert 'getEventColor' in body, (
            "drawEventBars must call getEventColor(...) to resolve the bar color. "
            "The legacy getMarkerColor call should be replaced — getEventColor "
            "is the single source of truth for spectral-aware coloring."
        )
        assert 'getMarkerColor(' not in body, (
            "drawEventBars still calls getMarkerColor directly. Replace the "
            "call with getEventColor so the spectral overlay is honored."
        )

    def test_legend_has_spectral_entry(self, waveform_js_text):
        """The waveform legend must show a 'Spectral (N)' entry when
        spectral candidates are present. The user-facing text is the
        contract — without it, the magenta bars on the canvas have no
        explanation."""
        assert "Spectral (" in waveform_js_text, (
            "expected a 'Spectral (...)' legend entry string in waveform.js. "
            "The number is interpolated at render time but the prefix must "
            "be present as a string literal."
        )

    def test_tooltip_surfaces_method_field(self, waveform_js_text):
        """The hover tooltip must surface ``event.method`` so the
        user can hover any bar and see whether it was energy- or
        spectral-detected. The color → method mapping is the core
        documentation of the A/B-comparison contract."""
        # Look for the drawTooltip function body and check it
        # references event.method.
        m = re.search(
            r"function\s+drawTooltip\s*\([^)]*\)\s*\{(.*?)\n\}",
            waveform_js_text,
            re.DOTALL,
        )
        assert m is not None, "could not locate drawTooltip function body"
        body = m.group(1)
        assert 'event.method' in body or "event['method']" in body, (
            "drawTooltip must read event.method to surface the detection "
            "method on hover. Without this, the user has no way to learn "
            "the color → method mapping except by reading the legend."
        )


# ─── 2. Node-based behavioral tests: the helper actually works ──────────

class TestWaveformJSColorHelperBehavior:
    """Behavioral tests for the color helper. Eval'd inside a minimal
    browser shim so the JS runs against a real JS engine. These tests
    are skipped when ``node`` is not available."""

    SHIM = textwrap.dedent(r"""
        const document = { getElementById: () => null };
        const window = { devicePixelRatio: 1 };
        const WAVEFORM_COLORS = {
            background: '#111827', axisLine: '#374151', axisText: '#9ca3af',
            markerKept: '#10b981', markerSpectral: '#ec4899',
            markerFiltered: '#ef4444', markerReverbCont: '#f59e0b',
            markerSensitive: 'rgba(156, 163, 175, 0.3)', markerUnknown: '#6b7280',
        };
        const CLASSIFICATION_COLORS = ['#10b981', '#a855f7', '#22d3ee', '#eab308'];
        const HIHAT_OPEN_COLOR = '#f97316';
        const HIHAT_CLOSED_COLOR = '#06b6d4';
    """)

    def _run_node(self, js_source: str) -> str:
        """Eval ``js_source`` inside the shim and return stdout.
        The test script is a single self-contained Node program."""
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

    def _extract_helpers(self, waveform_js_text: str) -> str:
        """Pull out the four helper functions so we can evaluate them
        in isolation. The function-declaration regex stops at the
        legend-bar comment header which is the next top-level
        declaration."""
        start = waveform_js_text.index('function getMarkerColor(')
        # Stop at the legend-bar section header.
        stop = waveform_js_text.index('// ─── Legend Bar')
        return waveform_js_text[start:stop]

    def test_energy_kept_no_method_is_green(self, node_available, waveform_js_text):
        """An energy KEPT event with no method field (older sidecars,
        or a method we don't recognize) must still render as the
        legacy green. The overlay flag is irrelevant here."""
        if not node_available:
            pytest.skip('node not on PATH')
        helpers = self._extract_helpers(waveform_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {helpers}
            const got = getEventColor({{status: 'KEPT'}}, true);
            console.log(got);
        """))
        assert out.strip() == EXPECTED_KEPT_COLOR, (
            f"got {out.strip()!r}, expected {EXPECTED_KEPT_COLOR}. "
            "Energy KEPT events with no method should fall through to "
            "the legacy green color."
        )

    def test_spectral_method_with_overlay_on_is_magenta(self, node_available, waveform_js_text):
        """A spectral KEPT event with the overlay flag on must
        render as the dedicated magenta — this is the A/B-comparison
        contract."""
        if not node_available:
            pytest.skip('node not on PATH')
        helpers = self._extract_helpers(waveform_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {helpers}
            const got = getEventColor({{status: 'KEPT', method: 'spectral'}}, true);
            console.log(got);
        """))
        assert out.strip() == EXPECTED_SPECTRAL_COLOR, (
            f"got {out.strip()!r}, expected {EXPECTED_SPECTRAL_COLOR}. "
            "Spectral KEPT events with the overlay flag on must use the "
            "dedicated magenta color."
        )

    def test_spectral_method_with_overlay_off_stays_green(self, node_available, waveform_js_text):
        """NO-REGRESSION contract: a spectral KEPT event with the
        overlay flag OFF (single-mode spectral project) must fall
        through to the legacy green. This is the spec's 'rendering
        with method=\"spectral\" only must look the same as before'
        rule."""
        if not node_available:
            pytest.skip('node not on PATH')
        helpers = self._extract_helpers(waveform_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {helpers}
            const got = getEventColor({{status: 'KEPT', method: 'spectral'}}, false);
            console.log(got);
        """))
        assert out.strip() == EXPECTED_KEPT_COLOR, (
            f"got {out.strip()!r}, expected {EXPECTED_KEPT_COLOR}. "
            "Spectral KEPT events with the overlay flag off must NOT use "
            "the magenta color — single-mode projects must look the same "
            "as before this work."
        )

    def test_rms_method_is_green_even_with_overlay(self, node_available, waveform_js_text):
        """An rms-detected KEPT event must render green even with the
        overlay on. ``method='rms'`` is energy detection, not
        spectral detection; the spec only colorizes bare
        ``method='spectral'`` events."""
        if not node_available:
            pytest.skip('node not on PATH')
        helpers = self._extract_helpers(waveform_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {helpers}
            const got = getEventColor({{status: 'KEPT', method: 'rms'}}, true);
            console.log(got);
        """))
        assert out.strip() == EXPECTED_KEPT_COLOR, (
            f"got {out.strip()!r}, expected {EXPECTED_KEPT_COLOR}. "
            "Energy-detected events (rms, peak_hold, spectral_flux) must "
            "remain green even when the overlay is active."
        )

    def test_filtered_wins_over_method(self, node_available, waveform_js_text):
        """FILTERED events must show as red regardless of method.
        A user actively triaging a song needs the red bar to win
        visual precedence over the spectral/energy distinction."""
        if not node_available:
            pytest.skip('node not on PATH')
        helpers = self._extract_helpers(waveform_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {helpers}
            const got = getEventColor({{status: 'FILTERED', method: 'spectral'}}, true);
            console.log(got);
        """))
        assert out.strip() == EXPECTED_FILTERED_COLOR, (
            f"got {out.strip()!r}, expected {EXPECTED_FILTERED_COLOR}. "
            "FILTERED must always win over the method-based color."
        )

    def test_hihat_open_wins_over_spectral(self, node_available, waveform_js_text):
        """Hihat open/closed is a hit-type identity and beats the
        spectral method color. Open/closed is what the user looks
        for in the waveform to understand a groove."""
        if not node_available:
            pytest.skip('node not on PATH')
        helpers = self._extract_helpers(waveform_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {helpers}
            const got = getEventColor(
                {{status: 'KEPT', method: 'spectral', hihat_state: 'open'}}, true,
            );
            console.log(got);
        """))
        # HIHAT_OPEN_COLOR is #f97316 (orange), distinct from both
        # green and magenta. Assert the actual color string to lock
        # the contract.
        assert out.strip() == '#f97316', (
            f"got {out.strip()!r}, expected '#f97316' (HIHAT_OPEN_COLOR). "
            "Hihat open is a hit-type identity and must take precedence "
            "over the spectral method color."
        )

    def test_classification_color_preserved(self, node_available, waveform_js_text):
        """A KEPT event with a classification index but no spectral
        method must use the classification palette (e.g. purple for
        type 1). The new spectral code must not break this path."""
        if not node_available:
            pytest.skip('node not on PATH')
        helpers = self._extract_helpers(waveform_js_text)
        out = self._run_node(textwrap.dedent(f"""
            {helpers}
            const got = getEventColor(
                {{status: 'KEPT', method: 'rms', classification: 1}}, true,
            );
            console.log(got);
        """))
        # CLASSIFICATION_COLORS[1] = '#a855f7' (purple).
        assert out.strip() == '#a855f7', (
            f"got {out.strip()!r}, expected '#a855f7' (classification 1). "
            "Events with a classification index must use the classification "
            "palette, not the default green."
        )


# ─── 3. hasMixedDetectionMethods behavioral tests ───────────────────────

class TestHasMixedDetectionMethods:
    """Behavioral tests for the per-render decision function. The
    function determines whether the spectral overlay should be
    activated; getting this wrong breaks the no-regression contract
    for single-mode projects."""

    SHIM = textwrap.dedent(r"""
        const document = { getElementById: () => null };
        const window = { devicePixelRatio: 1 };
        const WAVEFORM_COLORS = {
            background: '#111827', markerKept: '#10b981', markerSpectral: '#ec4899',
            markerFiltered: '#ef4444', markerReverbCont: '#f59e0b',
            markerSensitive: 'rgba(156, 163, 175, 0.3)', markerUnknown: '#6b7280',
        };
        const CLASSIFICATION_COLORS = ['#10b981', '#a855f7', '#22d3ee', '#eab308'];
        const HIHAT_OPEN_COLOR = '#f97316';
        const HIHAT_CLOSED_COLOR = '#06b6d4';
    """)

    def _extract_helpers(self, waveform_js_text: str) -> str:
        start = waveform_js_text.index('function getMarkerColor(')
        stop = waveform_js_text.index('// ─── Legend Bar')
        return waveform_js_text[start:stop]

    def _run(self, waveform_js_text: str, events_js: str) -> str:
        if shutil.which('node') is None:
            pytest.skip('node not on PATH')
        helpers = self._extract_helpers(waveform_js_text)
        script = self.SHIM + "\n" + helpers + "\n" + textwrap.dedent(f"""
            const got = hasMixedDetectionMethods({events_js});
            console.log(String(got));
        """)
        result = subprocess.run(
            ['node', '-e', script],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"node exited {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result.stdout.strip()

    def test_empty_list_is_false(self, waveform_js_text):
        assert self._run(waveform_js_text, "[]") == "false"

    def test_null_is_false(self, waveform_js_text):
        assert self._run(waveform_js_text, "null") == "false"

    def test_pure_energy_is_false(self, waveform_js_text):
        """All events are rms — this is the energy-only project case
        (detection_method='energy'). Overlay must be off."""
        assert self._run(
            waveform_js_text,
            "[{method: 'rms'}, {method: 'rms'}, {method: 'peak_hold'}]",
        ) == "false"

    def test_pure_spectral_is_false(self, waveform_js_text):
        """All events are spectral — this is the spectral-only project
        case (detection_method='spectral'). Overlay must be off so
        every bar stays green (no-regression contract)."""
        assert self._run(
            waveform_js_text,
            "[{method: 'spectral'}, {method: 'spectral'}]",
        ) == "false"

    def test_mixed_list_is_true(self, waveform_js_text):
        """A list with at least one spectral event AND at least one
        non-spectral event is the 'both' mode case. Overlay must be
        on so the user can see the A/B comparison."""
        assert self._run(
            waveform_js_text,
            "[{method: 'rms'}, {method: 'spectral'}, {method: 'rms'}]",
        ) == "true"

    def test_unknown_method_counts_as_non_spectral(self, waveform_js_text):
        """Events with no method (legacy sidecars) or a method value
        that isn't 'spectral' (rms / peak_hold / spectral_flux) all
        count as non-spectral for the decision. This is what the
        always-present field fix in midi.py guarantees — legacy
        sidecars (with the new always-present field added on first
        save) will still be detected as non-spectral."""
        assert self._run(
            waveform_js_text,
            "[{status: 'KEPT'}, {method: 'spectral'}]",
        ) == "true"

    def test_list_with_nulls_does_not_crash(self, waveform_js_text):
        """Defensive: a malformed event (null) in the list must not
        crash the loop. The implementation explicitly skips nulls."""
        assert self._run(
            waveform_js_text,
            "[null, {method: 'spectral'}, {method: 'rms'}]",
        ) == "true"


# ─── 4. Pipeline side: the method field survives the sidecar round-trip ─

class TestMethodFieldPlumbing:
    """Regression tests for the pipeline-side change: every
    energy-detected onset must be stamped with its method, and the
    ``method`` field must survive the sidecar round-trip as an
    always-present key. Without this, the WebUI color helper can
    never see ``method='spectral'`` on an event and the magenta
    overlay is unreachable.

    This class is in the webui/ test directory but covers a
    stems_to_midi/ module because the color rendering is meaningless
    without the data shape change.
    """

    def test_processing_shell_stamps_method_on_onsets(self):
        """``_build_events_configured`` and the per-stem energy-detector
        call sites must stamp the method on the onset dicts so the
        serializer can carry it to JSON.

        We assert by string search: the line that loops over
        ``all_onset_data`` and writes ``onset_d['method'] = energy_method``
        must exist in the post-`filter_onsets_by_spectral` block of
        ``process_stem_to_midi``."""
        text = PROCESSING_SHELL.read_text()
        # The new loop in process_stem_to_midi (right after
        # all_onset_data = filter_result['all_onset_data']).
        pattern = (
            r"all_onset_data\s*=\s*filter_result\['all_onset_data'\][\s\S]*?"
            r"onset_d\['method'\]\s*=\s*energy_method"
        )
        assert re.search(pattern, text), (
            "processing_shell.process_stem_to_midi must stamp "
            "onset_d['method'] = energy_method on every energy-detected "
            "onset. Without this, the sidecar events_configured entries "
            "have no method field and the WebUI color helper cannot "
            "distinguish energy from spectral."
        )

    def test_sensitive_detector_also_stamps_method(self):
        """``_run_sensitive_detection`` must stamp the method on its
        ``sensitive_onset_data`` so the background-layer rendering
        stays consistent (the same color discipline applies)."""
        text = PROCESSING_SHELL.read_text()
        # Look for the loop in _run_sensitive_detection right after
        # sensitive_onset_data = filter_result.get('all_onset_data', []).
        pattern = (
            r"sensitive_onset_data\s*=\s*filter_result\.get\('all_onset_data',\s*\[\]\)[\s\S]*?"
            r"onset_d\['method'\]\s*=\s*energy_method"
        )
        assert re.search(pattern, text), (
            "processing_shell._run_sensitive_detection must stamp "
            "onset_d['method'] = energy_method on every onset. The "
            "sensitive background layer shares the color discipline "
            "with the main layer."
        )

    def test_method_in_always_present_fields(self):
        """``_serialize_onset_events`` must list ``method`` in
        ALWAYS_PRESENT_FIELDS so the field is written on every event
        (with null when missing). Without this, legacy sidecars and
        new events that haven't been re-serialized would lack the
        key, and the WebUI helper would feature-detect per record."""
        text = MIDI_PY.read_text()
        # Find the ALWAYS_PRESENT_FIELDS tuple.
        m = re.search(
            r"ALWAYS_PRESENT_FIELDS\s*=\s*\(([^)]+)\)",
            text,
        )
        assert m is not None, (
            "could not locate ALWAYS_PRESENT_FIELDS tuple in midi.py"
        )
        fields_text = m.group(1)
        assert "'method'" in fields_text or '"method"' in fields_text, (
            f"ALWAYS_PRESENT_FIELDS does not include 'method'. Current "
            f"value: {fields_text!r}. The method field must be always-present "
            "so the WebUI color helper can read it on every event."
        )

    def test_round_trip_preserves_method_via_serializer(self, tmp_path):
        """End-to-end: a synthetic sidecar with method-tagged events
        must round-trip through ``_serialize_onset_events`` with the
        method field preserved. This is the data-flow test that
        would have caught the missing-always-present bug from the
        bug-C review (2026-06)."""
        sys.path.insert(0, str(REPO_ROOT))
        from stems_to_midi.midi import _serialize_onset_events  # noqa: PLC0415

        onset_data = [
            {
                'time': 0.5, 'status': 'KEPT', 'strength': 0.9,
                'geomean': 400, 'method': 'rms',
                'pan_confidence': 0.0, 'stereo_width': 0.0, 'pitch_hz': None,
            },
            {
                'time': 1.0, 'status': 'KEPT', 'strength': 1.0,
                'geomean': 500, 'method': 'spectral',
                'pan_confidence': 0.5, 'stereo_width': 0.3, 'pitch_hz': 220.0,
            },
        ]
        serialized = _serialize_onset_events(onset_data, midi_events=[])

        # Both events must have a method key.
        for ev in serialized:
            assert 'method' in ev, (
                f"event at t={ev.get('time')} is missing 'method' — the "
                "ALWAYS_PRESENT_FIELDS contract is broken. Events: {serialized}"
            )

        methods = [ev['method'] for ev in serialized]
        assert methods == ['rms', 'spectral'], (
            f"methods did not round-trip: {methods}. Expected ['rms', 'spectral']."
        )


# ─── 5. Cache-bust version on the script tag ───────────────────────────

class TestScriptTagBump:
    """Cache-bust check: the ``<script src="...waveform.js?v=N">``
    must be bumped whenever waveform.js changes, otherwise browsers
    serve the stale cached version and the user never sees the
    magenta overlay."""

    def test_waveform_js_version_bumped(self):
        text = INDEX_HTML.read_text()
        m = re.search(
            r'<script\s+src="[^"]*waveform\.js\?v=(\d+)"',
            text,
        )
        assert m is not None, (
            "could not find <script src=\"/static/js/waveform.js?v=N\"> "
            "in templates/index.html"
        )
        version = int(m.group(1))
        # The 4 prior tasks bumped 24 → 25 → 26 → 27. The 28 bump is
        # part of this work. We don't pin to exactly 28 because other
        # work could have bumped it further, but we require it to be
        # strictly greater than 27.
        assert version > 27, (
            f"waveform.js?v={version} is too old. The cache-bust version "
            "must be bumped on every waveform.js change so browsers fetch "
            "the new code. The most recent pre-change version was v=27."
        )
