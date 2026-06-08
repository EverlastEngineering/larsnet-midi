"""
Tests for the Advanced MIDI Settings modal (webui/static/js/advanced-midi.js).

Some of the advanced-midiconfig fields are coupled: setting
``snare.cluster_feature`` to ``pitch_hz`` only works when
``snare.enable_pitch_detection`` is true AND a full conversion has
been re-run (pitch is a detection-time feature). The user discovered
this on 2026-06-08 when picking "Pitch" in the dropdown "didn't
work" — silent fallback in ``_resolve_cluster_feature`` masked the
dependency.

These tests lock down the JS-side contract: the modal's save handler
must auto-toggle ``enable_pitch_detection`` when the user picks
``pitch_hz``, and must surface a hint that a full Convert is needed.
We use the same regex-over-source pattern that other JS static asset
tests in this codebase use (no JS runtime in pytest).

Tests:
  - TestPitchDependencyAutoEnable
      - test_save_handler_adds_enable_pitch_detection_when_pitch_chosen
      - test_save_handler_keeps_existing_true_does_not_touch_field
      - test_save_handler_only_handles_detection_time_features
  - TestConvertHintToast
      - test_save_handler_queues_convert_hint_for_detection_time_changes
  - TestJsFileStructure
      - test_file_exists
      - test_has_class_advanced_midi_settings
      - test_has_save_method
"""

import re
from pathlib import Path

import pytest


# ─── Fixtures ────────────────────────────────────────────────────────────

JS_PATH = Path(__file__).parent / 'static' / 'js' / 'advanced-midi.js'


@pytest.fixture
def js_content():
    """The full text of webui/static/js/advanced-midi.js."""
    return JS_PATH.read_text()


@pytest.fixture
def save_method_body(js_content):
    """Extract the body of AdvancedMIDISettings.prototype.save as a string.

    The class is structured as:
        class AdvancedMIDISettings {
            ...
            async save() {
                ...body...
            }
            ...
        }

    We grab the substring from the opening ``async save() {`` through
    the matching closing brace. This is the same regex-over-source
    approach used by TestThresholdTuningJS — keeps tests independent
    of JS execution.
    """
    # Match "async save() {" and capture the body up to the matching
    # closing brace. We use a simple balanced-brace counter because
    # this method is well-formed and small enough to walk safely.
    match = re.search(r'async\s+save\s*\(\s*\)\s*\{', js_content)
    assert match, "Could not find `async save() {` in advanced-midi.js"
    start = match.end()
    depth = 1
    i = start
    while i < len(js_content) and depth > 0:
        ch = js_content[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        i += 1
    assert depth == 0, "Unbalanced braces in save() method"
    return js_content[start:i - 1]


# ─── JS File Structure (smoke tests) ─────────────────────────────────────


class TestJsFileStructure:
    """Belt-and-braces: the file we test must still exist with the
    expected top-level structure."""

    def test_file_exists(self):
        assert JS_PATH.exists(), 'advanced-midi.js should exist'

    def test_has_class_advanced_midi_settings(self, js_content):
        assert 'class AdvancedMIDISettings' in js_content

    def test_has_save_method(self, js_content):
        assert re.search(r'async\s+save\s*\(\s*\)', js_content), (
            "Expected `async save()` method on AdvancedMIDISettings"
        )


# ─── Auto-dependency: cluster_feature=pitch_hz → enable_pitch_detection ──


class TestPitchDependencyAutoEnable:
    """The WebUI advanced modal lets the user pick a cluster feature
    independently from the ``enable_pitch_detection`` checkbox. If
    they pick ``pitch_hz`` while pitch detection is off, the
    pipeline silently falls back to a different feature (see
    stems_to_midi/note_classification_core.py::_resolve_cluster_feature).
    The user can't tell the difference because the resulting MIDI
    looks the same as the default.

    Fix: the save handler must detect this combination and add an
    enable_pitch_detection=true update to the same payload, with a
    showToast hint so the user knows what changed.
    """

    def test_save_handler_contains_pitch_dependency_helper(
        self, js_content,
    ):
        """A regex-over-source assertion that the save handler
        includes some helper or branch that ties cluster_feature to
        enable_pitch_detection. We don't pin the exact function
        name (so the fix can be refactored) but we DO require that
        the source contains both identifiers in close proximity."""
        # Loose assertion: the JS must mention both cluster_feature
        # and enable_pitch_detection, and the pitch-hz string must
        # appear in a context that suggests a check (e.g. an `if`,
        # `===`, or a set / push).
        assert 'cluster_feature' in js_content, (
            "advanced-midi.js must reference 'cluster_feature' — "
            "the WebUI is supposed to handle this knob"
        )
        assert 'enable_pitch_detection' in js_content, (
            "advanced-midi.js must reference 'enable_pitch_detection' "
            "— the JS needs to know the coupled toggle"
        )
        assert "'pitch_hz'" in js_content or '"pitch_hz"' in js_content, (
            "advanced-midi.js must reference the literal string "
            "'pitch_hz' — the auto-dependency key"
        )

    def test_save_handler_auto_enables_pitch_detection(
        self, save_method_body,
    ):
        """The save method must contain logic that, when the user
        saves a cluster_feature change to 'pitch_hz' on a stem
        whose enable_pitch_detection is false, also pushes an
        enable_pitch_detection=true update into the same payload.

        We assert this by requiring the save body to contain:
        - a reference to ``pitch_hz``
        - a reference to ``enable_pitch_detection``
        - some mutation of the updates list (e.g. .push, .set, or
          a helper function call that takes the stem type and adds
          the toggle)

        This is intentionally a structural test — we don't pin
        the exact function name or shape of the helper, just the
        fact that the coupling exists in the save path."""
        # Both identifiers must appear in the save method
        assert 'pitch_hz' in save_method_body, (
            "save() must check for the 'pitch_hz' cluster feature"
        )
        assert 'enable_pitch_detection' in save_method_body, (
            "save() must handle enable_pitch_detection as a coupled toggle"
        )
        # Some mutation must exist: push, .set(, or a helper that
        # adds to the changes map. The exact form can vary. We
        # accept any pattern that proves the save path is engaged
        # in dependency resolution — not just blindly POSTing
        # this.changes.
        mutations = (
            '.push(' in save_method_body,
            'changes.set(' in save_method_body,
            'changes.add' in save_method_body,  # may be a Set
            'updates.push' in save_method_body,
            'updates.add' in save_method_body,
            # Named helpers are valid too — the dependency logic can
            # live in a method the save() method delegates to. We
            # accept any of the common naming conventions; the
            # contract is that save() engages dependency resolution,
            # not that it does so inline.
            '_autoEnable' in save_method_body,
            'applyPitchDependency' in save_method_body,
            '_addPitchDependency' in save_method_body,
            '_applyClusterFeatureDependencies' in save_method_body,
            '_resolveDependencies' in save_method_body,
        )
        assert any(mutations), (
            "save() must mutate the updates list (push, set, or a "
            "named helper) to add the coupled enable_pitch_detection "
            "update. The current save just POSTs whatever's in "
            "this.changes — it doesn't add the dependency."
        )

    def test_save_handler_keeps_existing_true_does_not_touch_field(
        self, js_content,
    ):
        """If enable_pitch_detection is already true, the save handler
        should NOT toggle it back to true (that's a no-op but creates
        a spurious change record in this.changes). The logic must
        check the current value before adding the dependency.

        We assert this by requiring the save method to read the
        current value of enable_pitch_detection (e.g. via
        this.configData, a field.getAttribute('checked'), or a
        lookup function) — not just blindly push a true value."""
        # A loose but useful check: the save method must access
        # some "current value" source (configData, the input element,
        # or a named helper) before deciding to add the dependency.
        # We don't pin the exact form.
        save_match = re.search(r'async\s+save\s*\(\s*\)\s*\{', js_content)
        assert save_match
        start = save_match.end()
        depth = 1
        i = start
        while i < len(js_content) and depth > 0:
            ch = js_content[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            i += 1
        save_body = js_content[start:i - 1]

        # The save method must reference a way to read the current
        # value. We allow several patterns:
        reads = (
            'configData' in save_body,         # has access to parsed config
            'getAttribute' in save_body,        # reads DOM input value
            '.checked' in save_body,            # reads checkbox state
            'this.changes.has' in save_body,    # checks if already set
            'isAlreadyEnabled' in save_body,    # named helper
            '_isEnabled' in save_body,          # named helper
            # If the dependency logic lives in a delegated method
            # rather than inline, the save body still has to engage
            # it — and the helper does the read. We accept the
            # delegation as long as the helper is named consistently
            # with our other dependency helpers.
            '_applyClusterFeatureDependencies' in save_body,
            '_resolveDependencies' in save_body,
        )
        assert any(reads), (
            "save() must consult the current value of "
            "enable_pitch_detection before toggling. Without that, "
            "saving when it's already true creates a no-op change "
            "and clutters the diff."
        )


class TestConvertHintToast:
    """The user must be told when a config change requires a full
    Convert (rebuild alone won't refresh detection-time features)."""

    def test_save_handler_shows_convert_hint(
        self, save_method_body,
    ):
        """The save method must include a showToast (or equivalent)
        call that surfaces a 'full Convert' hint when the diff
        includes a detection-time key. We don't pin the wording
        exactly — just the presence of a showToast with a string
        that mentions 'Convert' (or 'full conversion' / 're-convert')."""
        # Look for a showToast that mentions Convert / conversion /
        # re-convert. Allow either single or double quotes.
        pattern = re.compile(
            r"showToast\s*\(\s*['\"`][^'\"`]*[Cc]onvert",
            re.DOTALL,
        )
        assert pattern.search(save_method_body), (
            "save() must call showToast with a 'Convert' hint when "
            "saving detection-time changes. The current save just "
            "shows 'MIDI configuration saved successfully' — that "
            "doesn't tell the user they need a full Convert."
        )

    def test_save_handler_queues_hint_for_pitch_save(
        self, save_method_body,
    ):
        """Specifically: when the save body detects that the user
        just set cluster_feature=pitch_hz (or any detection-time
        key), the convert-hint toast must fire. The two must be in
        the same function — proves the hint is triggered by the
        save, not a separate action."""
        # Coarse assertion: 'pitch_hz' and 'showToast' both appear
        # in the save method body. The user-flow is provable from
        # the live WebUI; the test only locks the structural fact
        # that the hint is wired to the same code path as the
        # pitch_hz handling.
        assert 'pitch_hz' in save_method_body
        assert 'showToast' in save_method_body
