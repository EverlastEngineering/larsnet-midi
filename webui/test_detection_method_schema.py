"""
Tests for the ``detection_method`` setting on the central settings schema.

Covers the schema entry, default value, CLI flag, and choice validation.
This is the schema-layer half of the detection-method work; the
processing_shell wiring has its own tests in
``stems_to_midi/test_detection_method.py``.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from webui.settings_schema import (
    SETTINGS_REGISTRY,
    SettingCategory,
    SettingType,
    UIControl,
    get_setting_by_key,
    get_settings_schema,
)
from webui.cli_builder import build_cli_parser


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/Users/jasoncopp/miniforge3/envs/drumtomidi/bin/python'


# ---------------------------------------------------------------------------
# 1. Schema presence and shape
# ---------------------------------------------------------------------------

class TestDetectionMethodSchemaEntry:
    """The setting is registered in the central settings schema."""

    def test_detection_method_in_registry(self):
        setting = get_setting_by_key('detection_method')
        assert setting is not None, (
            "detection_method must be added to SETTINGS_REGISTRY in "
            "webui/settings_schema.py"
        )

    def test_detection_method_is_choice_type(self):
        setting = get_setting_by_key('detection_method')
        assert setting.type == SettingType.CHOICE

    def test_detection_method_allowed_values(self):
        setting = get_setting_by_key('detection_method')
        assert set(setting.allowed_values) == {'energy', 'spectral', 'both'}

    def test_detection_method_default_is_both(self):
        """Default is 'both' so the new spectral candidates are visible
        out of the box without losing any energy-detector events."""
        setting = get_setting_by_key('detection_method')
        assert setting.default == 'both'

    def test_detection_method_category(self):
        setting = get_setting_by_key('detection_method')
        assert setting.category == SettingCategory.ONSET_DETECTION

    def test_detection_method_ui_control(self):
        setting = get_setting_by_key('detection_method')
        assert setting.ui_control == UIControl.SELECT

    def test_detection_method_label(self):
        setting = get_setting_by_key('detection_method')
        assert setting.label == 'Detection Method'

    def test_detection_method_yaml_path(self):
        """The yaml_path is what cli_builder uses to write CLI overrides
        back to the merged config dict."""
        setting = get_setting_by_key('detection_method')
        assert setting.yaml_path == ['onset_detection', 'detection_method']

    def test_detection_method_cli_flag(self):
        setting = get_setting_by_key('detection_method')
        assert setting.cli_flag == '--detection-method'

    def test_detection_method_description_mentions_spectral(self):
        setting = get_setting_by_key('detection_method')
        assert 'spectral' in setting.description.lower()
        assert 'energy' in setting.description.lower()


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------

class TestDetectionMethodValidation:
    """The CHOICE validator rejects bad values."""

    def test_valid_choices(self):
        setting = get_setting_by_key('detection_method')
        for value in ('energy', 'spectral', 'both'):
            ok, err = setting.validate(value)
            assert ok, f"validate({value!r}) should pass, got error: {err}"
            assert err is None

    def test_invalid_choice(self):
        setting = get_setting_by_key('detection_method')
        ok, err = setting.validate('librosa')
        assert not ok
        assert 'energy' in err and 'spectral' in err and 'both' in err


# ---------------------------------------------------------------------------
# 3. Schema API surface (settings_schema.get_settings_schema)
# ---------------------------------------------------------------------------

class TestDetectionMethodInSchemaApi:
    """The /api/settings/schema endpoint must expose the new setting."""

    def test_schema_dict_includes_detection_method(self):
        schema = get_settings_schema()
        assert 'detection_method' in schema['settings']
        entry = schema['settings']['detection_method']
        assert entry['key'] == 'detection_method'
        assert entry['type'] == 'choice'
        assert entry['default'] == 'both'
        assert 'energy' in entry['allowed_values']
        assert 'spectral' in entry['allowed_values']
        assert 'both' in entry['allowed_values']

    def test_schema_is_json_serializable(self):
        """Schema gets sent over the wire as JSON — must be safe."""
        schema = get_settings_schema()
        # Should not raise
        json.dumps(schema)

    def test_schema_listed_in_onset_detection_category(self):
        schema = get_settings_schema()
        cat = schema['categories']['onset_detection']
        assert 'detection_method' in cat['settings']


# ---------------------------------------------------------------------------
# 4. CLI flag appears in --help
# ---------------------------------------------------------------------------

class TestDetectionMethodCliFlag:
    """build_cli_parser must expose --detection-method."""

    def test_parser_accepts_detection_method_flag(self):
        parser = build_cli_parser(prog='stems_to_midi_cli')
        # If the flag isn't registered, parse_args will SystemExit. This
        # call returning without raising is the assertion.
        args = parser.parse_args(['--detection-method', 'spectral'])
        assert getattr(args, 'detection_method') == 'spectral'

    def test_parser_default_is_both(self):
        parser = build_cli_parser(prog='stems_to_midi_cli')
        args = parser.parse_args([])
        assert getattr(args, 'detection_method') == 'both'

    def test_parser_rejects_invalid_choice(self):
        parser = build_cli_parser(prog='stems_to_midi_cli')
        with pytest.raises(SystemExit):
            parser.parse_args(['--detection-method', 'bogus'])

    def test_help_output_contains_flag(self, capsys):
        """The --help text advertises the new flag with the allowed values."""
        parser = build_cli_parser(prog='stems_to_midi_cli')
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
        out = capsys.readouterr().out
        assert '--detection-method' in out
        assert 'energy' in out
        assert 'spectral' in out
        assert 'both' in out


# ---------------------------------------------------------------------------
# 5. Apply CLI override writes to the right yaml_path
# ---------------------------------------------------------------------------

class TestDetectionMethodCliOverride:
    """apply_cli_overrides must write --detection-method to onset_detection."""

    def test_override_writes_to_yaml_path(self):
        from webui.cli_builder import apply_cli_overrides

        parser = build_cli_parser(prog='stems_to_midi_cli')
        args = parser.parse_args(['--detection-method', 'spectral'])

        config = {
            'onset_detection': {
                'threshold': 0.3, 'delta': 0.01, 'wait': 3, 'hop_length': 512,
            },
        }
        n, applied = apply_cli_overrides(args, config)
        assert 'detection_method' in applied
        assert n == 1
        assert config['onset_detection']['detection_method'] == 'spectral'

    def test_no_override_keeps_default(self):
        """When the user doesn't pass the flag, the config should keep
        the schema default and apply_cli_overrides should not 'apply' it
        (since only_set=True skips default values)."""
        from webui.cli_builder import apply_cli_overrides

        parser = build_cli_parser(prog='stems_to_midi_cli')
        args = parser.parse_args([])

        config = {'onset_detection': {'threshold': 0.3}}
        n, applied = apply_cli_overrides(args, config)
        assert 'detection_method' not in applied
        # The user's config is not polluted by the schema default
        assert 'detection_method' not in config['onset_detection']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
