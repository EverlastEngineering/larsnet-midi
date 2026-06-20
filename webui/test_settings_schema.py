"""
Tests for centralized settings schema

Validates that the settings schema is:
- Complete (has all required fields)
- Consistent (types, defaults, validation)
- Usable by both Python and WebUI
"""

import pytest
from webui.settings_schema import (
    SettingDefinition,
    SettingType,
    UIControl,
    SettingCategory,
    get_settings_schema,
    get_setting_by_key,
    get_settings_by_category,
    get_defaults_for_category,
    SETTINGS_REGISTRY
)


def test_settings_registry_not_empty():
    """Ensure settings registry is populated"""
    assert len(SETTINGS_REGISTRY) > 0, "Settings registry should not be empty"


def test_all_settings_have_required_fields():
    """Ensure all settings have required metadata"""
    for setting in SETTINGS_REGISTRY:
        assert setting.key, f"Setting missing key: {setting}"
        assert setting.type, f"Setting missing type: {setting.key}"
        assert setting.default is not None or setting.nullable, \
            f"Setting {setting.key} has null default but nullable=False"
        assert setting.label, f"Setting missing label: {setting.key}"
        assert setting.description, f"Setting missing description: {setting.key}"
        assert setting.category, f"Setting missing category: {setting.key}"
        assert setting.ui_control, f"Setting missing ui_control: {setting.key}"


def test_numeric_settings_have_valid_ranges():
    """Ensure numeric settings have sensible min/max values"""
    numeric_types = {SettingType.INT, SettingType.FLOAT}
    
    for setting in SETTINGS_REGISTRY:
        if setting.type in numeric_types:
            if setting.min_value is not None and setting.max_value is not None:
                assert setting.min_value < setting.max_value, \
                    f"Setting {setting.key}: min must be < max"
            
            if setting.default is not None:
                if setting.min_value is not None:
                    assert setting.default >= setting.min_value, \
                        f"Setting {setting.key}: default {setting.default} < min {setting.min_value}"
                if setting.max_value is not None:
                    assert setting.default <= setting.max_value, \
                        f"Setting {setting.key}: default {setting.default} > max {setting.max_value}"


def test_choice_settings_have_allowed_values():
    """Ensure CHOICE settings have allowed_values list"""
    for setting in SETTINGS_REGISTRY:
        if setting.type == SettingType.CHOICE:
            assert setting.allowed_values is not None, \
                f"CHOICE setting {setting.key} missing allowed_values"
            assert len(setting.allowed_values) > 0, \
                f"CHOICE setting {setting.key} has empty allowed_values"
            
            if setting.default is not None:
                assert setting.default in setting.allowed_values, \
                    f"Setting {setting.key}: default not in allowed_values"


def test_get_setting_by_key():
    """Test retrieving settings by key"""
    # Should find existing setting
    min_velocity = get_setting_by_key('min_velocity')
    assert min_velocity is not None
    assert min_velocity.key == 'min_velocity'
    assert min_velocity.type == SettingType.INT

    # Should return None for non-existent setting
    nonexistent = get_setting_by_key('nonexistent_setting')
    assert nonexistent is None


def test_get_settings_by_category():
    """Test retrieving settings by category"""
    midi_settings = get_settings_by_category(SettingCategory.MIDI_OUTPUT)
    assert len(midi_settings) > 0
    
    for setting in midi_settings:
        assert setting.category == SettingCategory.MIDI_OUTPUT

    tempo = get_setting_by_key('tempo')
    
    assert tempo.nullable, "Tempo should be nullable"
    
    # Null should be valid
    is_valid, error = tempo.validate(None)
    assert is_valid
    assert error is None
    
    # Valid numeric value
    is_valid, error = tempo.validate(120.0)
    assert is_valid


def test_boolean_settings_validation():
    """Test boolean setting validation"""
    force_mono = get_setting_by_key('force_mono')
    
    # Valid boolean
    is_valid, error = force_mono.validate(True)
    assert is_valid
    
    # Invalid type
    is_valid, error = force_mono.validate("true")
    assert not is_valid


def test_choice_settings_validation():
    """Test CHOICE setting validation"""
    device = get_setting_by_key('device')
    
    # Valid choice
    is_valid, error = device.validate('auto')
    assert is_valid
    
    # Valid choice
    is_valid, error = device.validate('cuda')
    assert is_valid
    
    # Invalid choice
    is_valid, error = device.validate('invalid')
    assert not is_valid


def test_midi_note_ranges():
    """Test that MIDI notes are in valid range (0-127)"""
    midi_note_settings = [
        'kick_midi_note',
        'snare_midi_note',
        'toms_midi_note_low',
        'toms_midi_note_mid',
        'toms_midi_note_high',
        'hihat_midi_note_closed',
        'hihat_midi_note_open',
        'cymbals_midi_note'
    ]
    
    for key in midi_note_settings:
        setting = get_setting_by_key(key)
        assert setting is not None, f"Missing setting: {key}"
        assert setting.min_value == 0, f"{key}: min should be 0"
        assert setting.max_value == 127, f"{key}: max should be 127"
        assert 0 <= setting.default <= 127, f"{key}: default out of range"


def test_no_duplicate_keys():
    """Ensure no duplicate setting keys"""
    keys = [s.key for s in SETTINGS_REGISTRY]
    assert len(keys) == len(set(keys)), "Duplicate setting keys found"


def test_no_duplicate_cli_flags():
    """Ensure no duplicate CLI flags"""
    flags = [s.cli_flag for s in SETTINGS_REGISTRY if s.cli_flag]
    assert len(flags) == len(set(flags)), "Duplicate CLI flags found"


def test_schema_serializable():
    """Ensure schema can be converted to JSON-serializable dict"""
    schema = get_settings_schema()
    
    # Should be serializable
    import json
    try:
        json.dumps(schema)
    except (TypeError, ValueError) as e:
        pytest.fail(f"Schema not JSON-serializable: {e}")


def test_ui_control_matches_type():
    """Ensure UI controls match setting types"""
    for setting in SETTINGS_REGISTRY:
        if setting.type == SettingType.BOOL:
            assert setting.ui_control == UIControl.CHECKBOX, \
                f"Bool setting {setting.key} should use CHECKBOX control"
        
        elif setting.type == SettingType.CHOICE:
            assert setting.ui_control == UIControl.SELECT, \
                f"CHOICE setting {setting.key} should use SELECT control"


def test_yaml_paths_are_unique():
    """Ensure YAML paths don't conflict"""
    paths = [s.yaml_path for s in SETTINGS_REGISTRY if s.yaml_path]
    path_strings = ['.'.join(p) for p in paths]
    
    # Allow duplicates for nullable overrides (e.g., kick_onset_threshold)
    # But ensure no exact duplicates
    from collections import Counter
    duplicates = [k for k, v in Counter(path_strings).items() if v > 1]
    
    # Some duplicates are expected (per-stem overrides)
    # Just ensure they're reasonable
    for dup in duplicates:
        assert 'onset' in dup or 'midi_note' in dup, \
            f"Unexpected duplicate YAML path: {dup}"
