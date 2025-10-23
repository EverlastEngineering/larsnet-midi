"""
Tests for config schema validation and corruption prevention.

Tests the fix for the bug where saving config from webui would replace
dictionary values with primitives, corrupting the YAML structure.
"""

import pytest
import tempfile
from pathlib import Path
from webui.config_engine import YAMLConfigEngine
from webui.config_schema import validate_structure, MIDICONFIG_SCHEMA


# Sample valid midiconfig YAML
VALID_MIDICONFIG = """
audio:
  force_mono: true
  silence_threshold: 0.001

onset_detection:
  threshold: 0.3
  delta: 0.01
  wait: 3

kick:
  midi_note: 36
  onset_threshold: 0.1

snare:
  midi_note: 38
  onset_threshold: 0.2

toms:
  midi_note_high: 50
  midi_note_mid: 47
  midi_note_low: 45

hihat:
  midi_note_closed: 42
  midi_note_open: 46

cymbals:
  midi_note: 57

midi:
  tempo: 120
  time_signature: [4, 4]

debug:
  show_all_onsets: true
"""


class TestSchemaValidation:
    """Test schema validation functions"""
    
    def test_validate_structure_valid(self):
        """Valid structure should pass validation"""
        data = {
            'audio': {'force_mono': True},
            'onset_detection': {'threshold': 0.3, 'delta': 0.01},
            'kick': {'midi_note': 36},
            'toms': {'midi_note_low': 45},
            'midi': {'tempo': 120}
        }
        
        is_valid, error = validate_structure(data, MIDICONFIG_SCHEMA)
        assert is_valid
        assert error == ""
    
    def test_validate_structure_dict_as_primitive(self):
        """Dict replaced with primitive should fail validation"""
        data = {
            'audio': {'force_mono': True},
            'toms': 600,  # WRONG: should be dict
            'midi': 60,   # WRONG: should be dict
        }
        
        is_valid, error = validate_structure(data, MIDICONFIG_SCHEMA)
        assert not is_valid
        assert 'toms' in error
        assert 'should be a dictionary' in error
    
    def test_validate_structure_primitive_as_dict(self):
        """Primitive replaced with dict should fail validation"""
        data = {
            'audio': 123,  # WRONG: should be dict
        }
        
        is_valid, error = validate_structure(data, MIDICONFIG_SCHEMA)
        assert not is_valid
        assert 'audio' in error
        assert 'should be a dictionary' in error


class TestConfigEngineSaveValidation:
    """Test config engine prevents corruption on save"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(VALID_MIDICONFIG)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    def test_save_valid_config(self, temp_config_file):
        """Saving valid config should succeed"""
        engine = YAMLConfigEngine(temp_config_file, config_type='midiconfig')
        engine.load()
        
        # Update a valid leaf value
        success, error = engine.update_value(['kick', 'midi_note'], 35)
        assert success
        assert error == ""
        
        # Should save without error
        engine.save()
        
        # Verify saved correctly
        engine2 = YAMLConfigEngine(temp_config_file, config_type='midiconfig')
        data = engine2.load()
        assert data['kick']['midi_note'] == 35
    
    def test_prevent_dict_corruption(self, temp_config_file):
        """Attempting to replace dict with primitive should fail"""
        engine = YAMLConfigEngine(temp_config_file, config_type='midiconfig')
        engine.load()
        
        # Try to replace 'toms' dict with primitive
        success, error = engine.update_value(['toms'], 600)
        assert not success
        assert 'Cannot replace dictionary' in error
        assert 'toms' in error
    
    def test_prevent_nested_dict_corruption(self, temp_config_file):
        """Attempting to replace nested dict with primitive should fail"""
        engine = YAMLConfigEngine(temp_config_file, config_type='midiconfig')
        engine.load()
        
        # Try to replace nested 'audio' dict with primitive
        success, error = engine.update_value(['audio'], 123)
        assert not success
        assert 'Cannot replace dictionary' in error
    
    def test_update_nonexistent_path(self, temp_config_file):
        """Updating nonexistent path should return descriptive error"""
        engine = YAMLConfigEngine(temp_config_file, config_type='midiconfig')
        engine.load()
        
        success, error = engine.update_value(['nonexistent', 'key'], 123)
        assert not success
        assert 'not found' in error.lower()
    
    def test_save_with_corrupted_data_fails(self, temp_config_file):
        """Save should fail if data is manually corrupted"""
        engine = YAMLConfigEngine(temp_config_file, config_type='midiconfig')
        data = engine.load()
        
        # Manually corrupt the data (bypassing update_value validation)
        data['toms'] = 600
        data['midi'] = 60
        
        # Save should detect corruption and raise
        with pytest.raises(RuntimeError) as exc_info:
            engine.save()
        
        assert 'validation failed' in str(exc_info.value).lower()
        assert 'toms' in str(exc_info.value)
    
    def test_nested_update_works(self, temp_config_file):
        """Updating nested values should work normally"""
        engine = YAMLConfigEngine(temp_config_file, config_type='midiconfig')
        engine.load()
        
        # Update nested value
        success, error = engine.update_value(['toms', 'midi_note_low'], 43)
        assert success
        assert error == ""
        
        # Save should succeed
        engine.save()
        
        # Verify
        engine2 = YAMLConfigEngine(temp_config_file, config_type='midiconfig')
        data = engine2.load()
        assert data['toms']['midi_note_low'] == 43


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
