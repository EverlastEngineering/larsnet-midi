"""
Tests for YAML Configuration Engine

Tests parsing, validation, and round-trip saving of YAML configurations
with comment preservation.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from ruamel.yaml import YAML

from webui.config_engine import (
    ConfigField,
    ConfigSection,
    ValidationRule,
    YAMLConfigEngine,
    get_config_engine
)


# Sample YAML content for testing
SAMPLE_YAML = """
# Top-level comment
global:
  sr: 44100          # Sample rate in Hz
  enabled: true      # Enable processing
  volume: 0.75       # Volume level (0-1)
  
kick:
  midi_note: 36      # MIDI note number
  threshold: 0.5     # Detection threshold (0-1)
  model_path: '/app/models/kick.pth'  # Path to model file
"""


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(SAMPLE_YAML)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def config_engine(temp_yaml_file):
    """Create a YAMLConfigEngine instance for testing"""
    return YAMLConfigEngine(temp_yaml_file)


class TestValidationRule:
    """Tests for ValidationRule class"""
    
    def test_min_value_validation(self):
        """Test minimum value validation"""
        rule = ValidationRule(min_value=0)
        
        assert rule.validate(0)[0] is True
        assert rule.validate(1)[0] is True
        assert rule.validate(-1)[0] is False
    
    def test_max_value_validation(self):
        """Test maximum value validation"""
        rule = ValidationRule(max_value=1)
        
        assert rule.validate(1)[0] is True
        assert rule.validate(0)[0] is True
        assert rule.validate(2)[0] is False
    
    def test_range_validation(self):
        """Test range validation (min and max)"""
        rule = ValidationRule(min_value=0, max_value=1)
        
        assert rule.validate(0.5)[0] is True
        assert rule.validate(0)[0] is True
        assert rule.validate(1)[0] is True
        assert rule.validate(-0.1)[0] is False
        assert rule.validate(1.1)[0] is False
    
    def test_allowed_values_validation(self):
        """Test allowed values validation"""
        rule = ValidationRule(allowed_values=['cpu', 'cuda', 'mps'])
        
        assert rule.validate('cpu')[0] is True
        assert rule.validate('cuda')[0] is True
        assert rule.validate('gpu')[0] is False


class TestConfigField:
    """Tests for ConfigField class"""
    
    def test_type_inference_bool(self):
        """Test boolean type inference"""
        field = ConfigField('enabled', True)
        assert field.field_type == 'bool'
    
    def test_type_inference_int(self):
        """Test integer type inference"""
        field = ConfigField('count', 42)
        assert field.field_type == 'int'
    
    def test_type_inference_float(self):
        """Test float type inference"""
        field = ConfigField('threshold', 0.5)
        assert field.field_type == 'float'
    
    def test_type_inference_string(self):
        """Test string type inference"""
        field = ConfigField('name', 'test')
        assert field.field_type == 'string'
    
    def test_type_inference_path(self):
        """Test path type inference"""
        field = ConfigField('model', '/app/models/test.pth')
        assert field.field_type == 'path'
    
    def test_midi_note_validation(self):
        """Test MIDI note validation (0-127)"""
        field = ConfigField('midi_note', 36)
        assert field.validate()[0] is True
        
        field.value = 127
        assert field.validate()[0] is True
        
        field.value = 128
        assert field.validate()[0] is False
        
        field.value = -1
        assert field.validate()[0] is False
    
    def test_comment_range_extraction(self):
        """Test extraction of range from comments"""
        field = ConfigField('threshold', 0.5, comment='Detection threshold (0-1)')
        
        assert field.validation_rule.min_value == 0
        assert field.validation_rule.max_value == 1
    
    def test_to_ui_control(self):
        """Test conversion to UI control specification"""
        field = ConfigField(
            'threshold',
            0.5,
            comment='Detection threshold (0-1)',
            path=['kick']
        )
        
        control = field.to_ui_control()
        
        assert control['key'] == 'threshold'
        assert control['path'] == 'kick'
        assert control['type'] == 'float'
        assert control['value'] == 0.5
        assert 'threshold' in control['label'].lower()
        assert control['description'] == 'Detection threshold (0-1)'


class TestConfigSection:
    """Tests for ConfigSection class"""
    
    def test_section_creation(self):
        """Test section creation with fields"""
        fields = [
            ConfigField('midi_note', 36),
            ConfigField('threshold', 0.5)
        ]
        section = ConfigSection('kick', fields)
        
        assert section.name == 'kick'
        assert len(section.fields) == 2
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        fields = [
            ConfigField('midi_note', 36),
            ConfigField('threshold', 0.5)
        ]
        section = ConfigSection('kick', fields, comment='Kick drum settings')
        
        result = section.to_dict()
        
        assert result['name'] == 'kick'
        assert result['label'] == 'Kick'
        assert result['description'] == 'Kick drum settings'
        assert len(result['fields']) == 2
    
    def test_validate_all(self):
        """Test validation of all fields in section"""
        fields = [
            ConfigField('midi_note', 36),  # Valid
            ConfigField('midi_note_invalid', 200)  # Invalid (>127)
        ]
        section = ConfigSection('test', fields)
        
        errors = section.validate_all()
        
        # First field valid, second invalid
        assert len(errors) == 1
        assert errors[0][0] == 'midi_note_invalid'


class TestYAMLConfigEngine:
    """Tests for YAMLConfigEngine class"""
    
    def test_load_yaml(self, config_engine):
        """Test loading YAML file"""
        data = config_engine.load()
        
        assert 'global' in data
        assert 'kick' in data
        assert data['global']['sr'] == 44100
        assert data['kick']['midi_note'] == 36
    
    def test_parse_sections(self, config_engine):
        """Test parsing YAML into sections"""
        sections = config_engine.parse()
        
        # Should have 'global' and 'kick' sections
        section_names = [s.name for s in sections]
        assert 'global' in section_names
        assert 'kick' in section_names
    
    def test_parse_fields_with_comments(self, config_engine):
        """Test that comments are extracted correctly"""
        sections = config_engine.parse()
        
        # Find the 'global' section
        global_section = next(s for s in sections if s.name == 'global')
        
        # Find the 'sr' field
        sr_field = next(f for f in global_section.fields if f.key == 'sr')
        
        assert 'Sample rate' in sr_field.comment or 'Hz' in sr_field.comment
    
    def test_update_value(self, config_engine):
        """Test updating a value in the YAML"""
        config_engine.load()
        
        # Update value
        success = config_engine.update_value(['kick', 'midi_note'], 38)
        assert success is True
        
        # Verify update
        data = config_engine.load()
        assert data['kick']['midi_note'] == 38
    
    def test_update_nonexistent_value(self, config_engine):
        """Test updating a non-existent value"""
        config_engine.load()
        
        success = config_engine.update_value(['nonexistent', 'key'], 42)
        assert success is False
    
    def test_round_trip_save(self, config_engine, temp_yaml_file):
        """Test saving changes preserves formatting and comments"""
        # Load and modify
        config_engine.load()
        config_engine.update_value(['kick', 'midi_note'], 38)
        config_engine.save()
        
        # Read raw file content
        content = temp_yaml_file.read_text()
        
        # Check that comments are preserved
        assert '# Sample rate' in content or '# Hz' in content
        assert '# MIDI note' in content
        
        # Check that value was updated
        assert '38' in content
    
    def test_type_preservation(self, config_engine):
        """Test that value types are preserved during updates"""
        config_engine.load()
        
        # Update bool
        config_engine.update_value(['global', 'enabled'], False)
        assert config_engine._data['global']['enabled'] is False
        assert isinstance(config_engine._data['global']['enabled'], bool)
        
        # Update int
        config_engine.update_value(['kick', 'midi_note'], 38)
        assert config_engine._data['kick']['midi_note'] == 38
        assert isinstance(config_engine._data['kick']['midi_note'], int)
        
        # Update float
        config_engine.update_value(['global', 'volume'], 0.9)
        assert config_engine._data['global']['volume'] == 0.9
        assert isinstance(config_engine._data['global']['volume'], float)
    
    def test_validate_all(self, config_engine):
        """Test validating all configuration values"""
        config_engine.load()
        
        # Initially should be valid
        errors = config_engine.validate_all()
        assert len(errors) == 0
        
        # Introduce invalid value
        config_engine.update_value(['kick', 'midi_note'], 200)
        
        errors = config_engine.validate_all()
        assert 'kick' in errors
        assert len(errors['kick']) > 0


class TestGetConfigEngine:
    """Tests for get_config_engine factory function"""
    
    def test_invalid_config_type(self):
        """Test that invalid config type raises ValueError"""
        with pytest.raises(ValueError, match='config_type must be one of'):
            get_config_engine(1, 'invalid')
    
    def test_project_not_found(self):
        """Test that non-existent project raises ValueError"""
        with pytest.raises(ValueError, match='not found'):
            get_config_engine(99999, 'config')


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_yaml(self):
        """Test handling of empty YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = Path(f.name)
        
        try:
            engine = YAMLConfigEngine(temp_path)
            data = engine.load()
            
            # Empty YAML should load as None or empty dict
            assert data is None or data == {}
        finally:
            temp_path.unlink()
    
    def test_yaml_without_comments(self):
        """Test handling of YAML without comments"""
        yaml_content = """
global:
  sr: 44100
  enabled: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)
        
        try:
            engine = YAMLConfigEngine(temp_path)
            sections = engine.parse()
            
            # Should still parse successfully
            assert len(sections) > 0
            
            # Comments should be empty strings
            for section in sections:
                for field in section.fields:
                    assert isinstance(field.comment, str)
        finally:
            temp_path.unlink()
    
    def test_deeply_nested_yaml(self):
        """Test handling of deeply nested YAML structures"""
        yaml_content = """
level1:
  level2:
    level3:
      value: 42
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)
        
        try:
            engine = YAMLConfigEngine(temp_path)
            sections = engine.parse(max_depth=3)
            
            # Should parse nested structure
            assert len(sections) > 0
        finally:
            temp_path.unlink()
