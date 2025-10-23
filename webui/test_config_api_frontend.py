"""
Tests for config API from frontend perspective.

Tests the complete frontend workflow of reading config, modifying values,
and saving changes. Ensures that nested paths are correctly constructed
and that the API properly validates and saves updates.
"""

import pytest
import tempfile
import json
from pathlib import Path
from webui.app import create_app
from webui.config_engine import YAMLConfigEngine


# Sample midiconfig mimicking real structure
SAMPLE_MIDICONFIG = """
# Audio Processing
audio:
  force_mono: true
  silence_threshold: 0.001
  peak_window_sec: 0.10

onset_detection:
  threshold: 0.3
  delta: 0.01
  wait: 3
  hop_length: 512

kick:
  midi_note: 36
  onset_threshold: 0.1
  timing_offset: -0.014

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
  detect_open: true

cymbals:
  midi_note: 57
  onset_threshold: 0.35

midi:
  min_velocity: 80
  max_velocity: 110
  default_tempo: 124.0
  max_note_duration: 0.5

debug:
  show_all_onsets: true
  show_spectral_data: true

learning_mode:
  enabled: false
  export_all_detections: true
"""


class TestConfigAPIParsing:
    """Test that config parsing creates correct paths for frontend"""
    
    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config file"""
        config_file = tmp_path / "midiconfig.yaml"
        config_file.write_text(SAMPLE_MIDICONFIG)
        return config_file
    
    def test_nested_fields_have_full_paths(self, temp_config):
        """Ensure nested fields have complete paths including parent keys"""
        engine = YAMLConfigEngine(temp_config, config_type='midiconfig')
        sections = engine.parse()
        
        # Find midi section
        midi_section = next((s for s in sections if s.name == 'midi'), None)
        assert midi_section is not None, "midi section should exist"
        
        # Check that fields have full paths
        min_velocity_field = next((f for f in midi_section.fields if f.key == 'min_velocity'), None)
        assert min_velocity_field is not None, "min_velocity field should exist"
        assert min_velocity_field.path == ['midi', 'min_velocity'], \
            f"Expected path ['midi', 'min_velocity'], got {min_velocity_field.path}"
        
        # Check UI control has correct dotted path
        ui_control = min_velocity_field.to_ui_control()
        assert ui_control['path'] == 'midi.min_velocity', \
            f"Expected path 'midi.min_velocity', got {ui_control['path']}"
    
    def test_all_nested_fields_have_correct_paths(self, temp_config):
        """Test that ALL nested fields across all sections have correct paths"""
        engine = YAMLConfigEngine(temp_config, config_type='midiconfig')
        sections = engine.parse()
        
        test_cases = [
            ('audio', 'force_mono', ['audio', 'force_mono']),
            ('audio', 'silence_threshold', ['audio', 'silence_threshold']),
            ('onset_detection', 'threshold', ['onset_detection', 'threshold']),
            ('onset_detection', 'delta', ['onset_detection', 'delta']),
            ('kick', 'midi_note', ['kick', 'midi_note']),
            ('kick', 'onset_threshold', ['kick', 'onset_threshold']),
            ('toms', 'midi_note_low', ['toms', 'midi_note_low']),
            ('toms', 'midi_note_mid', ['toms', 'midi_note_mid']),
            ('hihat', 'detect_open', ['hihat', 'detect_open']),
            ('midi', 'min_velocity', ['midi', 'min_velocity']),
            ('midi', 'max_velocity', ['midi', 'max_velocity']),
            ('midi', 'default_tempo', ['midi', 'default_tempo']),
            ('debug', 'show_all_onsets', ['debug', 'show_all_onsets']),
        ]
        
        for section_name, field_key, expected_path in test_cases:
            section = next((s for s in sections if s.name == section_name), None)
            assert section is not None, f"Section '{section_name}' should exist"
            
            field = next((f for f in section.fields if f.key == field_key), None)
            assert field is not None, f"Field '{field_key}' should exist in section '{section_name}'"
            
            assert field.path == expected_path, \
                f"Section '{section_name}', field '{field_key}': expected path {expected_path}, got {field.path}"
    
    def test_ui_controls_have_dotted_paths(self, temp_config):
        """Test that to_ui_control() produces correct dotted paths for all fields"""
        engine = YAMLConfigEngine(temp_config, config_type='midiconfig')
        sections = engine.parse()
        
        # Collect all UI controls
        all_controls = []
        for section in sections:
            for field in section.fields:
                all_controls.append(field.to_ui_control())
        
        # Verify no single-element paths for nested configs
        for control in all_controls:
            path_parts = control['path'].split('.')
            # All paths should have at least 2 parts (section.field)
            # since we don't have top-level scalar values
            assert len(path_parts) >= 2, \
                f"Path '{control['path']}' should have at least 2 parts (section.field)"


class TestConfigAPIUpdate:
    """Test config update API from frontend perspective"""
    
    @pytest.fixture
    def mock_project_dir(self, tmp_path):
        """Create mock project directory with config"""
        project_dir = tmp_path / "user_files" / "1 - test_project"
        project_dir.mkdir(parents=True)
        
        config_file = project_dir / "midiconfig.yaml"
        config_file.write_text(SAMPLE_MIDICONFIG)
        
        return project_dir
    
    @pytest.fixture
    def client(self, mock_project_dir, monkeypatch):
        """Create test client with mocked project directory"""
        # Mock the project directory lookup
        def mock_glob(pattern):
            return [str(mock_project_dir)]
        
        import glob
        monkeypatch.setattr(glob, 'glob', mock_glob)
        
        test_app = create_app()
        test_app.config['TESTING'] = True
        with test_app.test_client() as client:
            yield client
    
    def test_update_nested_midi_value(self, client):
        """Test updating a nested midi value like min_velocity"""
        updates = {
            'updates': [
                {'path': ['midi', 'min_velocity'], 'value': 90}
            ]
        }
        
        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps(updates),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
    
    def test_update_multiple_nested_values(self, client):
        """Test updating multiple nested values in one request"""
        updates = {
            'updates': [
                {'path': ['midi', 'min_velocity'], 'value': 90},
                {'path': ['midi', 'max_velocity'], 'value': 120},
                {'path': ['audio', 'force_mono'], 'value': False},
                {'path': ['kick', 'onset_threshold'], 'value': 0.15}
            ]
        }
        
        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps(updates),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
    
    def test_reject_dict_replacement(self, client):
        """Test that replacing dict with primitive is rejected"""
        updates = {
            'updates': [
                {'path': ['midi'], 'value': 60}  # WRONG: trying to replace dict
            ]
        }
        
        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps(updates),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Cannot replace dictionary' in data['errors'][0]['error']
    
    def test_reject_multiple_with_one_bad_update(self, client):
        """Test that all updates are rejected if one is invalid"""
        updates = {
            'updates': [
                {'path': ['midi', 'min_velocity'], 'value': 90},  # Good
                {'path': ['audio'], 'value': 123},  # BAD: dict replacement
                {'path': ['kick', 'onset_threshold'], 'value': 0.15}  # Good
            ]
        }
        
        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps(updates),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        # Should have error about audio
        errors = [e['path'] for e in data['errors']]
        assert 'audio' in errors
    
    def test_update_all_midi_section_fields(self, client):
        """Test updating all fields in midi section individually"""
        updates = {
            'updates': [
                {'path': ['midi', 'min_velocity'], 'value': 85},
                {'path': ['midi', 'max_velocity'], 'value': 115},
                {'path': ['midi', 'default_tempo'], 'value': 130.0},
                {'path': ['midi', 'max_note_duration'], 'value': 1.0}
            ]
        }
        
        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps(updates),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True


class TestConfigAPIGet:
    """Test config GET API returns correct structure"""
    
    @pytest.fixture
    def mock_project_dir(self, tmp_path):
        """Create mock project directory with config"""
        project_dir = tmp_path / "user_files" / "1 - test_project"
        project_dir.mkdir(parents=True)
        
        config_file = project_dir / "midiconfig.yaml"
        config_file.write_text(SAMPLE_MIDICONFIG)
        
        return project_dir
    
    @pytest.fixture
    def client(self, mock_project_dir, monkeypatch):
        """Create test client with mocked project directory"""
        # Mock the project directory lookup
        def mock_glob(pattern):
            return [str(mock_project_dir)]
        
        import glob
        monkeypatch.setattr(glob, 'glob', mock_glob)
        
        test_app = create_app()
        test_app.config['TESTING'] = True
        with test_app.test_client() as client:
            yield client
    
    def test_get_config_returns_full_paths(self, client):
        """Test that GET API returns fields with complete dotted paths"""
        response = client.get('/api/config/1/midiconfig')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        
        # Find midi section
        midi_section = next((s for s in data['sections'] if s['name'] == 'midi'), None)
        assert midi_section is not None, "midi section should be in response"
        
        # Check fields have full paths
        field_paths = [f['path'] for f in midi_section['fields']]
        assert 'midi.min_velocity' in field_paths
        assert 'midi.max_velocity' in field_paths
        assert 'midi.default_tempo' in field_paths
        
        # Ensure no single-part paths
        for field in midi_section['fields']:
            assert '.' in field['path'], \
                f"Field path '{field['path']}' should contain a dot (section.field)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
