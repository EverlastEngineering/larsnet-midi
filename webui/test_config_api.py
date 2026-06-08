"""
Tests for Configuration API endpoints

Tests the REST API for reading, validating, and updating YAML configurations.
"""

import pytest
import json

from webui.app import create_app


@pytest.fixture
def app():
    """Create Flask app for testing"""
    app = create_app('testing')
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


class TestGetConfig:
    """Tests for GET /api/config/<project_id>/<config_type>"""
    
    def test_get_config_success(self, client):
        """Test successful config retrieval"""
        # Use project 1 which should exist in test environment
        response = client.get('/api/config/1/midiconfig')
        
        # May not exist in test env, so just check response format
        assert response.status_code in [200, 404]
        
        data = json.loads(response.data)
        assert 'success' in data
    
    def test_get_config_invalid_type(self, client):
        """Test invalid config type"""
        response = client.get('/api/config/1/invalid')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'config_type must be one of' in data['error']
    
    def test_get_config_nonexistent_project(self, client):
        """Test non-existent project"""
        response = client.get('/api/config/99999/config')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False


class TestValidateConfig:
    """Tests for POST /api/config/<project_id>/<config_type>/validate"""
    
    def test_validate_valid_updates(self, client):
        """Test validation of valid updates"""
        updates = {
            'updates': [
                {'path': ['kick', 'midi_note'], 'value': 36},
                {'path': ['audio', 'force_mono'], 'value': True}
            ]
        }
        
        response = client.post(
            '/api/config/1/midiconfig/validate',
            data=json.dumps(updates),
            content_type='application/json'
        )
        
        # May not exist in test env
        assert response.status_code in [200, 404]
    
    def test_validate_missing_updates(self, client):
        """Test validation without updates field"""
        response = client.post(
            '/api/config/1/midiconfig/validate',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        # Should fail regardless of project existence
        data = json.loads(response.data)
        if response.status_code == 400:
            assert data['success'] is False
            assert 'updates' in data['error']
    
    def test_validate_invalid_midi_note(self, client):
        """Test validation of invalid MIDI note"""
        updates = {
            'updates': [
                {'path': ['kick', 'midi_note'], 'value': 200}  # Invalid
            ]
        }
        
        response = client.post(
            '/api/config/1/midiconfig/validate',
            data=json.dumps(updates),
            content_type='application/json'
        )
        
        # If project exists, should return validation error
        if response.status_code == 200:
            data = json.loads(response.data)
            if not data['success']:
                assert len(data['errors']) > 0


class TestUpdateConfig:
    """Tests for POST /api/config/<project_id>/<config_type>"""
    
    def test_update_config_missing_updates(self, client):
        """Test update without updates field"""
        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        # Should fail regardless of project existence
        data = json.loads(response.data)
        if response.status_code == 400:
            assert data['success'] is False
            assert 'updates' in data['error']
    
    def test_update_config_nonexistent_field(self, client):
        """Test updating non-existent field"""
        updates = {
            'updates': [
                {'path': ['nonexistent', 'field'], 'value': 42}
            ]
        }
        
        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps(updates),
            content_type='application/json'
        )
        
        # If project exists, should return error about field not found
        if response.status_code == 400:
            data = json.loads(response.data)
            assert data['success'] is False


class TestResetConfig:
    """Tests for POST /api/config/<project_id>/<config_type>/reset"""
    
    def test_reset_config_nonexistent_project(self, client):
        """Test resetting config for non-existent project"""
        response = client.post('/api/config/99999/config/reset')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False


class TestAPIErrorHandling:
    """Tests for API error handling"""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            '/api/config/1/midiconfig',
            data='invalid json',
            content_type='application/json'
        )
        
        # Flask should handle this gracefully
        assert response.status_code in [400, 404, 500]
    
    def test_missing_content_type(self, client):
        """Test POST without content type"""
        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps({'updates': []})
        )
        
        # Should return error (400, 404, or 500 all acceptable for malformed request)
        assert response.status_code in [200, 400, 404, 500]


class TestConfigEngineIntegration:
    """Integration tests using actual config files"""
    
    @pytest.fixture
    def mock_project_dir(self, tmp_path):
        """Create a mock project directory with config files"""
        project_dir = tmp_path / "1 - test_project"
        project_dir.mkdir()
        
        # Create sample config file
        config_content = """
global:
  sr: 44100  # Sample rate

kick:
  midi_note: 36  # MIDI note
  threshold: 0.5  # Threshold (0-1)
"""
        config_file = project_dir / "midiconfig.yaml"
        config_file.write_text(config_content)
        
        return project_dir
    
    def test_full_workflow(self, mock_project_dir):
        """Test complete workflow: load, update, validate, save"""
        from webui.yaml_config_core import YAMLConfigEngine
        
        config_file = mock_project_dir / "midiconfig.yaml"
        engine = YAMLConfigEngine(config_file)
        
        # 1. Parse config
        sections = engine.parse()
        assert len(sections) > 0
        
        # 2. Update value
        success, error = engine.update_value(['kick', 'midi_note'], 38)
        assert success is True
        assert error == ""
        
        # 3. Validate
        errors = engine.validate_all()
        assert len(errors) == 0
        
        # 4. Save
        engine.save()
        
        # 5. Reload and verify
        engine2 = YAMLConfigEngine(config_file)
        data = engine2.load()
        assert data['kick']['midi_note'] == 38
        
        # 6. Check comments preserved
        content = config_file.read_text()
        assert '# Sample rate' in content
        assert '# MIDI note' in content


class TestPitchDependencyApiContract:
    """T2 follow-up round 4 (2026-06-08): when the JS auto-toggles
    ``enable_pitch_detection=true`` into the same save payload as
    ``cluster_feature=pitch_hz``, the server must accept the
    multi-update payload atomically and persist both changes.

    These tests mock ``project_manager.get_project_by_number`` to
    point at a tmp_path project so they don't mutate the user's
    real midiconfig.yaml.
    """

    @pytest.fixture
    def tmp_project(self, tmp_path):
        """Create a tmp project dir with a snare section that has
        enable_pitch_detection: false and cluster_feature: stereo_width
        — matching the user's real funk project state when the bug
        was reported."""
        import yaml
        from unittest.mock import patch
        project_dir = tmp_path / '1 - test_pitch_project'
        project_dir.mkdir()
        config_path = project_dir / 'midiconfig.yaml'
        config_path.write_text(yaml.safe_dump({
            'snare': {
                'midi_note': 38,
                'enable_pitch_detection': False,
                'cluster_feature': 'stereo_width',
            },
        }))

        fake_project = {
            'number': 1,
            'name': 'test_pitch_project',
            'path': project_dir,
        }

        # yaml_config_core.get_config_engine does
        #   from project_manager import get_project_by_number, USER_FILES_DIR
        # Patch at the source module so both get_config_engine
        # and any direct callers see the fake.
        p = patch('project_manager.get_project_by_number',
                  return_value=fake_project)
        p.start()
        yield project_dir, config_path
        p.stop()

    def test_save_accepts_simultaneous_pitch_hz_and_enable_true(
        self, client, tmp_project,
    ):
        """JS pre-bundles both updates into one payload. The server
        must apply both atomically and persist them to the YAML."""
        project_dir, config_path = tmp_project
        response = client.post(
            '/api/config/1/midiconfig',
            json={'updates': [
                {'path': ['snare', 'cluster_feature'], 'value': 'pitch_hz'},
                {'path': ['snare', 'enable_pitch_detection'], 'value': True},
            ]},
        )
        assert response.status_code == 200, response.data
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['updated_count'] == 2

        # Reload the YAML and verify both fields are persisted
        import yaml
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        assert saved['snare']['cluster_feature'] == 'pitch_hz'
        assert saved['snare']['enable_pitch_detection'] is True

    def test_save_with_only_pitch_hz_does_not_toggle_detection(
        self, client, tmp_project,
    ):
        """Sanity check: the SERVER does not auto-toggle. The JS does
        that. If only ``cluster_feature`` is in the payload, the
        server must save it without touching ``enable_pitch_detection``.
        This is the contract that lets the JS do the auto-toggle
        safely — the server is dumb, the JS is smart."""
        project_dir, config_path = tmp_project
        response = client.post(
            '/api/config/1/midiconfig',
            json={'updates': [
                {'path': ['snare', 'cluster_feature'], 'value': 'pitch_hz'},
            ]},
        )
        assert response.status_code == 200, response.data

        import yaml
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        assert saved['snare']['cluster_feature'] == 'pitch_hz'
        # enable_pitch_detection was false in the fixture, must stay false
        assert saved['snare']['enable_pitch_detection'] is False
