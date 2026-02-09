"""
Tests for Detection Analysis API Endpoints

Tests GET /api/projects/:number/analysis and
      GET /api/projects/:number/envelope/:stem_type

Run with: pytest webui/test_analysis_api.py
"""

import pytest
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from webui.app import create_app


@pytest.fixture
def app():
    """Create test Flask app."""
    return create_app('testing')


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def project_with_analysis(tmp_path):
    """Create a mock project directory with analysis.json and envelope .npz files."""
    project_path = tmp_path / '1 - Test Song'
    midi_dir = project_path / 'midi'
    midi_dir.mkdir(parents=True)

    # Create analysis.json (v3 format)
    analysis = {
        'version': '3.0',
        'tempo_bpm': 120.0,
        'stems': {
            'kick': {
                'logic': {
                    'geomean_threshold': 70.0,
                    'min_sustain_ms': None,
                    'passes': ['geomean']
                },
                'events_configured': [
                    {'time': 1.0, 'status': 'KEPT', 'strength': 1.5, 'note': 36, 'velocity': 100},
                    {'time': 2.0, 'status': 'FILTERED', 'strength': 0.3, 'geomean': 45.0},
                ],
                'events_sensitive': [
                    {'time': 0.5, 'status': 'KEPT', 'strength': 0.1},
                    {'time': 1.0, 'status': 'KEPT', 'strength': 1.5},
                    {'time': 1.5, 'status': 'KEPT', 'strength': 0.2},
                    {'time': 2.0, 'status': 'KEPT', 'strength': 0.3},
                ],
            },
            'snare': {
                'logic': {'geomean_threshold': 60.0, 'passes': ['geomean']},
                'events_configured': [
                    {'time': 1.5, 'status': 'KEPT', 'strength': 2.0, 'note': 38, 'velocity': 110},
                ],
                'events_sensitive': [],
            }
        }
    }
    with open(midi_dir / 'Test_Song.analysis.json', 'w') as f:
        json.dump(analysis, f)

    # Create envelope .npz for kick
    times = np.linspace(0, 3, 300, dtype=np.float32)
    left = np.random.rand(300).astype(np.float32)
    right = np.random.rand(300).astype(np.float32)
    np.savez_compressed(
        midi_dir / 'Test_Song.kick.envelope.npz',
        times=times, left=left, right=right,
        sr=np.array(44100), hop_length=np.array(512), method=np.array('rms')
    )

    return {
        'number': 1,
        'name': 'Test Song',
        'path': project_path,
        'created': datetime.now(),
        'metadata': {},
    }


@pytest.fixture
def project_without_analysis(tmp_path):
    """Create a mock project with no analysis data."""
    project_path = tmp_path / '2 - No Analysis'
    midi_dir = project_path / 'midi'
    midi_dir.mkdir(parents=True)
    # Create a .mid file but no analysis
    (midi_dir / 'No_Analysis.mid').write_bytes(b'')

    return {
        'number': 2,
        'name': 'No Analysis',
        'path': project_path,
        'created': datetime.now(),
        'metadata': {},
    }


class TestProjectDetailAnalysisFlags:
    """Test that project detail includes has_analysis and envelope_stems."""

    @patch('webui.api.projects.get_project_by_number')
    def test_project_with_analysis_flags(self, mock_get, client, project_with_analysis):
        """Project detail response includes has_analysis=true and envelope_stems."""
        mock_get.return_value = project_with_analysis
        response = client.get('/api/projects/1')

        assert response.status_code == 200
        data = json.loads(response.data)
        project = data['project']
        assert project['has_analysis'] is True
        assert 'kick' in project['envelope_stems']

    @patch('webui.api.projects.get_project_by_number')
    def test_project_without_analysis_flags(self, mock_get, client, project_without_analysis):
        """Project detail response includes has_analysis=false when no analysis.json."""
        mock_get.return_value = project_without_analysis
        response = client.get('/api/projects/2')

        assert response.status_code == 200
        data = json.loads(response.data)
        project = data['project']
        assert project['has_analysis'] is False
        assert project['envelope_stems'] == []


class TestAnalysisEndpoint:
    """Test GET /api/projects/:number/analysis."""

    @patch('webui.api.projects.get_project_by_number')
    def test_get_analysis_success(self, mock_get, client, project_with_analysis):
        """Returns analysis.json content."""
        mock_get.return_value = project_with_analysis
        response = client.get('/api/projects/1/analysis')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['version'] == '3.0'
        assert 'kick' in data['stems']
        assert 'snare' in data['stems']
        assert len(data['stems']['kick']['events_configured']) == 2
        assert len(data['stems']['kick']['events_sensitive']) == 4

    @patch('webui.api.projects.get_project_by_number')
    def test_get_analysis_not_found(self, mock_get, client, project_without_analysis):
        """Returns 404 when no analysis.json exists."""
        mock_get.return_value = project_without_analysis
        response = client.get('/api/projects/2/analysis')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

    @patch('webui.api.projects.get_project_by_number')
    def test_get_analysis_project_not_found(self, mock_get, client):
        """Returns 404 when project doesn't exist."""
        mock_get.return_value = None
        response = client.get('/api/projects/999/analysis')
        assert response.status_code == 404

    @patch('webui.api.projects.get_project_by_number')
    def test_get_analysis_no_midi_dir(self, mock_get, client, tmp_path):
        """Returns 404 when project has no midi directory."""
        project_path = tmp_path / '3 - Empty'
        project_path.mkdir(parents=True)
        mock_get.return_value = {
            'number': 3, 'name': 'Empty', 'path': project_path,
            'created': datetime.now(), 'metadata': {},
        }
        response = client.get('/api/projects/3/analysis')
        assert response.status_code == 404


class TestEnvelopeEndpoint:
    """Test GET /api/projects/:number/envelope/:stem_type."""

    @patch('webui.api.projects.get_project_by_number')
    def test_get_envelope_success(self, mock_get, client, project_with_analysis):
        """Returns envelope data as JSON arrays."""
        mock_get.return_value = project_with_analysis
        response = client.get('/api/projects/1/envelope/kick')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['stem_type'] == 'kick'
        assert isinstance(data['times'], list)
        assert isinstance(data['left'], list)
        assert isinstance(data['right'], list)
        assert data['sr'] == 44100
        assert data['hop_length'] == 512
        assert data['method'] == 'rms'
        assert data['sample_count'] == len(data['times'])
        # Should be 300 points (below 2000 threshold, no downsampling)
        assert data['sample_count'] == 300

    @patch('webui.api.projects.get_project_by_number')
    def test_get_envelope_downsampling(self, mock_get, client, tmp_path):
        """Downsamples large envelope arrays to ~8000 points."""
        project_path = tmp_path / '4 - Long Song'
        midi_dir = project_path / 'midi'
        midi_dir.mkdir(parents=True)

        # Create large envelope (30K points ~ 6 minute song at 86fps)
        n = 30000
        times = np.linspace(0, 350, n, dtype=np.float32)
        left = np.random.rand(n).astype(np.float32)
        right = np.random.rand(n).astype(np.float32)
        np.savez_compressed(
            midi_dir / 'Long.kick.envelope.npz',
            times=times, left=left, right=right,
            sr=np.array(44100), hop_length=np.array(512), method=np.array('peak_hold')
        )

        mock_get.return_value = {
            'number': 4, 'name': 'Long Song', 'path': project_path,
            'created': datetime.now(), 'metadata': {},
        }

        response = client.get('/api/projects/4/envelope/kick')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['sample_count'] == 8000

    @patch('webui.api.projects.get_project_by_number')
    def test_get_envelope_stem_not_found(self, mock_get, client, project_with_analysis):
        """Returns 404 when envelope for requested stem doesn't exist."""
        mock_get.return_value = project_with_analysis
        response = client.get('/api/projects/1/envelope/snare')  # No snare .npz

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

    @patch('webui.api.projects.get_project_by_number')
    def test_get_envelope_invalid_stem(self, mock_get, client, project_with_analysis):
        """Returns 400 for invalid stem type."""
        mock_get.return_value = project_with_analysis
        response = client.get('/api/projects/1/envelope/invalid')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    @patch('webui.api.projects.get_project_by_number')
    def test_get_envelope_project_not_found(self, mock_get, client):
        """Returns 404 when project doesn't exist."""
        mock_get.return_value = None
        response = client.get('/api/projects/999/envelope/kick')
        assert response.status_code == 404

    @patch('webui.api.projects.get_project_by_number')
    def test_get_envelope_values_are_numeric(self, mock_get, client, project_with_analysis):
        """All returned array values are valid JSON numbers."""
        mock_get.return_value = project_with_analysis
        response = client.get('/api/projects/1/envelope/kick')

        assert response.status_code == 200
        data = json.loads(response.data)
        for v in data['times']:
            assert isinstance(v, (int, float))
        for v in data['left']:
            assert isinstance(v, (int, float))
        for v in data['right']:
            assert isinstance(v, (int, float))
