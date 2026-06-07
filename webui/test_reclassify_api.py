"""
Tests for Reclassify API Endpoint

Tests POST /api/reclassify for note classification preview.

Run with: pytest webui/test_reclassify_api.py
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
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
def hihat_analysis(tmp_path):
    """Create a mock project with hihat analysis data including spectral features."""
    project_path = tmp_path / '1 - Test Song'
    midi_dir = project_path / 'midi'
    midi_dir.mkdir(parents=True)

    # Create a minimal midiconfig.yaml
    config_path = project_path / 'midiconfig.yaml'
    config_path.write_text(
        "hihat:\n"
        "  open_geomean_min: 262.0\n"
        "  open_sustain_ms: 150.0\n"
        "  midi_note_closed: 42\n"
        "  midi_note_open: 46\n"
    )

    analysis = {
        'version': '3.0',
        'tempo_bpm': 120.0,
        'stems': {
            'hihat': {
                'logic': {
                    'geomean_threshold': 70.0,
                    'passes': ['geomean'],
                    'open_geomean_min': 262.0,
                    'open_sustain_ms': 150.0,
                },
                'events_configured': [
                    {
                        'time': 1.0, 'status': 'KEPT', 'strength': 1.5,
                        'note': 42, 'velocity': 100,
                        'spectral_centroid_hz': 200.0, 'sustain_ms': 80.0,
                        'geomean': 300.0,
                        'hihat_state': 'closed',
                    },
                    {
                        'time': 2.0, 'status': 'KEPT', 'strength': 2.0,
                        'note': 46, 'velocity': 110,
                        'spectral_centroid_hz': 500.0, 'sustain_ms': 200.0,
                        'geomean': 400.0,
                        'hihat_state': 'open',
                    },
                    {
                        'time': 3.0, 'status': 'FILTERED', 'strength': 0.3,
                        'geomean': 45.0,
                    },
                ],
                'events_sensitive': [],
            },
        },
    }
    with open(midi_dir / 'Test_Song.analysis.json', 'w') as f:
        json.dump(analysis, f)

    return {
        'number': 1,
        'name': 'Test Song',
        'path': project_path,
        'created': datetime.now(),
        'metadata': {},
    }


@pytest.fixture
def empty_stem_analysis(tmp_path):
    """Create a mock project where all events are FILTERED."""
    project_path = tmp_path / '2 - Empty'
    midi_dir = project_path / 'midi'
    midi_dir.mkdir(parents=True)

    config_path = project_path / 'midiconfig.yaml'
    config_path.write_text("hihat:\n  open_geomean_min: 262.0\n")

    analysis = {
        'version': '3.0',
        'stems': {
            'hihat': {
                'logic': {},
                'events_configured': [
                    {'time': 1.0, 'status': 'FILTERED', 'strength': 0.2},
                ],
                'events_sensitive': [],
            },
        },
    }
    with open(midi_dir / 'Empty.analysis.json', 'w') as f:
        json.dump(analysis, f)

    return {
        'number': 2,
        'name': 'Empty',
        'path': project_path,
        'created': datetime.now(),
        'metadata': {},
    }


@pytest.fixture
def snare_analysis(tmp_path):
    """Create a mock project with snare analysis data including spectral features."""
    project_path = tmp_path / '3 - Snare Song'
    midi_dir = project_path / 'midi'
    midi_dir.mkdir(parents=True)

    config_path = project_path / 'midiconfig.yaml'
    config_path.write_text(
        "snare:\n"
        "  expected_clusters: 1\n"
        "  midi_note: 38\n"
        "  midi_note_rimshot: 37\n"
        "  midi_note_clap: 39\n"
    )

    analysis = {
        'version': '3.0',
        'stems': {
            'snare': {
                'logic': {
                    'geomean_threshold': 40.0,
                    'expected_clusters': 1,
                },
                'events_configured': [
                    {
                        'time': 1.0, 'status': 'KEPT', 'strength': 1.5,
                        'note': 38, 'velocity': 100,
                        'spectral_centroid_hz': 300.0, 'geomean': 200.0,
                    },
                    {
                        'time': 2.0, 'status': 'KEPT', 'strength': 2.0,
                        'note': 38, 'velocity': 110,
                        'spectral_centroid_hz': 5000.0, 'geomean': 300.0,
                    },
                    {
                        'time': 3.0, 'status': 'KEPT', 'strength': 1.8,
                        'note': 38, 'velocity': 105,
                        'spectral_centroid_hz': 800.0, 'geomean': 250.0,
                    },
                ],
                'events_sensitive': [],
            },
        },
    }
    with open(midi_dir / 'Snare_Song.analysis.json', 'w') as f:
        json.dump(analysis, f)

    return {
        'number': 3,
        'name': 'Snare Song',
        'path': project_path,
        'created': datetime.now(),
        'metadata': {},
    }


class TestReclassifyValidation:
    """Test request validation for POST /api/reclassify."""

    def test_missing_body(self, client):
        """Returns 400 when request body has no required fields."""
        response = client.post(
            '/api/reclassify',
            data=json.dumps({}),
            content_type='application/json',
        )
        assert response.status_code == 400

    def test_missing_project_number(self, client):
        """Returns 400 when project_number is missing."""
        response = client.post(
            '/api/reclassify',
            data=json.dumps({'stem_type': 'hihat'}),
            content_type='application/json',
        )
        assert response.status_code == 400

    def test_missing_stem_type(self, client):
        """Returns 400 when stem_type is missing."""
        response = client.post(
            '/api/reclassify',
            data=json.dumps({'project_number': 1}),
            content_type='application/json',
        )
        assert response.status_code == 400

    @patch('webui.api.operations.get_project_by_number')
    def test_project_not_found(self, mock_get, client):
        """Returns 404 when project does not exist."""
        mock_get.return_value = None
        response = client.post(
            '/api/reclassify',
            data=json.dumps({'project_number': 999, 'stem_type': 'hihat'}),
            content_type='application/json',
        )
        assert response.status_code == 404

    @patch('webui.api.operations.get_project_by_number')
    def test_no_analysis_file(self, mock_get, client, tmp_path):
        """Returns 404 when analysis.json does not exist."""
        project_path = tmp_path / '3 - No Analysis'
        midi_dir = project_path / 'midi'
        midi_dir.mkdir(parents=True)

        mock_get.return_value = {
            'number': 3, 'name': 'No Analysis', 'path': project_path,
            'created': datetime.now(), 'metadata': {},
        }
        response = client.post(
            '/api/reclassify',
            data=json.dumps({'project_number': 3, 'stem_type': 'hihat'}),
            content_type='application/json',
        )
        assert response.status_code == 404

    @patch('webui.api.operations.get_project_by_number')
    def test_stem_not_in_analysis(self, mock_get, client, hihat_analysis):
        """Returns 404 when requested stem is not in analysis data."""
        mock_get.return_value = hihat_analysis
        response = client.post(
            '/api/reclassify',
            data=json.dumps({'project_number': 1, 'stem_type': 'kick'}),
            content_type='application/json',
        )
        assert response.status_code == 404


class TestReclassifyHihat:
    """Test hihat reclassification with config overrides."""

    @patch('webui.api.operations.get_project_by_number')
    def test_reclassify_returns_only_kept_events(self, mock_get, client, hihat_analysis):
        """Only KEPT events are returned — FILTERED events are excluded."""
        mock_get.return_value = hihat_analysis
        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 1,
                'stem_type': 'hihat',
                'config_overrides': {},
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['events']) == 2  # Only the 2 KEPT events

    @patch('webui.api.operations.get_project_by_number')
    def test_reclassify_returns_time_and_note(self, mock_get, client, hihat_analysis):
        """Each event has time and note fields."""
        mock_get.return_value = hihat_analysis
        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 1,
                'stem_type': 'hihat',
                'config_overrides': {},
            }),
            content_type='application/json',
        )
        data = json.loads(response.data)
        for event in data['events']:
            assert 'time' in event
            assert 'note' in event

    @patch('webui.api.operations.get_project_by_number')
    def test_reclassify_hihat_state_present(self, mock_get, client, hihat_analysis):
        """Hihat events include hihat_state field."""
        mock_get.return_value = hihat_analysis
        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 1,
                'stem_type': 'hihat',
                'config_overrides': {},
            }),
            content_type='application/json',
        )
        data = json.loads(response.data)
        states = {e.get('hihat_state') for e in data['events']}
        # Both open and closed should be present in our fixture data
        assert 'open' in states or 'closed' in states

    @patch('webui.api.operations.get_project_by_number')
    def test_config_overrides_affect_classification(self, mock_get, client, hihat_analysis):
        """Changing open_geomean_min threshold changes which notes are open vs closed.

        Fixture event 1: centroid=200, sustain=80 (closed by default thresholds)
        Fixture event 2: centroid=500, sustain=200 (open by default thresholds)

        Lowering open_geomean_min to 100 and open_sustain_ms to 50 should make
        event 1 also classify as open (centroid 200 >= 100 AND sustain 80 >= 50).
        """
        mock_get.return_value = hihat_analysis

        # With very low thresholds, both events should be open
        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 1,
                'stem_type': 'hihat',
                'config_overrides': {
                    'open_geomean_min': 100,
                    'open_sustain_ms': 50,
                },
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        notes = [e['note'] for e in data['events']]
        # With such low thresholds, both events should be open hihat (note 46)
        assert all(n == 46 for n in notes), f"Expected all open (46), got {notes}"

    @patch('webui.api.operations.get_project_by_number')
    def test_high_thresholds_all_closed(self, mock_get, client, hihat_analysis):
        """With very high thresholds, all events should be closed."""
        mock_get.return_value = hihat_analysis

        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 1,
                'stem_type': 'hihat',
                'config_overrides': {
                    'open_geomean_min': 999,
                    'open_sustain_ms': 999,
                },
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        notes = [e['note'] for e in data['events']]
        # All events should be closed hihat (note 42)
        assert all(n == 42 for n in notes), f"Expected all closed (42), got {notes}"

    @patch('webui.api.operations.get_project_by_number')
    def test_empty_kept_returns_empty_events(self, mock_get, client, empty_stem_analysis):
        """When no KEPT events exist, returns empty events list."""
        mock_get.return_value = empty_stem_analysis
        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 2,
                'stem_type': 'hihat',
                'config_overrides': {},
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['events'] == []

    @patch('webui.api.operations.get_project_by_number')
    def test_no_disk_write(self, mock_get, client, hihat_analysis):
        """Reclassify is a preview-only operation — analysis.json should not change."""
        mock_get.return_value = hihat_analysis
        analysis_file = list((hihat_analysis['path'] / 'midi').glob('*.analysis.json'))[0]
        original_content = analysis_file.read_text()

        client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 1,
                'stem_type': 'hihat',
                'config_overrides': {'open_geomean_min': 100},
            }),
            content_type='application/json',
        )

        assert analysis_file.read_text() == original_content


class TestReclassifySettingsSchema:
    """Test that classification settings are in the schema."""

    def test_open_geomean_min_in_schema(self):
        """hihat_open_geomean_min exists in settings registry."""
        from webui.settings_schema import get_setting_by_key
        setting = get_setting_by_key('hihat_open_geomean_min')
        assert setting is not None
        assert setting.default == 262.0
        assert setting.yaml_path == ['hihat', 'open_geomean_min']

    def test_open_sustain_ms_in_schema(self):
        """hihat_open_sustain_ms exists in settings registry."""
        from webui.settings_schema import get_setting_by_key
        setting = get_setting_by_key('hihat_open_sustain_ms')
        assert setting is not None
        # Aligned with midiconfig.yaml default (100.0); previously was 150.0
        # but YAML always overrode, so this fix removes silent drift.
        assert setting.default == 100.0
        assert setting.yaml_path == ['hihat', 'open_sustain_ms']

    def test_open_geomean_min_validation(self):
        """hihat_open_geomean_min validates within range."""
        from webui.settings_schema import get_setting_by_key
        setting = get_setting_by_key('hihat_open_geomean_min')

        is_valid, _ = setting.validate(262.0)
        assert is_valid

        is_valid, _ = setting.validate(0.0)
        assert not is_valid

        is_valid, _ = setting.validate(2000.0)
        assert not is_valid

    def test_open_sustain_ms_validation(self):
        """hihat_open_sustain_ms validates within range."""
        from webui.settings_schema import get_setting_by_key
        setting = get_setting_by_key('hihat_open_sustain_ms')

        is_valid, _ = setting.validate(150.0)
        assert is_valid

        is_valid, _ = setting.validate(5.0)
        assert not is_valid

        is_valid, _ = setting.validate(1000.0)
        assert not is_valid

    def test_snare_expected_clusters_in_schema(self):
        """snare_expected_clusters exists in settings registry."""
        from webui.settings_schema import get_setting_by_key
        setting = get_setting_by_key('snare_expected_clusters')
        assert setting is not None
        assert setting.default == 1
        assert setting.yaml_path == ['snare', 'expected_clusters']

    def test_snare_expected_clusters_validation(self):
        """snare_expected_clusters validates within range."""
        from webui.settings_schema import get_setting_by_key
        setting = get_setting_by_key('snare_expected_clusters')

        is_valid, _ = setting.validate(1)
        assert is_valid

        is_valid, _ = setting.validate(4)
        assert is_valid

        is_valid, _ = setting.validate(0)
        assert not is_valid

        is_valid, _ = setting.validate(6)
        assert not is_valid


class TestReclassifySnare:
    """Test snare reclassification via POST /api/reclassify."""

    @patch('webui.api.operations.get_project_by_number')
    def test_default_clusters_all_snare(self, mock_get, client, snare_analysis):
        """expected_clusters=1 (default) → all events are note 38."""
        mock_get.return_value = snare_analysis
        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 3,
                'stem_type': 'snare',
                'config_overrides': {},
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        notes = [e['note'] for e in data['events']]
        assert all(n == 38 for n in notes), f'Expected all note 38, got {notes}'

    @patch('webui.api.operations.get_project_by_number')
    def test_override_clusters_3(self, mock_get, client, snare_analysis):
        """expected_clusters=3 → events split into up to 3 sub-types."""
        mock_get.return_value = snare_analysis
        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 3,
                'stem_type': 'snare',
                'config_overrides': {'expected_clusters': 3},
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        notes = set(e['note'] for e in data['events'])
        # With 3 clusters and 3 distinct spectral values, expect multiple sub-types
        assert len(notes) >= 2, f'Expected multiple note types, got {notes}'
        # All notes must be valid snare family
        valid_snare_notes = {38, 37, 39, 40}
        assert notes.issubset(valid_snare_notes), f'Unexpected notes: {notes - valid_snare_notes}'

    @patch('webui.api.operations.get_project_by_number')
    def test_reclassify_returns_classification(self, mock_get, client, snare_analysis):
        """Reclassified snare events include classification field."""
        mock_get.return_value = snare_analysis
        response = client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 3,
                'stem_type': 'snare',
                'config_overrides': {'expected_clusters': 3},
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        for event in data['events']:
            assert 'classification' in event, f'Missing classification in event: {event}'
            assert isinstance(event['classification'], int)

    @patch('webui.api.operations.get_project_by_number')
    def test_snare_no_disk_write(self, mock_get, client, snare_analysis):
        """Snare reclassify is preview-only — analysis.json should not change."""
        mock_get.return_value = snare_analysis
        analysis_file = list((snare_analysis['path'] / 'midi').glob('*.analysis.json'))[0]
        original_content = analysis_file.read_text()

        client.post(
            '/api/reclassify',
            data=json.dumps({
                'project_number': 3,
                'stem_type': 'snare',
                'config_overrides': {'expected_clusters': 3},
            }),
            content_type='application/json',
        )

        assert analysis_file.read_text() == original_content
