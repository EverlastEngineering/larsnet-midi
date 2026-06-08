"""
Tests for Rebuild MIDI API endpoint.

Covers POST /api/rebuild-midi. These tests were missing entirely before
2026-06-08 — T3 e2e Playwright drive caught the gap, but the unit tests
didn't. Tests here assert:
  1. The route is registered and reachable.
  2. config_overrides plumb through to rebuild_midi_for_project.
  3. Validation errors return 4xx not 5xx.
  4. Stem-type subset is respected.
  5. Honor_overrides flag is forwarded.
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
    return create_app('testing')


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def funk_project(tmp_path):
    """
    A minimal project with stems + analysis.json, modeled on the user's
    real user_files/1 - 2_funk_80_beat_4-4_4/ layout.
    """
    project_path = tmp_path / '1 - 2_funk_80_beat_4-4_4'
    stems_dir = project_path / 'stems'
    midi_dir = project_path / 'midi'
    stems_dir.mkdir(parents=True)
    midi_dir.mkdir(parents=True)

    # Minimal WAV (just enough for any audio loaders to be happy in unit tests)
    (stems_dir / '2_funk_80_beat_4-4_4-kick.wav').write_bytes(b'RIFF\x00\x00\x00\x00WAVE')
    (stems_dir / '2_funk_80_beat_4-4_4-snare.wav').write_bytes(b'RIFF\x00\x00\x00\x00WAVE')

    # Per-project config (matches the user's actual project)
    (project_path / 'midiconfig.yaml').write_text(
        "kick:\n  geomean_threshold: 800.0\n  midi_note: 36\n"
        "snare:\n  geomean_threshold: 19.0\n  midi_note: 38\n"
    )

    # Minimal analysis.json (5 stems, 2 KEPT events each, with hihat_state on hihat)
    analysis = {
        'version': '3.0',
        'tempo_bpm': 120.0,
        'stems': {
            'kick': {
                'logic': {'geomean_threshold': 800.0},
                'events_configured': [
                    {'time': 1.0, 'status': 'KEPT', 'strength': 0.84, 'note': 36, 'velocity': 99,
                     'geomean': 514.89, 'sustain_ms': 140, 'pan_confidence': 0.0, 'stereo_width': 0.0, 'pitch_hz': None},
                ],
                'events_sensitive': [],
            },
            'snare': {
                'logic': {'geomean_threshold': 19.0},
                'events_configured': [
                    {'time': 1.5, 'status': 'KEPT', 'strength': 1.2, 'note': 38, 'velocity': 95,
                     'geomean': 200.0, 'sustain_ms': 80, 'pan_confidence': -0.1, 'stereo_width': 0.05, 'pitch_hz': 250.0},
                ],
                'events_sensitive': [],
            },
            'hihat': {
                'logic': {'open_geomean_min': 262.0, 'open_sustain_ms': 150.0},
                'events_configured': [
                    {'time': 2.0, 'status': 'KEPT', 'note': 42, 'velocity': 100,
                     'sustain_ms': 80, 'geomean': 300.0, 'hihat_state': 'closed',
                     'pan_confidence': 0.05, 'stereo_width': 0.1, 'pitch_hz': None},
                    {'time': 2.5, 'status': 'KEPT', 'note': 46, 'velocity': 110,
                     'sustain_ms': 200, 'geomean': 400.0, 'hihat_state': 'open',
                     'pan_confidence': 0.05, 'stereo_width': 0.1, 'pitch_hz': None},
                ],
                'events_sensitive': [],
            },
        },
    }
    with open(midi_dir / '2_funk_80_beat_4-4_4.analysis.json', 'w') as f:
        json.dump(analysis, f)

    return {
        'number': 1,
        'name': '2_funk_80_beat_4-4_4',
        'path': project_path,
        'created': datetime.now(),
        'metadata': {},
    }


# ─── Tests ───────────────────────────────────────────────────────────────


class TestRebuildMidiRouteExists:
    """T3 found the rebuild-midi endpoint was untested. This is the first
    test that should have existed: 'does the route exist at all?'"""

    @patch('webui.api.operations.get_project_by_number')
    @patch('stems_to_midi.rebuild_shell.rebuild_midi_for_project')
    def test_rebuild_midi_endpoint_returns_200(self, mock_rebuild, mock_get, client, funk_project):
        """POST /api/rebuild-midi with a valid project returns 200, not 404/500."""
        mock_get.return_value = funk_project
        mock_rebuild.return_value = {
            'success': True,
            'stems_rebuilt': ['kick', 'snare', 'hihat', 'toms', 'cymbals'],
            'elapsed_ms': 42,
            'analysis_data': {},
            'events_by_stem': {},
        }

        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({'project_number': 1}),
            content_type='application/json',
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.data}"
        )
        data = json.loads(response.data)
        assert data['success'] is True


class TestRebuildMidiConfigOverrides:
    """T3 found bug D: WebUI slider values must reach the server. The
    config_overrides kwarg is the contract. These tests assert the
    contract is honored."""

    @patch('webui.api.operations.get_project_by_number')
    @patch('stems_to_midi.rebuild_shell.rebuild_midi_for_project')
    def test_rebuild_midi_passes_config_overrides_through(self, mock_rebuild, mock_get, client, funk_project):
        """config_overrides in the request body must reach
        rebuild_midi_for_project as the config_overrides kwarg. If the
        route silently drops them, the WebUI slider values never make it
        to the rebuild — the user sees a filter change but the MIDI
        doesn't reflect it (T3 finding)."""
        mock_get.return_value = funk_project
        mock_rebuild.return_value = {
            'success': True, 'stems_rebuilt': ['kick'],
            'elapsed_ms': 10, 'analysis_data': {}, 'events_by_stem': {},
        }

        overrides = {
            'filtering.reverb_continuation_attack_threshold': 0.3,
            'kick.geomean_threshold': 600,
            'hihat.open_geomean_min': 200,
        }
        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({
                'project_number': 1,
                'config_overrides': overrides,
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        # The mock should have been called with config_overrides=overrides
        mock_rebuild.assert_called_once()
        call_kwargs = mock_rebuild.call_args.kwargs
        assert call_kwargs.get('config_overrides') == overrides, (
            f"config_overrides dropped! got: {call_kwargs}"
        )

    @patch('webui.api.operations.get_project_by_number')
    @patch('stems_to_midi.rebuild_shell.rebuild_midi_for_project')
    def test_rebuild_midi_passes_stem_types_through(self, mock_rebuild, mock_get, client, funk_project):
        """stem_types list is forwarded so a partial rebuild works."""
        mock_get.return_value = funk_project
        mock_rebuild.return_value = {
            'success': True, 'stems_rebuilt': ['kick'],
            'elapsed_ms': 5, 'analysis_data': {}, 'events_by_stem': {},
        }

        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({
                'project_number': 1,
                'stem_types': ['kick'],
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        call_kwargs = mock_rebuild.call_args.kwargs
        assert call_kwargs.get('stem_types') == ['kick']

    @patch('webui.api.operations.get_project_by_number')
    @patch('stems_to_midi.rebuild_shell.rebuild_midi_for_project')
    def test_rebuild_midi_passes_honor_overrides_through(self, mock_rebuild, mock_get, client, funk_project):
        """honor_overrides flag is forwarded (defaults to True)."""
        mock_get.return_value = funk_project
        mock_rebuild.return_value = {
            'success': True, 'stems_rebuilt': [], 'elapsed_ms': 1,
            'analysis_data': {}, 'events_by_stem': {},
        }

        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({
                'project_number': 1,
                'honor_overrides': False,
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        call_kwargs = mock_rebuild.call_args.kwargs
        assert call_kwargs.get('honor_overrides') is False


class TestRebuildMidiErrors:
    """T3 found that error returns were inconsistent. These tests lock in
    the contract: validation errors are 4xx, real errors are 5xx."""

    def test_missing_project_number_returns_400(self, client):
        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({}),
            content_type='application/json',
        )
        assert response.status_code == 400

    @patch('webui.api.operations.get_project_by_number')
    def test_project_not_found_returns_404(self, mock_get, client):
        mock_get.return_value = None
        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({'project_number': 999}),
            content_type='application/json',
        )
        assert response.status_code == 404

    @patch('webui.api.operations.get_project_by_number')
    def test_missing_analysis_returns_409_or_404(self, mock_get, client, funk_project):
        """T3 found: a project without analysis.json must not 500.
        Expected: 409 (conflict, needs full pipeline) per the route's
        existing 'requires_full_pipeline' handling, or 404 with a clear
        message. Anything 5xx is a regression."""
        from pathlib import Path as _P
        # Build a project whose midi/ has no analysis.json
        import shutil
        midi_dir = funk_project['path'] / 'midi'
        for f in midi_dir.glob('*.analysis.json'):
            f.unlink()
        mock_get.return_value = funk_project

        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({'project_number': 1}),
            content_type='application/json',
        )
        # 409 (needs full pipeline) is the documented contract;
        # 404 is acceptable if the route returns "no analysis" cleanly.
        # 500 is NOT acceptable.
        assert response.status_code in (404, 409), (
            f"Expected 404/409 for missing analysis, got {response.status_code}: "
            f"{response.data}"
        )


class TestRebuildMidiFailureResult:
    """When rebuild_midi_for_project returns success=False, the route must
    return 4xx (not 200, not 500). T3 found the 409-for-requires-full-pipeline
    branch but no test exercised it."""

    @patch('webui.api.operations.get_project_by_number')
    @patch('stems_to_midi.rebuild_shell.rebuild_midi_for_project')
    def test_requires_full_pipeline_returns_409(self, mock_rebuild, mock_get, client, funk_project):
        """If the rebuild result indicates the user must run the full
        pipeline first (e.g. analysis data is missing the right
        structure), return 409 Conflict. Not 200, not 500."""
        mock_get.return_value = funk_project
        mock_rebuild.return_value = {
            'success': False,
            'requires_full_pipeline': True,
            'error': 'Analysis not in expected structure — run full pipeline first',
        }

        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({'project_number': 1}),
            content_type='application/json',
        )
        assert response.status_code == 409, (
            f"Expected 409 for requires_full_pipeline, got {response.status_code}: "
            f"{response.data}"
        )
        data = json.loads(response.data)
        assert data.get('requires_full_pipeline') is True


class TestRebuildMidiDataIntegrityWarnings:
    """T2's bug C: events_configured ⊆ events_sensitive validation. The
    rebuild endpoint must surface validation warnings in its response."""

    @patch('webui.api.operations.get_project_by_number')
    @patch('stems_to_midi.rebuild_shell.rebuild_midi_for_project')
    def test_data_integrity_warnings_in_response(self, mock_rebuild, mock_get, client, funk_project):
        """If rebuild_midi_for_project emits data integrity warnings, the
        response must include them in 'data_integrity_warnings' so the
        WebUI can toast them (T2 bug C)."""
        mock_get.return_value = funk_project
        mock_rebuild.return_value = {
            'success': True,
            'stems_rebuilt': ['kick'],
            'elapsed_ms': 5,
            'analysis_data': {},
            'events_by_stem': {},
            'data_integrity_warnings': [
                'kick: 1 event in events_configured has no matching time in events_sensitive',
            ],
        }

        response = client.post(
            '/api/rebuild-midi',
            data=json.dumps({'project_number': 1}),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'data_integrity_warnings' in data
        assert len(data['data_integrity_warnings']) >= 1
