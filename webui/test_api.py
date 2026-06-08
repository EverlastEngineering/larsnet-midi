"""
Unit Tests for DrumToMIDI Web UI API

Tests all API endpoints with mocked project_manager functions.
Run with: pytest webui/test_api.py
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webui.app import create_app
from webui.jobs import JobQueue, JobStatus


@pytest.fixture
def app():
    """Create test Flask app"""
    app = create_app('testing')
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def mock_project():
    """Mock project data"""
    return {
        'number': 1,
        'name': 'Test Song',
        'path': Path('/app/user_files/1 - Test Song'),
        'created': datetime.now(),
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'audio_file': 'test.wav'
        }
    }


class TestProjectsAPI:
    """Test projects endpoints"""
    
    @patch('webui.api.projects.discover_projects')
    def test_list_projects(self, mock_discover, client, mock_project):
        """Test GET /api/projects"""
        # Setup mock
        mock_discover.return_value = [mock_project]
        
        # Make request
        response = client.get('/api/projects')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'projects' in data
        assert len(data['projects']) == 1
        assert data['projects'][0]['number'] == 1
        assert data['projects'][0]['name'] == 'Test Song'
    
    @patch('webui.api.projects.get_project_by_number')
    def test_get_project_found(self, mock_get_project, client, mock_project):
        """Test GET /api/projects/:id when project exists"""
        mock_get_project.return_value = mock_project
        
        response = client.get('/api/projects/1')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'project' in data
        assert data['project']['number'] == 1
    
    @patch('webui.api.projects.get_project_by_number')
    def test_get_project_not_found(self, mock_get_project, client):
        """Test GET /api/projects/:id when project doesn't exist"""
        mock_get_project.return_value = None
        
        response = client.get('/api/projects/999')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


class TestUploadAPI:
    """Test upload endpoint"""
    
    @patch('webui.api.upload.create_project')
    def test_upload_success(self, mock_create_project, client, mock_project, tmp_path):
        """Test POST /api/upload with valid file"""
        mock_create_project.return_value = mock_project
        
        # Create test file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"fake wav data")
        
        with open(test_file, 'rb') as f:
            response = client.post(
                '/api/upload',
                data={'file': (f, 'test.wav')},
                content_type='multipart/form-data'
            )
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'project' in data
        assert data['project']['number'] == 1
    
    def test_upload_no_file(self, client):
        """Test POST /api/upload without file"""
        response = client.post('/api/upload')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_upload_invalid_extension(self, client, tmp_path):
        """Test POST /api/upload with invalid file type"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("not audio")
        
        with open(test_file, 'rb') as f:
            response = client.post(
                '/api/upload',
                data={'file': (f, 'test.txt')},
                content_type='multipart/form-data'
            )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid file type' in data['error']


class TestOperationsAPI:
    """Test operation endpoints"""
    
    @patch('webui.api.operations.get_project_by_number')
    @patch('webui.api.operations.get_job_queue')
    def test_separate_success(self, mock_get_queue, mock_get_project, client, mock_project):
        """Test POST /api/separate"""
        mock_get_project.return_value = mock_project
        mock_queue = Mock()
        mock_queue.submit.return_value = 'job-123'
        mock_get_queue.return_value = mock_queue
        
        response = client.post(
            '/api/separate',
            data=json.dumps({
                'project_number': 1,
                'device': 'cpu'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 202
        data = json.loads(response.data)
        assert 'job_id' in data
        assert data['job_id'] == 'job-123'
    
    @patch('webui.api.operations.get_project_by_number')
    def test_separate_project_not_found(self, mock_get_project, client):
        """Test POST /api/separate with non-existent project"""
        mock_get_project.return_value = None
        
        response = client.post(
            '/api/separate',
            data=json.dumps({'project_number': 999}),
            content_type='application/json'
        )
        
        assert response.status_code == 404
    
    def test_separate_invalid_device(self, client):
        """Test POST /api/separate with invalid device"""
        response = client.post(
            '/api/separate',
            data=json.dumps({
                'project_number': 1,
                'device': 'quantum'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid device' in data['error']
    
    @patch('webui.api.operations.get_project_by_number')
    @patch('webui.api.operations.get_job_queue')
    def test_cleanup(self, mock_get_queue, mock_get_project, client, mock_project):
        """Test POST /api/cleanup"""
        mock_get_project.return_value = mock_project
        mock_queue = Mock()
        mock_queue.submit.return_value = 'job-456'
        mock_get_queue.return_value = mock_queue
        
        response = client.post(
            '/api/cleanup',
            data=json.dumps({'project_number': 1}),
            content_type='application/json'
        )
        
        assert response.status_code == 202
        data = json.loads(response.data)
        assert data['job_id'] == 'job-456'
    
    @patch('webui.api.operations.get_project_by_number')
    @patch('webui.api.operations.get_job_queue')
    def test_stems_to_midi(self, mock_get_queue, mock_get_project, client, mock_project):
        """Test POST /api/stems-to-midi (legacy, mocks at queue level).

        NOTE: This test passes by mocking job_queue.submit() and never
        invoking run_stems_to_midi. It does NOT catch the route→work-function
        kwarg drift. See test_stems_to_midi_forwards_config_overrides and
        test_stems_to_midi_does_not_pass_stale_detection_kwargs for the
        contract that catches that bug.
        """
        mock_get_project.return_value = mock_project
        mock_queue = Mock()
        mock_queue.submit.return_value = 'job-789'
        mock_get_queue.return_value = mock_queue

        response = client.post(
            '/api/stems-to-midi',
            data=json.dumps({
                'project_number': 1,
                'onset_threshold': 0.3
            }),
            content_type='application/json'
        )

        assert response.status_code == 202
        data = json.loads(response.data)
        assert data['job_id'] == 'job-789'


class TestStemsToMidiKwargsContract:
    """T2 follow-up (2026-06-08): stems-to-midi route was blindly splatting
    request body into run_stems_to_midi(**kwargs), which forwarded them to
    stems_to_midi_for_project(**kwargs) — but T1's drift fix changed
    stems_to_midi_for_project's signature to take only (project, config,
    stems_to_process, max_duration, learning_mode). The stale kwargs
    (onset_threshold, onset_delta, onset_wait, hop_length, min_velocity,
    max_velocity, tempo, detect_hihat_open, etc.) hit the function and
    raised TypeError. The user saw this as a 500 toast in the WebUI.

    These tests mock at the WORK FUNCTION level (not the queue) so the
    real call path is exercised, and assert the contract: only the
    function's actual parameters are passed.
    """

    @patch('webui.api.operations.get_project_by_number')
    @patch('webui.api.operations.run_stems_to_midi')
    @patch('webui.api.operations.get_job_queue')
    def test_stems_to_midi_does_not_pass_stale_detection_kwargs(
        self, mock_get_queue, mock_run, mock_get_project, client, mock_project
    ):
        """The post-T1 stems_to_midi_for_project signature is
        (project, config, stems_to_process, max_duration, learning_mode).
        The route must NOT pass onset_threshold, onset_delta, onset_wait,
        hop_length, min_velocity, max_velocity, tempo, or detect_hihat_open
        as kwargs — those were the OLD signature, removed by the T1 drift fix.
        If the route still splats them, stems_to_midi_for_project raises
        TypeError at runtime."""
        mock_get_project.return_value = mock_project
        # Don't execute the work function (would hit the real
        # stems_to_midi_for_project). Just record the call.
        mock_get_queue.return_value = Mock(submit=Mock(return_value='job-x'))

        # Simulate the JS sending a body that includes the stale kwargs
        response = client.post(
            '/api/stems-to-midi',
            data=json.dumps({
                'project_number': 1,
                'onset_threshold': 0.3,
                'onset_delta': 0.01,
                'onset_wait': 3,
                'hop_length': 512,
                'min_velocity': 80,
                'max_velocity': 110,
                'tempo': 120.0,
                'detect_hihat_open': False,
            }),
            content_type='application/json',
        )
        assert response.status_code == 202

        # Extract the work function call from job_queue.submit kwargs
        # OR, since we mocked at run_stems_to_midi level, the route body
        # should NOT have called run_stems_to_midi with the stale kwargs.
        # The contract: when we eventually inspect what the work fn got,
        # none of the stale kwargs may be present.
        # Since this test patches run_stems_to_midi but the route still
        # uses job_queue.submit(func=run_stems_to_midi, **kwargs), the
        # submission captures the kwargs. Pull them out:
        submit_call = mock_get_queue.return_value.submit.call_args
        submit_kwargs = submit_call.kwargs
        # The function reference must be the work function
        assert submit_kwargs.get('func') is mock_run, (
            f"Expected func=run_stems_to_midi, got: {submit_kwargs}"
        )
        # No stale detection kwargs may reach the work function
        for stale in ('onset_threshold', 'onset_delta', 'onset_wait',
                      'hop_length', 'min_velocity', 'max_velocity',
                      'tempo', 'detect_hihat_open'):
            assert stale not in submit_kwargs, (
                f"Stale kwarg {stale!r} leaked to run_stems_to_midi: {submit_kwargs}"
            )

    @patch('webui.api.operations.get_project_by_number')
    @patch('webui.api.operations.run_stems_to_midi')
    @patch('webui.api.operations.get_job_queue')
    def test_stems_to_midi_forwards_modern_request_shape(
        self, mock_get_queue, mock_run, mock_get_project, client, mock_project
    ):
        """Modern request shape (post-T1): config_overrides as a dict of
        dotted YAML paths. The route must forward config_overrides, not
        flatten the request body into legacy kwargs."""
        mock_get_project.return_value = mock_project
        mock_get_queue.return_value = Mock(submit=Mock(return_value='job-y'))

        config_overrides = {
            'kick.geomean_threshold': 600,
            'hihat.open_geomean_min': 200,
            'filtering.reverb_continuation_attack_threshold': 0.3,
        }
        response = client.post(
            '/api/stems-to-midi',
            data=json.dumps({
                'project_number': 1,
                'config_overrides': config_overrides,
            }),
            content_type='application/json',
        )
        assert response.status_code == 202

        submit_kwargs = mock_get_queue.return_value.submit.call_args.kwargs
        # The config_overrides dict must reach the work function
        assert submit_kwargs.get('config_overrides') == config_overrides, (
            f"config_overrides dropped! got: {submit_kwargs}"
        )

    @patch('webui.api.operations.get_project_by_number')
    @patch('webui.api.operations.run_stems_to_midi')
    @patch('webui.api.operations.get_job_queue')
    def test_stems_to_midi_forwards_orchestration_kwargs(
        self, mock_get_queue, mock_run, mock_get_project, client, mock_project
    ):
        """Modern request shape also accepts stems_to_process (list),
        max_duration (float), learning_mode (bool). These are the only
        kwargs the new stems_to_midi_for_project signature accepts
        beyond project and config."""
        mock_get_project.return_value = mock_project
        mock_get_queue.return_value = Mock(submit=Mock(return_value='job-z'))

        response = client.post(
            '/api/stems-to-midi',
            data=json.dumps({
                'project_number': 1,
                'stems_to_process': ['kick', 'snare'],
                'max_duration': 60.0,
                'learning_mode': True,
            }),
            content_type='application/json',
        )
        assert response.status_code == 202

        submit_kwargs = mock_get_queue.return_value.submit.call_args.kwargs
        assert submit_kwargs.get('stems_to_process') == ['kick', 'snare']
        assert submit_kwargs.get('max_duration') == 60.0
        assert submit_kwargs.get('learning_mode') is True

    @patch('webui.api.operations.get_project_by_number')
    def test_stems_to_midi_project_not_found(self, mock_get_project, client):
        mock_get_project.return_value = None
        response = client.post(
            '/api/stems-to-midi',
            data=json.dumps({'project_number': 999}),
            content_type='application/json',
        )
        assert response.status_code == 404


class TestEventOverridesRoute:
    """T3 found: /api/projects/<n>/event-overrides double-prefixed URL.
    The route is registered as /projects/<n>/event-overrides inside a
    blueprint that already has url_prefix='/api/projects', so the full
    URL becomes /api/projects/projects/<n>/event-overrides — which the
    JS doesn't call. Result: click-to-toggle event bars never persist.

    These tests cover both GET and PUT for the canonical URL, so future
    blueprint/url_prefix renames can't silently break this.
    """

    @patch('webui.api.projects.get_project_by_number')
    def test_get_event_overrides(self, mock_get, client, tmp_path):
        """GET /api/projects/<n>/event-overrides returns 200 with the
        overrides dict (or empty dict if file doesn't exist)."""
        project_path = tmp_path / '1 - Test'
        (project_path / 'midi').mkdir(parents=True)
        (project_path / 'midi' / 'event_overrides.json').write_text(
            json.dumps({'snare': {'2.0782': 'FILTERED'}})
        )
        mock_get.return_value = {
            'number': 1, 'name': 'Test', 'path': project_path,
            'created': datetime.now(), 'metadata': {},
        }

        response = client.get('/api/projects/1/event-overrides')
        assert response.status_code == 200, (
            f"GET /api/projects/1/event-overrides returned {response.status_code}: "
            f"{response.data!r}. Likely cause: route is registered with "
            f"double 'projects/' prefix."
        )
        data = json.loads(response.data)
        assert data['overrides'] == {'snare': {'2.0782': 'FILTERED'}}

    @patch('webui.api.projects.get_project_by_number')
    def test_get_event_overrides_no_file_returns_empty(self, mock_get, client, tmp_path):
        """GET when event_overrides.json doesn't exist returns 200 with {}."""
        project_path = tmp_path / '1 - Test'
        (project_path / 'midi').mkdir(parents=True)
        mock_get.return_value = {
            'number': 1, 'name': 'Test', 'path': project_path,
            'created': datetime.now(), 'metadata': {},
        }

        response = client.get('/api/projects/1/event-overrides')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['overrides'] == {}

    @patch('webui.api.projects.get_project_by_number')
    def test_put_event_overrides_persists(self, mock_get, client, tmp_path):
        """PUT /api/projects/<n>/event-overrides writes the file and
        returns 200. Without this, click-to-toggle never persists across
        page reloads (T3 finding)."""
        project_path = tmp_path / '1 - Test'
        (project_path / 'midi').mkdir(parents=True)
        mock_get.return_value = {
            'number': 1, 'name': 'Test', 'path': project_path,
            'created': datetime.now(), 'metadata': {},
        }

        overrides = {'snare': {'2.0782': 'FILTERED'}, 'kick': {'1.0': 'KEPT'}}
        response = client.put(
            '/api/projects/1/event-overrides',
            data=json.dumps({'overrides': overrides}),
            content_type='application/json',
        )
        assert response.status_code == 200, (
            f"PUT /api/projects/1/event-overrides returned {response.status_code}: "
            f"{response.data!r}. Likely cause: route is registered with "
            f"double 'projects/' prefix."
        )
        # File actually written
        written = json.loads(
            (project_path / 'midi' / 'event_overrides.json').read_text()
        )
        assert written == overrides


class TestConfigUpdateMissingFile:
    """T3 found: /api/config/<id>/midiconfig PUT 500s when project has no
    per-project midiconfig.yaml. This breaks the Save & Reconvert flow
    for projects that only have the root config. The fix is a clean 4xx
    (e.g. 404 'no per-project config to update' or 200 'created from
    root'). Anything 5xx is a regression.
    """

    @patch('webui.api.config.get_config_engine')
    def test_config_update_missing_file_returns_clean_error(
        self, mock_engine, client, tmp_path
    ):
        """PUT to /api/config/<id>/midiconfig on a project with NO
        per-project midiconfig.yaml returns a 4xx with a useful message,
        not a 500."""
        # Engine throws FileNotFoundError when the per-project config
        # doesn't exist. The route currently lets it bubble to a 500.
        mock_engine.side_effect = FileNotFoundError(
            'No per-project midiconfig.yaml'
        )

        response = client.post(
            '/api/config/1/midiconfig',
            data=json.dumps({
                'updates': [
                    {'path': ['kick', 'geomean_threshold'], 'value': 700.0},
                ],
            }),
            content_type='application/json',
        )
        # Must be 4xx, NOT 5xx
        assert 400 <= response.status_code < 500, (
            f"PUT /api/config/1/midiconfig with no per-project config "
            f"returned {response.status_code} (5xx would be a regression). "
            f"Body: {response.data!r}"
        )


class TestJobsAPI:
    """Test job status endpoints"""
    
    @patch('webui.api.job_status.get_job_queue')
    def test_list_jobs(self, mock_get_queue, client):
        """Test GET /api/jobs"""
        mock_queue = Mock()
        mock_job = Mock()
        mock_job.to_dict.return_value = {
            'id': 'job-1',
            'operation': 'separate',
            'status': 'completed'
        }
        mock_queue.get_all_jobs.return_value = [mock_job]
        mock_get_queue.return_value = mock_queue
        
        response = client.get('/api/jobs')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'jobs' in data
        assert len(data['jobs']) == 1
    
    @patch('webui.api.job_status.get_job_queue')
    def test_get_job_found(self, mock_get_queue, client):
        """Test GET /api/jobs/:id when job exists"""
        mock_queue = Mock()
        mock_job = Mock()
        mock_job.to_dict.return_value = {
            'id': 'job-1',
            'status': 'running',
            'progress': 50
        }
        mock_queue.get_job.return_value = mock_job
        mock_get_queue.return_value = mock_queue
        
        response = client.get('/api/jobs/job-1')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['job']['id'] == 'job-1'
    
    @patch('webui.api.job_status.get_job_queue')
    def test_get_job_not_found(self, mock_get_queue, client):
        """Test GET /api/jobs/:id when job doesn't exist"""
        mock_queue = Mock()
        mock_queue.get_job.return_value = None
        mock_get_queue.return_value = mock_queue
        
        response = client.get('/api/jobs/nonexistent')
        
        assert response.status_code == 404
    
    @patch('webui.api.job_status.get_job_queue')
    def test_cancel_job(self, mock_get_queue, client):
        """Test POST /api/jobs/:id/cancel"""
        mock_queue = Mock()
        mock_queue.cancel_job.return_value = True
        mock_get_queue.return_value = mock_queue
        
        response = client.post('/api/jobs/job-1/cancel')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'Job cancelled' in data['message']


class TestJobQueue:
    """Test job queue functionality"""
    
    def test_job_queue_creation(self):
        """Test creating job queue"""
        queue = JobQueue(max_concurrent=2)
        assert queue.max_concurrent == 2
        assert len(queue.jobs) == 0
    
    def test_submit_job(self):
        """Test submitting a job"""
        queue = JobQueue()
        
        def dummy_func():
            return "success"
        
        job_id = queue.submit('test_op', dummy_func, project_id=1)
        
        assert job_id is not None
        job = queue.get_job(job_id)
        assert job is not None
        assert job.operation == 'test_op'
        assert job.project_id == 1
        assert job.status == JobStatus.QUEUED
    
    def test_get_project_jobs(self):
        """Test getting jobs for a specific project"""
        queue = JobQueue()
        
        def dummy_func():
            return "success"
        
        # Submit jobs for different projects
        _job1 = queue.submit('op1', dummy_func, project_id=1)
        _job2 = queue.submit('op2', dummy_func, project_id=2)
        _job3 = queue.submit('op3', dummy_func, project_id=1)
        
        # Get project 1 jobs
        project1_jobs = queue.get_project_jobs(1)
        assert len(project1_jobs) == 2
        assert all(job.project_id == 1 for job in project1_jobs)
    
    def test_cancel_queued_job(self):
        """Test cancelling a queued job"""
        queue = JobQueue()
        
        def dummy_func():
            return "success"
        
        job_id = queue.submit('test_op', dummy_func)
        success = queue.cancel_job(job_id)
        
        assert success is True
        job = queue.get_job(job_id)
        assert job.status == JobStatus.CANCELLED


class TestAudioFilesAPI:
    """Test audio file management endpoints"""
    
    @patch('webui.api.projects.get_project_by_number')
    def test_list_audio_files(self, mock_get_project, client, mock_project, tmp_path):
        """Test GET /api/projects/:id/audio-files"""
        # Setup mock project with files
        project_path = tmp_path / "1 - Test Song"
        project_path.mkdir()
        
        # Create original audio file
        original_audio = project_path / "Test Song.wav"
        original_audio.write_bytes(b"fake audio data")
        
        # Create alternate_mix directory with files
        alternate_mix = project_path / "alternate_mix"
        alternate_mix.mkdir()
        (alternate_mix / "no_drums.wav").write_bytes(b"fake alternate")
        
        mock_project['path'] = project_path
        mock_get_project.return_value = mock_project
        
        response = client.get('/api/projects/1/audio-files')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'audio_files' in data
        assert len(data['audio_files']) == 2
        
        # Check original audio
        original = next(f for f in data['audio_files'] if f['type'] == 'original')
        assert original['name'] == 'Test Song.wav'
        assert original['path'] == 'original'
        
        # Check alternate audio
        alternate = next(f for f in data['audio_files'] if f['type'] == 'alternate')
        assert alternate['name'] == 'no_drums.wav'
        assert alternate['path'] == 'alternate_mix/no_drums.wav'
    
    @patch('webui.api.projects.get_project_by_number')
    def test_list_audio_files_no_project(self, mock_get_project, client):
        """Test GET /api/projects/:id/audio-files when project not found"""
        mock_get_project.return_value = None
        
        response = client.get('/api/projects/999/audio-files')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    @patch('webui.api.projects.get_project_by_number')
    def test_upload_alternate_audio(self, mock_get_project, client, mock_project, tmp_path):
        """Test POST /api/projects/:id/upload-alternate-audio"""
        # Setup mock project
        project_path = tmp_path / "1 - Test Song"
        project_path.mkdir()
        mock_project['path'] = project_path
        mock_get_project.return_value = mock_project
        
        # Create test file data using BytesIO for proper file upload simulation
        from io import BytesIO
        data = {
            'file': (BytesIO(b'fake wav data'), 'test_audio.wav')
        }
        
        response = client.post(
            '/api/projects/1/upload-alternate-audio',
            data=data,
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 201
        result = json.loads(response.data)
        assert result['filename'] == 'test_audio.wav'
        assert result['path'] == 'alternate_mix/test_audio.wav'
        
        # Verify file was created
        alternate_mix = project_path / "alternate_mix"
        assert alternate_mix.exists()
        uploaded_file = alternate_mix / "test_audio.wav"
        assert uploaded_file.exists()
    
    @patch('webui.api.projects.get_project_by_number')
    def test_upload_alternate_audio_invalid_format(self, mock_get_project, client, mock_project, tmp_path):
        """Test upload with non-WAV file"""
        project_path = tmp_path / "1 - Test Song"
        project_path.mkdir()
        mock_project['path'] = project_path
        mock_get_project.return_value = mock_project
        
        from io import BytesIO
        data = {
            'file': (BytesIO(b'fake make up audio data'), 'test_audio.otheraudioformat')
        }
        
        response = client.post(
            '/api/projects/1/upload-alternate-audio',
            data=data,
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
        assert 'Audio file must be one of' in result['message']
    
    @patch('webui.api.projects.get_project_by_number')
    def test_upload_alternate_audio_no_project(self, mock_get_project, client):
        """Test upload when project doesn't exist"""
        mock_get_project.return_value = None
        
        from io import BytesIO
        data = {
            'file': (BytesIO(b'fake wav data'), 'test_audio.wav')
        }
        
        response = client.post(
            '/api/projects/999/upload-alternate-audio',
            data=data,
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 404
    
    @patch('webui.api.projects.get_project_by_number')
    def test_delete_audio_file(self, mock_get_project, client, mock_project, tmp_path):
        """Test DELETE /api/projects/:id/audio-files/:filename"""
        # Setup mock project with alternate audio
        project_path = tmp_path / "1 - Test Song"
        project_path.mkdir()
        alternate_mix = project_path / "alternate_mix"
        alternate_mix.mkdir()
        
        test_file = alternate_mix / "to_delete.wav"
        test_file.write_bytes(b"fake audio")
        
        mock_project['path'] = project_path
        mock_get_project.return_value = mock_project
        
        response = client.delete('/api/projects/1/audio-files/to_delete.wav')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['filename'] == 'to_delete.wav'
        
        # Verify file was deleted
        assert not test_file.exists()
    
    @patch('webui.api.projects.get_project_by_number')
    def test_delete_audio_file_path_traversal(self, mock_get_project, client, mock_project, tmp_path):
        """Test delete with path traversal attempt"""
        project_path = tmp_path / "1 - Test Song"
        project_path.mkdir()
        mock_project['path'] = project_path
        mock_get_project.return_value = mock_project
        
        # Try to delete file outside alternate_mix using path traversal
        response = client.delete('/api/projects/1/audio-files/../../../etc/passwd')
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
        assert 'Invalid' in result['error'] or 'invalid' in result['message'].lower()
    
    @patch('webui.api.projects.get_project_by_number')
    def test_delete_audio_file_not_found(self, mock_get_project, client, mock_project, tmp_path):
        """Test delete when file doesn't exist"""
        project_path = tmp_path / "1 - Test Song"
        project_path.mkdir()
        alternate_mix = project_path / "alternate_mix"
        alternate_mix.mkdir()
        
        mock_project['path'] = project_path
        mock_get_project.return_value = mock_project
        
        response = client.delete('/api/projects/1/audio-files/nonexistent.wav')
        
        assert response.status_code == 404


class TestHealthCheck:
    """Test health check endpoint"""

    def test_health(self, client):
        """Test GET /health"""
        response = client.get('/health')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'version' in data


# ============================================================================
# Route↔JS URL smoke test (T3 follow-up, 2026-06-08)
# ============================================================================


class TestRouteRegistration:
    """T3 found /api/projects/<n>/event-overrides double-prefixed — the
    route was registered as /projects/<n>/event-overrides inside a
    blueprint whose url_prefix was already /api/projects. Result: 404.

    This class catalogs the URL set the JS client uses (extracted from
    webui/static/js/api.js) and asserts each one resolves to a real
    Flask route. If a future rename or blueprint re-registration breaks
    a URL, this test fails BEFORE the user clicks the broken button.

    The list is intentionally hard-coded (not auto-generated) so it
    documents the contract — a developer changing the JS or the routes
    must update both sides."""

    def test_event_overrides_url_not_double_prefixed(self, client, tmp_path):
        """The JS calls /api/projects/<n>/event-overrides. The Flask
        blueprint projects_bp is registered with url_prefix='/api/projects'.
        The route must therefore be '/<n>/event-overrides' (not
        '/projects/<n>/event-overrides'). This test asserts the URL the
        JS uses resolves to a real route (404 → 200/404 with body, but
        NOT the URL-mismatch 404)."""
        project_path = tmp_path / '1 - Test'
        (project_path / 'midi').mkdir(parents=True)
        with patch('webui.api.projects.get_project_by_number') as mock_get:
            mock_get.return_value = {
                'number': 1, 'name': 'Test', 'path': project_path,
                'created': datetime.now(), 'metadata': {},
            }
            # Hit the URL the JS actually calls
            response = client.get('/api/projects/1/event-overrides')
            # If the URL is double-prefixed, Flask returns 404 with empty
            # body (no route matched). A working URL returns 200 with JSON.
            assert response.status_code == 200, (
                f"GET /api/projects/1/event-overrides returned {response.status_code}. "
                f"This is the JS URL — the Flask route is probably registered "
                f"as /projects/<n>/event-overrides inside a blueprint with "
                f"url_prefix='/api/projects', giving the wrong full URL. "
                f"Body: {response.data!r}"
            )

    def test_every_js_endpoint_resolves_to_a_flask_route(self, app, client):
        """Catalog of URL paths the JS uses. Each must resolve to a
        real Flask route (status != 404 for the canonical path).

        The point of this test: if anyone renames a route or changes a
        blueprint's url_prefix, the JS will silently 404. This catches
        the breakage here, before the user clicks."""
        # Build the JS endpoint set
        js_endpoints = [
            # (path, method, optional body)
            ('/api/projects', 'GET'),
            ('/api/projects/1', 'GET'),
            ('/api/projects/1/config/midiconfig', 'GET'),
            ('/api/projects/1/jobs', 'GET'),
            ('/api/projects/1/analysis', 'GET'),
            ('/api/projects/1/envelope/kick', 'GET'),
            ('/api/projects/1/event-overrides', 'GET'),
            ('/api/projects/1/event-overrides', 'PUT'),
            ('/api/projects/1/audio-files', 'GET'),
            ('/api/separate', 'POST'),
            ('/api/cleanup', 'POST'),
            ('/api/stems-to-midi', 'POST'),
            ('/api/rebuild-midi', 'POST'),
            ('/api/reclassify', 'POST'),
            ('/api/render-video', 'POST'),
            ('/api/jobs', 'GET'),
            ('/api/jobs/job-1', 'GET'),
            ('/api/jobs/job-1/cancel', 'POST'),
        ]

        # Get the full set of registered Flask routes for the app, with
        # a normalized method list.
        registered = []
        for rule in app.url_map.iter_rules():
            methods = sorted(m for m in rule.methods if m not in ('HEAD', 'OPTIONS'))
            registered.append((rule.rule, methods))

        failures = []
        for path, method in js_endpoints:
            ok = False
            for rule, methods in registered:
                if method in methods and _routes_match(rule, path):
                    ok = True
                    break
            if not ok:
                failures.append(f"  {method:6s} {path}  → NO MATCHING ROUTE")

        assert not failures, (
            "The following JS endpoints don't resolve to any registered Flask "
            "route. The route was likely renamed, the blueprint url_prefix "
            "was changed, or the URL is missing a path segment. Failures:\n"
            + "\n".join(failures)
            + "\n\nRegistered routes (sample):\n"
            + "\n".join(f"  {','.join(methods)!s:20s} {rule}"
                        for rule, methods in sorted(registered)[:20])
        )


def _routes_match(rule_str: str, js_path: str) -> bool:
    """
    Best-effort: does a Flask rule string (e.g. '/api/projects/<int:project_number>')
    match a JS-callable path (e.g. '/api/projects/1')?

    We do this by converting the Flask rule to a regex with integer-or-string
    placeholders, then matching the JS path. Catches the most common cases
    (int path params, string path params). Doesn't catch complex converters
    (uuid, path, etc.) — those are rare in this codebase.
    """
    import re
    # Convert Flask placeholders to a permissive regex
    pattern = re.sub(r'<[^>]+>', r'[^/]+', rule_str)
    return bool(re.fullmatch(pattern, js_path))


# ============================================================================
# Stems-to-Midi importlib loader contract (T2 follow-up #2, 2026-06-08)
# ============================================================================


class TestStemsToMidiImportlibContract:
    """T2 follow-up (round 2, 2026-06-08): the /api/stems-to-midi work
    function uses importlib.util.spec_from_file_location to load
    stems_to_midi_cli.py into a fresh module namespace, then calls
    helpers on that loaded module. If the helpers live in
    webui.api.operations instead of in the loaded file, the route
    crashes with `module 'stems_to_midi_cli' has no attribute
    '_load_project_config_for_project'` — which is exactly the 500
    toast the user saw in the WebUI.

    These tests mirror the production importlib load and assert the
    loaded module exposes everything the work function needs.
    """

    def test_loaded_module_exposes_config_loader(self):
        """The work function calls `stems_to_midi_cli._load_project_config_for_project(project)`.
        That helper must be defined in stems_to_midi_cli.py so the
        importlib-loaded module can see it."""
        import importlib.util
        from pathlib import Path

        cli_path = Path(__file__).parent.parent / 'stems_to_midi_cli.py'
        spec = importlib.util.spec_from_file_location('stems_to_midi_cli', cli_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, '_load_project_config_for_project'), (
            "stems_to_midi_cli.py is loaded via importlib in run_stems_to_midi "
            "(webui/api/operations.py:87). The work function then calls "
            "stems_to_midi_cli._load_project_config_for_project(project) — but "
            "the helper is defined in webui.api.operations, not in the loaded "
            "file, so the importlib-loaded module doesn't have it. Move the "
            "helper into stems_to_midi_cli.py or call it via a normal import."
        )

    def test_loaded_module_exposes_override_applier(self):
        """The work function calls `stems_to_midi_cli._apply_cli_overrides_to_config(config, overrides)`.
        That helper must be defined in stems_to_midi_cli.py so the
        importlib-loaded module can see it."""
        import importlib.util
        from pathlib import Path

        cli_path = Path(__file__).parent.parent / 'stems_to_midi_cli.py'
        spec = importlib.util.spec_from_file_location('stems_to_midi_cli', cli_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, '_apply_cli_overrides_to_config'), (
            "stems_to_midi_cli.py is loaded via importlib in run_stems_to_midi "
            "(webui/api/operations.py:87). The work function then calls "
            "stems_to_midi_cli._apply_cli_overrides_to_config(config, overrides) "
            "— but the helper is defined in webui.api.operations, not in the "
            "loaded file, so the importlib-loaded module doesn't have it. Move "
            "the helper into stems_to_midi_cli.py or call it via a normal import."
        )

    def test_run_stems_to_midi_loads_config_via_loaded_module(self, tmp_path, monkeypatch):
        """End-to-end via the loaded module: the helpers
        _load_project_config_for_project and
        _apply_cli_overrides_to_config must work when invoked
        through the importlib-loaded stems_to_midi_cli module —
        because that's the only access path run_stems_to_midi uses.

        We invoke the helpers on the importlib-loaded module
        directly (no run_stems_to_midi) so we don't have to fake
        the entire audio pipeline. This still proves the fix:
        the helpers must be reachable from the loaded module, with
        the right behavior, and the override applier must merge
        dotted paths into nested dicts.
        """
        import importlib.util
        from pathlib import Path

        cli_path = Path(__file__).parent.parent / 'stems_to_midi_cli.py'
        spec = importlib.util.spec_from_file_location('stems_to_midi_cli', cli_path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)

        project_dir = tmp_path / 'fake_project'
        project_dir.mkdir(exist_ok=True)
        # No midiconfig.yaml in project — should return empty config
        # (we monkeypatch get_project_config below to return None)
        fake_project = {
            'number': 99,
            'name': 'Test',
            'path': project_dir,
            'created': datetime.now(),
            'metadata': {},
        }

        # Force the loaded module's get_project_config to return None
        # so the helper takes the empty-config branch. (Without this
        # it would fall back to the larsnet root's midiconfig.yaml.)
        monkeypatch.setattr(loaded, 'get_project_config', lambda *a, **kw: None)

        # Call the helper through the loaded module — the exact
        # access pattern run_stems_to_midi uses.
        config = loaded._load_project_config_for_project(fake_project)
        assert config == {}, (
            f"Expected empty config when no midiconfig.yaml, got: {config!r}. "
            f"Check that _load_project_config_for_project is defined in "
            f"stems_to_midi_cli.py and returns {{}} when get_project_config "
            f"returns None."
        )

        # Now exercise the override applier — same access pattern.
        loaded._apply_cli_overrides_to_config(
            config,
            {'kick.onset_threshold': 0.42, 'snare.onset_delta': 0.01},
        )
        assert config == {
            'kick': {'onset_threshold': 0.42},
            'snare': {'onset_delta': 0.01},
        }, f"Override applier did not merge dotted paths: {config!r}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

