"""
Unit Tests for LarsNet Web UI API

Tests all API endpoints with mocked project_manager functions.
Run with: pytest webui/test_api.py
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
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
        """Test POST /api/stems-to-midi"""
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
        job1 = queue.submit('op1', dummy_func, project_id=1)
        job2 = queue.submit('op2', dummy_func, project_id=2)
        job3 = queue.submit('op3', dummy_func, project_id=1)
        
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
            'file': (BytesIO(b'fake mp3 data'), 'test_audio.mp3')
        }
        
        response = client.post(
            '/api/projects/1/upload-alternate-audio',
            data=data,
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
        assert 'WAV format' in result['message']
    
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
