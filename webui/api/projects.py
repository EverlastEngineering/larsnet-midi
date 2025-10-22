"""
Projects API Endpoints

Provides REST API for project discovery, retrieval, and status.
Uses project_manager functions as the functional core.
"""

from flask import jsonify, request # type: ignore
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_manager import (
    discover_projects,
    get_project_by_number,
    USER_FILES_DIR
)
from webui.api import projects_bp


@projects_bp.route('', methods=['GET'])
def list_projects():
    """
    GET /api/projects
    
    List all projects in user_files directory.
    
    Returns:
        200: List of projects with metadata
        500: Internal error
        
    Response format:
        {
            "projects": [
                {
                    "number": 1,
                    "name": "Song Name",
                    "path": "/path/to/project",
                    "created": "2025-10-19T12:00:00",
                    "metadata": {...},
                    "has_stems": true,
                    "has_cleaned": false,
                    "has_midi": true,
                    "has_video": false
                },
                ...
            ]
        }
    """
    try:
        projects = discover_projects(USER_FILES_DIR)
        
        # Enhance with status information
        enhanced_projects = []
        for project in projects:
            project_data = {
                **project,
                'path': str(project['path']),  # Convert Path to string for JSON
                'has_stems': (project['path'] / 'stems').exists() and 
                            any((project['path'] / 'stems').iterdir()),
                'has_cleaned': (project['path'] / 'cleaned').exists() and 
                              any((project['path'] / 'cleaned').iterdir()),
                'has_midi': (project['path'] / 'midi').exists() and 
                           any((project['path'] / 'midi').glob('*.mid')),
                'has_video': (project['path'] / 'video').exists() and 
                            any((project['path'] / 'video').glob('*.mp4'))
            }
            enhanced_projects.append(project_data)
        
        return jsonify({
            'projects': enhanced_projects
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to list projects',
            'message': str(e)
        }), 500


@projects_bp.route('/<int:project_number>', methods=['GET'])
def get_project(project_number):
    """
    GET /api/projects/:project_number
    
    Get detailed information about a specific project.
    
    Args:
        project_number: Project number
        
    Returns:
        200: Project details
        404: Project not found
        500: Internal error
        
    Response format:
        {
            "project": {
                "number": 1,
                "name": "Song Name",
                "path": "/path/to/project",
                "created": "2025-10-19T12:00:00",
                "metadata": {...},
                "files": {
                    "audio": ["song.wav"],
                    "stems": ["kick.wav", "snare.wav", ...],
                    "cleaned": [...],
                    "midi": ["song.mid"],
                    "video": ["song.mp4"]
                }
            }
        }
    """
    try:
        project = get_project_by_number(project_number, USER_FILES_DIR)
        
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        project_path = project['path']
        
        # Gather file information
        files = {
            'audio': [],
            'stems': [],
            'cleaned': [],
            'midi': [],
            'video': []
        }
        
        # Find audio files in root
        for ext in ['wav', 'mp3', 'flac', 'aiff', 'aif']:
            files['audio'].extend([f.name for f in project_path.glob(f'*.{ext}')])
        
        # Find stems
        stems_dir = project_path / 'stems'
        if stems_dir.exists():
            files['stems'] = [f.name for f in stems_dir.glob('*.wav')]
        
        # Find cleaned stems
        cleaned_dir = project_path / 'cleaned'
        if cleaned_dir.exists():
            files['cleaned'] = [f.name for f in cleaned_dir.glob('*.wav')]
        
        # Find MIDI files
        midi_dir = project_path / 'midi'
        if midi_dir.exists():
            files['midi'] = [f.name for f in midi_dir.glob('*.mid')]
        
        # Find videos
        video_dir = project_path / 'video'
        if video_dir.exists():
            files['video'] = [f.name for f in video_dir.glob('*.mp4')]
        
        project_data = {
            **project,
            'path': str(project['path']),
            'files': files
        }
        
        return jsonify({
            'project': project_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get project',
            'message': str(e)
        }), 500


@projects_bp.route('/<int:project_number>/config/<config_name>', methods=['GET'])
def get_project_config(project_number, config_name):
    """
    GET /api/projects/:project_number/config/:config_name
    
    Get project configuration file contents.
    
    Args:
        project_number: Project number
        config_name: Config file name (config.yaml, midiconfig.yaml, eq.yaml)
        
    Returns:
        200: Config file contents as YAML
        404: Project or config not found
        400: Invalid config name
        500: Internal error
    """
    try:
        # Validate config name
        allowed_configs = ['config.yaml', 'midiconfig.yaml', 'eq.yaml']
        if config_name not in allowed_configs:
            return jsonify({
                'error': 'Invalid config name',
                'message': f'Config must be one of: {", ".join(allowed_configs)}'
            }), 400
        
        project = get_project_by_number(project_number, USER_FILES_DIR)
        
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        config_path = project['path'] / config_name
        
        if not config_path.exists():
            return jsonify({
                'error': 'Config not found',
                'message': f'{config_name} not found in project'
            }), 404
        
        # Read and return config as text
        config_content = config_path.read_text()
        
        return jsonify({
            'config_name': config_name,
            'content': config_content
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get config',
            'message': str(e)
        }), 500


@projects_bp.route('/<int:project_number>', methods=['DELETE'])
def delete_project(project_number):
    """
    DELETE /api/projects/<project_number>
    
    Permanently delete a project and all its files.
    
    Path Parameters:
        project_number (int): Project number (e.g., 1, 2, 3)
    
    Returns:
        200: Project deleted successfully
        404: Project not found
        500: Internal error
        
    Response format:
        {
            "message": "Project deleted successfully",
            "project_number": 1,
            "project_name": "Song Name"
        }
    """
    try:
        import shutil
        
        # Get project details before deletion
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if not project:
            return jsonify({
                'error': 'Not found',
                'message': f'Project {project_number} not found'
            }), 404
        
        project_name = project['name']
        project_path = project['path']
        
        # Delete the entire project directory
        if project_path.exists():
            shutil.rmtree(project_path)
        
        return jsonify({
            'message': 'Project deleted successfully',
            'project_number': project_number,
            'project_name': project_name
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to delete project',
            'message': str(e)
        }), 500


@projects_bp.route('/<int:project_number>/upload-alternate-audio', methods=['POST'])
def upload_alternate_audio(project_number):
    """
    POST /api/projects/{number}/upload-alternate-audio
    
    Upload an alternate audio file (WAV) to a project's alternate_mix directory.
    
    Request:
        - multipart/form-data with 'file' field
        - file must be .wav format
        
    Returns:
        201: File uploaded successfully
        400: Bad request (no file, invalid format, etc.)
        404: Project not found
        500: Internal error
        
    Response format:
        {
            "message": "Alternate audio uploaded successfully",
            "filename": "my_audio.wav",
            "size": 12345678,
            "path": "alternate_mix/my_audio.wav"
        }
    """
    try:
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Request must include a file in the "file" field'
            }), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400
        
        # Check file extension - only allow WAV files for alternate audio
        if not file.filename.lower().endswith('.wav'):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Alternate audio must be WAV format (.wav)'
            }), 400
        
        # Secure the filename
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        
        # Ensure filename ends with .wav
        if not filename.lower().endswith('.wav'):
            filename += '.wav'
        
        # Create alternate_mix directory if it doesn't exist
        project_path = project['path']
        alternate_mix_dir = project_path / 'alternate_mix'
        alternate_mix_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = alternate_mix_dir / filename
        
        # Check if file already exists
        if file_path.exists():
            return jsonify({
                'error': 'File already exists',
                'message': f'A file named "{filename}" already exists in alternate_mix'
            }), 400
        
        file.save(str(file_path))
        
        # Get file size
        file_size = file_path.stat().st_size
        
        return jsonify({
            'message': 'Alternate audio uploaded successfully',
            'filename': filename,
            'size': file_size,
            'path': f'alternate_mix/{filename}'
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to upload alternate audio',
            'message': str(e)
        }), 500


@projects_bp.route('/<int:project_number>/audio-files', methods=['GET'])
def list_audio_files(project_number):
    """
    GET /api/projects/{number}/audio-files
    
    List all available audio files for a project (original + alternate mixes).
    
    Returns:
        200: List of audio files
        404: Project not found
        500: Internal error
        
    Response format:
        {
            "audio_files": [
                {
                    "name": "Song Name.wav",
                    "path": "original",
                    "type": "original",
                    "size": 12345678,
                    "exists": true
                },
                {
                    "name": "no_drums.wav",
                    "path": "alternate_mix/no_drums.wav",
                    "type": "alternate",
                    "size": 12345678,
                    "exists": true
                },
                ...
            ]
        }
    """
    try:
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        project_path = project['path']
        audio_files = []
        
        # Check for original audio file in project root
        audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.aif']
        for ext in audio_extensions:
            original_audio = project_path / f"{project['name']}{ext}"
            if original_audio.exists():
                audio_files.append({
                    'name': original_audio.name,
                    'path': 'original',
                    'type': 'original',
                    'size': original_audio.stat().st_size,
                    'exists': True
                })
                break  # Only include the first found original
        
        # Check for alternate audio files
        alternate_mix_dir = project_path / 'alternate_mix'
        if alternate_mix_dir.exists() and alternate_mix_dir.is_dir():
            for audio_file in sorted(alternate_mix_dir.glob('*.wav')):
                audio_files.append({
                    'name': audio_file.name,
                    'path': f'alternate_mix/{audio_file.name}',
                    'type': 'alternate',
                    'size': audio_file.stat().st_size,
                    'exists': True
                })
        
        return jsonify({
            'audio_files': audio_files
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to list audio files',
            'message': str(e)
        }), 500


@projects_bp.route('/<int:project_number>/audio-files/<path:filename>', methods=['DELETE'])
def delete_audio_file(project_number, filename):
    """
    DELETE /api/projects/{number}/audio-files/{filename}
    
    Delete an alternate audio file from a project.
    Only allows deletion of files in alternate_mix/ directory for safety.
    
    Args:
        project_number: Project number
        filename: Filename to delete (must be in alternate_mix/)
        
    Returns:
        200: File deleted successfully
        400: Bad request (trying to delete original, invalid path)
        404: Project or file not found
        500: Internal error
        
    Response format:
        {
            "message": "Alternate audio deleted successfully",
            "filename": "my_audio.wav"
        }
    """
    try:
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        project_path = project['path']
        
        # Security: Only allow deletion from alternate_mix directory
        # Prevent path traversal attacks
        if '..' in filename or filename.startswith('/') or '\\' in filename:
            return jsonify({
                'error': 'Invalid filename',
                'message': 'Filename contains invalid characters'
            }), 400
        
        # Ensure filename is just a basename (no directory components)
        from pathlib import Path as PathlibPath
        if PathlibPath(filename).parts != (filename,):
            return jsonify({
                'error': 'Invalid filename',
                'message': 'Filename must not contain directory separators'
            }), 400
        
        # Construct safe path
        alternate_mix_dir = project_path / 'alternate_mix'
        file_path = alternate_mix_dir / filename
        
        # Verify file is actually in alternate_mix directory (resolve any symlinks)
        try:
            file_path = file_path.resolve()
            alternate_mix_dir = alternate_mix_dir.resolve()
            if not str(file_path).startswith(str(alternate_mix_dir)):
                return jsonify({
                    'error': 'Invalid file location',
                    'message': 'Can only delete files from alternate_mix directory'
                }), 400
        except Exception:
            return jsonify({
                'error': 'Invalid file path',
                'message': 'Unable to resolve file path'
            }), 400
        
        # Check if file exists
        if not file_path.exists():
            return jsonify({
                'error': 'File not found',
                'message': f'File "{filename}" does not exist in alternate_mix'
            }), 404
        
        # Delete the file
        file_path.unlink()
        
        return jsonify({
            'message': 'Alternate audio deleted successfully',
            'filename': filename
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to delete alternate audio',
            'message': str(e)
        }), 500
