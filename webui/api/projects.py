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
