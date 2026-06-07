"""
Projects API Endpoints

Provides REST API for project discovery, retrieval, and status.
Uses project_manager functions as the functional core.
"""

from flask import jsonify, request # type: ignore
from pathlib import Path
import json
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
            'video': [],
            'comparison': {}
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
        
        # Check for analysis data
        has_analysis = False
        envelope_stems = []
        if midi_dir.exists():
            analysis_files = list(midi_dir.glob('*.analysis.json'))
            if analysis_files:
                has_analysis = True
            # Check for envelope .npz files
            for npz_file in midi_dir.glob('*.envelope.npz'):
                # Extract stem type from filename: base.STEM.envelope.npz
                parts = npz_file.suffixes  # ['.STEM', '.envelope', '.npz']
                if len(parts) >= 3:
                    stem_type = parts[-3].lstrip('.')
                    envelope_stems.append(stem_type)
        
        project_data = {
            **project,
            'path': str(project['path']),
            'files': files,
            'has_analysis': has_analysis,
            'envelope_stems': sorted(envelope_stems),
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
        config_name: Config file name (midiconfig.yaml)
        
    Returns:
        200: Config file contents as YAML
        404: Project or config not found
        400: Invalid config name
        500: Internal error
    """
    try:
        # Validate config name
        allowed_configs = ['midiconfig.yaml']
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


@projects_bp.route('/<int:project_number>/analysis', methods=['GET'])
def get_project_analysis(project_number):
    """
    GET /api/projects/:project_number/analysis

    Get analysis sidecar data (onset events, filtering decisions, spectral features).
    Supports both v2 (events) and v3 (events_configured/events_sensitive) formats.

    Bug C: also runs the events_configured ⊆ events_sensitive subset check
    on load and returns a ``data_integrity_warnings`` array when violations
    are detected, so the WebUI can surface them as toasts.

    Returns:
        200: Analysis JSON data
        404: Project or analysis file not found
    """
    try:
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404

        midi_dir = project['path'] / 'midi'
        if not midi_dir.exists():
            return jsonify({
                'error': 'No analysis data',
                'message': 'No MIDI directory found — run MIDI conversion first'
            }), 404

        analysis_files = list(midi_dir.glob('*.analysis.json'))
        if not analysis_files:
            return jsonify({
                'error': 'No analysis data',
                'message': 'No analysis.json found — run MIDI conversion first'
            }), 404

        with open(analysis_files[0], 'r') as f:
            analysis_data = json.load(f)

        # Run the bug C subset validation. The loader is the canonical
        # place for this check (called from both the API and the rebuild
        # shell), but a second pass here ensures GETs that bypass the
        # loader (e.g. direct file read) also surface warnings.
        from stems_to_midi.midi import _validate_events_subset
        integrity_warnings = _validate_events_subset(analysis_data)
        if integrity_warnings:
            analysis_data = dict(analysis_data)
            existing = analysis_data.get('data_integrity_warnings', [])
            analysis_data['data_integrity_warnings'] = list(existing) + integrity_warnings

        return jsonify(analysis_data), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get analysis',
            'message': str(e)
        }), 500


@projects_bp.route('/<int:project_number>/envelope/<stem_type>', methods=['GET'])
def get_project_envelope(project_number, stem_type):
    """
    GET /api/projects/:project_number/envelope/:stem_type

    Get energy envelope data for a specific stem, converted from .npz to JSON.
    Returns downsampled arrays suitable for Canvas rendering.
    
    Response is gzip-compressed for faster transfer.

    Args:
        stem_type: One of kick, snare, hihat, cymbals, toms

    Returns:
        200: {times: [...], left: [...], right: [...], sr, hop_length, method}
        404: Project, stem, or envelope file not found
    """
    import gzip
    import numpy as np
    from flask import make_response
    import time as time_module

    try:
        # Performance checkpoint 1: Validation
        t0 = time_module.perf_counter()
        
        valid_stems = ['kick', 'snare', 'hihat', 'cymbals', 'toms']
        if stem_type not in valid_stems:
            return jsonify({
                'error': 'Invalid stem type',
                'message': f'Stem must be one of: {", ".join(valid_stems)}'
            }), 400

        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404

        midi_dir = project['path'] / 'midi'
        if not midi_dir.exists():
            return jsonify({
                'error': 'No envelope data',
                'message': 'No MIDI directory found'
            }), 404

        # Find the .npz file for this stem
        pattern = f'*.{stem_type}.envelope.npz'
        npz_files = list(midi_dir.glob(pattern))
        if not npz_files:
            return jsonify({
                'error': 'No envelope data',
                'message': f'No envelope data for {stem_type} — re-run MIDI conversion to generate'
            }), 404
        data = np.load(npz_files[0], allow_pickle=False)

        # Downsample if needed (target ~48000 points for high-res Canvas rendering)
        # Load your data as usual
        times_raw = data['times']
        left_raw = data['left']
        right_raw = data['right']

        max_points = 32000
        n = len(times_raw)

        if n > max_points:
            # 1. Calculate window size
            window_size = n // max_points
            # 2. Trim to a perfect multiple of window_size
            limit = window_size * max_points
            
            # 3. Perform Vectorized Aggregation (Max)
            # We reshape to (32000, window_size) then find the max of each row
            left = left_raw[:limit].reshape(max_points, window_size).max(axis=1).astype(float)
            right = right_raw[:limit].reshape(max_points, window_size).max(axis=1).astype(float)
            
            # 4. Downsample Time
            # Taking the first timestamp of each window to keep them aligned
            times = times_raw[:limit:window_size].astype(float)
        else:
            # If data is already small, just ensure types are correct for the web viewer
            times = times_raw.astype(float)
            left = left_raw.astype(float)
            right = right_raw.astype(float)

        response_data = {
            'stem_type': stem_type,
            'times': [round(float(t), 4) for t in times],
            'left': [round(float(v), 6) for v in left],
            'right': [round(float(v), 6) for v in right],
            'sr': int(data['sr']),
            'hop_length': int(data['hop_length']),
            'method': str(data['method']),
            'sample_count': len(times),
        }
        
        # Only gzip if client explicitly accepts it (browser auto-decompresses)
        accept_encoding = request.headers.get('Accept-Encoding', '')
        
        if 'gzip' in accept_encoding.lower():
            json_str = json.dumps(response_data)
            json_bytes = json_str.encode('utf-8')
            compressed = gzip.compress(json_bytes, compresslevel=6)
            
            response = make_response(compressed)
            response.headers['Content-Type'] = 'application/json'
            response.headers['Content-Length'] = len(compressed)
            response.headers['Content-Encoding'] = 'gzip'
        else:
            # Return uncompressed JSON for test clients
            response = make_response(response_data)
            response.headers['Content-Type'] = 'application/json'
            
        return response, 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get envelope',
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
    
    Upload an alternate audio file to a project's alternate_mix directory.
    
    Request:
        - multipart/form-data with 'file' field
        - file must be a supported audio format (wav, mp3, flac, aiff, aac, ogg, m4a)
        
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
        
        # Check file extension - allow common audio formats
        allowed_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.aif', '.aac', '.ogg', '.m4a'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            allowed_formats = ', '.join(sorted(allowed_extensions))
            return jsonify({
                'error': 'Invalid file type',
                'message': f'Audio file must be one of: {allowed_formats}'
            }), 400
        
        # Secure the filename
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        
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
            # Support all common audio formats
            alternate_extensions = ['*.wav', '*.mp3', '*.flac', '*.aiff', '*.aif', '*.aac', '*.ogg', '*.m4a']
            alternate_files = []
            for pattern in alternate_extensions:
                alternate_files.extend(alternate_mix_dir.glob(pattern))
            
            for audio_file in sorted(alternate_files):
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


# ─── Event Overrides ─────────────────────────────────────────────────────

@projects_bp.route('/projects/<int:project_number>/event-overrides', methods=['GET'])
def get_event_overrides(project_number):
    """
    GET /api/projects/:project_number/event-overrides

    Retrieve manual event overrides for a project.

    Returns:
        200: { overrides: { stemType: { "time_str": "KEPT"|"FILTERED", ... }, ... } }
        404: Project not found
    """
    project = get_project_by_number(project_number)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    overrides_path = Path(project['path']) / 'midi' / 'event_overrides.json'
    if overrides_path.exists():
        try:
            overrides = json.loads(overrides_path.read_text())
        except (json.JSONDecodeError, OSError):
            overrides = {}
    else:
        overrides = {}

    return jsonify({'overrides': overrides}), 200


@projects_bp.route('/projects/<int:project_number>/event-overrides', methods=['PUT'])
def save_event_overrides(project_number):
    """
    PUT /api/projects/:project_number/event-overrides

    Save manual event overrides for a project.

    Body: { overrides: { stemType: { "time_str": "KEPT"|"FILTERED", ... }, ... } }

    Returns:
        200: { saved: true }
        404: Project not found
    """
    project = get_project_by_number(project_number)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json()
    if not data or 'overrides' not in data:
        return jsonify({'error': 'Missing overrides in request body'}), 400

    midi_dir = Path(project['path']) / 'midi'
    midi_dir.mkdir(parents=True, exist_ok=True)
    overrides_path = midi_dir / 'event_overrides.json'

    try:
        overrides_path.write_text(json.dumps(data['overrides'], indent=2))
        return jsonify({'saved': True}), 200
    except OSError as e:
        return jsonify({'error': 'Failed to save overrides', 'message': str(e)}), 500
