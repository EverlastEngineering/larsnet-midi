"""
Operations API Endpoints

Handles triggering of LarsNet operations (separate, cleanup, MIDI, video).
All operations run asynchronously via the job queue.
"""

from flask import jsonify, request # type: ignore
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_manager import get_project_by_number, USER_FILES_DIR
from webui.api import operations_bp
from webui.jobs import get_job_queue


def run_separate(project_number: int, device: str = 'cpu', wiener: float = None, eq: bool = False):
    """
    Execute stem separation for a project.
    
    This is the actual work function that runs in the job queue.
    """
    from separate import separate_project
    from project_manager import get_project_by_number, USER_FILES_DIR
    
    project = get_project_by_number(project_number, USER_FILES_DIR)
    if project is None:
        raise ValueError(f'Project {project_number} not found')
    
    separate_project(project, wiener, device, eq)
    
    return {'project_number': project_number, 'stems_created': True}


def run_comparison(project_number: int, device: str = 'cpu'):
    """
    Execute separation comparison with multiple configurations.
    
    This is the actual work function that runs in the job queue.
    Tests various Wiener/EQ combinations and saves to for_comparison/ directory.
    """
    from compare_separation_configs import process_comparison, DEFAULT_CONFIGS
    from project_manager import get_project_by_number, USER_FILES_DIR
    
    project = get_project_by_number(project_number, USER_FILES_DIR)
    if project is None:
        raise ValueError(f'Project {project_number} not found')
    
    process_comparison(project, DEFAULT_CONFIGS, device, cleanup=False)
    
    return {'project_number': project_number, 'comparison_completed': True}


def run_cleanup(project_number: int, threshold_db: float = -30.0, ratio: float = 10.0,
                attack_ms: float = 1.0, release_ms: float = 100.0):
    """
    Execute sidechain cleanup for a project.
    
    This is the actual work function that runs in the job queue.
    """
    from sidechain_cleanup import cleanup_project_stems
    
    # cleanup_project_stems takes project_number directly
    cleanup_project_stems(
        project_number=project_number,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms
    )
    
    return {'project_number': project_number, 'cleaned_stems_created': True}


def run_stems_to_midi(project_number: int, **kwargs):
    """
    Execute stems to MIDI conversion for a project.
    
    This is the actual work function that runs in the job queue.
    """
    # Import from stems_to_midi.py file using importlib
    # (stems_to_midi/ package directory shadows the .py file)
    import importlib.util
    import sys
    from pathlib import Path
    
    # Load stems_to_midi.py explicitly
    stems_to_midi_path = Path(__file__).parent.parent.parent / "stems_to_midi.py"
    spec = importlib.util.spec_from_file_location("stems_to_midi_cli", stems_to_midi_path)
    stems_to_midi_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stems_to_midi_cli)
    
    from project_manager import get_project_by_number, USER_FILES_DIR
    
    project = get_project_by_number(project_number, USER_FILES_DIR)
    if project is None:
        raise ValueError(f'Project {project_number} not found')
    
    stems_to_midi_cli.stems_to_midi_for_project(project, **kwargs)
    
    return {'project_number': project_number, 'midi_created': True}


def run_render_video(project_number: int, fps: int = 60, width: int = 1920, height: int = 1080, 
                     audio_source: str = 'original', include_audio: bool = None, fall_speed_multiplier: float = 1.0):
    """
    Execute MIDI to video rendering for a project.
    
    This is the actual work function that runs in the job queue.
    
    Args:
        audio_source: Audio source - None, 'original', or 'alternate_mix/{filename}'
        include_audio: DEPRECATED - kept for backward compatibility
        fall_speed_multiplier: Note fall speed multiplier (1.0 = default)
    """
    from render_midi_to_video import render_project_video
    from project_manager import get_project_by_number, USER_FILES_DIR
    
    project = get_project_by_number(project_number, USER_FILES_DIR)
    if project is None:
        raise ValueError(f'Project {project_number} not found')
    
    render_project_video(project, fps=fps, width=width, height=height, 
                        audio_source=audio_source, include_audio=include_audio,
                        fall_speed_multiplier=fall_speed_multiplier)
    
    return {'project_number': project_number, 'video_created': True}


@operations_bp.route('/separate', methods=['POST'])
def separate():
    """
    POST /api/separate
    
    Start stem separation for a project.
    
    Request body (JSON):
        {
            "project_number": 1,
            "device": "cpu",        # optional: "cpu" or "cuda"
            "wiener": 2.0,          # optional: Wiener filter exponent
            "eq": false             # optional: Apply EQ cleanup
        }
        
    Returns:
        202: Job created and queued
        400: Invalid request
        404: Project not found
        500: Internal error
        
    Response format:
        {
            "message": "Separation job started",
            "job_id": "uuid-here"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'project_number' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must include project_number'
            }), 400
        
        project_number = data['project_number']
        
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        # Extract optional parameters
        device = data.get('device', 'cpu')
        wiener = data.get('wiener', None)
        eq = data.get('eq', False)
        
        # Validate device
        if device not in ['cpu', 'cuda']:
            return jsonify({
                'error': 'Invalid device',
                'message': 'Device must be "cpu" or "cuda"'
            }), 400
        
        # Validate wiener
        if wiener is not None and wiener <= 0:
            return jsonify({
                'error': 'Invalid wiener value',
                'message': 'Wiener exponent must be positive'
            }), 400
        
        # Submit job
        job_queue = get_job_queue()
        job_id = job_queue.submit(
            operation='separate',
            func=run_separate,
            project_id=project_number,
            project_number=project_number,
            device=device,
            wiener=wiener,
            eq=eq
        )
        
        return jsonify({
            'message': 'Separation job started',
            'job_id': job_id
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to start separation',
            'message': str(e)
        }), 500


@operations_bp.route('/cleanup', methods=['POST'])
def cleanup():
    """
    POST /api/cleanup
    
    Start sidechain cleanup for a project.
    
    Request body (JSON):
        {
            "project_number": 1,
            "threshold_db": -30.0,   # optional
            "ratio": 10.0,           # optional
            "attack_ms": 1.0,        # optional
            "release_ms": 100.0      # optional
        }
        
    Returns:
        202: Job created and queued
        400: Invalid request
        404: Project not found
        500: Internal error
    """
    try:
        data = request.get_json()
        
        if not data or 'project_number' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must include project_number'
            }), 400
        
        project_number = data['project_number']
        
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        # Extract optional parameters
        threshold_db = data.get('threshold_db', -30.0)
        ratio = data.get('ratio', 10.0)
        attack_ms = data.get('attack_ms', 1.0)
        release_ms = data.get('release_ms', 100.0)
        
        # Submit job
        job_queue = get_job_queue()
        job_id = job_queue.submit(
            operation='cleanup',
            func=run_cleanup,
            project_id=project_number,
            project_number=project_number,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms
        )
        
        return jsonify({
            'message': 'Cleanup job started',
            'job_id': job_id
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to start cleanup',
            'message': str(e)
        }), 500


@operations_bp.route('/compare', methods=['POST'])
def compare():
    """
    POST /api/compare
    
    Start separation comparison with multiple configurations.
    Tests various Wiener/EQ combinations and saves to for_comparison/ directory.
    
    Request body (JSON):
        {
            "project_number": 1,
            "device": "cpu"        # optional: "cpu" or "cuda"
        }
        
    Returns:
        202: Job created and queued
        400: Invalid request
        404: Project not found
        500: Internal error
        
    Response format:
        {
            "message": "Comparison job started",
            "job_id": "uuid-here"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'project_number' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must include project_number'
            }), 400
        
        project_number = data['project_number']
        
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        # Extract optional parameters
        device = data.get('device', 'cpu')
        
        # Validate device
        if device not in ['cpu', 'cuda']:
            return jsonify({
                'error': 'Invalid device',
                'message': 'Device must be "cpu" or "cuda"'
            }), 400
        
        # Submit job
        job_queue = get_job_queue()
        job_id = job_queue.submit(
            operation='compare',
            func=run_comparison,
            project_id=project_number,
            project_number=project_number,
            device=device
        )
        
        return jsonify({
            'message': 'Comparison job started',
            'job_id': job_id
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to start comparison',
            'message': str(e)
        }), 500


@operations_bp.route('/delete-comparison', methods=['POST'])
def delete_comparison():
    """
    POST /api/delete-comparison
    
    Delete the for_comparison directory from a project.
    
    Request body (JSON):
        {
            "project_number": 1
        }
        
    Returns:
        200: Comparison folder deleted
        400: Invalid request
        404: Project or comparison folder not found
        500: Internal error
    """
    try:
        data = request.get_json()
        
        if not data or 'project_number' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must include project_number'
            }), 400
        
        project_number = data['project_number']
        
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        # Check if comparison folder exists
        comparison_dir = Path(project['path']) / 'for_comparison'
        if not comparison_dir.exists():
            return jsonify({
                'error': 'Comparison folder not found',
                'message': 'No comparison folder exists for this project'
            }), 404
        
        # Delete the comparison directory
        import shutil
        shutil.rmtree(comparison_dir)
        
        return jsonify({
            'message': 'Comparison folder deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to delete comparison folder',
            'message': str(e)
        }), 500


@operations_bp.route('/stems-to-midi', methods=['POST'])
def stems_to_midi():
    """
    POST /api/stems-to-midi
    
    Start stems to MIDI conversion for a project.
    
    Request body (JSON):
        {
            "project_number": 1,
            "onset_threshold": 0.3,  # optional
            "onset_delta": 0.01,     # optional
            "onset_wait": 3,         # optional
            "hop_length": 512,       # optional
            "min_velocity": 80,      # optional
            "max_velocity": 110,     # optional
            "tempo": 120.0,          # optional
            "detect_hihat_open": false  # optional
        }
        
    Returns:
        202: Job created and queued
        400: Invalid request
        404: Project not found
        500: Internal error
    """
    try:
        data = request.get_json()
        
        if not data or 'project_number' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must include project_number'
            }), 400
        
        project_number = data['project_number']
        
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        # Extract optional parameters (all kwargs will be passed to stems_to_midi_for_project)
        kwargs = {k: v for k, v in data.items() if k != 'project_number'}
        
        # Submit job
        job_queue = get_job_queue()
        job_id = job_queue.submit(
            operation='stems-to-midi',
            func=run_stems_to_midi,
            project_id=project_number,
            project_number=project_number,
            **kwargs
        )
        
        return jsonify({
            'message': 'MIDI conversion job started',
            'job_id': job_id
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to start MIDI conversion',
            'message': str(e)
        }), 500


@operations_bp.route('/render-video', methods=['POST'])
def render_video():
    """
    POST /api/render-video
    
    Start MIDI to video rendering for a project.
    
    Request body (JSON):
        {
            "project_number": 1,
            "fps": 60,           # optional: 30, 60, 120
            "width": 1920,       # optional
            "height": 1080,      # optional
            "audio_source": null # optional: null, 'original', or 'alternate_mix/{filename}'
            "include_audio": false,  # DEPRECATED: use audio_source instead
            "fall_speed_multiplier": 1.0  # optional: 0.5-2.0, controls note fall speed
        }
        
    Returns:
        202: Job created and queued
        400: Invalid request
        404: Project not found
        500: Internal error
    """
    try:
        data = request.get_json()
        
        if not data or 'project_number' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must include project_number'
            }), 400
        
        project_number = data['project_number']
        
        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404
        
        # Extract optional parameters
        fps = data.get('fps', 60)
        width = data.get('width', 1920)
        height = data.get('height', 1080)
        audio_source = data.get('audio_source', None)
        include_audio = data.get('include_audio', None)  # Deprecated but still supported
        fall_speed_multiplier = data.get('fall_speed_multiplier', 1.0)
        
        # Submit job
        job_queue = get_job_queue()
        job_id = job_queue.submit(
            operation='render-video',
            func=run_render_video,
            project_id=project_number,
            project_number=project_number,
            fps=fps,
            width=width,
            height=height,
            audio_source=audio_source,
            include_audio=include_audio,
            fall_speed_multiplier=fall_speed_multiplier
        )
        
        return jsonify({
            'message': 'Video rendering job started',
            'job_id': job_id
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to start video rendering',
            'message': str(e)
        }), 500
