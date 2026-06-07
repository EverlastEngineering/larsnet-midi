"""
Operations API Endpoints

Handles triggering of DrumToMIDI operations (separate, cleanup, MIDI, video).
All operations run asynchronously via the job queue.
"""

from flask import jsonify, request # type: ignore
from pathlib import Path
import sys
import platform

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_manager import get_project_by_number, USER_FILES_DIR
from webui.api import operations_bp
from webui.jobs import get_job_queue


def run_separate(project_number: int, device: str = 'cpu', overlap: int = 4, wiener: float = None):
    """
    Execute stem separation for a project.
    
    This is the actual work function that runs in the job queue.
    """
    from separate import separate_project
    from project_manager import get_project_by_number, USER_FILES_DIR
    
    project = get_project_by_number(project_number, USER_FILES_DIR)
    if project is None:
        raise ValueError(f'Project {project_number} not found')
    
    separate_project(project, overlap=overlap, wiener_exponent=wiener, device=device)
    
    return {'project_number': project_number, 'stems_created': True}


def run_cleanup(project_number: int, threshold_db: float = -30.0, ratio: float = 10.0,
                attack_ms: float = 1.0, release_ms: float = 100.0):
    """
    Execute sidechain cleanup for a project.
    
    This is the actual work function that runs in the job queue.
    """
    from sidechain_shell import cleanup_project_stems
    
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
    # Import from stems_to_midi_cli.py file using importlib
    import importlib.util
    from pathlib import Path
    
    # Load stems_to_midi_cli.py explicitly
    stems_to_midi_path = Path(__file__).parent.parent.parent / "stems_to_midi_cli.py"
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
                     audio_source: str = 'original', include_audio: bool = None, fall_speed_multiplier: float = 1.0,
                     use_moderngl: bool = None):
    """
    Execute MIDI to video rendering for a project.
    
    This is the actual work function that runs in the job queue.
    
    Args:
        audio_source: Audio source - None, 'original', or 'alternate_mix/{filename}'
        include_audio: DEPRECATED - kept for backward compatibility
        fall_speed_multiplier: Note fall speed multiplier (1.0 = default)
        use_moderngl: Use GPU-accelerated ModernGL renderer (default: True on macOS, False otherwise)
    """
    from render_midi_video_shell import render_project_video
    from project_manager import get_project_by_number, USER_FILES_DIR
    
    # Auto-detect ModernGL on macOS if not explicitly specified
    if use_moderngl is None:
        use_moderngl = platform.system() == 'Darwin'
    
    project = get_project_by_number(project_number, USER_FILES_DIR)
    if project is None:
        raise ValueError(f'Project {project_number} not found')
    
    render_project_video(project, fps=fps, width=width, height=height, 
                        audio_source=audio_source, include_audio=include_audio,
                        fall_speed_multiplier=fall_speed_multiplier,
                        use_moderngl=use_moderngl)
    
    return {'project_number': project_number, 'video_created': True}


@operations_bp.route('/separate', methods=['POST'])
def separate():
    """
    POST /api/separate
    
    Start stem separation for a project.
    
    Request body (JSON):
        {
            "project_number": 1,
            "device": "auto",       # optional: "auto", "cpu", "cuda", or "mps"
            "overlap": 4,           # optional: MDX23C overlap (2-8, default: 4)
            "wiener": 2.0           # optional: Wiener filter exponent
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
        device = data.get('device', 'auto')
        overlap = data.get('overlap', 4)
        wiener = data.get('wiener', None)
        
        # Validate overlap
        if overlap < 2 or overlap > 50:
            return jsonify({
                'error': 'Invalid overlap',
                'message': 'Overlap must be between 2 and 50'
            }), 400
        
        # Auto-detect device if requested
        if device == 'auto':
            from device_shell import detect_best_device
            device = detect_best_device(verbose=False)
        
        # Validate device
        if device not in ['cpu', 'cuda', 'mps']:
            return jsonify({
                'error': 'Invalid device',
                'message': 'Device must be "cpu", "cuda", "mps", or "auto"'
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
            overlap=overlap,
            wiener=wiener
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
            "tempo": 120.0           # optional
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


@operations_bp.route('/rebuild-midi', methods=['POST'])
def rebuild_midi():
    """
    POST /api/rebuild-midi

    Re-filter cached analysis data and rebuild MIDI without re-running
    audio detection. Returns updated analysis data synchronously (no job queue).

    Request body (JSON):
        {
            "project_number": 1,
            "stem_types": ["kick", "snare"],   # optional: null = all stems
            "honor_overrides": true,            # optional: default true
            "config_overrides": {               # optional: bug D
                "filtering.reverb_continuation_attack_threshold": 0.3,
                "kick.geomean_threshold": 600,
                "hihat.open_geomean_min": 200
            }
        }

    Returns:
        200: Rebuild complete with analysis data
        400: Invalid request or missing analysis
        404: Project not found
        500: Internal error

    Response format:
        {
            "success": true,
            "stems_rebuilt": ["kick", "snare"],
            "elapsed_ms": 42,
            "analysis_data": { ... },
            "events_by_stem": { ... },
            "data_integrity_warnings": [...]    # bug C
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
        stem_types = data.get('stem_types', None)
        honor_overrides = data.get('honor_overrides', True)
        # Bug D: WebUI slider values (e.g. reverb_continuation_attack_threshold)
        # must reach the server so the actual filter matches what the user
        # sees in the tuning panel. Keys are dotted YAML paths.
        config_overrides = data.get('config_overrides', None)

        # Run rebuild synchronously (sub-second, no job queue needed)
        from stems_to_midi.rebuild_shell import rebuild_midi_for_project

        result = rebuild_midi_for_project(
            project_dir=project['path'],
            stem_types=stem_types,
            honor_overrides=honor_overrides,
            config_overrides=config_overrides,
        )

        if not result['success']:
            status_code = 400
            if result.get('requires_full_pipeline'):
                status_code = 409  # Conflict — needs full pipeline
            return jsonify(result), status_code

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to rebuild MIDI',
            'message': str(e)
        }), 500


@operations_bp.route('/reclassify', methods=['POST'])
def reclassify():
    """
    POST /api/reclassify

    Re-run note classification on KEPT events with optional config overrides.
    Lightweight preview endpoint — no MIDI rebuild, no disk write. Returns
    updated classification fields for each event so the frontend can
    re-render with note-type colors.

    Request body (JSON):
        {
            "project_number": 1,
            "stem_type": "hihat",
            "config_overrides": {
                "open_geomean_min": 300,
                "open_sustain_ms": 120
            }
        }

    Returns:
        200: Classification results
        400: Invalid request
        404: Project or analysis not found
        500: Internal error

    Response format:
        {
            "events": [
                {
                    "time": 0.5,
                    "note": 46,
                    "hihat_state": "open",
                    "classification": null
                },
                ...
            ]
        }
    """
    try:
        data = request.get_json()

        if not data or 'project_number' not in data or 'stem_type' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must include project_number and stem_type'
            }), 400

        project_number = data['project_number']
        stem_type = data['stem_type']
        config_overrides = data.get('config_overrides', {})

        # Validate project exists
        project = get_project_by_number(project_number, USER_FILES_DIR)
        if project is None:
            return jsonify({
                'error': 'Project not found',
                'message': f'No project with number {project_number}'
            }), 404

        # Load analysis data
        import json
        import copy
        midi_dir = project['path'] / 'midi'
        analysis_files = list(midi_dir.glob('*.analysis.json'))
        if not analysis_files:
            return jsonify({
                'error': 'No analysis data',
                'message': 'No analysis.json found — run MIDI conversion first'
            }), 404

        with open(analysis_files[0], 'r') as f:
            analysis_data = json.load(f)

        stem_data = analysis_data.get('stems', {}).get(stem_type)
        if not stem_data:
            return jsonify({
                'error': 'Stem not found',
                'message': f'No analysis data for stem: {stem_type}'
            }), 404

        # Get KEPT events (work on copies to avoid side effects)
        configured_events = stem_data.get('events_configured', [])
        kept_events = [
            copy.deepcopy(e) for e in configured_events
            if e.get('status') == 'KEPT'
        ]

        if not kept_events:
            return jsonify({'events': []}), 200

        # Load config and merge overrides
        from stems_to_midi.config import load_config, DrumMapping
        from stems_to_midi.note_classification_core import (
            classify_notes,
            analyze_clusters,
        )

        config = load_config(project['path'] / 'midiconfig.yaml')
        drum_mapping = DrumMapping.from_config(config)

        # Apply config overrides for this stem's classification params
        if config_overrides:
            if stem_type not in config:
                config[stem_type] = {}
            for key, value in config_overrides.items():
                config[stem_type][key] = value

        # Run classification. force_reclassify=True because the reclassify
        # endpoint is only called when the user has changed a classification
        # slider (open_geomean_min, open_sustain_ms, expected_clusters, or
        # cluster_feature) — we WANT the new thresholds to take effect on
        # every event, not preserve the old hihat_state / classification.
        classify_notes(kept_events, stem_type, drum_mapping, config,
                       force_reclassify=True)

        # Analyze cluster characteristics for the UI
        cluster_info = analyze_clusters(
            kept_events, stem_type, drum_mapping,
        )

        # Return minimal payload: time + classification fields
        result_events = []
        for event in kept_events:
            result_event = {'time': event.get('time')}
            if 'note' in event:
                result_event['note'] = event['note']
            if 'hihat_state' in event:
                result_event['hihat_state'] = event['hihat_state']
            if 'classification' in event:
                result_event['classification'] = event['classification']
            result_events.append(result_event)

        return jsonify({
            'events': result_events,
            'cluster_info': cluster_info,
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to reclassify',
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
            "fall_speed_multiplier": 1.0,  # optional: 0.5-2.0, controls note fall speed
            "use_moderngl": null  # optional: use GPU-accelerated renderer (default: true on macOS)
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
        use_moderngl = data.get('use_moderngl', None)  # None allows platform detection
        
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
            fall_speed_multiplier=fall_speed_multiplier,
            use_moderngl=use_moderngl
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
