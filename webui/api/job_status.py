"""
Job Status API Endpoints

Provides job status tracking and Server-Sent Events for real-time updates.
"""

from flask import jsonify, Response, stream_with_context
import json
import time
from webui.api import jobs_bp
from webui.jobs import get_job_queue


@jobs_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """
    GET /api/jobs
    
    List all jobs in the queue.
    
    Returns:
        200: List of all jobs
        500: Internal error
        
    Response format:
        {
            "jobs": [
                {
                    "id": "uuid-here",
                    "operation": "separate",
                    "project_id": 1,
                    "status": "running",
                    "progress": 50,
                    "logs": [...],
                    "created_at": "2025-10-19T12:00:00",
                    ...
                },
                ...
            ]
        }
    """
    try:
        job_queue = get_job_queue()
        jobs = job_queue.get_all_jobs()
        
        return jsonify({
            'jobs': [job.to_dict() for job in jobs]
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to list jobs',
            'message': str(e)
        }), 500


@jobs_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """
    GET /api/jobs/:job_id
    
    Get status of a specific job.
    
    Args:
        job_id: Job ID
        
    Returns:
        200: Job details
        404: Job not found
        500: Internal error
        
    Response format:
        {
            "job": {
                "id": "uuid-here",
                "operation": "separate",
                "project_id": 1,
                "status": "running",
                "progress": 50,
                "logs": [...],
                "result": null,
                "error": null,
                "created_at": "2025-10-19T12:00:00",
                "started_at": "2025-10-19T12:00:05",
                "completed_at": null
            }
        }
    """
    try:
        job_queue = get_job_queue()
        job = job_queue.get_job(job_id)
        
        if job is None:
            return jsonify({
                'error': 'Job not found',
                'message': f'No job with ID {job_id}'
            }), 404
        
        return jsonify({
            'job': job.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get job',
            'message': str(e)
        }), 500


@jobs_bp.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """
    POST /api/jobs/:job_id/cancel
    
    Cancel a job.
    
    Note: If the job is already running, it will continue until
    the current operation completes, but will be marked as cancelled.
    
    Args:
        job_id: Job ID
        
    Returns:
        200: Job cancelled
        404: Job not found
        400: Job cannot be cancelled (already completed/failed)
        500: Internal error
    """
    try:
        job_queue = get_job_queue()
        
        if job_queue.cancel_job(job_id):
            return jsonify({
                'message': 'Job cancelled',
                'job_id': job_id
            }), 200
        else:
            job = job_queue.get_job(job_id)
            if job is None:
                return jsonify({
                    'error': 'Job not found',
                    'message': f'No job with ID {job_id}'
                }), 404
            else:
                return jsonify({
                    'error': 'Cannot cancel job',
                    'message': f'Job is already {job.status.value}'
                }), 400
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to cancel job',
            'message': str(e)
        }), 500


@jobs_bp.route('/jobs/<job_id>/stream', methods=['GET'])
def stream_job_status(job_id):
    """
    GET /api/jobs/:job_id/stream
    
    Stream job status updates using Server-Sent Events (SSE).
    
    The client should connect to this endpoint and listen for events.
    Events are sent whenever the job status changes.
    
    Args:
        job_id: Job ID
        
    Returns:
        200: SSE stream
        404: Job not found
        
    Event format:
        event: job_update
        data: {"id": "uuid", "status": "running", "progress": 50, ...}
        
        event: job_complete
        data: {"id": "uuid", "status": "completed", "result": {...}}
        
        event: job_error
        data: {"id": "uuid", "status": "failed", "error": "..."}
    """
    job_queue = get_job_queue()
    job = job_queue.get_job(job_id)
    
    if job is None:
        return jsonify({
            'error': 'Job not found',
            'message': f'No job with ID {job_id}'
        }), 404
    
    def generate():
        """Generate SSE events for job updates"""
        last_status = None
        last_log_count = 0
        
        while True:
            job = job_queue.get_job(job_id)
            
            if job is None:
                # Job was deleted
                yield f'event: job_error\n'
                yield f'data: {json.dumps({"error": "Job no longer exists"})}\n\n'
                break
            
            # Check if status changed or new logs added
            status_changed = job.status != last_status
            new_logs = len(job.logs) > last_log_count
            
            if status_changed or new_logs:
                last_status = job.status
                last_log_count = len(job.logs)
                
                # Send update
                job_data = job.to_dict()
                
                if job.status.value in ('completed', 'failed', 'cancelled'):
                    # Final event
                    event_name = 'job_complete' if job.status.value == 'completed' else 'job_error'
                    yield f'event: {event_name}\n'
                    yield f'data: {json.dumps(job_data)}\n\n'
                    break
                else:
                    # Progress update
                    yield f'event: job_update\n'
                    yield f'data: {json.dumps(job_data)}\n\n'
            
            # Poll every 500ms
            time.sleep(0.5)
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@jobs_bp.route('/projects/<int:project_number>/jobs', methods=['GET'])
def get_project_jobs(project_number):
    """
    GET /api/projects/:project_number/jobs
    
    Get all jobs for a specific project.
    
    Args:
        project_number: Project number
        
    Returns:
        200: List of jobs for project
        500: Internal error
    """
    try:
        job_queue = get_job_queue()
        jobs = job_queue.get_project_jobs(project_number)
        
        return jsonify({
            'jobs': [job.to_dict() for job in jobs]
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get project jobs',
            'message': str(e)
        }), 500
