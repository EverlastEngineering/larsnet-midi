"""
Upload API Endpoint

Handles file uploads and automatic project creation.
"""

from flask import jsonify, request # type: ignore
from pathlib import Path
from werkzeug.utils import secure_filename # type: ignore
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_manager import (
    create_project,
    USER_FILES_DIR
)
from webui.api import upload_bp
from webui.config import Config


@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    POST /api/upload
    
    Upload an audio file and create a new project.
    
    Request:
        - multipart/form-data with 'file' field
        - file must be audio format (.wav, .mp3, .flac, .aiff, .aif)
        
    Returns:
        201: File uploaded and project created
        400: Bad request (no file, invalid format, etc.)
        500: Internal error
        
    Response format:
        {
            "message": "File uploaded successfully",
            "project": {
                "number": 1,
                "name": "Song Name",
                "path": "/path/to/project"
            }
        }
    """
    try:
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
        
        # Check file extension
        if not Config.allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': f'File must be one of: {", ".join(Config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Save to user_files directory temporarily
        temp_path = USER_FILES_DIR / filename
        file.save(str(temp_path))
        
        try:
            # Create project using project_manager
            project = create_project(temp_path, USER_FILES_DIR, Path('.'))
            
            return jsonify({
                'message': 'File uploaded successfully',
                'project': {
                    'number': project['number'],
                    'name': project['name'],
                    'path': str(project['path'])
                }
            }), 201
            
        except Exception as e:
            # If project creation fails, clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            raise e
        
    except Exception as e:
        return jsonify({
            'error': 'Upload failed',
            'message': str(e)
        }), 500
