"""
Downloads API Endpoints

Provides file download endpoints for project outputs.
"""

from flask import send_file, jsonify, abort # type: ignore
from pathlib import Path
import sys
import zipfile
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_manager import get_project_by_number
from webui.api import downloads_bp


@downloads_bp.route('/projects/<int:project_number>/download/<file_type>/<path:filename>', methods=['GET'])
def download_file(project_number, file_type, filename):
    """
    GET /api/projects/:project_number/download/:file_type/:filename
    
    Download a specific file from a project.
    
    Args:
        project_number: Project number
        file_type: Type of file (stems, cleaned, midi, video)
        filename: Name of file to download
        
    Returns:
        200: File download
        404: Project or file not found
        400: Invalid file type
    """
    try:
        # Get project
        project = get_project_by_number(project_number)
        if not project:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project #{project_number} does not exist'
            }), 404
        
        # Validate file type
        valid_types = ['stems', 'cleaned', 'midi', 'video']
        if file_type not in valid_types:
            return jsonify({
                'error': 'Invalid file type',
                'message': f'File type must be one of: {", ".join(valid_types)}'
            }), 400
        
        # Build file path
        project_path = Path(project['path'])
        file_path = project_path / file_type / filename
        
        # Security check: ensure file is within project directory
        if not str(file_path.resolve()).startswith(str(project_path.resolve())):
            return jsonify({
                'error': 'Invalid file path',
                'message': 'File path is outside project directory'
            }), 400
        
        # Check if file exists
        if not file_path.exists():
            return jsonify({
                'error': 'File not found',
                'message': f'File {filename} not found in {file_type}'
            }), 404
        
        # Send file - stream media files for playback, download others
        if file_type == 'video':
            # Stream video with range request support for playback
            return send_file(
                file_path,
                mimetype='video/mp4',
                as_attachment=False,
                conditional=True  # Enable range request support
            )
        elif file_type in ['stems', 'cleaned'] and filename.endswith('.wav'):
            # Stream audio with range request support for playback
            return send_file(
                file_path,
                mimetype='audio/wav',
                as_attachment=False,
                conditional=True  # Enable range request support
            )
        else:
            # Download other file types
            return send_file(
                file_path,
                as_attachment=True,
                download_name=filename
            )
        
    except Exception as e:
        return jsonify({
            'error': 'Download failed',
            'message': str(e)
        }), 500


@downloads_bp.route('/projects/<int:project_number>/download/<file_type>', methods=['GET'])
def download_all_files(project_number, file_type):
    """
    GET /api/projects/:project_number/download/:file_type
    
    Download all files of a type as a ZIP archive.
    
    Args:
        project_number: Project number
        file_type: Type of files (stems, cleaned, midi, video)
        
    Returns:
        200: ZIP file download
        404: Project not found or no files
        400: Invalid file type
    """
    try:
        # Get project
        project = get_project_by_number(project_number)
        if not project:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project #{project_number} does not exist'
            }), 404
        
        # Validate file type
        valid_types = ['stems', 'cleaned', 'midi', 'video']
        if file_type not in valid_types:
            return jsonify({
                'error': 'Invalid file type',
                'message': f'File type must be one of: {", ".join(valid_types)}'
            }), 400
        
        # Get directory
        project_path = Path(project['path'])
        file_dir = project_path / file_type
        
        if not file_dir.exists():
            return jsonify({
                'error': 'Directory not found',
                'message': f'No {file_type} directory found'
            }), 404
        
        # Get all files
        files = list(file_dir.glob('*'))
        files = [f for f in files if f.is_file()]
        
        if not files:
            return jsonify({
                'error': 'No files found',
                'message': f'No files in {file_type} directory'
            }), 404
        
        # If only one file, send it directly
        if len(files) == 1:
            return send_file(
                files[0],
                as_attachment=True,
                download_name=files[0].name
            )
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                # Add file to ZIP with just its filename (no path)
                zip_file.write(file_path, arcname=file_path.name)
        
        # Rewind buffer
        zip_buffer.seek(0)
        
        # Generate ZIP filename
        project_name = project['name'].replace(' ', '_')
        zip_filename = f"{project_name}_{file_type}.zip"
        
        # Send ZIP
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
    except Exception as e:
        return jsonify({
            'error': 'Download failed',
            'message': str(e)
        }), 500
