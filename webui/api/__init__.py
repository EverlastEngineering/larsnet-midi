"""
API Blueprints

Organizes Flask API endpoints into logical blueprints.
"""

from flask import Blueprint

# Create blueprints
projects_bp = Blueprint('projects', __name__, url_prefix='/api/projects')
operations_bp = Blueprint('operations', __name__, url_prefix='/api')
upload_bp = Blueprint('upload', __name__, url_prefix='/api')
jobs_bp = Blueprint('jobs', __name__, url_prefix='/api')
downloads_bp = Blueprint('downloads', __name__, url_prefix='/api')
