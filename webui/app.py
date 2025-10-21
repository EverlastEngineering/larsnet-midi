"""
LarsNet Web UI Flask Application

Main application entry point. Configures Flask, registers blueprints,
and starts the web server.

Usage:
    python -m webui.app
    
    Or with environment:
    FLASK_ENV=production python -m webui.app
"""

from flask import Flask, render_template, send_from_directory # type: ignore
from flask_cors import CORS # type: ignore
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from webui.config import get_config
from webui.jobs import get_job_queue, shutdown_job_queue

# Import API blueprints
from webui.api.projects import projects_bp
from webui.api.operations import operations_bp
from webui.api.upload import upload_bp
from webui.api.job_status import jobs_bp
from webui.api.downloads import downloads_bp


def create_app(config_name=None):
    """
    Create and configure Flask application.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
                    If None, uses FLASK_ENV environment variable
    
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enable CORS if configured
    if config.CORS_ENABLED:
        CORS(app)
    
    # Register blueprints
    app.register_blueprint(projects_bp)
    app.register_blueprint(operations_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(downloads_bp)
    
    # Initialize job queue
    get_job_queue()
    
    # Main routes
    @app.route('/')
    def index():
        """Serve main web UI page"""
        return render_template('index.html')
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files"""
        return send_from_directory('static', filename)
    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        return {'status': 'healthy', 'version': config.APP_VERSION}, 200
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return {'error': 'Not found', 'message': str(error)}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        app.logger.error(f'Internal error: {error}')
        return {
            'error': 'Internal server error',
            'message': 'An unexpected error occurred. Please check the logs.'
        }, 500
    
    # Cleanup on shutdown
    @app.teardown_appcontext
    def cleanup(error=None):
        """Cleanup resources on shutdown"""
        if error:
            app.logger.error(f'Teardown error: {error}')
    
    return app


def main():
    """Run the Flask development server"""
    app = create_app()
    
    print("\n" + "="*60)
    print(f"LarsNet Web UI v{app.config['APP_VERSION']}")
    print("="*60)
    print(f"\nStarting server at http://0.0.0.0:5000")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        shutdown_job_queue()
        print("Goodbye!\n")


if __name__ == '__main__':
    main()
