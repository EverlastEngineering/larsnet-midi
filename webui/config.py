"""
Flask Application Configuration

Provides configuration settings for the LarsNet Web UI Flask application.
Supports development and production environments.
"""

import os
from pathlib import Path


class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Application settings
    APP_NAME = 'LarsNet MIDI'
    APP_VERSION = '0.1.0'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    UPLOAD_FOLDER = Path('user_files')
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'aiff', 'aif'}
    
    # Job queue settings
    MAX_CONCURRENT_JOBS = 2  # Limit concurrent operations
    JOB_TIMEOUT = 3600  # 1 hour timeout for long operations
    
    # API settings
    API_PREFIX = '/api'
    CORS_ENABLED = True
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    @staticmethod
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # In production, SECRET_KEY must be set via environment variable
    

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """
    Get configuration for specified environment.
    
    Args:
        env: Environment name ('development', 'production', 'testing')
             If None, uses FLASK_ENV environment variable or 'default'
    
    Returns:
        Configuration class
    """
    if env is None:
        env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])
