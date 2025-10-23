"""
Configuration API endpoints for Web UI

Provides endpoints to read, validate, and update YAML configuration files
for projects using the YAMLConfigEngine.
"""

from flask import Blueprint, jsonify, request
from pathlib import Path
import glob
from typing import Dict, Any, List

from ..config_engine import YAMLConfigEngine, get_config_engine

config_bp = Blueprint('config', __name__, url_prefix='/api/config')


@config_bp.route('/<int:project_id>/<config_type>', methods=['GET'])
def get_config(project_id: int, config_type: str):
    """
    Get parsed configuration for a project.
    
    Args:
        project_id: Project number
        config_type: One of 'config', 'midiconfig', or 'eq'
    
    Returns:
        JSON with sections and fields for UI rendering
    """
    try:
        engine = get_config_engine(project_id, config_type)
        sections = engine.parse()
        
        return jsonify({
            'success': True,
            'config_type': config_type,
            'project_id': project_id,
            'sections': [section.to_dict() for section in sections]
        })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load configuration: {str(e)}'
        }), 500


@config_bp.route('/<int:project_id>/<config_type>/validate', methods=['POST'])
def validate_config(project_id: int, config_type: str):
    """
    Validate configuration values without saving.
    
    Expected JSON body:
    {
        "updates": [
            {"path": ["kick", "midi_note"], "value": 36},
            {"path": ["audio", "force_mono"], "value": true}
        ]
    }
    
    Returns:
        JSON with validation results for each field
    """
    try:
        engine = get_config_engine(project_id, config_type)
        engine.load()
        
        data = request.get_json()
        if not data or 'updates' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "updates" in request body'
            }), 400
        
        # Apply updates to in-memory copy (don't save)
        validation_errors = []
        for update in data['updates']:
            path = update.get('path')
            value = update.get('value')
            
            if not path:
                validation_errors.append({
                    'path': None,
                    'error': 'Missing path in update'
                })
                continue
            
            success = engine.update_value(path, value)
            if not success:
                validation_errors.append({
                    'path': '.'.join(path),
                    'error': f'Field not found: {".".join(path)}'
                })
        
        # Validate all fields
        all_errors = engine.validate_all()
        
        # Convert to flat list for response
        for section_name, field_errors in all_errors.items():
            for field_key, error_msg in field_errors:
                validation_errors.append({
                    'path': f'{section_name}.{field_key}',
                    'error': error_msg
                })
        
        return jsonify({
            'success': len(validation_errors) == 0,
            'errors': validation_errors
        })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Validation failed: {str(e)}'
        }), 500


@config_bp.route('/<int:project_id>/<config_type>', methods=['POST'])
def update_config(project_id: int, config_type: str):
    """
    Update configuration values and save to file.
    
    Expected JSON body:
    {
        "updates": [
            {"path": ["kick", "midi_note"], "value": 36},
            {"path": ["audio", "force_mono"], "value": true}
        ]
    }
    
    Returns:
        JSON with success status and any errors
    """
    try:
        engine = get_config_engine(project_id, config_type)
        engine.load()
        
        data = request.get_json()
        if not data or 'updates' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "updates" in request body'
            }), 400
        
        # Apply all updates
        update_errors = []
        for update in data['updates']:
            path = update.get('path')
            value = update.get('value')
            
            if not path:
                update_errors.append({
                    'path': None,
                    'error': 'Missing path in update'
                })
                continue
            
            success, error_msg = engine.update_value(path, value)
            if not success:
                update_errors.append({
                    'path': '.'.join(path),
                    'error': error_msg or f'Field not found: {".".join(path)}'
                })
        
        # If any updates failed, don't save
        if update_errors:
            return jsonify({
                'success': False,
                'errors': update_errors
            }), 400
        
        # Validate before saving
        validation_errors = engine.validate_all()
        if validation_errors:
            # Convert to flat list
            error_list = []
            for section_name, field_errors in validation_errors.items():
                for field_key, error_msg in field_errors:
                    error_list.append({
                        'path': f'{section_name}.{field_key}',
                        'error': error_msg
                    })
            
            return jsonify({
                'success': False,
                'errors': error_list,
                'message': 'Validation failed. Changes not saved.'
            }), 400
        
        # Save changes
        engine.save()
        
        return jsonify({
            'success': True,
            'message': f'Configuration saved successfully',
            'updated_count': len(data['updates'])
        })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to update configuration: {str(e)}'
        }), 500


@config_bp.route('/<int:project_id>/<config_type>/reset', methods=['POST'])
def reset_config(project_id: int, config_type: str):
    """
    Reset configuration to default values.
    
    Copies the default config file from the root directory to the project directory.
    
    Returns:
        JSON with success status
    """
    try:
        import shutil
        
        # Get project directory
        project_pattern = f'/app/user_files/{project_id} - *'
        matches = glob.glob(project_pattern)
        if not matches:
            return jsonify({
                'success': False,
                'error': f'Project {project_id} not found'
            }), 404
        
        project_dir = Path(matches[0])
        
        # Get default config file
        default_config = Path(f'/app/{config_type}.yaml')
        if not default_config.exists():
            return jsonify({
                'success': False,
                'error': f'Default config file not found: {config_type}.yaml'
            }), 404
        
        # Copy default to project
        target_config = project_dir / f'{config_type}.yaml'
        shutil.copy2(default_config, target_config)
        
        return jsonify({
            'success': True,
            'message': f'Configuration reset to defaults'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to reset configuration: {str(e)}'
        }), 500
