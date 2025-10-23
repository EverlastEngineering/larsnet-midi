"""
Configuration schema definitions for YAML validation.

Defines the expected structure of configuration files to prevent corruption
during updates. Each schema specifies which keys are dictionaries vs primitives.
"""

from typing import Dict, Any, Set


# Schema definition: True = dict/section, False = primitive value
MIDICONFIG_SCHEMA = {
    'audio': True,  # Dict containing audio processing settings
    'onset_detection': True,  # Dict containing onset detection parameters
    'kick': True,  # Dict containing kick settings
    'snare': True,  # Dict containing snare settings
    'toms': True,  # Dict containing toms settings
    'hihat': True,  # Dict containing hihat settings
    'cymbals': True,  # Dict containing cymbals settings
    'midi': True,  # Dict containing MIDI output settings
    'debug': True,  # Dict containing debug settings
    'learning_mode': True,  # Dict containing learning mode settings
}

CONFIG_SCHEMA = {
    'global': True,  # Dict containing global settings
    'audio': True,  # Dict containing audio settings
    'separation': True,  # Dict containing separation settings
    'kick': True,  # Dict containing kick settings
    'snare': True,  # Dict containing snare settings
    'hihat': True,  # Dict containing hihat settings
    'cymbals': True,  # Dict containing cymbal settings
    'toms': True,  # Dict containing tom settings
    # Add more as needed
}

EQ_SCHEMA = {
    # EQ schema - typically flat structure with band definitions
    # Add as needed
}


def get_schema(config_type: str) -> Dict[str, bool]:
    """
    Get the schema definition for a config type.
    
    Args:
        config_type: One of 'midiconfig', 'config', or 'eq'
    
    Returns:
        Schema dictionary mapping keys to True (dict) or False (primitive)
    
    Raises:
        ValueError: If config_type is invalid
    """
    schemas = {
        'midiconfig': MIDICONFIG_SCHEMA,
        'config': CONFIG_SCHEMA,
        'eq': EQ_SCHEMA,
    }
    
    if config_type not in schemas:
        raise ValueError(f"Unknown config_type: {config_type}. Must be one of: {list(schemas.keys())}")
    
    return schemas[config_type]


def validate_structure(data: Dict[str, Any], schema: Dict[str, bool]) -> tuple[bool, str]:
    """
    Validate that a configuration dictionary matches the expected schema.
    
    Args:
        data: Configuration data to validate
        schema: Schema defining expected structure
    
    Returns:
        (is_valid, error_message) tuple
    """
    errors = []
    
    for key, should_be_dict in schema.items():
        if key not in data:
            continue  # Missing keys are OK (will use defaults)
        
        value = data[key]
        is_dict = isinstance(value, dict)
        
        if should_be_dict and not is_dict:
            errors.append(
                f"'{key}' should be a dictionary but is {type(value).__name__}: {value}"
            )
        elif not should_be_dict and is_dict:
            errors.append(
                f"'{key}' should be a primitive value but is a dictionary"
            )
    
    if errors:
        return False, "; ".join(errors)
    
    return True, ""


def get_dict_keys(config_type: str) -> Set[str]:
    """
    Get the set of keys that should be dictionaries for a config type.
    
    Args:
        config_type: One of 'midiconfig', 'config', or 'eq'
    
    Returns:
        Set of keys that must be dictionaries
    """
    schema = get_schema(config_type)
    return {key for key, is_dict in schema.items() if is_dict}
