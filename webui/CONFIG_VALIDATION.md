# Configuration Validation System

## Overview

The webui configuration system includes schema validation to prevent YAML corruption during save operations. This system ensures that dictionary structures in configuration files are preserved and not accidentally replaced with primitive values.

## Files

- **`config_schema.py`**: Defines expected structure for each config type
- **`config_engine.py`**: Enforces validation during update and save operations
- **`test_config_schema_validation.py`**: Tests for validation system

## How It Works

### Schema Definition

Each config type (midiconfig, config, eq) has a schema that defines which keys should contain dictionaries vs primitive values:

```python
MIDICONFIG_SCHEMA = {
    'audio': True,        # Must be a dict
    'onset_detection': False,  # Must be primitive (float)
    'kick': True,         # Must be a dict
    'toms': True,         # Must be a dict
    # ...
}
```

### Update Validation

When updating a value via `update_value()`, the system:

1. Checks if attempting to replace a dict with a primitive
2. Returns `(False, error_message)` if invalid
3. Returns `(True, "")` if valid

Example protection:
```python
# PROTECTED: Cannot do this
engine.update_value(['toms'], 600)  
# Returns: (False, "Cannot replace dictionary 'toms' with primitive...")

# ALLOWED: Can update nested values
engine.update_value(['toms', 'midi_note_low'], 45)
# Returns: (True, "")
```

### Save Validation

Before writing to disk, `save()` validates the entire structure:

```python
engine.save()  # Raises RuntimeError if structure is invalid
```

This catches any manual corruption that bypassed `update_value()`.

## Protected Against

✅ Replacing dict sections with primitives (e.g., `toms: 600`)  
✅ Malformed updates from API  
✅ Manual data manipulation before save  

## API Integration

The config API in `webui/api/config.py` handles the tuple return from `update_value()`:

```python
success, error = engine.update_value(path, value)
if not success:
    return jsonify({'error': error}), 400
```

## Testing

Run validation tests:
```bash
pytest webui/test_config_schema_validation.py -v
```

9 tests cover:
- Schema validation logic
- Update protection
- Save-time validation
- Error message clarity
