# YAML Configuration Engine

## Overview

The YAML Configuration Engine provides automatic parsing, validation, and round-trip editing of YAML configuration files with comment preservation. It extracts inline comments to generate user-friendly labels and descriptions for web UI controls.

## Architecture

### Core Classes

```
ValidationRule
  └─ Defines validation constraints (min/max, allowed values, regex)

ConfigField
  ├─ Represents a single configuration parameter
  ├─ Automatically infers type (bool/int/float/string/path)
  ├─ Extracts validation rules from comments
  └─ Converts to UI control specification

ConfigSection
  ├─ Groups related fields (e.g., 'kick', 'audio')
  └─ Validates all fields in section

YAMLConfigEngine
  ├─ Loads YAML with ruamel.yaml (preserves formatting)
  ├─ Parses into ConfigSection/ConfigField hierarchy
  ├─ Updates values with type preservation
  └─ Saves changes preserving comments and structure
```

## Usage

### Backend (Python)

```python
from webui.config_engine import YAMLConfigEngine

# Load and parse YAML
engine = YAMLConfigEngine('/path/to/config.yaml')
sections = engine.parse()

# Convert to UI-ready format
ui_data = [section.to_dict() for section in sections]

# Update value
engine.update_value(['kick', 'midi_note'], 38)

# Validate all fields
errors = engine.validate_all()

# Save (preserves comments and formatting)
engine.save()
```

### Frontend (JavaScript)

```javascript
// GET configuration for UI rendering
const response = await fetch('/api/config/1/midiconfig');
const { sections } = await response.json();

// sections is an array of:
// {
//   name: 'kick',
//   label: 'Kick',
//   description: 'Kick drum settings',
//   fields: [
//     {
//       key: 'midi_note',
//       path: 'kick',
//       type: 'int',
//       value: 36,
//       label: 'Midi Note',
//       description: 'MIDI note number',
//       validation: { min: 0, max: 127, required: true }
//     },
//     // ...more fields
//   ]
// }

// Update configuration
await fetch('/api/config/1/midiconfig', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    updates: [
      { path: ['kick', 'midi_note'], value: 38 },
      { path: ['audio', 'force_mono'], value: true }
    ]
  })
});
```

## Type Inference

Types are automatically inferred from Python values:

| Python Type | Field Type | UI Control |
|------------|-----------|-----------|
| `bool` | `'bool'` | Checkbox |
| `int` | `'int'` | Number input (step=1) |
| `float` | `'float'` | Number input (step=0.01) |
| `str` | `'string'` | Text input |
| `str` with `/` or `.pth` | `'path'` | File path input |

## Validation Rules

Validation rules are extracted from inline comments:

```yaml
threshold: 0.5  # Detection threshold (0-1)
# Auto-extracts: min_value=0, max_value=1

midi_note: 36   # MIDI note number
# Auto-detects: min_value=0, max_value=127 (MIDI range)

model_path: '/app/models/kick.pth'  # Path to model
# Auto-detects: field_type='path'
```

### Custom Validation

Add custom validation by extending `ValidationRule`:

```python
rule = ValidationRule(
    min_value=0.0,
    max_value=1.0,
    allowed_values=['cpu', 'cuda', 'mps'],
    must_exist=True,  # For file paths
    regex=r'^[a-z_]+$'  # Custom pattern
)

is_valid, error_msg = rule.validate(value)
```

## Comment Extraction

The engine extracts inline comments for UI labels:

```yaml
audio:
  force_mono: true     # Convert stereo to mono before analysis
  #                    ^ This becomes the field description
  
  peak_window_sec: 0.10  # Window size for peak detection (seconds)
  #                      ^ Extracted as description + validation hint
```

## Round-Trip Editing

Uses `ruamel.yaml` to preserve:
- Comments (inline and preceding)
- Key ordering
- Formatting and indentation
- Blank lines

```python
# Before
"""
kick:
  midi_note: 36  # MIDI note number
  threshold: 0.5  # Detection threshold (0-1)
"""

# Update value
engine.update_value(['kick', 'midi_note'], 38)
engine.save()

# After (comments preserved!)
"""
kick:
  midi_note: 38  # MIDI note number
  threshold: 0.5  # Detection threshold (0-1)
"""
```

## API Endpoints

### GET /api/config/:project_id/:config_type

Get parsed configuration with UI-ready structure.

**Parameters:**
- `project_id`: Project number (1, 2, 3, etc.)
- `config_type`: One of `'config'`, `'midiconfig'`, `'eq'`

**Response:**
```json
{
  "success": true,
  "config_type": "midiconfig",
  "project_id": 1,
  "sections": [...]
}
```

### POST /api/config/:project_id/:config_type

Update configuration values and save to file.

**Request Body:**
```json
{
  "updates": [
    { "path": ["kick", "midi_note"], "value": 38 },
    { "path": ["audio", "force_mono"], "value": true }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration saved successfully",
  "updated_count": 2
}
```

### POST /api/config/:project_id/:config_type/validate

Validate configuration changes without saving.

**Request Body:** Same as update endpoint

**Response:**
```json
{
  "success": false,
  "errors": [
    {
      "path": "kick.midi_note",
      "error": "Value must be <= 127"
    }
  ]
}
```

### POST /api/config/:project_id/:config_type/reset

Reset configuration to default values (copies from root directory).

**Response:**
```json
{
  "success": true,
  "message": "Configuration reset to defaults"
}
```

## Testing

Run tests with pytest:

```bash
# Inside Docker container
pytest webui/test_config_engine.py -v
pytest webui/test_config_api.py -v
```

Tests cover:
- Type inference for all types
- Comment extraction and parsing
- Validation rule creation and execution
- Round-trip save with comment preservation
- Type preservation during updates
- API endpoints (GET, POST, validate, reset)
- Error handling and edge cases

## Frontend Integration

Example React/Vue/vanilla JS component:

```javascript
class ConfigPanel {
  async loadConfig(projectId, configType) {
    const res = await fetch(`/api/config/${projectId}/${configType}`);
    const { sections } = await res.json();
    
    sections.forEach(section => {
      this.renderSection(section);
    });
  }
  
  renderSection(section) {
    section.fields.forEach(field => {
      switch (field.type) {
        case 'bool':
          return this.renderCheckbox(field);
        case 'int':
        case 'float':
          return this.renderNumberInput(field);
        case 'string':
        case 'path':
          return this.renderTextInput(field);
      }
    });
  }
  
  renderNumberInput(field) {
    const input = document.createElement('input');
    input.type = 'number';
    input.value = field.value;
    input.min = field.validation.min;
    input.max = field.validation.max;
    input.step = field.type === 'int' ? 1 : 0.01;
    
    const label = document.createElement('label');
    label.textContent = field.label;
    label.title = field.description;  // Tooltip
    
    return { label, input };
  }
  
  async saveConfig(projectId, configType, updates) {
    const res = await fetch(`/api/config/${projectId}/${configType}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates })
    });
    
    const { success, errors } = await res.json();
    
    if (!success) {
      this.showErrors(errors);
    }
  }
}
```

## Configuration Files

The engine works with three configuration files per project:

1. **config.yaml** - Audio processing and model settings
   - Global parameters (sample rate, segment length)
   - Per-stem model paths
   - Data augmentation settings (training only)

2. **midiconfig.yaml** - MIDI conversion settings
   - Audio processing (mono conversion, silence threshold)
   - Onset detection parameters
   - Per-stem settings (thresholds, spectral filtering, MIDI notes)
   - Advanced: pitch detection, statistical filtering

3. **eq.yaml** - Frequency filtering
   - Per-stem highpass/lowpass frequencies
   - Applied after stem separation to reduce bleed

## Design Principles

1. **Functional Core**: ConfigField/Section/Engine are pure data transformations
2. **Imperative Shell**: API endpoints handle HTTP and file I/O
3. **Comment-Driven**: UI labels and hints extracted from YAML comments
4. **Validation First**: Rules created and applied automatically
5. **Round-Trip Safe**: Preserves all formatting and comments
6. **Type Safe**: Values maintain their types through update cycle

## Future Enhancements

Potential improvements for future phases:

- [ ] Schema file for advanced validation rules
- [ ] Enum support for dropdown menus
- [ ] File existence checking for paths
- [ ] Dependency validation (field X requires field Y)
- [ ] Undo/redo functionality
- [ ] Diff view showing changes before save
- [ ] Import/export configuration presets
- [ ] Configuration templates (beginner/advanced/pro)
