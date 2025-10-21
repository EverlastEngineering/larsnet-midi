"""
YAML Configuration Engine for Web UI

Provides parsing, validation, and round-trip saving of YAML configuration files
with comment preservation and type inference.

This module implements the "functional core" for configuration management,
separating YAML parsing logic from UI rendering.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
from dataclasses import dataclass
from ruamel.yaml import YAML # type: ignore
from ruamel.yaml.comments import CommentedMap # type: ignore


@dataclass
class ValidationRule:
    """Defines validation rules for a configuration field"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    must_exist: bool = False  # For file paths
    regex: Optional[str] = None
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        """
        Validate a value against this rule.
        
        Returns:
            (is_valid, error_message)
        """
        if self.min_value is not None and value < self.min_value:
            return False, f"Value must be >= {self.min_value}"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Value must be <= {self.max_value}"
        
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Value must be one of: {', '.join(map(str, self.allowed_values))}"
        
        if self.regex is not None:
            if not re.match(self.regex, str(value)):
                return False, f"Value does not match required pattern"
        
        if self.must_exist:
            if not Path(value).exists():
                return False, f"Path does not exist: {value}"
        
        return True, ""


class ConfigField:
    """Represents a single configuration field with metadata"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        comment: str = "",
        field_type: Optional[str] = None,
        path: Optional[List[str]] = None
    ):
        """
        Initialize a configuration field.
        
        Args:
            key: Field name
            value: Field value
            comment: Inline or preceding comment from YAML
            field_type: Type hint ('bool', 'int', 'float', 'string', 'path')
            path: Hierarchical path to this field (e.g., ['kick', 'midi_note'])
        """
        self.key = key
        self.value = value
        self.comment = comment
        self.field_type = field_type or self._infer_type(value)
        self.path = path or []
        self.validation_rule = self._create_validation_rule()
    
    def _infer_type(self, value: Any) -> str:
        """Infer field type from Python value"""
        if isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            # Distinguish between paths and regular strings
            if '/' in value or '\\' in value or value.endswith(('.pth', '.py', '.yaml', '.yml')):
                return 'path'
            return 'string'
        else:
            return 'string'  # Default fallback
    
    def _create_validation_rule(self) -> ValidationRule:
        """Create validation rule based on field type and comment hints"""
        rule = ValidationRule()
        
        # Extract range hints from comments like "(0-1)" or "Hz"
        if self.comment:
            # Look for range patterns like (0-1), (0.0-1.0), etc.
            range_match = re.search(r'\((\d+\.?\d*)-(\d+\.?\d*)\)', self.comment)
            if range_match:
                rule.min_value = float(range_match.group(1))
                rule.max_value = float(range_match.group(2))
        
        # Type-specific validation
        if self.field_type == 'int' and 'midi' in self.key.lower():
            # MIDI notes are 0-127
            rule.min_value = 0
            rule.max_value = 127
        elif self.field_type == 'path':
            # Paths should exist (optional - we can make this configurable)
            # For now, don't enforce existence as some paths are in Docker container
            rule.must_exist = False
        
        return rule
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate the current field value.
        
        Returns:
            (is_valid, error_message)
        """
        return self.validation_rule.validate(self.value)
    
    def to_ui_control(self) -> Dict[str, Any]:
        """
        Convert field to UI control specification.
        
        Returns:
            Dictionary with control metadata for frontend rendering
        """
        return {
            'key': self.key,
            'path': '.'.join(self.path),
            'type': self.field_type,
            'value': self.value,
            'label': self._format_label(),
            'description': self.comment,
            'validation': {
                'min': self.validation_rule.min_value,
                'max': self.validation_rule.max_value,
                'allowed': self.validation_rule.allowed_values,
                'required': True
            }
        }
    
    def _format_label(self) -> str:
        """Format field key into human-readable label"""
        # Convert snake_case to Title Case
        return self.key.replace('_', ' ').title()


class ConfigSection:
    """Represents a logical section of configuration (e.g., 'kick', 'audio')"""
    
    def __init__(self, name: str, fields: List[ConfigField], comment: str = ""):
        """
        Initialize a configuration section.
        
        Args:
            name: Section name (e.g., 'kick', 'audio')
            fields: List of fields in this section
            comment: Section-level comment
        """
        self.name = name
        self.fields = fields
        self.comment = comment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary format"""
        return {
            'name': self.name,
            'label': self._format_label(),
            'description': self.comment,
            'fields': [f.to_ui_control() for f in self.fields]
        }
    
    def _format_label(self) -> str:
        """Format section name into human-readable label"""
        return self.name.replace('_', ' ').title()
    
    def validate_all(self) -> List[Tuple[str, str]]:
        """
        Validate all fields in section.
        
        Returns:
            List of (field_key, error_message) tuples for invalid fields
        """
        errors = []
        for field in self.fields:
            is_valid, error_msg = field.validate()
            if not is_valid:
                errors.append((field.key, error_msg))
        return errors


class YAMLConfigEngine:
    """
    Parse, render, and save YAML configuration files with comment preservation.
    
    Uses ruamel.yaml for round-trip editing that preserves formatting,
    comments, and key ordering.
    """
    
    def __init__(self, yaml_path: Union[str, Path]):
        """
        Initialize the config engine for a specific YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
        """
        self.yaml_path = Path(yaml_path)
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False
        self.yaml.width = 4096  # Prevent line wrapping
        self._data: Optional[CommentedMap] = None
    
    def load(self) -> CommentedMap:
        """Load YAML file and return commented map"""
        if self._data is None:
            with open(self.yaml_path, 'r') as f:
                self._data = self.yaml.load(f)
        return self._data
    
    def parse(self, max_depth: int = 2) -> List[ConfigSection]:
        """
        Parse YAML into structured sections with comments.
        
        Args:
            max_depth: Maximum nesting depth to parse (1 = top-level only, 2 = one level of nesting)
        
        Returns:
            List of ConfigSection objects representing the configuration
        """
        data = self.load()
        sections = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                # This is a section (e.g., 'kick', 'audio')
                section_comment = self._get_comment(data, key)
                fields = self._parse_dict(value, path=[key], max_depth=max_depth-1)
                sections.append(ConfigSection(key, fields, section_comment))
            else:
                # Top-level scalar value (less common but possible)
                comment = self._get_comment(data, key)
                field = ConfigField(key, value, comment, path=[])
                # Create a pseudo-section for top-level values
                if not sections or sections[0].name != '_root':
                    sections.insert(0, ConfigSection('_root', [], "Top-level configuration"))
                sections[0].fields.append(field)
        
        return sections
    
    def _parse_dict(
        self,
        data: Union[Dict, CommentedMap],
        path: List[str],
        max_depth: int
    ) -> List[ConfigField]:
        """
        Recursively parse dictionary into fields.
        
        Args:
            data: Dictionary or CommentedMap to parse
            path: Current hierarchical path
            max_depth: Remaining depth to parse
        
        Returns:
            List of ConfigField objects
        """
        fields = []
        
        for key, value in data.items():
            current_path = path + [key]
            comment = self._get_comment(data, key)
            
            if isinstance(value, dict) and max_depth > 0:
                # Nested dictionary - recurse
                nested_fields = self._parse_dict(value, current_path, max_depth - 1)
                fields.extend(nested_fields)
            elif not isinstance(value, (list, dict)):
                # Scalar value - create field
                field = ConfigField(key, value, comment, path=path)
                fields.append(field)
            # Skip lists and deep nested dicts
        
        return fields
    
    def _get_comment(self, data: CommentedMap, key: str) -> str:
        """
        Extract comment for a key from CommentedMap.
        
        Args:
            data: CommentedMap containing the key
            key: Key to get comment for
        
        Returns:
            Comment string (cleaned of # and whitespace)
        """
        if not hasattr(data, 'ca'):
            return ""
        
        # Try inline comment first (most common in our configs)
        if hasattr(data.ca, 'items') and key in data.ca.items:
            comment_token = data.ca.items.get(key)
            if comment_token and len(comment_token) >= 2 and comment_token[2]:
                # comment_token[2] is the inline comment
                comment = comment_token[2].value
                # Clean up: remove # and strip whitespace
                return comment.lstrip('#').strip()
        
        return ""
    
    def update_value(self, path: List[str], new_value: Any) -> bool:
        """
        Update a value in the loaded YAML data.
        
        Args:
            path: Hierarchical path to the value (e.g., ['kick', 'midi_note'])
            new_value: New value to set
        
        Returns:
            True if update successful, False otherwise
        """
        data = self.load()
        
        # Navigate to the correct location
        current = data
        for key in path[:-1]:
            if key not in current:
                return False
            current = current[key]
        
        # Update the final key
        final_key = path[-1]
        if final_key not in current:
            return False
        
        # Preserve type if possible
        old_value = current[final_key]
        if isinstance(old_value, bool):
            current[final_key] = bool(new_value)
        elif isinstance(old_value, int):
            current[final_key] = int(new_value)
        elif isinstance(old_value, float):
            current[final_key] = float(new_value)
        else:
            current[final_key] = new_value
        
        return True
    
    def save(self) -> None:
        """Save the modified YAML data back to file, preserving formatting"""
        if self._data is None:
            raise RuntimeError("No data loaded. Call load() or parse() first.")
        
        with open(self.yaml_path, 'w') as f:
            self.yaml.dump(self._data, f)
    
    def validate_all(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Validate all configuration values.
        
        Returns:
            Dictionary mapping section names to list of (field, error) tuples
        """
        sections = self.parse()
        errors = {}
        
        for section in sections:
            section_errors = section.validate_all()
            if section_errors:
                errors[section.name] = section_errors
        
        return errors


def get_config_engine(project_id: int, config_type: str) -> YAMLConfigEngine:
    """
    Factory function to get config engine for a specific project and config type.
    
    Args:
        project_id: Project number
        config_type: One of 'config', 'midiconfig', or 'eq'
    
    Returns:
        YAMLConfigEngine instance
    
    Raises:
        ValueError: If config_type is invalid or file doesn't exist
    """
    from pathlib import Path
    
    valid_types = ['config', 'midiconfig', 'eq']
    if config_type not in valid_types:
        raise ValueError(f"config_type must be one of: {', '.join(valid_types)}")
    
    # Construct path to project config file
    project_dir = Path(f'/app/user_files/{project_id} - *').expanduser()
    
    # Handle glob pattern (project name after number)
    import glob
    matches = glob.glob(str(project_dir))
    if not matches:
        raise ValueError(f"Project {project_id} not found")
    
    project_dir = Path(matches[0])
    config_file = project_dir / f'{config_type}.yaml'
    
    if not config_file.exists():
        raise ValueError(f"Config file not found: {config_file}")
    
    return YAMLConfigEngine(config_file)
