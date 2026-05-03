#!/usr/bin/env python3
"""
Normalize YAML files for comparison by alphabetizing keys and removing comments.

Usage:
    python normalize_yaml.py input.yaml [output.yaml]
    
If output.yaml is not specified, prints to stdout.
"""

import sys
import yaml


def sort_dict_recursive(obj):
    """
    Recursively sort all dictionaries in the object by key (alphabetically).
    Preserves lists and other data types as-is.
    Returns a regular dict (not OrderedDict) for clean YAML output.
    """
    if isinstance(obj, dict):
        return {k: sort_dict_recursive(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [sort_dict_recursive(item) for item in obj]
    else:
        return obj


def normalize_yaml_file(input_path, output_path=None):
    """
    Load YAML, sort all keys alphabetically, and save without comments.
    
    Args:
        input_path: Path to input YAML file
        output_path: Path to output file (None = print to stdout)
    """
    # Load the YAML file
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Sort all keys recursively
    sorted_data = sort_dict_recursive(data)
    
    # Output clean YAML without comments
    yaml_output = yaml.dump(
        sorted_data,
        default_flow_style=False,
        sort_keys=True,  # Ensure keys stay sorted in output
        allow_unicode=True,
        width=1000  # Prevent line wrapping
    )
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(yaml_output)
        print(f"Normalized YAML written to: {output_path}")
    else:
        print(yaml_output)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        normalize_yaml_file(input_path, output_path)
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
