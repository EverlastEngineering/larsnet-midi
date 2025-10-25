"""
Compare multiple separation configurations for troublesome audio files.

This script processes a project with various combinations of Wiener filter
and EQ settings, organizing outputs into comparison folders. This helps users
identify the best configuration for problematic audio.

Usage:
    python compare_separation_configs.py              # Auto-detect project
    python compare_separation_configs.py 1            # Compare configs for project #1
    python compare_separation_configs.py --device cuda  # Use GPU acceleration
    python compare_separation_configs.py --config custom_configs.yaml  # Custom configs

Output Structure:
    project_dir/
        for_comparison/
            baseline/           # No filtering
            wiener-2.0/        # Wiener filter with α=2.0
            wiener-1.0/        # Wiener filter with α=1.0
            wiener-0.4/        # Wiener filter with α=0.4
            eq/                # EQ cleanup only
            wiener-2.0_eq/     # Combined Wiener 2.0 + EQ
            wiener-1.0_eq/     # Combined Wiener 1.0 + EQ
"""

from separation_utils import process_stems_for_project
from project_manager import (
    get_project_by_number,
    select_project,
    get_project_config,
    USER_FILES_DIR
)
from pathlib import Path
from typing import List, Dict, Optional, Any
import argparse
import sys
import yaml # type: ignore
import shutil


# Default comparison configurations
DEFAULT_CONFIGS = [
    {"name": "baseline", "wiener": None, "eq": False},
    {"name": "wiener-2.0", "wiener": 2.0, "eq": False},
    {"name": "wiener-1.0", "wiener": 1.0, "eq": False},
    {"name": "wiener-0.4", "wiener": 0.4, "eq": False},
    {"name": "eq", "wiener": None, "eq": True},
    {"name": "wiener-2.0_eq", "wiener": 2.0, "eq": True},
    {"name": "wiener-1.0_eq", "wiener": 1.0, "eq": True},
]


def load_custom_configs(config_path: Path) -> List[Dict]:
    """
    Load custom comparison configurations from YAML file.
    
    Expected format:
        configs:
          - name: "my_config"
            wiener: 1.5
            eq: true
          - name: "another_config"
            wiener: null
            eq: true
    
    Args:
        config_path: Path to custom configuration YAML file
        
    Returns:
        List of configuration dictionaries
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict) or 'configs' not in data:
        raise ValueError("Config file must contain 'configs' key with list of configurations")
    
    configs = data['configs']
    
    # Validate configs is a non-null list
    if configs is None or not isinstance(configs, list):
        raise ValueError("'configs' must be a list of configuration dictionaries")
    
    # Validate each config
    for i, config in enumerate(configs):
        if 'name' not in config:
            raise ValueError(f"Config {i} missing required 'name' field")
        if 'wiener' not in config:
            config['wiener'] = None
        if 'eq' not in config:
            config['eq'] = False
        if config['wiener'] is not None and config['wiener'] <= 0:
            raise ValueError(f"Config '{config['name']}': Wiener exponent must be positive")
    
    return configs


def process_comparison(
    project: dict,
    configs: List[Dict],
    device: str = 'cpu',
    cleanup: bool = False
):
    """
    Process a project with multiple separation configurations for comparison.
    
    Args:
        project: Project info dictionary from project_manager
        configs: List of configuration dicts with 'name', 'wiener', and 'eq' keys
        device: 'cpu' or 'cuda'
        cleanup: If True, remove original stems directory before processing
    """
    project_dir = project["path"]
    comparison_dir = project_dir / "for_comparison"
    
    print(f"\n{'='*70}")
    print(f"Comparison Mode: Project {project['number']} - {project['name']}")
    print(f"{'='*70}")
    print(f"Configurations to test: {len(configs)}")
    print(f"Output directory: {comparison_dir}")
    print(f"{'='*70}\n")
    
    # Get project-specific config
    config_path = get_project_config(project_dir, "config.yaml")
    if config_path is None:
        print("ERROR: config.yaml not found in project or root directory")
        sys.exit(1)
    
    print(f"Using config: {config_path}")
    
    # Optional cleanup of original stems
    if cleanup:
        original_stems = project_dir / "stems"
        if original_stems.exists():
            print(f"Removing original stems directory: {original_stems}")
            shutil.rmtree(original_stems)
    
    # Create comparison directory
    comparison_dir.mkdir(exist_ok=True)
    
    # Process each configuration
    for i, config in enumerate(configs, 1):
        config_name = config['name']
        wiener = config['wiener']
        eq = config['eq']
        
        print(f"\n[{i}/{len(configs)}] Processing: {config_name}")
        print(f"  Wiener: {wiener if wiener is not None else 'Disabled'}")
        print(f"  EQ: {'Enabled' if eq else 'Disabled'}")
        print("-" * 70)
        
        # Output to for_comparison/<config_name>/
        output_dir = comparison_dir / config_name
        
        try:
            process_stems_for_project(
                project_dir=project_dir,
                stems_dir=output_dir,
                config_path=config_path,
                wiener_exponent=wiener,
                device=device,
                apply_eq=eq,
                verbose=True
            )
            print(f"✓ Completed: {config_name}")
            print(f"  Output: {output_dir}")
        
        except Exception as e:
            print(f"✗ Failed: {config_name}")
            print(f"  Error: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Comparison Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {comparison_dir}")
    print(f"\nYou can now compare the stems in each subdirectory:")
    for config in configs:
        print(f"  - {comparison_dir / config['name']}")
    print(f"\nTip: Load stems from different folders into your DAW to A/B compare")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare multiple separation configurations for troublesome audio.",
        epilog="""
Examples:
  python compare_separation_configs.py              # Auto-detect project
  python compare_separation_configs.py 1            # Compare configs for project #1
  python compare_separation_configs.py --device cuda  # Use GPU
  python compare_separation_configs.py --config custom.yaml  # Custom configs
  python compare_separation_configs.py --cleanup    # Remove original stems first
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('project_number', type=int, nargs='?', default=None,
                       help="Project number to process (optional)")
    parser.add_argument('-d', '--device', type=str, default='cpu',
                       help="Torch device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument('-c', '--config', type=str, default=None,
                       help="Path to custom configuration YAML file")
    parser.add_argument('--cleanup', action='store_true',
                       help="Remove original stems directory before processing")
    
    args = parser.parse_args()
    
    # Load configurations
    if args.config:
        try:
            configs = load_custom_configs(Path(args.config))
            print(f"Loaded {len(configs)} custom configurations from {args.config}")
        except Exception as e:
            print(f"ERROR: Failed to load custom config: {e}")
            sys.exit(1)
    else:
        configs = DEFAULT_CONFIGS
        print(f"Using default configurations ({len(configs)} configs)")
    
    # Select project
    if args.project_number is not None:
        project = get_project_by_number(args.project_number, USER_FILES_DIR)
        if project is None:
            print(f"ERROR: Project {args.project_number} not found")
            sys.exit(1)
    else:
        project = select_project(None, USER_FILES_DIR, allow_interactive=True)
        if project is None:
            print("\nNo projects found in user_files/")
            print("Drop an audio file (.wav, .mp3, .flac) in user_files/ to get started!")
            sys.exit(0)
    
    # Process comparison
    process_comparison(project, configs, args.device, args.cleanup)
