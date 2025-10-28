"""
Separate drums into individual stems using MDX23C or LarsNet.

Uses project-based workflow: automatically detects projects in user_files/
or processes new audio files dropped there.

Usage:
    python separate.py              # Auto-detect project (uses MDX23C by default)
    python separate.py 1            # Process specific project by number
    python separate.py --device cuda  # Use GPU acceleration
    python separate.py --model larsnet  # Use LarsNet instead of MDX23C
"""

from separation_utils import process_stems_for_project
from project_manager import (
    discover_projects,
    find_loose_files,
    create_project,
    select_project,
    get_project_by_number,
    get_project_config,
    update_project_metadata,
    USER_FILES_DIR
)
from device_utils import detect_best_device
from pathlib import Path
from typing import Optional
import argparse
import sys


def separate_project(
    project: dict,
    model: str = 'mdx23c',
    overlap: int = 8,
    wiener_exponent: Optional[float] = None,
    device: str = 'cpu',
    apply_eq: bool = False
):
    """
    Separate drums for a specific project.
    
    Args:
        project: Project info dictionary from project_manager
        model: Separation model to use ('mdx23c' or 'larsnet')
        overlap: Overlap value for MDX23C (2-50, higher=better quality but slower, default=8)
        wiener_exponent: Wiener filter exponent (None to disable, LarsNet only)
        device: 'cpu' or 'cuda'
        apply_eq: Whether to apply frequency cleanup
    """
    project_dir = project["path"]
    
    print(f"\n{'='*60}")
    print(f"Processing Project {project['number']}: {project['name']}")
    print(f"{'='*60}\n")
    
    # Get project-specific config
    config_path = get_project_config(project_dir, "config.yaml")
    if config_path is None:
        print("ERROR: config.yaml not found in project or root directory")
        sys.exit(1)
    
    print(f"Using config: {config_path}")
    
    # Input: project directory (original audio file)
    # Output: project/stems/ subdirectory
    stems_dir = project_dir / "stems"
    
    # Process stems
    process_stems_for_project(
        project_dir=project_dir,
        stems_dir=stems_dir,
        config_path=config_path,
        model=model,
        overlap=overlap,
        wiener_exponent=wiener_exponent,
        device=device,
        apply_eq=apply_eq,
        verbose=True
    )
    
    # Update project metadata
    update_project_metadata(project_dir, {
        "status": {
            "separated": True,
            "cleaned": project["metadata"]["status"].get("cleaned", False) if project["metadata"] else False,
            "midi_generated": project["metadata"]["status"].get("midi_generated", False) if project["metadata"] else False,
            "video_rendered": project["metadata"]["status"].get("video_rendered", False) if project["metadata"] else False
        }
    })
    
    print(f"Status Update: Process complete!")
    print(f"  Stems saved to: {stems_dir}")
    print(f"  Project status updated\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Separate drums into individual stems using LarsNet.",
        epilog="""
Examples:
  python separate.py                    # Auto-detect project or new file
  python separate.py 1                  # Process project #1
  python separate.py --device cuda      # Use GPU
  python separate.py --wiener 2.0       # Apply Wiener filtering
        """
    )
    
    parser.add_argument('project_number', type=int, nargs='?', default=None,
                       help="Project number to process (optional)")
    parser.add_argument('-m', '--model', type=str, default='mdx23c',
                       choices=['mdx23c', 'larsnet'],
                       help="Separation model: 'mdx23c' or 'larsnet' (default: mdx23c)")
    parser.add_argument('-o', '--overlap', type=int, default=8,
                       help="MDX23C overlap (2-50): higher=better quality but slower (default: 8)")
    parser.add_argument('-w', '--wiener', type=float, default=None,
                       help="Wiener filter exponent (default: disabled, LarsNet only)")
    parser.add_argument('-d', '--device', type=str, default=None,
                       help="Torch device: 'cpu', 'cuda', 'mps', or auto-detect (default: auto)")
    parser.add_argument('--eq', action='store_true',
                       help="Apply frequency cleanup (experimental)")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = detect_best_device(verbose=False)
        print(f"Auto-detected device: {args.device}")
    
    # Validate
    if args.overlap < 2 or args.overlap > 50:
        print("ERROR: Overlap must be between 2 and 50")
        sys.exit(1)
    
    if args.wiener is not None:
        if args.wiener <= 0:
            print("ERROR: Wiener exponent must be positive")
            sys.exit(1)
        if args.model == 'mdx23c':
            print("WARNING: Wiener filter only works with LarsNet model, ignoring --wiener")
            args.wiener = None
    
    # Check for loose files first (new audio files to process)
    loose_files = find_loose_files(USER_FILES_DIR)
    
    if loose_files:
        print(f"\nFound {len(loose_files)} new audio file(s):")
        for f in loose_files:
            print(f"  - {f.name}")
        
        if len(loose_files) == 1:
            print(f"\nCreating project for: {loose_files[0].name}")
            try:
                project = create_project(loose_files[0], USER_FILES_DIR, Path("."))
                print(f"✓ Created project {project['number']}: {project['name']}")
                
                # Process the newly created project
                separate_project(project, args.model, args.overlap, args.wiener, args.device, args.eq)
                
            except Exception as e:
                print(f"ERROR: Failed to create project: {e}")
                sys.exit(1)
        else:
            # Multiple loose files - ask user to process them one at a time
            print("\nPlease move all but one file to process them individually,")
            print("or organize them into project folders.")
            sys.exit(0)
    
    else:
        # No loose files - look for existing projects
        if args.project_number is not None:
            # Specific project requested
            project = get_project_by_number(args.project_number, USER_FILES_DIR)
            if project is None:
                print(f"ERROR: Project {args.project_number} not found")
                sys.exit(1)
        else:
            # Auto-select project
            project = select_project(None, USER_FILES_DIR, allow_interactive=True)
            if project is None:
                print("\nNo projects found in user_files/")
                print("Drop an audio file (.wav, .mp3, .flac) in user_files/ to get started!")
                sys.exit(0)
        
        # Process the selected project
        separate_project(project, args.model, args.overlap, args.wiener, args.device, args.eq)
